"""Combine per-objective gradients (primary first) into one update direction.

Every combiner takes ``grads: dict[str, Tensor]`` of *raw* flattened gradients (see
:func:`pcdkd.optim.grads.flat_grad`) and returns ``(direction, info)``; write the direction into
``.grad`` with :func:`pcdkd.optim.grads.unflatten_to` and call any optimizer.

Notation: g1 = primary (CE) gradient, g2 = secondary (KD) gradient, g~i = s_i g_i (EMA-normalized).

* ``WeightedSum(lam)``: d = (1 - lam) g1 + lam g2 (raw Hinton KD).
* ``NormalizedWeightedSum(lam)``: d~ = g~1 + lam g~2, rescaled to ||g1|| (NWKD).
* ``PCGrad(lam=1)``: d = g1 + lam g2', where g2' is g2 minus its g1-component if g1^T g2 < 0
  (one-sided "protect the primary" variant on raw gradients).
* ``PCD(tau)``: d~ = PCD QP solution (Cor. 4.6 / KKT), rescaled to ||g1|| (Alg. 1).
* ``AdaptiveLambda(tau)``: NWKD with lam_t = max(0, tau - g~1^T g~2 / ||g~2||^2), the Cor. 4.6
  coefficient computed independently of the QP; must equal PCD for K = 2.

Every combiner updates its own EMA normalizer on every call (also those whose direction ignores
it), so ``cos_norm`` and the scales are logged identically for all arms.

``info`` keys (all JSON-serialisable; secondary-specific scalars refer to the first secondary):

* method, step, primary, secondaries
* cos_raw: cosine of raw g1, g2
* cos_norm: g~1^T g~2, the c of PCD Prop. 4.7 (equals the cosine only in EMA steady state)
* active: the method's conditional branch fired (PCD constraint binding / PCGrad projection
  applied / AdaptiveLambda lam_t > 0); always False for the weighted sums
* feasible: False only for an infeasible PCD QP (fallback to the primary)
* lambda_eff: secondary/primary coefficient in the space the method combines (raw for
  WeightedSum and PCGrad, normalized otherwise); lambda_raw: the same ratio in raw space
* prop47_margin: 1 + tau c - c^2 with c = cos_raw (PCD Prop. 4.7 under the unit-norm
  idealization); None for methods without tau
* norm_primary, norm_secondary, norm_direction, scale_primary, scale_secondary
* align_primary = cos(g1, d), align_secondary = cos(g2, d), zero_primary, tau
* PCD only: mu, active_set, cos_raw_all, norm_secondary_all
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch

from pcdkd.optim.normalize import EMANormalizer
from pcdkd.optim.pcd import MAX_K, gram_matrix, solve_pcd_qp


def _cos(dot: float, n1: float, n2: float) -> float:
    return dot / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0


def _rescale(d_tilde: torch.Tensor, norm_primary: float) -> torch.Tensor:
    """d = d~ ||g1|| / ||d~||; zero if ||g1|| = 0 or d~ = 0 (PCD Alg. 1, lines 7-8, 17-20)."""
    d_norm = float(torch.linalg.vector_norm(d_tilde).item())
    if norm_primary == 0.0 or d_norm == 0.0:
        return torch.zeros_like(d_tilde)
    return d_tilde * (norm_primary / d_norm)


class Combiner:
    """Base class: ordering, EMA bookkeeping and the diagnostics shared by all methods."""

    name = "base"
    max_objectives = 2

    def __init__(self, beta: float = 0.99, eps: float = 1e-8):
        self.normalizer = EMANormalizer(beta, eps)
        self.step = 0

    def _ordered(self, grads: dict[str, torch.Tensor], primary: str) -> list[str]:
        if primary not in grads:
            raise KeyError(f"primary objective {primary!r} not in grads {list(grads)}")
        names = [primary] + [k for k in grads if k != primary]
        if len(names) < 2:
            raise ValueError("need at least one secondary objective")
        if len(names) > self.max_objectives:
            msg = f"{type(self).__name__} supports at most {self.max_objectives} objectives"
            raise ValueError(msg)
        shape = grads[primary].shape
        if any(grads[k].dim() != 1 or grads[k].shape != shape for k in names):
            raise ValueError("all gradients must be 1-D tensors of equal length")
        return names

    def combine(
        self, grads: dict[str, torch.Tensor], primary: str = "ce"
    ) -> tuple[torch.Tensor, dict]:
        names = self._ordered(grads, primary)
        g = [grads[k].detach() for k in names]
        G = gram_matrix(g)
        norms = np.sqrt(np.maximum(np.diag(G), 0.0))
        scales = [self.normalizer.update_sq(k, float(G[i, i])) for i, k in enumerate(names)]
        self.step += 1

        direction, extra = self._direction(g, G, norms, scales)

        dG = gram_matrix([direction, g[0], g[1]])
        d_norm = math.sqrt(max(dG[0, 0], 0.0))
        info = {
            "method": self.name,
            "step": self.step,
            "primary": names[0],
            "secondaries": names[1:],
            "cos_raw": _cos(G[0, 1], norms[0], norms[1]),
            "cos_norm": float(scales[0] * scales[1] * G[0, 1]),
            "active": False,
            "feasible": True,
            "lambda_eff": None,
            "lambda_raw": None,
            "prop47_margin": None,
            "norm_primary": float(norms[0]),
            "norm_secondary": float(norms[1]),
            "norm_direction": d_norm,
            "scale_primary": scales[0],
            "scale_secondary": scales[1],
            "align_primary": _cos(dG[0, 1], d_norm, norms[0]),
            "align_secondary": _cos(dG[0, 2], d_norm, norms[1]),
            "zero_primary": bool(norms[0] == 0.0),
            "tau": None,
        }
        info.update(extra)
        return direction, info

    def _direction(
        self, g: list[torch.Tensor], G: np.ndarray, norms: np.ndarray, scales: list[float]
    ) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    def state_dict(self) -> dict:
        return {"step": self.step, "normalizer": self.normalizer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.step = int(state["step"])
        self.normalizer.load_state_dict(state["normalizer"])


class WeightedSum(Combiner):
    """Raw weighted sum d = (1 - lam) g1 + lam g2 (the gradient of (1-lam) CE + lam KD)."""

    name = "wkd"

    def __init__(self, lam: float, beta: float = 0.99, eps: float = 1e-8):
        super().__init__(beta, eps)
        if not 0.0 <= lam < 1.0:
            raise ValueError(f"lam must be in [0, 1), got {lam}")
        self.lam = float(lam)

    def _direction(self, g, G, norms, scales):
        d = (1.0 - self.lam) * g[0] + self.lam * g[1]
        ratio = self.lam / (1.0 - self.lam)
        return d, {"lambda_eff": ratio, "lambda_raw": ratio}


class NormalizedWeightedSum(Combiner):
    """EMA-normalized weighted sum d~ = g~1 + lam g~2, optionally rescaled to ||g1|| like PCD."""

    name = "nwkd"

    def __init__(self, lam: float, beta: float = 0.99, eps: float = 1e-8, rescale: bool = True):
        super().__init__(beta, eps)
        if lam < 0 or not math.isfinite(lam):
            raise ValueError(f"lam must be finite and >= 0, got {lam}")
        self.lam = float(lam)
        self.rescale = rescale

    def _weighted(self, g, norms, scales, lam: float) -> torch.Tensor:
        g1n, g2n = g[0] * scales[0], g[1] * scales[1]
        d_tilde = g1n * 1.0
        if lam != 0.0:
            d_tilde = d_tilde + lam * g2n
        return _rescale(d_tilde, float(norms[0])) if self.rescale else d_tilde

    def _direction(self, g, G, norms, scales):
        lam_raw = self.lam * scales[1] / scales[0] if scales[0] > 0 else None
        return self._weighted(g, norms, scales, self.lam), {
            "lambda_eff": self.lam,
            "lambda_raw": lam_raw,
        }


class PCGrad(Combiner):
    """One-sided PCGrad: remove the component of g2 along g1 only when they conflict (g1^T g2 < 0).

    d = g1 + lam g2', so g1^T d = ||g1||^2 + lam * max(0, g1^T g2): the projection on g1 is
    preserved.
    """

    name = "pcgrad"

    def __init__(self, lam: float = 1.0, beta: float = 0.99, eps: float = 1e-8):
        super().__init__(beta, eps)
        if lam < 0 or not math.isfinite(lam):
            raise ValueError(f"lam must be finite and >= 0, got {lam}")
        self.lam = float(lam)

    def _direction(self, g, G, norms, scales):
        dot = float(G[0, 1])
        conflict = dot < 0.0 and G[0, 0] > 0.0
        g2 = g[1] - (dot / float(G[0, 0])) * g[0] if conflict else g[1]
        return g[0] + self.lam * g2, {
            "active": bool(conflict),
            "lambda_eff": self.lam,
            "lambda_raw": self.lam,
        }


def _check_tau(tau: float | Sequence[float]) -> float | list[float]:
    arr = np.asarray(tau, dtype=np.float64)
    if np.any(arr < 0) or not np.all(np.isfinite(arr)):
        raise ValueError(f"tau must be finite and >= 0, got {tau}")
    return float(arr) if arr.ndim == 0 else arr.tolist()


class PCD(Combiner):
    """Priority-Constrained Descent (PCD Alg. 1): EMA normalization, QP, magnitude rescale.

    Supports 2 <= K <= 4 objectives. With ``rescale=False`` the normalized direction d~ is returned.
    """

    name = "pcd"
    max_objectives = MAX_K

    def __init__(
        self,
        tau: float | Sequence[float],
        beta: float = 0.99,
        eps: float = 1e-8,
        rescale: bool = True,
    ):
        super().__init__(beta, eps)
        self.tau = _check_tau(tau)
        self.rescale = rescale

    def _tau1(self) -> float:
        return float(np.asarray(self.tau).reshape(-1)[0])

    def _direction(self, g, G, norms, scales):
        K = len(g)
        tau1 = self._tau1()
        c = _cos(G[0, 1], norms[0], norms[1])
        extra = {
            "tau": self.tau,
            "prop47_margin": 1.0 + tau1 * c - c * c,
            "cos_raw_all": [_cos(G[0, j], norms[0], norms[j]) for j in range(1, K)],
            "norm_secondary_all": [float(norms[j]) for j in range(1, K)],
        }
        if norms[0] == 0.0:  # Alg. 1 line 7: primary-stationary halt
            extra.update(
                {"mu": [0.0] * (K - 1), "active_set": [], "lambda_eff": 0.0, "lambda_raw": 0.0}
            )
            return torch.zeros_like(g[0]), extra
        gn = [g[i] * scales[i] for i in range(K)]
        d_tilde, qp = solve_pcd_qp(gn, self.tau)
        mu = qp["mu"]
        extra.update(
            {
                "active": len(qp["active"]) > 0,
                "active_set": list(qp["active"]),
                "feasible": qp["feasible"],
                "mu": mu,
                "lambda_eff": mu[0],
                "lambda_raw": mu[0] * scales[1] / scales[0] if scales[0] > 0 else None,
            }
        )
        d = _rescale(d_tilde, float(norms[0])) if self.rescale else d_tilde
        return d, extra


class AdaptiveLambda(NormalizedWeightedSum):
    """NWKD with PCD's own implicit coefficient lam_t = max(0, tau - g~1^T g~2 / ||g~2||^2) (K = 2).

    Computed from the closed form directly (no QP); an identity check that must reproduce ``PCD``.
    """

    name = "adaptive_lambda"

    def __init__(self, tau: float, beta: float = 0.99, eps: float = 1e-8, rescale: bool = True):
        super().__init__(0.0, beta, eps, rescale)
        self.tau = float(_check_tau(tau))

    def _direction(self, g, G, norms, scales):
        n2 = scales[1] ** 2 * float(G[1, 1])  # ||g~2||^2
        inner = scales[0] * scales[1] * float(G[0, 1])  # g~1^T g~2
        lam = max(0.0, self.tau - inner / n2) if n2 > 0 else 0.0
        c = _cos(G[0, 1], norms[0], norms[1])
        extra = {
            "tau": self.tau,
            "active": lam > 0.0,
            "lambda_eff": lam,
            "lambda_raw": lam * scales[1] / scales[0] if scales[0] > 0 else None,
            "prop47_margin": 1.0 + self.tau * c - c * c,
        }
        if norms[0] == 0.0:  # same halt rule as PCD
            return torch.zeros_like(g[0]), extra
        return self._weighted(g, norms, scales, lam), extra


COMBINERS = {
    "wkd": WeightedSum,
    "nwkd": NormalizedWeightedSum,
    "pcgrad": PCGrad,
    "pcd": PCD,
    "adaptive_lambda": AdaptiveLambda,
}
