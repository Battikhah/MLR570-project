"""Baseline combiners for the experiment suite (same ``combine(grads, primary)`` contract).

* ``MultiWeightedSum(lams, normalize)``: K-objective d = sum w_i g_i (raw, "sum") or the
  EMA-normalized d~ = sum w_i g~_i rescaled to ||g1|| ("nws"); w = [1, *lams].
* ``MinNorm(log_loss)``: symmetric min-norm point on the simplex (MGDA; Desideri 2012, Sener &
  Koltun 2018). With ``log_loss=True`` it is applied to grad log L_i = g_i / L_i, the exact
  (non-amortized) DTO-KD rule (Hayder et al., ICLR 2026, Eqs. 11-12), without DTO-KD's feature
  projection or separate heads. Pass ``losses`` to ``combine``.
* ``DOT(delta, momentum, lam)``: distillation-oriented trainer (Zhao et al., ICCV 2023): separate
  heavy-ball buffers, beta - delta for the task gradient and beta + delta for the KD gradient.
  Use with a momentum-free optimizer.
* ``LayerPCD(tau)`` / ``LayerPCDM(tau)``: per-layer PCD (decision 0004), the latter on DOT-style
  separate momentum buffers.
* ``SymmetricPCGrad(lams)``: gradient surgery of Yu et al. (2020), random projection order.
* ``shadow_cosines``: from one K=2 ``info`` dict, the pairwise cosines of the directions that
  CE-only, PCD, NWKD, WKD and DTO-KD *would* take on the same batch (no extra backward).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import combinations

import numpy as np
import torch

from pcdkd.optim.combiners import Combiner, _cos, _rescale
from pcdkd.optim.normalize import EMANormalizer
from pcdkd.optim.pcd import MAX_K, gram_matrix, pcd_k2_closed_form


def _lams(lams) -> list[float]:
    out = [float(x) for x in np.atleast_1d(lams)]
    if any(x < 0 or not math.isfinite(x) for x in out):
        raise ValueError(f"weights must be finite and >= 0, got {lams}")
    return out


class MultiWeightedSum(Combiner):
    max_objectives = MAX_K

    def __init__(self, lams, normalize=True, beta=0.99, eps=1e-8, rescale=True):
        super().__init__(beta, eps)
        self.lams = _lams(lams)
        self.normalize = normalize
        self.rescale = rescale and normalize
        self.name = "nws" if normalize else "sum"

    def _weights(self, K):
        if len(self.lams) == 1:
            return [1.0] + self.lams * (K - 1)
        if len(self.lams) != K - 1:
            raise ValueError(f"need {K - 1} secondary weights, got {len(self.lams)}")
        return [1.0, *self.lams]

    def _direction(self, g, G, norms, scales):
        w = self._weights(len(g))
        s = scales if self.normalize else [1.0] * len(g)
        d = sum(w[i] * s[i] * g[i] for i in range(len(g)))
        if self.rescale:
            d = _rescale(d, float(norms[0]))
        return d, {"lambda_eff": w[1], "weights": w}


def min_norm_weights(G: np.ndarray) -> np.ndarray:
    """argmin_{pi in simplex} pi^T G pi, exactly: closed form for K=2, else enumerate supports
    (on support S the optimum is pi_S ~ G_S^+ 1 if positive; K <= 4 gives <= 15 supports)."""
    G = np.asarray(G, dtype=np.float64)
    K = G.shape[0]
    if K == 2:
        den = G[0, 0] + G[1, 1] - 2 * G[0, 1]
        p = 0.5 if den <= 0 else float(np.clip((G[1, 1] - G[0, 1]) / den, 0.0, 1.0))
        return np.array([p, 1.0 - p])
    best, best_val = None, np.inf
    for size in range(1, K + 1):
        for S in combinations(range(K), size):
            w = np.linalg.pinv(G[np.ix_(S, S)]) @ np.ones(size)
            if w.sum() <= 0 or np.any(w < -1e-12):
                continue
            pi = np.zeros(K)
            pi[list(S)] = np.maximum(w, 0) / w.sum()
            val = pi @ G @ pi
            if val < best_val - 1e-15:
                best, best_val = pi, val
    return best if best is not None else np.full(K, 1.0 / K)


class MinNorm(Combiner):
    max_objectives = MAX_K

    def __init__(self, log_loss=False, rescale=False, beta=0.99, eps=1e-8):
        super().__init__(beta, eps)
        self.log_loss = log_loss
        self.rescale = rescale
        self.name = "dto" if log_loss else "mgda"

    def combine(self, grads, primary="ce", losses=None):
        if self.log_loss:
            if losses is None:
                raise ValueError("DTO-KD needs the loss values (losses=...)")
            if any(not losses[k] > 0 for k in grads):
                raise ValueError("log-loss gradients need positive losses")
            grads = {k: v / float(losses[k]) for k, v in grads.items()}
        return super().combine(grads, primary)

    def _direction(self, g, G, norms, scales):
        pi = min_norm_weights(G)
        d = sum(float(pi[i]) * g[i] for i in range(len(g)))
        if self.rescale:
            d = _rescale(d, float(norms[0]))
        return d, {"pi": pi.tolist(), "lambda_eff": float(pi[1] / pi[0]) if pi[0] > 0 else None}


class DOT(Combiner):
    name = "dot"

    def __init__(self, delta=0.075, momentum=0.9, lam=0.9, beta=0.99, eps=1e-8):
        super().__init__(beta, eps)
        if not 0 <= delta <= momentum < 1 or momentum + delta >= 1:
            raise ValueError("need 0 <= delta <= momentum and momentum + delta < 1")
        self.betas = (momentum - delta, momentum + delta)
        self.w = (1.0 - lam, lam)  # task / KD loss weights, as in the weighted-sum KD it extends
        self.buf: list[torch.Tensor] | None = None

    def _direction(self, g, G, norms, scales):
        if self.buf is None:
            self.buf = [torch.zeros_like(g[0]), torch.zeros_like(g[1])]
        for i in range(2):
            self.buf[i] = self.betas[i] * self.buf[i] + self.w[i] * g[i]
        return self.buf[0] + self.buf[1], {"lambda_eff": self.w[1] / self.w[0]}


class SymmetricPCGrad(Combiner):
    name = "pcgrad_sym"
    max_objectives = MAX_K

    def __init__(self, lams=1.0, beta=0.99, eps=1e-8):
        super().__init__(beta, eps)
        self.lams = _lams(lams)

    def _direction(self, g, G, norms, scales):
        K = len(g)
        w = [1.0] + (self.lams * (K - 1) if len(self.lams) == 1 else self.lams)
        v = [w[i] * g[i] for i in range(K)]
        rng = np.random.default_rng(self.step)  # deterministic per step
        projected = 0
        out = torch.zeros_like(g[0])
        for i in range(K):
            p = v[i].clone()
            for j in rng.permutation(K):
                if j == i:
                    continue
                dot, nj = float(p @ v[j]), float(v[j] @ v[j])
                if dot < 0 and nj > 0:
                    p = p - (dot / nj) * v[j]
                    projected += 1
            out = out + p
        return out, {"active": projected > 0, "lambda_eff": w[1]}


def shadow_cosines(info: dict, losses: dict, tau=0.5, lam_nwkd=1.0, lam_wkd=0.9) -> dict:
    """Pairwise cosines of the K=2 directions CE / PCD(tau) / NWKD / WKD / DTO-KD on one batch."""
    n1, n2, c = info["norm_primary"], info["norm_secondary"], info["cos_raw"]
    s1, s2 = info["scale_primary"], info["scale_secondary"]
    if n1 == 0 or n2 == 0:
        return {}
    G = np.array([[n1 * n1, c * n1 * n2], [c * n1 * n2, n2 * n2]])
    Gn = G * np.outer([s1, s2], [s1, s2])
    mu = pcd_k2_closed_form(Gn, tau)[0][1]
    L = np.array([losses[info["primary"]], losses[info["secondaries"][0]]], dtype=np.float64)
    pi = min_norm_weights(G / np.outer(L, L)) if np.all(L > 0) else np.array([0.5, 0.5])
    W = {
        "ce": np.array([1.0, 0.0]),
        "pcd": np.array([s1, mu * s2]),
        "nwkd": np.array([s1, lam_nwkd * s2]),
        "wkd": np.array([1 - lam_wkd, lam_wkd]),
        "dto": pi / L,
    }

    def cos(a, b):
        den = math.sqrt(max(a @ G @ a, 0) * max(b @ G @ b, 0))
        return float(a @ G @ b / den) if den > 0 else 0.0

    pairs = [("pcd", "ce"), ("pcd", "nwkd"), ("pcd", "wkd"), ("pcd", "dto"), ("dto", "nwkd")]
    return {f"shadow_{a}_{b}": cos(W[a], W[b]) for a, b in pairs} | {"shadow_pcd_mu": float(mu)}


class LayerPCD(Combiner):
    """Per-layer PCD (``pcdl``): the K = 2 PCD QP solved separately on each parameter group.

    Each group (a slice of the flattened gradient, see :meth:`set_groups`) has its own EMA norm
    normalization and its own mu_l = max(0, tau - cos-like term), and its direction is rescaled to
    that group's ||g_CE^l||. Reuses ``pcd_k2_closed_form`` and ``_rescale``; with a single group
    it is exactly :class:`pcdkd.optim.combiners.PCD`. Diagnostics: ``lambda_eff`` = numel-weighted
    mean mu, ``active`` = fraction of groups with mu > 0, per-group ``mu_group`` / ``cos_group``.
    """

    name = "pcdl"

    def __init__(self, tau: float, beta: float = 0.99, eps: float = 1e-8, rescale: bool = True):
        super().__init__(beta, eps)
        if not (math.isfinite(tau) and tau >= 0):
            raise ValueError(f"tau must be finite and >= 0, got {tau}")
        self.tau = float(tau)
        self.rescale = rescale
        self.group_norm = EMANormalizer(beta, eps)  # one scale per (group, objective)
        self.groups: list[tuple[str, int]] | None = None

    def set_groups(self, groups: Sequence[tuple[str, int]]) -> None:
        """Group (name, numel) in the order of the flattened gradient."""
        self.groups = [(str(n), int(k)) for n, k in groups]

    def _inputs(self, g: list[torch.Tensor]) -> list[torch.Tensor]:
        return g

    def _direction(self, g, G, norms, scales):
        if self.groups is None:
            raise RuntimeError("call set_groups(...) before combine")
        a, b = self._inputs(g)
        if sum(k for _, k in self.groups) != a.numel():
            raise ValueError("group sizes do not add up to the gradient length")
        parts, mus, coss, sizes, off = [], [], [], [], 0
        for name, k in self.groups:
            ga, gb = a[off : off + k], b[off : off + k]
            off += k
            Gl = gram_matrix([ga, gb])
            sa = self.group_norm.update_sq(f"{name}/primary", float(Gl[0, 0]))
            sb = self.group_norm.update_sq(f"{name}/secondary", float(Gl[1, 1]))
            na, nb = math.sqrt(max(Gl[0, 0], 0.0)), math.sqrt(max(Gl[1, 1], 0.0))
            coss.append(_cos(Gl[0, 1], na, nb))
            if na == 0.0:  # primary-stationary group: no step, nothing active
                parts.append(torch.zeros_like(ga))
                mus.append(0.0)
            else:
                w, _ = pcd_k2_closed_form(Gl * np.outer([sa, sb], [sa, sb]), self.tau)
                d_tilde = ga * sa + float(w[1]) * sb * gb if w[1] != 0.0 else ga * sa
                parts.append(_rescale(d_tilde, na) if self.rescale else d_tilde)
                mus.append(float(w[1]))
            sizes.append(k)
        total = float(sum(sizes))
        return torch.cat(parts), {
            "tau": self.tau,
            "lambda_eff": sum(m * k for m, k in zip(mus, sizes, strict=True)) / total,
            "active": sum(m > 0 for m in mus) / len(mus),
            "mu_group": mus,
            "cos_group": coss,
        }

    def state_dict(self) -> dict:
        return super().state_dict() | {"group_norm": self.group_norm.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self.group_norm.load_state_dict(state["group_norm"])


class LayerPCDM(LayerPCD):
    """Per-layer PCD on separate momentum buffers (``pcdlm``), following :class:`DOT`.

    m_CE <- beta_ce m_CE + g_CE, m_KD <- beta_kd m_KD + g_KD; the per-layer QP is solved on
    (m_CE, m_KD) and each group is rescaled to ||m_CE^l||. Use with a momentum-free optimizer and
    weight decay folded into the CE gradient by the trainer (as for DOT; ``w`` = loss weights).
    """

    name = "pcdlm"

    def __init__(
        self,
        tau: float,
        beta_ce: float = 0.9,
        beta_kd: float = 0.9,
        beta: float = 0.99,
        eps: float = 1e-8,
        rescale: bool = True,
    ):
        super().__init__(tau, beta, eps, rescale)
        if not (0.0 <= beta_ce < 1.0 and 0.0 <= beta_kd < 1.0):
            raise ValueError("beta_ce and beta_kd must be in [0, 1)")
        self.betas = (float(beta_ce), float(beta_kd))
        self.w = (1.0, 1.0)  # CE weight 1: the trainer's weight-decay term needs w[0]
        self.buf: list[torch.Tensor] | None = None

    def _inputs(self, g):
        if self.buf is None:
            self.buf = [torch.zeros_like(g[0]), torch.zeros_like(g[1])]
        for i in range(2):
            self.buf[i] = self.betas[i] * self.buf[i] + g[i]
        return self.buf

    def state_dict(self) -> dict:
        return super().state_dict() | {
            "buf": None if self.buf is None else [x.clone() for x in self.buf]
        }

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self.buf = None if state["buf"] is None else [x.clone() for x in state["buf"]]
