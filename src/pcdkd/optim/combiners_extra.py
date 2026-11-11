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
* ``SymmetricPCGrad(lams)``: gradient surgery of Yu et al. (2020), random projection order.
* ``shadow_cosines``: from one K=2 ``info`` dict, the pairwise cosines of the directions that
  CE-only, PCD, NWKD, WKD and DTO-KD *would* take on the same batch (no extra backward).
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import torch

from pcdkd.optim.combiners import Combiner, _rescale
from pcdkd.optim.pcd import MAX_K, pcd_k2_closed_form


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
