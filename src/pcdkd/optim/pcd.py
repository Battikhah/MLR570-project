"""The PCD quadratic program (PCD Def. 4.1 / Eq. 4) on already-normalized gradients.

    d~ = argmin_d 1/2 ||d - g~_1||^2   s.t.   g~_j^T d >= tau ||g~_j||^2,  j = 2..K

By KKT (PCD Thm. 4.2) d~ = g~_1 + sum_j mu_j g~_j with mu_j >= 0, so the QP is solved on the
K x K Gram matrix (float64, CPU, NumPy) and the direction is formed on the gradients' own
device and dtype.

* K = 2: closed form (PCD Cor. 4.6).
* 3 <= K <= 4: KKT working-set enumeration in order of increasing size (PCD App. B.2). This
  follows the algorithm of the official ``pcd/qp.py`` (MIT,
  DaraVaram/priority-constrained-descent @ e9afdbc) with the same tolerances, and is
  cross-checked against it in ``tests/unit/test_pcd.py``.
* Infeasible constraint set (only possible for K >= 3): fall back to g~_1 (PCD Alg. 1, l. 14).
* Zero secondaries (||g~_j|| = 0, constraint 0 >= 0) are skipped.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np
import torch

MAX_K = 4


def _as_taus(tau: float | Sequence[float], k: int) -> np.ndarray:
    taus = np.broadcast_to(np.asarray(tau, dtype=np.float64), (k - 1,)).copy()
    if np.any(taus < 0) or not np.all(np.isfinite(taus)):
        raise ValueError(f"tau must be finite and non-negative, got {tau}")
    return taus


def pcd_k2_closed_form(gram: np.ndarray, tau: float) -> tuple[np.ndarray, tuple[int, ...]]:
    """PCD Cor. 4.6 on a 2 x 2 Gram matrix. Returns (weights, active); weights[0] == 1."""
    G = np.asarray(gram, dtype=np.float64)
    if G.shape != (2, 2):
        raise ValueError(f"gram must be 2 x 2, got {G.shape}")
    tau = float(_as_taus(tau, 2)[0])
    w = np.array([1.0, 0.0])
    if G[1, 1] <= 0.0 or G[1, 0] >= tau * G[1, 1]:  # zero secondary or inactive constraint
        return w, ()
    w[1] = tau - G[1, 0] / G[1, 1]
    return w, (1,)


def pcd_kkt_enumeration(
    gram: np.ndarray, tau: float | Sequence[float], tol: float = 1e-9
) -> tuple[np.ndarray, tuple[int, ...], bool]:
    """General-K solver by KKT working-set enumeration. Returns (weights, active, feasible)."""
    G = np.asarray(gram, dtype=np.float64)
    K = G.shape[0]
    if G.ndim != 2 or G.shape != (K, K):
        raise ValueError(f"gram must be square, got {G.shape}")
    w = np.zeros(K)
    w[0] = 1.0
    if K == 1:
        return w, (), True
    taus = _as_taus(tau, K)
    Gs = G[1:, 1:]
    b = taus * np.diag(Gs) - G[1:, 0]  # b_j > 0  <=>  g~_1 violates constraint j
    if np.all(b <= 0):  # PCD Cor. 4.3: g~_1 feasible
        return w, (), True
    atol = tol * max(float(np.max(np.diag(G))), np.finfo(np.float64).tiny)
    candidates = [j for j in range(K - 1) if Gs[j, j] > 0]
    for size in range(1, len(candidates) + 1):
        for subset in combinations(candidates, size):
            A = list(subset)
            GA = Gs[np.ix_(A, A)]
            if not _independent(GA, tol):
                continue
            mu = np.linalg.solve(GA, b[A])
            if np.any(mu < -tol):
                continue
            mu = np.maximum(mu, 0.0)
            slack = Gs[:, A] @ mu - b
            if np.all(slack >= -atol):
                w[1 + np.asarray(A)] = mu
                return w, tuple(1 + j for j in A), True
    return w, (), False


def _independent(GA: np.ndarray, tol: float) -> bool:
    if GA.shape[0] == 1:
        return True
    d = np.sqrt(np.diag(GA))
    return bool(np.linalg.eigvalsh(GA / np.outer(d, d))[0] > tol)


def primary_margin(gram_normalized, weights: Sequence[float]) -> float | None:
    """Actual primary margin g~1^T d~ / ||g~1||^2 for d~ = sum_i w_i g~_i (None if g~1 = 0).

    Under ||g~1|| = ||g~2|| = 1 and an active K = 2 constraint this equals PCD Prop. 4.7's
    1 + tau c - c^2; in general (EMA-normalized, non-unit vectors) it does not.
    """
    G = np.asarray(gram_normalized, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if G[0, 0] <= 0.0:
        return None
    return float(w @ G[0, : len(w)] / G[0, 0])


def gram_matrix(vectors: Sequence[torch.Tensor]) -> np.ndarray:
    """K x K float64 Gram matrix of 1-D tensors (dots on device, assembled on CPU)."""
    K = len(vectors)
    dots = torch.stack(
        [torch.dot(vectors[i], vectors[j]) for i in range(K) for j in range(i, K)]
    ).detach()
    dots = dots.cpu().double().numpy()  # move to CPU before float64 (MPS has no float64)
    G = np.empty((K, K))
    it = iter(dots)
    for i in range(K):
        for j in range(i, K):
            G[i, j] = G[j, i] = next(it)
    return G


def solve_pcd_qp(
    G_normalized_list: Sequence[torch.Tensor],
    tau: float | Sequence[float],
    tol: float = 1e-9,
) -> tuple[torch.Tensor, dict]:
    """Solve the PCD QP for normalized 1-D gradients (primary first).

    Returns ``(d_tilde, info)`` with info keys: ``weights`` (list of K, weights[0] == 1),
    ``mu`` (K-1 multipliers), ``active`` (tuple of 1-based secondary indices whose constraint
    binds), ``feasible``, ``gram`` (K x K nested list), ``primary_margin``
    (g~1^T d~ / ||g~1||^2; None if g~1 = 0) and ``method`` ("trivial" | "closed_form" | "kkt").
    """
    vecs = list(G_normalized_list)
    K = len(vecs)
    if K == 0:
        raise ValueError("need at least the primary gradient")
    if K > MAX_K:
        raise ValueError(f"solve_pcd_qp supports K <= {MAX_K}, got {K}")
    if any(v.dim() != 1 or v.shape != vecs[0].shape for v in vecs):
        raise ValueError("all gradients must be 1-D with the same length")
    G = gram_matrix(vecs)
    if K == 1:
        w, active, feasible, method = np.array([1.0]), (), True, "trivial"
    elif K == 2:
        w, active = pcd_k2_closed_form(G, float(np.asarray(tau).reshape(-1)[0]))
        feasible, method = True, "closed_form"
    else:
        w, active, feasible = pcd_kkt_enumeration(G, tau, tol)
        method = "kkt"
    d = vecs[0] * float(w[0])
    for i in range(1, K):
        if w[i] != 0.0:
            d = d + float(w[i]) * vecs[i]
    info = {
        "weights": w.tolist(),
        "mu": w[1:].tolist(),
        "active": active,
        "feasible": bool(feasible),
        "gram": G.tolist(),
        "primary_margin": primary_margin(G, w),
        "method": method,
    }
    return d, info
