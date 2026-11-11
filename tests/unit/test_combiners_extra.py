"""Baseline combiners: MGDA / DTO-KD min-norm, DOT momenta, symmetric PCGrad, K-objective sums."""

import itertools
import math

import numpy as np
import pytest
import torch

from pcdkd.optim.combiners import PCD, NormalizedWeightedSum
from pcdkd.optim.combiners_extra import (
    DOT,
    MinNorm,
    MultiWeightedSum,
    SymmetricPCGrad,
    min_norm_weights,
    shadow_cosines,
)


def _rand(k, n=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, generator=g, dtype=torch.float64) for _ in range(k)]


def _brute_min_norm(G, steps=200):
    K = G.shape[0]
    best = (math.inf, None)
    grid = np.linspace(0, 1, steps + 1)
    for w in itertools.product(grid, repeat=K - 1):
        if sum(w) <= 1 + 1e-12:
            pi = np.array([*w, 1 - sum(w)])
            best = min(best, (pi @ G @ pi, tuple(pi)), key=lambda t: t[0])
    return best[0]


@pytest.mark.parametrize("k,seed", [(2, 0), (2, 1), (3, 2), (3, 3), (4, 4)])
def test_min_norm_matches_brute_force(k, seed):
    v = _rand(k, seed=seed)
    G = np.array([[float(a @ b) for b in v] for a in v])
    pi = min_norm_weights(G)
    assert pi.min() >= -1e-12 and pi.sum() == pytest.approx(1.0)
    brute = _brute_min_norm(G, steps=400 if k == 2 else (60 if k == 3 else 20))
    assert pi @ G @ pi <= brute + 1e-9


def test_min_norm_k2_closed_form_and_dto_equal_contribution():
    """DTO-KD Cor. 3.3: interior min-norm point has equal inner products with both inputs."""
    g1 = torch.tensor([1.0, 0.2, 0.0], dtype=torch.float64)
    g2 = torch.tensor([-0.3, 1.0, 0.5], dtype=torch.float64)
    L = {"ce": 2.0, "kd": 0.5}
    d, info = MinNorm(log_loss=True).combine({"ce": g1, "kd": g2}, losses=L)
    v1, v2 = g1 / L["ce"], g2 / L["kd"]
    assert 0 < info["pi"][0] < 1
    assert float(d @ v1) == pytest.approx(float(d @ v2), rel=1e-10)
    assert float(d @ v1) > 0  # aligned with both objectives (Cor. 3.2)
    G = np.array([[v1 @ v1, v1 @ v2], [v2 @ v1, v2 @ v2]], dtype=float)
    expected = (G[1, 1] - G[0, 1]) / (G[0, 0] + G[1, 1] - 2 * G[0, 1])
    assert info["pi"][0] == pytest.approx(expected)


def test_dto_requires_losses_and_is_symmetric():
    g = _rand(2)
    with pytest.raises(ValueError):
        MinNorm(log_loss=True).combine({"ce": g[0], "kd": g[1]})
    L = {"ce": 1.3, "kd": 0.7}
    d1, _ = MinNorm(log_loss=True).combine({"ce": g[0], "kd": g[1]}, "ce", L)
    d2, _ = MinNorm(log_loss=True).combine({"ce": g[0], "kd": g[1]}, "kd", L)
    assert torch.allclose(d1, d2)  # no priority: swapping the primary changes nothing


def test_dot_momenta():
    g1, g2 = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    comb = DOT(delta=0.075, momentum=0.9, lam=0.9)
    d1, _ = comb.combine({"ce": g1, "kd": g2})
    assert torch.allclose(d1, torch.tensor([0.1, 0.9]))
    d2, _ = comb.combine({"ce": g1, "kd": g2})
    # task buffer decays with 0.825, KD buffer with 0.975
    assert torch.allclose(d2, torch.tensor([0.1 * 1.825, 0.9 * 1.975]))
    with pytest.raises(ValueError):
        DOT(delta=0.2, momentum=0.9)


def test_symmetric_pcgrad_removes_pairwise_conflict_k2():
    g1 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    g2 = torch.tensor([-1.0, 1.0], dtype=torch.float64)
    d, info = SymmetricPCGrad().combine({"ce": g1, "kd": g2})
    assert info["active"]
    # each projected gradient is orthogonal to the other raw one: p1 = (0.5, 0.5), p2 = (0, 1)
    assert torch.allclose(d, torch.tensor([0.5, 1.5], dtype=torch.float64))
    assert float(d @ g1) > 0 and float(d @ g2) > 0
    d2, info2 = SymmetricPCGrad().combine({"ce": g1, "kd": g1})
    assert not info2["active"] and torch.allclose(d2, 2 * g1)


def test_multi_weighted_sum_matches_two_objective_nwkd_and_raw_sum():
    g = _rand(2, seed=5)
    a, ia = MultiWeightedSum([0.7], normalize=True).combine({"ce": g[0], "kd": g[1]})
    b, _ = NormalizedWeightedSum(0.7).combine({"ce": g[0], "kd": g[1]})
    assert torch.allclose(a, b) and ia["method"] == "nws"
    g3 = _rand(3, seed=6)
    d, info = MultiWeightedSum([2.0, 0.5], normalize=False).combine(
        dict(zip("abc", g3, strict=True)), "a"
    )
    assert torch.allclose(d, g3[0] + 2.0 * g3[1] + 0.5 * g3[2]) and info["method"] == "sum"
    with pytest.raises(ValueError):
        MultiWeightedSum([1.0, 2.0]).combine(dict(zip("abcd", _rand(4), strict=True)), "a")


def test_shadow_cosines_consistent_with_real_combiners():
    g = _rand(2, seed=8)
    grads = {"ce": g[0], "kd": g[1]}
    L = {"ce": 1.7, "kd": 0.4}
    p = PCD(0.5)
    d_pcd, info = p.combine(grads)
    sh = shadow_cosines(info, L, tau=0.5)
    # NWKD direction with lam = 1 from a combiner that saw the same single gradient
    d_nw, _ = NormalizedWeightedSum(1.0).combine(grads)
    d_dto, _ = MinNorm(log_loss=True).combine(grads, losses=L)

    def cos(a, b):
        return float(a @ b / (a.norm() * b.norm()))

    assert sh["shadow_pcd_nwkd"] == pytest.approx(cos(d_pcd, d_nw), abs=1e-9)
    assert sh["shadow_pcd_dto"] == pytest.approx(cos(d_pcd, d_dto), abs=1e-9)
    assert sh["shadow_pcd_ce"] == pytest.approx(cos(d_pcd, g[0]), abs=1e-9)
    assert sh["shadow_pcd_mu"] == pytest.approx(info["mu"][0])
