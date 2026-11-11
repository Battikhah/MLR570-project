"""PCD's predicted strengths and failure modes, checked on constructed gradients.

These are the *mechanisms* the experiment suite is designed to detect at scale; each test names
the suite family that measures its consequence on CIFAR. They hold by construction (algebra),
so passing them says nothing about accuracy -- it says the suite's predictions are well-posed.
"""

import numpy as np
import pytest
import torch

from pcdkd.optim.combiners import PCD, NormalizedWeightedSum, PCGrad, WeightedSum
from pcdkd.optim.combiners_extra import MinNorm


def _pair(c, n=6, seed=0):
    """Unit vectors a, b with cosine c (float64)."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, generator=g, dtype=torch.float64)
    a /= a.norm()
    r = torch.randn(n, generator=g, dtype=torch.float64)
    r -= (r @ a) * a
    r /= r.norm()
    return a, c * a + (1 - c * c) ** 0.5 * r


def _cos(x, y):
    return float(x @ y / (x.norm() * y.norm()))


@pytest.mark.parametrize("m", [1e-3, 1e-1, 10.0, 1e3])
def test_loss_scale_invariance(m):
    """family loss-scale: multiplying L_KD by m changes raw WKD / PCGrad but not PCD or DTO-KD.

    PCD's invariance holds while ||g||^2 >> eps (1e-8): a KD gradient of norm ~1e-4 would hit
    the EMA floor (checked below), which is why the suite logs norm_secondary.
    """
    a, b = _pair(-0.3)
    a, b = 20.0 * a, 20.0 * b
    ref = {k: None for k in ("pcd", "dto", "wkd", "pcgrad")}
    for scale in (1.0, m):
        grads = {"ce": 2.0 * a, "kd": scale * 0.5 * b}
        out = {
            "pcd": PCD(0.5).combine(grads)[0],
            "dto": MinNorm(True).combine(grads, losses={"ce": 1.0, "kd": scale})[0],
            "wkd": WeightedSum(0.9).combine(grads)[0],
            "pcgrad": PCGrad().combine(grads)[0],
        }
        if scale == 1.0:
            ref = out
    assert _cos(out["pcd"], ref["pcd"]) == pytest.approx(1.0, abs=1e-6)
    assert out["pcd"].norm() == pytest.approx(ref["pcd"].norm())  # rescaled to ||g_CE||
    assert _cos(out["dto"], ref["dto"]) == pytest.approx(1.0, abs=1e-9)
    assert _cos(out["wkd"], ref["wkd"]) < 0.999
    assert _cos(out["pcgrad"], ref["pcgrad"]) < 0.999


def test_scale_invariance_breaks_at_eps_floor():
    a, b = _pair(-0.3)
    d1 = PCD(0.5).combine({"ce": a, "kd": b})[0]
    d2 = PCD(0.5).combine({"ce": a, "kd": 1e-4 * b})[0]
    assert _cos(d1, d2) < 0.9999


def test_aligned_teacher_gives_pure_ce_step():
    """H0' (families core, tau-sweep): with c >= tau the constraint is inactive and PCD discards
    the KD gradient entirely -- PCD(CE|KD) at small tau can reduce to CE-only training."""
    a, b = _pair(0.6)
    d, info = PCD(0.1).combine({"ce": a, "kd": b})
    assert not info["active"] and torch.allclose(d, a)
    d, info = PCD(1.0).combine({"ce": a, "kd": b})
    assert info["active"] and _cos(d, a) < 0.999


@pytest.mark.parametrize("tau", [0.1, 0.5, 1.0])
def test_kd_weight_grows_with_conflict(tau):
    """families teacher-corruption / label-noise: implicit KD weight lam = max(0, tau - c) rises as
    the teacher conflicts more with the labels (decision 0001), unlike a fixed NWKD lambda."""
    lams = [
        PCD(tau).combine(dict(zip(("ce", "kd"), _pair(c), strict=True)))[1]["lambda_eff"]
        for c in (0.9, 0.3, 0.0, -0.4, -0.8)
    ]
    assert lams == sorted(lams)
    assert lams[-1] == pytest.approx(tau + 0.8)
    nw = [
        NormalizedWeightedSum(tau).combine(dict(zip(("ce", "kd"), _pair(c), strict=True)))[1][
            "lambda_eff"
        ]
        for c in (0.9, -0.8)
    ]
    assert nw[0] == nw[1]


def test_near_opposition_sacrifices_primary_prop47():
    """PCD Prop. 4.7: at c = -1 the primary moves backwards for every tau > 0 -- the regime
    (strong teacher-label conflict) where CE-primary PCD is predicted to LOSE."""
    a = torch.tensor([1.0, 0.0], dtype=torch.float64)
    d, info = PCD(0.2).combine({"ce": a, "kd": -a})
    assert float(d @ a) < 0
    assert info["prop47_margin_actual"] == pytest.approx(-0.2)


def test_priority_order_matters_kd_primary_keeps_teacher_anchor():
    """family label-noise: with KD as primary (pcdrev), a noisy CE gradient only enters as a
    tau-constraint; the update stays closer to the teacher direction than under CE priority."""
    kd, ce = _pair(-0.3)  # noisy labels: CE disagrees with the (clean) teacher
    pcd = dict(rescale=False, beta=0.0, eps=0.0)  # unit inputs: normalization is the identity
    d_rev, _ = PCD(0.1, **pcd).combine({"kd": kd, "ce": ce}, primary="kd")
    d_fwd, _ = PCD(0.1, **pcd).combine({"ce": ce, "kd": kd}, primary="ce")
    assert _cos(d_rev, kd) > _cos(d_fwd, kd)
    assert float(d_rev @ ce) == pytest.approx(0.1)  # CE gets exactly its tau-progress
    # with unit gradients and tau - c = 1 the two orders give the SAME direction (a + b):
    # the priority order only matters when tau - c != 1
    d_rev, _ = PCD(0.7, **pcd).combine({"kd": kd, "ce": ce}, primary="kd")
    d_fwd, _ = PCD(0.7, **pcd).combine({"ce": ce, "kd": kd}, primary="ce")
    assert torch.allclose(d_rev, d_fwd)
    # symmetric rules have no priority
    d1, _ = MinNorm(True).combine({"ce": ce, "kd": kd}, "ce", {"ce": 1.0, "kd": 1.0})
    d2, _ = MinNorm(True).combine({"ce": ce, "kd": kd}, "kd", {"ce": 1.0, "kd": 1.0})
    assert torch.allclose(d1, d2)


def test_k3_single_tau_guarantees_every_secondary_progress():
    """families multi-loss-* / multi-teacher: one tau gives every secondary tau-progress (when
    feasible), which a fixed weighted sum only achieves for a lucky weight vector."""
    g = torch.Generator().manual_seed(3)
    a = torch.randn(10, generator=g, dtype=torch.float64)
    b = -0.7 * a + 0.3 * torch.randn(10, generator=g, dtype=torch.float64)
    c = 0.2 * a + torch.randn(10, generator=g, dtype=torch.float64)
    comb = PCD(0.3, rescale=False, beta=0.0, eps=0.0)
    d, info = comb.combine({"ce": a, "kd": b, "at": c})
    assert info["feasible"]
    for v in (b, c):
        vn = v / v.norm()  # beta = 0, eps = 0: EMA scale is exactly 1/||g||
        assert float(vn @ d) >= 0.3 * float(vn @ vn) - 1e-9


def test_lambda_tracks_nonstationary_conflict():
    """families low-data / long-tail: when c drifts over training, PCD's coefficient follows it
    step by step; any fixed lambda is right for at most one phase."""
    cs = np.linspace(0.8, -0.6, 15)
    comb = PCD(0.5)
    lams = [
        comb.combine(dict(zip(("ce", "kd"), _pair(float(c)), strict=True)))[1]["lambda_eff"]
        for c in cs
    ]
    assert lams[0] == 0.0 and lams[-1] == pytest.approx(1.1)
    assert np.all(np.diff(lams) >= -1e-12)
