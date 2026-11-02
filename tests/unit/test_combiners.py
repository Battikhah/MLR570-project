import json
import math

import pytest
import torch

from pcdkd.optim.combiners import (
    PCD,
    AdaptiveLambda,
    NormalizedWeightedSum,
    PCGrad,
    WeightedSum,
)
from pcdkd.optim.grads import flat_grad, unflatten_to

REQUIRED_KEYS = {
    "cos_raw",
    "dot_norm",
    "active",
    "feasible",
    "lambda_eff",
    "prop47_margin",
    "norm_primary",
    "norm_secondary",
}


def _stream(seed, n=40, dim=25, conflict=0.8, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    for t in range(n):
        g1 = torch.randn(dim, generator=g, dtype=dtype) * (1 + 0.1 * t)
        g2 = 3.0 * (-conflict * g1 + torch.randn(dim, generator=g, dtype=dtype))
        yield g1, g2


ALL = [
    lambda: WeightedSum(0.5),
    lambda: NormalizedWeightedSum(0.4),
    lambda: PCGrad(),
    lambda: PCD(0.1),
    lambda: AdaptiveLambda(0.1),
]


@pytest.mark.parametrize("make", ALL)
def test_info_keys_and_json(make):
    comb = make()
    for g1, g2 in _stream(0, n=3):
        d, info = comb.combine({"kd": g2, "ce": g1})  # primary need not be first in the dict
        assert REQUIRED_KEYS <= set(info)
        json.dumps(info, allow_nan=False)
        assert d.shape == g1.shape and torch.isfinite(d).all()
        assert info["cos_raw"] == pytest.approx(
            float(g1 @ g2) / float(g1.norm() * g2.norm()), abs=1e-5
        )


def test_weighted_sum():
    g1, g2 = torch.randn(5), torch.randn(5)
    d, info = WeightedSum(0.25).combine({"ce": g1, "kd": g2})
    assert torch.allclose(d, 0.75 * g1 + 0.25 * g2)
    assert info["lambda_eff"] == pytest.approx(1 / 3) and not info["active"]


def test_nwkd_rescale_and_formula():
    comb = NormalizedWeightedSum(0.5, rescale=False)
    g1, g2 = torch.randn(9, dtype=torch.float64), torch.randn(9, dtype=torch.float64)
    d, info = comb.combine({"ce": g1, "kd": g2})
    s1, s2 = info["scale_primary"], info["scale_secondary"]
    assert torch.allclose(d, s1 * g1 + 0.5 * s2 * g2)
    d2, _ = NormalizedWeightedSum(0.5).combine({"ce": g1, "kd": g2})
    assert float(d2.norm()) == pytest.approx(float(g1.norm()), rel=1e-12)


# ---- identity checks (required test 4) ----


def test_pcd_equals_nwkd_with_alpha_in_active_case():
    pcd = PCD(0.3, beta=0.9)
    n_active = 0
    for g1, g2 in _stream(1):
        state = pcd.state_dict()
        d_pcd, info = pcd.combine({"ce": g1, "kd": g2})
        if not info["active"]:
            continue
        n_active += 1
        nwkd = NormalizedWeightedSum(info["lambda_eff"], beta=0.9)
        nwkd.load_state_dict(state)  # same EMA state before the step
        d_nw, _ = nwkd.combine({"ce": g1, "kd": g2})
        assert torch.allclose(d_pcd, d_nw, rtol=0, atol=1e-6)
        assert pcd.normalizer.state_dict() == nwkd.normalizer.state_dict()
    assert n_active >= 20


@pytest.mark.parametrize("tau", [0.0, 0.1, 0.5])
@pytest.mark.parametrize("rescale", [True, False])
@pytest.mark.parametrize(
    ("dtype", "lam_tol"), [(torch.float64, 1e-9), (torch.float32, 1e-5)]
)  # float32: PCD's Gram is formed from float32 normalized vectors -> ~1e-8 relative rounding
def test_adaptive_lambda_equals_pcd(tau, rescale, dtype, lam_tol):
    pcd, ada = PCD(tau, rescale=rescale), AdaptiveLambda(tau, rescale=rescale)
    seen = set()
    for g1, g2 in _stream(2, dim=3, conflict=0.0, dtype=dtype):
        d1, i1 = pcd.combine({"ce": g1, "kd": g2})
        d2, i2 = ada.combine({"ce": g1, "kd": g2})
        if dtype == torch.float64:
            assert torch.allclose(d1, d2, rtol=0, atol=1e-6)
        else:  # float32 last-ulp differences on |d| ~ 10
            assert torch.allclose(d1, d2, rtol=1e-5, atol=1e-6)
        assert i1["active"] == i2["active"]
        assert i1["lambda_eff"] == pytest.approx(i2["lambda_eff"], rel=lam_tol, abs=lam_tol)
        seen.add(i1["active"])
    assert seen == {True, False}  # both regimes exercised
    # boundary: primary-stationary halt (PCD Alg. 1 lines 7-8) -- identical diagnostics too
    z = torch.zeros(3, dtype=dtype)
    g2 = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    d1, i1 = pcd.combine({"ce": z, "kd": g2})
    d2, i2 = ada.combine({"ce": z, "kd": g2})
    assert torch.all(d1 == 0) and torch.all(d2 == 0)
    for key in ("active", "lambda_eff", "lambda_raw", "prop47_margin_actual"):
        assert i1[key] == i2[key], key
    assert i2["active"] is False and i2["lambda_eff"] == 0.0 and i2["lambda_raw"] == 0.0


def test_pcd_alpha_is_tau_minus_c_for_unit_vectors():
    """Cor. 4.6: with ||g~1|| = ||g~2|| = 1 (eps = 0, first step), lambda_t = max(0, tau - c)."""
    g = torch.Generator().manual_seed(5)
    for _ in range(20):
        pcd = PCD(0.2, eps=0.0)
        g1, g2 = (
            torch.randn(7, generator=g, dtype=torch.float64),
            torch.randn(7, generator=g, dtype=torch.float64),
        )
        _, info = pcd.combine({"ce": g1, "kd": g2})
        assert info["lambda_eff"] == pytest.approx(max(0.0, 0.2 - info["cos_raw"]), abs=1e-12)
        assert info["dot_norm"] == pytest.approx(info["cos_raw"], abs=1e-12)


# ---- PCGrad (required test 8) ----


def test_pcgrad_projects_only_when_conflicting():
    g1 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    g2c = torch.tensor([-2.0, 1.0, 0.5], dtype=torch.float64)  # cos < 0
    d, info = PCGrad().combine({"ce": g1, "kd": g2c})
    assert info["active"]
    assert torch.allclose(d, g1 + torch.tensor([0.0, 1.0, 0.5], dtype=torch.float64))
    assert float(g1 @ d) == pytest.approx(float(g1 @ g1))  # projection onto CE preserved
    g2a = torch.tensor([2.0, 1.0, 0.5], dtype=torch.float64)  # cos > 0: untouched
    d, info = PCGrad().combine({"ce": g1, "kd": g2a})
    assert not info["active"] and torch.allclose(d, g1 + g2a)


def test_pcgrad_random_property():
    for g1, g2 in _stream(3, n=50, conflict=0.2):
        d, info = PCGrad().combine({"ce": g1, "kd": g2})
        dot = float(g1 @ g2)
        assert info["active"] == (dot < 0)
        expected = float(g1 @ g1) + max(0.0, dot)
        assert float(g1 @ d) == pytest.approx(expected, rel=1e-4, abs=1e-4)


# ---- misc ----


def test_k3_pcd_and_two_objective_guard():
    g = torch.Generator().manual_seed(9)
    grads = {k: torch.randn(6, generator=g) for k in ("ce", "kd", "reg")}
    d, info = PCD(0.1).combine(grads)
    assert len(info["mu"]) == 2 and info["secondaries"] == ["kd", "reg"]
    with pytest.raises(ValueError):
        NormalizedWeightedSum(0.5).combine(grads)
    with pytest.raises(KeyError):
        PCD(0.1).combine({"a": torch.ones(2), "b": torch.ones(2)})


def test_flat_grad_and_unflatten_roundtrip():
    torch.manual_seed(0)
    lin = torch.nn.Linear(3, 2)
    unused = torch.nn.Parameter(torch.ones(4))
    params = [*lin.parameters(), unused]
    x = torch.randn(5, 3)
    loss = lin(x).pow(2).sum()
    v = flat_grad(loss, params, retain_graph=True)
    assert v.shape == (3 * 2 + 2 + 4,) and torch.all(v[-4:] == 0)
    loss.backward()
    ref = torch.cat([lin.weight.grad.reshape(-1), lin.bias.grad])
    assert torch.allclose(v[:8], ref)
    unflatten_to(params, 2 * v)
    assert torch.allclose(lin.weight.grad, 2 * ref[:6].view(2, 3))
    assert torch.all(unused.grad == 0)
    with pytest.raises(ValueError):
        unflatten_to(params, v[:-1])
    assert torch.all(flat_grad(torch.tensor(1.0), params, False) == 0)


def test_bad_hyperparameters():
    with pytest.raises(ValueError):
        WeightedSum(1.0)
    with pytest.raises(ValueError):
        PCD(-0.1)
    with pytest.raises(ValueError):
        NormalizedWeightedSum(math.inf)


def test_adaptive_lambda_zero_primary_reviewer_input():
    """Codex review F4: g1 = [0, 0], g2 = [1, 0], tau = 0.2 must not report an active step."""
    g1, g2 = torch.zeros(2), torch.tensor([1.0, 0.0])
    _, ip = PCD(0.2).combine({"ce": g1, "kd": g2})
    d, ia = AdaptiveLambda(0.2).combine({"ce": g1, "kd": g2})
    assert torch.all(d == 0)
    assert (ia["active"], ia["lambda_eff"], ia["lambda_raw"]) == (False, 0.0, 0.0)
    assert (ip["active"], ip["lambda_eff"], ip["lambda_raw"]) == (False, 0.0, 0.0)


def test_dot_norm_is_inner_product_not_cosine():
    """Codex review F2: dot_norm = g~1.g~2 may leave [-1, 1]; cos_raw is the true angle."""
    comb = NormalizedWeightedSum(0.5, beta=0.9)
    g1 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    comb.combine({"ce": g1, "kd": g1})  # warm-up at norm 1
    _, info = comb.combine({"ce": 10 * g1, "kd": 10 * g1})  # norm jump -> ||g~|| > 1
    assert "cos_norm" not in info
    assert info["cos_raw"] == pytest.approx(1.0)
    s1, s2 = info["scale_primary"], info["scale_secondary"]
    assert info["dot_norm"] == pytest.approx(s1 * s2 * 100.0)
    assert info["dot_norm"] > 1.0


def test_nwkd_actual_margin():
    comb = NormalizedWeightedSum(0.7)
    g1 = torch.tensor([1.0, 2.0, 0.0], dtype=torch.float64)
    g2 = torch.tensor([-3.0, 0.5, 1.0], dtype=torch.float64)
    d, info = comb.combine({"ce": g1, "kd": g2})
    g1n = info["scale_primary"] * g1
    expected = 1.0 + 0.7 * info["dot_norm"] / float(g1n @ g1n)
    assert info["prop47_margin_actual"] == pytest.approx(expected, rel=1e-12)
    # the final rescale preserves the sign of the margin
    assert math.copysign(1, float(g1 @ d)) == math.copysign(1, expected)
    assert info["prop47_margin_unit"] is None  # no tau


def test_one_sided_pcgrad_method_name_is_unambiguous():
    _, info = PCGrad().combine({"ce": torch.ones(2), "kd": -torch.ones(2)})
    assert info["method"] == "pcgrad_onto_primary"
    from pcdkd.optim.combiners import COMBINERS

    assert "pcgrad" not in COMBINERS and COMBINERS["pcgrad_onto_primary"] is PCGrad
