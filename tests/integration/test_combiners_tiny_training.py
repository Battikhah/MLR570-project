"""Tiny end-to-end training with every combiner (CPU, synthetic data, 50 steps each)."""

import json
import math

import pytest
import torch
import torch.nn.functional as F

from pcdkd.eval.diagnostics import DiagnosticsAggregator
from pcdkd.losses.gate import reliability_mask
from pcdkd.losses.kd import kd_loss
from pcdkd.optim.combiners import (
    PCD,
    AdaptiveLambda,
    NormalizedWeightedSum,
    PCGrad,
    WeightedSum,
)
from pcdkd.optim.grads import flat_grad, unflatten_to

pytestmark = pytest.mark.integration

FACTORIES = {
    "wkd": lambda: WeightedSum(0.5),
    "nwkd": lambda: NormalizedWeightedSum(0.3),
    "pcgrad": lambda: PCGrad(),
    "pcd": lambda: PCD(0.1),
    "adaptive_lambda": lambda: AdaptiveLambda(0.1),
}


def _data(seed=0, n=256, dim=8, corrupt=0.3):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, dim, generator=g)
    w = torch.randn(dim, generator=g)
    y = (x @ w > 0).long()
    # teacher: confident and right, except a corrupted fraction that is confident and wrong
    flip = torch.rand(n, generator=g) < corrupt
    t_label = torch.where(flip, 1 - y, y)
    t_logits = torch.zeros(n, 2)
    t_logits[torch.arange(n), t_label] = 2.0 * (x @ w).abs() + 1.0
    return x, y, torch.softmax(t_logits, dim=1), flip


def _train(name, gate_mode="none", seed=0, steps=50, batch=32, T=2.0):
    torch.manual_seed(seed)
    x, y, p_t, _ = _data(seed)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 2))
    params = list(model.parameters())
    opt = torch.optim.SGD(params, lr=0.1)
    comb = FACTORIES[name]()
    agg = DiagnosticsAggregator()
    order = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(seed + 1))
    losses = []
    for step in range(steps):
        idx = order[(step * batch) % x.shape[0] :][:batch]
        logits = model(x[idx])
        ce = F.cross_entropy(logits, y[idx])
        mask = reliability_mask(p_t[idx], y[idx], 0.6, gate_mode)
        kd = kd_loss(logits, p_t[idx], T, mask=mask)
        grads = {
            "ce": flat_grad(ce, params, retain_graph=True),
            "kd": flat_grad(kd, params, retain_graph=False),
        }
        d, info = comb.combine(grads, primary="ce")
        info["gate_accept"] = float(mask.float().mean())
        agg.update(info)
        opt.zero_grad(set_to_none=True)
        unflatten_to(params, d)
        opt.step()
        losses.append((ce.item(), kd.item()))
    return model, losses, agg


@pytest.mark.parametrize("name", list(FACTORIES))
def test_training_finite_populated_deterministic(name):
    m1, l1, a1 = _train(name)
    m2, l2, a2 = _train(name)
    assert all(math.isfinite(c) and math.isfinite(k) for c, k in l1)
    s = a1.summary()
    assert s["steps"] == 50
    for key in ("cos_raw", "cos_norm", "norm_primary", "norm_secondary", "lambda_eff"):
        assert s["numeric"][key]["count"] == 50, key
    assert set(s["fractions"]) >= {"active", "feasible"}
    assert 0.0 <= s["fractions"]["active"] <= 1.0
    if name in ("pcd", "adaptive_lambda"):
        assert s["numeric"]["prop47_margin"]["count"] == 50
        assert s["prop47_nonpositive_fraction"] is not None
        print(f"{name}: active fraction = {s['fractions']['active']:.3f}")
    # determinism across two identical runs
    assert l1 == l2
    for p, q in zip(m1.parameters(), m2.parameters(), strict=True):
        assert torch.equal(p, q)
    assert a1.to_json() == a2.to_json()
    json.loads(a1.to_json())


def test_gated_pcd_runs_and_gate_rejects():
    _, losses, agg = _train("pcd", gate_mode="correct_and_conf")
    s = agg.summary()
    assert all(math.isfinite(c) and math.isfinite(k) for c, k in losses)
    assert s["numeric"]["gate_accept"]["mean"] < 1.0


def test_corruption_is_confident_and_wrong():
    _, y, p_t, flip = _data()
    assert flip.any()
    assert torch.all(p_t[flip].argmax(1) != y[flip])
    assert torch.all(p_t[~flip].argmax(1) == y[~flip])
    assert torch.all(p_t.max(1).values > 0.7)
