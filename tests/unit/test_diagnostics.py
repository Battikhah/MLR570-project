import json

import pytest
import torch

from pcdkd.eval.diagnostics import DiagnosticsAggregator
from pcdkd.optim.combiners import PCD, WeightedSum


def test_means_fractions_histograms():
    agg = DiagnosticsAggregator(n_bins=4)
    infos = [
        {
            "cos_raw": -0.9,
            "active": True,
            "prop47_margin_actual": -0.1,
            "lambda_eff": 1.0,
            "x": None,
        },
        {
            "cos_raw": 0.1,
            "active": False,
            "prop47_margin_actual": 0.5,
            "lambda_eff": 3.0,
            "s": "pcd",
        },
        {"cos_raw": 1.0, "active": True, "prop47_margin_actual": 0.0, "lambda_eff": float("nan")},
    ]
    for i in infos:
        agg.update(i)
    s = agg.summary()
    assert s["steps"] == 3
    assert s["numeric"]["lambda_eff"]["mean"] == pytest.approx(2.0)
    assert s["numeric"]["lambda_eff"]["nonfinite"] == 1
    assert s["numeric"]["cos_raw"]["min"] == -0.9 and s["numeric"]["cos_raw"]["max"] == 1.0
    assert s["fractions"]["active"] == pytest.approx(2 / 3)
    assert s["prop47_nonpositive_fraction"] == pytest.approx(2 / 3)
    h = s["histograms"]["cos_raw"]
    assert h["counts"] == [1, 0, 1, 1] and h["below"] == 0 and h["above"] == 0
    assert "x" not in s["numeric"] and "s" not in s["numeric"]
    json.loads(agg.to_json())


def test_out_of_range_and_reset():
    agg = DiagnosticsAggregator()
    agg.update({"dot_norm": 1.3})
    agg.update({"dot_norm": -1.2})
    h = agg.summary()["histograms"]["dot_norm"]
    assert h["above"] == 1 and h["below"] == 1 and sum(h["counts"]) == 0
    agg.reset()
    assert agg.summary()["steps"] == 0 and agg.fraction("active") is None


def test_with_real_combiner_infos():
    g = torch.Generator().manual_seed(0)
    agg_p, agg_w = DiagnosticsAggregator(), DiagnosticsAggregator()
    pcd, wkd = PCD(0.2), WeightedSum(0.5)
    for _ in range(30):
        g1 = torch.randn(10, generator=g)
        g2 = -0.5 * g1 + torch.randn(10, generator=g)
        agg_p.update(pcd.combine({"ce": g1, "kd": g2})[1])
        agg_w.update(wkd.combine({"ce": g1, "kd": g2})[1])
    sp, sw = agg_p.summary(), agg_w.summary()
    assert 0.0 <= sp["fractions"]["active"] <= 1.0
    assert sp["numeric"]["prop47_margin_actual"]["count"] == 30
    assert sp["numeric"]["prop47_margin_unit"]["count"] == 30
    assert sw["prop47_nonpositive_fraction"] is None  # not applicable to WKD
    assert sum(sp["histograms"]["cos_raw"]["counts"]) == 30
    json.loads(agg_p.to_json())


def test_nonpositive_fraction_uses_actual_not_unit_margin():
    """Codex review F1: the unit-norm formula can be positive while the actual margin is not."""
    agg = DiagnosticsAggregator()
    agg.update({"prop47_margin_unit": 0.28, "prop47_margin_actual": -0.44})
    agg.update({"prop47_margin_unit": 0.5, "prop47_margin_actual": 0.7})
    agg.update({"prop47_margin_unit": None, "prop47_margin_actual": None})  # halt step
    s = agg.summary()
    assert s["prop47_nonpositive_fraction"] == pytest.approx(0.5)
    assert s["numeric"]["prop47_margin_actual"]["count"] == 2
