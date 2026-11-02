import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pcdkd.optim.combiners import PCD
from pcdkd.optim.pcd import pcd_k2_closed_form, pcd_kkt_enumeration, solve_pcd_qp

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "pcd_reference_qp.json"


def _vecs(V):
    return [torch.as_tensor(v, dtype=torch.float64) for v in np.asarray(V)]


@pytest.mark.parametrize("seed", range(30))
def test_k2_closed_form_equals_general_solver(seed):
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((2, 12))
    V[1] -= rng.uniform(0, 1.5) * V[0]
    tau = float(rng.uniform(0, 1))
    G = V @ V.T
    w_cf, act_cf = pcd_k2_closed_form(G, tau)
    w_kkt, act_kkt, feas = pcd_kkt_enumeration(G, tau)
    assert feas and act_cf == act_kkt
    np.testing.assert_allclose(w_cf, w_kkt, rtol=0, atol=1e-12)


def test_inactive_returns_primary():
    g1 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    g2 = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)  # g2.g1 = 1 >= tau*2 for tau <= 0.5
    d, info = solve_pcd_qp([g1, g2], 0.3)
    assert torch.equal(d, g1) and info["active"] == () and info["feasible"]


def test_zero_secondary_skipped():
    g1 = torch.tensor([1.0, 2.0], dtype=torch.float64)
    z = torch.zeros(2, dtype=torch.float64)
    d, info = solve_pcd_qp([g1, z], 0.5)
    assert torch.equal(d, g1) and info["active"] == ()
    g3 = torch.tensor([-1.0, 0.0], dtype=torch.float64)
    d, info = solve_pcd_qp([g1, z, g3], 0.5)
    assert info["active"] == (2,) and info["mu"][0] == 0.0
    assert float(g3 @ d) >= 0.5 * float(g3 @ g3) - 1e-9


def test_tau_zero_vs_positive():
    g1 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    g2 = torch.tensor([-1.0, 1.0], dtype=torch.float64)  # conflicting
    d0, _ = solve_pcd_qp([g1, g2], 0.0)
    assert abs(float(g2 @ d0)) < 1e-12  # tau = 0: projected onto the orthogonal complement
    d3, _ = solve_pcd_qp([g1, g2], 0.3)
    assert float(g2 @ d3) == pytest.approx(0.3 * float(g2 @ g2), abs=1e-12)
    assert float(g1 @ d3) < float(g1 @ d0)  # larger tau sacrifices more primary progress


@pytest.mark.parametrize("K", [2, 3, 4])
def test_constraints_satisfied_when_feasible(K):
    rng = np.random.default_rng(100 + K)
    for _ in range(200):
        V = rng.standard_normal((K, 8))
        V[1:] -= rng.uniform(0, 1.5) * V[0]
        tau = float(rng.uniform(0, 1))
        vs = _vecs(V)
        d, info = solve_pcd_qp(vs, tau)
        if info["feasible"]:
            for g in vs[1:]:
                assert float(g @ d) >= tau * float(g @ g) - 1e-6
            assert all(m >= 0 for m in info["mu"])


@pytest.mark.parametrize("K", [3, 4])
def test_antiparallel_secondaries_infeasible_fallback(K):
    rng = np.random.default_rng(K)
    V = rng.standard_normal((K, 5))
    V[2] = -V[1]
    vs = _vecs(V)
    d, info = solve_pcd_qp(vs, 0.2)
    assert not info["feasible"] and info["active"] == ()
    assert torch.equal(d, vs[0])
    # tau = 0 with anti-parallel secondaries is feasible (the orthogonal complement)
    _, info0 = solve_pcd_qp(vs, 0.0)
    assert info0["feasible"]


def test_k_limit_and_shapes():
    vs = [torch.randn(3, dtype=torch.float64) for _ in range(5)]
    with pytest.raises(ValueError):
        solve_pcd_qp(vs, 0.1)
    with pytest.raises(ValueError):
        solve_pcd_qp([torch.randn(3), torch.randn(4)], 0.1)
    with pytest.raises(ValueError):
        solve_pcd_qp(vs[:2], -0.1)


def test_prop47_identity_unit_vectors():
    """Prop. 4.7: for unit g~1, g~2 in the active regime, g~1^T d~ = 1 + tau c - c^2."""
    rng = np.random.default_rng(47)
    checked = 0
    for _ in range(500):
        V = rng.standard_normal((2, 10))
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        tau = float(rng.uniform(0, 1))
        c = float(V[0] @ V[1])
        vs = _vecs(V)
        d, info = solve_pcd_qp(vs, tau)
        if c < tau:  # active regime for unit vectors
            assert info["active"] == (1,)
            assert float(vs[0] @ d) == pytest.approx(1 + tau * c - c * c, abs=1e-12)
            checked += 1
        else:
            assert info["active"] == ()
    assert checked > 200


def _fixture():
    return json.loads(FIXTURE.read_text())


def test_fixture_provenance():
    doc = _fixture()
    assert doc["provenance"]["commit"].startswith("e9afdbc")
    assert "MIT" in doc["provenance"]["license"]
    kinds = {p["kind"] for p in doc["problems"]}
    assert {"random", "inactive", "infeasible", "zero_secondary"} <= kinds
    assert {p["K"] for p in doc["problems"]} == {2, 3, 4}
    assert len(doc["problems"]) >= 50


@pytest.mark.parametrize("idx", range(len(json.loads(FIXTURE.read_text())["problems"])))
def test_matches_official_qp(idx):
    p = _fixture()["problems"][idx]
    vs = _vecs(p["vectors"])
    d, info = solve_pcd_qp(vs, p["tau"])
    np.testing.assert_allclose(info["weights"], p["weights"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(d.numpy(), p["direction"], rtol=0, atol=1e-8)
    assert list(info["active"]) == p["active"]
    assert info["feasible"] == p["feasible"]


# ---- PCD combiner: rescale and boundary branches (Alg. 1) ----


def _pair(seed, conflict=1.0):
    g = torch.Generator().manual_seed(seed)
    g1 = torch.randn(30, generator=g)
    g2 = -conflict * g1 + 0.5 * torch.randn(30, generator=g)
    return g1, g2


def test_rescale_norm_equals_primary_norm():
    pcd = PCD(0.3, beta=0.9)
    for s in range(10):
        g1, g2 = _pair(s)
        d, info = pcd.combine({"ce": g1, "kd": 7.0 * g2})
        assert float(d.norm()) == pytest.approx(float(g1.norm()), rel=1e-5)
        assert info["active"]


def test_zero_primary_gives_zero_step():
    pcd = PCD(0.3)
    g1, g2 = _pair(0)
    d, info = pcd.combine({"ce": torch.zeros_like(g1), "kd": g2})
    assert torch.all(d == 0) and info["zero_primary"]
    assert pcd.normalizer.t == {"ce": 1, "kd": 1}  # EMA still updated (Alg. 1 line 5)


def test_zero_dtilde_gives_zero_step():
    # g1 = -g2 exactly, unit norms at step 1, tau = 0: d~ = g~1 - (c/||g~2||^2) g~2 = g~1 + g~2 ~ 0
    g1 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    pcd = PCD(0.0, eps=0.0)
    d, info = pcd.combine({"ce": g1, "kd": -g1})
    assert info["active"] and torch.all(d == 0)


def test_rescale_false_returns_normalized_direction():
    g1, g2 = _pair(3)
    a, b = PCD(0.2, rescale=False), PCD(0.2, rescale=True)
    da, _ = a.combine({"ce": g1, "kd": g2})
    db, _ = b.combine({"ce": g1, "kd": g2})
    cos = float(da @ db) / float(da.norm() * db.norm())
    assert cos == pytest.approx(1.0, abs=1e-6)
