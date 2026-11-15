import numpy as np
import torch

from pcdkd.optim.combiners import PCD
from pcdkd.optim.combiners_extra import LayerPCD, LayerPCDM


def _grads(n=12, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return {"ce": torch.randn(n, generator=gen), "kd": torch.randn(n, generator=gen)}


def test_single_group_equals_pcd():
    ref, lay = PCD(0.5), LayerPCD(0.5)
    lay.set_groups([("all", 12)])
    for seed in range(4):
        g = _grads(seed=seed)
        d_ref, _ = ref.combine(g)
        d, info = lay.combine(g)
        assert torch.allclose(d, d_ref, atol=1e-6)
        assert len(info["mu_group"]) == 1


def test_groups_get_own_mu_constraint_and_norm():
    g = _grads(10)
    g["ce"][:5], g["kd"][:5] = torch.tensor([1.0, 2, 0, 1, 1]), torch.tensor([1.0, 2, 0, 1, 1])
    g["ce"][5:], g["kd"][5:] = torch.tensor([1.0, 0, 1, 1, 0]), torch.tensor([-1.0, 0, -1, -1, 0])
    tau = 1.0
    lay = LayerPCD(tau, rescale=False)  # before rescale
    lay.set_groups([("a", 5), ("b", 5)])
    d, info = lay.combine(g)
    mu = info["mu_group"]
    assert mu[0] != mu[1] and mu[1] > 0
    # constraint on normalized gradients: <d, g~kd> >= tau ||g~kd||^2 for every group
    for k, sl in enumerate((slice(0, 5), slice(5, 10))):
        v = lay.group_norm.v_hat
        s_kd = 1.0 / np.sqrt(v(f"{'ab'[k]}/secondary") + lay.group_norm.eps)
        gk = g["kd"][sl] * s_kd
        assert float(d[sl] @ gk) >= tau * float(gk @ gk) - 1e-5
    lay2 = LayerPCD(tau)
    lay2.set_groups([("a", 5), ("b", 5)])
    d2, _ = lay2.combine(g)
    for sl in (slice(0, 5), slice(5, 10)):
        assert torch.isclose(d2[sl].norm(), g["ce"][sl].norm(), rtol=1e-5)


def test_pcdlm_zero_beta_equals_pcdl_and_buffers():
    groups = [("a", 5), ("b", 7)]
    a, b = LayerPCD(0.5), LayerPCDM(0.5, 0.0, 0.0)
    a.set_groups(groups), b.set_groups(groups)
    for seed in range(3):
        g = _grads(seed=seed)
        assert torch.allclose(a.combine(g)[0], b.combine(g)[0], atol=1e-6)
    m = LayerPCDM(0.5, 0.9, 0.8)
    m.set_groups(groups)
    g1, g2 = _grads(seed=1), _grads(seed=2)
    m.combine(g1)
    m.combine(g2)
    assert torch.allclose(m.buf[0], 0.9 * g1["ce"] + g2["ce"], atol=1e-6)
    assert torch.allclose(m.buf[1], 0.8 * g1["kd"] + g2["kd"], atol=1e-6)


def test_state_dict_round_trip():
    for make in (lambda: LayerPCD(0.5), lambda: LayerPCDM(0.5)):
        c1, c2 = make(), make()
        c1.set_groups([("a", 5), ("b", 7)]), c2.set_groups([("a", 5), ("b", 7)])
        for seed in range(3):
            c1.combine(_grads(seed=seed))
        c2.load_state_dict(c1.state_dict())
        g = _grads(seed=9)
        d1, i1 = c1.combine(g)
        d2, i2 = c2.combine(g)
        assert torch.equal(d1, d2) and i1["mu_group"] == i2["mu_group"]
