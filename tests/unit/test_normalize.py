import math

import pytest
import torch

from pcdkd.optim.combiners import PCD, NormalizedWeightedSum
from pcdkd.optim.normalize import EMANormalizer


def test_bias_correction_first_step_and_constant_norm():
    n = EMANormalizer(beta=0.99, eps=1e-8)
    g = torch.full((10,), 3.0, dtype=torch.float64)  # ||g||^2 = 90
    for _ in range(5):
        s = n.scale("a", g)
        assert n.v_hat("a") == pytest.approx(90.0, rel=1e-12)
        assert s == pytest.approx(1 / math.sqrt(90.0 + 1e-8), rel=1e-12)


def test_matches_manual_recursion():
    torch.manual_seed(0)
    beta, eps = 0.9, 1e-8
    n = EMANormalizer(beta, eps)
    v = 0.0
    for t in range(1, 30):
        g = torch.randn(7) * (1 + t)
        sq = float(torch.linalg.vector_norm(g)) ** 2
        v = beta * v + (1 - beta) * sq
        want = 1 / math.sqrt(v / (1 - beta**t) + eps)
        assert n.scale("x", g) == pytest.approx(want, rel=1e-6)


def test_independent_names_and_state_roundtrip():
    n = EMANormalizer()
    n.scale("a", torch.ones(3))
    n.scale("a", torch.ones(3))
    n.scale("b", torch.ones(3) * 5)
    assert n.t == {"a": 2, "b": 1}
    m = EMANormalizer(0.5, 1.0)
    m.load_state_dict(n.state_dict())
    assert m.scale("a", torch.ones(3)) == n.scale("a", torch.ones(3))


def test_bad_args():
    with pytest.raises(ValueError):
        EMANormalizer(beta=1.0)
    with pytest.raises(ValueError):
        EMANormalizer().update_sq("a", float("nan"))


@pytest.mark.parametrize("c", [1e3, 1e-1])
@pytest.mark.parametrize("make", [lambda: PCD(0.3), lambda: NormalizedWeightedSum(0.7)])
def test_scale_invariance_of_direction(c, make):
    """Multiplying the KD loss (hence g2) by c leaves the normalized direction unchanged."""
    torch.manual_seed(1)
    a, b = make(), make()
    for t in range(60):
        g1 = torch.randn(20, dtype=torch.float64)
        g2 = -0.5 * g1 + torch.randn(20, dtype=torch.float64)
        da, _ = a.combine({"ce": g1, "kd": g2})
        db, _ = b.combine({"ce": g1, "kd": c * g2})
        if t >= 10:  # after warm-up
            assert torch.allclose(da, db, rtol=1e-5, atol=1e-7)
