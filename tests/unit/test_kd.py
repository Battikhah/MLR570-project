import math

import pytest
import torch
import torch.nn.functional as F

from pcdkd.losses.kd import kd_loss, soften_probs


def _data(seed=0, B=8, C=5):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(B, C, generator=g, requires_grad=True)
    t_logits = torch.randn(B, C, generator=g) * 2
    return s, t_logits


def _manual(s, t_logits, T):
    p = torch.softmax(t_logits / T, -1)
    logq = torch.log_softmax(s / T, -1)
    return (T * T) * (p * (torch.log(p) - logq)).sum(-1)


@pytest.mark.parametrize("T", [1.0, 2.0, 4.0])
def test_equals_manual_T2_KL(T):
    s, tl = _data()
    got = kd_loss(s, torch.softmax(tl, -1), T)
    want = _manual(s, tl, T).mean()
    assert torch.allclose(got, want, atol=1e-5)
    # also equals T^2 * F.kl_div(batchmean)
    ref = F.kl_div(torch.log_softmax(s / T, -1), torch.softmax(tl / T, -1), reduction="batchmean")
    assert torch.allclose(got, T * T * ref, atol=1e-5)


def test_pretempered_matches():
    s, tl = _data(1)
    T = 3.0
    a = kd_loss(s, torch.softmax(tl, -1), T)
    b = kd_loss(s, torch.softmax(tl / T, -1), T, pretempered=True)
    assert torch.allclose(a, b, atol=1e-5)
    assert torch.allclose(
        soften_probs(torch.softmax(tl, -1), T), torch.softmax(tl / T, -1), atol=1e-6
    )


def test_mask_mean_over_masked_in_and_reductions():
    s, tl = _data(2)
    T = 2.0
    mask = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0], dtype=torch.bool)
    per = _manual(s, tl, T)
    p = torch.softmax(tl, -1)
    assert torch.allclose(kd_loss(s, p, T, mask), per[mask].mean(), atol=1e-5)
    assert torch.allclose(kd_loss(s, p, T, mask, reduction="sum"), per[mask].sum(), atol=1e-5)
    none = kd_loss(s, p, T, mask, reduction="none")
    assert torch.allclose(none[mask], per[mask], atol=1e-5)
    assert torch.all(none[~mask] == 0)


def test_zero_mask_returns_zero_with_graph():
    s, tl = _data(3)
    out = kd_loss(s, torch.softmax(tl, -1), 2.0, torch.zeros(8, dtype=torch.bool))
    assert out.item() == 0.0 and out.requires_grad
    (g,) = torch.autograd.grad(out, s)
    assert torch.all(g == 0)


def test_gradient_finite_with_zero_teacher_probs():
    s, _ = _data(4)
    p = torch.zeros(8, 5)
    p[:, 0] = 1.0  # one-hot teacher (log 0 entries)
    out = kd_loss(s, p, 4.0)
    (g,) = torch.autograd.grad(out, s)
    assert math.isfinite(out.item()) and torch.isfinite(g).all()
    assert out.item() >= 0


def test_teacher_detached_and_bad_inputs():
    s, tl = _data(5)
    p = torch.softmax(tl, -1).requires_grad_(True)
    kd_loss(s, p, 2.0).backward()
    assert p.grad is None
    with pytest.raises(ValueError):
        kd_loss(s, p[:, :3], 2.0)
    with pytest.raises(ValueError):
        kd_loss(s, p, 2.0, reduction="bad")
    with pytest.raises(ValueError):
        kd_loss(s, p, 0.0)


@pytest.mark.parametrize("T", [0.0, -1.0, float("nan"), float("inf")])
@pytest.mark.parametrize("pretempered", [True, False])
def test_invalid_temperature_rejected_even_if_pretempered(T, pretempered):
    """Codex review F3: T is validated before the pretempered branch."""
    s, tl = _data(6)
    with pytest.raises(ValueError, match="temperature"):
        kd_loss(s, torch.softmax(tl, -1), T, pretempered=pretempered)
    with pytest.raises(ValueError, match="temperature"):
        kd_loss(s, torch.log_softmax(tl, -1), T, kind="logprobs", pretempered=pretempered)


@pytest.mark.parametrize("T", [1.0, 4.0])
@pytest.mark.parametrize("pretempered", [False, True])
def test_logprobs_kind_matches_probs(T, pretempered):
    s, tl = _data(7)
    src = tl / T if pretempered else tl
    a = kd_loss(s, torch.softmax(src, -1), T, pretempered=pretempered)
    b = kd_loss(s, torch.log_softmax(src, -1), T, pretempered=pretempered, kind="logprobs")
    c = kd_loss(s, torch.log_softmax(src, -1) + 3.0, T, pretempered=pretempered, kind="logprobs")
    assert torch.allclose(a, b, atol=1e-5) and torch.allclose(b, c, atol=1e-5)
    assert torch.allclose(b, _manual(s, tl, T).mean(), atol=1e-5)


def test_logprobs_avoid_float32_underflow():
    """z = [-200, 0]: float32 probs underflow to one-hot; log-probs keep the softened mass."""
    z = torch.tensor([[-200.0, 0.0]])
    s = torch.zeros(1, 2, requires_grad=True)
    T = 20.0
    assert torch.softmax(z, 1)[0, 0].item() == 0.0  # the underflow being avoided
    out = kd_loss(s, torch.log_softmax(z.double(), 1), T, kind="logprobs")
    p = torch.softmax(z / T, 1)  # exact target, independently computed
    want = (T * T) * (p * (p.log() - torch.log_softmax(s.detach() / T, 1))).sum()
    assert out.item() == pytest.approx(want.item(), rel=1e-6)
    assert p[0, 0].item() == pytest.approx(4.54e-5, rel=1e-2)


def test_logprobs_with_neg_inf_entries_finite():
    s, _ = _data(8)
    lp = torch.full((8, 5), float("-inf"))
    lp[:, 2] = 0.0
    out = kd_loss(s, lp, 4.0, kind="logprobs")
    (g,) = torch.autograd.grad(out, s)
    assert math.isfinite(out.item()) and torch.isfinite(g).all()
    ref = kd_loss(s, lp.exp(), 4.0)
    assert torch.allclose(out, ref, atol=1e-5)
    with pytest.raises(ValueError, match="kind"):
        kd_loss(s, lp, 4.0, kind="logits")
