"""DKD parts, attention transfer, data regimes, synthetic data, and the model factory."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pcdkd.data.build import build_data
from pcdkd.data.regimes import class_groups, fraction_indices, label_noise, long_tail_indices
from pcdkd.losses.extra import at_loss, dkd_parts
from pcdkd.losses.kd import kd_loss
from pcdkd.models import get_model, resnet8


@pytest.mark.parametrize("T", [1.0, 4.0])
def test_dkd_decomposition_identity(T):
    """KL(p_t || p_s) = TCKD + (1 - p_t[target]) NCKD, per example (DKD Eq. 7)."""
    g = torch.Generator().manual_seed(0)
    s, t = torch.randn(6, 10, generator=g), 3 * torch.randn(6, 10, generator=g)
    y = torch.randint(0, 10, (6,), generator=g)
    for i in range(6):
        tc, nc = dkd_parts(s[i : i + 1], t[i : i + 1], y[i : i + 1], T)
        kd = kd_loss(s[i : i + 1], F.log_softmax(t[i : i + 1], -1), T, kind="logprobs")
        pt = F.softmax(t[i] / T, -1)[y[i]]
        assert float(kd) == pytest.approx(float(tc + (1 - pt) * nc), rel=1e-5)
    tc, nc = dkd_parts(t, t, y, T)  # student = teacher -> both terms vanish
    assert abs(float(tc)) < 1e-6 and abs(float(nc)) < 1e-6


def test_dkd_parts_have_distinct_gradients():
    s = torch.randn(4, 10, requires_grad=True)
    t, y = torch.randn(4, 10), torch.tensor([0, 1, 2, 3])
    tc, nc = dkd_parts(s, t, y, 4.0)
    g1 = torch.autograd.grad(tc, s, retain_graph=True)[0].flatten()
    g2 = torch.autograd.grad(nc, s)[0].flatten()
    assert abs(float(F.cosine_similarity(g1, g2, 0))) < 0.999


def test_attention_transfer():
    f = [torch.randn(2, 16, 8, 8), torch.randn(2, 32, 4, 4)]
    assert float(at_loss(f, f)) == 0.0
    wide = [torch.cat([x, x], 1) for x in f]  # channel count differs, same attention map
    assert float(at_loss(f, wide)) == pytest.approx(0.0, abs=1e-7)
    assert float(at_loss(f, [torch.randn_like(x) for x in f])) > 0
    with pytest.raises(ValueError):
        at_loss(f, f[:1])


def test_label_noise_symmetric_and_asymmetric():
    labels = np.repeat(np.arange(10), 100)
    train = np.arange(0, 1000, 2)
    noisy = label_noise(labels, train, 0.4, seed=0)
    changed = np.flatnonzero(noisy != labels)
    assert len(changed) == 200 and set(changed) <= set(train)
    asym = label_noise(labels, train, 0.4, seed=0, kind="asymmetric")
    ch = asym != labels
    assert ch.sum() == 200 and np.all(asym[ch] == (labels[ch] + 1) % 10)
    assert np.array_equal(noisy, label_noise(labels, train, 0.4, seed=0))
    with pytest.raises(ValueError):
        label_noise(labels, train, 1.5, 0)


def test_long_tail_profile_and_groups():
    labels = np.repeat(np.arange(100), 450)
    idx = long_tail_indices(labels, np.arange(len(labels)), 100, seed=0)
    counts = np.bincount(labels[idx], minlength=100)
    assert counts[0] == 450 and counts[-1] == 4 and np.all(np.diff(counts) <= 0)
    groups = class_groups(labels[idx], 100)
    assert sum(len(v) for v in groups.values()) == 100
    assert counts[groups["few"]].max() < 20 and counts[groups["many"]].min() > 100
    assert len(long_tail_indices(labels, np.arange(len(labels)), 1, 0)) == len(labels)


def test_fraction_is_per_class():
    labels = np.repeat(np.arange(5), 100)
    idx = fraction_indices(labels, np.arange(500), 0.05, seed=1)
    assert np.all(np.bincount(labels[idx]) == 5)
    assert len(fraction_indices(labels, np.arange(500), 0.001, 1)) == 5  # at least one per class


def test_build_data_synthetic_regimes():
    spec = {
        "dataset": "cifar100",
        "val_size": 5000,
        "split_seed": 0,
        "seed": 1,
        "label_noise": 0.5,
        "teacher_corruption": 0.3,
        "train_fraction": 0.5,
    }
    d = build_data(spec, synthetic=True)
    idx = d["train_indices"]
    x, y, i = d["train"][0]
    assert x.shape == (3, 32, 32) and i == idx[0]
    assert not set(idx) & {d["val"][k][2] for k in range(len(d["val"]))}
    noisy = (d["train_labels"][idx] != d["clean_labels"][idx]).float().mean()
    assert float(noisy) == pytest.approx(0.5, abs=0.01)
    assert set(d["corruption_map"]) <= set(idx.tolist())
    assert len(d["corruption_map"]) == round(0.3 * len(idx))


def test_get_model_and_features():
    m = get_model("resnet8x4", 100)
    logits, feats = m(torch.randn(2, 3, 32, 32), return_features=True)
    assert logits.shape == (2, 100) and [f.shape[1] for f in feats] == [64, 128, 256]
    assert [f.shape[-1] for f in feats] == [32, 16, 8]
    assert sum(p.numel() for p in get_model("resnet8", 10).parameters()) == sum(
        p.numel() for p in resnet8(10).parameters()
    )
    with pytest.raises(ValueError):
        get_model("vgg13")
    with pytest.raises(ValueError):
        get_model("resnet9")
