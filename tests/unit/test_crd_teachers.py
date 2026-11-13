"""CRD/mdistiller ResNets, pre-trained teachers in the runner, the lit-p1 family, val_size 0."""

import json

import numpy as np
import pytest
import torch

from pcdkd.experiments.runner import run_student, run_teacher, teacher_paths
from pcdkd.experiments.suite import build_specs, queue_order
from pcdkd.models import get_model

KEY = "cifar100-md_resnet32x4-crd"
# key counts of the published checkpoints (cifar_teachers.tar, no num_batches_tracked buffers)
RELEASED_KEYS = {"md_resnet32x4": 172, "md_resnet56": 287, "md_resnet110": 557}


def _mdistiller_format(model):
    """State dict as the released checkpoints store it (older torch: no num_batches_tracked)."""
    return {k: v for k, v in model.state_dict().items() if not k.endswith("num_batches_tracked")}


@pytest.mark.parametrize(
    "arch", ["md_resnet8x4", "md_resnet32x4", "md_resnet20", "md_resnet56", "md_resnet110"]
)
def test_md_strict_round_trip(arch):
    torch.manual_seed(0)
    a, b = get_model(arch, 100).eval(), get_model(arch, 100).eval()
    sd = _mdistiller_format(a)
    assert {"conv1.weight", "bn1.running_var", "fc.weight", "fc.bias"} <= set(sd)
    assert not any("linear" in k or "shortcut" in k for k in sd)
    if arch in RELEASED_KEYS:
        assert len(sd) == RELEASED_KEYS[arch]
    b.load_state_dict(sd, strict=True)
    x = torch.randn(2, 3, 32, 32)
    logits, feats = b(x, return_features=True)
    assert torch.equal(logits, a(x)) and logits.shape == (2, 100)
    w = 256 if "x4" in arch else 64
    assert [tuple(f.shape[1:]) for f in feats][-1] == (w, 8, 8)


def test_md_downsample_only_where_shape_changes():
    x4, r56 = get_model("md_resnet32x4", 100).state_dict(), get_model("md_resnet56", 100)
    assert "layer1.0.downsample.0.weight" in x4  # 32 -> 64 channels
    assert x4["layer3.0.downsample.0.weight"].shape == (256, 128, 1, 1)
    keys = set(r56.state_dict())
    assert "layer1.0.downsample.0.weight" not in keys and "layer2.0.downsample.1.weight" in keys
    assert "layer2.1.downsample.0.weight" not in keys


def test_lit_p1_specs():
    for profile in ("screen", "full"):
        students, teachers = build_specs(profile, families=["lit-p1"])
        assert len(students) == 9 and {s["seed"] for s in students} == {0}
        assert {s["epochs"] for s in students} == {240}
        assert {s["val_size"] for s in students} == {0}
        assert {tuple(s["teacher_keys"]) for s in students} == {(KEY,)}
        assert {s["arch"] for s in students} == {"md_resnet8x4"}
        assert {s["arm"] for s in students} == {
            "ce", "wkd_l0.9", "nwkd_l1.0", "pcd_t0.5", "pcd_t1.0",
            "pcgrad1s", "mgda", "dtokd", "dot_d0.075",
        }  # fmt: skip
        assert {s["T"] for s in students} == {4.0}
        t = teachers[KEY]
        assert t["arch"] == "md_resnet32x4" and t["epochs"] == 0 and t["val_size"] == 0
        assert t["pretrained"]["file"].endswith("resnet32x4_vanilla/ckpt_epoch_240.pth")
        assert t["pretrained"]["key"] == "model"
    screen, _ = build_specs("screen")
    assert queue_order(screen)[0]["family"] == "lit-p1"
    smoke, smoke_t = build_specs("smoke", families=["lit-p1"])
    assert {s["epochs"] for s in smoke} == {1} and "pretrained" not in smoke_t[KEY]


def _pretrained_spec(file):
    _, teachers = build_specs("smoke", families=["lit-p1"])
    t = teachers[KEY]
    t["pretrained"] = {
        "source": "CRD/mdistiller",
        "url": "unused",
        "dir": "fake",
        "file": file,
        "key": "model",
        "archive_sha256": "0" * 64,
        "published_top1": 79.42,
    }
    t["epochs"] = 0
    return t, teachers


def test_run_teacher_loads_pretrained(tmp_path):
    torch.manual_seed(1)
    fake = get_model("md_resnet32x4", 100)
    (tmp_path / "fake").mkdir()
    # like the release: weights under "model" plus numpy scalars (need the weights_only allowlist)
    torch.save(
        {"model": _mdistiller_format(fake), "accuracy": np.float64(79.42), "epoch": 240},
        tmp_path / "fake" / "ckpt.pth",
    )
    tspec, teachers = _pretrained_spec("fake/ckpt.pth")
    out = tmp_path / "out"
    run_teacher(tspec, out, data_root=tmp_path, device="cpu", log=lambda m: None)
    pt, meta_path = teacher_paths(out, "smoke", KEY)
    loaded = get_model("md_resnet32x4", 100)
    loaded.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))
    for k, v in _mdistiller_format(fake).items():
        assert torch.equal(loaded.state_dict()[k], v), k
    meta = json.loads(meta_path.read_text())
    assert meta["source"] == "CRD/mdistiller" and meta["archive_sha256"] == "0" * 64
    assert len(meta["file_sha256"]) == 64 and meta["val"] is None
    assert 0 <= meta["test"]["top1"] <= 1 and meta["train"] == {}

    # a lit-p1 student (val_size 0) trains against it end to end
    students, _ = build_specs("smoke", families=["lit-p1"])
    spec = next(s for s in students if s["arm"] == "wkd_l0.9")
    assert run_student(spec, teachers, out, device="cpu", log=lambda m: None) == "done"
    res = json.loads((out / "smoke" / spec["id"] / "result.json").read_text())
    assert res["val"] is None and res["teacher_test_top1"] == [meta["test"]["top1"]]
    assert all(e.get("val_top1") is None for e in res["train"]["epochs"])


def test_run_teacher_missing_pretrained_file(tmp_path):
    tspec, _ = _pretrained_spec("fake/missing.pth")
    with pytest.raises(FileNotFoundError, match="missing.pth not found"):
        run_teacher(tspec, tmp_path / "out", data_root=tmp_path, device="cpu", log=lambda m: None)
