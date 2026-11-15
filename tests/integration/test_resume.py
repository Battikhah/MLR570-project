"""Pause/resume for the workstation: DeepFreeze wipes the disk and the access window closes on
Monday 08:00, so a run must stop with a checkpoint and resume to the same result."""

import pytest
import torch

from pcdkd.data.build import build_data
from pcdkd.experiments.runner import run_dir, run_many
from pcdkd.experiments.suite import PROFILES, build_specs
from pcdkd.models import get_model
from pcdkd.train.trainer import Paused, _loader, _rng_state, _set_rng_state, train

pytestmark = pytest.mark.integration
SMOKE, SMOKE_TEACHERS = build_specs("smoke")
quiet = lambda m: None  # noqa: E731


def _flat(model):
    return torch.cat(
        [p.detach().flatten() for p in model.state_dict().values() if p.is_floating_point()]
    )


@pytest.mark.parametrize("method", ["pcd", "dot", "pcdl", "pcdlm"])  # EMA / momentum-buffer state
def test_paused_then_resumed_run_matches_uninterrupted(tmp_path, method):
    spec = next(s for s in SMOKE if s["method"] == method and len(s["teacher_keys"]) == 1)
    spec = dict(spec, epochs=2, batch_size=256, log_every=1)  # 3 steps/epoch keeps CPU time low
    data = build_data(spec, synthetic=True)
    torch.manual_seed(0)
    teacher = [get_model(SMOKE_TEACHERS[spec["teacher_keys"][0]]["arch"], data["classes"])]

    straight, res_a = train(spec, data, teacher, device="cpu", log=quiet)

    ckpt = tmp_path / "checkpoint.pt"
    with pytest.raises(Paused):
        train(
            spec,
            data,
            teacher,
            device="cpu",
            log=quiet,
            checkpoint=ckpt,
            should_stop=lambda s: True,
        )
    assert ckpt.exists()
    resumed, res_b = train(spec, data, teacher, device="cpu", log=quiet, checkpoint=ckpt)

    assert torch.equal(_flat(straight), _flat(resumed))
    assert [e["epoch"] for e in res_b["epochs"]] == [0, 1]
    assert res_a["steps"] == res_b["steps"]
    assert res_a["diagnostics"].keys() == res_b["diagnostics"].keys()


def test_sweep_pauses_then_resumes_and_cleans_up(tmp_path, monkeypatch):
    monkeypatch.setitem(PROFILES["smoke"], "max_steps", None)  # the smoke cap ends epoch 0 early
    students = [s for s in SMOKE if s["family"] == "core-p0"][:2]
    students = [dict(s, epochs=2, batch_size=256) for s in students]
    calls = {"n": 0}

    def stop_after_first_epoch(next_epoch_seconds):
        calls["n"] += 1
        return next_epoch_seconds > 0  # pause at the first epoch boundary; starting is allowed

    teachers = {k: v for k, v in SMOKE_TEACHERS.items() if k in students[0]["teacher_keys"]}
    run_many(students[:1], teachers, tmp_path, stage="teachers", device="cpu", log=quiet)
    status = run_many(
        students,
        teachers,
        tmp_path,
        stage="students",
        device="cpu",
        log=quiet,
        should_stop=stop_after_first_epoch,
        ckpt_every=1,
    )
    first = run_dir(tmp_path, students[0])
    assert status["paused"] == 1 and status["done"] == 0
    assert (first / "checkpoint.pt").exists() and not (first / "result.json").exists()
    assert not run_dir(tmp_path, students[1]).exists()  # nothing started after the pause

    status = run_many(students, teachers, tmp_path, stage="students", device="cpu", log=quiet)
    assert status == {"done": 2, "skipped": 0, "failed": 0, "paused": 0}
    assert (first / "result.json").exists() and not (first / "checkpoint.pt").exists()


def test_closing_window_starts_nothing(tmp_path):
    status = run_many(
        SMOKE[:1], SMOKE_TEACHERS, tmp_path, device="cpu", log=quiet, should_stop=lambda s: True
    )
    assert status["paused"] == 1 and not any(tmp_path.rglob("*.pt"))


def _tiny(method="pcd"):
    spec = next(s for s in SMOKE if s["method"] == method and len(s["teacher_keys"]) == 1)
    spec = dict(spec, epochs=2, batch_size=256, log_every=1)
    data = build_data(spec, synthetic=True)
    torch.manual_seed(0)
    teacher = [get_model(SMOKE_TEACHERS[spec["teacher_keys"][0]]["arch"], data["classes"])]
    return spec, data, teacher


ACCEL = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None
)


@pytest.mark.skipif(ACCEL is None, reason="needs a GPU (cuda or mps)")
def test_rng_state_restores_after_loading_onto_the_gpu(tmp_path):
    """On the GPU the checkpoint is loaded with map_location=device, which moves the RNG states
    off the CPU; restoring them must still work (failed on the workstation, 2026-11-12)."""
    spec, data, _ = _tiny()
    train_dl = _loader(data["train"], 64, True, 0, 0, "cpu")
    torch.save({"rng": _rng_state(train_dl)}, tmp_path / "c.pt")
    st = torch.load(tmp_path / "c.pt", map_location=ACCEL, weights_only=False)
    assert st["rng"]["torch"].device.type == ACCEL
    _set_rng_state(st["rng"], train_dl)


def test_a_busy_checkpoint_file_does_not_stop_training(tmp_path, monkeypatch):
    """Windows refuses to replace a file the Drive uploader is reading; keep training."""
    import pcdkd.train.trainer as trainer

    def busy(obj, path):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(trainer, "save_torch", busy)
    spec, data, teacher = _tiny()
    msgs = []
    model, res = train(
        spec,
        data,
        teacher,
        device="cpu",
        log=msgs.append,
        checkpoint=tmp_path / "checkpoint.pt",
        ckpt_every=1,
    )
    assert [e["epoch"] for e in res["epochs"]] == [0, 1]
    assert any("checkpoint not updated" in m for m in msgs)
