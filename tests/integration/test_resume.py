"""Pause/resume for the workstation: DeepFreeze wipes the disk and the access window closes on
Monday 08:00, so a run must stop with a checkpoint and resume to the same result."""

import pytest
import torch

from pcdkd.data.build import build_data
from pcdkd.experiments.runner import run_dir, run_many
from pcdkd.experiments.suite import PROFILES, build_specs
from pcdkd.models import get_model
from pcdkd.train.trainer import Paused, train

pytestmark = pytest.mark.integration
SMOKE, SMOKE_TEACHERS = build_specs("smoke")
quiet = lambda m: None  # noqa: E731


def _flat(model):
    return torch.cat(
        [p.detach().flatten() for p in model.state_dict().values() if p.is_floating_point()]
    )


@pytest.mark.parametrize("method", ["pcd", "dot"])  # EMA normaliser state / DOT momentum buffers
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
