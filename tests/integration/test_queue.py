"""Shared queue ordering and execution behavior on synthetic data."""

import pytest

from pcdkd.experiments import suite
from pcdkd.experiments.runner import (
    _claim,
    _claim_path,
    clear_claims,
    queue_tasks,
    run_dir,
    run_queue,
    teacher_paths,
)
from pcdkd.experiments.suite import PRIORITY, WEEKEND_PLAN, build_specs, queue_order

pytestmark = pytest.mark.integration
SMOKE, SMOKE_TEACHERS = build_specs("smoke")
SCREEN, _ = build_specs("screen")
quiet = lambda m: None  # noqa: E731


def test_queue_order_matches_blocks_once_then_old_order_tail():
    q = queue_order(SCREEN)
    assert len(q) == len(SCREEN) and len({s["id"] for s in q}) == len(q)
    # Every selected block is contiguous and sorted by seed then original design position.
    cursor = 0
    pos = {id(s): i for i, s in enumerate(SCREEN)}
    matched = set()
    for family, seeds, conditions, arms in WEEKEND_PLAN:
        block = [
            s
            for s in SCREEN
            if id(s) not in matched
            and s["family"] == family
            and s["seed"] in seeds
            and (conditions is None or s["condition"] in conditions)
            and (s["arm"] in arms if arms is not None else s["arm"] not in suite.DEFERRED_ARMS)
        ]
        expected = sorted(block, key=lambda s: (s["seed"], pos[id(s)]))
        assert q[cursor : cursor + len(expected)] == expected
        cursor += len(expected)
        matched.update(map(id, expected))
    rank = {name: i for i, name in enumerate(PRIORITY)}
    tail = [s for s in SCREEN if id(s) not in matched]
    assert q[cursor:] == sorted(tail, key=lambda s: (s["seed"], rank[s["family"]], pos[id(s)]))


def test_high_rho_core_seeds_come_before_low_rho_seeds_and_dtokd_is_deferred():
    q = queue_order(SCREEN)
    tc = [
        s
        for s in q
        if s["family"] == "teacher-corruption"
        and s["seed"] in (1, 2)
        and s["arm"] in suite.TC_CORE_ARMS
    ]
    first = tc[: 2 * 2 * len(suite.TC_CORE_ARMS)]
    assert {s["condition"] for s in first} == {"rho0.5", "rho0.7"}
    assert {s["arm"] for s in first} == set(suite.TC_CORE_ARMS)
    plan_len = sum(1 for s in SCREEN if any(suite.in_block(s, b) for b in WEEKEND_PLAN))
    planned = q[:plan_len]
    assert not any(s["arm"] == "dtokd" and s["family"] != "lit-p1" for s in planned)
    assert any(s["arm"] == "dtokd" for s in q[plan_len:])


def test_extended_plan_runs_first_and_matched_weights_lead():
    q = queue_order(SCREEN)
    n = sum(1 for s in SCREEN if any(suite.in_block(s, b) for b in suite.EXTENDED_PLAN))
    assert suite.WEEKEND_PLAN[: len(suite.EXTENDED_PLAN)] == suite.EXTENDED_PLAN
    first = q[: 5 * 5 * len(suite.MATCHED_NWKD_ARMS)]
    assert {s["arm"] for s in first} == {"nwkd_l0.25", "nwkd_l0.75"}
    assert {s["family"] for s in first} == {"teacher-corruption"}
    headline = {"teacher-corruption", "label-noise", "core-p0", "lit-p1"}
    assert headline <= {s["family"] for s in q[:n]}
    assert not any(s["arm"] == "dtokd" for s in q[:n])


def test_weekend_plan_rejects_unknown_names(monkeypatch):
    monkeypatch.setattr(suite, "WEEKEND_PLAN", (("teacher-corruption", (0,), None, ("typo",)),))
    with pytest.raises(ValueError, match="unknown weekend-plan arm"):
        suite._validate_weekend_plan()
    monkeypatch.setattr(suite, "WEEKEND_PLAN", (("teacher-corruption", (0,), ("typo",), None),))
    with pytest.raises(ValueError, match="unknown weekend-plan condition"):
        suite._validate_weekend_plan()
    monkeypatch.setattr(suite, "WEEKEND_PLAN", (("missing", (0,), None, None),))
    with pytest.raises(ValueError, match="unknown weekend-plan family"):
        suite._validate_weekend_plan()


def test_each_teacher_comes_just_before_its_first_student():
    tasks = queue_tasks(queue_order(SCREEN), _)
    seen = set()
    for kind, x in tasks:
        if kind == "teacher":
            seen.add(x)
        else:
            assert set(x["teacher_keys"]) <= seen


def test_claims_are_exclusive_and_clearable(tmp_path):
    task = ("student", SMOKE[0])
    path = _claim_path(tmp_path, "smoke", task)
    assert _claim(path) and not _claim(path)
    assert clear_claims(tmp_path, "smoke") == 1 and _claim(path)


def _cheap(n=3):
    fam = [s for s in queue_order(SMOKE) if s["family"] == "core-p0"][:n]
    keys = {k for s in fam for k in s["teacher_keys"]}
    return fam, {k: SMOKE_TEACHERS[k] for k in keys}


def test_queue_trains_teacher_then_students_and_is_idempotent(tmp_path):
    students, teachers = _cheap()
    status = run_queue(students, teachers, tmp_path, device="cpu", log=quiet)
    assert status == {"done": 1 + len(students), "skipped": 0, "failed": 0, "paused": 0}
    assert all((run_dir(tmp_path, s) / "result.json").exists() for s in students)
    again = run_queue(students, teachers, tmp_path, device="cpu", log=quiet)
    assert again["done"] == 0 and again["failed"] == 0


def test_queue_leaves_tasks_claimed_by_another_process(tmp_path):
    students, teachers = _cheap()
    run_queue(students[:1], teachers, tmp_path, device="cpu", log=quiet)
    assert _claim(_claim_path(tmp_path, "smoke", ("student", students[1])))
    status = run_queue(students, teachers, tmp_path, device="cpu", log=quiet)
    assert not (run_dir(tmp_path, students[1]) / "result.json").exists()
    assert (run_dir(tmp_path, students[2]) / "result.json").exists()
    assert status["failed"] == 0


def test_student_waits_for_a_teacher_in_training_elsewhere(tmp_path):
    students, teachers = _cheap(1)
    (key,) = teachers
    assert _claim(_claim_path(tmp_path, "smoke", ("teacher", key)))

    class Waited(Exception):
        pass

    def sleep(_):
        raise Waited

    with pytest.raises(Waited):
        run_queue(students, teachers, tmp_path, device="cpu", log=quiet, sleep=sleep)


def test_students_of_a_failed_teacher_are_skipped(tmp_path):
    students, teachers = _cheap(2)
    (key,) = teachers
    err = teacher_paths(tmp_path, "smoke", key)[0].with_suffix(".error.json")
    err.parent.mkdir(parents=True)
    err.write_text("{}")
    assert _claim(_claim_path(tmp_path, "smoke", ("teacher", key)))
    status = run_queue(students, teachers, tmp_path, device="cpu", log=quiet)
    assert status["failed"] == len(students) and status["done"] == 0


def test_closing_window_stops_the_queue(tmp_path):
    students, teachers = _cheap()
    status = run_queue(
        students, teachers, tmp_path, device="cpu", log=quiet, should_stop=lambda s: True
    )
    assert status["paused"] == 1 and not any(tmp_path.rglob("*.pt"))


def test_retry_succeeds_and_keeps_checkpoint_for_resume(tmp_path, monkeypatch):
    import pcdkd.experiments.runner as runner

    (students, teachers) = _cheap(1)
    for key in teachers:
        pt = teacher_paths(tmp_path, "smoke", key)[0]
        pt.parent.mkdir(parents=True, exist_ok=True)
        pt.touch()
    calls, logs = [], []

    def flaky(spec, *_args, **_kwargs):
        checkpoint = run_dir(tmp_path, spec) / "checkpoint.pt"
        if calls:
            assert checkpoint.exists()
            calls.append("resume")
            return "done"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.touch()
        calls.append("failed")
        raise OSError(22, "spawn failed")

    monkeypatch.setattr(runner, "run_student", flaky)
    status = run_queue(students, teachers, tmp_path, log=logs.append)
    assert calls == ["failed", "resume"]
    assert status["done"] == 1 and status["failed"] == 0
    assert any("FAILED (attempt 1): OSError(22, 'spawn failed'); retrying" in x for x in logs)
    assert not (run_dir(tmp_path, students[0]) / "error.json").exists()


def test_retry_writes_error_only_after_second_failure(tmp_path, monkeypatch):
    import pcdkd.experiments.runner as runner

    (students, teachers) = _cheap(1)
    for key in teachers:
        pt = teacher_paths(tmp_path, "smoke", key)[0]
        pt.parent.mkdir(parents=True, exist_ok=True)
        pt.touch()
    calls, logs = [], []

    def broken(*_args, **_kwargs):
        calls.append(1)
        raise RuntimeError("persistent")

    monkeypatch.setattr(runner, "run_student", broken)
    status = run_queue(students, teachers, tmp_path, log=logs.append)
    assert len(calls) == 2 and status["failed"] == 1
    assert (run_dir(tmp_path, students[0]) / "error.json").exists()
    assert any("FAILED (attempt 1): RuntimeError('persistent'); retrying" in x for x in logs)


def test_clearing_claims_retries_failed_teachers(tmp_path):
    students, teachers = _cheap(1)
    (key,) = teachers
    err = teacher_paths(tmp_path, "smoke", key)[0].with_suffix(".error.json")
    err.parent.mkdir(parents=True)
    err.write_text("{}")
    clear_claims(tmp_path, "smoke")
    assert not err.exists()
    status = run_queue(students, teachers, tmp_path, device="cpu", log=quiet)
    assert status["failed"] == 0 and status["done"] == 2


def test_a_gpu_fault_hands_the_task_back_and_stops_the_worker(tmp_path, monkeypatch):
    import pcdkd.experiments.runner as runner

    students, teachers = _cheap(2)
    for key in teachers:
        pt = teacher_paths(tmp_path, "smoke", key)[0]
        pt.parent.mkdir(parents=True, exist_ok=True)
        pt.touch()
    calls = []

    def dead_gpu(spec, *_args, **_kwargs):
        calls.append(spec["id"])
        raise RuntimeError("CUDA error: unknown error")

    monkeypatch.setattr(runner, "run_student", dead_gpu)
    rdir = run_dir(tmp_path, students[0])
    for n in (1, 2):
        with pytest.raises(runner.GpuFault):
            run_queue(students, teachers, tmp_path, log=quiet)
        assert calls == [students[0]["id"]] * n  # no retry, and the next task is never touched
        assert not (rdir / "claim").exists() and not (rdir / "error.json").exists()
    with pytest.raises(runner.GpuFault):  # third fault on the same task: it counts as failed
        run_queue(students, teachers, tmp_path, log=quiet)
    assert (rdir / "error.json").exists()


def test_clearing_claims_also_clears_student_failures(tmp_path):
    students, _ = _cheap(1)
    err = run_dir(tmp_path, students[0]) / "error.json"
    err.parent.mkdir(parents=True)
    err.write_text("{}")
    clear_claims(tmp_path, "smoke")
    assert not err.exists()
