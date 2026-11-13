"""Shared priority queue for a time-limited GPU window: several processes, one queue."""

import pytest

from pcdkd.experiments.runner import (
    _claim,
    _claim_path,
    clear_claims,
    queue_tasks,
    run_dir,
    run_queue,
    teacher_paths,
)
from pcdkd.experiments.suite import PRIORITY, build_specs, queue_order

pytestmark = pytest.mark.integration
SMOKE, SMOKE_TEACHERS = build_specs("smoke")
SCREEN, _ = build_specs("screen")
quiet = lambda m: None  # noqa: E731


def test_queue_order_is_breadth_first_by_priority():
    q = queue_order(SCREEN)
    seeds = [s["seed"] for s in q]
    assert seeds == sorted(seeds)  # every seed-0 run before any seed-1 run
    seed0 = [s["family"] for s in q if s["seed"] == 0]
    ranks = [PRIORITY.index(f) for f in seed0]
    assert ranks == sorted(ranks) and seed0[0] == "pilot"
    # within a family the design order is kept: one condition's arms finish together
    tc = [s["condition"] for s in q if s["family"] == "teacher-corruption" and s["seed"] == 0]
    assert tc == sorted(tc, key=tc.index)


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
    run_queue(students[:1], teachers, tmp_path, device="cpu", log=quiet)  # trains the teacher
    assert _claim(_claim_path(tmp_path, "smoke", ("student", students[1])))  # other process
    status = run_queue(students, teachers, tmp_path, device="cpu", log=quiet)
    assert not (run_dir(tmp_path, students[1]) / "result.json").exists()
    assert (run_dir(tmp_path, students[2]) / "result.json").exists()
    assert status["failed"] == 0


def test_student_waits_for_a_teacher_in_training_elsewhere(tmp_path):
    students, teachers = _cheap(1)
    (key,) = teachers
    assert _claim(_claim_path(tmp_path, "smoke", ("teacher", key)))  # another process trains it

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
