"""Run suite specs: teachers first (cached checkpoints), then students; one JSON per run.

Layout under ``out`` (default ``outputs/suite``)::

    <profile>/teachers/<key>.pt | <key>.json
    <profile>/<family>/<condition>/<arm>/s<seed>/result.json   (or error.json on failure)

Completed runs are skipped, so a crashed job can simply be restarted. A run in progress keeps
``checkpoint.pt`` in its directory (teachers: ``teachers/<key>.ckpt``) and resumes from it; the
checkpoint is deleted when the run finishes. If ``should_stop`` says the access window is closing,
the current run saves and the sweep ends with ``paused`` in the status.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from pcdkd.data.build import build_data
from pcdkd.data.regimes import class_groups
from pcdkd.experiments.suite import PROFILES
from pcdkd.models import get_model
from pcdkd.train.trainer import Paused, _loader, device_auto, evaluate, train
from pcdkd.utils.atomic import remove, replace_with_retry, save_torch


def _git_commit():
    try:
        return subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "rev-parse", "HEAD"],  # noqa: S607 - git from PATH is intended
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, allow_nan=False, default=_jsonable))
    replace_with_retry(tmp, path)


def _jsonable(x):
    if isinstance(x, np.integer | np.floating):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f"not JSON-serialisable: {type(x)}")


def run_dir(out, spec):
    return Path(out) / spec["profile"] / spec["id"]


def teacher_paths(out, profile, key):
    base = Path(out) / profile / "teachers"
    return base / f"{key}.pt", base / f"{key}.json"  # not with_suffix: keys contain dots


def _env(device):
    return {
        "git_commit": _git_commit(),
        "torch": torch.__version__,
        "device": device,
        "python": platform.python_version(),
        "host": platform.node(),
        "cuda": torch.cuda.get_device_name() if device == "cuda" else None,
    }


def run_teacher(
    tspec,
    out,
    *,
    data_root="data",
    device=None,
    workers=0,
    log=print,
    ckpt_every=5,
    should_stop=None,
):
    ckpt, meta = teacher_paths(out, tspec["profile"], tspec["condition"])
    if ckpt.exists():
        return
    resume = ckpt.with_name(ckpt.name.removesuffix(".pt") + ".ckpt")
    device = device or device_auto()
    prof = PROFILES[tspec["profile"]]
    data = build_data(tspec, data_root, synthetic=prof["synthetic"])
    pre = tspec.get("pretrained")
    provenance = {}
    if pre:  # published weights: evaluate and cache, no training
        src = Path(data_root) / pre["file"]
        if not src.exists():
            raise FileNotFoundError(
                f"pretrained teacher {tspec['condition']}: {src} not found; download {pre['url']} "
                f"and extract it into {Path(data_root) / pre['dir']}"
            )
        model = get_model(tspec["arch"], data["classes"])
        model.load_state_dict(_load_pretrained(src, pre["key"]), strict=True)
        model, res = model.to(device), {}
        with src.open("rb") as f:
            provenance = {
                "source": pre["source"],
                "file": pre["file"],
                "file_sha256": hashlib.file_digest(f, "sha256").hexdigest(),
                "archive_sha256": pre["archive_sha256"],
                "published_top1": pre.get("published_top1"),
            }
    else:
        model, res = train(
            tspec,
            data,
            device=device,
            workers=workers,
            max_steps=prof["max_steps"],
            log=log,
            checkpoint=resume,
            ckpt_every=ckpt_every,
            should_stop=should_stop,
        )
    test = _evaluate(model, data["test"], data["classes"], device, workers)
    val = _evaluate(model, data["val"], data["classes"], device, workers)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    save_torch(model.state_dict(), ckpt)
    _write_json(
        meta,
        {"spec": tspec, "val": val, "test": test, "train": res, "env": _env(device)} | provenance,
    )
    remove(resume)
    log(f"[teacher {tspec['condition']}] test top1={test['top1']:.4f}")


def _evaluate(model, dataset, classes, device, workers, **kw):
    """Metrics on ``dataset``; None when there is no such split (``val_size: 0``)."""
    if dataset is None:
        return None
    return evaluate(model, _loader(dataset, 512, False, 0, 0, device), classes, device, **kw)


def _load_pretrained(path, key):
    """The published checkpoints hold numpy scalars beside the weights. Allowlisting just those
    numpy types keeps the load ``weights_only`` (no arbitrary unpickling)."""
    safe = [
        (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),  # name in numpy < 2 pickles
        np._core.multiarray.scalar,
        np.dtype,
        type(np.dtype(np.float64)),
    ]
    with torch.serialization.safe_globals(safe):
        return torch.load(path, map_location="cpu", weights_only=True)[key]


def load_teacher(out, profile, key, arch, classes, device):
    ckpt, _ = teacher_paths(out, profile, key)
    if not ckpt.exists():
        raise FileNotFoundError(f"teacher {key} missing at {ckpt}; run the teachers stage first")
    model = get_model(arch, classes)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def run_student(
    spec,
    teacher_specs,
    out,
    *,
    data_root="data",
    device=None,
    workers=0,
    log=print,
    ckpt_every=5,
    should_stop=None,
):
    rdir = run_dir(out, spec)
    if (rdir / "result.json").exists():
        return "skipped"
    device = device or device_auto()
    prof = PROFILES[spec["profile"]]
    data = build_data(spec, data_root, synthetic=prof["synthetic"])
    teachers = [
        load_teacher(out, spec["profile"], k, teacher_specs[k]["arch"], data["classes"], device)
        for k in spec["teacher_keys"]
    ]
    model, res = train(
        spec,
        data,
        teachers,
        device=device,
        workers=workers,
        max_steps=prof["max_steps"],
        log=log,
        checkpoint=rdir / "checkpoint.pt",
        ckpt_every=ckpt_every,
        should_stop=should_stop,
    )
    train_labels = data["clean_labels"][data["train_indices"]].numpy()
    groups = class_groups(train_labels, data["classes"]) if spec["imbalance"] > 1 else None
    val = _evaluate(model, data["val"], data["classes"], device, workers)
    test = _evaluate(
        model,
        data["test"],
        data["classes"],
        device,
        workers,
        teacher=teachers[0] if teachers else None,
        groups=groups,
    )
    teacher_test = []
    for k in spec["teacher_keys"]:
        meta = teacher_paths(out, spec["profile"], k)[1]
        teacher_test.append(json.loads(meta.read_text())["test"]["top1"] if meta.exists() else None)
    _write_json(
        rdir / "result.json",
        {
            "spec": spec,
            "val": val,
            "test": test,
            "teacher_test_top1": teacher_test,
            "train": res,
            "env": _env(device),
        },
    )
    remove(rdir / "error.json")
    remove(rdir / "checkpoint.pt")
    vtxt = f"{val['top1']:.4f}" if val else "none"
    log(f"[{spec['id']}] val={vtxt} test={test['top1']:.4f}")
    return "done"


def run_many(students, teacher_specs, out, *, stage="all", log=print, should_stop=None, **kw):
    """Train the teachers the selected students need, then the students. Returns a status count.

    ``paused`` is 1 when the sweep stopped because ``should_stop`` fired (window closing).
    """
    status = {"done": 0, "skipped": 0, "failed": 0, "paused": 0}

    def closing():
        if should_stop is not None and should_stop(0.0):
            status["paused"] = 1
            log("access window closing: not starting another run")
            return True
        return False

    if stage in ("all", "teachers"):
        needed = sorted(
            {k for s in students for k in s["teacher_keys"]} if students else teacher_specs
        )
        for key in needed:
            if closing():
                return status
            try:
                run_teacher(teacher_specs[key], out, log=log, should_stop=should_stop, **kw)
            except Paused as p:
                status["paused"] = 1
                log(f"PAUSED {p}; rerun the same command in the next window to resume")
                return status
    if stage in ("all", "students"):
        for spec in students:
            if closing():
                return status
            try:
                status[
                    run_student(spec, teacher_specs, out, log=log, should_stop=should_stop, **kw)
                ] += 1
            except Paused as p:
                status["paused"] = 1
                log(f"PAUSED {p}; rerun the same command in the next window to resume")
                return status
            except Exception as exc:  # noqa: BLE001 - one bad run must not stop the sweep
                status["failed"] += 1
                _write_json(
                    run_dir(out, spec) / "error.json",
                    {"spec": spec, "error": repr(exc), "traceback": traceback.format_exc()},
                )
                log(f"[{spec['id']}] FAILED: {exc!r}")
    return status


# --- shared work queue: several processes on one GPU, highest-priority work first ----------------


class GpuFault(RuntimeError):
    """The CUDA context of this process is dead: every later task here would fail instantly."""


def _gpu_fault(exc):
    return "CUDA error" in str(exc)


def _claim_path(out, profile, task):
    kind, x = task
    if kind == "teacher":
        return teacher_paths(out, profile, x)[0].with_suffix(".claim")
    return run_dir(out, x) / "claim"


def _claim(path):
    """Atomically take a task; False if another process already holds it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        f.write(f"{platform.node()} pid={os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    return True


def clear_claims(out, profile):
    """Drop every claim and every failure so it is retried (run once before starting workers;
    claims of a dead session are stale; Drive keeps the failure records). Returns the number of
    claims dropped."""
    base = Path(out) / profile
    for p in [*(base / "teachers").glob("*.error.json"), *base.rglob("error.json")]:
        p.unlink(missing_ok=True)
    n = 0
    for p in [*base.rglob("claim"), *base.rglob("*.claim")]:
        p.unlink(missing_ok=True)
        n += 1
    return n


def queue_tasks(students, teacher_specs):
    """``students`` (already in priority order) with each teacher inserted before its first user."""
    tasks, seen = [], set()
    for s in students:
        for k in s["teacher_keys"]:
            if k not in seen:
                seen.add(k)
                tasks.append(("teacher", k))
        tasks.append(("student", s))
    return tasks


def run_queue(
    students, teacher_specs, out, *, should_stop=None, log=print, poll=60, sleep=time.sleep, **kw
):
    """Work through ``queue_tasks`` with any number of cooperating processes.

    Each process takes the first task that is not finished, not claimed and not waiting for a
    teacher, preserving the selected queue plan. A student whose
    teacher another process is still training waits; one whose teacher failed is skipped. Returns
    the status count of the tasks this process ran.
    """
    status = {"done": 0, "skipped": 0, "failed": 0, "paused": 0}
    profile = students[0]["profile"] if students else None
    pending = queue_tasks(students, teacher_specs)

    def finished(task):
        kind, x = task
        if kind == "teacher":
            return teacher_paths(out, profile, x)[0].exists()
        return (run_dir(out, x) / "result.json").exists()

    def teacher_failed(key):
        return teacher_paths(out, profile, key)[0].with_suffix(".error.json").exists()

    while pending:
        if should_stop is not None and should_stop(0.0):
            status["paused"] = 1
            log("access window closing: not starting another task")
            return status
        task, waiting = None, False
        for t in list(pending):
            if finished(t):
                pending.remove(t)
                continue
            if t[0] == "student":
                keys = t[1]["teacher_keys"]
                if any(teacher_failed(k) for k in keys):
                    pending.remove(t)
                    status["failed"] += 1
                    log(f"[{t[1]['id']}] skipped: its teacher failed")
                    continue
                if not all(teacher_paths(out, profile, k)[0].exists() for k in keys):
                    waiting = True  # teacher in training elsewhere (or later in this queue)
                    continue
            if _claim(_claim_path(out, profile, t)):
                task = t
                break
            pending.remove(t)  # another process owns it
        if task is None:
            if not waiting:
                break
            sleep(poll)
            continue
        pending.remove(task)
        kind, x = task
        name = x if kind == "teacher" else x["id"]
        target = (
            teacher_paths(out, profile, x)[0].with_suffix(".error.json")
            if kind == "teacher"
            else run_dir(out, x) / "error.json"
        )
        for attempt in (1, 2):
            try:
                if kind == "teacher":
                    run_teacher(teacher_specs[x], out, log=log, should_stop=should_stop, **kw)
                    status["done"] += 1
                else:
                    status[
                        run_student(x, teacher_specs, out, log=log, should_stop=should_stop, **kw)
                    ] += 1
                break
            except Paused as p:
                status["paused"] = 1
                log(f"PAUSED {p}; rerun the same command in the next window to resume")
                return status
            except Exception as exc:  # noqa: BLE001 - one bad task must not stop the queue
                if _gpu_fault(exc):
                    # retrying here fails too and would burn through the whole queue: hand the
                    # task back (a task that faults 3 times counts as failed) and exit
                    mark = target.with_suffix(".gpufault")
                    n = int(mark.read_text()) + 1 if mark.exists() else 1
                    _write_json(mark, n)
                    if n >= 3:
                        status["failed"] += 1
                        _write_json(
                            target,
                            {"task": name, "error": repr(exc), "traceback": traceback.format_exc()},
                        )
                    else:
                        _claim_path(out, profile, task).unlink(missing_ok=True)
                    log(f"[{name}] GPU FAULT {n}/3: {exc!r}; stopping this worker")
                    raise GpuFault(name) from exc
                if attempt == 1:
                    log(f"[{name}] FAILED (attempt 1): {exc!r}; retrying")
                    continue
                status["failed"] += 1
                _write_json(
                    target, {"task": name, "error": repr(exc), "traceback": traceback.format_exc()}
                )
                log(f"[{name}] FAILED: {exc!r}")
    return status
