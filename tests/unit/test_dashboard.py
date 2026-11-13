import json
import os
import threading
import time
import urllib.error
import urllib.request

import pytest

from pcdkd.experiments.runner import teacher_paths
from pcdkd.experiments.suite import build_specs
from scripts.dashboard import Scanner, compact_result, forecast, make_server, parse_line


def test_log_lines():
    e = parse_line(
        "[capacity-gap/resnet14/pcd_t0.5/s0] epoch 3/60 lr=0.05 time=43.13 "
        "train_ce=3.192 train_kd=6.585"
    )
    assert e["kind"] == "epoch" and (e["epoch"], e["epochs"], e["train_kd"]) == (3, 60, 6.585)
    t = parse_line("[teacher/cifar100-resnet32x4] epoch 10/240 lr=0.05 time=445.1 val_top1=0.519")
    assert t["id"] == "teacher/cifar100-resnet32x4" and t["val_top1"] == 0.519
    r = parse_line("[capacity-gap/resnet14/pcd_t0.5/s0] val=0.5538 test=0.5619")
    assert r["kind"] == "result" and r["test"] == 0.5619
    assert parse_line("[teacher cifar100-resnet56] test top1=0.7100") == {
        "kind": "result",
        "id": "teacher/cifar100-resnet56",
    }
    f = parse_line("[cifar100-resnet32x4] FAILED: PermissionError(13, 'Access is denied')")
    assert f["kind"] == "failed" and f["msg"].startswith("PermissionError")
    assert parse_line("[pilot/clean/ce/s0] skipped: its teacher failed")["kind"] == "skipped"
    assert parse_line("[x] resumed from C:\\a\\b.pt at epoch 25")["kind"] == "resumed"
    assert parse_line("[window] stopping before Mon 2026-11-16 07:40 GST")["kind"] == "window"
    assert parse_line("PAUSED stopping; rerun")["kind"] == "paused"
    assert parse_line("Downloading 7.1%") is None


def test_forecast_fills_free_workers_in_queue_order():
    def task(i, units):
        return {"id": i, "teacher": False, "arch": "a", "method": "m", "units": units}

    med = {("a", "m"): 10.0}
    # one worker busy for 100 s, one free; tasks of 50 s, 50 s and 100 s
    out = forecast([task(1, 5), task(2, 5), task(3, 10)], [100], 2, 0, 120, med, 1.0)
    assert [o["finish"] for o in out] == [50, 100, 200]
    assert [o["by_stop"] for o in out] == [True, True, False]


def _tree(tmp_path):
    """A small smoke-profile outputs tree: one finished run, one failed run, a teacher that is
    done, a half-written result, and a worker training another run right now."""
    students, teachers = build_specs("smoke")
    out, logs = tmp_path / "outputs", tmp_path / "logs"
    done, failed, running = students[0], students[1], students[2]
    d = out / "smoke" / done["id"]
    d.mkdir(parents=True)
    result = {
        "spec": done,
        "val": {"top1": 0.5},
        "test": {"top1": 0.55, "ece": 0.05},
        "train": {
            "train_seconds": 20.0,
            "epochs": [{"epoch": 0, "train_ce": 2.0}],
            "diagnostics": {
                "steps": 10,
                "numeric": {"cos_raw": {"mean": 0.4}, "active_set_0": {"count": 4, "mean": 1.0}},
                "fractions": {"active": 0.4},
            },
            "timeseries": [{"step": 0, "cos_raw": 0.4}],
        },
    }
    (d / "result.json").write_text(json.dumps(result))
    f = out / "smoke" / failed["id"]
    f.mkdir(parents=True)
    (f / "error.json").write_text(json.dumps({"error": "RuntimeError('x')", "traceback": "tb"}))
    (out / "smoke" / "half").mkdir()
    (out / "smoke" / "half" / "result.json").write_text('{"spec": ')  # still being written
    key = next(iter(teachers))
    pt, meta = teacher_paths(out, "smoke", key)
    pt.parent.mkdir(parents=True, exist_ok=True)
    pt.write_bytes(b"x")
    meta.write_text(json.dumps({"test": {"top1": 0.7}, "train": {"train_seconds": 10.0}}))
    logs.mkdir()
    (logs / "worker1-20261112-1955.log").write_text(f"[{done['id']}] epoch 1/9 time=1\n")
    (logs / "worker1-20261113-1700.log").write_text(
        f"[{done['id']}] epoch 1/2 time=10 train_ce=2.0\n[{done['id']}] val=0.5 test=0.55\n"
        f"[{running['id']}] epoch 1/4 time=5 train_ce=2.5\n"
        f"[{running['id']}] epoch 2/4 time=10 train_ce=2.1\n"
        "[pilot/x/s0] skipped: its teacher failed\n"
    )
    return out, logs, done, failed, running, key


def test_scan_of_a_small_tree(tmp_path):
    out, logs, done, failed, running, key = _tree(tmp_path)
    s = Scanner(out, logs, "smoke").scan()
    assert s["counts"]["done"] == 1 and s["results"][0]["test"] == 0.55
    assert s["results"][0]["diag"] == {"cos_raw": 0.4}  # a CE run: no PCD active rate
    pcd = json.loads((out / "smoke" / done["id"] / "result.json").read_text())
    pcd["spec"] = {**pcd["spec"], "method": "pcd"}
    assert compact_result(pcd, 0)["diag"]["active_rate"] == 0.4
    assert [x["id"] for x in s["failures"]] == [failed["id"]]
    assert [w["log"] for w in s["workers"]] == ["worker1-20261113-1700.log"]  # newest session
    (a,) = s["active"]
    assert a["id"] == running["id"] and a["epoch"] == 2 and a["sec_per_epoch"] == 5
    assert a["seconds_left"] == 10
    assert {t["key"]: t["status"] for t in s["teachers"]}[key] == "done"
    assert s["counts"]["skipped_lines"] == 1
    assert s["forecast"]["remaining"] == s["counts"]["remaining"] > 0
    json.dumps(s)  # served as JSON


def test_a_silent_worker_is_not_running_anything(tmp_path):
    out, logs, *_ = _tree(tmp_path)
    old = time.time() - 3600
    os.utime(logs / "worker1-20261113-1700.log", (old, old))
    s = Scanner(out, logs, "smoke").scan()
    assert s["active"] == [] and s["workers"][0]["alive"] is False


@pytest.fixture
def server(tmp_path):
    out, logs, *_ = _tree(tmp_path)
    try:
        srv = make_server(Scanner(out, logs, "smoke"), "127.0.0.1", 0)
    except PermissionError:
        pytest.skip("cannot bind a local socket here")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_http_routes(server):
    assert b"PCD suite monitor" in urllib.request.urlopen(server + "/").read()
    state = json.load(urllib.request.urlopen(server + "/api/state"))
    run_id = state["results"][0]["id"]
    detail = json.load(urllib.request.urlopen(f"{server}/api/run?id={run_id}"))
    assert detail["test"]["top1"] == 0.55 and detail["timeseries"][0]["cos_raw"] == 0.4
    for bad in ("../../logs/worker1-20261113-1700.log", "nope"):
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(f"{server}/api/run?id={bad}")
        assert e.value.code == 404


def test_a_checkpointed_teacher_is_paused_unless_a_worker_is_on_it(tmp_path):
    out, logs, *_ = _tree(tmp_path)
    _, teachers = build_specs("smoke")
    a, b = list(teachers)[1:3]
    for key in (a, b):
        pt = teacher_paths(out, "smoke", key)[0]
        pt.with_name(pt.name.removesuffix(".pt") + ".ckpt").write_bytes(b"x")
    (logs / "worker2-20261113-1700.log").write_text(f"[teacher/{a}] epoch 3/9 time=6\n")
    status = {t["key"]: t["status"] for t in Scanner(out, logs, "smoke").scan()["teachers"]}
    assert (status[a], status[b]) == ("training", "paused")
