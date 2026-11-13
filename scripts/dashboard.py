"""Read-only live dashboard for a PCD suite run: http://127.0.0.1:8765

    uv run --no-sync python scripts/dashboard.py --out <outputs> [--logs <logs>] [--profile screen]

Reads what the suite already writes (result.json, error.json, checkpoints, teacher files, worker
logs) and never writes to those folders. Works on the workstation while runs are going, and on a
Mac against a Google Drive download (pass --logs where the logs folder is).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from pcdkd.experiments.runner import queue_tasks, teacher_paths
from pcdkd.experiments.suite import PRIORITY, build_specs, queue_order
from pcdkd.experiments.suite import cost as work_units
from pcdkd.experiments.window import UAE, window_close

STOP_MARGIN = timedelta(minutes=20)
DIAG = ("cos_raw", "lambda_eff", "prop47_margin_actual")
EPOCH_RE = re.compile(r"^\[(?P<id>[^\]]+)\] epoch (?P<e>\d+)/(?P<n>\d+) (?P<rest>.*)$")
KV_RE = re.compile(r"(\w+)=([-+0-9.eE]+|nan|inf)")
WORKER_LOG_RE = re.compile(r"^worker(?P<n>\d+)-(?P<stamp>\d{8}-\d{4})\.log$")


# --- parsing --------------------------------------------------------------------------------------
def parse_line(line):
    """One worker-log line -> dict with ``kind`` (epoch, result, failed, skipped, resumed, paused,
    window) and ``id`` where the line names a task; None for anything else."""
    line = line.strip()
    m = EPOCH_RE.match(line)
    if m:
        d = {"kind": "epoch", "id": m["id"], "epoch": int(m["e"]), "epochs": int(m["n"])}
        d.update({k: float(v) for k, v in KV_RE.findall(m["rest"])})
        return d
    if line.startswith("PAUSED") or line.startswith("access window closing"):
        return {"kind": "paused", "msg": line}
    m = re.match(r"^\[(?P<id>[^\]]+)\] (?P<msg>.*)$", line)
    if not m:
        return None
    tid, msg = m["id"], m["msg"]
    if tid == "window":
        return {"kind": "window", "msg": msg}
    if msg.startswith("FAILED"):
        return {"kind": "failed", "id": tid, "msg": msg.removeprefix("FAILED: ")}
    if msg.startswith("skipped"):
        return {"kind": "skipped", "id": tid, "msg": msg}
    if msg.startswith("resumed"):
        return {"kind": "resumed", "id": tid, "msg": msg}
    if tid.startswith("teacher ") and msg.startswith("test top1="):  # [teacher <key>] test top1=
        return {"kind": "result", "id": "teacher/" + tid.removeprefix("teacher ")}
    if msg.startswith("val="):
        return {"kind": "result", "id": tid, **{k: float(v) for k, v in KV_RE.findall(msg)}}
    return None


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None  # missing or half-written: picked up on the next scan


def mtime(path):
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return None


def units(spec):
    """Work in full-training-set passes (epochs x backward passes x share of the data used), so
    a low-data or long-tail run is not charged as a full-data one."""
    try:
        return work_units(spec)
    except (KeyError, TypeError):
        return spec.get("epochs", 0)


def compact_result(r, ts):
    s, train = r["spec"], r.get("train", {})
    diagnostics = train.get("diagnostics") or {}
    nums = diagnostics.get("numeric", {})
    test = r.get("test") or {}
    return {
        "id": s["id"],
        "family": s["family"],
        "condition": s["condition"],
        "arm": s["arm"],
        "method": s.get("method"),
        "seed": s["seed"],
        "arch": s.get("arch"),
        "epochs": s.get("epochs"),
        "units": units(s),
        "test": test.get("top1"),
        "val": (r.get("val") or {}).get("top1"),
        "ece": test.get("ece"),
        "seconds": train.get("train_seconds"),
        "diag": {
            **{k: nums[k]["mean"] for k in DIAG if k in nums},
            # share of steps with PCD's constraint active (as in analyze_suite.py)
            **(
                {"active_rate": diagnostics["fractions"]["active"]}
                if "active" in diagnostics.get("fractions", {})
                else {}
            ),
        },
        "finished": ts,
    }


# --- forecast -------------------------------------------------------------------------------------
def cost_model(results, teacher_meta):
    """Seconds per work unit (see ``units``), measured while sharing the GPU: medians by
    (arch, method), by (arch, None) and by ("teacher", arch); plus the overall student median."""
    groups = {}
    for r in results:
        if r["seconds"] and r["units"]:
            x = r["seconds"] / r["units"]
            groups.setdefault((r["arch"], r["method"]), []).append(x)
            groups.setdefault((r["arch"], None), []).append(x)
    every = sorted(x for k, v in groups.items() if k[1] is None for x in v)
    for arch, sec in teacher_meta:
        groups.setdefault(("teacher", arch), []).append(sec)
    med = {k: sorted(v)[len(v) // 2] for k, v in groups.items()}
    return med, (every[len(every) // 2] if every else 12.0)


def task_cost(t, med, fallback):
    """Estimated seconds per work unit for one queued task."""
    if t["teacher"]:
        return med.get(("teacher", t["arch"])) or med.get((t["arch"], None)) or 1.5 * fallback
    return med.get((t["arch"], t["method"])) or med.get((t["arch"], None)) or fallback


def forecast(remaining, running, lanes, now, stop, med, fallback):
    """``lanes`` workers take ``remaining`` tasks in queue order, each starting when a worker frees
    up; tasks in progress (``running``: seconds left) occupy workers first. Returns each task with
    its projected finish time (unix seconds) and whether that is before ``stop``."""
    lanes = max(1, lanes)
    free = sorted([now + s for s in running] + [now] * max(0, lanes - len(running)))[:lanes]
    free += [now] * (lanes - len(free))
    out = []
    for t in remaining:
        start = min(free)
        free.remove(start)
        end = start + task_cost(t, med, fallback) * t["units"]
        free.append(end)
        out.append({**t, "finish": end, "by_stop": end <= stop})
    return out


# --- the scanner ----------------------------------------------------------------------------------
class Scanner:
    def __init__(self, out, logs, profile):
        self.out, self.logs, self.profile = Path(out), Path(logs), profile
        self.base = self.out / profile
        students, self.teacher_specs = build_specs(profile)
        self.tasks = []
        for kind, x in queue_tasks(queue_order(students), self.teacher_specs):
            spec = self.teacher_specs[x] if kind == "teacher" else x
            self.tasks.append(
                {
                    "id": spec["id"],
                    "key": x if kind == "teacher" else None,
                    "teacher": kind == "teacher",
                    "family": spec["family"],
                    "tier": spec["tier"],
                    "seed": spec["seed"],
                    "epochs": spec["epochs"],
                    "units": units(spec),
                    "arch": spec["arch"],
                    "method": spec.get("method"),
                    "teacher_keys": [] if kind == "teacher" else spec["teacher_keys"],
                }
            )
        self._results = {}  # path -> (mtime, compact result)
        self._logs = {}  # path -> (mtime, size, parsed lines)
        self.gpu = deque(maxlen=6 * 360)  # 6 h at one sample per 10 s
        self.lock = threading.Lock()
        self.cache = (0.0, None)

    def sample_gpu(self, every=10):
        query = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
        while True:
            try:
                line = subprocess.run(  # noqa: S603 - fixed argv
                    ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],  # noqa: S607
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                ).stdout.splitlines()[0]
                util, used, total, temp, power = (float(x) for x in line.split(","))
                self.gpu.append(
                    {
                        "t": time.time(),
                        "util": util,
                        "mem": used,
                        "mem_total": total,
                        "temp": temp,
                        "power": power,
                    }
                )
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass  # no NVIDIA driver here (e.g. the Mac): the page says "no GPU telemetry"
            time.sleep(every)

    def _result(self, path):
        ts = mtime(path)
        hit = self._results.get(path)
        if hit and hit[0] == ts:
            return hit[1]
        r = read_json(path)
        c = compact_result(r, ts) if isinstance(r, dict) and r.get("spec") else None
        if c:
            self._results[path] = (ts, c)
        return c

    def _log(self, path):
        try:
            st = path.stat()
            hit = self._logs.get(path)
            if hit and hit[0] == st.st_mtime and hit[1] == st.st_size:
                return hit[2]
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        parsed = [p for p in map(parse_line, text.splitlines()) if p]
        self._logs[path] = (st.st_mtime, st.st_size, parsed)
        return parsed

    def worker_logs(self):
        """The newest log of each worker number (the current session)."""
        newest = {}
        for p in self.logs.glob("worker*.log") if self.logs.exists() else []:
            m = WORKER_LOG_RE.match(p.name)
            if m and (m["n"] not in newest or m["stamp"] > newest[m["n"]][0]):
                newest[m["n"]] = (m["stamp"], p)
        return [newest[n][1] for n in sorted(newest, key=int)]

    def scan(self):
        with self.lock:
            if time.time() - self.cache[0] < 10 and self.cache[1] is not None:
                return self.cache[1]
            state = self._scan()
            self.cache = (time.time(), state)
            return state

    def _teacher_file(self, key, suffix):
        pt = teacher_paths(self.out, self.profile, key)[0]
        return pt.with_name(pt.name.removesuffix(".pt") + suffix)

    def _scan(self):  # noqa: C901, PLR0912, PLR0915 - one pass over everything the page shows
        now = time.time()
        now_dt = datetime.now(UAE)
        close = window_close(now_dt)
        if close is None:  # outside the window: show the next one
            close = window_close(now_dt + timedelta(days=(3 - now_dt.weekday()) % 7, hours=12))
        stop = (close - STOP_MARGIN).timestamp() if close else now

        results = {}
        for p in self.base.rglob("result.json") if self.base.exists() else []:
            c = self._result(str(p))
            if c:
                results[c["id"]] = c

        # teachers
        teachers, teacher_meta = [], []
        for key, spec in self.teacher_specs.items():
            pt, meta_path = teacher_paths(self.out, self.profile, key)
            err_path = self._teacher_file(key, ".error.json")
            meta = read_json(meta_path) if pt.exists() else None
            err = read_json(err_path) if err_path.exists() else None
            status = (
                "done"
                if pt.exists()
                else "failed"
                if err
                else "training"
                if self._teacher_file(key, ".ckpt").exists()
                else "not started"
            )
            secs = ((meta or {}).get("train") or {}).get("train_seconds")
            if secs and units(spec):
                teacher_meta.append((spec["arch"], secs / units(spec)))
            teachers.append(
                {
                    "key": key,
                    "id": spec["id"],
                    "arch": spec["arch"],
                    "dataset": spec["dataset"],
                    "status": status,
                    "epochs": spec["epochs"],
                    "test": ((meta or {}).get("test") or {}).get("top1"),
                    "error": (err or {}).get("error"),
                }
            )
        teachers_done = {t["key"] for t in teachers if t["status"] == "done"}
        teachers_failed = {t["key"] for t in teachers if t["status"] == "failed"}

        # failures; a later success hides an old error
        failures = []
        for p in self.base.rglob("error.json") if self.base.exists() else []:
            if p.parent.name == "teachers":
                continue
            run_id = p.parent.relative_to(self.base).as_posix()
            if run_id in results:
                continue
            d = read_json(p) or {}
            failures.append(
                {
                    "id": run_id,
                    "error": d.get("error"),
                    "traceback": d.get("traceback"),
                    "time": mtime(p),
                }
            )
        for t in teachers:
            if t["status"] == "failed":
                p = self._teacher_file(t["key"], ".error.json")
                d = read_json(p) or {}
                failures.append(
                    {
                        "id": t["id"],
                        "error": d.get("error"),
                        "traceback": d.get("traceback"),
                        "time": mtime(p),
                    }
                )
        failures.sort(key=lambda f: f["time"] or 0, reverse=True)

        # workers and what each is doing now
        workers, active, skipped, events = [], [], 0, []
        for path in self.worker_logs():
            lines = self._log(path)
            age = now - (mtime(path) or now)
            current, curve, finished_ids = None, [], set()
            for ln in lines:
                if ln["kind"] == "skipped":
                    skipped += 1
                elif ln["kind"] in ("result", "failed"):
                    finished_ids.add(ln["id"])
                elif ln["kind"] == "epoch":
                    if current != ln["id"]:
                        current, curve = ln["id"], []
                    curve.append(ln)
                if ln["kind"] in ("failed", "paused", "window", "resumed"):
                    events.append({"worker": path.name, **ln})
            name = path.name.split("-")[0]
            done_teacher = bool(current) and current.removeprefix("teacher/") in teachers_done
            ended = current in finished_ids or current in results or done_teacher
            busy = bool(current) and not ended and age < 180  # a silent worker is not running it
            workers.append(
                {
                    "name": name,
                    "log": path.name,
                    "age": age,
                    "alive": age < 180,
                    "current": current if busy else None,
                }
            )
            if busy:
                last = curve[-1]
                per_epoch = last.get("time", 0.0) / max(1, last["epoch"])
                left = per_epoch * (last["epochs"] - last["epoch"])
                active.append(
                    {
                        "id": current,
                        "worker": name,
                        "epoch": last["epoch"],
                        "epochs": last["epochs"],
                        "sec_per_epoch": per_epoch,
                        "seconds_left": left,
                        "eta": now + left,
                        "log_age": age,
                        "curve": [
                            {k: c.get(k) for k in ("epoch", "train_ce", "train_kd", "val_top1")}
                            for c in curve
                        ],
                    }
                )
        active_ids = {a["id"] for a in active}
        for t in teachers:  # a checkpoint alone means paused; "training" needs a live worker on it
            if t["status"] == "training" and t["id"] not in active_ids:
                t["status"] = "paused"

        # queue status and forecast
        med, fallback = cost_model(list(results.values()), teacher_meta)
        failed_ids = {f["id"] for f in failures}
        remaining, fam = [], {}
        for t in self.tasks:
            f = fam.setdefault(
                t["family"],
                {
                    "family": t["family"],
                    "tier": t["tier"],
                    "total": 0,
                    "done": 0,
                    "running": 0,
                    "failed": 0,
                    "blocked": 0,
                    "left": 0,
                    "seed0_total": 0,
                    "seed0_done": 0,
                    "by_stop": 0,
                    "gpu_hours_left": 0.0,
                },
            )
            f["total"] += 1
            f["seed0_total"] += t["seed"] == 0
            if t["key"] in teachers_done if t["teacher"] else t["id"] in results:
                f["done"] += 1
                f["seed0_done"] += t["seed"] == 0
                continue
            if t["id"] in active_ids:
                f["running"] += 1
                continue
            if t["id"] in failed_ids:
                f["failed"] += 1
            elif any(k in teachers_failed for k in t["teacher_keys"]):
                f["blocked"] += 1
            else:
                f["left"] += 1
            remaining.append(t)  # failed and blocked tasks are retried in the next session
        alive = sum(w["alive"] for w in workers)
        lanes = alive or len(workers) or 3
        proj = forecast(
            remaining, [a["seconds_left"] for a in active], lanes, now, stop, med, fallback
        )
        for p in proj:
            f = fam[p["family"]]
            f["by_stop"] += p["by_stop"]
            f["gpu_hours_left"] += task_cost(p, med, fallback) * p["units"] / 3600 / lanes
        order = {name: i for i, name in enumerate(PRIORITY)}
        families = sorted(
            fam.values(), key=lambda f: (f["family"] != "teachers", order.get(f["family"], 99))
        )

        # throughput: student epoch-passes finished per hour
        fin = sorted((r for r in results.values() if r["finished"]), key=lambda r: r["finished"])
        recent = [r for r in fin if r["finished"] > now - 2 * 3600]
        span_h = (fin[-1]["finished"] - fin[0]["finished"]) / 3600 if len(fin) > 1 else 0
        seed0 = [p["finish"] for p in proj if p["seed"] == 0]
        first_miss = next((p for p in proj if not p["by_stop"]), None)
        return {
            "now": now,
            "close": close.timestamp() if close else None,
            "stop": stop,
            "lanes": lanes,
            "workers": workers,
            "gpu": list(self.gpu)[-360:],  # the last hour
            "ram": windows_ram(),
            "counts": {
                "done": len(results),
                "running": len(active),
                "failed": len(failures),
                "remaining": len(remaining),
                "teachers_done": len(teachers_done),
                "teachers": len(teachers),
                "skipped_lines": skipped,
                "total": len(self.tasks),
            },
            "last_result": fin[-1]["finished"] if fin else None,
            "throughput": {
                "recent_per_hour": sum(r["epochs"] for r in recent) / 2,
                "overall_per_hour": sum(r["epochs"] for r in fin) / span_h if span_h else None,
            },
            "forecast": {
                "remaining": len(proj),
                "by_stop": sum(p["by_stop"] for p in proj),
                "first_miss": first_miss
                and {"id": first_miss["id"], "family": first_miss["family"]},
                "seed0_left": len(seed0),
                "seed0_end": max(seed0) if seed0 else None,
                "end": proj[-1]["finish"] if proj else None,
                "sec_per_unit": fallback,
                "next": [
                    {"id": p["id"], "finish": p["finish"], "by_stop": p["by_stop"]}
                    for p in proj[:12]
                ],
            },
            "families": families,
            "teachers": teachers,
            "active": active,
            "events": events[-30:],
            "failures": failures[:200],
            "results": list(results.values()),
            "timeline": [{"t": r["finished"], "n": i + 1} for i, r in enumerate(fin)],
        }

    def run_detail(self, run_id):
        path = (self.base / run_id / "result.json").resolve()
        if self.base.resolve() not in path.parents:  # no reading outside the outputs folder
            return None
        d = read_json(path)
        if not isinstance(d, dict):
            return None
        train = d.get("train") or {}
        keep = ("step", "epoch", "cos_raw", "lambda_eff", "loss_ce", "loss_kd")
        return {
            "id": run_id,
            "epochs": train.get("epochs", []),
            "timeseries": [{k: x.get(k) for k in keep} for x in train.get("timeseries") or []],
            "test": {k: v for k, v in (d.get("test") or {}).items() if k != "per_class_acc"},
        }


def windows_ram():
    if os.name != "nt":
        return None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
            (n, ctypes.c_ulonglong) for n in ("total", "avail", "tpf", "apf", "tv", "av", "aev")
        ]

    m = MemoryStatus()
    m.dwLength = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        return None
    return {"used": m.total - m.avail, "total": m.total}


# --- HTTP -----------------------------------------------------------------------------------------
def make_server(scanner, host, port):
    page = (Path(__file__).parent / "dashboard" / "index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body, ctype, code=200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 - http.server API
            u = urlparse(self.path)
            try:
                if u.path == "/":
                    self._send(page, "text/html; charset=utf-8")
                elif u.path == "/api/state":
                    self._send(json.dumps(scanner.scan()).encode(), "application/json")
                elif u.path == "/api/run":
                    d = scanner.run_detail(parse_qs(u.query).get("id", [""])[0])
                    self._send(
                        json.dumps(d or {"error": "not found"}).encode(),
                        "application/json",
                        200 if d else 404,
                    )
                else:
                    self._send(b"not found", "text/plain", 404)
            except Exception as exc:  # noqa: BLE001 - one bad scan must not kill the server
                self._send(json.dumps({"error": repr(exc)}).encode(), "application/json", 500)

        def log_message(self, *a):
            pass

    return ThreadingHTTPServer((host, port), Handler)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.environ.get("PCDKD_OUT"))
    ap.add_argument("--logs", help="worker logs folder (default: <out>/../logs)")
    ap.add_argument("--profile", default="screen")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    if not a.out:
        ap.error("--out (or PCDKD_OUT) is required")
    scanner = Scanner(a.out, a.logs or Path(a.out).parent / "logs", a.profile)
    threading.Thread(target=scanner.sample_gpu, daemon=True).start()
    server = make_server(scanner, a.host, a.port)
    print(f"dashboard: http://{a.host}:{a.port}  (out={a.out})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
