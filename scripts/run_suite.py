#!/usr/bin/env python3
"""List or run the PCD-in-KD experiment suite.

    uv run python scripts/run_suite.py list --profile screen --tier 0 1
    uv run python scripts/run_suite.py run --profile screen --stage teachers
    uv run python scripts/run_suite.py run --profile screen --tier 1 --shard 0/4   # GPU 0 of 4
    uv run python scripts/run_suite.py run --profile smoke                         # CPU, synthetic
    uv run python scripts/run_suite.py clear-claims --profile screen               # once, then:
    uv run python scripts/run_suite.py run --profile screen --queue                # x N processes

``--queue`` runs the whole selection in ``suite.WEEKEND_PLAN`` order, then the unmatched tail in
the legacy seed / ``suite.PRIORITY`` / design order, training each teacher just before its first
user. Any number of processes can share one queue: each claims its next task.

Shards split students only; run ``--stage teachers`` once before launching parallel shards.

On the AUS workstation use ``--window aus`` (or PCDKD_WINDOW=aus): the run refuses to start outside
Thursday 18:00 - Monday 08:00 (UAE time) and pauses with a checkpoint before the window closes.
Rerunning the same command in the next window resumes. ``--out`` defaults to $PCDKD_OUT.
DeepFreeze wipes the local disk there, so scripts/workstation/run_weekend.ps1 copies it to
Google Drive with rclone.
"""

import argparse
import collections
import ctypes
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

from pcdkd.experiments.runner import (
    GpuFault,
    clear_claims,
    queue_tasks,
    run_many,
    run_queue,
)
from pcdkd.experiments.suite import (
    FAMILIES,
    FAMILY_NAMES,
    PROFILES,
    build_specs,
    cost,
    queue_order,
)
from pcdkd.experiments.window import UAE, StopGuard, window_close


def select(args):
    students, teachers = build_specs(args.profile, args.family, args.tier)
    if args.ids:
        students = [s for s in students if any(i in s["id"] for i in args.ids)]
    if getattr(args, "queue", False):
        students = queue_order(students)
    else:
        students.sort(key=lambda s: (s["tier"], s["id"]))
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        students = students[i::n]
    return students, teachers


def cmd_list(args):
    students, teachers = select(args)
    by_fam = collections.defaultdict(list)
    for s in students:
        by_fam[s["family"]].append(s)
    print("| tier | family | conditions | arms | seeds | runs | student epoch-passes |")
    print("|---|---|---|---|---|---|---|")
    total_runs = total_cost = 0
    for fam in FAMILIES:
        runs = by_fam.get(fam["name"])
        if not runs:
            continue
        fam_cost = round(sum(cost(s) for s in runs))
        total_runs, total_cost = total_runs + len(runs), total_cost + fam_cost
        print(
            f"| {fam['tier']} | {fam['name']} | {len({s['condition'] for s in runs})} | "
            f"{len({s['arm'] for s in runs})} | {len({s['seed'] for s in runs})} | "
            f"{len(runs)} | {fam_cost:,} |"
        )
    t_cost = sum(
        t["epochs"] for k, t in teachers.items() if any(k in s["teacher_keys"] for s in students)
    )
    print(
        f"\n**{total_runs} student runs, {total_cost:,} student epoch-passes; "
        f"{len({k for s in students for k in s['teacher_keys']})} teachers "
        f"({t_cost:,} teacher epochs).** Multiply by the pilot's measured seconds per "
        "epoch-pass for each architecture to get GPU-hours."
    )
    if args.verbose:
        for s in students:
            print(s["id"])


def keep_awake():
    """Stop Windows sleeping while this process runs (no admin needed); no-op elsewhere."""
    if sys.platform == "win32":
        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


def make_guard(window, margin_min, now=None):
    """StopGuard for ``window`` ('aus' or 'none'); None if outside the AUS access window."""
    if window == "none":
        return StopGuard(None)
    close = window_close(now or datetime.now(UAE))
    if close is None:
        return None
    return StopGuard(close, timedelta(minutes=margin_min))


def cmd_run(args):
    students, teachers = select(args)
    if args.dry_run:
        if args.queue:
            for kind, x in queue_tasks(students, teachers):
                print(f"teacher {x}" if kind == "teacher" else x["id"])
        else:
            for s in students:
                print(s["id"])
        return 0
    guard = make_guard(args.window, args.stop_margin_min)
    if guard is None:
        print(
            "outside the workstation access window (Thursday 18:00 - Monday 08:00, UAE time); "
            "not starting",
            file=sys.stderr,
        )
        return 2
    print(f"[window] {guard.describe()}; results -> {args.out}")
    keep_awake()
    if args.queue:
        if args.shard or args.stage != "all":
            print("--queue replaces --shard and --stage", file=sys.stderr)
            return 2
        try:
            status = run_queue(
                students,
                teachers,
                args.out,
                data_root=args.data_root,
                device=args.device,
                workers=args.workers,
                ckpt_every=args.ckpt_every,
                should_stop=guard,
            )
        except GpuFault as e:
            print(f"GPU fault on {e}; this process exits so a fresh one can take over")
            return GPU_FAULT
        print(status)
        if status["paused"]:
            print("paused for the access window; rerun the same command next window to resume")
        return 1 if status["failed"] else 0
    status = run_many(
        students,
        teachers,
        args.out,
        stage=args.stage,
        data_root=args.data_root,
        device=args.device,
        workers=args.workers,
        ckpt_every=args.ckpt_every,
        should_stop=guard,
    )
    print(status)
    if status["paused"]:
        print("paused for the access window; rerun the same command next window to resume")
    return 1 if status["failed"] else 0


GPU_FAULT = 3


def supervise(cmd, guard, run=subprocess.call, sleep=time.sleep, pause=120):
    """Run the queue worker ``cmd`` in a child process, and start a fresh one after a GPU fault
    (a dead CUDA context cannot recover in-process) until the window closes."""
    while True:
        code = run(cmd, env={**os.environ, "PCDKD_CHILD": "1"})
        if code != GPU_FAULT or guard is None or guard(0.0):
            return code
        print(f"[worker] GPU fault; starting a fresh worker in {pause} s", flush=True)
        sleep(pause)
        if guard(0.0):
            return code


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("command", choices=["list", "run", "clear-claims"])
    p.add_argument("--profile", choices=list(PROFILES), default="screen")
    p.add_argument("--family", nargs="*", choices=FAMILY_NAMES)
    p.add_argument("--tier", nargs="*", type=int)
    p.add_argument("--ids", nargs="*", help="substring filter on run ids")
    p.add_argument("--shard", help="i/n: run every n-th student starting at i")
    p.add_argument("--stage", choices=["all", "teachers", "students"], default="all")
    p.add_argument("--out", default=os.environ.get("PCDKD_OUT", "outputs/suite"))
    p.add_argument("--data-root", default="data")
    p.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--window",
        choices=["aus", "none"],
        default=os.environ.get("PCDKD_WINDOW", "none"),
        help="aus: enforce the workstation access window (default $PCDKD_WINDOW or none)",
    )
    p.add_argument(
        "--stop-margin-min",
        type=float,
        default=20,
        help="stop this many minutes before the window closes (checkpoint sync time)",
    )
    p.add_argument("--ckpt-every", type=int, default=5, help="checkpoint every N epochs")
    p.add_argument(
        "--queue", action="store_true", help="shared priority queue (run several processes)"
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    if args.command == "clear-claims":
        print(f"cleared {clear_claims(args.out, args.profile)} claims")
        return 0
    if args.command == "list":
        return cmd_list(args)
    if args.queue and not args.dry_run and not os.environ.get("PCDKD_CHILD"):
        argv = sys.argv[1:] if argv is None else argv
        guard = make_guard(args.window, args.stop_margin_min)
        return supervise([sys.executable, os.path.abspath(__file__), *argv], guard)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
