#!/usr/bin/env python3
"""Summarise suite results into a "where does PCD win?" report (markdown + CSV).

    uv run python scripts/analyze_suite.py --profile screen

Per family x condition, every method group (pcd, pcdrev, nwkd, wkd, dtokd, dot, ...) gets ONE
configuration, chosen by mean *validation* accuracy; its *test* accuracy is then compared with
the best non-PCD group by seed-paired differences and a bootstrap 95% CI over seeds.
Verdicts are screening evidence (no multiplicity correction; few seeds) -- not confirmatory.
"""

import argparse
import collections
import csv
import json
import math
from pathlib import Path

import numpy as np


def load(root):
    runs = []
    for f in sorted(Path(root).rglob("result.json")):
        r = json.loads(f.read_text())
        s, d = r["spec"], r["train"]["diagnostics"]
        num = d["numeric"]
        ts = r["train"]["timeseries"]

        def mean(k, num=num):
            return num.get(k, {}).get("mean")

        shadow = [x["shadow_pcd_nwkd"] for x in ts if "shadow_pcd_nwkd" in x]
        runs.append(
            {
                "family": s["family"],
                "condition": s["condition"],
                "arm": s["arm"],
                "group": s["group"],
                "pcd": s["method"] == "pcd",
                "seed": s["seed"],
                "val": r["val"]["top1"],
                "test": r["test"]["top1"],
                "ece": r["test"]["ece"],
                "teacher_agreement": r["test"].get("teacher_agreement"),
                "acc_few": r["test"].get("acc_few"),
                "cos": mean("cos_raw"),
                "active": d["fractions"].get("active"),
                "lambda": mean("lambda_eff"),
                "margin_nonpos": d.get("prop47_nonpositive_fraction"),
                "rules_differ": (float(np.mean(np.asarray(shadow) < 0.99)) if shadow else None),
                "cos_by_third": [
                    t["numeric"].get("cos_raw", {}).get("mean")
                    for t in r["train"]["diagnostics_by_third"]
                ],
                "hours": r["train"]["train_seconds"] / 3600,
            }
        )
    return runs


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def fmt(x, pct=True):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{100 * x:.2f}" if pct else f"{x:.3f}"


def best_config(rows):
    """{group: (arm, {seed: row})} picking the arm with the highest mean val accuracy."""
    by_arm = collections.defaultdict(dict)
    for r in rows:
        by_arm[r["arm"]][r["seed"]] = r
    best = {}
    for a, seeds in by_arm.items():
        g = next(iter(seeds.values()))["group"]
        score = np.mean([r["val"] for r in seeds.values()])
        if g not in best or score > best[g][2]:
            best[g] = (a, seeds, score)
    return {g: (a, s) for g, (a, s, _) in best.items()}


def paired(a, b):
    common = sorted(set(a) & set(b))
    diffs = [a[k]["test"] - b[k]["test"] for k in common]
    return (np.mean(diffs) if diffs else math.nan), boot_ci(diffs), len(diffs)


def verdict(lo, hi):
    if math.isnan(lo):
        return "n<2"
    return "**PCD better**" if lo > 0 else ("PCD worse" if hi < 0 else "inconclusive")


def report(runs):
    out = [
        "# PCD-in-KD suite report",
        "",
        "Screening evidence: configs chosen on validation, test reported; paired-seed "
        "bootstrap 95% CIs; no multiplicity correction.",
        "",
    ]
    wins = [
        "## Where does PCD win?",
        "",
        "| family | condition | best PCD arm | best baseline arm | Δ test (pp) | 95% CI | n "
        "| verdict | PCD − NWKD (pp) | mean cos(CE,KD) | PCD active | rules differ |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    detail = []
    cells = collections.defaultdict(list)
    for r in runs:
        cells[(r["family"], r["condition"])].append(r)
    for (fam, cond), rows in sorted(cells.items()):
        best = best_config(rows)
        is_pcd = {g: next(iter(best[g][1].values()))["pcd"] for g in best}
        pcd = [g for g in best if is_pcd[g]]
        base = [g for g in best if not is_pcd[g]]

        def val_mean(g, best=best):
            return np.mean([r["val"] for r in best[g][1].values()])

        if pcd and base:
            p, b = max(pcd, key=val_mean), max(base, key=val_mean)
            m, (lo, hi), n = paired(best[p][1], best[b][1])
            nw = ""
            if "nwkd" in best:
                nm, _, _ = paired(best[p][1], best["nwkd"][1])
                nw = fmt(nm)
            prow = list(best[p][1].values())

            def pmean(k, prow=prow):
                v = [r[k] for r in prow if r[k] is not None]
                return float(np.mean(v)) if v else math.nan

            wins.append(
                f"| {fam} | {cond} | {best[p][0]} | {best[b][0]} | {fmt(m)} | "
                f"[{fmt(lo)}, {fmt(hi)}] | {n} | {verdict(lo, hi)} | {nw} | "
                f"{fmt(pmean('cos'), False)} | {fmt(pmean('active'))} | "
                f"{fmt(pmean('rules_differ'))} |"
            )
        detail += [
            f"### {fam} / {cond}",
            "",
            "| arm | seeds | val | test (mean ± sd) | Δ vs CE | ECE | few-shot acc "
            "| cos(CE,KD) by third | active | λ_eff | margin ≤ 0 |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        by_arm = collections.defaultdict(list)
        for r in rows:
            by_arm[r["arm"]].append(r)
        ce = np.mean([r["test"] for r in by_arm.get("ce", [])]) if "ce" in by_arm else None
        for a, rs in sorted(by_arm.items(), key=lambda kv: -np.mean([r["val"] for r in kv[1]])):
            t = [r["test"] for r in rs]

            def avg(k, rs=rs):
                v = [r[k] for r in rs if r[k] is not None]
                return float(np.mean(v)) if v else None

            thirds = " / ".join(
                fmt(
                    np.mean(
                        [r["cos_by_third"][i] for r in rs if r["cos_by_third"][i] is not None]
                        or [math.nan]
                    ),
                    False,
                )
                for i in range(3)
            )
            detail.append(
                f"| {a} | {len(rs)} | {fmt(avg('val'))} | {fmt(np.mean(t))} ± "
                f"{fmt(np.std(t, ddof=1) if len(t) > 1 else math.nan)} | "
                f"{fmt(np.mean(t) - ce) if ce is not None else '–'} | {fmt(avg('ece'), False)} | "
                f"{fmt(avg('acc_few'))} | {thirds} | {fmt(avg('active'))} | "
                f"{fmt(avg('lambda'), False)} | {fmt(avg('margin_nonpos'))} |"
            )
        detail.append("")
    sens = sensitivity(runs)
    return "\n".join(out + wins + [""] + sens + ["## Per-cell detail", ""] + detail)


def sensitivity(runs):
    """Best / median / worst test accuracy over each group's grid (tuning robustness)."""
    rows = [
        "## Tuning robustness (all configs of each method group)",
        "",
        "| family | condition | group | configs | best | median | worst | spread |",
        "|---|---|---|---|---|---|---|---|",
    ]
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in runs:
        if r["family"] in ("sensitivity", "loss-scale", "temperature", "tau-sweep"):
            key = (r["family"], r["condition"] if r["family"] == "sensitivity" else "all")
            cells[key][(r["group"], r["arm"], r["condition"])].append(r["test"])
    for (fam, cond), arms in sorted(cells.items()):
        groups = collections.defaultdict(list)
        for (g, _, _), t in arms.items():
            groups[g].append(np.mean(t))
        for g, v in sorted(groups.items()):
            rows.append(
                f"| {fam} | {cond} | {g} | {len(v)} | {fmt(max(v))} | "
                f"{fmt(float(np.median(v)))} | {fmt(min(v))} | {fmt(max(v) - min(v))} |"
            )
    return rows + [""]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default="screen")
    p.add_argument("--out", default="outputs/suite")
    args = p.parse_args()
    root = Path(args.out) / args.profile
    runs = load(root)
    if not runs:
        raise SystemExit(f"no result.json under {root}")
    (root / "REPORT.md").write_text(report(runs))
    with (root / "runs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0]))
        w.writeheader()
        w.writerows(runs)
    errors = list(root.rglob("error.json"))
    print(f"{len(runs)} runs -> {root / 'REPORT.md'}, {root / 'runs.csv'}; {len(errors)} failed")


if __name__ == "__main__":
    main()
