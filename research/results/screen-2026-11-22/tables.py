"""Rebuild every table in this folder's README from the committed CSVs (standard library only).

    python research/results/screen-2026-11-22/tables.py                  # print the tables
    python research/results/screen-2026-11-22/tables.py --extract <dir>  # rebuild diagnostics.csv

<dir> holds the raw result.json files (rclone copy gdrive:pcd-suite/outputs/screen <dir>
--include result.json). runs.csv and REPORT.md come from scripts/analyze_suite.py.
"""

import csv
import json
import math
import pathlib
import statistics as st
import sys
from collections import defaultdict

HERE = pathlib.Path(__file__).parent
# two-sided 95% and one-sided 95% (for the 90% TOST interval) t quantiles by degrees of freedom
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
T95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833}
DIAG_ARMS = (
    "pcd_t0.5",
    "pcd_t1.0",
    "g_pcd_t0.5",
    "pcdl_t1.0",
    "pcdlm_t1.0",
    "pcdrev_t0.5",
    "pcdrev_t1.0",
)


def load_runs():
    acc = defaultdict(dict)
    for r in csv.DictReader(open(HERE / "runs.csv", encoding="utf-8")):
        acc[(r["family"], r["condition"], r["arm"])][int(r["seed"])] = 100 * float(r["test"])
    return acc


def paired(acc, fam, cond, a, b, cond_b=None, fam_b=None):
    """Mean of seed-paired differences a - b with a 95% t-interval and a 90% (TOST) interval."""
    A, B = acc.get((fam, cond, a), {}), acc.get((fam_b or fam, cond_b or cond, b), {})
    d = [A[s] - B[s] for s in sorted(set(A) & set(B))]
    if len(d) < 2:
        return None
    m, se = st.mean(d), st.stdev(d) / math.sqrt(len(d))
    k = len(d) - 1
    return m, (m - T975[k] * se, m + T975[k] * se), (m - T95[k] * se, m + T95[k] * se), len(d)


def fmt_diff(p):
    if p is None:
        return "n < 2"
    m, ci, _, n = p
    return f"{m:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] (n={n})"


def mean_table(acc, fam, conds, arms):
    print(f"\n### {fam}: mean test top-1 (%) (seeds)\n")
    print("| Arm | " + " | ".join(conds) + " |")
    print("|---" * (len(conds) + 1) + "|")
    for a in arms:
        cells = []
        for c in conds:
            v = acc.get((fam, c, a), {})
            cells.append(f"{st.mean(v.values()):.2f} ({len(v)})" if v else "–")
        if any(x != "–" for x in cells):
            print(f"| {a} | " + " | ".join(cells) + " |")


def diff_table(acc, title, rows):
    """rows: (fam, cond, a, b) or (fam, cond, a, b, fam_b, cond_b) when b comes from elsewhere."""
    print(f"\n### {title}\n")
    print("| Comparison | Condition | Difference (pp) [95% interval] |")
    print("|---|---|---|")
    for fam, cond, a, b, *other in rows:
        fb, cb = other or (None, None)
        label = f"{a} − {b}" + (f" ({fb} {cb})" if fb else "")
        print(f"| {label} | {fam} {cond} | {fmt_diff(paired(acc, fam, cond, a, b, cb, fb))} |")


def tost_row(acc, label, fam, cond, a, b, delta=0.5, one_sided=False):
    """Equivalence: the 90% interval inside ±delta. one_sided: H2's "a is not higher than b"."""
    p = paired(acc, fam, cond, a, b)
    if p is None:
        print(f"| {label} | {a} − {b}, {cond} | n < 2 | | |")
        return
    m, _, (lo, hi), n = p
    if one_sided:
        verdict = "consistent (PCD not shown higher)" if lo < 0 else "contradicted (PCD higher)"
    else:
        verdict = "equivalent" if -delta < lo and hi < delta else "not shown equivalent"
    print(
        f"| {label} | {a} − {b}, {cond} | {m:+.2f} (n={n}) | [{lo:+.2f}, {hi:+.2f}] | {verdict} |"
    )


def extract(results_dir):
    """Per-run diagnostics of the PCD arms: cosine, weight, its spread and its drift by thirds."""
    rows = []
    for p in sorted(pathlib.Path(results_dir).rglob("result.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        s = r["spec"]
        if s.get("kind") != "student" or s["arm"] not in DIAG_ARMS:
            continue
        d = r["train"]["diagnostics"]
        num = d["numeric"]
        get = lambda k, f="mean", n=num: (n.get(k) or {}).get(f)  # noqa: E731
        thirds = [t["numeric"] for t in r["train"].get("diagnostics_by_third", [])]
        rows.append(
            {
                "family": s["family"],
                "condition": s["condition"],
                "arm": s["arm"],
                "seed": s["seed"],
                "test": r["test"]["top1"],
                "cos_mean": get("cos_raw"),
                "cos_sd": get("cos_raw", "std"),
                "lambda_mean": get("lambda_eff"),
                "lambda_sd": get("lambda_eff", "std"),
                **{f"lambda_third{i + 1}": get("lambda_eff", n=t) for i, t in enumerate(thirds)},
                **{f"cos_third{i + 1}": get("cos_raw", n=t) for i, t in enumerate(thirds)},
                "active": d.get("fractions", {}).get("active"),
                "cos_step_pcd_nwkd": get("shadow_pcd_nwkd"),
                "cos_step_pcd_ce": get("shadow_pcd_ce"),
            }
        )
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(HERE / "diagnostics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} runs -> {HERE / 'diagnostics.csv'}")


def diagnostics_table():
    path = HERE / "diagnostics.csv"
    if not path.exists():
        return
    g = defaultdict(list)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        g[(r["family"], r["condition"], r["arm"])].append(r)
    cols = [
        "cos_mean",
        "cos_sd",
        "lambda_mean",
        "lambda_sd",
        "lambda_third1",
        "lambda_third2",
        "lambda_third3",
        "active",
        "cos_step_pcd_nwkd",
        "cos_step_pcd_ce",
    ]
    print("\n### PCD diagnostics, mean over seeds\n")
    print("| Family | Condition | Arm | n | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 4) + "|")
    for (fam, cond, arm), rs in sorted(g.items()):

        def m(k, rs=rs):
            v = [float(r[k]) for r in rs if r.get(k) not in (None, "")]
            return f"{st.mean(v):.3f}" if v else "–"

        print(f"| {fam} | {cond} | {arm} | {len(rs)} | " + " | ".join(m(k) for k in cols) + " |")


def main():
    acc = load_runs()
    rho = ["rho0.0", "rho0.1", "rho0.3", "rho0.5", "rho0.7"]
    tc_arms = [
        "ce",
        "nwkd_l0.25",
        "nwkd_l0.5",
        "nwkd_l0.75",
        "nwkd_l1.0",
        "pcd_t0.1",
        "pcd_t0.5",
        "pcd_t1.0",
        "pcdl_t0.5",
        "pcdl_t1.0",
        "pcdlm_t0.5",
        "pcdlm_t1.0",
        "pcdrev_t0.5",
        "g_pcd_t0.5",
        "g_nwkd_l1.0",
        "g_pcdrev_t0.5",
        "pcgrad1s",
        "wkd_l0.9",
        "dot_d0.075",
        "dtokd",
    ]
    mean_table(acc, "teacher-corruption", rho, tc_arms)
    diff_table(
        acc,
        "Matched weights: PCD against the fixed weight nearest its average λ_eff",
        [
            ("teacher-corruption", c, a, b)
            for c in rho
            for a, b in (
                ("pcd_t0.5", "nwkd_l0.25"),
                ("pcd_t1.0", "nwkd_l0.75"),
                ("pcd_t1.0", "nwkd_l0.5"),
            )
        ],
    )
    diff_table(
        acc,
        "PCGrad, PCD and CE under teacher corruption",
        [
            ("teacher-corruption", c, a, b)
            for c in ("rho0.0", "rho0.5", "rho0.7")
            for a, b in (
                ("pcgrad1s", "pcd_t1.0"),
                ("pcgrad1s", "ce"),
                ("pcd_t0.5", "ce"),
                ("dot_d0.075", "pcd_t1.0"),
            )
        ],
    )
    diff_table(
        acc,
        "PCDLM against fixed weights of similar size",
        [
            ("teacher-corruption", c, "pcdlm_t1.0", b)
            for c in ("rho0.0", "rho0.5", "rho0.7")
            for b in ("nwkd_l0.75", "nwkd_l1.0", "pcd_t1.0")
        ]
        + [("core-p0", "clean", "pcdlm_t1.0", b) for b in ("pcd_t1.0", "nwkd_l1.0", "dot_d0.075")],
    )
    ln = ["sym0.0", "sym0.2", "sym0.4", "sym0.6", "sym0.8", "asym0.2", "asym0.4"]
    mean_table(
        acc,
        "label-noise",
        ln,
        [
            "ce",
            "nwkd_l1.0",
            "nwkd_l4.0",
            "wkd_l0.9",
            "pcd_t1.0",
            "pcdrev_t0.5",
            "pcdrev_t1.0",
            "pcdlm_t1.0",
            "pcgrad1s",
            "dot_d0.075",
        ],
    )
    diff_table(
        acc,
        "Label noise: which signal PCD puts first",
        [
            ("label-noise", c, a, b)
            for c in ("sym0.4", "sym0.8", "asym0.4")
            for a, b in (
                ("pcdrev_t0.5", "pcd_t1.0"),
                ("pcdrev_t0.5", "nwkd_l4.0"),
                ("dot_d0.075", "pcdrev_t0.5"),
            )
        ],
    )
    for fam in ("core-p0", "core-p1", "lit-p1"):
        arms = sorted({a for (f, _, a) in acc if f == fam})
        mean_table(acc, fam, ["clean"], arms)
    plan = "research/experiment-plans/pcd-kd-conflict.md"
    print(f"\n### Pre-registered hypotheses ({plan}), δ = 0.5 pp\n")
    print("| Hypothesis | Contrast | Mean (pp) | 90% interval | Verdict |")
    print("|---|---|---|---|---|")
    tost_row(
        acc, "H3 (clean, equivalence)", "teacher-corruption", "rho0.0", "pcd_t1.0", "nwkd_l0.5"
    )
    tost_row(
        acc, "H3 (clean, equivalence)", "teacher-corruption", "rho0.0", "pcd_t1.0", "nwkd_l0.75"
    )
    tost_row(
        acc,
        "H4 (gated, ρ0.3, equivalence)",
        "teacher-corruption",
        "rho0.3",
        "g_pcd_t0.5",
        "g_nwkd_l1.0",
    )
    for nw in ("nwkd_l0.5", "nwkd_l0.25"):
        tost_row(
            acc,
            "H2 (ρ0.3, PCD not higher)",
            "teacher-corruption",
            "rho0.3",
            "pcd_t1.0",
            nw,
            one_sided=True,
        )
    diff_table(
        acc,
        "Per-sample gate under heavy corruption (decision 0006)",
        [
            ("teacher-corruption", c, a, b)
            for c in ("rho0.3", "rho0.5", "rho0.7")
            for a, b in (
                ("g_pcd_t0.5", "g_nwkd_l1.0"),
                ("g_pcd_t0.5", "pcd_t0.5"),
                ("g_nwkd_l1.0", "nwkd_l1.0"),
                ("g_pcd_t0.5", "nwkd_l0.25"),
                ("g_pcd_t0.5", "ce"),
            )
        ],
    )
    sw = sorted({c for (f, c, _) in acc if f == "teacher-switch"})
    sw_arms = ["nwkd_l0.25", "nwkd_l0.75", "pcd_t0.5", "pcd_t1.0", "pcdrev_t0.5", "pcdrev_t1.0"]
    if sw:
        mean_table(acc, "teacher-switch", sw, sw_arms)
        diff_table(
            acc,
            "Teacher switch: PCD against the matched fixed weight, and against CE "
            "(CE ignores the teacher, so its teacher-corruption runs are the reference)",
            [
                ("teacher-switch", c, a, b)
                for c in sw
                for a, b in (
                    ("pcd_t0.5", "nwkd_l0.25"),
                    ("pcd_t1.0", "nwkd_l0.75"),
                    ("pcdrev_t0.5", "nwkd_l0.75"),
                    ("pcdrev_t1.0", "nwkd_l0.75"),
                )
            ]
            + [
                ("teacher-switch", c, a, "ce", "teacher-corruption", c.split("_")[0])
                for c in sw
                for a in sw_arms
            ],
        )
    diagnostics_table()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--extract":
        extract(sys.argv[2])
    else:
        main()
