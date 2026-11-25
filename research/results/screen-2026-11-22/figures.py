"""Draw the README figures from runs.csv and diagnostics.csv in this folder.

    uv run --with matplotlib python research/results/screen-2026-11-22/figures.py

matplotlib is not a project dependency; `--with` installs it for this one command only.
"""

import csv
import math
import pathlib
import statistics as st
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).parent
OUT = HERE / "figures"
# validated categorical slots 1-4 (dataviz reference palette, light surface); CE is a gray reference
BLUE, ORANGE, AQUA, VIOLET, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#8a8984"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def runs():
    acc = defaultdict(dict)
    for r in csv.DictReader(open(HERE / "runs.csv", encoding="utf-8")):
        acc[(r["family"], r["condition"], r["arm"])][int(r["seed"])] = 100 * float(r["test"])
    return acc


def mse(acc, fam, cond, arm):
    v = list(acc[(fam, cond, arm)].values())
    return st.mean(v), (st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0)


def diag():
    g = defaultdict(list)
    for r in csv.DictReader(open(HERE / "diagnostics.csv", encoding="utf-8")):
        g[(r["family"], r["condition"], r["arm"])].append(r)
    return lambda fam, cond, arm, k: st.mean(float(r[k]) for r in g[(fam, cond, arm)])


def matched_weights(acc, d):
    """Fixed-weight curve with PCD placed at its average weight, one panel per corruption level."""
    fam, lams = "teacher-corruption", (0.25, 0.5, 0.75, 1.0)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for ax, (cond, title) in zip(
        axes,
        [("rho0.0", "clean teacher"), ("rho0.5", "50% corrupted"), ("rho0.7", "70% corrupted")],
        strict=True,
    ):
        m = [mse(acc, fam, cond, f"nwkd_l{x}") for x in lams]
        ax.errorbar(
            lams,
            [a for a, _ in m],
            yerr=[e for _, e in m],
            color=BLUE,
            lw=2,
            marker="o",
            ms=6,
            capsize=3,
            label="fixed weight (NWKD)",
        )
        ce, _ = mse(acc, fam, cond, "ce")
        ax.axhline(ce, color=GRAY, lw=1.5, ls="--")
        ax.text(
            1.02,
            ce,
            "CE (no teacher)",
            color=INK2,
            va="center",
            fontsize=8,
            transform=ax.get_yaxis_transform(),
        )
        for arm, mk in (("pcd_t0.5", "D"), ("pcd_t1.0", "s")):
            y, e = mse(acc, fam, cond, arm)
            x = d(fam, cond, arm, "lambda_mean")
            ax.errorbar(
                [x],
                [y],
                yerr=[e],
                color=ORANGE,
                marker=mk,
                ms=8,
                capsize=3,
                lw=0,
                elinewidth=1.5,
                mec=SURFACE,
                mew=1.5,
                label="PCD at its average weight" if arm == "pcd_t1.0" else None,
            )
            off = (8, 6) if arm == "pcd_t0.5" else (8, -12)
            ax.annotate(
                f"PCD τ{arm[-3:]}",
                (x, y),
                xytext=off,
                textcoords="offset points",
                fontsize=8,
                color=INK,
            )
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("KD weight λ")
    axes[0].set_ylabel("test top-1 (%)")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(
        "PCD lands on the fixed-weight curve: same weight, same accuracy (5 seeds, ±1 SE)",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT / "matched-weights.png", dpi=160)


def weight_over_training(d):
    """PCD tau=1.0: the weight moves with the corruption level, but hardly within a run."""
    fam = "teacher-corruption"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    thirds = ["first third", "middle third", "last third"]
    for (cond, label), color, mk in zip(
        [("rho0.0", "clean"), ("rho0.5", "ρ 0.5"), ("rho0.7", "ρ 0.7")],
        (BLUE, ORANGE, AQUA),
        "o^s",
        strict=True,
    ):
        lam = [d(fam, cond, "pcd_t1.0", f"lambda_third{i}") for i in (1, 2, 3)]
        cos = [d(fam, cond, "pcd_t1.0", f"cos_third{i}") for i in (1, 2, 3)]
        a1.plot(thirds, cos, color=color, lw=2, marker=mk, ms=7, label=label)
        a2.plot(thirds, lam, color=color, lw=2, marker=mk, ms=7, label=label)
        a1.annotate(
            label, (2, cos[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8, va="center"
        )
        a2.annotate(
            label, (2, lam[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8, va="center"
        )
    a1.set_title("cos(g_CE, g_KD): set by the corruption level", fontsize=10, loc="left")
    a2.set_title("PCD τ1.0 weight λ = 1 − cos: follows it", fontsize=10, loc="left")
    a1.set_ylabel("mean per third of training")
    a1.set_ylim(0, 0.8)
    a2.set_ylim(0, 1.0)
    a2.legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "The weight changes between conditions, not within a run (5 seeds)",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT / "weight-over-training.png", dpi=160)


def robustness(acc):
    """Accuracy as the teacher is corrupted: conflict-aware methods win clean and collapse later."""
    fam = "teacher-corruption"
    rho = [0.0, 0.1, 0.3, 0.5, 0.7]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    series = [
        ("dot_d0.075", "DOT", VIOLET, "v"),
        ("pcgrad1s", "PCGrad", AQUA, "^"),
        ("nwkd_l0.75", "fixed weight 0.75", BLUE, "o"),
        ("pcd_t1.0", "PCD τ1.0", ORANGE, "s"),
        ("ce", "CE (no teacher)", GRAY, ""),
    ]
    for arm, label, color, mk in series:
        y = [mse(acc, fam, f"rho{r}", arm)[0] for r in rho]
        ax.plot(
            rho,
            y,
            color=color,
            lw=2,
            marker=mk or None,
            ms=7,
            ls="--" if arm == "ce" else "-",
            label=label,
        )
        if arm in ("dot_d0.075", "ce"):  # the three middle lines end within 0.6 pp: legend only
            ax.annotate(
                label,
                (0.7, y[-1]),
                xytext=(8, 0),
                textcoords="offset points",
                fontsize=8,
                va="center",
            )
    ax.set_xlabel("fraction of training images with a corrupted teacher output (ρ)")
    ax.set_ylabel("test top-1 (%)")
    ax.set_xlim(-0.03, 0.9)
    ax.set_xticks(rho)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(
        "Under a bad teacher, only a small KD weight is safe (3–5 seeds per point)",
        fontsize=11,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(OUT / "robustness.png", dpi=160)


def paired_diff(acc, fam, cond, a, b):
    A, B = acc[(fam, cond, a)], acc[(fam, cond, b)]
    d = [A[k] - B[k] for k in sorted(set(A) & set(B))]
    t975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}[len(d) - 1]
    return st.mean(d), t975 * st.stdev(d) / math.sqrt(len(d))


def teacher_switch(acc, d):
    """Teacher clean for the first half, corrupted after: PCD's weight moves, and the wrong way."""
    fam = "teacher-switch"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.1, 1]})
    x = [0, 1, 2]
    a1.axvspan(1, 2.6, color=GRID, alpha=0.6, lw=0)
    a1.text(1.05, 0.98, "teacher corrupted from epoch 30", fontsize=8, color=INK2, va="top")
    for arm, label, color, ls, mk in (
        ("pcd_t1.0", "PCD τ1.0, CE first: KD weight", ORANGE, "-", "s"),
        ("pcd_t0.5", "PCD τ0.5, CE first: KD weight", ORANGE, ":", "D"),
        ("pcdrev_t1.0", "PCD τ1.0, KD first: CE weight", AQUA, "-", "o"),
        ("pcdrev_t0.5", "PCD τ0.5, KD first: CE weight", AQUA, ":", "^"),
    ):
        y = [d(fam, "rho0.7_from0.5", arm, f"lambda_third{i}") for i in (1, 2, 3)]
        a1.plot(x, y, color=color, ls=ls, lw=2, marker=mk, ms=7, mec=SURFACE, label=label)
    for lam in (0.25, 0.75):
        a1.axhline(lam, color=BLUE, lw=1.2, ls="--")
        a1.text(2.55, lam + 0.015, f"fixed {lam}", color=INK2, fontsize=8, va="bottom", ha="right")
    a1.set_xticks(x, ["first third", "middle third", "last third"])
    a1.set_xlim(-0.15, 2.6)
    a1.set_ylim(0, 1.32)
    a1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    a1.set_ylabel("weight on the second loss (mean per third)")
    a1.set_title("Weight over training, ρ0.7 after the switch", fontsize=10, loc="left")
    a1.legend(loc="upper left", fontsize=7.5, ncol=2)
    pairs = [
        ("pcd_t0.5", "nwkd_l0.25", "PCD τ0.5 (CE first)\n− fixed 0.25"),
        ("pcd_t1.0", "nwkd_l0.75", "PCD τ1.0 (CE first)\n− fixed 0.75"),
        ("pcdrev_t1.0", "nwkd_l0.75", "PCD τ1.0 (KD first)\n− fixed 0.75"),
    ]
    for j, (cond, mk, off, lab) in enumerate(
        (("rho0.5_from0.5", "o", -0.12, "ρ0.5"), ("rho0.7_from0.5", "s", 0.12, "ρ0.7"))
    ):
        m = [paired_diff(acc, fam, cond, a, b) for a, b, _ in pairs]
        a2.errorbar(
            [i + off for i in range(len(pairs))],
            [v for v, _ in m],
            yerr=[e for _, e in m],
            color=INK2 if j == 0 else INK,
            marker=mk,
            ms=7,
            lw=0,
            elinewidth=1.5,
            capsize=3,
            label=f"{lab} after the switch",
        )
    a2.axhline(0, color=GRAY, lw=1.2)
    a2.set_xticks(range(len(pairs)), [p[2] for p in pairs], fontsize=8)
    a2.set_ylabel("difference in test top-1 (pp)")
    a2.set_title("Against the matched fixed weight (3 seeds, 95% CI)", fontsize=10, loc="left")
    a2.legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "When the teacher turns bad, CE-first PCD gives it more weight, and does no better",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT / "teacher-switch.png", dpi=160)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    acc, d = runs(), diag()
    matched_weights(acc, d)
    weight_over_training(d)
    robustness(acc)
    teacher_switch(acc, d)
    print("figures ->", OUT)
