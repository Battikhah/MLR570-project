"""The PCD-in-KD experiment suite: every family, condition, arm and seed as flat run specs.

Each family probes one regime where PCD's mechanism predicts a *difference* from the baselines
(see ``FAMILIES[...]["hypothesis"]`` and docs/experiment-suite.md). No family assumes PCD wins:
each carries the matched controls needed to attribute a difference to PCD's projection
(EMA-normalized weighted sum with the same rescale), to symmetric balancing (MGDA / DTO-KD), or
to momentum (DOT).

Profiles scale the schedule, not the design: ``full`` (paper protocol, 240 epochs), ``screen``
(60 epochs, <= 2 seeds; a first GPU pass to find where to spend the full budget), ``smoke``
(synthetic data, a few steps on CPU; used by the tests).
"""

from __future__ import annotations

import copy
import re

PROFILES = {
    "full": {"epochs": 240, "max_seeds": 99, "synthetic": False, "max_steps": None},
    "screen": {
        "epochs": 60,
        "max_seeds": 2,
        "synthetic": False,
        "max_steps": None,
        "seed_overrides": {
            "teacher-corruption": 5, "label-noise": 3, "core-p0": 3, "core-p1": 3, "lit-p1": 2,
        },
    },
    "smoke": {"epochs": 1, "max_seeds": 1, "synthetic": True, "max_steps": 2, "batch_size": 8},
}
TEACHER_EPOCHS = {"full": 240, "screen": 240, "smoke": 1}

# mdistiller CIFAR-100 protocol (configs/cifar100/*): SGD 0.9, wd 5e-4, lr 0.05, batch 64,
# x0.1 at 150/180/210 of 240 epochs, T = 4.
BASE = {
    "kind": "student",
    "dataset": "cifar100",
    "arch": "resnet8",
    "teachers": ["resnet32"],
    "primary": "ce",
    "objectives": ["ce", "kd"],
    "method": "ce",
    "params": {},
    "loss_scale": {},
    "T": 4.0,
    "gate": "none",
    "gate_q": 0.8,
    "teacher_corruption": 0.0,
    "corrupt_teacher": 0,
    "label_noise": 0.0,
    "noise_kind": "symmetric",
    "train_fraction": 1.0,
    "imbalance": 1,
    "split_seed": 0,
    "val_size": 5000,
    "epochs": 240,
    "batch_size": 64,
    "lr": 0.05,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "schedule": "step",
    "log_every": 50,
    "eval_every": 10,
}
P0 = {"arch": "resnet8", "teachers": ["resnet32"]}  # the submitted proposal's pair (cheap)
P1 = {"arch": "resnet8x4", "teachers": ["resnet32x4"]}  # mdistiller / DTO-KD reference pair
P1_MD = {"arch": "md_resnet8x4", "teachers": ["md_resnet32x4"]}  # the same pair, CRD architecture

# CRD's CIFAR-100 teachers as packaged by mdistiller (paths relative to --data-root).
# published_top1: the CRD / mdistiller tables; run_teacher measures and records our own test top-1.
CRD_URL = (
    "https://github.com/megvii-research/mdistiller/releases/download/checkpoints/cifar_teachers.tar"
)
CRD_SHA256 = "e7c8b2f7d449af08e1222663e845ad23ca4a2044735b980740d86a96088de52c"
CRD_TEACHERS = {
    arch: {
        "source": "CRD/mdistiller",
        "url": CRD_URL,
        "dir": "mdistiller",
        "file": f"mdistiller/cifar_teachers/{name}_vanilla/ckpt_epoch_240.pth",
        "key": "model",
        "archive_sha256": CRD_SHA256,
        "published_top1": top1,
    }
    for arch, name, top1 in (
        ("md_resnet32x4", "resnet32x4", 79.42),
        ("md_resnet56", "resnet56", 72.34),
        ("md_resnet110", "resnet110", 74.31),
    )
}


def arm(name, method, objectives=("ce", "kd"), primary="ce", group=None, **params):
    """``group`` = method family used by the analysis to pick one config per family on val."""
    return {
        "arm": name,
        "group": group or name,
        "method": method,
        "objectives": list(objectives),
        "primary": primary,
        "params": params,
    }


CE = arm("ce", "ce", ("ce",))


def wkd(lam):
    return arm(f"wkd_l{lam}", "wkd", group="wkd", lam=lam)


def nwkd(lam, **kw):
    name = f"nwkd_l{lam}" + "".join(f"_{k}{v}" for k, v in kw.items())
    return arm(name, "nwkd", group="nwkd", lam=lam, **kw)


def pcd(tau, primary="ce", **kw):
    obj = ("ce", "kd") if primary == "ce" else ("kd", "ce")
    tag = "pcd" if primary == "ce" else "pcdrev"
    name = f"{tag}_t{tau}" + "".join(f"_{k}{v}" for k, v in kw.items())
    return arm(name, "pcd", obj, primary, group=tag, tau=tau, **kw)


def pcdl(tau, **kw):
    """Per-layer PCD (decision 0004)."""
    return arm(f"pcdl_t{tau}", "pcdl", group="pcdl", tau=tau, **kw)


def pcdlm(tau, **kw):
    """Per-layer PCD on separate CE / KD momentum buffers (decision 0004)."""
    return arm(f"pcdlm_t{tau}", "pcdlm", group="pcdlm", tau=tau, **kw)


PCGRAD = arm("pcgrad1s", "pcgrad", lam=1.0)
MGDA = arm("mgda", "mgda")
DTO = arm("dtokd", "dto")
DOT_ = arm("dot_d0.075", "dot", delta=0.075, lam=0.9)

# Arm sets. "full" = every two-objective rule; "regime" = the set run under every stressor.
ARMS_FULL = [
    CE,
    wkd(0.5),
    wkd(0.9),
    *(nwkd(x) for x in (0.25, 0.5, 1.0, 2.0)),
    *(pcd(t) for t in (0.1, 0.5, 1.0, 2.0)),
    pcd(0.1, "kd"),
    pcd(0.5, "kd"),
    PCGRAD,
    MGDA,
    DTO,
    DOT_,
]
ARMS_REGIME = [
    CE,
    wkd(0.9),
    nwkd(0.5),
    nwkd(1.0),
    *(pcd(t) for t in (0.1, 0.5, 1.0)),
    pcd(0.5, "kd"),
    PCGRAD,
    DTO,
    DOT_,
]
ARMS_LITE = [CE, wkd(0.9), nwkd(1.0), pcd(0.5), pcd(1.0), DTO, DOT_]


def gated(a, mode="correct_and_conf"):
    return {**a, "arm": f"g_{a['arm']}", "group": f"g_{a['group']}", "gate": mode}


def _sweep(values, key, fmt="{}"):
    return [({key: v}, fmt.format(v)) for v in values]


def _family(name, hypothesis, tier, seeds, conditions, arms, pair=P0, **extra):
    return {
        "name": name,
        "hypothesis": hypothesis,
        "tier": tier,
        "seeds": seeds,
        "conditions": conditions,
        "arms": arms,
        "pair": pair,
        "extra": extra,
    }


K3_AT = [
    *(
        arm(f"sum_kd{a}_at{b}", "sum", ("ce", "kd", "at"), group="sum", lams=[a, b])
        for a in (1.0, 9.0)
        for b in (100.0, 1000.0)
    ),
    *(
        arm(f"nws_kd{a}_at{b}", "nws", ("ce", "kd", "at"), group="nws", lams=[a, b])
        for a in (0.5, 1.0)
        for b in (0.5, 1.0)
    ),
    *(arm(f"pcd_t{t}", "pcd", ("ce", "kd", "at"), group="pcd", tau=t) for t in (0.1, 0.5, 1.0)),
    arm("mgda", "mgda", ("ce", "kd", "at")),
    arm("dtokd", "dto", ("ce", "kd", "at")),
    arm("pcgrad_sym", "pcgrad_sym", ("ce", "kd", "at"), lams=[9.0, 1000.0]),
    CE,
    wkd(0.9),
    arm("ce_at1000", "sum", ("ce", "at"), lams=[1000.0]),
]
K3_DKD = [
    *(
        arm(f"dkd_a1_b{b}", "sum", ("ce", "tckd", "nckd"), group="dkd", lams=[1.0, b])
        for b in (2.0, 8.0)
    ),
    *(
        arm(f"nws_{a}_{b}", "nws", ("ce", "tckd", "nckd"), group="nws", lams=[a, b])
        for a in (0.5, 1.0)
        for b in (0.5, 1.0)
    ),
    *(arm(f"pcd_t{t}", "pcd", ("ce", "tckd", "nckd"), group="pcd", tau=t) for t in (0.1, 0.5, 1.0)),
    arm("mgda", "mgda", ("ce", "tckd", "nckd")),
    arm("dtokd", "dto", ("ce", "tckd", "nckd")),
    CE,
    wkd(0.9),
]
MT = ("ce", "kd", "kd@1", "kd@2")
MULTI_TEACHER = [
    arm("kdavg_wkd0.9", "wkd", ("ce", "kdavg"), lam=0.9),
    arm("kdavg_pcd_t0.5", "pcd", ("ce", "kdavg"), tau=0.5),
    arm("sum_equal3", "sum", MT, lams=[3.0]),
    *(arm(f"nws_l{x}", "nws", MT, group="nws", lams=[x]) for x in (0.33, 1.0)),
    *(arm(f"pcd_t{t}", "pcd", MT, group="pcd", tau=t) for t in (0.1, 0.5, 1.0)),
    arm("mgda_aekd", "mgda", MT),
    arm("pcgrad_sym", "pcgrad_sym", MT, lams=[3.0]),
    CE,
]
TAUS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)
SENS = (
    [wkd(x) for x in (0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99)]
    + [nwkd(x) for x in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)]
    + [pcd(t) for t in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)]
    + [pcd(t, "kd") for t in (0.05, 0.25, 1.0, 4.0)]
)

TC_LAYERWISE_ARMS = [pcdl(0.5), pcdl(1.0), pcdlm(0.5), pcdlm(1.0)]
# Decision 0005: fixed NWKD weights at the lambda_eff PCD actually used (0.01-0.75), so PCD can be
# compared with a fixed weight it matches; and PCDLM on the clean benchmarks.
MATCHED_NWKD_ARMS = [nwkd(0.25), nwkd(0.75)]
CLEAN_PCDLM_ARMS = [pcdlm(0.5), pcdlm(1.0)]

FAMILIES = [
    _family(
        "lit-p1",
        "Literature-matched anchor: CRD teacher, mdistiller protocol, full training set; checks "
        "our KD/DOT against the published 73.33 / 75.12 and places PCD on the survey's ranking "
        "pair.",
        0,
        1,
        [({}, "clean")],
        [CE, wkd(0.9), nwkd(1.0), pcd(0.5), pcd(1.0), PCGRAD, MGDA, DTO, DOT_, pcd(2.0)]
        + [pcdlm(1.0)],  # decision 0005, appended last
        P1_MD,
        val_size=0,  # no val set, so every arm runs at its paper/default setting (no grid)
        epochs=240,  # the protocol's length in every profile except smoke
        crd_teacher=True,
    ),
    _family(
        "pilot",
        "Go/no-go (plan §3): measured cost per arm, the CE-KD cosine distribution, PCD's "
        "active rate, and whether the PCD / NWKD / DTO-KD directions differ on >= 10% of steps.",
        0,
        1,
        [({}, "clean")],
        [CE, wkd(0.9), DTO, pcd(0.5)],
        P1,
    ),
    _family(
        "mve-cifar10",
        "Decision 0001 MVE: does 30% teacher corruption shift cos(g_CE, g_KD) "
        "negative and activate PCD's constraint (H0)?",
        0,
        3,
        _sweep((0.0, 0.3), "teacher_corruption", "rho{}"),
        [CE, wkd(0.9), nwkd(1.0), pcd(0.1), pcd(0.5), pcd(0.5, "kd")],
        P0,
        dataset="cifar10",
    ),
    _family(
        "core-p1",
        "I1*: priority (PCD) vs symmetry (DTO-KD, MGDA) vs momentum split (DOT) vs a "
        "budget-matched normalized sum, on the reference ResNet32x4 -> ResNet8x4 benchmark.",
        1,
        5,
        [({}, "clean")],
        [*ARMS_FULL, *CLEAN_PCDLM_ARMS],  # appended last (decision 0005)
        P1,
    ),
    _family(
        "core-p0",
        "I1* replicated on the cheap ResNet32 -> ResNet8 pair (does the ranking "
        "depend on the pair?).",
        1,
        5,
        [({}, "clean")],
        [*ARMS_FULL, *CLEAN_PCDLM_ARMS],  # appended last (decision 0005)
        P0,
    ),
    _family(
        "loss-scale",
        "PCD's documented strength is objective SCALE, not conflict (PCD §5.1.1). "
        "Multiplying L_KD by m leaves PCD/NWKD/DTO-KD directions invariant (EMA or log "
        "normalization) but shifts raw WKD/PCGrad. Prediction: flat PCD curve, WKD collapses at "
        "large or small m.",
        1,
        3,
        [({"loss_scale": {"kd": m}}, f"m{m}") for m in (0.01, 0.1, 1.0, 10.0, 100.0)],
        [wkd(0.9), nwkd(1.0), pcd(0.5), pcd(1.0), DTO, MGDA, PCGRAD],
        P0,
    ),
    _family(
        "temperature",
        "T changes both the KD gradient scale (T^2) and its direction. Normalized "
        "rules should be less T-sensitive than raw WKD.",
        2,
        3,
        _sweep((1.0, 2.0, 4.0, 8.0, 16.0), "T", "T{}"),
        [wkd(0.9), nwkd(1.0), pcd(0.5), pcd(1.0), DTO],
        P0,
    ),
    _family(
        "teacher-corruption",
        "Unreliable teacher (submitted proposal). PCD's KD weight "
        "max(0, tau - c) GROWS with conflict (decision 0001), so CE-primary PCD is predicted to "
        "follow a corrupted teacher more than NWKD; KD-primary PCD and the gate are the "
        "contrasts.",
        1,
        3,
        _sweep((0.0, 0.1, 0.3, 0.5, 0.7), "teacher_corruption", "rho{}"),
        [
            *ARMS_REGIME,
            gated(nwkd(1.0)),
            gated(pcd(0.5)),
            gated(pcd(0.5, "kd")),
            *TC_LAYERWISE_ARMS,  # appended last: existing spec ids and content are unchanged
            *MATCHED_NWKD_ARMS,  # decision 0005, appended after those
        ],
        P0,
    ),
    _family(
        "label-noise",
        "Noisy labels, clean teacher: the label (CE) gradient is the unreliable "
        "one. KD-primary PCD (pcdrev) keeps the teacher as anchor and admits CE only as a "
        "tau-constraint. HYPOTHESIS: this is where priority helps most.",
        1,
        3,
        [({"label_noise": e}, f"sym{e}") for e in (0.0, 0.2, 0.4, 0.6, 0.8)]
        + [({"label_noise": e, "noise_kind": "asymmetric"}, f"asym{e}") for e in (0.2, 0.4)],
        [*ARMS_REGIME, pcd(0.1, "kd"), pcd(1.0, "kd"), nwkd(4.0), pcdlm(1.0)],
        P0,
    ),
    _family(
        "label-noise-noisy-teacher",
        "As label-noise, but the teacher is trained on the same "
        "noisy labels (realistic): is KD-priority still useful when the teacher memorised noise?",
        3,
        3,
        [({"label_noise": e, "teacher_noise": True}, f"sym{e}") for e in (0.2, 0.4, 0.6)],
        [*ARMS_LITE, pcd(0.5, "kd"), pcd(1.0, "kd")],
        P0,
    ),
    _family(
        "low-data",
        "Few labels per class: the strongest measured CE-KD conflict is few-shot "
        "(DHO, cos -0.39..-0.76). Iteration-matched schedules (epochs / fraction).",
        1,
        3,
        _sweep((0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0), "train_fraction", "f{}"),
        ARMS_REGIME,
        P0,
    ),
    _family(
        "long-tail",
        "CIFAR-100-LT: CE is dominated by head classes while the teacher carries "
        "tail-class structure; per-batch conflict varies with class mix. Report many/medium/few.",
        2,
        3,
        _sweep((1, 10, 50, 100, 200), "imbalance", "if{}"),
        ARMS_REGIME,
        P0,
    ),
    _family(
        "capacity-gap",
        "Large teacher-student gaps make the KD target hard to match (more "
        "conflict late in training?).",
        2,
        3,
        [
            ({"teachers": [t]}, t)
            for t in ("resnet14", "resnet20", "resnet32", "resnet56", "resnet110", "resnet32x4")
        ],
        ARMS_LITE,
        P0,
    ),
    _family(
        "multi-loss-at",
        "K=3 (CE + KD + attention transfer): PCD needs ONE tau while weighted "
        "sums need a 2-D grid. Prediction: PCD ~ best grid point at a fraction of tuning cost.",
        2,
        3,
        [({}, "clean"), ({"teacher_corruption": 0.3}, "rho0.3")],
        K3_AT,
        P0,
    ),
    _family(
        "multi-loss-dkd",
        "K=3 (CE + TCKD + NCKD): PCD vs DKD's hand-set alpha/beta and "
        "symmetric MGDA/DTO-KD on the DKD reference pair.",
        2,
        3,
        [({}, "clean")],
        K3_DKD,
        P1,
    ),
    _family(
        "multi-teacher",
        "K=4 (CE + 3 teachers): anchored PCD vs symmetric MGDA (AE-KD-style), "
        "teacher averaging and equal sums; with one teacher corrupted, does PCD follow the bad "
        "teacher (constraint grows with conflict) or is it outvoted?",
        2,
        3,
        [({}, "clean"), ({"teacher_corruption": 0.5}, "t0rho0.5")],
        MULTI_TEACHER,
        {"arch": "resnet8", "teachers": ["resnet32", "resnet56", "resnet20"]},
    ),
    _family(
        "tau-sweep",
        "PCD's only knob: tau from 0 (CE-only when aligned, H0') to 4 (always "
        "active), clean and corrupted, both priority orders.",
        2,
        2,
        _sweep((0.0, 0.3), "teacher_corruption", "rho{}"),
        [*(pcd(t) for t in TAUS), *(pcd(t, "kd") for t in TAUS), CE, wkd(0.9), nwkd(1.0)],
        P0,
    ),
    _family(
        "sensitivity",
        "Tuning robustness: equal 8-point grids for WKD lambda, NWKD lambda and "
        "PCD tau. Report best, median and worst; a flat PCD curve is a usability win even with "
        "equal best accuracy.",
        2,
        2,
        [({}, "clean"), ({"teacher_corruption": 0.3}, "rho0.3"), ({"label_noise": 0.4}, "sym0.4")],
        SENS,
        P0,
    ),
    _family(
        "pcd-ablations",
        "Which part of PCD matters: EMA beta, final rescale, optimizer momentum, "
        "batch size (gradient noise changes c).",
        3,
        2,
        [({}, "clean")],
        [
            *(pcd(0.5, beta=b) for b in (0.0, 0.9, 0.999)),
            pcd(0.5, rescale=False),
            nwkd(1.0, rescale=False),
            pcd(0.5),
            nwkd(1.0),
            *(
                {**a, "arm": a["arm"] + "_mom0", "momentum": 0.0}
                for a in (pcd(0.5), nwkd(1.0), wkd(0.9))
            ),
            *(
                {**a, "arm": f"{a['arm']}_bs{bs}", "batch_size": bs}
                for bs in (32, 128, 256)
                for a in (pcd(0.5), nwkd(1.0))
            ),
        ],
        P0,
    ),
]

FAMILY_NAMES = [f["name"] for f in FAMILIES]


def teacher_key(dataset, arch, imbalance=1, label_noise=0.0, noise_kind="symmetric", crd=False):
    key = f"{dataset}-{arch}" + ("-crd" if crd else "")
    if imbalance > 1:
        key += f"-if{imbalance}"
    if label_noise > 0:
        key += f"-{noise_kind[:3]}{label_noise}"
    return key


def _slug(s):
    return re.sub(r"[^A-Za-z0-9_.=/-]+", "-", s)


def build_specs(profile="full", families=None, tiers=None):
    """Return (student specs, teacher specs keyed by teacher key) for the selected families."""
    prof = PROFILES[profile]
    students, teachers = [], {}
    for fam in FAMILIES:
        if (
            families
            and fam["name"] not in families
            or tiers is not None
            and fam["tier"] not in tiers
        ):
            continue
        for cond, cname in fam["conditions"]:
            for a in fam["arms"]:
                n_seeds = prof.get("seed_overrides", {}).get(
                    fam["name"], min(fam["seeds"], prof["max_seeds"])
                )
                for seed in range(n_seeds):
                    s = (
                        copy.deepcopy(BASE)
                        | copy.deepcopy(fam["pair"])
                        | copy.deepcopy(fam["extra"])
                    )
                    s |= copy.deepcopy(cond)
                    s |= {k: copy.deepcopy(v) for k, v in a.items()}
                    s.update(
                        family=fam["name"],
                        tier=fam["tier"],
                        condition=cname,
                        seed=seed,
                        profile=profile,
                    )
                    if profile == "smoke" or "epochs" not in fam["extra"]:
                        s["epochs"] = prof["epochs"]
                    s["batch_size"] = prof.get("batch_size", s["batch_size"])
                    if s["train_fraction"] < 1:  # iteration-matched
                        s["epochs"] = max(1, round(prof["epochs"] / s["train_fraction"]))
                        s["eval_every"] = max(10, s["epochs"] // 24)
                    noisy_t = s.pop("teacher_noise", False)
                    crd = s.pop("crd_teacher", False)
                    s["teacher_keys"] = []
                    for t in s.pop("teachers"):
                        key = teacher_key(
                            s["dataset"],
                            t,
                            s["imbalance"],
                            s["label_noise"] if noisy_t else 0.0,
                            s["noise_kind"],
                            crd,
                        )
                        s["teacher_keys"].append(key)
                        teachers.setdefault(key, _teacher_spec(key, s, t, noisy_t, profile, crd))
                    s["id"] = _slug(f"{fam['name']}/{cname}/{a['arm']}/s{seed}")
                    students.append(s)
    return students, teachers


# Queue order for a time-limited GPU window: the family most likely to show where PCD helps or
# hurts comes first. INTERPRETATION, from the hypotheses table in docs/experiment-suite.md:
# pilot and MVE gate everything; core-p0 is the cheap clean baseline; label noise is the most
# promising regime and teacher corruption the predicted failure mode; loss scale tests the
# "scale, not conflict" explanation; the reference pair (core-p1) is costlier per run. Families
# that need extra teachers (long-tail, capacity-gap, noisy teachers) or many grid points come last.
PRIORITY = (
    "lit-p1",  # first: anchors our numbers to the published ones
    "pilot",
    "mve-cifar10",
    "core-p0",
    "label-noise",
    "teacher-corruption",
    "loss-scale",
    "core-p1",
    "low-data",
    "tau-sweep",
    "multi-loss-dkd",
    "pcd-ablations",
    "multi-teacher",
    "capacity-gap",
    "temperature",
    "long-tail",
    "multi-loss-at",
    "label-noise-noisy-teacher",
    "sensitivity",
)


# Each block is (family, seeds, conditions, arms); None selects every condition/arm.
# Blocks spend this weekend on paired evidence for the paper's central claims, then defer the rest.
# Decision 0003: high-rho teacher-corruption seeds for the arms the robustness claim compares.
TC_CORE_ARMS = (
    "ce", "pcd_t0.5", "pcd_t1.0", "nwkd_l0.5", "nwkd_l1.0", "wkd_l0.9", "dot_d0.075", "pcdrev_t0.5",
)
# Arms left out of blocks that select every arm (decision 0003); they still run in the tail.
DEFERRED_ARMS = ("dtokd", *(a["arm"] for a in TC_LAYERWISE_ARMS))  # new arms: explicit blocks
TC_CONDS = ("rho0.0", "rho0.5", "rho0.7")
LN_CORE_ARMS = (
    "ce", "pcd_t1.0", "pcdrev_t0.5", "pcdrev_t1.0", "nwkd_l1.0", "nwkd_l4.0", "wkd_l0.9",
    "dot_d0.075",
)
NEW_TC_ARMS = tuple(a["arm"] for a in TC_LAYERWISE_ARMS)
# Decision 0005 (extended access, 18-23 Nov 2026): the runs the first weekend's results asked for
# go first, most important first. Everything in EARLIER_PLAN keeps its order after them.
EXTENDED_PLAN = (
    # 1. is PCD's adaptive weight better than a fixed weight at the same level?
    ("teacher-corruption", (0, 1, 2, 3, 4), None, tuple(a["arm"] for a in MATCHED_NWKD_ARMS)),
    # 2. PCGrad, the strongest competitor, had one seed at rho 0.5 / 0.7
    ("teacher-corruption", (0, 1, 2, 3, 4), None, ("pcgrad1s",)),
    # 3-5. PCDLM (the best clean-teacher PCD variant) on the clean benchmarks and under label noise
    ("core-p0", (0, 1, 2), None, tuple(a["arm"] for a in CLEAN_PCDLM_ARMS)),
    ("lit-p1", (0, 1), None, ("pcdlm_t1.0", "pcd_t0.5", "pcd_t1.0", "mgda")),
    ("label-noise", (0, 1), None, ("pcdlm_t1.0",)),
    # 6. five seeds for the core arms at the low-rho points too
    ("teacher-corruption", (3, 4), ("rho0.1", "rho0.3"), TC_CORE_ARMS),
    # 7. the stronger-student benchmark, three seeds
    ("core-p1", (0, 1, 2), None, None),
)
# Decision 0004: blocks 1-12 are the earlier plan (done or nearly done); the rest is ordered by
# how much each block can change the paper's PCD conclusion.
EARLIER_PLAN = (
    (
        "lit-p1",
        (0,),
        None,
        (
            "ce", "wkd_l0.9", "nwkd_l1.0", "pcd_t0.5", "pcd_t1.0", "pcgrad1s", "mgda",
            "dtokd", "dot_d0.075",
        ),
    ),
    ("pilot", (0,), None, None),
    ("mve-cifar10", (0,), None, None),
    ("core-p0", (0,), None, None),
    ("label-noise", (0,), None, None),
    ("teacher-corruption", (0,), None, None),
    ("teacher-corruption", (1, 2), ("rho0.5", "rho0.7"), TC_CORE_ARMS),
    ("teacher-corruption", (1, 2), ("rho0.0", "rho0.1", "rho0.3"), None),
    ("lit-p1", (0, 1), None, ("ce", "wkd_l0.9", "nwkd_l1.0", "pcd_t2.0", "pcgrad1s", "dot_d0.075")),
    ("temperature", (0,), None, None),
    ("multi-loss-dkd", (0,), None, None),
    ("pcd-ablations", (0,), None, None),
    ("teacher-corruption", (0, 1, 2), TC_CONDS, NEW_TC_ARMS),  # a: per-layer PCD, paired s0-2
    ("teacher-corruption", (3, 4), TC_CONDS, TC_CORE_ARMS),  # b: 5-seed confirmatory headline
    ("label-noise", (1,), None, LN_CORE_ARMS),  # c: CE-first vs KD-first, second seed
    ("core-p0", (1, 2), None, None),  # d
    ("core-p1", (0,), None, None),  # e
    ("teacher-corruption", (3, 4), TC_CONDS, NEW_TC_ARMS),  # f
    ("label-noise", (1,), None, None),  # g: the old plan's order from here
    ("sensitivity", (0,), None, None),
    ("mve-cifar10", (1,), None, None),
    ("label-noise-noisy-teacher", (0,), None, None),
    ("label-noise", (2,), None, None),
)
WEEKEND_PLAN = (*EXTENDED_PLAN, *EARLIER_PLAN)


def in_block(s, block):
    """Whether student spec ``s`` belongs to a WEEKEND_PLAN block (before first-match-wins)."""
    family, seeds, conditions, arms = block
    return (
        s["family"] == family
        and s["seed"] in seeds
        and (conditions is None or s["condition"] in conditions)
        and (s["arm"] in arms if arms is not None else s["arm"] not in DEFERRED_ARMS)
    )


def _validate_weekend_plan():
    by_name = {f["name"]: f for f in FAMILIES}
    for family, _seeds, conditions, arms in WEEKEND_PLAN:
        if family not in by_name:
            raise ValueError(f"unknown weekend-plan family: {family}")
        fam = by_name[family]
        known_conditions = {name for _, name in fam["conditions"]}
        known_arms = {a["arm"] for a in fam["arms"]}
        if conditions is not None and not set(conditions) <= known_conditions:
            unknown = set(conditions) - known_conditions
            raise ValueError(f"unknown weekend-plan condition for {family}: {unknown}")
        if arms is not None and not set(arms) <= known_arms:
            raise ValueError(f"unknown weekend-plan arm for {family}: {set(arms) - known_arms}")


_validate_weekend_plan()


def queue_order(students):
    """Apply the explicit weekend blocks, then retain all unmatched work in the old order."""
    pos = {id(s): i for i, s in enumerate(students)}
    matched, ordered = set(), []
    for plan_block in WEEKEND_PLAN:
        block = [s for s in students if id(s) not in matched and in_block(s, plan_block)]
        block.sort(key=lambda s: (s["seed"], pos[id(s)]))
        ordered.extend(block)
        matched.update(map(id, block))
    rank = {f: i for i, f in enumerate(PRIORITY)}
    tail = [s for s in students if id(s) not in matched]
    tail.sort(key=lambda s: (s["seed"], rank[s["family"]], pos[id(s)]))
    return ordered + tail


def _teacher_spec(key, s, arch, noisy, profile, crd=False):
    t = copy.deepcopy(BASE)
    t.update(
        kind="teacher",
        id=f"teacher/{key}",
        arch=arch,
        dataset=s["dataset"],
        batch_size=PROFILES[profile].get("batch_size", BASE["batch_size"]),
        imbalance=s["imbalance"],
        label_noise=s["label_noise"] if noisy else 0.0,
        noise_kind=s["noise_kind"],
        seed=0,
        epochs=TEACHER_EPOCHS[profile],
        profile=profile,
        family="teachers",
        tier=-1,
        condition=key,
        arm="ce",
        objectives=["ce"],
        teacher_keys=[],
        val_size=s["val_size"],
    )
    if crd and profile != "smoke":  # smoke (synthetic data) trains a stand-in instead
        t.update(pretrained=copy.deepcopy(CRD_TEACHERS[arch]), epochs=0)
    for k in ("teachers", "params", "loss_scale"):
        t.pop(k, None)
    return t


def passes(spec):
    """Student backward passes per step (cost model; 1 for CE / single-pass sums)."""
    return 1 if spec["method"] in ("ce", "wkd", "sum") else len(spec["objectives"])


def cost(spec):
    """Epoch-passes over 45k images: epochs x passes x (share of the training set used; 50k/45k
    with ``val_size: 0``)."""
    size = spec["train_fraction"] * (50000 - spec["val_size"]) / 45000
    if spec["imbalance"] > 1:  # mean of the exponential long-tail profile over 100 classes
        size *= sum(spec["imbalance"] ** (-k / 99) for k in range(100)) / 100
    return spec["epochs"] * passes(spec) * size
