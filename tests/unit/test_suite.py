"""Static checks on the experiment suite: catch spec mistakes before any GPU time is spent."""

import collections

import pytest

from pcdkd.experiments.suite import FAMILIES, PROFILES, build_specs, cost, teacher_key
from pcdkd.optim.pcd import MAX_K
from pcdkd.train.trainer import make_combiner

STUDENTS, TEACHERS = build_specs("full")


def test_ids_unique_and_seeds_paired():
    ids = [s["id"] for s in STUDENTS]
    assert len(ids) == len(set(ids))
    cells = collections.defaultdict(set)
    for s in STUDENTS:
        cells[(s["family"], s["condition"], s["arm"])].add(s["seed"])
    fam_seeds = {f["name"]: f["seeds"] for f in FAMILIES}
    for (fam, _, _), seeds in cells.items():
        assert seeds == set(range(fam_seeds[fam])), fam  # same seeds for every arm -> pairing


@pytest.mark.parametrize("spec", STUDENTS[:: max(1, len(STUDENTS) // 400)], ids=lambda s: s["id"])
def test_every_spec_builds_a_valid_combiner(spec):
    assert len(spec["objectives"]) <= MAX_K
    assert spec["primary"] == spec["objectives"][0]
    if spec["method"] != "ce":
        make_combiner(spec)
    assert set(spec["teacher_keys"]) <= set(TEACHERS)


def test_all_specs_valid_quickly():
    for s in STUDENTS:
        if s["method"] != "ce":
            make_combiner(s)
        assert s["T"] > 0 and s["lr"] > 0 and s["epochs"] >= 1
        assert 0 <= s["teacher_corruption"] <= 1 and 0 <= s["label_noise"] < 1
        n_teachers = len(s["teacher_keys"])
        for o in s["objectives"]:
            if o.startswith("kd@"):
                assert int(o[3:]) < n_teachers
        assert s["gate"] == "none" or "kd" in s["objectives"]
        assert set(s["loss_scale"]) <= set(s["objectives"])


def test_each_condition_has_controls():
    """Every cell that contains PCD also contains a non-PCD comparator; regime families include
    the decisive NWKD control and CE-only for negative transfer."""
    cells = collections.defaultdict(set)
    for s in STUDENTS:
        cells[(s["family"], s["condition"])].add((s["method"], s["group"]))
    for (fam, cond), arms in cells.items():
        methods = {m for m, _ in arms}
        if "pcd" in methods:
            assert methods - {"pcd"}, (fam, cond)
        if fam in (
            "teacher-corruption",
            "label-noise",
            "low-data",
            "long-tail",
            "core-p0",
            "core-p1",
            "capacity-gap",
        ):
            assert {"nwkd", "ce", "dto", "pcd"} <= methods, (fam, cond)


def test_teachers_regimes():
    lt = [s for s in STUDENTS if s["family"] == "long-tail" and s["imbalance"] == 100][0]
    assert lt["teacher_keys"] == ["cifar100-resnet32-if100"]
    noisy = [s for s in STUDENTS if s["family"] == "label-noise-noisy-teacher"][0]
    assert "sym" in noisy["teacher_keys"][0]
    clean = [s for s in STUDENTS if s["family"] == "label-noise" and s["label_noise"] > 0][0]
    assert clean["teacher_keys"] == ["cifar100-resnet32"]
    assert teacher_key("cifar100", "resnet32", label_noise=0.2) != teacher_key(
        "cifar100", "resnet32", label_noise=0.4
    )
    for k, t in TEACHERS.items():
        assert t["kind"] == "teacher" and t["condition"] == k and t["teacher_corruption"] == 0


def test_low_data_is_iteration_matched():
    for s in STUDENTS:
        if s["train_fraction"] < 1:
            assert cost(s) == pytest.approx(
                240
                * (1 if s["method"] in ("ce", "wkd") else 2)
                * s["train_fraction"]
                * s["epochs"]
                / 240,
                rel=1e-9,
            )
            assert s["epochs"] == round(240 / s["train_fraction"])


def test_profiles_scale_schedule_only():
    screen, _ = build_specs("screen")
    smoke, _ = build_specs("smoke")
    # lit-p1 fixes the protocol's 240 epochs in every profile except smoke
    assert {
        s["epochs"] for s in screen if s["train_fraction"] == 1 and s["family"] != "lit-p1"
    } == {60}
    assert {s["batch_size"] for s in screen} == {s["batch_size"] for s in STUDENTS}
    full_cells = {(s["family"], s["condition"], s["arm"]) for s in STUDENTS}
    assert {(s["family"], s["condition"], s["arm"]) for s in screen} == full_cells
    assert {(s["family"], s["condition"], s["arm"]) for s in smoke} == full_cells
    assert max(s["seed"] for s in screen) < PROFILES["screen"]["max_seeds"]


def test_filters():
    s, t = build_specs("full", families=["pilot"])
    assert {x["family"] for x in s} == {"pilot"} and set(t) == {"cifar100-resnet32x4"}
    s, _ = build_specs("full", tiers=[0])
    assert {x["family"] for x in s} == {"lit-p1", "pilot", "mve-cifar10"}
