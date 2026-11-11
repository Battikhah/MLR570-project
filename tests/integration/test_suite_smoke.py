"""End-to-end CPU smoke run of every distinct experiment *shape* in the suite (synthetic data).

A shape is everything that changes the code path: method, objectives, priority, gate, regime,
architectures, loss scaling and combiner parameters. Running one spec per shape exercises every
code path the GPU runs will take, in about a minute. ``-m slow`` runs the whole smoke profile.
"""

import json
import subprocess
import sys

import pytest
import torch

from pcdkd.experiments.runner import run_many, run_student
from pcdkd.experiments.suite import build_specs
from pcdkd.train.trainer import train

pytestmark = pytest.mark.integration


def _shape(s):
    return (
        s["method"],
        tuple(s["objectives"]),
        s["primary"],
        s["gate"],
        s["arch"],
        len(s["teacher_keys"]),
        s["teacher_corruption"] > 0,
        s["label_noise"] > 0,
        s["noise_kind"],
        s["imbalance"] > 1,
        s["train_fraction"] < 1,
        bool(s["loss_scale"]),
        tuple(sorted(s["params"])),
        s["momentum"] > 0,
        s["dataset"],
    )


SMOKE, SMOKE_TEACHERS = build_specs("smoke")
SHAPES = list({_shape(s): s for s in SMOKE}.values())


@pytest.fixture(scope="module")
def smoke_out(tmp_path_factory):
    out = tmp_path_factory.mktemp("suite")
    status = run_many(SHAPES, SMOKE_TEACHERS, out, device="cpu", workers=0, log=lambda m: None)
    return out, status


def test_every_shape_runs(smoke_out):
    out, status = smoke_out
    errors = [json.loads(p.read_text())["error"] for p in out.rglob("error.json")]
    assert status["failed"] == 0, errors[:5]
    assert status["done"] == len(SHAPES) > 50


def test_results_complete_and_resumable(smoke_out):
    out, _ = smoke_out
    results = [json.loads(p.read_text()) for p in out.rglob("result.json")]
    assert len(results) == len(SHAPES)
    for r in results:
        assert {"top1", "top5", "nll", "ece", "per_class_acc"} <= set(r["test"])
        assert r["env"]["torch"] == torch.__version__
        d = r["train"]["diagnostics"]
        if r["spec"]["method"] != "ce":
            assert d["steps"] >= 1 and "cos_raw" in d["numeric"]
        if r["spec"]["method"] == "pcd":
            assert "mu_0" in d["numeric"] and "active" in d["fractions"]
        if r["spec"]["gate"] != "none":
            assert "gate_accept" in d["numeric"]
        if r["spec"]["imbalance"] > 1:
            assert "acc_few" in r["test"]
    assert run_student(SHAPES[0], SMOKE_TEACHERS, out, device="cpu") == "skipped"


def test_analysis_report(smoke_out):
    out, _ = smoke_out
    proc = subprocess.run(
        [sys.executable, "scripts/analyze_suite.py", "--profile", "smoke", "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    report = (out / "smoke" / "REPORT.md").read_text()
    assert "## Where does PCD win?" in report and "teacher-corruption" in report


def test_single_pass_and_two_pass_wkd_agree():
    """WKD's cheap one-backward path must give the same update as its logged two-backward path."""
    from pcdkd.data.build import build_data

    spec = next(s for s in SMOKE if s["method"] == "wkd" and s["family"] == "core-p0")
    data = build_data(spec, synthetic=True)
    teacher = __import__("pcdkd.models", fromlist=["get_model"]).get_model("resnet32", 100)
    params = []
    for log_every in (1, 10**9):  # every step logged (two-pass) vs never logged (one pass)
        s = dict(spec, log_every=log_every, momentum=0.0)
        model, _ = train(s, data, [teacher], device="cpu", max_steps=3, log=lambda m: None)
        params.append(torch.cat([p.detach().flatten() for p in model.parameters()]))
    assert torch.allclose(params[0], params[1], atol=1e-5)


@pytest.mark.slow
def test_full_smoke_profile(tmp_path):
    status = run_many(SMOKE, SMOKE_TEACHERS, tmp_path, device="cpu", log=lambda m: None)
    assert status["failed"] == 0 and status["done"] == len(SMOKE)


def test_non_finite_loss_stops_the_run():
    """Logged steps stop in the EMA normalizer; unlogged steps at the per-epoch finiteness check."""
    from pcdkd.data.build import build_data
    from pcdkd.models import get_model

    wkd = next(s for s in SMOKE if s["method"] == "wkd" and s["family"] == "core-p0")
    data, teacher = build_data(wkd, synthetic=True), [get_model("resnet32", 100)]
    with pytest.raises(ValueError, match="finite"):
        train(
            dict(wkd, loss_scale={"kd": float("nan")}),
            data,
            teacher,
            device="cpu",
            max_steps=2,
            log=lambda m: None,
        )
    ce = next(s for s in SMOKE if s["method"] == "ce" and s["family"] == "core-p0")
    with pytest.raises(FloatingPointError):  # diverging CE run: no combiner, epoch-level check
        train(dict(ce, lr=1e30), data, [], device="cpu", max_steps=3, log=lambda m: None)
