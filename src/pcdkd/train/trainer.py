"""One training run (teacher or student) from a flat run-spec dict; returns a JSON-able result.

Student objectives (``spec["objectives"]``, primary = ``spec["primary"]``):

* ``ce``: cross-entropy on the (possibly noisy) training labels.
* ``kd`` / ``kd@j``: Hinton KD to teacher 0 / teacher j (log-space, gated by ``spec["gate"]``).
* ``kdavg``: Hinton KD to the mean of all teachers' probabilities.
* ``tckd``, ``nckd``: DKD's target / non-target terms w.r.t. teacher 0.
* ``at``: attention transfer w.r.t. teacher 0's three stage outputs.

Single-pass methods (``ce``, ``wkd``, ``sum``) take one backward of the weighted loss and compute
per-objective gradients only on logging steps (for diagnostics), so their cost matches standard
KD. Every other method takes one backward per objective and combines them.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pcdkd.data.corruption import corrupted_teacher_distribution
from pcdkd.eval.diagnostics import DiagnosticsAggregator
from pcdkd.losses.extra import at_loss, dkd_parts
from pcdkd.losses.gate import reliability_mask
from pcdkd.losses.kd import kd_loss
from pcdkd.models import get_model
from pcdkd.optim.combiners import PCD, NormalizedWeightedSum, PCGrad, WeightedSum
from pcdkd.optim.combiners_extra import (
    DOT,
    MinNorm,
    MultiWeightedSum,
    SymmetricPCGrad,
    shadow_cosines,
)
from pcdkd.optim.grads import flat_grad, unflatten_to
from pcdkd.utils.atomic import save_torch
from pcdkd.utils.seed import dataloader_generator, seed_everything, worker_init_fn

SINGLE_PASS = {"ce", "wkd", "sum"}
TS_KEYS = (
    "cos_raw",
    "dot_norm",
    "active",
    "lambda_eff",
    "lambda_raw",
    "prop47_margin_actual",
    "norm_primary",
    "norm_secondary",
    "align_primary",
    "align_secondary",
)


class Paused(Exception):  # noqa: N818 - a pause, not an error
    """Training stopped early (access window closing); a checkpoint was saved for resuming."""


def _rng_state(train_dl):
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),  # noqa: NPY002 - global state set by seed_everything
        "python": random.getstate(),
        "loader": train_dl.generator.get_state() if train_dl.generator is not None else None,
    }


def _set_rng_state(state, train_dl):
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])  # noqa: NPY002
    random.setstate(state["python"])
    if state["loader"] is not None:
        train_dl.generator.set_state(state["loader"])


def device_auto():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_combiner(spec):
    m, p = spec["method"], spec.get("params", {})
    beta = p.get("beta", 0.99)
    table = {
        "wkd": lambda: WeightedSum(p["lam"], beta),
        "sum": lambda: MultiWeightedSum(p["lams"], normalize=False, beta=beta),
        "nwkd": lambda: NormalizedWeightedSum(p["lam"], beta, rescale=p.get("rescale", True)),
        "nws": lambda: MultiWeightedSum(p["lams"], True, beta, rescale=p.get("rescale", True)),
        "pcd": lambda: PCD(p["tau"], beta, rescale=p.get("rescale", True)),
        "pcgrad": lambda: PCGrad(p.get("lam", 1.0), beta),
        "pcgrad_sym": lambda: SymmetricPCGrad(p.get("lams", 1.0), beta),
        "mgda": lambda: MinNorm(False, rescale=p.get("rescale", True), beta=beta),
        "dto": lambda: MinNorm(True, rescale=p.get("rescale", False), beta=beta),
        "dot": lambda: DOT(p.get("delta", 0.075), spec["momentum"], p.get("lam", 0.9), beta),
    }
    if m not in table:
        raise ValueError(f"unknown method {m!r}")
    return table[m]()


def _loader(ds, batch, shuffle, seed, workers, device):
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        generator=dataloader_generator(seed) if shuffle else None,
        worker_init_fn=worker_init_fn if workers else None,
        pin_memory=device == "cuda",
        drop_last=shuffle and len(ds) >= batch,
        persistent_workers=workers > 0,
    )


@torch.no_grad()
def evaluate(model, loader, classes, device, teacher=None, groups=None):
    model.eval()
    correct, correct5, nll, n = 0, 0, 0.0, 0
    agree = 0
    conf_all, hit_all, per_class_hit, per_class_n = (
        [],
        [],
        torch.zeros(classes),
        torch.zeros(classes),
    )
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        p = logits.softmax(-1)
        conf, pred = p.max(-1)
        hit = pred == y
        correct += int(hit.sum())
        correct5 += int((logits.topk(min(5, classes), -1).indices == y[:, None]).any(-1).sum())
        nll += float(F.cross_entropy(logits, y, reduction="sum"))
        n += len(y)
        conf_all.append(conf.cpu())
        hit_all.append(hit.cpu())
        per_class_hit.index_add_(0, y.cpu(), hit.float().cpu())
        per_class_n.index_add_(0, y.cpu(), torch.ones(len(y)))
        if teacher is not None:
            agree += int((teacher(x).argmax(-1) == pred).sum())
    conf, hit = torch.cat(conf_all), torch.cat(hit_all).float()
    edges = torch.linspace(0, 1, 16)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += float(m.float().mean() * (conf[m].mean() - hit[m].mean()).abs())
    per_class = (per_class_hit / per_class_n.clamp_min(1)).tolist()
    out = {
        "top1": correct / n,
        "top5": correct5 / n,
        "nll": nll / n,
        "ece": ece,
        "n": n,
        "per_class_acc": per_class,
    }
    if teacher is not None:
        out["teacher_agreement"] = agree / n
    for name, ids in (groups or {}).items():
        out[f"acc_{name}"] = float(np.mean([per_class[i] for i in ids])) if len(ids) else None
    model.train()
    return out


def _lr_at(spec, epoch):
    lr, E = spec["lr"], spec["epochs"]
    if spec["schedule"] == "cosine":
        return 0.5 * lr * (1 + math.cos(math.pi * epoch / E))
    if spec["schedule"] == "step":  # mdistiller: x0.1 at 150/180/210 of 240 epochs
        return lr * 0.1 ** sum(epoch >= round(f * E) for f in (0.625, 0.75, 0.875))
    return lr


def _teacher_targets(spec, teachers, x, idx, labels, data, device, need_feats):
    """Return per-teacher log-probs (T=1), teacher-0 features, corruption flags for the batch."""
    logps, feats = [], None
    for j, t in enumerate(teachers):
        with torch.no_grad():
            out = t(x, return_features=need_feats and j == 0)
        if need_feats and j == 0:
            out, feats = out
        lp = F.log_softmax(out.float(), -1)
        if j == spec.get("corrupt_teacher", 0) and data["corruption_map"]:
            cmap = data["corruption_map"]
            rows = {r: cmap[int(i)] for r, i in enumerate(idx.tolist()) if int(i) in cmap}
            if rows:
                clean = data["clean_labels"][idx].numpy()
                lp = (
                    torch.from_numpy(
                        corrupted_teacher_distribution(
                            lp.cpu().double().numpy(),
                            clean,
                            rows,
                            kind="logits",
                            return_log_probs=True,
                        )
                    )
                    .float()
                    .to(device)
                )
        logps.append(lp)
    flags = torch.tensor([int(i) in data["corruption_map"] for i in idx.tolist()])
    return logps, feats, flags


def _objective_losses(spec, logits, s_feats, logps, t_feats, y):
    T = spec["T"]
    mask = None
    if spec.get("gate", "none") != "none" and logps:
        mask = reliability_mask(logps[0].exp(), y, spec.get("gate_q", 0.8), spec["gate"])
    out = {}
    for name in spec["objectives"]:
        if name == "ce":
            out[name] = F.cross_entropy(logits, y)
        elif name == "kd" or name.startswith("kd@"):
            j = 0 if name == "kd" else int(name[3:])
            out[name] = kd_loss(logits, logps[j], T, mask=mask, kind="logprobs")
        elif name == "kdavg":
            avg = torch.logsumexp(torch.stack(logps), 0) - math.log(len(logps))
            out[name] = kd_loss(logits, avg, T, mask=mask, kind="logprobs")
        elif name in ("tckd", "nckd"):
            if "_dkd" not in out:
                out["_dkd"] = dkd_parts(logits, logps[0], y, T)
            out[name] = out["_dkd"][0 if name == "tckd" else 1]
        elif name == "at":
            out[name] = at_loss(s_feats, t_feats)
        else:
            raise ValueError(f"unknown objective {name!r}")
    out.pop("_dkd", None)
    for k, m in spec.get("loss_scale", {}).items():
        out[k] = out[k] * m
    return out, mask


def _flatten_info(info):
    flat = {}
    for k, v in info.items():
        if isinstance(v, list) and v and all(isinstance(x, int | float) for x in v):
            flat.update({f"{k}_{i}": float(x) for i, x in enumerate(v)})
        elif not isinstance(v, list | dict | str | tuple):
            flat[k] = v
    return flat


def train(
    spec,
    data,
    teachers=(),
    *,
    device=None,
    workers=0,
    max_steps=None,
    log=print,
    checkpoint=None,
    ckpt_every=5,
    should_stop=None,
):
    """Train the model described by ``spec`` on ``data`` (from ``build_data``).

    With ``checkpoint`` (a path), the full training state is saved every ``ckpt_every`` epochs and
    an existing checkpoint is resumed, so a wiped or interrupted machine loses at most that many
    epochs. ``should_stop(next_epoch_seconds)`` is asked after each epoch; when it returns True the
    state is saved and :class:`Paused` is raised. The caller deletes the checkpoint after success.
    """
    device = device or device_auto()
    seed_everything(spec["seed"])
    torch.backends.cudnn.benchmark = True
    model = get_model(spec["arch"], data["classes"]).to(device)
    teachers = [t.to(device).eval() for t in teachers]
    params = [p for p in model.parameters() if p.requires_grad]
    method = spec.get("method", "ce") if spec["kind"] == "student" else "ce"
    objectives = spec["objectives"] if method != "ce" else ["ce"]
    comb = make_combiner(spec) if method != "ce" else None
    single = method in SINGLE_PASS
    dot = method == "dot"
    opt = torch.optim.SGD(
        params,
        lr=spec["lr"],
        momentum=0.0 if dot else spec["momentum"],
        weight_decay=0.0 if dot else spec["weight_decay"],
        nesterov=False,
    )
    train_dl = _loader(data["train"], spec["batch_size"], True, spec["seed"], workers, device)
    val_dl = _loader(data["val"], 512, False, 0, workers, device)
    labels_all = data["train_labels"]
    need_feats = "at" in objectives
    whole, thirds = DiagnosticsAggregator(), [DiagnosticsAggregator() for _ in range(3)]
    series, epoch_log = [], []
    step, start_epoch, elapsed = 0, 0, 0.0
    if checkpoint is not None and Path(checkpoint).exists():
        # our own file, written by save_torch below: full pickle (combiner and aggregator objects)
        st = torch.load(checkpoint, map_location=device, weights_only=False)  # noqa: S614
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        comb, whole, thirds = st["comb"], st["whole"], st["thirds"]
        series, epoch_log, step = st["series"], st["epoch_log"], st["step"]
        start_epoch, elapsed = st["epoch"], st["elapsed"]
        _set_rng_state(st["rng"], train_dl)
        log(f"[{spec['id']}] resumed from {checkpoint} at epoch {start_epoch}")

    def save(next_epoch):
        save_torch(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "comb": comb,
                "whole": whole,
                "thirds": thirds,
                "series": series,
                "epoch_log": epoch_log,
                "step": step,
                "epoch": next_epoch,
                "elapsed": time.time() - t0,
                "rng": _rng_state(train_dl),
            },
            checkpoint,
        )

    weights = None  # single-pass loss weights, primary first (same as the combiner's direction)
    if method == "wkd":
        weights = [1.0 - comb.lam, comb.lam]
    elif method == "sum":
        weights = comb._weights(len(objectives))

    t0 = time.time() - elapsed
    t_session, epochs_this_session = time.time(), 0
    total_steps = spec["epochs"] * max(len(train_dl), 1)
    for epoch in range(start_epoch, spec["epochs"]):
        for g in opt.param_groups:
            g["lr"] = _lr_at(spec, epoch)
        model.train()
        sums, n_b = {}, 0  # loss sums stay on device: no host sync per step
        for x, _, idx in train_dl:
            x = x.to(device, non_blocking=True)
            y = labels_all[idx].to(device)
            logits = model(x, return_features=need_feats)
            s_feats = None
            if need_feats:
                logits, s_feats = logits
            logps, t_feats, flags = [], None, torch.zeros(len(idx), dtype=torch.bool)
            if method != "ce":
                logps, t_feats, flags = _teacher_targets(
                    spec, teachers, x, idx, labels_all, data, device, need_feats
                )
            if method == "ce":
                losses, mask = {"ce": F.cross_entropy(logits, y)}, None
            else:
                losses, mask = _objective_losses(spec, logits, s_feats, logps, t_feats, y)
            names = [spec.get("primary", "ce")] + [
                k for k in losses if k != spec.get("primary", "ce")
            ]
            log_now = step % spec["log_every"] == 0 and comb is not None
            opt.zero_grad(set_to_none=True)
            info = None
            if comb is None:
                losses[names[0]].backward()
            elif single and not log_now:
                sum(w * losses[k] for w, k in zip(weights, names, strict=True)).backward()
            else:
                grads = {
                    k: flat_grad(losses[k], params, retain_graph=i < len(names) - 1)
                    for i, k in enumerate(names)
                }
                if dot and spec["weight_decay"]:
                    flat_p = torch.cat([p.detach().reshape(-1) for p in params])
                    grads[names[0]] = grads[names[0]] + spec["weight_decay"] / comb.w[0] * flat_p
                vals = {k: float(v.detach()) for k, v in losses.items()}
                if isinstance(comb, MinNorm):
                    d, info = comb.combine(grads, primary=names[0], losses=vals)
                else:
                    d, info = comb.combine(grads, primary=names[0])
                unflatten_to(params, d)
            opt.step()
            if info is not None and log_now:
                rec = _flatten_info(info)
                rec.update({f"loss_{k}": float(v.detach()) for k, v in losses.items()})
                if len(names) == 2 and "norm_secondary" in info and not isinstance(comb, MinNorm):
                    tau = spec.get("params", {}).get("tau", 0.5)
                    tau = float(np.atleast_1d(tau)[0])
                    rec.update(shadow_cosines(info, vals, tau=tau))
                if mask is not None:
                    rec["gate_accept"] = float(mask.float().mean())
                    if flags.any():
                        rec["gate_accept_corrupted"] = float(mask.cpu()[flags].float().mean())
                rec["corrupted_frac"] = float(flags.float().mean())
                noisy = labels_all[idx] != data["clean_labels"][idx]
                rec["noisy_label_frac"] = float(noisy.float().mean())
                whole.update(rec)
                thirds[min(2, 3 * step // max(total_steps, 1))].update(rec)
                series.append(
                    {"step": step, "epoch": epoch}
                    | {k: rec[k] for k in TS_KEYS if k in rec and rec[k] is not None}
                    | {k: v for k, v in rec.items() if k.startswith(("loss_", "shadow_", "mu_"))}
                )
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + v.detach()
            step, n_b = step + 1, n_b + 1
            if max_steps and step >= max_steps:
                break
        entry = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "time": time.time() - t0}
        entry |= {f"train_{k}": float(v) / max(n_b, 1) for k, v in sums.items()}
        if not all(math.isfinite(v) for k, v in entry.items() if k.startswith("train_")):
            raise FloatingPointError(f"non-finite training loss in epoch {epoch}: {entry}")
        if (epoch + 1) % spec.get("eval_every", 10) == 0 or epoch + 1 == spec["epochs"]:
            entry["val_top1"] = evaluate(model, val_dl, data["classes"], device)["top1"]
        epoch_log.append(entry)
        log(
            f"[{spec['id']}] epoch {epoch + 1}/{spec['epochs']} "
            + " ".join(f"{k}={v:.4g}" for k, v in entry.items() if isinstance(v, float))
        )
        if max_steps and step >= max_steps:
            break
        epochs_this_session += 1
        if checkpoint is not None and epoch + 1 < spec["epochs"]:
            per_epoch = (time.time() - t_session) / epochs_this_session
            stop = should_stop is not None and should_stop(per_epoch)
            if stop or (epoch + 1) % ckpt_every == 0:
                save(epoch + 1)
            if stop:
                raise Paused(f"{spec['id']}: paused after epoch {epoch + 1}/{spec['epochs']}")
    return model, {
        "epochs": epoch_log,
        "diagnostics": whole.summary(),
        "diagnostics_by_third": [a.summary() for a in thirds],
        "timeseries": series,
        "train_seconds": time.time() - t0,
        "steps": step,
        "device": device,
    }
