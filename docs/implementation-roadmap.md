# Implementation roadmap (objective S)

Plan: `research/experiment-plans/pcd-kd-conflict.md`. Every stage runs in Orca workers with separate worktrees and
exclusive file ownership; every substantive change gets an independent cross-agent review (max two cycles), tests,
and a re-run of affected tests on the integration branch. Human approval gates are marked ⛔.

## Environment

- Python 3.11, `uv` (`uv sync --frozen`), dependencies pinned in `uv.lock` (torch 2.15.0, torchvision 0.29.2,
  numpy, pyyaml; dev: pytest, ruff). Apple M4 / MPS verified available (`torch.backends.mps.is_available()`).
- Orca setup hook: `orca.yaml` runs `uv sync --frozen` when `pyproject.toml` + `uv.lock` exist. ⛔ The repo's local
  Orca setting still overrides it with `pnpm install` (see final report for the one-time Settings fix).

## Stage 1 — Foundations (in progress: Orca Run `run_8242320e918a`)

| Track | Owner | Files | Done when |
|---|---|---|---|
| A: data, models, reproducibility, configs, CI, literature tooling | Codex | `src/pcdkd/{data,models,utils}`, `configs/`, `scripts/`, `.github/workflows/ci.yml`, `docs/literature/` | unit + synthetic integration tests pass; ruff clean; smoke throughput measured on synthetic data |
| B: KD loss, reliability gate, EMA normalizer, PCD QP, combiners, diagnostics | Claude | `src/pcdkd/{losses,optim,eval}`, `tests/fixtures/` | closed-form/KKT/Prop. 4.7/identity tests pass; matches official `pcd/qp.py` fixtures to 1e-8 |
| Cross-review | Claude ↔ Codex | `docs/reviews/stage1-*.md` | every finding AGREE/DISAGREE/UNSUPPORTED with evidence; verified issues fixed; tests re-run |

## Stage 2 — Trainer and experiment runner

Owner Codex (trainer loop, checkpoint-free logging, per-step diagnostics hook, deterministic data order, run manifest
with config + seed + git commit + torch/device info); owner Claude (teacher-output caching, corruption application,
gate integration, arm registry for CE/WKD/NWKD/PCGrad/PCD/G-PCD/G-NWKD). Integration test: 2-epoch run on a
synthetic CIFAR-shaped dataset for every arm on CPU.

## Stage 3 — ⛔ Data and smoke run (needs authorization)

Download CIFAR-10/100 with `scripts/fetch_cifar.py --yes` (md5-verified, 163 MB + 161 MB); 1-epoch ResNet-8 throughput
on MPS per arm. Decide the epoch budget.

## Stage 4 — ⛔ Minimum viable experiment (needs authorization)

Teacher ResNet-32 on CIFAR-10; 15 student runs (plan §9). Apply the go/no-go rule in decision 0001.

## Stage 5 — ⛔ Selection and confirmatory runs

Freeze the plan (commit hash), CIFAR-100 selection on validation, confirmatory 5 seeds; analysis scripts compute
paired bootstrap CIs, TOST, Holm; results reported with every seed.

## Quality gates (all stages)

`uv run ruff check .` · `uv run pytest -m "not slow and not requires_data"` · `scripts/check_repo_hygiene.py` ·
literature docs regenerated and unchanged · cross-agent review file present · no test weakened (reviewers diff tests).

## GitHub / CI (⛔ needs authorization)

No remote exists. When approved: create the GitHub repository, push the integration branch, open a PR to the default
branch; CI (`.github/workflows/ci.yml`) must pass; branch protection requiring CI + one human approval; merges only by
a human.
