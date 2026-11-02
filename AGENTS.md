# AGENTS.md — shared instructions for all agents (Claude Code, Codex, others)

This repository holds the MLR570 (Advanced Machine Learning, AUS CSE) project:
a **survey paper** on knowledge distillation / gradient-conflict optimization and an
**ML research project** built around Priority-Constrained Descent (PCD).
Read `README.md` for layout and `docs/requirements/assignment-requirements.md` for what is graded.

## 1. Orchestration (Orca)

- All work is coordinated through **Orca**. If your prompt has Task/Dispatch IDs you are a
  supervised worker: report with `worker_done`, ask with `ask`, and put your **complete report in
  the message body** (a file path alone is not a report).
- Work only in **your own Orca worktree** and only on the **files your task assigns you**.
  Need a file someone else owns? Say so in your report instead of editing it.
- Do not create checkouts, worktrees, or branches yourself; Orca does that.
- Commit locally on your worktree branch when your task says to. **Never** push, merge, rebase
  shared branches, force-push, delete branches/worktrees, or open PRs unless the task explicitly says so.

## 2. Safety and approvals (needs explicit human approval)

- Pushing, merging, publishing a repository, deploying, deleting work.
- Downloading datasets or anything large (> ~200 MB), starting training runs longer than a smoke test,
  paid services, installing dependencies not already declared in `pyproject.toml`.
- Never commit credentials, datasets, checkpoints, or copyrighted PDFs (see `.gitignore`).

## 3. Research integrity (non-negotiable)

- **Never invent** references, authors, venues, URLs, repository names, numbers, or results.
- Every source goes in `literature/sources.yaml` with the fields defined in `literature/README.md`,
  including how it was verified. Unverified items stay `status: unverified`.
- Label claims explicitly: **VERIFIED** (checked against a primary source; cite it),
  **INTERPRETATION** (your judgement), **HYPOTHESIS** (untested), **RECOMMENDATION** (proposed action).
- Distinguish **theoretical guarantees**, **existing empirical evidence**, and **untested hypotheses**.
  Do not assume PCD outperforms alternatives; actively look for contradicting evidence.
- Do not bypass paywalls or access restrictions. Prefer official publisher / author / arXiv sources.
- Do not edit `Project Files/` (original course materials and the existing survey draft).
  Survey revisions go in `paper/` and must preserve the original draft's content and attributions.

## 4. Engineering conventions

- Python ≥ 3.11, managed with `uv` (`uv sync`, `uv run pytest`). Code lives in `src/`, tests in `tests/`.
- Every substantive code change ships with tests. Never weaken a test, change a metric, or alter
  measurements to make results look better. Report failing tests verbatim.
- Determinism: all experiments take an explicit seed; configs live in `configs/`; results must be
  reproducible from (config, seed, git commit).
- Small, focused commits: `<area>: <imperative summary>` (e.g. `literature: add PCD source records`).

## 5. Review protocol

- All substantive changes get an **independent cross-agent review** (Claude ↔ Codex).
- Reviews classify each finding AGREE / DISAGREE / UNSUPPORTED with document, code, or test evidence
  (file:line, page, command + output + exit code).
- At most **two** review/correction cycles per issue; then record it as an unresolved disagreement
  in `docs/reviews/` and escalate to the coordinator.
