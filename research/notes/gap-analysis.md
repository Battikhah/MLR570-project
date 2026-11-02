# Gap analysis — PCD / FedPCD and the MLR570 project

Sources are ids in `literature/sources.yaml` / `literature/repositories.yaml`. Labels: **VERIFIED** (checked against a
primary source by at least one agent and cross-reviewed), **INTERPRETATION**, **HYPOTHESIS**. Derived from Orca Run
`run_f9e2e8953f54` (see `docs/reviews/`).

## 1. Gaps in the PCD / FedPCD evidence base

| # | Gap | Status | Evidence | Addressed by this project? |
|---|---|---|---|---|
| E1 | No convergence theorem for the deployed PCD algorithm (only per-step direction properties) | VERIFIED | varam2026pcd §4.7 | No (theory); results reported as empirical only |
| E2 | PCD's empirical wins are in compression, attributed to objective **scale**, not conflict (cos ≤ 1.2e-3) | VERIFIED | varam2026pcd §5.1.1, App. E.3 Table 10 | Yes: S tests PCD where conflict is measured, not assumed |
| E3 | Missing control: low-weight, EMA-normalized, rescale-matched weighted sum (E.2 used w ≥ 0.1) | VERIFIED (narrowed in review) | varam2026pcd App. E.2, E.4 | Yes: NWKD arm with grid containing τ − c̄ |
| E4 | No independent replication of PCD or FedPCD | VERIFIED (search 2026-11-02, search-limited) | — | Partly (independent PCD implementation cross-checked vs official QP) |
| E5 | Official PCD release lacks baselines and the physical-pruning export; README reports an in-loop proximal step not disclosed in the paper | VERIFIED (repo) / paper disclosure gap VERIFIED (HTML keyword audit) | varam2026pcdcode `experiments/README.md`, `objectives.py` | Recorded; relevant only to backup M1 |
| E6 | PCD paper used no held-out validation split | VERIFIED | varam2026pcd App. H | Our protocol uses a 45k/5k split and single test evaluation |
| E7 | Realized-update guarantees fail under momentum (47–73% sign flips with SGD+momentum) | VERIFIED | varam2026pcd App. G Table 18 | Ablation (SGD vs SGD+momentum) |
| E8 | FedPCD: code not publicly released; no FedProx/SCAFFOLD/FedNova baselines; larger tuning grid than baselines; selection protocol not described | VERIFIED | irshaid2026fedpcd §5, App. A; project page | No (M2 not selected) — documented for the survey |
| E9 | FedPCD theorem requires full participation, deterministic local GD, H = 0, and an assumed alignment margin γ_eff > 0 that drift erodes | VERIFIED | irshaid2026fedpcd §4, App. B.1 | No — survey critical analysis |

## 2. Gaps in KD-under-conflict research

| # | Gap | Status | Evidence |
|---|---|---|---|
| K1 | Measured CE–KD gradient cosines exist only in a few-shot VLM-teacher setting | VERIFIED (single agent; arXiv record re-checked by coordinator) | kang2025dho App. G.2 Tables 26–27 (−0.39…−0.45 backbone, −0.76 head) |
| K2 | No measurement of CE–KD conflict on full-data CIFAR-100 with a trained CNN teacher | VERIFIED absence (search-limited) | — → hypothesis H0 |
| K3 | No study of how teacher corruption changes CE–KD conflict | VERIFIED absence (search-limited) | — → H0/H1 |
| K4 | Classification-priority KD already exists via projection/selection (SCOPE, TPKD, SCKD) and adaptive trade-offs (DTO-KD) | VERIFIED | chen2026scope, yuan2026tpkd, zhu2021sckd, hayder2026dtokd |
| K5 | No comparison of PCD against a tuned normalized weighted KD | VERIFIED absence (search-limited) | → H2–H4 |
| K6 | Teacher confidence is a weak predictor of per-sample KD usefulness (\|r\| ≤ 0.09) | VERIFIED (single agent) | sumit2026chad §5.2 — threat to the confidence gate |
| K7 | SCKD reports cosine thresholds > 1 (Table 6), an internal inconsistency | VERIFIED (single agent) | zhu2021sckd Table 6 |

## 3. Contradicting evidence the project must confront

- Well-tuned scalarization often matches specialized multi-task optimizers: xin2022mtohelp, kurin2022unitary,
  royer2023scalarization (VERIFIED as published; claims summarized by Claude).
- PCD's own Table 3: 54/102 ties with tuned baselines; one genuine loss in the compression regime (VERIFIED).
- **Theory predicts PCD up-weights a conflicting (corrupted) teacher**: λ_t = max(0, τ − c_t) (THEORETICAL GUARANTEE
  from varam2026pcd Cor. 4.6); Prop. 4.7 permits CE ascent. The project's headline method may lose to its controls.

## 4. Unexplored applications (INTERPRETATION; not pursued)

Group fairness as priority constraints (no PCD fairness work found); PCD for QAT (overlap with authors' stated
applications; GAQAT jiang2024gaqat is prior gradient-conflict QAT); continual learning (near-zero novelty:
PCD ≡ GEM/A-GEM at τ = 0 per varam2026pcd §3.2, positive margin in hu2020softgem).
