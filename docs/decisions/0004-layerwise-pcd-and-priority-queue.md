# Decision 0004 — Per-layer PCD variants and a most-important-first queue

- **Status:** Adopted for the remaining workstation window (ends Monday 07:40 UAE); amends the block order of decisions 0002 and 0003.
- **Date:** 2026-11-15
- **Scope:** Two new arms (code), four new arm specs in `teacher-corruption`, and the order of `suite.WEEKEND_PLAN`. Existing spec IDs and contents do not change.

## Context

- **VERIFIED (screen results so far):** CE-primary PCD behaves like NWKD at a matched weight.
- **INTERPRETATION (closed form):** For K = 2, PCD on the whole flattened gradient gives
  d~ = g~_CE + mu g~_KD with mu = max(0, tau − g~_CE·g~_KD / ||g~_KD||²) (Cor. 4.6). One
  constraint can only produce a weighted sum, and mu barely moves over training, so PCD ≈ NWKD with
  lambda ≈ mean(mu). No single-vector PCD can differ from a tuned NWKD except through the time variation of mu.
- About 15 GPU hours remain at about 9 runs/h (60-epoch runs), not enough for the whole screen plan.

## Decision

### Two variants (the concept stays: CE primary, KD as a minimum-progress constraint, smallest change to the CE direction)

- **`pcdl` (`LayerPCD`)**: the same K = 2 QP, solved per parameter group (first component of the
  parameter name: `conv1`, `bn1`, `layer1`–`layer3`, `linear` for the CIFAR ResNets and `fc` for the `md_` ones). Each group has its own
  EMA norm normalization and its own mu_l, and its direction is rescaled to that group's ||g_CE^l||.
  Different layers get different KD weights, which no single NWKD lambda reproduces.
- **`pcdlm` (`LayerPCDM`)**: as `pcdl`, on DOT-style separate momentum buffers
  m_CE ← β_ce m_CE + g_CE, m_KD ← β_kd m_KD + g_KD (both 0.9 by default, `spec["momentum"]`),
  rescaled per group to ||m_CE^l||. Optimizer momentum and weight decay are off in the optimizer;
  weight decay is added to the CE gradient as for DOT.

Both reuse `pcd_k2_closed_form` and `_rescale` (no new solver) and save their state
(per-group EMA norms, buffers) in the checkpoint like DOT. Group boundaries come from the
trainer (`comb.set_groups`, computed once from `named_parameters()`); other combiners are unchanged.

Diagnostics: `lambda_eff` is the numel-weighted mean of mu_l, `active` the fraction of groups with
mu_l > 0; the time series and result keep `mu_group_i` / `cos_group_i` (index = position in
`result["train"]["param_groups"]`). `analyze_suite.py` treats the arms as their own groups
(`pcdl`, `pcdlm`) and leaves them out of the PCD-vs-baseline headline selection.

### Arms

`pcdl_t0.5`, `pcdl_t1.0`, `pcdlm_t0.5`, `pcdlm_t1.0`, appended at the end of the `teacher-corruption` arm list only.
Its rho = 0.0 condition is the clean baseline, and seeds 0–2 of every core arm exist, so pairing is free.
They are in `DEFERRED_ARMS`, so "all arms" blocks do not pick them up; only blocks (a) and (f) below
select them, and the remaining 100 − 60 = 40 runs (rho 0.1/0.3 and the other seeds) fall to the tail.

### Queue (blocks 1–12 of the previous plan are unchanged and stay first)

| Block | Selection | Runs |
|---|---|---|
| 1–12 | as in 0002 / 0003 (lit-p1 s0 … pcd-ablations s0) | 9, 3, 12, 16, 91, 65, 32, 78, 7, 20, 12, 16 |
| a | teacher-corruption s0–2, rho0.0/0.5/0.7, the 4 new arms | 36 |
| b | teacher-corruption s3–4, rho0.0/0.5/0.7, `TC_CORE_ARMS` (5-seed headline) | 48 |
| c | label-noise s1, ce, pcd_t1.0, pcdrev_t0.5/1.0, nwkd_l1.0/4.0, wkd_l0.9, dot_d0.075 | 56 |
| d | core-p0 s1–2 | 32 |
| e | core-p1 s0 | 16 |
| f | teacher-corruption s3–4, rho0.0/0.5/0.7, the 4 new arms | 24 |
| g | label-noise s1 (rest), sensitivity s0, mve-cifar10 s1, label-noise-noisy-teacher s0, label-noise s2 | 35, 84, 12, 24, 91 |
| tail | everything else, old order | 1018 |

Rationale: (a) is the only open question that can change the paper's PCD conclusion; (b) makes the
headline five-seed; (c) gives the CE-first vs KD-first ordering result a second seed; (d, e) are
the clean baselines.

## Consequences / deviations to report

- The new arms are post-hoc, motivated by the screen results; they are exploratory, not pre-registered. Report them as such, with tau fixed in advance at 0.5 and 1.0 (no tuning on test).
- **HYPOTHESIS (untested):** per-layer mu differs enough across layers to separate `pcdl` from NWKD. If mu_l is flat across layers and time, `pcdl` ≡ NWKD again; that is a result to report.
- `pcdlm` changes two things at once (per-layer QP and momentum split); comparing with `pcdl` and DOT separates them only approximately.
- `core-p1` s0 and the label-noise s1 remainder moved later than in 0003; `dtokd` stays deferred.
- Group choice (first name component) is coarse; `bn1` is a tiny group and its mu_l is noisy.
