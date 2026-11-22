# Decision 0006 — Testing why PCD ties a fixed weight: gated seeds and a teacher that switches

- **Status:** Adopted. Two blocks are added at the front of decision 0005's `EXTENDED_PLAN`; everything else keeps its order.
- **Date:** 2026-11-22
- **Scope:**
  - A new spec field, `corrupt_from` (a fraction of training; absent means corrupted from the start), read by the trainer (`_corruption_active`).
  - A new family, `teacher-switch`.
  - Two queue blocks.

  Existing specs do not change: 0 of the 1,939 screen specs and 0 of the 2,146 full specs differ from the previous commit, and the teacher specs are identical. The new family adds 60 screen and 36 full runs.

## Context

These are the results of the extended window up to Sunday 22 Nov, 09:23 (784 runs). The same numbers, from the 10:08 copy (786 runs), are committed in `research/results/screen-2026-11-22/`.

**VERIFIED, 5 seeds, teacher corruption:**
- PCD ties NWKD at the fixed weight matching its average λ_eff at every ρ:
  - PCD τ0.5 − NWKD 0.25: from −0.07 to +0.36 pp at ρ ≥ 0.3, all intervals include 0.
  - PCD τ1.0 − NWKD 0.75: from −0.28 to +0.38 pp, all intervals include 0.
  - The one significant gap goes against PCD: τ0.5 trails NWKD 0.25 with a clean teacher, −0.69 [−1.27, −0.11].

**VERIFIED, diagnostics:**
- **PCD's weight barely moves within a run.** λ_eff sd is about 0.1. For PCD τ1.0 at ρ0, it is 0.35 / 0.41 / 0.49 by thirds of training.
- **The CE–KD cosine is steady:** sd 0.08–0.09 within a run, mean 0.57 at ρ0 and 0.25 at ρ0.7.
- **PCD's step points almost the same way as the NWKD step:** cosine 0.98–0.99 for τ1.0.

**INTERPRETATION:**
- The cosine is set by the teacher–label disagreement rate, which is fixed for a run.
- The batch average hides which samples the teacher gets wrong.
- So PCD reduces to a fixed weight, as Cor. 4.6 predicts when c is steady.

## Decision

1. **Per-sample check.** `teacher-corruption`, seeds 1–4, ρ0.5/0.7, arms `g_pcd_t0.5` and `g_nwkd_l1.0`: 16 runs.
   - Tests whether judging the teacher per sample recovers what the batch cosine misses.
   - Until now these arms had 1 seed at high ρ.
2. **A teacher whose quality changes within the run.** `teacher-switch`: the teacher is clean until half of training (epoch 30 of 60 in screen), then corrupted at ρ0.5 or ρ0.7 with the same per-sample map as `teacher-corruption`.
   - **Arms:** NWKD 0.25 and 0.75, PCD τ0.5 and τ1.0, and KD-primary PCD τ0.5 and τ1.0. That is 5 seeds, 60 runs.
   - This is the one setting where c changes within a run, so an adaptive weight can differ from a fixed one.
   - **Expected direction (HYPOTHESIS):**
     - CE-primary PCD raises its KD weight when the teacher turns bad, which is the wrong way.
     - KD-primary PCD raises its CE weight, which is the right way.
   - **CE needs no rerun.** It ignores the teacher, so its `teacher-corruption` runs (same data and seeds) are the reference.

At the measured pace of about 11 runs/h, the two blocks take about 7 h of the roughly 21 h left in the window.

## Consequences

- **If PCD (either priority) beats its matched fixed weight here, but not under constant corruption:** adaptivity helps exactly when the teacher's reliability changes. That is a positive, bounded claim.
- **If it ties here too:** PCD's batch-level rule is too slow or too coarse even in its best case. The paper then reports PCD as a principled way to bound the KD weight, not as an adaptive combiner.
- **Decision 0005's later blocks lose about 7 h:** the tail of tier 3 (core-p1 seeds, the noisy-teacher label noise, sensitivity on sym0.4, label-noise seed 2) is less likely to run this window.
- **Not yet done:** a cross-agent review of this change, required by AGENTS.md for substantive changes.
