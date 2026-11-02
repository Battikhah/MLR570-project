# Experiment plan — PCD vs tuned normalized KD under unreliable teachers (objective S)

Decision record: `docs/decisions/0001-research-objective.md`. Status: **pre-registration draft**; freeze this file
(commit hash recorded in results) before any confirmatory run. Nothing below is a result.

## 1. Research question

In CE-primary / KD-secondary distillation on CIFAR-100, does PCD's conflict-adaptive KD coefficient improve, match, or
degrade student accuracy and negative transfer relative to a budget-matched, tuned, EMA-normalized weighted KD, as
teacher reliability is degraded — and does a correctness+confidence reliability gate change that comparison?

## 2. Hypotheses

| ID | Type | Statement | Basis |
|---|---|---|---|
| H0 | Precondition (pilot) | Ungated, the per-step normalized cos(g_CE, g_KD) is lower at 30% corruption than at 0%, and the PCD constraint is active on a non-trivial fraction of steps. | UNTESTED HYPOTHESIS; DHO (few-shot) measured −0.39…−0.76; head-level inner product scales with ‖p−y‖² (DHO Cor. 1) |
| H1 | Mechanism check | PCD's implicit λ_t = max(0, τ − c_t) is larger under 30% corruption than under 0%. | THEORETICAL GUARANTEE given H0 (PCD Cor. 4.6); verifies the implementation, not a finding |
| H2 | **Confirmatory** | Ungated, 30% corruption: PCD-KD test accuracy is **not higher** than tuned normalized WKD (one-sided; theory predicts harm). | PCD Cor. 4.6 + Prop. 4.7 (UNTESTED empirically) |
| H3 | **Confirmatory** | Clean teacher (0%): \|PCD − tuned normalized WKD\| < δ = 0.5 pp (equivalence, TOST). | Steady c ⇒ constant λ reproduces PCD (UNTESTED) |
| H4 | **Confirmatory** | Gated, 30% corruption: \|gated PCD − gated normalized WKD\| < δ (equivalence). | Gate removes corrupted outputs by construction (VERIFIED design property); residual conflict small (UNTESTED) |
| H5 | Exploratory | The original proposal's question: gated PCD vs CE-only and vs the strongest tuned baseline; negative transfer reduction. | Proposal p. 2 |

Confirmatory family = {H2, H3, H4}; Holm correction at α = 0.05. Everything else is exploratory and labelled so.
If H0 fails, H2–H4 are not run as confirmatory (see gating rule).

## 3. Arms

| Arm | Description | Tuned hyper-parameters (validation only, equal budget) |
|---|---|---|
| CE | Label-only student | — |
| WKD | Hinton KD, raw weighted sum `(1−λ)CE + λ T²·KL` | λ grid (4 values) |
| NWKD | EMA-normalized weighted sum `g̃_CE + λ g̃_KD`, rescaled to ‖g_CE‖ exactly as PCD | λ grid (4 values) **including pilot τ − c̄** |
| PCGrad | Project KD gradient off CE when conflicting (protect-the-primary baseline; SCOPE-like) | — |
| PCD | PCD(τ), EMA normalization, final rescale; ungated | τ ∈ {0.05, 0.1, 0.3} |
| G-PCD | PCD with correctness+confidence gate on KD examples (training labels only) | τ, q |
| G-NWKD | NWKD with the identical gate mask | λ, q |

Shared: temperature T selected once on validation for WKD and reused; same optimizer (SGD, no momentum for the main
study, per proposal), schedule, epochs, batch size, augmentation, initialization, data order, and corruption map
per seed. Momentum is an ablation (PCD App. G reports 47–73% realized-sign flips with SGD+momentum).

Dropped from the proposal (scope + no verified reference code): DTO-KD, SCKD, SCOPE re-implementations, CAGrad.
Recorded as a limitation.

## 4. Data and models

- CIFAR-10 (pilot), CIFAR-100 (main). Official archives (163 MB / 161 MB), checksum-verified; no redistribution;
  cite Krizhevsky 2009. **Download requires authorization.**
- Split: 45,000 train / 5,000 validation, stratified, fixed split seed shared by all runs; official test set
  evaluated **once** per final configuration.
- Teacher: CIFAR ResNet-32, trained once per dataset with a fixed seed; frozen; its train/val/test accuracy and ECE
  are reported. Students: ResNet-8 (main), ResNet-14 (capacity check, exploratory).
- Corruption: for each seed, a fixed map selects 10% or 30% of **training** examples; the teacher's output on those is
  replaced by a high-confidence wrong distribution (wrong class drawn uniformly from the non-true classes; fixed
  confidence p_wrong = teacher's own max-prob on that example). Map is saved with the run.

## 5. Metrics

Primary: student top-1 test accuracy. Secondary: negative transfer = acc(arm) − acc(CE); accuracy drop (30% vs 0%);
CE and KD losses; ECE. Per-step diagnostics (logged every k steps): raw and normalized cos(g_CE, g_KD) on the full
batch and on gated-in examples; ‖g_CE‖, ‖g_KD‖, EMA scales; PCD active-constraint rate; infeasible/fallback rate;
implicit λ_t; Prop. 4.7 margin `1 + τc − c²` and fraction of steps with a non-positive margin; gate acceptance rate by
corruption status; wall-clock and peak memory.

## 6. Statistics

Paired seeds (same init, data order, corruption map across arms). Report mean ± SD, every seed, paired differences
with 95% bootstrap CIs (10,000 resamples over seeds; descriptive given n = 5). Equivalence via TOST at δ = 0.5 pp.
Tuning budget parity: identical number of validation evaluations per arm; all grids published. No test-set access
during selection.

## 7. Ablations

Gate on/off; confidence-only vs correctness+confidence gate; τ sweep; SGD vs SGD+momentum; EMA vs instantaneous
normalization; with vs without PCD's final norm rescale (isolates direction from magnitude); adaptive-λ_t scalarization
using PCD's own λ_t (implementation identity check, expected equal to PCD — unit-test level, not an arm).

## 8. Threats to validity

- Full-data CIFAR-100 CE–KD cosine may be ≈ 0, making PCD ≈ NWKD everywhere (H0 failure) — handled by the gating rule.
- The gate removes the corrupted outputs, so gated comparisons probe low conflict only (by design; H4).
- PCD is outside its theory: non-convex KD secondary, no common stationary point (PCD §4.8) — results are empirical.
- Single teacher per dataset; teacher quality affects conflict. Report teacher metrics; ResNet-14 check is exploratory.
- Five seeds give low power; CIs are descriptive; equivalence margins pre-declared.
- MPS non-determinism; report device, torch version, and run-to-run variance from a repeated seed.
- Implementation fidelity: our PCD is cross-checked against the official `pcd/qp.py` closed form (MIT) in tests.
- Artificial corruption does not represent all real teacher errors (acknowledged in the proposal).
- Tuning-budget asymmetry; test leakage — controlled by §6.
- Scope: instructor acceptance of the narrowed design is unconfirmed.

## 9. Minimum viable experiment (MVE)

1. Smoke test (no dataset needed for timing on synthetic tensors; then 1 epoch CIFAR-10 once authorized):
   images/s for ResNet-8 with CE-only, WKD, and PCD on MPS.
2. CIFAR-10, teacher ResNet-32 (1 run), students 20 epochs, 3 seeds × corruption {0, 30%} × {CE (seed only),
   NWKD (λ from a 3-point validation pilot), PCD τ = 0.1} ⇒ 3 + 12 = 15 student runs, all diagnostics on.
3. Go/no-go per `docs/decisions/0001-research-objective.md`.

## 10. Run budget (counts; time is UNTESTED until the smoke test)

| Stage | Runs |
|---|---|
| Teachers (CIFAR-10, CIFAR-100) | 2 |
| MVE | 15 |
| CIFAR-100 selection (1 seed, validation) | ≈ 48 |
| CIFAR-100 confirmatory (5 seeds) | ≈ 80 |
| **Total** | **≈ 145** (+ ablations) |

If throughput < ~1,000 img/s: reduce confirmatory to 3 seeds and corruption {0, 30%} (≈ 100 runs total).
