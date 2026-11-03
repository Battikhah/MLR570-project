# EXPERIMENT PLAN rev. 2 — Priority vs symmetry in KD gradient combination (I1*)

Status: **DRAFT, revised after Codex review cycle 1** (`docs/reviews/phase7-codex-reviews-claude.md`).
- Nothing runs until approved: ⛔ the CIFAR-100 download on the GPU server, ⛔ the mdistiller teacher checkpoint, ⛔ every GPU run.
- GPU-hours are **ESTIMATES** until the pilot measures them.
- [PIN] items must be fixed to a commit SHA or config value before the plan is frozen. The frozen plan's commit hash is recorded before the first confirmatory run.

## 0. Protocol (fixed for all arms)
- **Data:** CIFAR-100. Train on 45,000 images; 5,000 stratified **validation** images (`pcdkd.data.cifar_splits`, `split_seed = 0`); the test set is used only for final reporting.
- **Pair P1:** ResNet32x4 teacher (mdistiller checkpoint [PIN]) → ResNet8x4 student. This is the pair reported by mdistiller, LSKD and DTO-KD.
- **Schedule:**
  - mdistiller `configs/cifar100/dkd/res32x4_res8x4.yaml` (Codex-verified on `master`, SHA [PIN]);
  - 240 epochs, SGD with momentum 0.9, batch 64, weight decay 5e-4;
  - learning rate 0.05, ×0.1 at epochs 150, 180 and 210.
- **KD loss:** Hinton KD with the mdistiller KD config (temperature, loss weights [PIN]); identical in every arm.
- **Seeds:** a seed fixes initialisation, data order and augmentation, and is **matched across arms** so differences can be paired.
- **Endpoint (primary):** test top-1 at the final epoch, for the configuration selected on validation. No checkpoint or config selection on test.

## 1. Arms
| ID | Arm | Combination rule | Native grid (exactly 3 configs) | Backward passes per step |
|---|---|---|---|---|
| A0 | CE only | — | — (1 config) | 1 |
| A1 | KD | fixed weighted sum (mdistiller weights) | KD weight ×{0.5, 1, 2} of the mdistiller default | 1 |
| A2 | KD + DOT | separate momenta | δ ∈ {0.05, **0.075** (official), 0.1} | 2 |
| A3 | **DTO-KD**, reimplemented, logit KD | min-norm on [∇log L_KD, ∇log L_CE], closed form (Eqs. 11–12) | {exact, amortised η_π = 1e-3, amortised η_π = 1e-2} | 2 (exact) / 1 (amortised) |
| A4 | **PCD(CE \| KD)** | priority projection, EMA normalisation, rescale to ‖g_CE‖ | τ ∈ {0.1, 0.5, 1.0} | 2 |
| A5 | **Normalised-WS control** | d = s_CE·g_CE + λ·s_KD·g_KD, rescaled to ‖g_CE‖ | λ ∈ {0.1, λ̄_PCD (pilot mean of PCD's μ, counted as a grid point), 1.0} | 2 |
| A6a | KD + LS (LSKD) | fixed sum, standardised logits | published LS settings (1 config) | 1 |
| A6b | PCD(CE \| KD+LS) | as A4, with the LS KD loss | τ ∈ {0.1, 0.5, 1.0} | 2 |

Notes:
- **Why the τ grid reaches 1.0 (TG, PCD Cor. 4.3/4.6, unit-norm view):**
  - With CE primary, the constraint is inactive whenever KD already gets τ-progress (c ≥ τ). The update is then the pure CE direction.
  - With small τ and aligned gradients, PCD(CE | KD) may therefore approximate CE-only training (A0) and discard KD's extra signal.
  - At τ = 1 the constraint is always active, giving d = g̃_CE + (1 − c)·g̃_KD.
  - **HYPOTHESIS H0′:** PCD at small τ performs close to A0, and only larger τ recovers KD's benefit. The pilot's distribution of c decides whether τ ∈ {0.1, 0.5, 1.0} brackets the active regime.
- **Exact definition of the A5 control** (identical to A4 except the coefficient):
  - the same `pcdkd.optim.EMANormalizer` (β = 0.99 [PIN], ε = 1e-8, bias correction, per-loss state starting at t = 0, updated every step);
  - λ multiplies the *normalised* KD gradient;
  - the same final rescale to ‖g_CE‖;
  - the same SGD momentum buffer, weight decay and learning rate, with no state reset;
  - the same mean batch reduction for both losses.

  A4 − A5 therefore isolates *adaptive projection versus a fixed coefficient*.
- **A3 reimplementation:**
  - Eqs. 11–12 on the Gram matrix of the log-loss gradients, π projected to the simplex;
  - the amortised Eq. 18 update followed by a softmax renormalisation, as stated in the paper;
  - DTO-KD's feature projection P and its separate heads are **omitted**, so that every arm shares one logit-KD loss. This is a limitation: we compare combination rules, not the full DTO-KD system.
  - Unit tests check π* against a brute-force simplex minimisation and check Cor. 3.3 (equal inner products) numerically.
- **Equal tuning budget:** 3 validation configurations per tunable arm, 1 seed each. Shared settings are never tuned per arm. Published-default arms (A0, A6a) get 1 configuration and are labelled as such.

## 2. Hypotheses and statistics
- **Confirmatory family** (Holm-corrected, α = 0.05; 10 paired seeds each):
  - **C1:** A4 vs A3 (priority vs symmetry);
  - **C2:** A4 vs A5 (projection vs a fixed normalised coefficient);
  - **C3:** A4 vs A2 (vs the momentum split).
- **Analysis of each contrast:**
  - the 10 per-seed differences, their mean, a 95% t-interval and a paired bootstrap interval;
  - all seeds listed.
- **Decision rules:**
  - **Superiority:** claimed if the Holm-adjusted p-value is < 0.05.
  - **Equivalence (secondary analysis, not gated):** for each contrast, report the 90% TOST interval (two one-sided tests at α = 0.05 each), with δ = 0.5 points fixed now.
    - Claim equivalence only if that interval lies entirely inside ±δ.
    - It is always computed and reported, whatever the pilot shows. No pilot-variance gate applies (cycle-2 disposition: three pilot runs cannot estimate SD reliably).
    - Expected precision, for transparency only: with n = 10 pairs the half-width ≈ 0.58·SD_diff (t₀.₉₅,₉/√10). With independent seed noise, SD_diff ≈ √2·SD_seed, so half-width ≈ 0.82·SD_seed.
    - The realised SD_diff from the 10 confirmatory pairs is reported, and equivalence conclusions are read against it.
  - **Otherwise the result is "inconclusive".**
- **Exploratory, uncorrected, labelled as such:**
  - A6b vs A6a (does PCD add to LS?);
  - A4 vs A1 and A0;
  - H0′.
- **Mechanism outcomes** (descriptive; logged every 50 steps):
  - cos(g_CE, g_KD) distribution;
  - PCD active rate, μ, and `prop47_margin_actual`;
  - DTO-KD π;
  - realised update-direction alignment after momentum;
  - **shadow directions:** on the same batch gradients, compute the A3, A4 and A5 directions and their pairwise cosines. This is cheap, adds no backward pass, and shows how often the rules actually differ.

## 3. Pilot (go/no-go before any confirmatory run)
- **Runs:** A0, A1, A3 (exact), A4 (τ = 0.5); 1 seed; full 240 epochs; ≈ 4 runs.
- **Outputs:**
  - measured GPU-h per arm, which replaces §4;
  - the c distribution and active rate;
  - seed-noise SD from A1 run on 2 extra seeds (+2 runs). **Descriptive only:** three runs give a low-precision SD estimate, so it gates nothing.
- **Go** if:
  - (i) the measured budget for §4 fits the approved GPU allowance;
  - (ii) the shadow directions show A3, A4 and A5 differ on a non-trivial fraction of steps (direction cosine < 0.99 on ≥ 10% of logged steps).
- **If (ii) fails:** the rules are numerically near-identical on this benchmark. Report that as the finding, run only C2 and the exploratory set, and skip C1/C3 superiority.

## 4. Run count and budget (ESTIMATE; row by row)
| Block | Runs | GPU-h per run (1-pass / 2-pass) | GPU-h |
|---|---|---|---|
| Pilot | 6 (4 + 2 seed-noise) | ~1–2 / ~1.5–3 | ~8–14 |
| Selection: A1, A2, A3, A4, A5, A6b × 3; A0, A6a × 1 | 20 | 3 single-pass (A1) + 2 published (A0, A6a) + 15 two-pass | ~28–57 |
| Confirmatory: A2–A5 × 10 seeds | 40 | two-pass (A3 exact) | ~60–120 |
| Exploratory: A0, A1, A6a, A6b × 5 seeds | 20 | 15 single-pass + 5 two-pass | ~23–45 |
| **Total** | **86** | | **≈ 119–236 GPU-h (say ~120–240)** |

The range reflects unmeasured throughput. Codex's mechanism note gives 1–4 h per CIFAR-100 two-pass run on one A100-class GPU, also unmeasured.

## 5. Follow-up (only after I1*, separately approved): I4, priority-order ablation
- P1, with teacher outputs corrupted on 30% of training indices plus a 0% control.
- Arms: KD, tuned NWKD, PCD(CE | KD), PCD(KD | CE), gated NWKD, gated PCD(KD | CE), and RLD (identical teacher logits; licence and protocol [PIN]).
- **Claim scope:**
  - PCD's μ is batch-level and does not select corrupted examples;
  - per-example teacher correctness, confidence and loss are logged as diagnostics only;
  - no reliability-protection claim is made.
- **Budget:** about 7 arms × 2 conditions × 5 seeds + selection ≈ 90 runs (ESTIMATE, re-derived after the I1* pilot).

## 6. Code work (Stage 2 replacement; all cross-reviewed, with synthetic CPU tests before GPU)
- Port PCD (reusing `pcdkd.optim`) and the A5 control into mdistiller as a trainer/distiller pair, so DOT and the teachers are byte-identical to the official code ([PIN] SHA).
- DTO-KD exact and amortised combiners, with closed-form and Cor. 3.3 tests.
- LS standardisation for A6.
- Shadow-direction logging.
- On CUDA, compute the Gram matrix on the device in float64 (removes the MPS-motivated CPU sync).
