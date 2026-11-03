# EXPERIMENT PLAN — PCD vs state-of-the-art KD (I1) and reversed priority under unreliable teachers (I4)

Status: **DRAFT for cross-review and human approval.** Nothing in this plan runs until approved.
- Gates: ⛔ teacher-checkpoint download, ⛔ CIFAR download on the GPU server, ⛔ every GPU run.
- Compute figures are **ESTIMATES** (one A100-class GPU), to be replaced by a measured pilot.
- Hyperparameters marked [VERIFY] are recalled from the mdistiller configs and must be checked against the repo
  (`megvii-research/mdistiller`, MIT) before freezing.

## 0. Shared protocol
- **Benchmark:** CIFAR-100, mdistiller protocol.
  - [VERIFY] 240 epochs, SGD with momentum 0.9, batch 64, weight decay 5e-4.
  - [VERIFY] Learning rate 0.05 (0.01 for ShuffleNet/MobileNet students), decayed ×0.1 at epochs 150/180/210.
  - Teachers: the mdistiller pretrained checkpoints (⛔ download).
- **Pairs:**
  - P1: ResNet32x4 → ResNet8x4 (the main reference pair; most published numbers);
  - P2: VGG13 → VGG8;
  - P3: ResNet32x4 → ShuffleNet-V2 (heterogeneous; DOT and LSKD report it).
- **Model selection:**
  - a 5,000-image stratified validation split from train (`pcdkd.data.cifar_splits`, Stage 1 code);
  - hyperparameters chosen on validation with 1 seed, then **frozen**;
  - final results over **5 seeds**, reported on the test set with every seed listed;
  - published numbers are quoted as context only, never as our baseline runs.
- **Equal tuning budget:** each method with a tunable weight gets the same number of grid points (4).
- **Statistics:** paired by seed; bootstrap 95% CIs on differences; TOST equivalence with δ = 0.5 points; Holm correction within each RQ.
- **Diagnostics logged every step (Stage 1 `pcdkd.eval.diagnostics`):**
  - cos(g_CE, g_KD) and, for K=3, every pairwise cosine;
  - raw and EMA norms;
  - active-set rate and implicit λ_t;
  - `prop47_margin_actual` and its non-positive fraction;
  - the infeasible rate (K=3).
- **Reproducibility:** run manifest (config, seed, commit, torch/CUDA versions); deterministic data order.

## 1. Pilot (go/no-go), P1 only, 1 seed, full 240 epochs
Arms: KD, PCD(CE | KD) τ = 0.05, PCD(CE | TCKD, NCKD) τ = 0.05, DKD at its default.

Measure:
- the cosine distributions;
- the active rate;
- wall-clock per epoch per arm (replaces every ESTIMATE below).

**Go** if both hold:
- (i) the cosine distribution or the active rate shows PCD is not just a fixed weighted sum. Concretely, an active rate < 95% or a cosine standard deviation > 0.05.
- (ii) the measured cost fits the approved budget.

**Otherwise:** report the pilot as a negative mechanism result and run only the reduced I1-B (§2, arms 1–6) plus I4.

Estimate: 4 runs × ~1.5 GPU-h ≈ 6 GPU-h.

## 2. RQ1 / I1 — PCD vs state-of-the-art KD
Arms, all on the same student, teacher and schedule:

| # | Arm | Family | Tuned on validation |
|---|---|---|---|
| 1 | KD (Hinton) | loss baseline | T |
| 2 | DKD | loss state of the art | β ∈ 4 values [VERIFY default α = 1] |
| 3 | KD + DOT | optimizer state of the art | DOT momentum gap, 4 values [VERIFY] |
| 4 | KD + LS (LSKD) | loss state of the art | as published |
| 5 | **PCD(CE \| KD)** | ours | τ ∈ {0.01, 0.05, 0.1, 0.2} |
| 6 | **Normalised-WS control**: EMA-normalised CE + λ·KD with PCD's final rescale | decisive control | λ grid of 4 that includes the pilot's mean τ − c̄ |
| 7 | **PCD(CE \| TCKD, NCKD)** K=3 | ours ("tuning-free DKD") | τ ∈ 4 values |
| 8 | PCGrad / CAGrad / MGDA on (CE, TCKD, NCKD) | symmetric multi-objective | CAGrad c ∈ {0.2, 0.5} |
| 9 | **PCD + LS** (LSKD loss as the secondary) | plug-in | τ ∈ 4 values |
| 10 | DKD + LS | loss state of the art | as published |

**Hypotheses (preregistered):**
- **H1, non-inferiority:** arm 7 with a *single τ fixed across P1–P3* is within δ of DKD with β tuned per pair.
- **H2, mechanism:** arm 5 vs arm 6 is equivalent (TOST) when the pilot's active rate is ≥ 95%. A difference is expected only when the conflict varies.
- **H3:** arm 5 vs DOT (arm 3); two-sided, exploratory.
- **H4:** arm 9 > arm 4 (PCD adds to LSKD).

Runs:
- selection ≈ 22 per pair;
- final 10 arms × 5 seeds = 50 per pair (arm 8 counts as one row of three methods; budget it as 3 arms);
- about 3 × 80 ≈ 240 runs in total.

Estimate:

| Arm type | GPU-h per run |
|---|---|
| KD, DKD, LS | ~1.0 |
| K=2 multi-objective (two backward passes) | ~1.6 |
| K=3 multi-objective | ~2.2 |

**About 350–450 GPU-h for all three pairs.**
- **I1-A, reduced (P1 + P3 only):** ≈ 250 GPU-h.
- **I1-B, minimal (P1 only):** ≈ 130 GPU-h.

Optional ImageNet with ResNet34 → ResNet18 (⛔ ImageNet licence and download; 100 epochs, roughly 8 GPU-days per run, ESTIMATE): one seed each of KD, DOT, PCD(CE | KD) and the normalised-WS control, as an external-validity check only.

## 3. RQ2 / I4 — reversed priority under unreliable teachers
- **Setting:** P1 (ResNet32x4 → ResNet8x4).
- **Teacher outputs:**
  - (a) natural, with the teacher's own test errors kept;
  - (b) 30% training-index corruption (Stage 1 `pcdkd.data.corruption`, log-space, nested);
  - (c) 0% as the control.
- **Arms:**
  1. KD;
  2. RLD (official code `zju-SWJ/RLD`) [VERIFY licence];
  3. tuned NWKD (normalised WS) whose λ grid includes high CE weights;
  4. PCD(CE | KD), CE primary (the former objective S);
  5. **PCD(KD | CE)**, KD primary (reversed);
  6. gated NWKD (correctness + confidence gate, Stage 1 `pcdkd.losses.gate`);
  7. gated reversed PCD.
- **Hypotheses:**
  - **H5:** under 30% corruption, arm 5 > arm 4.
  - **H6:** under 30% corruption, arm 5 vs arm 2 (RLD); two-sided, exploratory.
  - **H7:** at 0% corruption, arm 5 is non-inferior to KD within δ.
- **Mechanism check:** the implicit CE weight λ_t on batches with a high corrupted fraction vs a low one. This needs per-batch corruption-fraction logging.
- **Runs:** 7 arms × 3 conditions × 5 seeds, plus selection, ≈ 120 runs. **Estimate ≈ 160 GPU-h.**

## 4. Total budget (ESTIMATE, before the pilot)
| Package | GPU-h |
|---|---|
| Pilot | ~6 |
| I1-B + I4 (minimum viable paper) | ~300 |
| I1-A + I4 | ~420 |
| Full I1 + I4 | ~570–620 |

## 5. Code reuse and new code
- **Reused from Stage 1:**
  - combiners (wkd, nwkd, pcgrad_onto_primary, pcd, adaptive_lambda);
  - the PCD QP (K ≤ MAX_K), EMA normaliser and diagnostics;
  - corruption, gate, splits and config.
- **New code:**
  - mdistiller-compatible trainer, or porting PCD into mdistiller as a "distiller + trainer" pair (RECOMMENDATION: port into mdistiller, so that DOT, DKD and the teachers are byte-identical to the published code);
  - TCKD/NCKD loss terms;
  - CAGrad/MGDA (K=3);
  - LS standardisation;
  - an RLD adapter;
  - removal of the CPU-side Gram matrix sync on CUDA (Stage 1 review finding 1).
- **Review:** every new component gets Claude↔Codex cross-review and a synthetic CPU test before any GPU run (roadmap Stage 2 rules).
