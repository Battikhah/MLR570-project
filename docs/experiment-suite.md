# PCD-in-KD experiment suite

Goal: find **where PCD helps knowledge distillation, and where it hurts**, before spending the
confirmatory budget. The design lives in `src/pcdkd/experiments/suite.py`, the trainer in
`src/pcdkd/train/trainer.py`, the runner in `scripts/run_suite.py`, and the report in
`scripts/analyze_suite.py`.

Nothing here is a result. Every family states a prediction (from PCD's algebra or prior
evidence) and carries the controls that could refute it. Labels follow `AGENTS.md`.

## How to run on the GPU server

```bash
uv sync --frozen
uv run python scripts/fetch_cifar.py --yes            # ⛔ needs approval (CIFAR-10/100, ~324 MB)
uv run pytest -m "not slow and not requires_data"     # all green before any GPU time

# 1. Teachers (14 in total; trained once and cached under outputs/suite/<profile>/teachers/)
uv run python scripts/run_suite.py run --profile screen --stage teachers

# 2. Tier 0: pilot + MVE. Measures seconds per epoch for each arm and checks the go/no-go rule
uv run python scripts/run_suite.py run --profile screen --tier 0
uv run python scripts/analyze_suite.py --profile screen

# 3. Tier 1 screen, sharded: one process per shard. ResNet-8 barely loads a GPU, so 3-4 shards per GPU
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$((i % 2)) uv run python scripts/run_suite.py run --profile screen \
      --tier 1 --shard $i/4 --stage students > logs/shard$i.log 2>&1 &
done

# 4. Read outputs/suite/screen/REPORT.md. Then run --profile full only on the families or cells
#    the screen flags, e.g.:
uv run python scripts/run_suite.py run --profile full --family label-noise --ids sym0.4 sym0.6
```

- **Restarts are safe:** finished runs (`result.json`) are skipped, and a run in progress resumes
  from its last checkpoint (`checkpoint.pt`, every `--ckpt-every` epochs, default 5).
- **Failures don't stop the sweep:** a failed run writes `error.json` and the sweep continues.
- **Previews:** `list` prints run counts and cost, and `--dry-run` prints the run IDs without training.

## Running on the AUS workstation (ESB-1043-20)

The workstation is reached through GlobalProtect and RDP, and its rules shape how the suite runs:

| Rule | What the suite does |
|---|---|
| Use only Thursday 18:00 to Monday 08:00 (UAE time); use outside it is monitored | `--window aus` refuses to start outside the window and, after each epoch, pauses with a checkpoint once fewer than 20 min (`--stop-margin-min`) plus 1.5 epochs remain. It also starts no new run in that period. |
| DeepFreeze wipes local data on any restart, power cut or crash | Runs write to the local disk (`$PCDKD_OUT`), and rclone copies results, teacher weights, run checkpoints and logs to Google Drive, `My Drive/pcd-suite/`, every 5 minutes (`-SyncMinutes`); each session first pulls that copy back, so runs resume. No Drive app is needed: rclone is a portable exe and signs in through the browser (scope `drive.file`, so it sees only its own files). The code is a git clone of this repo, pulled each session and re-cloned after a wipe; the Python environment and CIFAR are rebuilt locally. Each `result.json` records the commit it ran. A wipe loses at most the last upload interval plus `--ckpt-every` epochs of the run in progress. |
| No admin rights, Windows | uv installs per user. The CUDA build of the locked torch version is installed from the PyTorch index, because PyPI's Windows wheels are CPU-only. The run keeps Windows from sleeping while it works. |

Steps:

1. **Code changes:** push to `main` on GitHub. The workstation pulls it at the start of each session.
2. **On the workstation, first window (and after any wipe):** run
   `git clone https://github.com/Battikhah/MLR570-project "$env:LOCALAPPDATA\pcd-suite\code"` (sign in
   to GitHub when Git asks; the repo is private). From `scripts\workstation` in that clone, test access
   without training: `powershell -ExecutionPolicy Bypass -File run_weekend.ps1 -SetupOnly`. It signs rclone in to Google (browser), restores earlier results from Drive, pulls,
   checks that CUDA works, runs the CPU tests, and downloads CIFAR (⛔ the first CIFAR download needs approval).
3. **Each window:** `run_weekend.ps1` (default: whole screen profile, 3 processes). It clears stale
   claims, then starts `-Procs` workers of `run_suite.py run --queue`. They share one priority queue:
   seed 0 of every family before any second seed, families ordered by `suite.PRIORITY` (most
   revealing first), each teacher trained just before its first user. Whatever finishes by Monday is
   the most informative subset; rerunning next window continues. `-Family "label-noise ..."` narrows it.
4. **When leaving:** keep the PowerShell window open (it does the uploads) and disconnect the RDP window. Do not sign out, because signing out ends the run.
5. **Afterwards, on the Mac:** Google Drive's `My Drive/pcd-suite/outputs/<profile>/` holds every `result.json`;
   download it (or `rclone copy gdrive:pcd-suite/outputs <dir>`) and point `scripts/analyze_suite.py --out` at it.

**Profiles** change the schedule only; the design is the same:
- `full`: 240 epochs, the family's seed count.
- `screen`: 60 epochs, at most 2 seeds.
- `smoke`: synthetic data, 2 steps, on CPU. The tests use this profile.

## Protocol shared by every run

- **Data:** CIFAR-100 (CIFAR-10 only for the MVE), 45k train / 5k stratified validation (`split_seed` 0). The test set is evaluated once, at the end. The analysis selects configurations on validation only.
- **Optimiser and schedule** (mdistiller CIFAR-100 protocol): SGD, momentum 0.9, weight decay 5e-4, lr 0.05, batch 64, ×0.1 at 62.5 / 75 / 87.5% of training, KD temperature T = 4.
- **Pairs:**
  - P0: ResNet-32 → ResNet-8, the submitted proposal's pair; cheap.
  - P1: ResNet32x4 → ResNet8x4, the mdistiller / DTO-KD reference pair.
- **Teachers:** we train our own with CE, seed 0. The mdistiller checkpoints are not used, so absolute numbers will not match the published tables.
- **Seeds:** a seed fixes initialisation, data order and the teacher-corruption map, and is identical across arms. Every comparison is therefore paired.
- **Fixed per dataset:** label noise, the long-tail subsample and the low-data subsample are set by `split_seed`. Teacher and student see the same noisy labels.
- **Logged every 50 steps:** cos(g_CE, g_KD); the EMA-normalised inner product; PCD active rate, μ and the actual Prop. 4.7 margin; λ_eff; alignment of the update with each gradient; per-objective losses; gate acceptance (overall and on corrupted samples); and *shadow cosines*. Shadow cosines measure how far apart the CE, PCD, NWKD, WKD and DTO-KD directions are on the same batch, computed from the 2×2 Gram matrix at no extra backward cost. Each run also gets diagnostics summarised by thirds of training.
- **Metrics:** top-1/top-5, NLL, ECE (15 bins), per-class accuracy, many/medium/few-shot accuracy (long tail), and student–teacher agreement on the test set.

## Arms

| Arm | Rule | Why it is there |
|---|---|---|
| `ce` | CE only | negative-transfer reference |
| `wkd_l*` | (1−λ)CE + λKD, one backward | standard KD (λ = 0.9 is the mdistiller default) |
| `nwkd_l*` | g̃_CE + λ g̃_KD, rescaled to ‖g_CE‖ | **the decisive control**: same EMA normalisation and rescale as PCD, but a fixed coefficient |
| `pcd_t*` | PCD(CE \| KD), τ | the method |
| `pcdrev_t*` | PCD(KD \| CE), τ | priority-order ablation: teacher as anchor |
| `pcgrad1s` | project KD off CE when they conflict | "protect the primary" baseline |
| `pcgrad_sym` | Yu et al. gradient surgery (K ≥ 2) | symmetric conflict removal |
| `mgda` | min-norm point of the raw gradients (AE-KD-style for multi-teacher) | symmetric balancing |
| `dtokd` | min-norm point of the log-loss gradients (DTO-KD, Eqs. 11–12, exact) | the symmetric state of the art in optimizer-side KD |
| `dot_d0.075` | separate momenta, β ∓ δ (DOT) | the momentum-side state of the art in optimizer-side KD |
| `g_*` | the same rule with the correctness + confidence gate | the proposal's gate |
| `sum_*`, `nws_*`, `dkd_*` | K-objective raw / normalised sums | K = 3–4 grids |

## Families: what "PCD excels" would look like

Tier 0 runs first, then tier 1. Tiers 2–3 broaden coverage.

| Tier | Family | Probe | PCD predicted to… | Basis |
|---|---|---|---|---|
| 0 | `pilot` | P1: CE, WKD, DTO-KD, PCD(0.5) | measure cost, the c distribution, the active rate, and whether the rules differ on ≥ 10% of steps (plan §3 go/no-go) | plan rev. 2 |
| 0 | `mve-cifar10` | ρ ∈ {0, 0.3} | show a negative shift in c and a non-trivial active rate (decision 0001, H0) | decision 0001 |
| 1 | `core-p1`, `core-p0` | all 17 two-objective arms, clean | match or beat NWKD only if c varies within training; tie with NWKD if c is steady | Cor. 4.6; I1* |
| 1 | `loss-scale` | L_KD × m, m ∈ 10^{−2..2} | **excel**: direction invariant to m (EMA normalisation), while raw WKD/PCGrad degrade. DTO-KD and NWKD are also invariant, so the win is over *raw* KD only | PCD §5.1.1: its wins come from scale; `test_pcd_mechanism.py` |
| 1 | `teacher-corruption` | ρ ∈ {0, 0.1, 0.3, 0.5, 0.7}, ± gate | CE-primary **lose** at high ρ (its KD weight max(0, τ−c) grows with conflict); the gate and KD-primary are the contrasts | decision 0001, Prop. 4.7 |
| 1 | `label-noise` | η ∈ {0, 0.2, …, 0.8} symmetric + asymmetric, clean teacher | **excel with `pcdrev`**: the reliable teacher is the anchor and noisy CE gets only τ-progress. Most promising regime (HYPOTHESIS) | priority order; `test_priority_order…` |
| 1 | `low-data` | 1–100% of labels, iteration-matched | excel if few labels create strong, varying conflict | DHO measured c ≈ −0.4…−0.76 few-shot |
| 2 | `long-tail` | imbalance factor 1–200 (teacher trained on the same long-tailed set) | help on few-shot classes, if per-batch class mix drives c | INTERPRETATION |
| 2 | `capacity-gap` | teachers ResNet-14 … ResNet32x4 → ResNet-8 | help when the KD target is hard to match | INTERPRETATION |
| 2 | `multi-loss-at` | CE + KD + AT, clean and ρ = 0.3 | **excel on tuning cost**: one τ vs a 2-D weight grid | K ≥ 3 QP; `test_k3…` |
| 2 | `multi-loss-dkd` | CE + TCKD + NCKD vs DKD's α/β | match DKD without hand-set β | C2 in the mechanism note |
| 2 | `multi-teacher` | 3 teachers, one corrupted at 50% | either be outvoted by the good teachers or follow the bad one (its constraint grows with conflict); MGDA ≈ AE-KD is the symmetric rival | C4 |
| 2 | `temperature` | T ∈ {1, …, 16} | flatter T-curve than raw WKD | T² scale |
| 2 | `tau-sweep` | 11 τ values × both orders × ρ ∈ {0, 0.3}, plus CE/WKD/NWKD references | CE-only behaviour at τ → 0 when aligned (H0′) | Cor. 4.3 |
| 2 | `sensitivity` | equal 8-point grids: WKD λ, NWKD λ, PCD τ, plus a 4-point grid for PCD-rev τ (clean, ρ = 0.3, η = 0.4) | **excel on worst case and spread** even at equal best accuracy | usability claim |
| 3 | `label-noise-noisy-teacher` | teacher trained on the same noisy labels | test whether KD priority survives a teacher that memorised the noise | realism check |
| 3 | `pcd-ablations` | EMA β, rescale off, momentum 0, batch 32–256 | identify which part of PCD matters | plan §7 |

**Mechanism caveats the analysis must respect** (proved in `tests/unit/test_pcd_mechanism.py`):
- With unit-normalised gradients and τ − c = 1, PCD(CE | KD) and PCD(KD | CE) give the **same direction**. The priority order matters only away from that line.
- With c ≥ τ, PCD(CE | KD) **drops KD entirely** and takes the pure CE step.
- At c = −1 the CE margin is −τ (Prop. 4.7): the primary objective moves backwards.
- Scale invariance holds only while ‖g‖² ≫ ε = 1e-8, so check the logged `norm_secondary` in the `loss-scale` family.

## Size and cost

| Profile | Student runs | Full-dataset epoch-passes | Teachers |
|---|---|---|---|
| screen, tiers 0–1 | 656 | 72,240 | 3 |
| screen, all tiers | 1,392 | 150,011 | 14 |
| full, tiers 0–1 | 1,050 | 462,480 | 3 |
| full, all tiers | 2,004 | 864,065 | 14 |

An epoch-pass is one pass over 45k images with one student backward. GPU-hours = epoch-passes × the
pilot's measured seconds per epoch-pass for each pair. ESTIMATE, unmeasured: at about 10 s per
ResNet-8 epoch-pass, screen tiers 0–1 come to roughly 200 GPU-hours before running several shards
per GPU. Run the pilot first and re-plan from its numbers.

## Analysis (`scripts/analyze_suite.py`)

Writes `REPORT.md` and `runs.csv` under `outputs/suite/<profile>/`.

- **"Where does PCD win?" table:** for each family × condition, every method group gets its best configuration on validation. The best PCD group is then compared with the best non-PCD group on test accuracy, using seed-paired differences with a bootstrap 95% CI, plus PCD − NWKD, mean c, the active rate, and the fraction of steps where PCD and NWKD point > 8° apart.
- **Tuning-robustness table:** best / median / worst accuracy and the spread over each group's grid.
- **Per-cell detail:** every arm, with c by thirds of training (non-stationarity), λ_eff, and the fraction of steps with a non-positive CE margin.

Verdicts from the screen are leads, not findings. Confirm them with `--profile full` and the
pre-registered statistics in `refine-logs/EXPERIMENT_PLAN.md` (Holm correction, 90% TOST, δ = 0.5 pp).

## Known limitations

- **DTO-KD:** the exact rule only, without its feature projection or separate heads, and no amortised variant.
- **DOT:** reimplemented as two heavy-ball buffers on flattened gradients, not mdistiller's optimizer. Weight decay enters the task buffer.
- **Teachers:** trained here, so the absolute numbers are not comparable with mdistiller's tables.
- **Scale:** CIFAR only. Robust distillation (C1) and DHO/ImageNet (C3) are out of scope because of compute and data access.
- **Precision:** no mixed precision, because the Gram matrix is computed in float64 on the CPU.
