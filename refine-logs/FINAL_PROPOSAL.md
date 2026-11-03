# FINAL PROPOSAL (draft, pending cross-review): Priority-Constrained Distillation

**Title (working):** *Which objective comes first? Priority-constrained descent against state-of-the-art knowledge distillation.*

## Problem
- KD trains a student on a task loss (CE) plus distillation losses whose gradients can differ in scale and direction.
- **What is published:**
  - State-of-the-art gains come mostly from loss design (DKD, MLKD, LSKD, RLD).
  - Optimizer-side fixes are fewer: DOT's momentum split, MEMA's penalty, and MGDA for ensembles. MoKD is withdrawn.
- **Two gaps:**
  - No study has measured how much CE and KD actually conflict in standard full-data KD. Measured conflict exists only in VLM few-shot distillation (DHO) and in adversarial training (CA-AT).
  - No study has tested a *priority* rule (one objective first, the others guaranteed minimal progress) on KD.

## Method
Use PCD (Varam & AlHajri, TMLR 2026) as the gradient combiner in KD, in two priority orders:
- **(a)** CE primary, distillation terms as secondaries. K=2 with KD, or K=3 with DKD's TCKD/NCKD, aiming for a single-τ, scale-invariant replacement for DKD's weight tuning.
- **(b)** KD primary, CE as a progress constraint. Theory (PCD Cor. 4.6) gives a CE weight max(0, τ − c) that grows only when the teacher contradicts the label.

## Contributions (all conditional on results; no claim of improvement is made in advance)
1. The first per-step measurement of CE–KD and TCKD–NCKD gradient conflict on the standard CIFAR-100 KD benchmark.
2. A fair, preregistered comparison of PCD with the optimizer-side state of the art (DOT) and symmetric multi-objective methods. It includes a normalisation-only control that separates projection effects from scale effects.
3. PCD as a plug-in on DKD and LSKD.
4. A reversed-priority variant for unreliable teachers, compared with target-repair (RLD) and gating.

## Expected outcomes (HYPOTHESES)
- PCD ≈ a normalised weighted sum wherever conflict is weak, so any gain over KD comes mainly from normalisation.
- K=3 PCD matches tuned DKD without per-pair tuning.
- The reversed priority beats CE-primary PCD under corruption.

Null results are reported as such.

## Relation to the course deliverables
- The survey gains a section contrasting optimizer-side and loss-side KD, with PCD and FedPCD positioned against DOT, MoKD, MEMA, AE-KD and CA-AT.
- The project keeps the submitted unreliable-teacher question (RQ2) and adds a state-of-the-art comparison (RQ1).
- This supersedes decision 0001 (objective S), which needs a new decision 0002 and instructor acknowledgement.

## Plan and budget
See `refine-logs/EXPERIMENT_PLAN.md`:
- a pilot of about 6 GPU-h;
- a minimum viable paper of about 300 GPU-h (estimate);
- all GPU work, downloads and checkpoint use need approval.
