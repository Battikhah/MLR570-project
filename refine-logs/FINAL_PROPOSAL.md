# FINAL PROPOSAL rev. 2 (pending cycle-2 review): Priority vs symmetry in knowledge distillation

**Working title:** *Should the label come first? Priority-constrained descent versus symmetric trade-off optimisation in knowledge distillation.*

## Problem
- KD trains a student on a task loss (CE) and a distillation loss whose gradients can differ in scale and direction.
- **The optimizer-side state of the art treats the two symmetrically:**
  - **DTO-KD** (ICLR 2026): min-norm weighting of log-loss gradients with equal first-order contribution to both (Cor. 3.3).
  - **DOT** (ICCV 2023): a separate, larger momentum for the KD gradient.
- **PCD** (TMLR 2026) argues that objectives are not equal. It keeps a fixed primary and guarantees each secondary only τ-progress. It has never been tested on KD.
- No study reports how much CE and KD actually conflict per step on the standard CIFAR-100 benchmark.

## Research question
On an identical CIFAR-100 protocol (ResNet32x4 → ResNet8x4), does a fixed-priority rule (CE first, KD guaranteed τ-progress) give a better, equal, or worse student than:
- the symmetric rule (DTO-KD);
- the momentum split (DOT);
- a budget-matched normalised weighted sum?

How often do these rules actually produce different update directions?

## Contributions (conditional on results; no improvement is claimed in advance)
1. A controlled comparison of combination rules on one loss, teacher, schedule and data order, with 10 paired seeds, a Holm-corrected confirmatory family, and a preregistered equivalence margin as a secondary analysis.
2. The first per-step measurements, on this benchmark, of CE–KD conflict, PCD's active rate, DTO-KD's weights, and the shadow-direction agreement between rules.
3. A theory-derived prediction, H0′, tested explicitly: with CE primary and small τ, PCD reduces toward CE-only training (PCD Cor. 4.3).
4. Exploratory: PCD as a plug-in on a loss-side method (LS).

## Explicit non-claims
- No claim of a new absolute state of the art. Loss-side methods such as MLKD+LS (78.28) are outside the comparison family.
- DTO-KD is reimplemented without its feature projection or separate heads; the comparison is of combination rules only.
- The unreliable-teacher study (I4) is a separately approved follow-up and an ablation of priority order. PCD's coefficient is batch-level and does not identify corrupted examples.

## Course fit
- **Survey:** gains a section contrasting optimizer-side and loss-side KD, with PCD, FedPCD, DTO-KD, DOT, MEMA, CA-AT and DHO.
- **Project:** keeps gradient conflict central, and keeps the proposal's unreliable-teacher question as the follow-up.
- **Decision:** supersedes decision 0001, which needs a decision 0002 and instructor acknowledgement.

## Budget
- A pilot of about 8–14 GPU-h.
- About 120–240 GPU-h in total (ESTIMATE, `EXPERIMENT_PLAN.md` §4).
- Every download and GPU run requires approval.
