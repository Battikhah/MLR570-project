# Decision 0001 — Research objective

- **Status:** Proposed (coordinator). Needs team / instructor acknowledgement because it narrows the submitted proposal.
- **Date:** 2026-11-02
- **Evidence:** Orca Run `run_f9e2e8953f54`: Claude literature task, Codex math/feasibility task, two rounds of
  adversarial cross-review (`docs/reviews/`). Ranked comparison: `research/directions/ranked-comparison.md`.
  Key arXiv records (PCD 2606.29521, FedPCD 2610.13508, DHO 2505.07675, TPKD 2609.39338, CHAD 2607.19956,
  GAQAT 2412.05551) were additionally re-checked by the coordinator against the arXiv API on 2026-11-02.

## Context

The submitted MLR570 proposal asks whether reliability-gated PCD improves student accuracy and reduces negative
transfer when teacher predictions are unreliable (CIFAR-100, ResNet-32 teacher, ResNet-8 student).
The investigation established (VERIFIED unless labelled):

1. **PCD** (Varam & AlHajri, TMLR 2026) projects the normalized primary gradient onto secondary-progress half-spaces.
   It guarantees per-step directional properties only; there is **no convergence theorem** for the deployed
   algorithm (PCD §4.7). Secondaries are assumed convex; KD-KL is not.
2. In the active two-objective case PCD is exactly an **adaptive normalized weighted sum**:
   `d̃ = a + α b`, `α = τ − aᵀb/‖b‖²` (PCD Cor. 4.6; derived independently by Codex; checked numerically by Claude
   against the official `pcd/qp.py`, max error 4.4e-16).
3. PCD's empirical wins are on compression, where the authors attribute them to **objective scale, not gradient
   conflict** (cosine ≤ 1.2e-3, §5.1.1; App. E.3 Table 10). Against tuned baselines: 44 wins / 4 losses / 54 ties
   out of 102 (Table 3).
4. The official repository (MIT, HEAD `e9afdbc`) contains PCD only: no baselines and no physical-pruning export.
   Its README states that the published runs used a per-step proximal step not disclosed in the paper.
5. **FedPCD** (arXiv preprint, under review) reports favourable federated results but **code is not publicly released**
   (as of 2026-11-02) and drift-aware baselines (FedProx/SCAFFOLD/FedNova) are absent.
6. Measured, materially negative CE–KD gradient cosines exist only in DHO (few-shot, CLIP teacher; −0.39 to −0.45
   backbone, −0.76 head; single-agent verified). **No source measures this for full-data CIFAR-100 or under teacher
   corruption.**

## Key theoretical consequence (THEORETICAL GUARANTEE, from Cor. 4.6)

With unit-normalized gradients and an active constraint, PCD's implicit KD weight is `λ_t = max(0, τ − c_t)`.
It **grows as the teacher conflicts more with the labels**. PCD protects *progress on the secondary* (KD); it protects
the primary (CE) only through the anchor, and Prop. 4.7 allows CE ascent when `1 + τc − c² ≤ 0`.
Therefore the intuition "PCD protects the student from an unreliable teacher" is **opposite to PCD's mechanism**
in the ungated setting. Separately, the proposal's correctness gate removes every high-confidence-wrong teacher
output by construction, so in gated arms the corruption only reduces the number of KD examples.

## Decision

Adopt **objective S (amended)** as the project's research objective; keep **M1** as backup.

**Research question.** In CE-primary / KD-secondary distillation on CIFAR-100, does PCD's conflict-adaptive KD
coefficient improve, match, or degrade student accuracy and negative transfer relative to a budget-matched, tuned,
EMA-normalized weighted KD, as teacher reliability is degraded — and does a correctness+confidence reliability gate
change that comparison?

Mandatory amendments agreed by both agents:
1. The control is a **budget-matched tuned** EMA-normalized weighted KD whose λ grid contains the pilot-measured
   `τ − c̄`, with PCD's final norm rescaling applied identically — not "λ matched to τ" (a strawman when `c̄ ≠ 0`).
2. Teacher corruption is a **stressor on the ungated arms** for the mechanism test; gated arms answer a separate
   gate-effect question.

The full design is in `research/experiment-plans/pcd-kd-conflict.md`.

## Gating rule (go / no-go)

The confirmatory study proceeds only if the minimum viable experiment shows (i) a measurable shift toward negative
CE–KD cosine under 30% corruption in ungated training and (ii) a non-trivial PCD active-constraint rate, and a smoke test
shows throughput compatible with the run budget. Otherwise: report the null/precondition failure honestly and switch
to backup M1 (or a reduced S at {0, 30%} corruption and 3 seeds if only throughput fails).

## Consequences

- The proposal's research question is retained; its method matrix shrinks (DTO-KD, SCKD, SCOPE re-implementations
  dropped: no verified reference code; PCGrad retained as the "remove conflicting KD component" baseline).
- The expected headline is plausibly *negative or null for PCD*. That is acceptable scientifically and must be
  reported as such; measurements, metrics, or protocols must not be changed after seeing results.
- Requires authorization before execution: CIFAR-10/100 download (163 MB + 161 MB), teacher training, pilot runs.
