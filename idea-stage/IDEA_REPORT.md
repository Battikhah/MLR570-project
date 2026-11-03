# IDEA REPORT — PCD in knowledge distillation vs. state-of-the-art KD (Phase 7)

- **Run:** ORCA `run_8e2818695e01`; ARIS state `kd-pcd-sota-20261103` (Stage 1 / idea discovery only).
- **Date:** 2026-11-03. **Executor:** Claude Opus 5.5 (coordinator literature track).
- **Revision 2 (after the Codex review, cycle 1):** DTO-KD added; the I4 mechanism wording corrected; scope narrowed; DOT is no longer called the optimizer-side state of the art. See §External critical review.
- **Independent track:** Codex, ORCA task `task_15a692ceefea`, `research/notes/kd-pcd-mechanism-codex.md` (mechanism math and GPU feasibility).
- **Scope limits:**
  - No training or downloads were run.
  - The project's approval gates still apply. The CIFAR, ImageNet and checkpoint downloads and the GPU runs need human approval.
  - ARIS Stages 2–3 (deploying experiments and an autonomous review loop) are **not** run. The user's standing instruction forbids unattended experiment loops and a second orchestration system.
- **Labels follow `AGENTS.md`:**
  - **VERIFIED**: checked against the primary page named.
  - **EE**: existing empirical evidence, as reported by the authors.
  - **TG**: theoretical guarantee.
  - **INTERPRETATION**.
  - **HYPOTHESIS**: untested.
  - **RECOMMENDATION**.
  - "search-only" means the claim was seen only in a search snippet and is unverified.

## Literature landscape

Summary: state-of-the-art KD gains come mainly from loss and target design (DKD, MLKD, LSKD, RLD). PCD belongs to the smaller optimizer-side family (DTO-KD, DOT, MEMA, AE-KD; CA-AT outside KD), where DTO-KD's symmetric min-norm rule is the direct counterpart to PCD's fixed priority. Measured CE–KD conflict exists only in DHO's few-shot VLM setting, and no measurement exists for standard CIFAR-100 KD.

### L0. Where PCD sits
PCD is an **optimizer-side combiner**. Given the gradients of the primary and the secondaries, it returns one update direction:
- EMA normalisation of each gradient;
- projection of g̃₁ onto the half-spaces g̃ⱼᵀd ≥ τ‖g̃ⱼ‖²;
- a final rescale to ‖g₁‖.

It does **not** change the KD target or loss (VERIFIED, `literature/notes/varam2026pcd.md`). State-of-the-art KD splits into two families that PCD relates to differently:

| Family | Changes… | Representative methods | Relation to PCD |
|---|---|---|---|
| **Loss / target design** | what the student imitates | KD (Hinton 2015); DKD (CVPR 2022); CTKD (AAAI 2023); MLKD (CVPR 2023); LSKD logit standardisation (CVPR 2024); RLD refined logits (ICCV 2025); feature KD: FitNets, AT, CRD, ReviewKD | **Orthogonal.** PCD can be stacked on any of them by making their loss a secondary. |
| **Optimizer / loss balancing** | how the task and distillation gradients are combined | **DTO-KD (ICLR 2026)**; DOT (ICCV 2023); MEMA in SPOFA (preprint 2026); MoKD (withdrawn, superseded by DTO-KD); AE-KD (NeurIPS 2020, multi-teacher MGDA); learnable α / AKD (2025, search-only); RW-KD (EMNLP-F 2021, search-only); CHAD gating (2026); CA-AT (adversarial training, not KD) | **Direct competitors.** PCD's claim must be judged inside this family. |

**INTERPRETATION (central):** "PCD vs state-of-the-art KD" is a category error unless the comparison is split.
1. Within the optimizer family: PCD vs DTO-KD and DOT (plus MGDA/PCGrad/CAGrad and tuned normalised weighted sums), all on the same KD loss.
2. As a plug-in: X vs PCD+X for the strongest loss-design X (LSKD, MLKD, DKD).

A PCD student with vanilla KD loss is not expected to beat MLKD+LS. The loss-design gains on the reference pair are much larger than anything an optimizer change has reported (L1).

### L1. Benchmark anchor: CIFAR-100, ResNet32x4 → ResNet8x4 (VERIFIED from official repos)
| Method | Top-1 | Family | Source |
|---|---|---|---|
| KD | 73.33 | loss | mdistiller README; LSKD README |
| KD + DOT | 75.12 | **optimizer** | mdistiller README (DOT is the official implementation, MIT) |
| DKD | 76.32 | loss | mdistiller README |
| DTO-KD | 76.40 | **optimizer** | DTO-KD paper Table 2 (that table lists DKD 75.44, not 76.32: cross-paper numbers differ) |
| KD + LS | 76.62 | loss | LSKD README |
| DKD + LS | 77.01 | loss | LSKD README |
| MLKD | 77.08 | loss | LSKD README |
| MLKD + LS | **78.28** | loss | LSKD README |

On ImageNet with ResNet34 → ResNet18: KD 71.03, DKD 71.70, KD+DOT 71.72 (VERIFIED, mdistiller README).

**INTERPRETATION (descriptive, cross-paper table differences, not controlled estimates):** optimizer-side methods report KD+DOT 75.12 and DTO-KD 76.40 on this pair. Loss-side methods reach 76.3 to 78.3. Realistically, PCD is a candidate to match or beat DTO-KD/DOT on an identical protocol and to add on top of a loss-side method; it is not a candidate to set a new absolute record.

### L2. Optimizer-side KD methods: the direct competitors
- **DTO-KD** (Hayder, Cheraghian, Petersson, Harandi, Hartley; **ICLR 2026**; VERIFIED full PDF; no code link).
  - The closest prior work. KD is framed as a two-objective problem against "gradient conflict" and "gradient dominance".
  - Following FAMO (Liu et al. 2023), the update is g* = π₁∇log L_distill + π₂∇log L_task, with π the **min-norm point on the simplex**. Closed form (Eqs. 11–12): π₁* = (g₂₂ − g₁₂)/(g₁₁ + g₂₂ − 2g₁₂), where g is the Gram matrix of the log-loss gradients.
  - Properties: g* aligns with both objectives (Cor. 3.2) and has **equal inner products with both log-loss gradients** (Cor. 3.3: equal first-order contribution, not equal finite-step loss reduction). It is symmetric MGDA-style balancing. Scale handling comes from the log-loss (each gradient divided by its loss value).
  - An amortised variant (Eq. 18) updates π without a second backward pass.
  - Results: CIFAR-100 ResNet32x4 → ResNet8x4 76.40 (Table 2), plus DeiT/ImageNet and ViDT/COCO.
  - Reimplementable from Eqs. 11–12 and Alg. 1.
  - **Relation:** DTO-KD is the symmetric counterpart of PCD. PCD pins the primary coefficient at 1 and guarantees only τ-progress on the secondary; DTO-KD equalises first-order contribution on the log-losses. *Priority vs symmetry* in KD is the open, testable question (the PCD paper's "symmetry problem", now with a published KD instance).
  - Lineage: MoKD (withdrawn 2025-08-02, "major changes to the paper and the authorship") has the same first four authors. INTERPRETATION: DTO-KD supersedes it.
- **DOT** (Zhao, Cui, Song, Liang; ICCV 2023; arXiv 2307.08436; VERIFIED abs).
  - It identifies a trade-off: adding the distillation loss slows the task loss.
  - It computes the two gradients separately and gives the distillation gradient a larger momentum. The supplement says this applies only to parameters both losses touch (search-only).
  - Code is in mdistiller.
  - **Relation:** a mandatory optimizer-side baseline that works through momentum, not projection. (Revision 2: it is *not* the closest state of the art. That is DTO-KD, below.)
- **MEMA / SPOFA** (Yang, Zhang, Zhao; arXiv 2606.24557, 2026-06-23, "under review"; VERIFIED abs).
  - For heterogeneous KD (Transformer → CNN), it judges task–distillation conflict against an EMA baseline and "adaptively penalizes harmful distillation signals".
  - The exact rule is not in the abstract, and no code is linked.
  - **Relation:** close in spirit (task first, KD suppressed when it conflicts). It is the opposite of PCD, which *guarantees* KD progress.
- **MoKD** (Hayder et al.; arXiv 2505.08170; VERIFIED abs).
  - It framed KD as multi-objective optimisation against "gradient conflicts" and "gradient dominance".
  - **It was withdrawn on 2025-08-02** ("major changes to the paper and the authorship"), and no PDF is available.
  - **Relation:** evidence that the framing is considered publishable. It cannot be used as a baseline.
- **AE-KD** (Du et al., NeurIPS 2020; proceedings URL seen in search results, not opened; `sources.yaml` status unverified): ensemble or multi-teacher KD as MGDA with a tolerance for teacher disagreement. It is symmetric, with no primary.
- **CHAD** (arXiv 2607.19956, already in `sources.yaml`): per-sample KD gating by cos(∇L_val, ∇L_KD). In its Bangla-summarisation setting, 51.3% of samples have s ≤ 0 (EE).
- **Loss-weight learning:** AKD (learnable α, 2025), RW-KD (meta-learned per-sample weights), AdaKD (instance weights). All search-only; none verified.

### L3. Evidence of real CE–KD or task–KD gradient conflict
| Setting | Measured conflict | Source |
|---|---|---|
| CLIP RN50 zero-shot teacher → ResNet-18, 16-shot, 8 datasets | cos(∇CE, ∇KD) at the final epoch: −0.39 (layer1) to −0.45 (layer4); **−0.76 at the head** | DHO arXiv 2505.07675v3, App. G.2 Table 26 (VERIFIED in project note; author-stated "Accepted at NeurIPS 2026") |
| Clean vs adversarial loss in adversarial training (not KD) | the conflict μ = ‖g_c‖‖g_a‖(1−cos) grows about as δ² with the attack budget (TG + EE) | CA-AT, arXiv 2410.16579 (VERIFIED HTML) |
| Full-data CIFAR-100, trained CNN teacher | **No measurement found** | gap (VERIFIED absence in phase 4; still unfilled) |
| Teacher wrong on some samples | KD and CE become "inconsistent" (qualitative) | RLD, ICCV 2025 (VERIFIED abs) |
| Dual clean/robust teachers in ARD | "divergent optimization objectives" (qualitative) | CIARD, arXiv 2509.12633 (VERIFIED abs) |

### L4. Robust distillation
- **Methods:**
  - ARD (Goldblum et al., AAAI 2020);
  - RSLAD (robust soft labels);
  - AdaAD (CVPR 2023);
  - PeerAiD (CVPR 2024: up to +1.66 AutoAttack points with ResNet-18 on TinyImageNet, search-only);
  - MTARD (ECCV 2022: adaptive normalisation loss balancing the clean and robust teachers, search-only);
  - CIARD (2025, code `eminentgu/CIARD`, VERIFIED abs);
  - gradient-matching ARD (ICLR 2025, arXiv 2312.03286, search-only).
- **CA-AT** (Xue, Wang, Qin, Pedarsani; arXiv 2410.16579 v1, no venue; VERIFIED HTML).
  - Rule: if φ = cos(g_a, g_c) ≤ γ, then g* = g_a + λ*·g_c with λ* = ‖g_a‖(γ√(1−φ²) − φ√(1−γ²)) / (‖g_c‖√(1−γ²)); otherwise g* = g_c.
  - This is a **cosine-margin, priority-style projection**, structurally close to PCD with g_a as anchor: PCD's margin is on progress, CA-AT's is on the angle.
  - Tested on CIFAR-10/100 with ResNet-18/34 and WRN-28-10, and with AutoAttack. **No KD.**
  - **INTERPRETATION:** this removes most of the novelty of "PCD for the clean/robust trade-off" outside KD. Any robust-KD PCD study must include CA-AT transplanted to ARD as a baseline.

### L5. Semi-supervised distillation from vision-language models
DHO (Kang, Lee, Jang, Hwang; arXiv 2505.07675 v3, 2026-10-24; code `erjui/DHO`; VERIFIED abs):
- It resolves the CE/KD conflict *architecturally*, with two heads (one CE, one KD) combined linearly at test time.
- It reports state-of-the-art semi-supervised ImageNet results and evaluates on 15 datasets.
- **Relation:** the cleanest "optimizer fix (PCD, single head) vs architecture fix (DHO)" comparison available, and the setting with the strongest *measured* conflict.

### L6. Unreliable teachers (the submitted proposal's question)
- RLD (ICCV 2025, code `zju-SWJ/RLD`, VERIFIED abs): uses labels to refine teacher logits without destroying class correlations.
- LR-DS label revision and data selection (arXiv 2404.03693, search-only).
- Both fix the *target*; PCD with reversed priority (KD primary, CE secondary) would fix the *combination*.
- No source found that treats the label loss as a progress constraint on a teacher-primary objective (searched 2026-11-03).

### L7. Constrained-optimisation views of KD
- Scaffolding a student (Kag et al., ICLR 2023, search-only).
- LLM distillation as a constrained MDP (arXiv 2509.22921, search-only).
- An exact-phrase search for "priority-constrained" with "distillation" returns only the PCD paper itself (2026-11-03).
- **INTERPRETATION:** PCD has never been evaluated on KD. The PCD paper contains no KD (VERIFIED, phase 4).

## Ranked ideas

Scores are 1–5 and are the coordinator's judgement (INTERPRETATION), **revised after the cycle-1 review**. The novelty of I1 drops because of DTO-KD; the novelty and risk of I4 drop because of the batch-level mechanism. Criteria: fit with PCD theory, open gap, comparability to published state of the art, GPU feasibility, course fit, and risk of an uninformative result (5 = low risk). These ordinal sums are a summary, not a measurement.

| Rank | ID | Idea | Fit | Gap | Comp | Feas | Course | Risk | Total |
|---|---|---|---|---|---|---|---|---|---|
| **1** | **I1\*** | **Priority vs symmetry in KD gradient combination**: PCD(CE \| KD) vs DTO-KD vs DOT vs a tuned normalised weighted sum, one loss-side plug-in, on CIFAR-100 ResNet32x4 → ResNet8x4 | 4 | 3 | 5 | 5 | 5 | 4 | **26** |
| 2 | I4 | Reversed priority (KD primary, CE secondary) under corrupted teachers: a **follow-up** priority-order ablation, not a reliability-protection method | 4 | 4 | 3 | 5 | 5 | 2 | 23 |
| 3 | I1-K3 | K=3 PCD(CE \| TCKD, NCKD) vs tuned DKD (exploratory extension of I1\*) | 3 | 3 | 4 | 4 | 4 | 3 | 21 |
| 4 | I3 | Single-head PCD vs dual-head DHO, semi-supervised VLM distillation | 4 | 4 | 4 | 3 | 3 | 3 | 21 |
| 5 | I2 | Priority-constrained robust distillation vs RSLAD/AdaAD/MTARD and CA-AT-in-ARD | 3 | 2 | 4 | 2 | 3 | 2 | 16 |
| 6 | I5 | Multi-teacher PCD vs AE-KD / MMKD | 3 | 3 | 3 | 2 | 3 | 3 | 17 |

### I1\* — priority vs symmetry (sole confirmatory study)
- **Research question:** on an identical CIFAR-100 KD protocol, does PCD's fixed-priority rule beat or match the symmetric DTO-KD rule, DOT's momentum split, and a budget-matched normalised weighted sum? PCD's rule is: CE coefficient pinned at 1, KD guaranteed τ-progress.
- **Why the comparison is informative (TG + INTERPRETATION):**
  - DTO-KD equalises the two losses' progress (Cor. 3.3).
  - PCD leaves the primary untouched whenever KD already makes τ-progress (Cor. 4.3), and otherwise adds the minimum KD needed (Cor. 4.6).
  - The two rules coincide only in special geometries. A difference in outcome would be attributable to the *combination rule*, because the loss, teacher, schedule and data order are held fixed.
- **Decisive control:** a tuned EMA-normalised weighted sum with PCD's rescale, specified executably in `EXPERIMENT_PLAN.md`. Without it, a PCD win cannot be attributed to projection rather than normalisation.
- **What it adds whatever the outcome:**
  - per-step CE–KD cosine distributions on standard CIFAR-100 KD, a gap no source fills;
  - DTO-KD's weights π and PCD's active rate and μ on the same runs.
- **Risks:**
  - small effects relative to seed noise, hence 10 seeds on the confirmatory contrasts;
  - DTO-KD has no released code, so it is reimplemented from Eqs. 11–12 and Alg. 1, and unit-tested against the closed form;
  - the primary can ascend under conflict (Prop. 4.7), so actual margins are logged.

### I4 — reversed priority (follow-up only)
- **Corrected mechanism (agreeing with the Codex review and the mechanism note C5):**
  - With KD primary, PCD's CE coefficient μ = max(0, τ − aᵀb/‖b‖²) is a **batch-level** quantity computed from aggregate, normalised parameter gradients.
  - It grows when the batch KD direction fails the CE progress threshold.
  - It does **not** identify which examples have wrong teacher outputs, and it applies CE to the whole batch.
  - Any "protects the student from the teacher" claim is unsupported. I4 is a **priority-order ablation**. Selective filtering stays with the explicit gate.
- **Prerequisites before it can run:**
  - frozen definitions of corrupted-teacher generation (Stage 1 log-space corruption);
  - per-example diagnostics: teacher correctness, confidence, per-example loss;
  - an identical teacher-logit pipeline for RLD;
  - RLD's licence and protocol checked.

### I1-K3, I3, I2, I5
- **I1-K3:** μ values are geometry-dependent multipliers, not DKD's α/β. It is exploratory only, against a jointly tuned two-weight normalised control (Codex review §1).
- **I3:** the strongest measured conflict, but the cost and architecture confounds are high.
- **I2:** CA-AT already supplies the projection mechanism; attack and evaluation costs dominate.
- **I5:** a costly multi-teacher setup.

All four go to the survey's future-work section.

## Novelty verification

Searches 2026-11-03 by Claude (WebSearch/WebFetch) and independently by Codex (`docs/reviews/phase7-codex-reviews-claude.md` §1):

| Claim | Closest prior work | Claude (rev. 2) | Codex cycle 1 |
|---|---|---|---|
| Optimizer-side task–KD conflict handling is sparse; DOT is the closest state of the art | **DTO-KD (ICLR 2026)**, DOT, MEMA, MoKD (withdrawn), CORD (arXiv 2610.29965, RL + self-distillation) | Claim **withdrawn**. The neighbourhood is crowded. | ABANDON as written |
| PCD's specific rule (fixed priority + τ-progress constraint) on KD | DTO-KD is symmetric; no PCD-in-KD found | **PROCEED WITH CAUTION**: novelty is the priority rule, not projection in KD | PROCEED WITH CAUTION |
| K=3 PCD as a replacement for DKD's α/β | DKD; DTO-KD; LSKD | Exploratory only | PROCEED WITH CAUTION |
| Label loss as a progress constraint (I4) | RLD, LR-DS, CHAD; none uses a progress constraint | PROCEED WITH CAUTION, with batch-level wording | PROCEED WITH CAUTION |
| Projection for clean/robust trade-off in KD (I2) | CA-AT (AT, not KD) | Incremental | PROCEED WITH CAUTION (baseline only) |
| Single-head PCD vs DHO (I3) | DHO | Open but setting-specific | PROCEED WITH CAUTION |

Novelty verdict for the selected idea I1\*: **PROCEED WITH CAUTION** (both reviewers). Codex dispatch `ctx_87dc917fe9c8`.

## External critical review

**Cycle 1** (Codex, ORCA task `task_e32b7cee9b3a`, dispatch `ctx_87dc917fe9c8`, file `docs/reviews/phase7-codex-reviews-claude.md`): **score 4/10, ABANDON in current form**, with six minimum fixes. Claude's response to each:

| # | Codex finding / required fix | Claude verdict | Evidence / action |
|---|---|---|---|
| 1 | DTO-KD (ICLR 2026) omitted; novelty must be recast around the fixed-priority rule; demote "DOT is the optimizer state of the art" | **AGREE** | Verified in the DTO-KD full PDF (Sec. 3.1, Eqs. 11–12, Cor. 3.3, Table 2). The record was already in `sources.yaml` (`hayder2026dtokd`, phase 4); my literature pass missed it. That is my error. DTO-KD is now a core baseline, reimplemented from the paper. Refinement: its "gradient projection" is FAMO-style min-norm weighting of log-loss gradients (symmetric), which sharpens the priority-vs-symmetry question. |
| 2 | Make I1 the sole confirmatory study on P1 with a small arm set | **AGREE** | `EXPERIMENT_PLAN.md` rev. 2: P1 only; 6 arms; K=3, other pairs and I4 exploratory or follow-up. |
| 3 | I4 wording: batch alignment, not teacher–label contradiction or per-example trust | **AGREE** | I4 rewritten above; claim downgraded to a priority-order ablation. |
| 4 | Executable controls and equal tuning budget; pin commits; verify α, the P3 learning rate, DOT δ; fix the run arithmetic | **AGREE** | Plan rev. 2 §2–§4. DOT `DELTA 0.075` taken from the official config (Codex-verified); P3 is dropped from the confirmatory design. DKD α and commit SHAs remain [PIN] items before freezing. Arithmetic is shown row by row. |
| 5 | Five seeds can't establish equivalence; predefine endpoints, margins, families | **AGREE** | 10 paired seeds for the confirmatory contrasts; one primary endpoint; a Holm family of 3; equivalence is a secondary analysis claimed only if the 90% TOST interval (α = 0.05 per side) lies inside ±δ = 0.5. There is no pilot-variance gate (cycle-2 disposition). |
| 6 | Reduce scope and GPU budget; get approval | **AGREE** | About 125 GPU-h confirmatory (ESTIMATE) after a 6 GPU-h pilot; approval still required. |
| — | `du2020aekd` described as "VERIFIED" in the report while its record says unverified | **AGREE** | Report wording corrected. |
| — | H2's "≥95% active ⇒ equivalence" reasoning | **AGREE** | H2 is now an effect-size estimate plus per-step direction comparison; no causal-equivalence inference. |

**Cycle 2** (Codex, task `task_71c07d07ef05`, dispatch `ctx_4180e1f47c37`, message `msg_6395188246c1`; appended to `docs/reviews/phase7-codex-reviews-claude.md`): **score 7/10, PROCEED.**
- Fixes 1, 2, 3 and 6: RESOLVED.
- Fix 4: PARTIAL. [PIN] items must be pinned before freeze; the plan already requires that.
- Fix 5: UNRESOLVED. It covered two things: the 90% vs 95% inconsistency, and a variance gate based on three pilot runs.
- New claims: DTO-KD description AGREE (wording made precise); H0′ AGREE on the algebra, HYPOTHESIS for accuracy; TOST arithmetic AGREE; run count AGREE; GPU bounds should read 119–236.

**Coordinator disposition** (maximum cycles reached; recorded in `docs/reviews/phase7-summary-and-unresolved.md`):
- Fix 5 is resolved by *removing* the pilot-variance gate. Equivalence becomes an always-reported secondary analysis with a 90% TOST, and the report text is aligned to it.
- The DTO-KD wording now says "equal first-order contribution"; the budget now reads 119–236.
- No third review cycle; the remaining verification happens at the [PIN] freeze before any run.

## RECOMMENDATION (revision 2, pending cycle-2 verdict)
- **Adopt I1\* as the sole confirmatory study:** *priority vs symmetry in KD gradient combination*. PCD(CE | KD) vs DTO-KD vs DOT vs a tuned normalised weighted sum, with KD and one loss-side plug-in (KD+LS, PCD on top), on CIFAR-100 ResNet32x4 → ResNet8x4 with 10 seeds. This answers "how does PCD compare with state-of-the-art KD" where the comparison is meaningful (the optimizer family), and produces the missing conflict measurements.
- **Keep I4 as a pre-specified follow-up.** It is a priority-order ablation under corrupted teachers, run only if the I1\* pilot shows a non-trivial active rate.
- **Future work in the survey:** I1-K3, I3, I2 and I5.

<!-- ARIS_IDEA_DISCOVERY_EVIDENCE_GATE:START -->
## Evidence Gate
**Status:** PASS

All required stage records, review receipts, artifacts, and report sections are present.
<!-- ARIS_IDEA_DISCOVERY_EVIDENCE_GATE:END -->
