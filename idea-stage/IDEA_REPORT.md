# IDEA REPORT — PCD in knowledge distillation vs. state-of-the-art KD (Phase 7)

- **Run:** ORCA `run_8e2818695e01`; ARIS state `kd-pcd-sota-20261103` (Stage 1 / idea discovery only).
- **Date:** 2026-11-03. **Executor:** Claude Opus 5.5 (coordinator literature track).
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

### L0. Where PCD sits
PCD is an **optimizer-side combiner**. Given the gradients of the primary and the secondaries, it returns one update direction:
- EMA normalisation of each gradient;
- projection of g̃₁ onto the half-spaces g̃ⱼᵀd ≥ τ‖g̃ⱼ‖²;
- a final rescale to ‖g₁‖.

It does **not** change the KD target or loss (VERIFIED, `literature/notes/varam2026pcd.md`). State-of-the-art KD splits into two families that PCD relates to differently:

| Family | Changes… | Representative methods | Relation to PCD |
|---|---|---|---|
| **Loss / target design** | what the student imitates | KD (Hinton 2015); DKD (CVPR 2022); CTKD (AAAI 2023); MLKD (CVPR 2023); LSKD logit standardisation (CVPR 2024); RLD refined logits (ICCV 2025); feature KD: FitNets, AT, CRD, ReviewKD | **Orthogonal.** PCD can be stacked on any of them by making their loss a secondary. |
| **Optimizer / loss balancing** | how the task and distillation gradients are combined | DOT (ICCV 2023); MEMA in SPOFA (preprint 2026); MoKD (withdrawn); AE-KD (NeurIPS 2020, multi-teacher MGDA); learnable α / AKD (2025, search-only); RW-KD (EMNLP-F 2021, search-only); CHAD gating (2026); CA-AT (adversarial training, not KD) | **Direct competitors.** PCD's claim must be judged inside this family. |

**INTERPRETATION (central):** "PCD vs state-of-the-art KD" is a category error unless the comparison is split.
1. Within the optimizer family: PCD vs DOT, MEMA-style penalties, MGDA/PCGrad/CAGrad and tuned normalised weighted sums, all on the same KD loss.
2. As a plug-in: X vs PCD+X for the strongest loss-design X (LSKD, MLKD, DKD).

A PCD student with vanilla KD loss is not expected to beat MLKD+LS. The loss-design gains on the reference pair are much larger than anything an optimizer change has reported (L1).

### L1. Benchmark anchor: CIFAR-100, ResNet32x4 → ResNet8x4 (VERIFIED from official repos)
| Method | Top-1 | Family | Source |
|---|---|---|---|
| KD | 73.33 | loss | mdistiller README; LSKD README |
| KD + DOT | 75.12 | **optimizer** | mdistiller README (DOT is the official implementation, MIT) |
| DKD | 76.32 | loss | mdistiller README |
| KD + LS | 76.62 | loss | LSKD README |
| DKD + LS | 77.01 | loss | LSKD README |
| MLKD | 77.08 | loss | LSKD README |
| MLKD + LS | **78.28** | loss | LSKD README |

On ImageNet with ResNet34 → ResNet18: KD 71.03, DKD 71.70, KD+DOT 71.72 (VERIFIED, mdistiller README).

**INTERPRETATION:** the best optimizer-side method (DOT) adds +1.8 on this CIFAR pair. Loss-side methods add +3.0 to +5.0. Realistically, PCD is a candidate to match or beat DOT and to add on top of LSKD/DKD; it is not a candidate to set a new absolute record.

### L2. Optimizer-side KD methods: the direct competitors
- **DOT** (Zhao, Cui, Song, Liang; ICCV 2023; arXiv 2307.08436; VERIFIED abs).
  - It identifies a trade-off: adding the distillation loss slows the task loss.
  - It computes the two gradients separately and gives the distillation gradient a larger momentum. The supplement says this applies only to parameters both losses touch (search-only).
  - Code is in mdistiller.
  - **Relation:** DOT is the closest optimizer-side state of the art to PCD-for-KD, and it fixes the trade-off through momentum, not projection. It is a mandatory baseline.
- **MEMA / SPOFA** (Yang, Zhang, Zhao; arXiv 2606.24557, 2026-06-23, "under review"; VERIFIED abs).
  - For heterogeneous KD (Transformer → CNN), it judges task–distillation conflict against an EMA baseline and "adaptively penalizes harmful distillation signals".
  - The exact rule is not in the abstract, and no code is linked.
  - **Relation:** close in spirit (task first, KD suppressed when it conflicts). It is the opposite of PCD, which *guarantees* KD progress.
- **MoKD** (Hayder et al.; arXiv 2505.08170; VERIFIED abs).
  - It framed KD as multi-objective optimisation against "gradient conflicts" and "gradient dominance".
  - **It was withdrawn on 2025-08-02** ("major changes to the paper and the authorship"), and no PDF is available.
  - **Relation:** evidence that the framing is considered publishable. It cannot be used as a baseline.
- **AE-KD** (Du et al., NeurIPS 2020; VERIFIED proceedings page in an earlier search): ensemble or multi-teacher KD as MGDA with a tolerance for teacher disagreement. It is symmetric, with no primary.
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

Scores are 1–5 and are the coordinator's judgement (INTERPRETATION). Criteria:
- **Fit:** fit with PCD theory.
- **Gap:** whether the question is open.
- **Comp:** comparability to published state of the art.
- **Feas:** GPU feasibility in one semester.
- **Course:** fit with MLR570 (survey on KD and gradient conflict; proposal on unreliable teachers).
- **Risk:** of an uninformative result (5 = low risk).

| Rank | ID | Idea | Fit | Gap | Comp | Feas | Course | Risk | Total |
|---|---|---|---|---|---|---|---|---|---|
| **1** | **I1** | **PCD as an optimizer-side plug-in on the mdistiller/LSKD benchmark**: CE primary; secondaries = KD, or TCKD+NCKD (K=3); vs DOT, tuned normalised WS, PCGrad/CAGrad/MGDA, and stacked on LSKD/DKD | 4 | 4 | **5** | 5 | **5** | 4 | **27** |
| **2** | **I4** | **Reversed priority under unreliable teachers** (KD primary, CE secondary) on CIFAR-100 with natural teacher errors + 30% corruption; vs RLD, label revision, tuned NWKD, CE-primary PCD | **5** | **5** | 3 | 5 | **5** | 3 | 26 |
| 3 | I3 | Single-head PCD vs dual-head DHO, semi-supervised VLM → small student (strongest measured conflict) | 4 | 4 | 4 | 3 (CLIP + datasets, ImageNet optional) | 3 | 3 | 21 |
| 4 | I2 | Priority-constrained robust distillation (robust KD primary, clean secondary, and the reverse) vs RSLAD/AdaAD/MTARD and CA-AT-in-ARD | 3 (strong-conflict regime: Prop. 4.7) | 3 (CA-AT covers the mechanism) | 4 | 2 (10-step attacks ≈ 11× cost; robust teacher; AutoAttack) | 3 | 2 (gradient-masking pitfalls) | 17 |
| 5 | I5 | Multi-teacher PCD vs AE-KD / MMKD | 3 | 3 | 3 | 2 (several teachers, M+1 backward passes) | 3 | 3 | 17 |

### I1 — PCD vs state-of-the-art KD, done fairly
- **Research question:** on the standard CIFAR-100 (and optional ImageNet) KD benchmark, does PCD's priority projection improve on the best optimizer-side KD method (DOT) and add to the best loss-side methods (LSKD, DKD)? And is any gain due to the projection, or to EMA normalisation alone?
- **Why PCD could help (HYPOTHESIS, grounded in TG):**
  - DKD's NCKD weight β must be hand-tuned per teacher/student pair, and TCKD/NCKD/CE differ in scale.
  - PCD's normalisation is asymptotically scale-invariant (PCD Thm B.9, TG), which is the mechanism behind PCD's only strong published wins (EE, pruning).
  - A K=3 PCD (CE | TCKD, NCKD) is a **tuning-free DKD** candidate with a single τ.
- **Why it may not:** the PCD authors' own Table 10 shows that at cosine ≈ 0, PCD equals a normalised weighted sum (EE). The decisive control is therefore a budget-matched EMA-normalised weighted sum with PCD's rescale.
- **What it adds whatever the outcome:** the **first measurement of the per-step CE–KD (and TCKD–NCKD) cosine distribution for full-data CIFAR-100 KD** (gap L3), plus a fair optimizer-side comparison with DOT.
- **Risks:**
  - A small or null effect size. That is still informative, and the plan is preregistered.
  - The cost of K backward passes (2–3×).
  - Teacher checkpoints must come from mdistiller's release (a download needing approval).

### I4 — Reversed priority: protect the student from the teacher through the label constraint
- **Mechanism (derived from the K=2 closed form; TG under unit normalisation):** with a = g̃_KD as primary and b = g̃_CE as secondary, the CE weight is α = max(0, τ − c).
  - It is zero when the teacher agrees with the label (c > τ).
  - It grows as the teacher contradicts the label.
  - The Prop. 4.7 primary-ascent case becomes "stop following a teacher that contradicts the label", which is the desired behaviour. This is the reverse of objective S, where theory predicts harm.
- **Comparators:**
  - RLD (ICCV 2025), which fixes the target, not the combination;
  - LR-DS;
  - tuned NWKD with a high CE weight (the decisive control);
  - CE-primary PCD;
  - gated arms (from Stage 1 code).
- **What it adds:** a new framing (label as a progress constraint), with no prior source found (L6). It keeps the submitted proposal's question and reuses the Stage 1 corruption, gate and combiner code almost unchanged.
- **Risks:**
  - On clean full-data CIFAR the constraint may rarely bind.
  - Labels themselves must be trustworthy.
  - Per-batch conflict mixes corrupted and clean rows, so a batch-level projection may dilute the per-sample signal (HYPOTHESIS; to be measured by Codex C5).

### I3 / I2 / I5
- **I3 (DHO setting):** the strongest scientific case for conflict. Ranked third because it depends on a CLIP teacher, several dataset downloads, and DHO's architecture fix already being strong and accepted.
- **I2 (robust KD):** a real conflict, but in the regime where PCD's primary can ascend (Prop. 4.7). CA-AT already offers a cosine-margin projection, and the compute and evaluation burden is high (10-step attacks in training; AutoAttack; checks for gradient masking). Recommended only as a stretch if the GPU budget is large.
- **I5 (multi-teacher):** structurally FedPCD-like, but costly. AE-KD is a dated baseline, and MMKD's code would need adaptation.

## Novelty verification

Searches run 2026-11-03 (WebSearch/WebFetch; details in the ORCA transcript):

| Claim | Closest prior work found | Verdict (coordinator) |
|---|---|---|
| PCD (priority projection with EMA normalisation) applied to KD | none; the exact phrase returns only PCD itself; the PCD paper has no KD | **Novel application** (VERIFIED absence, limited search) |
| Optimizer-side handling of task–KD conflict | DOT (momentum, ICCV 2023), MEMA (penalty, preprint 2026), MoKD (MOO, withdrawn), AE-KD (MGDA, multi-teacher) | **Crowded neighbourhood.** PCD's distinct element is a *guaranteed-progress* constraint with a fixed primary. Must beat DOT. |
| K=3 PCD replacing DKD's α/β tuning | DKD (hand-tuned α, β); LSKD (standardisation removes some scale issues) | Novel as stated; LSKD partly overlaps the scale argument (INTERPRETATION) |
| Label loss as progress constraint on a teacher-primary objective (I4) | RLD and LR-DS fix targets; CHAD gates per sample; none uses a progress constraint | **Novel framing** (limited search) |
| Priority projection for the clean/robust trade-off in KD (I2) | CA-AT (cosine-margin projection in AT, not KD); MTARD (normalisation); CIARD (push loss) | **Incremental:** a CA-AT-in-KD transplant is the obvious baseline |
| Single-head optimizer fix vs DHO dual head (I3) | DHO itself; MEMA | Novel comparison |

Cross-family novelty check: **pending** (Codex via ORCA). See the next section.

## External critical review

*Pending: an independent Codex reviewer will be dispatched through ORCA after the mechanism worker reports (ARIS's MCP reviewer is not configured; ORCA's Claude↔Codex review protocol replaces it). Findings will be folded in here with AGREE / DISAGREE / UNSUPPORTED verdicts and the dispatch id as the verdict handle.*

## RECOMMENDATION (pending review)
Adopt **I1 as the core study and I4 as the unreliable-teacher study**, one paper with two research questions:
- **RQ1 (I1):** how does PCD compare with state-of-the-art KD? Inside the optimizer family vs DOT and MOO baselines; as a plug-in on DKD and LSKD; with the normalisation-vs-projection control.
- **RQ2 (I4):** when the teacher is unreliable, does putting the teacher first and the label behind a progress constraint protect the student better than target-repair methods (RLD) and gating?

This keeps the course fit (KD plus gradient conflict, plus the original proposal's question) and gives published comparators. Every run sits on one standard benchmark, so the GPU budget stays modest (see `refine-logs/EXPERIMENT_PLAN.md`). I2 and I3 are documented stretch options for the survey's future-work section.
