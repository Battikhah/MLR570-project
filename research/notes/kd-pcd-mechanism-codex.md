# PCD in KD: mechanism and GPU feasibility (Phase 7)

**Evidence labels.** VERIFIED means checked in project primary notes/code or a linked primary paper/repository. INTERPRETATION is a mathematical consequence or methodological judgment. HYPOTHESIS is untested. RECOMMENDATION is proposed. Runtime ranges are ESTIMATE. No training or dataset download was run.

## Common PCD algebra

Let raw gradients be g_i=∇θL_i. The implementation tracks EMA squared norms v_i←βv_i+(1−β)||g_i||², bias-corrects, and forms a_i=s_i g_i, s_i=1/sqrt(v̂_i+ε). PCD solves min_d ½||d−a_1||² subject to a_jᵀd≥τ_j||a_j||². Its KKT direction is d~=a_1+Σμ_j a_j, μ_j≥0; it then rescales d=d~||g_1||/||d~||. Rescaling preserves alignment signs, but momentum/Adam can change the realized parameter update. **VERIFIED:** src/pcdkd/optim/pcd.py, combiners.py and literature/notes/varam2026pcd.md.

For K=2, a=a_1, b=a_2. The constraint is inactive if aᵀb≥τ||b||², so d~=a. Otherwise α=τ−aᵀb/||b||²>0 and d~=a+αb. α is the secondary/primary coefficient in normalized-gradient space; in raw-gradient space it is αs_2/s_1. This is adaptive normalized weighted sum (NWKD) with PCD EMA and final rescale. **VERIFIED:** pcd_k2_closed_form and AdaptiveLambda implement this identity.

For unit normalized vectors and c=aᵀb, active PCD has α=τ−c and primary margin aᵀd~=1+τc−c² (Proposition 4.7). At c=−1, α=τ+1 and d~=−τa, so primary progress is negative for every τ>0. At τ=0 the exactly opposite pair cancels. **VERIFIED algebra; INTERPRETATION implication.**

For K=3, primary a and secondaries b,c, define G=[[bᵀb,bᵀc],[cᵀb,cᵀc]] and q=[τ_b||b||²−bᵀa, τ_c||c||²−cᵀa]. If both constraints bind and G is nonsingular, μ=G⁻¹q, with μ≥0 and all inactive inequalities feasible. For b-only active, μ_b=q_b/||b||²≥0 and bᵀ(a+μ_bb)≥τ_c||c||²; likewise for c-only. If q≤0 componentwise, direction is inactive. If no feasible KKT working set exists, project code falls back to a; K≥3 feasibility is not guaranteed. **VERIFIED:** pcd_kkt_enumeration.

**VERIFIED theory scope:** guarantees are per-step directional statements; the paper gives no general convergence theorem for deployed PCD. KD/adversarial secondaries do not satisfy the paper's convex-secondary assumption. Secondary progress does not imply primary progress or test accuracy. Norm rescaling does not undo optimizer-state effects.

## Candidate analyses

### C1 — Robust/adversarial distillation

Let g_R be robust KD (for example robust-teacher soft targets on natural/adversarial inputs as in RSLAD) and g_C clean CE/KD. These objective labels do not assert all ARD/RSLAD formulations are identical.

**(i) Robust primary, clean secondary.** With a_R=s_Rg_R and b_C=s_Cg_C, d~=a_R+max(0,τ−a_Rᵀb_C/||b_C||²)b_C, rescaled to ||g_R||. If a_Rᵀb_C≥τ||b_C||², clean loss has no effect; otherwise clean progress is enforced but robust primary margin can be negative. For unit vectors it is 1+τc−c²; at opposition d~=−τa_R. Thus robustness progress is lost in the near-opposite regime. **VERIFIED algebra; INTERPRETATION:** this ordering spends robustness to satisfy clean progress.

**(ii) Clean primary, robust secondary.** Swap roles: d~=a_C+max(0,τ−a_Cᵀb_R/||b_R||²)b_R, rescale to ||g_C||. Conflict increases robust-secondary weight but CE margin may vanish or turn negative under Proposition 4.7. This is not a robustness-protection mechanism. **INTERPRETATION.**

**Reduction:** adaptive NWKD for K=2; inactive constraint is primary-only. Related to dynamic-barrier GD; not symmetric PCGrad. **VERIFIED** in PCD paper note. **Predicted direction:** conflict raises α and ensures constructed secondary progress, while primary descent becomes less certain. **Decisive control:** EMA-normalized weighted sum sweeping λ including measured α, same rescale, plus robust-only/conventional robust KD; report clean/robust accuracy and AutoAttack. **Likeliest failure:** nonconvex attack loss and noisy minibatch conflict yield lost primary performance without robust generalization.

### C2 — DKD as K=3: CE, TCKD, NCKD

Let g_0=∇CE, g_T=∇TCKD, g_N=∇NCKD on common student logits, with fixed teacher/labels. Normalize each separately. PCD returns d~=a_0+μ_Ta_T+μ_Na_N and rescales to ||g_0||. DKD itself uses CE_WEIGHT·CE + α_DKD·TCKD + β_DKD·NCKD (with warmup). **VERIFIED:** official MDistiller DKD.py and configuration.

PCD τ values do not directly substitute for DKD α/β: PCD imposes normalized directional-progress halfspaces, and μ is solved jointly from gradient geometry; DKD coefficients scale loss terms. **INTERPRETATION:** it could replace them as a training rule, but controls need EMA-normalized DKD weights and same rescale. The QP is inactive iff a_0ᵀa_T≥τ_T||a_T||² and a_0ᵀa_N≥τ_N||a_N||². Violation activates one or both constraints; both-active solution uses the 2×2 secondary Gram system. Nearly singular coupling can make it ill-conditioned, and opposing constraints may be infeasible.

TCKD separates target-vs-rest probability mass; NCKD distills conditional distribution among non-target classes. They are distinct functions of the same logits, but their parameter gradients are not generally orthogonal: shared softmax/logit Jacobian and student representation couple them. **VERIFIED objective definitions/code; INTERPRETATION gradient relation.**

**Reduction:** anchored adaptive normalized scalarization, not symmetric MGDA. **Predicted direction:** CE remains anchor; T/N constraints only fire when their directional thresholds are violated. **Decisive control:** official DKD alpha/beta, EMA-normalized weighted CE+TCKD+NCKD with equal joint tuning budget/rescale. **Likeliest failure:** T/N gradients are coupled or redundant, making PCD add little or encounter ill-conditioning/infeasibility. **RECOMMENDATION:** later mechanistic study, not first confirmatory candidate.

### C3 — Semi-supervised CLIP distillation / DHO

Let g_C be supervised CE on labeled examples and g_U KD on unlabeled teacher predictions. Single shared head with CE primary gives d~=a_C+max(0,τ−a_Cᵀb_U/||b_U||²)b_U, rescaled to ||g_C||. Reverse order makes KD the anchor and CE secondary. **INTERPRETATION:** PCD can reweight one shared parameter-space update but cannot remove representational interference; CE-primary does not guarantee CE descent in conflict.

DHO reports single-head CE/KD conflict and uses separate supervised and KD heads over shared features, combining predictions at inference. Head-specific parameters avoid direct conflict in those head weights; shared-feature gradients can still conflict. **VERIFIED:** arXiv:2505.07675 and official erjui/DHO repo. Project decision 0001 records negative cosines in DHO's own few-shot setting; not transferable to CIFAR or other models.

**Reduction:** adaptive NWKD, not DHO architecture. **Predicted direction:** conflict raises KD coefficient, potentially reversing CE descent. **Decisive control:** DHO dual head, standard single-head CE+KD, CE-only, PCD and matched normalized-sum weights, same labeled fractions/encoder/augmentations/inference tuning. **Likeliest failure:** architecture differences, CLIP preprocessing, and few-label variance swamp optimizer effect.

### C4 — Multi-teacher KD (K=1+M) versus AE-KD/MGDA

Use CE primary and each teacher KD loss as separate secondary: d~=a_0+Σ_m μ_ma_m, jointly solving the anchored QP. For two teachers, use K=3 formula. More teachers increase working-set complexity and infeasibility risk; this checkout supports at most K=4. Infeasibility falls back to primary only. **VERIFIED:** project pcd.py.

AE-KD formulates teacher losses as symmetric MOO and uses MGDA with tolerance to accommodate disagreement. **VERIFIED:** NeurIPS 2020 proceedings paper, “Agree to Disagree: Adaptive Ensemble Knowledge Distillation in Gradient Space.” PCD with CE primary fixes CE coefficient at one and treats teachers as subordinate, so is not a Pareto compromise among all losses. **INTERPRETATION.**

**Reduction:** anchored normalized multiobjective projection; it is one-sided and differs from symmetric AE-KD. **Predicted direction:** CE anchor, but teacher constraints may collectively be infeasible or force CE ascent; teacher diversity is no guarantee of usefulness. **Decisive control:** averaged-teacher target with same forward budget, equal teacher KD, tuned normalized weights, and AE-KD/MGDA matched compute. **Likeliest failure:** correlated/adversarial teachers create incompatible constraints; unreported fallback silently drops teacher supervision.

### C5 — Reversed priority in corrupted-teacher CIFAR: KD primary, CE secondary

Let batch g_KD=Σ_i q_i g_KD,i and g_CE=Σ_i r_i g_CE,i with same exact reductions across arms. For a=s_KDg_KD, b=s_CEg_CE, direction is a+μb, μ=max(0,τ−aᵀb/||b||²), rescaled to ||g_KD||. Batch-level implicit raw-space CE coefficient is μs_CE/s_KD. **VERIFIED algebra.**

For corrupted fraction ρ, g_KD=(1−ρ)g_KD,clean+ρg_KD,corr only if sampling/reduction match those proportions; CE still uses labels across examples. Corruption affects μ indirectly through aggregate KD direction/norm; PCD has no per-example reliability weighting. If corrupted KD conflicts more, μ may grow and increase global CE influence, applying to all batch examples. KD-primary margin may still turn negative, so CE can be “corrected” at the expense of KD progress. **INTERPRETATION:** reversing priority changes anchor, not reliability semantics.

**Reduction:** adaptive NWKD with KD anchor. **Predicted direction:** corruption can change cosine and raise CE multiplier but cannot selectively suppress corrupted examples. **Decisive control:** CE-primary versus KD-primary PCD, matched normalized-sum arms, explicit correctness/confidence gate, per-example diagnostics. **Likeliest failure:** batch aggregation is mistaken for selective trust or causal reliability handling.

## GPU feasibility — ESTIMATES, not measurements

Ranges assume one A100-class GPU, mixed precision where stable, frozen teacher, one seed, no sweeps, no input pipeline bottleneck. PCD requires separate objective-gradient extraction; backward counts are student objective backprops, excluding teacher forwards. C1 attack steps include input-gradient steps. Timings need a pilot. **INTERPRETATION / ESTIMATE.**

| Candidate | Suggested protocol and evaluation | Student backward/step | ESTIMATED hours/run | Dataset/licence |
|---|---|---:|---:|---|
| C1 robust KD | CIFAR-10, robust WRN-34-10 teacher→WRN-10-1 student or published RSLAD pair; 100 epochs; PGD-10 first feasibility run; clean and AutoAttack robust accuracy. Verify source protocol before reproduction. | ~11 (10 input attack grads + objective/student grad; separate objectives may add) | 10–30 h | CIFAR owner lists 50k/10k, 163 MB; content licence/terms unresolved. AutoAttack eval overhead excluded. |
| C2 DKD K=3 | CIFAR-100 MDistiller ResNet32x4→ResNet8x4; 240 epochs, batch 64 repository default; test top-1 and active/feasibility rates. | 3 | 2–6 h | 50k/10k, 161 MB; content terms unresolved. Official config path is mutable master; SHA not captured. |
| C3 DHO | ImageNet-1K semi-supervised 1% then 10%, CLIP and source student/augmentations; evaluate top-1 and identical head combination. Confirm source schedule before run. | 2 shared-trunk objectives | 12–48 h (1%); 25–80 h (10%) | ImageNet requires registered access and non-commercial terms review/approval; do not fetch. |
| C4 multi-teacher | CIFAR-100 ResNet32x4→ResNet8x4; 2–3 frozen heterogeneous teachers; 240 epochs; top-1 plus teacher-wise diagnostics. | 1+M | 3–10 h (M=2); 5–16 h (M=4) | CIFAR rights caveat above. Cached logits trade disk/generation time for teacher-forward cost. |
| C5 reversed priority | CIFAR-100 ResNet-32→ResNet-8; current corruption map/seeds; 40–100 epochs; clean test top-1 and per-corruption diagnostics. | 2 | 1–4 h | CIFAR rights caveat above; keep test set untouched. |

**Estimate assumptions:** CIFAR architectures/batch sizes as above, single seed and no tuning. C3 wide range reflects ImageNet pipeline/student. C1 is costly because iterative input attacks. **RECOMMENDATION:** time one epoch per arm on target GPU and scale with evaluation/setup margin; do not budget a grid from these placeholders.

## Ranking and open risks

1. **C5/current CIFAR (RECOMMENDATION, conditional):** cheapest/directly connected, and reversed order is a falsifiable mechanism ablation; KD-primary should not headline a teacher-reliability protection claim.
2. **C2 DKD K=3 (INTERPRETATION):** clean test of K=3 beyond hand-set weights; gradient coupling and controls complicate it.
3. **C1 robust KD (INTERPRETATION):** real conflict/deployment metrics, but attack cost, nonconvexity, and priority policy complicate claims.
4. **C4 multi-teacher (INTERPRETATION):** relevant AE-KD comparator but teacher selection, feasibility, and K limit add confounds; start with two teachers.
5. **C3 DHO/ImageNet (INTERPRETATION):** rich case but most expensive and architecture/head design confounds optimizer attribution.

Ranking is mechanism isolation/feasibility, not predicted accuracy. PCD superiority over these KD baselines is unverified. **HYPOTHESIS:** C5 with both priority orders and matched controls is the strongest economical null-result test; C2 best demonstrates K>2 solver behavior.

Open risks: **VERIFIED** KD/adversarial objectives are nonconvex and outside PCD's convex-secondary assumptions; the project's solver caps K at four; optimizer state may alter realized alignment. **INTERPRETATION:** EMA history/batch reductions must match in controls. **UNVERIFIED:** GPU timings, proposed accuracy effects, and exact DHO schedule for this project. **RECOMMENDATION:** pin source commits and exact protocols; verify dataset terms/access; authorized pilot should log active rates, fallbacks, margins, and optimizer-step alignments.

## Sources and verification

- **VERIFIED:** PCD primary details in literature/notes/varam2026pcd.md and project code src/pcdkd/optim/{pcd.py,combiners.py}.
- **VERIFIED:** DKD, Zhao et al., CVPR 2022, arXiv:2203.08679 https://arxiv.org/abs/2203.08679. Official code path https://github.com/megvii-research/mdistiller/blob/master/mdistiller/distillers/DKD.py; config https://github.com/megvii-research/mdistiller/blob/master/configs/cifar100/dkd/res32x4_res8x4.yaml. Mutable master inspected; SHA not captured.
- **VERIFIED:** RSLAD, Zi et al., arXiv:2108.07969 https://arxiv.org/abs/2108.07969; paper abstract reports natural/adversarial robust soft labels and AutoAttack evaluation. Exact code/attack schedule not audited.
- **VERIFIED:** DHO, Kang et al., arXiv:2505.07675 https://arxiv.org/abs/2505.07675; official repository https://github.com/erjui/DHO. Project decision 0001 records study-specific measured conflict; values are not generalized here.
- **VERIFIED:** AE-KD, NeurIPS 2020, https://proceedings.neurips.cc/paper/2020/hash/91c77393975889bd08f301c9e13a44b7-Abstract.html; symmetric MGDA with tolerance.
- **VERIFIED:** MDistiller README documents CIFAR-100 commands and ImageNet entrypoint https://github.com/megvii-research/mdistiller. Pin commit before reproduction.
- **VERIFIED, limited:** CIFAR counts/archive sizes owner page https://www.cs.toronto.edu/~kriz/cifar.html and research/notes/experimental-feasibility.md; data content terms unresolved. ImageNet terms/access must be checked directly before any use.

## Synthetic numerical check

Command: Python/NumPy constructed unit vectors with cosine c∈{−0.5,−0.9,−1.0}, τ=0.2, then printed α, Proposition 4.7 margin, aᵀd~, rescaled cosine. Exit 0.

    c=-0.5 alpha=0.700 prop47=0.650 dot(a,d)=0.650 rescaled_cos=0.731
    c=-0.9 alpha=1.100 prop47=0.010 dot(a,d)=0.010 rescaled_cos=0.021
    c=-1.0 alpha=1.200 prop47=-0.200 dot(a,d)=-0.200 rescaled_cos=-1.000

**INTERPRETATION:** confirms the primary directional margin collapses and reverses near opposition; it is an algebraic check, not neural evidence.
