# Knowledge Distillation: Methods and Comparisons

Khalid Hamdan, Aly Khaled Ismail Mahmoud, Hasan Akel, and Salim Akel

# Abstract

The abstract goes here.  
Keywords: Knowledge distillation, teacher reliability, gradient conflict, multi-objective optimization, objective priority.

# Introduction

* Knowledge distillation enables compact student networks to learn from larger teachers, supporting deployment with limited computational resources [https\://arxiv.org/pdf/1503.02531](https://arxiv.org/pdf/1503.02531).  
* A more accurate or larger teacher does not necessarily produce a better student; differences in capacity can limit effective knowledge transfer [https\://openaccess.thecvf.com/content\_ICCV\_2019/html/Cho\_On\_the\_Efficacy\_of\_Knowledge\_Distillation\_ICCV\_2019\_paper.html](https://openaccess.thecvf.com/content_ICCV_2019/html/Cho_On_the_Efficacy_of_Knowledge_Distillation_ICCV_2019_paper.html).  
* Ground-truth supervision and teacher imitation can produce conflicting gradients, while differences in gradient magnitude can cause one objective to dominate training [https\://arxiv.org/pdf/2001.06782](https://arxiv.org/pdf/2001.06782) [https\://proceedings.iclr.cc/paper\_files/paper/2026/hash/e546ce4230c15bc422a500bb47b341ea-Abstract-Conference.html](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e546ce4230c15bc422a500bb47b341ea-Abstract-Conference.html).  
* Teacher reliability and gradient compatibility are distinct concerns: selecting reliable supervision does not necessarily remove optimization conflicts [https\://doi.org/10.3390/jimaging12090436](https://doi.org/10.3390/jimaging12090436).

# Foundations and Knowledge Types

## Conventional Knowledge Distillation

* A pretrained teacher provides additional supervision to a student, typically while the teacher parameters remain fixed [https\://arxiv.org/pdf/1503.02531](https://arxiv.org/pdf/1503.02531).  
* The student learns from both ground-truth labels and the teacher's softened output distribution.  
* Temperature controls the softness of teacher and student probabilities, exposing relationships among classes beyond the highest-probability prediction.  
* A common objective combines supervised cross-entropy with temperature-scaled KL divergence:  
* L \= L\_CE \+ λT² KL(p\_t^(T) \|\| p\_s^(T)), where λ controls the distillation contribution [https\://arxiv.org/pdf/1503.02531](https://arxiv.org/pdf/1503.02531).  
* Transfer effectiveness depends on teacher–student compatibility and the training procedure, rather than teacher accuracy alone [https\://openaccess.thecvf.com/content\_ICCV\_2019/html/Cho\_On\_the\_Efficacy\_of\_Knowledge\_Distillation\_ICCV\_2019\_paper.html](https://openaccess.thecvf.com/content_ICCV_2019/html/Cho_On_the_Efficacy_of_Knowledge_Distillation_ICCV_2019_paper.html).

## Logit-Based, Feature-Based, and Relational Knowledge Transfer

* Logit-based transfer: Matches teacher and student output distributions, transferring information about relative class probabilities [https\://arxiv.org/pdf/1503.02531](https://arxiv.org/pdf/1503.02531).  
* Feature-based transfer: Uses intermediate teacher representations as supervision; adaptation layers may be needed to reconcile teacher and student feature dimensions [https\://arxiv.org/pdf/1412.6550](https://arxiv.org/pdf/1412.6550).  
* Relational transfer: Matches relationships among examples, such as distances and angles between representations, rather than matching each representation independently [https\://openaccess.thecvf.com/content\_CVPR\_2019/html/Park\_Relational\_Knowledge\_Distillation\_CVPR\_2019\_paper.html](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html).  
* These approaches differ in the information transferred, architectural compatibility requirements, and training overhead.  
* Combining output, feature, and relational losses introduces additional objectives whose reliability and interaction must be considered.

# Teacher Reliability and Selective Transfer

## What reliability means in this review

* Correctness: Does the teacher's highest-probability class match the available training label? Conditional T/S provides a direct precedent for correctness-dependent teacher admission. [https\://doi.org/10.1109/icassp.2019.8683438](https://doi.org/10.1109/icassp.2019.8683438)  
* Confidence: How much probability does the teacher assign to a prediction or to the known correct class? These are different quantities: a confidently wrong teacher can have high maximum probability and low probability on the correct class. CCKD uses the latter. [https\://arxiv.org/pdf/2107.06993](https://arxiv.org/pdf/2107.06993)  
* Calibration: Does stated confidence correspond to empirical correctness frequency? Raw softmax confidence is not automatically calibrated; temperature scaling is one calibration procedure. The temperature used for calibration has a different purpose from the temperature used to soften KD targets. [https\://proceedings.mlr.press/v70/guo17a.html](https://proceedings.mlr.press/v70/guo17a.html)  
* Student usefulness: Would following the teacher help this student at this stage? Correctness and confidence alone do not establish this. Gradient-selection methods and repaired-target methods address different aspects of that question. [https\://doi.org/10.1109/ICCV48922.2021.00501](https://doi.org/10.1109/ICCV48922.2021.00501); [https\://doi.org/10.1109/iccv51701.2025.00111](https://doi.org/10.1109/iccv51701.2025.00111)  
* Applicability limit: Calibration assessed in-distribution may deteriorate under dataset shift. Agreement with observed labels also inherits any label errors; the proposed gate assumes those labels are trustworthy. [https\://papers.neurips.cc/paper\_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html](https://papers.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html).

## Conditional Teacher-Student Learning — correctness-dependent admission

* Source and status: Zhong Meng, Jinyu Li, Yong Zhao, Yifan Gong; ICASSP 2019\. Full text, selected methods. [https\://doi.org/10.1109/icassp.2019.8683438](https://doi.org/10.1109/icassp.2019.8683438).  
* Reliability signal: Whether the teacher predicts the known correct class.  
* Ground-truth supervision: Uses hard-label targets when the teacher is wrong; uses teacher targets when it is correct. It does not retain a separate all-example ground-truth CE term in the same way as the proposed method.  
* Update mechanism: Switch the supervision target, then optimize the resulting loss. No PCD program is established in this source.  
* Strength: A direct and interpretable precedent for excluding teacher supervision based on teacher correctness.  
* Limitation: Correctness alone does not measure calibration, student compatibility, or the usefulness of the teacher's remaining class probabilities. The reported setting is speech adaptation.

## Confidence Conditioned Knowledge Distillation — CCKD

* Source and status: Sourav Mishra and Suresh Sundaram; arXiv preprint, 2021; no later publication verified in the packet. Full text, selected methods. [https\://arxiv.org/pdf/2107.06993](https://arxiv.org/pdf/2107.06993).  
* Reliability signal: Teacher probability assigned to the known correct class, used as a sample-dependent confidence weight.  
* Ground-truth supervision: The loss-based variant uses λ L\_KD+(1-λ)L\_CE; the label coefficient therefore varies with confidence. The target-based variant blends teacher and one-hot targets rather than retaining an independent, fixed CE term. The self-regulation variant can skip samples the student already finds easy.  
* Update mechanism: Change sample loss weights or construct confidence-conditioned targets; optional student-based sample selection. This differs from solving PCD constraints.  
* Strength: Uses label information to reduce reliance on teacher predictions that assign little probability to the correct class.  
* Limitation: A soft correct-class confidence weight is different from a Boolean teacher-correctness AND maximum-confidence test. Its variants also have different supervision behavior and should not be described as one identical gate.

## Hard Gate Knowledge Distillation — calibrated supervision routing

* Source and status: Dongkyu Lee et al.; EMNLP 2022\. Full text, selected methods. [https\://aclanthology.org/2022.emnlp-main.665/](https://aclanthology.org/2022.emnlp-main.665/).  
* Reliability signal: Compares student probability on the true token or sentence with the corresponding calibrated-teacher quantity.  
* Ground-truth supervision: Routes between label and teacher supervision; the label branch can be off when the teacher branch is on.  
* Update mechanism: A hard instance-dependent switch between supervision sources. It does not use the proposed PCD program.  
* Strength: Makes teacher admission conditional and explicitly considers calibration and the student's current state.  
* Limitation: The NLG setting and student-versus-teacher comparison differ from the proposal's teacher argmax correctness plus a fixed confidence threshold.

## Confidence-Aware Multi-Teacher KD — CA-MKD

* Source and status: Hailin Zhang, Defang Chen, Can Wang; ICASSP 2022\. Full text, selected methods. [https\://arxiv.org/pdf/2201.00007](https://arxiv.org/pdf/2201.00007).  
* Reliability signal: Teacher cross-entropy against the ground-truth label determines sample-wise relative teacher weights.  
* Ground-truth supervision: Ground-truth labels are explicitly used to assess teacher reliability. The retained extraction does not establish the exact persistence of a separate student CE term on every example; that stronger comparison remains unchecked.  
* Update mechanism: Adaptively weight and integrate information from multiple teachers, including intermediate features.  
* Strength: Uses label agreement to distinguish teacher quality for each example rather than assigning equal trust to every teacher.  
* Limitation: Relative weighting among several teachers is different from hard admission of one teacher's signal. It changes both the teacher configuration and transferred knowledge.

## Label Revision and Data Selection — repair or select teacher guidance

* Source and status: Weichao Lan et al.; IEEE Transactions on Cognitive and Developmental Systems, 2025\. Selected full text inspected in the preprint; published metadata verified. [https\://doi.org/10.1109/tcds.2025.3559881](https://doi.org/10.1109/tcds.2025.3559881).  
* Reliability signal: Label information identifies teacher errors; influence-based scores select examples for distillation.  
* Ground-truth supervision: Examples outside the selected distillation subset remain under label supervision. The exact final-journal loss formulation was not compared line by line with the preprint.  
* Update mechanism: Revise teacher targets and restrict which examples receive teacher supervision; no PCD mechanism was established.  
* Strength: Offers an alternative to accepting every teacher output without correction.  
* Limitation: Influence-based selection is different from the proposed correctness-plus-confidence predicate. The final author PDF was inaccessible during the recorded pass.

## Knowledge Distillation with Refined Logits — retain useful parts of imperfect advice

* Source and status: Wujie Sun et al.; ICCV 2025\. Full text, selected methods. [https\://doi.org/10.1109/iccv51701.2025.00111](https://doi.org/10.1109/iccv51701.2025.00111).  
* Reliability signal: Label-conditioned refinement separates sample confidence and selected class correlations from misleading teacher rankings.  
* Ground-truth supervision: Labels participate in constructing the refined supervision. The retained evidence does not independently settle whether every configuration keeps a separate, fixed all-example CE term; do not equate label use with that guarantee.  
* Update mechanism: Modify the transferred logit information and loss rather than impose the proposed PCD constraint.  
* Strength: Shows a concrete alternative to discarding an incorrect teacher's entire output distribution.  
* Limitation: Useful class correlations do not establish that every wrong prediction helps; target repair also changes what is transferred.

## CHAD — a nearby reliability-aware preprint

* Source and status: Sumit et al., "When Does Knowledge Distillation Hurt? Reliability-Aware Distillation for Low-Resource Language Summarization"; arXiv preprint, 2026\. Full text, selected methods/results. [https\://arxiv.org/pdf/2607.19956](https://arxiv.org/pdf/2607.19956).  
* Reliability signal: Alignment of a sample's KD gradient with a validation-loss gradient, used to train a sample-weight predictor.  
* Update mechanism: Learn sample-specific KD weights from a local estimate of usefulness, rather than apply the proposed correctness-plus-confidence gate and PCD.  
* Strength: Makes the distinction between teacher confidence and student usefulness explicit; teacher-only confidence gates offered little benefit in its reported summarization comparison.  
* Limitation: Alignment is a local proxy for usefulness, not a causal retraining test of each sample's harm. The multilingual extension evaluates EWAD+CPDP, not CHAD, and does not establish CHAD's multilingual effectiveness.

# Gradient-Based and Multi-Objective KD Methods

## Why a reliability gate and a gradient rule answer different questions

* Meaning of an update: A training step changes the student parameters using a direction computed from its losses. In ordinary weighted KD, that direction combines the supervised and teacher-loss gradients with selected weights.  
* Meaning of conflict: For ordinary gradient-descent directions, a negative inner product between two loss gradients indicates opposing local first-order preferences. This is an optimization property, distinct from whether the teacher predicted the correct class. [https\://arxiv.org/pdf/2001.06782](https://arxiv.org/pdf/2001.06782); [https\://doi.org/10.1109/ICCV48922.2021.00501](https://doi.org/10.1109/ICCV48922.2021.00501)  
* Illustrative calculation: At temperature 1, teacher probabilities (0.85,0.10,0.05), student probabilities (0.95,0.03,0.02), and label class 1 pass a teacher threshold of 0.8. Their CE and forward-KL logit gradients have inner product \-0.0077. A correct, confident teacher can therefore still conflict with the current student's label gradient.  
* Limit of that example: This is a local logit-space calculation, not an experiment or a guarantee about parameter-space gradients in every network.

## PCGrad — remove conflicting gradient components

* Source and status: Tianhe Yu et al.; NeurIPS 2020\. Abstract and metadata inspected only; discussion is limited to the mechanism stated by the authors. [https\://arxiv.org/pdf/2001.06782](https://arxiv.org/pdf/2001.06782).  
* Update mechanism: Project a conflicting task gradient onto the normal plane of another task gradient.  
* Strength: Provides a direct optimization-level response to interference without requiring a teacher-confidence model.  
* Limitation: The original evidence is multi-task learning. It does not establish a supervised-primary hierarchy or an effective PCD-KD baseline implementation.

## CAGrad — balance average progress and the least-improving objective

* Source and status: Bo Liu et al.; NeurIPS 2021\. Abstract and metadata inspected only; no detailed theorem or implementation claim is made here. [https\://arxiv.org/pdf/2110.14048](https://arxiv.org/pdf/2110.14048).  
* Update mechanism: Optimize average loss while regularizing the direction using the worst local objective improvement.  
* Strength: Offers a comparator that addresses objective imbalance and interference together.  
* Limitation: Its convergence statements require their own assumptions and settings; the source abstract does not verify effectiveness in this KD application.

## Student Customized Knowledge Distillation — SCKD

* Source and status: Yichen Zhu and Yi Wang; ICCV 2021\. Full text, selected methods; published p. 5 visually checked. [https\://doi.org/10.1109/ICCV48922.2021.00501](https://doi.org/10.1109/ICCV48922.2021.00501).  
* Reliability/compatibility signal: Cosine similarity between the supervised gradient and individual output- or feature-distillation gradients. This tests the student's optimization state, not the teacher's top-1 correctness and confidence.  
* Ground-truth supervision: The supervised loss remains as an ungated term in Eq. (6); separate KD terms are switched on or off.  
* Update mechanism: Gate individual KD losses using gradient similarity, then optimize the remaining supervised-plus-KD loss. This is not the PCD constrained program.  
* Strength: A close precedent for supervised-primary, selective KD; it already distinguishes the main task from optional teacher objectives.  
* Limitation: Gating by gradient similarity alone does not establish teacher reliability, nor does it prove better final generalization.  
* Unresolved implementation issue — verify later: Published Sec. 3.4 and Algorithm 1 disable a KD term for cosine greater than or equal to a threshold, despite surrounding explanations about eliminating conflicting gradients. Confirm threshold direction, per-iteration gate reset, gradient parameter subset, and intended behavior against authoritative code. Do not silently reverse the published condition.

## Dynamic Trade-off Optimization for KD — DTO-KD

* Source and status: Zeeshan Hayder et al.; ICLR 2026\. Full text, selected methods. [https\://proceedings.iclr.cc/paper\_files/paper/2026/hash/e546ce4230c15bc422a500bb47b341ea-Abstract-Conference.html](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e546ce4230c15bc422a500bb47b341ea-Abstract-Conference.html).  
* Reliability/compatibility signal: Task and distillation loss gradients and relative progress; no matching correctness-plus-confidence admission gate was established.  
* Ground-truth supervision: The task loss remains part of the objective vector, but its coefficient is dynamically optimized. This differs from fixing CE as the preferred gradient in PCD; strictly positive task weight throughout training is not verified here.  
* Update mechanism: The ideal formulation combines gradients of log task and log distillation losses through a multi-objective weight problem. The practical method amortizes weight adaptation; the two should not be treated as identical implementations.  
* Strength: A direct KD optimization comparator addressing conflict and dominance rather than relying on fixed task/KD weights.  
* Limitation: Its practical recipe includes representation projections and task/distillation heads; a plain-logit adaptation would need to be described explicitly. Its theoretical claims are not automatically guarantees for the project's recipe.  
* Unresolved implementation issue — verify later: Eq. (9) refers to simplex weights, while the displayed Eq. (10) lists only a sum-to-one constraint. Check nonnegative-weight handling, clamping or renormalization, identical/opposed-gradient cases, and zero-loss safeguards for logarithms against authoritative code.

## Earlier gradient-space KD and cross-method synthesis

* Earlier KD/MOO precedent: Shangchen Du et al., "Agree to Disagree: Adaptive Ensemble Knowledge Distillation in Gradient Space," NeurIPS 2020, formulates ensemble KD in gradient space with a disagreement tolerance. Only its official abstract and publisher metadata were inspected; exact loss handling and optimization details remain outside verified scope. [https\://proceedings.neurips.cc/paper\_files/paper/2020/file/91c77393975889bd08f301c9e13a44b7-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2020/file/91c77393975889bd08f301c9e13a44b7-Paper.pdf)  
* PCGrad/CAGrad: General optimization ideas that can be adapted to KD; their original papers do not establish the proposed KD application. [https\://arxiv.org/pdf/2001.06782](https://arxiv.org/pdf/2001.06782); [https\://arxiv.org/pdf/2110.14048](https://arxiv.org/pdf/2110.14048)  
* SCKD/DTO-KD: Optimization or gradient-selection methods already applied directly to distillation. Their existence narrows the contribution beyond simply "address gradient conflict in KD." [https\://doi.org/10.1109/ICCV48922.2021.00501](https://doi.org/10.1109/ICCV48922.2021.00501); [https\://proceedings.iclr.cc/paper\_files/paper/2026/hash/e546ce4230c15bc422a500bb47b341ea-Abstract-Conference.html](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e546ce4230c15bc422a500bb47b341ea-Abstract-Conference.html)

## Medical SCOPE — confidence-aware relations and classification-priority correction

* Source and status: Tao Chen et al.; Journal of Imaging, 2026\. Full text, selected methods/experiments. [https\://doi.org/10.3390/jimaging12090436](https://doi.org/10.3390/jimaging12090436).  
* Reliability signal: Entropy-derived confidence weights for pairwise teacher relations, combined with gradient-conflict detection.  
* Ground-truth supervision: Classification supervision is retained as the reference objective; the gradient-correction component modifies conflicting distillation contributions.  
* Update mechanism: Selective Relation Alignment regulates relational teacher information; Gradient Conflict Resolution removes distillation-gradient components that conflict with classification.  
* Strength: A particularly relevant precedent because it combines reliability-sensitive teacher information with asymmetric treatment of classification and KD.  
* Limitation: Its entropy-weighted relations differ from a hard per-example correctness-plus-confidence gate. Its full system also contains auxiliary priors and relational losses; the retained extraction does not support transcribing every equation exactly.

## Task-Preserving Knowledge Distillation — TPKD

* Source and status: Qianfeng Yuan and Wenbing Tao; "Learning Beyond Full Imitation: Task-Preserving Knowledge Distillation," arXiv preprint, 2026\. Full text, selected theory/methods. [https\://arxiv.org/pdf/2609.39338](https://arxiv.org/pdf/2609.39338).  
* Reliability/protection signal: Correct-class margin behavior in output space, rather than the proposed hard teacher-correctness-plus-confidence test.  
* Ground-truth supervision: Retains the label-gradient contribution while conditionally modifying teacher directions.  
* Update mechanism: Project teacher directions to protect correct-class margins in output space. This differs from requiring a minimum amount of progress on an admitted secondary objective.  
* Strength: A close asymmetric-KD precedent that explicitly questions unrestricted imitation.  
* Limitation: Output-space protection transfers to parameter updates only under stated conditions; it is not a universal neural-training guarantee. The reported projection-removal ablation is closely matched, so the projection's isolated benefit should not be overstated.

## Asymmetry and anchored projection before PCD

* Dynamic Barrier Gradient Descent — Gong, Xingchao Liu, Qiang Liu, NeurIPS 2021: An earlier construction stays close to a preferred gradient subject to a half-space requirement. Its barrier schedule and normalization differ from PCD. It is optimization ancestry, not a verified correctness-gated KD implementation. Full text, selected passages. [https\://papers.nips.cc/paper\_files/paper/2021/hash/f7b027d45fd7484f6d0833823b98907e-Abstract.html](https://papers.nips.cc/paper_files/paper/2021/hash/f7b027d45fd7484f6d0833823b98907e-Abstract.html)  
* Pareto Navigation Gradient Descent — Ye and Liu, UAI 2022: Uses a reference criterion and gradient constraints to navigate within a Pareto set. It supports preference-aware constrained optimization as prior work, but does not establish the proposed gate or PCD-KD. Full text, selected passages. [https\://proceedings.mlr.press/v180/ye22a.html](https://proceedings.mlr.press/v180/ye22a.html)  
* AuxiNash — Shamsian et al., ICML 2023: Explicitly treats auxiliary learning as asymmetric bargaining and learns preferences for the main task. Its bargaining and bilevel preference mechanism differs from PCD's anchored program; no matching KD reliability gate was verified. Full text, selected passages. [https\://proceedings.mlr.press/v202/shamsian23a.html](https://proceedings.mlr.press/v202/shamsian23a.html)  
* Gradient-similarity auxiliary gating — Yunshu Du et al., arXiv preprint, first posted 2018; v2 updated 2020: Uses main/auxiliary gradient cosine similarity, including a binary auxiliary-admission variant. The main loss supplies the reference direction; gating is based on compatibility, not teacher correctness. This predates SCKD but does not establish PCD-KD or improved final generalization. Full text, selected passages. [https\://arxiv.org/pdf/1812.02224](https://arxiv.org/pdf/1812.02224)  
* Synthesis: Asymmetry, gradient gating, and anchored projection each have antecedents. PCD's own related-work discussion acknowledges dynamic-barrier ancestry; the proposed project should credit these ingredients while distinguishing their combination and application. [https\://openreview.net/forum?id=HT01yGHLEt](https://openreview.net/forum?id=HT01yGHLEt)

## Original PCD — the optimization foundation

* Source and status: Dara Varam and Mohamed I. AlHajri; "Not All Objectives Are Born Equal: Priority-Constrained Descent for Hierarchical Multi-Objective Optimization," 2026\. TMLR status is reported in the arXiv journal reference; direct OpenReview verification was blocked in the recorded pass. Full text, selected methods/theory. [https\://openreview.net/forum?id=HT01yGHLEt](https://openreview.net/forum?id=HT01yGHLEt).  
* Update mechanism: Normalize objective gradients using moving averages; choose the direction closest to the primary gradient subject to secondary-progress inequalities. If the primary direction already satisfies them, no directional correction is needed.  
* Limitation: Primary priority means proximity to the primary gradient in the optimization objective. It does not impose a supervised-descent inequality or an upper bound on angular deviation. Strong conflict can produce a direction that fails to descend the primary loss. [https\://openreview.net/forum?id=HT01yGHLEt](https://openreview.net/forum?id=HT01yGHLEt)

# Open Challenges and Future Directions

* Rejection may discard useful information: Refined Logits preserves selected teacher correlations; label-revision methods repair guidance instead of simply dropping it. These alternatives weaken a universal argument for rejecting every incorrect teacher output. [https\://doi.org/10.1109/iccv51701.2025.00111](https://doi.org/10.1109/iccv51701.2025.00111); [https\://doi.org/10.1109/tcds.2025.3559881](https://doi.org/10.1109/tcds.2025.3559881)  
* Teacher accuracy is not transfer utility: Yuan et al.'s CVPR 2020 "Revisiting Knowledge Distillation via Label Smoothing Regularization" reports useful weak/reversed-teacher settings. That does not mean every wrong example helps, but it rules out treating teacher weakness as automatic uselessness. Full text, selected arguments/experiments. [https\://doi.org/10.1109/cvpr42600.2020.00396](https://doi.org/10.1109/cvpr42600.2020.00396)  
* Imitation and generalization differ: Stanton et al., NeurIPS 2021, show that teacher-student fidelity and student generalization need not improve together. KD-loss reduction is therefore not itself evidence of better student performance. Full text, selected results/discussion. [https\://proceedings.neurips.cc/paper\_files/paper/2021/file/376c6b9ff3bedbbea56751a84fffc10c-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2021/file/376c6b9ff3bedbbea56751a84fffc10c-Paper.pdf)  
* Recipe choices can change conclusions: Beyer et al., CVPR 2022, identify training duration and consistent teacher/student image views as consequential. Any later optimizer comparison must separate the direction rule from recipe differences. Full text, selected methods. [https\://openaccess.thecvf.com/content/CVPR2022/html/Beyer\_Knowledge\_Distillation\_A\_Good\_Teacher\_Is\_Patient\_and\_Consistent\_CVPR\_2022\_paper.html](https://openaccess.thecvf.com/content/CVPR2022/html/Beyer_Knowledge_Distillation_A_Good_Teacher_Is_Patient_and_Consistent_CVPR_2022_paper.html)  
* No demonstrated effectiveness: PCD, FedPCD, SCKD, DTO-KD, and SCOPE results cannot be presented as PCD-KD results. The proposed evaluation remains prospective.