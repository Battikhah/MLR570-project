# Assignment requirements and project objectives

Extracted from `Project Files/` at commit `07af803`. Cross-checked by two independent agents
(Orca Run `run_1a90ee37419e`). PDF page numbers are physical pages; the proposal's printed
"Page 1 of 3" is physical p. 2.

## Course and team (VERIFIED — proposal p. 1)

MLR570 Advanced Machine Learning, AUS CSE. Team: Khalid Hamdan, Aly Khaled Ismail Mahmoud,
Hasan Akel, Salim Akel. No roles are assigned in the documents.

## Deliverables

1. **Survey paper** (governed by `Survey_Guideline.pdf`; example: `Survey_Sample.pdf`;
   draft: `Current_Survey.md`).
2. **ML project** (governed by `MLR570_Project_Proposal_Submission.pdf`).

## Deadlines, length, rubric

**None stated** in any provided document (VERIFIED by full-text search of all four files).
Open questions for the instructor: length/page limit, deadline(s), grading rubric and weights,
whether IEEE citation style is mandatory ("such as IEEE"), whether per-claim citations are expected,
template/format.

## Survey guideline checklist (VERIFIED — Survey_Guideline.pdf)

| # | Requirement | Page | Key wording |
|---|---|---|---|
| G1 | Descriptive title | 1 | "reflect the key focus of the paper" |
| G2 | Abstract: motivation, key contributions, high-level structure, key trends | 1 | "the motivation for the survey, the key contributions, and the high-level structure" |
| G3 | Introduction — problem statement, importance, recent trends | 1 | "why this field is important, recent trends, and developments" |
| G4 | Introduction — motivation for the survey; gaps | 1 | "challenges or gaps in the field and why a comprehensive overview is necessary" |
| G5 | Introduction — scope, categorisation dimensions, outline | 1 | "Clearly define the scope"; "provide an outline of the rest of the paper" |
| G6 | Detailed discussion of influential methods per category, with comparisons | 1 | "how different methodologies solve similar problems or how one approach improves on another" |
| G7 | Visual aids (figures, diagrams of models/workflows/comparisons) | 1 | "Include figures and diagrams to illustrate key concepts" |
| G8 | Critical analysis: strengths, weaknesses, applicability; open problems | 2 | "Highlight both strengths and weaknesses, as well as their applicability" |
| G9 | Future directions grounded in identified gaps | 2 | "based on the gaps or limitations identified" |
| G10 | Conclusion | 2 | "Recap the important contributions" |
| G11 | References: consistent format (such as IEEE), relevant, recent, properly attributed | 2 | "Use a consistent format (such as IEEE)" |

## Project objectives (VERIFIED — proposal pp. 2–3)

- **Research question:** "We ask whether reliability-gated PCD can improve student accuracy and
  reduce negative transfer when teacher predictions are unreliable." (p. 2, §2)
- **Method:** frozen teacher; L1 = cross-entropy on all examples; L2 = temperature-scaled KL on
  examples where the teacher is correct **and** max probability > q; PCD combines them with the
  supervised objective as priority (d* = argmin ½‖d−h1‖² s.t. h2ᵀd ≥ τ‖h2‖²), moving-average
  gradient normalisation, rescaling to the supervised-gradient norm; SGD without momentum, then
  "a smaller set of experiments with momentum". (pp. 2–3, §3)
- **Data/models:** CIFAR-100 main, CIFAR-10 early checks [7]; 5,000 of 50,000 training images for
  validation; test set only for final evaluation; teacher CIFAR ResNet-32; students ResNet-8
  (main) and ResNet-14 (capacity check). (p. 3, §4)
- **Comparisons:** label-only; weighted KD, DTO-KD, PCGrad, CAGrad, SCKD, SCOPE gradient
  correction; ungated vs gated PCD-KD; gated weighted KD; confidence-only vs correctness+confidence
  gate; τ ∈ {0, 0.1, 0.3, 0.5}; "comparable search effort". (p. 3)
- **Robustness test:** replace 10% or 30% of teacher outputs with high-confidence wrong predictions.
- **Metrics:** top-1 test accuracy (mean, SD, paired differences over 5 seeds); accuracy drop,
  negative transfer vs label-only, training time, gate rate, gradient alignment,
  secondary-progress satisfaction.
- **Not specified:** timeline, roles, compute budget, statistical test, augmentation/training
  recipe, how "comparable search effort" is measured, corruption assignment protocol.

## Known gaps in the current survey draft (VERIFIED — see Run `run_1a90ee37419e` reports)

Placeholder abstract (L7); no survey motivation/scope/roadmap; no figures or tables; no
conclusion; no reference list (49 inline links, 28 unique URLs); PCGrad/CAGrad reviewed from
abstracts only (L119, L126); "negative transfer", CIFAR, ResNet absent; "FedPCD" mentioned at
L195 without introduction or citation; PCD venue stated as TMLR in the proposal but "not directly
verified" in the draft (L185); research-process notes at L57, L77, L89, L139, L149, L185.
