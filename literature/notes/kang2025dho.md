# kang2025dho: Dual-Head Optimization (DHO) for semi-supervised KD from VLMs

Read 2026-11-02 (cross-review round 2): §2.3 and App. G.2 of arXiv 2505.07675v3, saved locally as
`literature/pdfs/kang2025dho.pdf`. The arXiv comment says "Accepted at NeurIPS 2026" (author-stated).

## Why it matters

It is the only source I found that reports **measured** CE–KD gradient cosines in parameter space.

## Theory and measurements

- **[THEORY] Eq. 3, Prop. 1, Cor. 1.** At a linear head, cos(∇_W L_CE, ∇_W L_KD) ∝ (p̂−y)ᵀ(p̂−p)·‖z‖².
  This equals ‖p̂−m‖² − r² with m = (y+p)/2 and r = ‖y−p‖/2. Conflict is therefore guaranteed whenever p̂ lies
  inside the ball whose diameter is the segment [y, p]. For p̂ = (1−t)y + tp, the value is −t(1−t)‖p−y‖² < 0.
  Prop. 2 argues that the conflict propagates to backbone parameters through the PSD form JᵀWᵀWJ.
- **[EE] Fig. 3.** The average cosine over 10 datasets "turns negative" during training. Exact values are graphical only.
- **[EE] App. G.2, Table 26.** Single-head CE+KD (SHO), final epoch, mean over 8 datasets (Caltech-101, DTD,
  EuroSAT, FGVC-Aircraft, Flowers, Pets, Cars, UCF101), 16-shot, ResNet-18 student, CLIP RN50 zero-shot teacher:

  | Parameters | cos(∇L_CE, ∇L_KD) |
  |---|---|
  | layer1 | −0.3924 |
  | layer2 | −0.3931 |
  | layer3 | −0.4048 |
  | layer4 | −0.4526 |
  | classifier head | −0.7570 |

- **[EE] Table 27.** Most severe layer4 values: EuroSAT −0.80, DTD −0.67, FGVC-Aircraft −0.62.

## Scope caveats (INTERPRETATION)

- The setting is few-shot labels with a zero-shot VLM teacher, not full-data CIFAR-100 with a trained CNN teacher.
- The values are per-layer cosines at the final epoch, not a whole-vector per-step distribution.
- Teacher corruption is not studied.
