# lopezpaz2017gem — Gradient Episodic Memory (GEM)

Read 2026-11-02 (targeted passages only, arXiv 1706.08840v6 PDF, p. 4–5, Eqs. 9–11).
Saved locally as `literature/pdfs/lopezpaz2017gem.pdf`.

## Why it matters here

VERIFIED: GEM solves the primal QP minimize_z ½‖g − z‖² s.t. ⟨z, g_k⟩ ≥ 0 for every past task k. Here g is the
current-task gradient and g_k are gradients on episodic memories. It solves the dual in (t−1) variables. The authors
also write: "In practice, we found that adding a small constant γ ≥ 0 to v* biased the gradient projection to updates
that favoured beneficial backwards transfer."

INTERPRETATION: this is the PCD QP with a priority order (current task primary, memories secondary) and τ = 0.
The GEM γ-margin is applied in the dual rather than as a normalized τ-fraction margin, and there is no gradient
normalization. Any "PCD for continual learning" project is therefore a re-parameterization of GEM plus EMA
normalization, and must be positioned as such. A-GEM (`chaudhry2019agem`) is the single averaged-constraint
(K = 2) case. VERIFIED (varam2026pcd v2, Sec. 2 "Anchored Projections and Dynamic Barriers", p. 4): the PCD
authors themselves state that GEM and A-GEM "are algorithmically equivalent to PCD with τ = 0", and that
ε-SOFT-GEM (`hu2020softgem`) "adds a unit-normalized positive margin at K = 2 in the same continual-learning
setting". So even the normalized-margin variant has prior art in continual learning.

## Benchmarks used by GEM (from the paper; recalled at the dataset level only)

MNIST Permutations, MNIST Rotations, CIFAR-100 incremental. The exact protocols were not re-read in this pass.
