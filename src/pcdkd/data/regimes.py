"""Training-set regimes for the experiment suite: label noise, long tail, low data.

All functions are pure NumPy, seeded, and operate on *original dataset indices*, so they compose
with :func:`pcdkd.data.splits.stratified_split` and the teacher-corruption map.
"""

import numpy as np


def label_noise(labels, train_indices, rate, seed, kind="symmetric"):
    """Copy of ``labels`` with ``round(rate * len(train_indices))`` training labels flipped.

    ``symmetric``: the new label is uniform over the other classes.
    ``asymmetric``: the new label is ``(y + 1) % C`` (class-dependent "pair flip" noise).
    Only training indices change; validation/test labels are untouched.
    """
    labels = np.asarray(labels)
    if not 0 <= rate <= 1:
        raise ValueError("rate must be in [0, 1]")
    if kind not in ("symmetric", "asymmetric"):
        raise ValueError("kind must be 'symmetric' or 'asymmetric'")
    classes = int(labels.max()) + 1
    rng = np.random.default_rng(seed)
    chosen = rng.permutation(np.asarray(train_indices, dtype=int))[
        : round(rate * len(train_indices))
    ]
    noisy = labels.copy()
    for i in chosen:
        if kind == "asymmetric":
            noisy[i] = (labels[i] + 1) % classes
        else:
            noisy[i] = rng.choice(np.delete(np.arange(classes), labels[i]))
    return noisy


def long_tail_indices(labels, train_indices, imbalance, seed):
    """Exponential long-tail subsample (Cui et al. 2019 profile): class k keeps
    ``n_max * imbalance ** (-k / (C - 1))`` examples; ``imbalance = 1`` keeps everything."""
    labels = np.asarray(labels)
    train_indices = np.asarray(train_indices, dtype=int)
    if imbalance < 1:
        raise ValueError("imbalance must be >= 1")
    classes = np.unique(labels[train_indices])
    rng = np.random.default_rng(seed)
    per_class = [train_indices[labels[train_indices] == c] for c in classes]
    n_max = min(len(ids) for ids in per_class)
    keep = []
    for k, ids in enumerate(per_class):
        n = max(1, int(n_max * imbalance ** (-k / max(len(classes) - 1, 1))))
        keep.extend(rng.permutation(ids)[:n])
    return np.sort(np.asarray(keep, dtype=int))


def fraction_indices(labels, train_indices, fraction, seed):
    """Per-class stratified subsample keeping ``fraction`` of each class (at least one example)."""
    labels = np.asarray(labels)
    train_indices = np.asarray(train_indices, dtype=int)
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(labels[train_indices]):
        ids = train_indices[labels[train_indices] == c]
        keep.extend(rng.permutation(ids)[: max(1, round(fraction * len(ids)))])
    return np.sort(np.asarray(keep, dtype=int))


def class_groups(train_labels, classes, many=100, few=20):
    """Many/medium/few-shot class groups by training count (Liu et al. 2019 thresholds)."""
    counts = np.bincount(np.asarray(train_labels), minlength=classes)
    return {
        "many": np.flatnonzero(counts > many),
        "medium": np.flatnonzero((counts <= many) & (counts >= few)),
        "few": np.flatnonzero(counts < few),
    }
