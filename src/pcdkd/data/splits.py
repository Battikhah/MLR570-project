"""Deterministic stratified dataset splits."""
import numpy as np


def stratified_split(labels, val_size=5000, split_seed=0):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not 0 <= val_size < len(labels):
        raise ValueError("labels must be 1D and val_size in [0, len(labels))")
    rng = np.random.default_rng(split_seed)
    train, val = [], []
    classes = np.unique(labels)
    class_ids = [np.flatnonzero(labels == cls) for cls in classes]
    raw = np.asarray([len(ids) for ids in class_ids]) * (val_size / len(labels))
    allocations = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - allocations))[: val_size - allocations.sum()]:
        allocations[i] += 1
    for ids, nval in zip(class_ids, allocations):
        ids = ids.copy()
        rng.shuffle(ids)
        val.extend(ids[:nval])
        train.extend(ids[nval:])
    return np.asarray(sorted(train)), np.asarray(sorted(val))
