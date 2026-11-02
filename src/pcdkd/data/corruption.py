"""Seeded teacher-output corruption maps."""
import json

import numpy as np


def make_corruption_map(labels, train_indices, rate, seed):
    labels = np.asarray(labels)
    train_indices = np.asarray(train_indices, dtype=int)
    if not 0 <= rate <= 1 or np.any((train_indices < 0) | (train_indices >= len(labels))):
        raise ValueError("invalid rate or training indices")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(train_indices, round(rate * len(train_indices)), replace=False)
    classes = np.unique(labels)
    wrong = {int(i): int(rng.choice(classes[classes != labels[i]])) for i in chosen}
    return wrong


def corrupted_teacher_distribution(logits_or_probs, labels, corruption_map):
    values = np.asarray(logits_or_probs, dtype=np.float64)
    labels = np.asarray(labels)
    row_sums = values.sum(axis=1, keepdims=True)
    if np.all(values >= 0) and np.allclose(row_sums, 1):
        probs = values.copy()
    else:
        probs = np.exp(values - values.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
    out = probs.copy()
    k = probs.shape[1]
    for raw_i, raw_wrong in corruption_map.items():
        i, wrong = int(raw_i), int(raw_wrong)
        if wrong == labels[i] or not 0 <= wrong < k:
            raise ValueError("corruption class must be valid and differ from label")
        confidence = probs[i].max()
        out[i] = (1 - confidence) / (k - 1)
        out[i, wrong] = confidence
    return out


def save_corruption_map(path, mapping):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({str(k): int(v) for k, v in sorted(mapping.items())}, f, sort_keys=True)


def load_corruption_map(path):
    with open(path, encoding="utf-8") as f:
        return {int(k): int(v) for k, v in json.load(f).items()}
