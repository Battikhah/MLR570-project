"""Seeded, nested teacher-output corruption maps and distributions."""
import json

import numpy as np


def make_corruption_map(labels, train_indices, rate, seed):
    labels = np.asarray(labels)
    train_indices = np.asarray(train_indices, dtype=int)
    if not 0 <= rate <= 1 or np.any((train_indices < 0) | (train_indices >= len(labels))):
        raise ValueError("invalid rate or training indices")
    classes = np.unique(labels)
    if len(classes) < 2:
        raise ValueError("at least two classes are required")
    streams = np.random.SeedSequence(seed).spawn(2)
    order = np.random.default_rng(streams[0]).permutation(train_indices)
    count = round(rate * len(train_indices))
    class_rng = np.random.default_rng(streams[1])
    return {
        int(index): int(class_rng.choice(classes[classes != labels[index]]))
        for index in order[:count]
    }


def corrupted_teacher_distribution(
    logits_or_probs, labels, corruption_map, *, kind, return_log_probs=False
):
    """Return float32 probabilities or float64 log-probabilities.

    Log-probability output preserves confidence for saturated logits that cannot
    be represented by a float32 probability row; cast only when a consumer needs
    probabilities on-device.
    """
    values = np.asarray(logits_or_probs, dtype=np.float64)
    labels = np.asarray(labels)
    if values.ndim != 2 or values.shape[0] != len(labels) or values.shape[1] < 2:
        raise ValueError("expected (N, C) values and N labels, with C >= 2")
    if kind not in {"logits", "probs"}:
        raise ValueError("kind must be 'logits' or 'probs'")
    if kind == "logits":
        if not np.isfinite(values).all():
            raise ValueError("logits must be finite")
        shifted = values - values.max(axis=1, keepdims=True)
        log_probs = shifted - np.logaddexp.reduce(shifted, axis=1, keepdims=True)
    else:
        if np.any(values < 0) or not np.isfinite(values).all():
            raise ValueError("probabilities must be finite and nonnegative")
        sums = values.sum(axis=1, keepdims=True)
        if np.any(sums <= 0) or not np.allclose(sums, 1):
            raise ValueError("probability rows must sum to one")
        with np.errstate(divide="ignore"):
            log_probs = np.log(values)

    out = log_probs.copy()
    classes = values.shape[1]
    for raw_i, raw_wrong in corruption_map.items():
        i, wrong = int(raw_i), int(raw_wrong)
        if not 0 <= i < len(labels) or wrong == labels[i] or not 0 <= wrong < classes:
            raise ValueError("corruption index/class must be valid and class differ from label")
        teacher = log_probs[i]
        true_class = int(np.argmax(teacher))
        log_confidence = teacher[true_class]
        other = np.arange(classes) != true_class
        log_other_mass = np.logaddexp.reduce(teacher[other])
        out[i] = log_other_mass - np.log(classes - 1)
        out[i, wrong] = log_confidence

    if return_log_probs:
        return out
    return np.exp(out).astype(np.float32)


def save_corruption_map(path, mapping):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({str(k): int(v) for k, v in sorted(mapping.items())}, f, sort_keys=True)


def load_corruption_map(path):
    with open(path, encoding="utf-8") as f:
        return {int(k): int(v) for k, v in json.load(f).items()}
