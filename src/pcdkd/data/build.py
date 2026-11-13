"""Build train/val/test sets for one run spec, applying the training-set regime.

Datasets yield ``(image, clean_label, original_index)``. The labels the student trains on (possibly
noisy) are returned separately as ``train_labels`` and looked up by original index, so the clean
label stays available for diagnostics and the teacher-corruption map keys stay valid.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR10, CIFAR100

from .cifar import IndexedDataset, _transform
from .corruption import make_corruption_map
from .regimes import fraction_indices, label_noise, long_tail_indices
from .splits import stratified_split


class SyntheticCifar(Dataset):
    """CIFAR-shaped tensors with a learnable class signal (CPU smoke tests; no download)."""

    def __init__(self, n, classes, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.targets = torch.arange(n) % classes
        patterns = torch.randn(classes, 3, 32, 32, generator=g)
        self.x = patterns[self.targets] + 2.0 * torch.randn(n, 3, 32, 32, generator=g)
        self.targets = self.targets.tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.x[i], self.targets[i]


def _bases(dataset, root, synthetic):
    classes = {"cifar10": 10, "cifar100": 100}[dataset]
    if synthetic:
        n_train = max(10 * classes, 400)
        return classes, SyntheticCifar(n_train, classes, 0), SyntheticCifar(2 * classes, classes, 1)
    cls = CIFAR10 if classes == 10 else CIFAR100
    return classes, cls(root, train=True, download=False), cls(root, train=False, download=False)


def build_data(spec, root="data", synthetic=False):
    """Return a dict with train/val/test datasets, label tensors and the corruption map.

    ``spec`` keys used: dataset, val_size, split_seed, seed, train_fraction, imbalance,
    label_noise, noise_kind, teacher_corruption. ``val_size: 0`` trains on the whole training set
    and returns ``val`` = None.
    """
    classes, base, test_base = _bases(spec["dataset"], root, synthetic)
    labels = np.asarray(base.targets)
    val_size = min(spec["val_size"], len(labels) // 10) if synthetic else spec["val_size"]
    train_idx, val_idx = stratified_split(labels, val_size, spec["split_seed"])
    # subsets and label noise are fixed by split_seed; teacher corruption varies with seed
    if spec.get("imbalance", 1) > 1:
        train_idx = long_tail_indices(labels, train_idx, spec["imbalance"], spec["split_seed"])
    if spec.get("train_fraction", 1.0) < 1.0:
        train_idx = fraction_indices(labels, train_idx, spec["train_fraction"], spec["split_seed"])
    noisy = labels
    if spec.get("label_noise", 0.0) > 0:
        noisy = label_noise(  # a dataset property (split_seed): teacher and student share it
            labels,
            train_idx,
            spec["label_noise"],
            spec["split_seed"],
            spec.get("noise_kind", "symmetric"),
        )
    corruption = {}
    if spec.get("teacher_corruption", 0.0) > 0:
        corruption = make_corruption_map(
            labels, train_idx, spec["teacher_corruption"], spec["seed"]
        )

    aug = None if synthetic else _transform(classes, True)
    plain = None if synthetic else _transform(classes, False)
    return {
        "classes": classes,
        "train": Subset(IndexedDataset(base, aug), train_idx.tolist()),
        "val": Subset(IndexedDataset(base, plain), val_idx.tolist()) if len(val_idx) else None,
        "test": IndexedDataset(test_base, plain),
        "train_indices": train_idx,
        "train_labels": torch.as_tensor(noisy, dtype=torch.long),  # by original index
        "clean_labels": torch.as_tensor(labels, dtype=torch.long),
        "corruption_map": corruption,
    }
