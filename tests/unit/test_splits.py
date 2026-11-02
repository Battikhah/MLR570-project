import numpy as np
from PIL import Image
from torchvision.transforms import RandomCrop, RandomHorizontalFlip

import pcdkd.data.cifar as cifar
from pcdkd.data.splits import stratified_split


def test_stratified_split_properties_and_seed():
    labels = np.repeat(np.arange(10), 100)
    train, val = stratified_split(labels, 50, 3)
    train2, val2 = stratified_split(labels, 50, 3)
    _, val3 = stratified_split(labels, 50, 4)
    assert len(train) == 950 and len(val) == 50
    assert not set(train) & set(val) and sorted(np.r_[train, val]) == list(range(1000))
    assert np.all(np.bincount(labels[val]) == 5)
    assert np.array_equal(val, val2) and not np.array_equal(val, val3)


def test_cifar_splits_val_is_unaugmented_and_preserves_original_index(monkeypatch):
    class SyntheticCifar:
        targets = [0, 0, 1, 1, 2, 2, 3, 3]

        def __init__(self, root, *, train, download, transform):
            assert train and not download and transform is None

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            return Image.new("RGB", (32, 32)), self.targets[index]

    monkeypatch.setattr(cifar, "CIFAR10", SyntheticCifar)
    train, val = cifar.cifar_splits("unused", val_size=4, split_seed=9)
    assert len(train) == len(val) == 4
    train_ops = train.dataset.transform.transforms
    val_ops = val.dataset.transform.transforms
    assert any(isinstance(op, (RandomCrop, RandomHorizontalFlip)) for op in train_ops)
    assert not any(isinstance(op, (RandomCrop, RandomHorizontalFlip)) for op in val_ops)
    for subset in (train, val):
        for image, _, index in subset:
            assert image.shape == (3, 32, 32)
            assert index in subset.indices
