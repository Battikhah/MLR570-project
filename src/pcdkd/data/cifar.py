"""CIFAR datasets returning image, label, and original index."""
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor

from .splits import stratified_split

CIFAR10_MEAN, CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN, CIFAR100_STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)


class IndexedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y, index


def _transform(classes, augment):
    mean, std = (CIFAR10_MEAN, CIFAR10_STD) if classes == 10 else (CIFAR100_MEAN, CIFAR100_STD)
    transforms = []
    if augment:
        transforms.extend((RandomCrop(32, padding=4), RandomHorizontalFlip()))
    transforms.extend((ToTensor(), Normalize(mean, std)))
    return Compose(transforms)


def cifar_dataset(root, *, classes=10, train=True, download=False, augment=None):
    if classes not in (10, 100):
        raise ValueError("classes must be 10 or 100")
    if augment is None:
        augment = train
    dataset_class = CIFAR10 if classes == 10 else CIFAR100
    dataset = dataset_class(root, train=train, download=download, transform=None)
    return IndexedDataset(dataset, _transform(classes, augment))


def cifar_splits(root, *, classes=10, val_size=5000, split_seed=0, download=False):
    """Return augmented train and deterministic unaugmented val subsets with original indices."""
    if classes not in (10, 100):
        raise ValueError("classes must be 10 or 100")
    dataset_class = CIFAR10 if classes == 10 else CIFAR100
    base = dataset_class(root, train=True, download=download, transform=None)
    train_indices, val_indices = stratified_split(base.targets, val_size, split_seed)
    train = Subset(IndexedDataset(base, _transform(classes, True)), train_indices.tolist())
    val = Subset(IndexedDataset(base, _transform(classes, False)), val_indices.tolist())
    return train, val
