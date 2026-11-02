"""CIFAR datasets returning image, label, and original index."""
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor

CIFAR10_MEAN, CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN, CIFAR100_STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        return x, y, index


def cifar_dataset(root, *, classes=10, train=True, download=False, augment=None):
    if classes not in (10, 100):
        raise ValueError("classes must be 10 or 100")
    mean, std = (CIFAR10_MEAN, CIFAR10_STD) if classes == 10 else (CIFAR100_MEAN, CIFAR100_STD)
    transforms = []
    if augment is None:
        augment = train
    if augment:
        transforms.extend((RandomCrop(32, padding=4), RandomHorizontalFlip()))
    transforms.extend((ToTensor(), Normalize(mean, std)))
    dataset_class = CIFAR10 if classes == 10 else CIFAR100
    dataset = dataset_class(root, train=train, download=download, transform=Compose(transforms))
    return IndexedDataset(dataset)
