"""Reproducible random seeding helpers."""
import random

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seed NumPy's global API for reproducible callers.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    seed_everything(seed)


def dataloader_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
