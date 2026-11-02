import random

import numpy as np
import torch

from pcdkd.utils.seed import dataloader_generator, seed_everything


def test_seed_everything_and_loader_generator():
    seed_everything(12)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    seed_everything(12)
    assert first == (random.random(), np.random.rand(), torch.rand(1).item())
    assert torch.equal(torch.randperm(20, generator=dataloader_generator(4)), torch.randperm(20, generator=dataloader_generator(4)))
