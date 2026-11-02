import random
from unittest.mock import patch

import numpy as np
import torch

from pcdkd.utils.seed import dataloader_generator, seed_everything


def test_seed_everything_and_loader_generator():
    seed_everything(12)
    first = (random.random(), np.random.default_rng(12).random(), torch.rand(1).item())
    seed_everything(12)
    second = (random.random(), np.random.default_rng(12).random(), torch.rand(1).item())
    assert first == second
    left = torch.randperm(20, generator=dataloader_generator(4))
    right = torch.randperm(20, generator=dataloader_generator(4))
    assert torch.equal(left, right)


def test_seed_everything_seeds_numpy_global():
    with patch("pcdkd.utils.seed.np.random.seed") as numpy_seed:
        seed_everything(12)
    numpy_seed.assert_called_once_with(12)
