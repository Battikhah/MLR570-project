"""Typed experiment configuration loading with strict keys."""
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ExperimentConfig:
    dataset: str = "cifar10"
    num_classes: int = 10
    seed: int = 0
    split_seed: int = 0
    corruption_rate: float = 0.3
    batch_size: int = 128
    epochs: int = 1
    teacher: str = "resnet32"
    student: str = "resnet8"
    tau: float = 0.1
    temperature: float = 4.0
    learning_rate: float = 0.1


def load_config(path):
    with Path(path).open(encoding="utf-8") as f:
        values = yaml.safe_load(f) or {}
    if not isinstance(values, dict):
        raise ValueError("configuration must be a mapping")
    allowed = ExperimentConfig.__dataclass_fields__
    unknown = set(values) - set(allowed)
    if unknown:
        raise ValueError(f"unknown config keys: {', '.join(sorted(unknown))}")
    config = ExperimentConfig(**values)
    expected_classes = {"cifar10": 10, "cifar100": 100}
    if (
        config.dataset not in expected_classes
        or expected_classes[config.dataset] != config.num_classes
    ):
        raise ValueError("dataset/classes must be cifar10/10 or cifar100/100")
    if not 0 <= config.corruption_rate <= 1 or config.batch_size < 1 or config.epochs < 1:
        raise ValueError("invalid rate, batch_size, or epochs")
    if config.temperature <= 0 or config.learning_rate <= 0 or config.tau < 0:
        raise ValueError("temperature and learning_rate must be positive; tau nonnegative")
    return config
