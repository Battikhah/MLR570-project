"""Typed experiment configuration loading with strict keys."""
import math
from dataclasses import dataclass, fields
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
    method: str = "pcd"
    lambda_grid: list[float] = None
    tau_grid: list[float] = None
    beta: float = 0.9
    epsilon: float = 1e-8
    gate_mode: str = "none"
    gate_q: float = 0.8
    momentum: float = 0.0
    weight_decay: float = 0.0
    lr_schedule: str = "cosine"
    val_size: int = 5000
    temperature_grid: list[float] = None
    select_temperature_on_val: bool = True

    def __post_init__(self):
        if self.lambda_grid is None:
            self.lambda_grid = [0.0, 0.1, 0.5, 1.0]
        if self.tau_grid is None:
            self.tau_grid = [0.05, 0.1, 0.3]
        if self.temperature_grid is None:
            self.temperature_grid = [2.0, 4.0, 8.0]


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
    int_fields = {"num_classes", "seed", "split_seed", "batch_size", "epochs", "val_size"}
    float_fields = {
        "corruption_rate", "tau", "temperature", "learning_rate", "beta", "epsilon",
        "gate_q", "momentum", "weight_decay",
    }
    for field in fields(config):
        value = getattr(config, field.name)
        if field.name in int_fields and (type(value) is not int):
            raise ValueError(f"{field.name} must be an integer")
        if field.name in float_fields and (
            type(value) not in (int, float) or not math.isfinite(value)
        ):
            raise ValueError(f"{field.name} must be a finite number")
        if field.name in {"lambda_grid", "tau_grid", "temperature_grid"} and (
            not isinstance(value, list)
            or not value
            or any(type(item) not in (int, float) or not math.isfinite(item) for item in value)
        ):
            raise ValueError(f"{field.name} must be a nonempty list of finite numbers")
        if field.name in {
            "dataset", "teacher", "student", "method", "gate_mode", "lr_schedule"
        } and not isinstance(value, str):
            raise ValueError(f"{field.name} must be a string")
    if type(config.select_temperature_on_val) is not bool:
        raise ValueError("select_temperature_on_val must be a boolean")
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
    if config.method not in {
        "ce", "wkd", "nwkd", "pcgrad_onto_primary", "pcd", "adaptive_lambda"
    }:
        raise ValueError("unsupported method")
    if config.gate_mode not in {"none", "conf_only", "correct_and_conf"}:
        raise ValueError("unsupported gate_mode")
    if config.lr_schedule not in {"constant", "step", "cosine"}:
        raise ValueError("unsupported lr_schedule")
    if (
        not 0 <= config.corruption_rate <= 1
        or config.batch_size < 1
        or config.epochs < 1
        or config.val_size < 0
    ):
        raise ValueError("invalid rate, batch_size, epochs, or val_size")
    if not 0 <= config.beta < 1 or config.epsilon <= 0 or not 0 <= config.gate_q <= 1:
        raise ValueError("invalid beta, epsilon, or gate_q")
    if config.momentum < 0 or config.weight_decay < 0:
        raise ValueError("momentum and weight_decay must be nonnegative")
    if any(value < 0 for value in config.lambda_grid + config.tau_grid):
        raise ValueError("lambda_grid and tau_grid values must be nonnegative")
    if any(value <= 0 for value in config.temperature_grid):
        raise ValueError("temperature_grid values must be positive")
    return config
