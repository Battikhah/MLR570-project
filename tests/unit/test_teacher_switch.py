"""Decision 0006: teacher corruption that switches on partway through training."""

import torch

from pcdkd.experiments.suite import build_specs
from pcdkd.train.trainer import _corruption_active, _teacher_targets


class _Teacher(torch.nn.Module):
    def forward(self, x, return_features=False):
        return torch.nn.functional.one_hot(torch.arange(len(x)) % 5, 5).float() * 10


def _data():
    labels = torch.arange(4) % 5
    return {"corruption_map": {1: 3, 2: 4}, "clean_labels": labels}, labels


def test_corruption_switches_on_at_the_given_fraction():
    spec = {"epochs": 60, "corrupt_from": 0.5}
    assert not _corruption_active(spec, 29) and _corruption_active(spec, 30)
    assert _corruption_active({"epochs": 60}, 0)  # default: corrupted from the start


def test_teacher_targets_are_clean_before_the_switch_and_corrupted_after():
    data, labels = _data()
    spec = {"epochs": 60, "corrupt_from": 0.5}
    x, idx = torch.zeros(4, 1), torch.arange(4)
    before, _, f0 = _teacher_targets(spec, [_Teacher()], x, idx, labels, data, "cpu", False, 29)
    after, _, f1 = _teacher_targets(spec, [_Teacher()], x, idx, labels, data, "cpu", False, 30)
    assert torch.equal(before[0].argmax(1), labels) and not f0.any()
    assert f1.tolist() == [False, True, True, False]
    assert torch.equal(after[0].argmax(1)[[0, 3]], labels[[0, 3]])
    assert (after[0].argmax(1)[[1, 2]] != labels[[1, 2]]).all()


def test_teacher_switch_family_reuses_the_teacher_corruption_teacher():
    screen, _ = build_specs("screen", families=["teacher-switch", "teacher-corruption"])
    sw = [s for s in screen if s["family"] == "teacher-switch"]
    tc = next(s for s in screen if s["family"] == "teacher-corruption")
    assert len(sw) == 2 * 6 * 5 and {s["corrupt_from"] for s in sw} == {0.5}
    assert {s["teacher_keys"][0] for s in sw} == {tc["teacher_keys"][0]}
    assert "corrupt_from" not in tc
