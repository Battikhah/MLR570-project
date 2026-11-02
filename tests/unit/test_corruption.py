import numpy as np
import pytest

from pcdkd.data.corruption import (
    corrupted_teacher_distribution,
    load_corruption_map,
    make_corruption_map,
    save_corruption_map,
)


def test_corruption_is_seeded_training_only_and_confident_wrong():
    labels = np.tile(np.arange(4), 20)
    train = np.arange(60)
    mapping = make_corruption_map(labels, train, 0.3, 10)
    assert len(mapping) == round(0.3 * len(train)) and set(mapping) <= set(train)
    assert mapping == make_corruption_map(labels, train, 0.3, 10)
    assert all(labels[i] != cls for i, cls in mapping.items())
    logits = np.tile([4., 1., 0., -1.], (len(labels), 1))
    out = corrupted_teacher_distribution(logits, labels, mapping, kind="logits")
    assert out.dtype == np.float32
    assert np.allclose(out.sum(1), 1)
    for i, cls in mapping.items():
        assert out[i, cls] == out[i].max()
        probs = np.exp(logits[i] - np.logaddexp.reduce(logits[i]))
        assert np.isclose(out[i, cls], probs.max(), rtol=1e-6)
        other = np.arange(len(probs)) != cls
        assert np.allclose(out[i, other], (1 - probs.max()) / (len(probs) - 1), rtol=1e-6)


def test_zero_rate_is_identity():
    labels = np.array([0, 1, 2])
    logits = np.array([[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]])
    assert make_corruption_map(labels, np.arange(3), 0, 0) == {}
    expected = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    assert np.allclose(corrupted_teacher_distribution(logits, labels, {}, kind="logits"), expected)


def test_saturated_logits_keep_teacher_confidence_in_log_space():
    labels = np.array([0])
    logits = np.array([[0.0, -20.0, -30.0]], dtype=np.float64)
    mapping = {0: 1}
    logs = corrupted_teacher_distribution(
        logits, labels, mapping, kind="logits", return_log_probs=True
    )
    probs = corrupted_teacher_distribution(logits, labels, mapping, kind="logits")
    # The wrong class receives the teacher maximum, while the other classes retain nonzero log mass.
    assert logs[0, 1] < 0
    assert np.isclose(np.exp(logs[0, 1]), np.exp(logits[0, 0] - np.logaddexp.reduce(logits[0])))
    assert np.all(np.isfinite(logs))
    assert probs.dtype == np.float32


def test_kind_is_required_and_probabilities_are_explicit():
    labels = np.array([0])
    values = np.array([[1.0, 0.0, 0.0]])
    with pytest.raises(TypeError):
        corrupted_teacher_distribution(values, labels, {})
    with pytest.raises(ValueError, match="kind"):
        corrupted_teacher_distribution(values, labels, {}, kind="auto")
    probs = corrupted_teacher_distribution(values, labels, {}, kind="probs")
    assert np.allclose(probs, values)


def test_corruption_maps_are_nested_across_rates():
    labels = np.tile(np.arange(5), 100)
    train = np.arange(len(labels))
    low = make_corruption_map(labels, train, 0.1, 17)
    high = make_corruption_map(labels, train, 0.3, 17)
    assert set(low) <= set(high)
    assert all(high[i] == wrong for i, wrong in low.items())


def test_map_json_roundtrip(tmp_path):
    mapping = {2: 1, 4: 3}
    path = tmp_path / "map.json"
    save_corruption_map(path, mapping)
    assert load_corruption_map(path) == mapping
