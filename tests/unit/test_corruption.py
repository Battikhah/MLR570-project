import numpy as np

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
    out = corrupted_teacher_distribution(logits, labels, mapping)
    assert np.allclose(out.sum(1), 1)
    for i, cls in mapping.items():
        assert out[i, cls] == out[i].max()


def test_zero_rate_is_identity():
    labels = np.array([0, 1, 2])
    logits = np.array([[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]])
    assert make_corruption_map(labels, np.arange(3), 0, 0) == {}
    expected = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    assert np.allclose(corrupted_teacher_distribution(logits, labels, {}), expected)


def test_map_json_roundtrip(tmp_path):
    mapping = {2: 1, 4: 3}
    path = tmp_path / "map.json"
    save_corruption_map(path, mapping)
    assert load_corruption_map(path) == mapping
