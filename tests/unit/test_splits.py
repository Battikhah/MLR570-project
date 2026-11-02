import numpy as np

from pcdkd.data.splits import stratified_split


def test_stratified_split_properties_and_seed():
    labels = np.repeat(np.arange(10), 100)
    train, val = stratified_split(labels, 50, 3)
    train2, val2 = stratified_split(labels, 50, 3)
    _, val3 = stratified_split(labels, 50, 4)
    assert len(train) == 950 and len(val) == 50
    assert not set(train) & set(val) and sorted(np.r_[train, val]) == list(range(1000))
    assert np.all(np.bincount(labels[val]) == 5)
    assert np.array_equal(val, val2) and not np.array_equal(val, val3)
