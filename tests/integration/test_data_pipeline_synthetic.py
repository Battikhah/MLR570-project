import numpy as np

from pcdkd.data import make_corruption_map, stratified_split


def test_synthetic_pipeline_split_and_map():
    labels = np.repeat(np.arange(10), 50)
    train, val = stratified_split(labels, 50, 7)
    mapping = make_corruption_map(labels, train, 0.1, 8)
    assert len(mapping) == 45 and not set(mapping) & set(val)
