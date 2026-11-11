"""pcdkd.train: the single-run trainer shared by teachers and students."""

from pcdkd.train.trainer import evaluate, make_combiner, train

__all__ = ["evaluate", "make_combiner", "train"]
