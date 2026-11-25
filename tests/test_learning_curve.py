"""lib.learning_curve (nested stratified subsamples, fold summary)."""
import numpy as np
import pandas as pd
import pytest
from lib.learning_curve import nested_stratified_subsamples, summarize_learning_curve


def test_subsamples_are_nested_and_stratified():
    y = np.array([1] * 40 + [0] * 160)
    subsamples = nested_stratified_subsamples(y, [0.5, 0.1, 1.0], np.random.default_rng(0))
    assert list(subsamples) == [0.1, 0.5, 1.0]
    assert len(subsamples[0.1]) == 20 and y[subsamples[0.1]].sum() == 4
    assert len(subsamples[0.5]) == 100 and y[subsamples[0.5]].sum() == 20
    assert set(subsamples[0.1]) <= set(subsamples[0.5]) <= set(subsamples[1.0])
    np.testing.assert_array_equal(subsamples[1.0], np.arange(200))


def test_subsamples_minimum_per_class_and_validation():
    y = np.array([1] * 5 + [0] * 95)
    small = nested_stratified_subsamples(y, [0.1], np.random.default_rng(0), min_per_class=3)
    assert y[small[0.1]].sum() == 3
    with pytest.raises(ValueError):
        nested_stratified_subsamples(y, [0.0, 1.0], np.random.default_rng(0))


def test_summary_per_fraction():
    curve = pd.DataFrame(
        {
            "model": "Full",
            "fold": [0, 1, 0, 1],
            "fraction": [0.5, 0.5, 1.0, 1.0],
            "n_train": [100, 100, 200, 200],
            "n_events": [20, 20, 40, 40],
            "events_per_feature": [1.0, 1.0, 2.0, 2.0],
            "train_AUC": [0.99, 0.97, 0.95, 0.93],
            "test_AUC": [0.70, 0.74, 0.78, 0.80],
        }
    )
    summary = summarize_learning_curve(curve)
    assert summary["fraction"].tolist() == [0.5, 1.0]
    assert summary["test_AUC_mean"].tolist() == pytest.approx([0.72, 0.79])
    assert summary["AUC_gap_mean"].tolist() == pytest.approx([0.98 - 0.72, 0.94 - 0.79])
    assert summary["n_folds"].tolist() == [2, 2]
