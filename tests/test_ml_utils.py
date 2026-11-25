"""compute_results: vectorized metrics equal the sklearn/imblearn ones."""
import warnings

import numpy as np
import pytest
from imblearn.metrics import sensitivity_score, specificity_score
from lib.ml_utils import compute_results
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score


@pytest.mark.parametrize("seed", range(5))
def test_metrics_match_sklearn(seed):
    warnings.simplefilter("ignore")  # sklearn warns when a metric is undefined (returns 0)
    rng = np.random.default_rng(seed)
    y = (rng.random(80) < 0.4).astype(int)
    prob = np.clip(0.3 * y + rng.random(80) * 0.7, 0, 1).round(2)
    ths = [0.0, 0.05, 0.3, 0.5, 0.8, 1.0]  # includes "all positive" and "none positive"
    rows = compute_results(3, "m", prob, y, [], ths_range=ths)
    assert len(rows) == len(ths)
    for row, t in zip(rows, ths):
        pred = (prob > t).astype(int)
        expected = [
            balanced_accuracy_score(y, pred), roc_auc_score(y, prob),
            f1_score(y, pred), specificity_score(y, pred),
            sensitivity_score(y, pred), recall_score(y, pred),
        ]
        assert row[:6] == [3, "m", False, 0, t, 0]
        assert row[6:] == pytest.approx(expected)


def test_single_threshold():
    rows = compute_results(0, "m", np.array([0.2, 0.7, 0.9, 0.1]), np.array([0, 1, 1, 0]), [], ths_range=0.5)
    assert len(rows) == 1 and rows[0][7] == 1.0
