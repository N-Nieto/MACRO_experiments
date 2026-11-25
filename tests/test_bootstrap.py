"""Patient-level bootstrap of the cross-validated AUC."""
import numpy as np
import pandas as pd
import pytest
from lib.bootstrap import (
    _bootstrap_weights,
    _RepetitionAUC,
    bootstrap_auc,
    bootstrap_auc_difference,
    read_fold_predictions,
    single_test_set,
)
from sklearn.metrics import roc_auc_score


def test_weighted_auc_matches_sklearn_with_ties_and_weights():
    rng = np.random.default_rng(0)
    pred = rng.integers(0, 20, 300) / 20  # many ties
    y = (rng.random(300) < 0.4).astype(int)
    weights = rng.integers(0, 3, (5, 300)).astype(float)
    ours = _RepetitionAUC(np.arange(300), pred, y)(weights)
    for k in range(5):
        assert ours[k] == pytest.approx(roc_auc_score(y, pred, sample_weight=weights[k]))


def test_bootstrap_weights_resample_all_patients():
    w = _bootstrap_weights(50, 7, np.random.default_rng(1))
    assert w.shape == (7, 50) and (w.sum(axis=1) == 50).all()


def make_cv_predictions(tmp_path, n=200, n_splits=5, n_repetitions=3, signal=1.0, seed=0, name="m"):
    rng = np.random.default_rng(seed)
    ids = np.array([f"p{i}" for i in range(n)])
    y = (rng.random(n) < 0.45).astype(int)
    preds, ys, pids = [], [], []
    for _ in range(n_repetitions):
        folds = np.array_split(rng.permutation(n), n_splits)
        for test in folds:
            preds.append(1 / (1 + np.exp(-(signal * (2 * y[test] - 1) + rng.normal(0, 1, len(test))))))
            ys.append(y[test])
            pids.append(ids[test])
    for values, suffix in [(preds, "predictions"), (ys, "y_true"), (pids, "patient_ID")]:
        pd.DataFrame([list(v) for v in values]).T.to_csv(tmp_path / f"{name}_{suffix}.csv")
    return [tmp_path / f"{name}_{s}.csv" for s in ("predictions", "y_true", "patient_ID")]


def test_cv_auc_is_mean_of_pooled_repetitions(tmp_path):
    files = make_cv_predictions(tmp_path)
    long = read_fold_predictions(*files, n_splits=5)
    assert long["repetition"].nunique() == 3 and long.groupby("repetition").size().eq(200).all()
    expected = np.mean([roc_auc_score(d["y"], d["prediction"]) for _, d in long.groupby("repetition")])
    result = bootstrap_auc(long, n_boot=300)
    assert result["AUC"] == pytest.approx(expected)
    assert result["CI_low"] < result["AUC"] < result["CI_high"]
    assert result["n_patients"] == 200


def test_paired_difference_detects_a_better_model(tmp_path):
    good = read_fold_predictions(*make_cv_predictions(tmp_path, signal=2.0, name="good"), n_splits=5)
    weak = read_fold_predictions(*make_cv_predictions(tmp_path, signal=0.3, name="weak"), n_splits=5)
    result = bootstrap_auc_difference(good, weak, n_boot=300)
    assert result["difference"] > 0 and result["CI_low"] > 0 and result["p_value"] < 0.05
    same = bootstrap_auc_difference(good, good, n_boot=50)
    assert same["difference"] == 0


def test_single_test_set():
    rng = np.random.default_rng(2)
    y = (rng.random(150) < 0.5).astype(int)
    pred = y + rng.normal(0, 1, 150)
    result = bootstrap_auc(single_test_set(pred, y, np.arange(150)), n_boot=200)
    assert result["AUC"] == pytest.approx(roc_auc_score(y, pred))


def test_fold_summary_uses_every_fold(tmp_path):
    from lib.bootstrap import fold_auc_summary
    long = read_fold_predictions(*make_cv_predictions(tmp_path), n_splits=5)
    summary = fold_auc_summary(long)
    per_fold = [roc_auc_score(d["y"], d["prediction"]) for _, d in long.groupby("fold")]
    assert len(per_fold) == 15
    assert summary["fold_AUC_mean"] == pytest.approx(np.mean(per_fold))
