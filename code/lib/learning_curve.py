"""
Learning curves (Reviewer 3, comment 6): performance as a function of the
number of training patients, to show how far the gap between training and
held-out AUC closes as the training set grows.

Within each outer CV fold, the training patients are subsampled to a series of
fractions. The subsamples are nested (every smaller subsample is contained in
the larger ones) and stratified (same event rate as the full training fold),
so the curve reflects the amount of data and not a different draw of patients.
The held-out test fold is always the complete fold.
"""

import numpy as np
import pandas as pd
from lib.experiment_config import QUICK
from lib.ml_utils import get_inner_loop_optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

# Fractions of the training fold (quick mode: three)
LEARNING_CURVE_FRACTIONS = (
    (0.25, 0.5, 1.0) if QUICK else (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)


def nested_stratified_subsamples(
    y: np.ndarray,
    fractions: tuple[float, ...] | list[float],
    rng: np.random.Generator,
    min_per_class: int = 1,
) -> dict[float, np.ndarray]:
    """
    Positions of nested, stratified subsamples of y, one per fraction.

    Each class is shuffled once and the first round(fraction x class size)
    positions of every class are taken, so a smaller fraction is always a
    subset of a larger one and the class proportions are kept.

    Parameters
    ----------
    y : array of class labels
    fractions : fractions in (0, 1]
    rng : random generator (fixes the shuffle)
    min_per_class : minimum number of patients of each class in a subsample
        (e.g. the number of inner CV folds)

    Returns
    -------
    dict fraction -> sorted positions (np.ndarray of int) into y
    """
    y = np.asarray(y)
    if any(not 0 < f <= 1 for f in fractions):
        raise ValueError(f"fractions must be in (0, 1], got {list(fractions)}")
    shuffled = {c: rng.permutation(np.flatnonzero(y == c)) for c in np.unique(y)}
    subsamples = {}
    for fraction in sorted(fractions):
        positions = []
        for c, idx in shuffled.items():
            n = max(int(round(fraction * len(idx))), min(min_per_class, len(idx)))
            positions.append(idx[:n])
        subsamples[fraction] = np.sort(np.concatenate(positions))
    return subsamples


def run_learning_curve(
    X: pd.DataFrame,
    Y: np.ndarray,
    model_name: str,
    params_optuna: dict,
    fractions: tuple[float, ...] | list[float] = LEARNING_CURVE_FRACTIONS,
    n_splits: int = 10,
    inner_n_splits: int = 3,
    random_state: int = 23,
) -> pd.DataFrame:
    """
    Learning curve over the outer folds of one repetition of the repeated
    stratified K-fold CV. With the same n_splits and random_state, these are
    the folds of the first repetition of the main experiment.

    In every outer fold and for every training fraction, the hyperparameters
    are tuned with Optuna on the inner CV of the subsample and the model is
    fitted on the subsample (lib.ml_utils.get_inner_loop_optuna), as in the
    main experiment.

    Returns
    -------
    pd.DataFrame, one row per outer fold and fraction: model, fold, fraction,
    n_train, n_events, n_features, events_per_feature, train_AUC (in-sample,
    on the training subsample), test_AUC (complete held-out fold) and the
    selected hyperparameters.
    """
    kf_out = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=1, random_state=random_state)
    kf_inner = StratifiedKFold(n_splits=inner_n_splits, shuffle=True, random_state=random_state)
    rng = np.random.default_rng(random_state)
    rows = []
    for i_fold, (train_index, test_index) in enumerate(kf_out.split(X, Y)):
        X_test, Y_test = X.iloc[test_index], Y[test_index]
        subsamples = nested_stratified_subsamples(
            Y[train_index], fractions, rng, min_per_class=inner_n_splits
        )
        for fraction, positions in subsamples.items():
            X_train, Y_train = X.iloc[train_index[positions]], Y[train_index[positions]]
            print(f"{model_name} fold {i_fold}: {fraction:.0%} of the training fold ({len(Y_train)} patients)")
            tuned = get_inner_loop_optuna(X_train, Y_train, kf_inner, params_optuna)
            model = tuned["model"]
            n_events = int(Y_train.sum())
            rows.append(
                {
                    "model": model_name,
                    "fold": i_fold,
                    "fraction": fraction,
                    "n_train": len(Y_train),
                    "n_events": n_events,
                    "n_features": X.shape[1],
                    "events_per_feature": n_events / X.shape[1],
                    "train_AUC": roc_auc_score(Y_train, model.predict_proba(X_train)[:, 1]),
                    "test_AUC": roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1]),
                    **tuned["best_params"],
                    "n_estimators": model.n_estimators,
                }
            )
    return pd.DataFrame(rows)


def summarize_learning_curve(curve: pd.DataFrame) -> pd.DataFrame:
    """
    Mean and SD over the outer folds of the per-fold learning-curve results
    (columns model, fraction, n_train, n_events, events_per_feature,
    train_AUC, test_AUC), one row per model and fraction.
    """
    grouped = curve.groupby(["model", "fraction"])
    summary = grouped[["n_train", "n_events", "events_per_feature"]].mean()
    for metric in ("train_AUC", "test_AUC"):
        summary[f"{metric}_mean"] = grouped[metric].mean()
        summary[f"{metric}_sd"] = grouped[metric].std()
    summary["AUC_gap_mean"] = summary["train_AUC_mean"] - summary["test_AUC_mean"]
    summary["n_folds"] = grouped.size()
    return summary.reset_index()
