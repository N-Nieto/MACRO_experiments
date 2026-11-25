"""
Patient-level bootstrap confidence intervals for the AUC of cross-validated
(repeated k-fold) and external-validation predictions.

For repeated k-fold CV, each repetition gives one out-of-fold prediction per
patient. The AUC of a model is the mean over repetitions of the AUC of the
pooled out-of-fold predictions. Bootstrap samples resample patients (with
replacement) and apply the same patient weights to every repetition, so the
interval reflects the uncertainty due to the finite cohort rather than the
fold-to-fold spread. Paired comparisons use the same resampled patients for
both models.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def read_fold_predictions(
    pred_file: Path, y_file: Path, id_file: Path, n_splits: int
) -> pd.DataFrame:
    """
    Out-of-fold predictions saved as one column per outer fold (rows padded
    with NaN), as long table: patient_ID, fold, repetition, prediction, y.
    The repetition of fold f is f // n_splits.
    """
    pred = pd.read_csv(pred_file, index_col=0)
    y = pd.read_csv(y_file, index_col=0)
    ids = pd.read_csv(id_file, index_col=0)
    rows = []
    for i_fold, col in enumerate(pred.columns):
        mask = pred[col].notna()
        rows.append(
            pd.DataFrame(
                {
                    "patient_ID": ids.loc[mask, col].astype(str).to_numpy(),
                    "fold": i_fold,
                    "repetition": i_fold // n_splits,
                    "prediction": pred.loc[mask, col].to_numpy(dtype=float),
                    "y": y.loc[mask, col].to_numpy().astype(int),
                }
            )
        )
    long = pd.concat(rows, ignore_index=True)
    duplicated = long.duplicated(["patient_ID", "repetition"])
    if duplicated.any():
        raise ValueError(f"{pred_file}: a patient appears twice in one repetition")
    return long


def single_test_set(prediction: np.ndarray, y: np.ndarray, ids: np.ndarray) -> pd.DataFrame:
    """External validation (one prediction per patient) in the same long format."""
    return pd.DataFrame(
        {
            "patient_ID": np.asarray(ids).astype(str),
            "fold": 0,
            "repetition": 0,
            "prediction": np.asarray(prediction, dtype=float),
            "y": np.asarray(y).astype(int),
        }
    )


class _RepetitionAUC:
    """AUC of one repetition for many patient-weight vectors at once."""

    def __init__(self, patient_index: np.ndarray, prediction: np.ndarray, y: np.ndarray):
        values, groups = np.unique(prediction, return_inverse=True)
        self.patient_index = patient_index
        self.positive = y == 1
        # one-hot matrix: prediction -> group of tied predictions (sorted)
        self.groups = np.zeros((len(prediction), len(values)))
        self.groups[np.arange(len(prediction)), groups] = 1.0

    def __call__(self, weights: np.ndarray) -> np.ndarray:
        """weights: (n_samples, n_patients) -> AUC per sample (Mann-Whitney, ties 1/2)."""
        w = weights[:, self.patient_index]
        pos = (w * self.positive) @ self.groups
        neg = (w * ~self.positive) @ self.groups
        neg_below = np.cumsum(neg, axis=1) - neg
        numerator = np.sum(pos * (neg_below + 0.5 * neg), axis=1)
        return numerator / (pos.sum(axis=1) * neg.sum(axis=1))


def _auc_function(long: pd.DataFrame, patients: np.ndarray):
    position = pd.Series(np.arange(len(patients)), index=patients)
    long = long[long["patient_ID"].isin(position.index)]
    per_repetition = [
        _RepetitionAUC(
            position.loc[d["patient_ID"]].to_numpy(), d["prediction"].to_numpy(), d["y"].to_numpy()
        )
        for _, d in long.groupby("repetition")
    ]
    return lambda weights: np.mean([auc(weights) for auc in per_repetition], axis=0)


def _bootstrap_weights(n_patients: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Patient counts of n_boot bootstrap samples, shape (n_boot, n_patients)."""
    return rng.multinomial(n_patients, np.full(n_patients, 1 / n_patients), size=n_boot).astype(float)


def bootstrap_auc(
    long: pd.DataFrame, n_boot: int = 2000, seed: int = 23, alpha: float = 0.05
) -> dict:
    """AUC (mean over repetitions of the pooled out-of-fold AUC) with a percentile CI."""
    patients = np.sort(long["patient_ID"].unique())
    auc = _auc_function(long, patients)
    point = float(auc(np.ones((1, len(patients))))[0])
    boots = auc(_bootstrap_weights(len(patients), n_boot, np.random.default_rng(seed)))
    y = long.drop_duplicates("patient_ID")["y"]
    return {
        "n_patients": len(patients),
        "n_events": int(y.sum()),
        "AUC": point,
        "CI_low": float(np.quantile(boots, alpha / 2)),
        "CI_high": float(np.quantile(boots, 1 - alpha / 2)),
    }


def bootstrap_auc_difference(
    long_a: pd.DataFrame,
    long_b: pd.DataFrame,
    n_boot: int = 2000,
    seed: int = 23,
    alpha: float = 0.05,
) -> dict:
    """
    Paired AUC difference (a - b) on the patients predicted by both models,
    with a percentile CI and a two-sided bootstrap p-value.
    """
    patients = np.intersect1d(long_a["patient_ID"].unique(), long_b["patient_ID"].unique())
    auc_a = _auc_function(long_a, patients)
    auc_b = _auc_function(long_b, patients)
    ones = np.ones((1, len(patients)))
    point_a, point_b = float(auc_a(ones)[0]), float(auc_b(ones)[0])
    weights = _bootstrap_weights(len(patients), n_boot, np.random.default_rng(seed))
    diff = auc_a(weights) - auc_b(weights)
    p_value = min(1.0, 2 * min(np.mean(diff <= 0), np.mean(diff >= 0)))
    return {
        "n_patients": len(patients),
        "AUC_a": point_a,
        "AUC_b": point_b,
        "difference": point_a - point_b,
        "CI_low": float(np.quantile(diff, alpha / 2)),
        "CI_high": float(np.quantile(diff, 1 - alpha / 2)),
        "p_value": float(p_value),
    }


def fold_auc_summary(long: pd.DataFrame) -> dict:
    """Mean and SD of the AUC over the outer test folds (as reported so far)."""
    aucs = np.asarray(
        [
            roc_auc_score(fold["y"], fold["prediction"])
            for _, fold in long.groupby("fold")
            if fold["y"].nunique() == 2
        ]
    )
    sd = float(aucs.std(ddof=1)) if len(aucs) > 1 else float("nan")
    return {"fold_AUC_mean": float(aucs.mean()), "fold_AUC_sd": sd}
