"""SHAP values of the cross-validated models saved by the main experiments."""

from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import RepeatedStratifiedKFold


def load_fold_model(models_dir: Path, model_name: str, i_fold: int) -> xgb.XGBClassifier:
    """XGBoost model of one outer fold (saved by <model>_model_experiment.py)."""
    path = Path(models_dir) / f"{model_name}_fold_{i_fold:03d}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found: run {model_name}_model_experiment.py first "
            "(it saves the model of every outer fold)."
        )
    model = xgb.XGBClassifier()
    model.load_model(path)
    return model


def cross_validated_shap(
    X: pd.DataFrame,
    Y: np.ndarray,
    patient_ids: np.ndarray,
    models_dir: Path,
    model_name: str,
    n_splits: int,
    n_repetitions: int,
    random_state: int,
    saved_patient_ids: pd.DataFrame | None = None,
) -> dict:
    """
    SHAP values of every patient from the outer-fold models of the main
    experiment, i.e. the same models whose performance is reported. The outer
    folds are re-created with the same seed; if ``saved_patient_ids`` (the
    experiment's <model>_patient_ID.csv) is given, the folds are checked
    against it.

    The test patients of every fold are explained with an interventional
    TreeExplainer, using the fold's training patients as background.

    Returns a dict with
        values      (n_repetitions, n_patients, n_features) SHAP values
        base_values (n_repetitions, n_patients)
        predictions list of out-of-fold predicted probabilities, one per fold
        last_model  model of the last fold
    """
    kf_out = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repetitions, random_state=random_state
    )
    n_patients, n_features = X.shape
    values = np.full((n_repetitions, n_patients, n_features), np.nan)
    base_values = np.full((n_repetitions, n_patients), np.nan)
    predictions = []
    model = None
    for i_fold, (train_index, test_index) in enumerate(kf_out.split(X, Y)):
        if saved_patient_ids is not None:
            saved = saved_patient_ids.iloc[:, i_fold].dropna().astype(str).to_numpy()
            if not np.array_equal(saved, patient_ids[test_index].astype(str)):
                raise ValueError(
                    f"Fold {i_fold}: test patients differ from the experiment's "
                    "patient_ID.csv (data or settings changed since the run)."
                )
        model = load_fold_model(models_dir, model_name, i_fold)
        X_test = X.iloc[test_index]
        explainer = shap.TreeExplainer(model, X.iloc[train_index])
        explanation = explainer(X_test)
        repetition = i_fold // n_splits
        values[repetition, test_index, :] = explanation.values
        base_values[repetition, test_index] = explanation.base_values
        predictions.append(model.predict_proba(X_test)[:, 1])
    return {
        "values": values,
        "base_values": base_values,
        "predictions": predictions,
        "last_model": model,
    }
