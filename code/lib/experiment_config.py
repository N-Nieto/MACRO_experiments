"""
Settings shared by all experiments, so that the main experiments, permutation
tests, SHAP analyses, eICU validation, score comparisons and final models use
the same cohort, cross-validation and XGBoost search space.

Quick mode: set the environment variable MACRO_QUICK=1 to run any script with
tiny settings (3 Optuna trials, 1 CV repetition, 2 permutations). Results then
go to output_quick/ instead of output/ (see lib/paths.py), so a smoke run on
the real data never overwrites the real results.
"""

import os

import numpy as np
import pandas as pd
from lib.data_load_utils import get_data_from_features
from lib.data_processing import remove_low_variance_features
from lib.experiment_definitions import get_features

QUICK = os.environ.get("MACRO_QUICK", "0") == "1"

RANDOM_STATE = 23
# Minimum feature variance (binary features: prevalence between ~11% and ~89%)
VARIANCE_THS = 0.10
# Outcome: all-cause death within 30 days
ENDPOINT = "fu_ce_death_le30d_yn"  # or "fu_ce_death_le365d_yn"
# Day of death (0 = died on the admission day)
DEATH_DAY = "fu_ce_Death_d"

# Outer (evaluation) and inner (hyperparameter) cross-validation
OUT_N_SPLITS = 10
OUT_N_REPETITIONS = 1 if QUICK else 10
INNER_N_SPLITS = 3
# Label permutations for the permutation tests
N_PERMUTATIONS = 2 if QUICK else 100
# Decision thresholds evaluated by compute_results
THS_RANGE = [round(float(t), 2) for t in np.linspace(0, 1, 101)]

# Feature set (lib.experiment_definitions) and cohort of each model. The
# Admission model is evaluated on all patients; the Full model (first 24 h)
# excludes patients who died on the admission day.
FEATURE_SET = {"Admission": "Admission", "Full": "24hs"}
EXCLUDE_EARLY_DEATHS = {"Admission": False, "Full": True}

# XGBoost/Optuna search space, the same for both models (Supplementary Table 10:
# max_depth [1, 5], alpha and lambda [1e-8, 10], eta [0.1, 1]). The first
# submission used max_depth up to 10 for the Admission model and lambda up to 100.
_SEARCH_SPACE = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "optuna_trials": 3 if QUICK else 100,
    "random_state": RANDOM_STATE,
    "alpha_min": 1e-8,
    "alpha_max": 10,
    "lambda_min": 1e-8,
    "lambda_max": 10,
    "eta_min": 0.1,
    "eta_max": 1,
    "early_stopping_rounds": 10 if QUICK else 100,
    "num_boost_round": 100 if QUICK else 10000,
}
_MAX_DEPTH = {"Admission": (1, 5), "Full": (1, 5)}


def get_params_optuna(model_name: str) -> dict:
    """Optuna search space of the Admission or Full model."""
    depth_min, depth_max = _MAX_DEPTH[model_name]
    return {**_SEARCH_SPACE, "max_depth_min": depth_min, "max_depth_max": depth_max}


def model_cohort(patient_info: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Patients used for the Admission or Full model (same order as the input)."""
    if EXCLUDE_EARLY_DEATHS[model_name]:
        return patient_info[patient_info[DEATH_DAY] != 0]
    return patient_info


def get_model_data(
    patient_info: pd.DataFrame, model_name: str
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Feature matrix, labels and patient IDs of the Admission or Full model:
    cohort selection, feature set and low-variance filter, as in the paper.

    Returns
    -------
    X : pd.DataFrame (index of patient_info kept)
    Y : np.ndarray of 0/1 labels (ENDPOINT)
    patient_ids : np.ndarray
    """
    cohort = model_cohort(patient_info, model_name)
    X = get_data_from_features(cohort, get_features(FEATURE_SET[model_name]))
    X = remove_low_variance_features(X, VARIANCE_THS)
    # All features as float: columns without missing values are read as int,
    # and pandas 3 refuses to write NaN into them (feature-removal scenarios)
    X = X.astype(float)
    Y = cohort[ENDPOINT].to_numpy().astype(int)
    patient_ids = cohort["patient_ID"].to_numpy()
    print(f"{model_name} model: {X.shape[0]} patients, {X.shape[1]} features")
    return X, Y, patient_ids
