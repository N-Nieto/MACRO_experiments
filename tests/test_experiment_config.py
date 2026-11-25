"""Shared cohort/feature selection (lib/experiment_config.py)."""
import numpy as np
import pytest
from lib.data_load_utils import load_CULPRIT_data
from lib.experiment_config import get_model_data, get_params_optuna
from synthetic_data import write_synthetic_culprit


@pytest.fixture(scope="module")
def patient_info(tmp_path_factory):
    folder = tmp_path_factory.mktemp("culprit")
    write_synthetic_culprit(folder, n=120)
    return load_CULPRIT_data(folder)


def test_cohorts(patient_info):
    X_adm, Y_adm, ids_adm = get_model_data(patient_info, "Admission")
    X_full, Y_full, ids_full = get_model_data(patient_info, "Full")
    early = (patient_info["fu_ce_Death_d"] == 0).sum()
    assert early > 0
    assert len(X_adm) == len(patient_info) and len(X_full) == len(patient_info) - early
    assert len(Y_full) == len(ids_full) == len(X_full)
    assert set(Y_adm) <= {0, 1}


def test_features_accept_nan(patient_info):
    # pandas 3: writing NaN into an int column raises; all features must be float
    X, _, _ = get_model_data(patient_info, "Admission")
    assert (X.dtypes == float).all()
    X.loc[:, X.columns[0]] = np.nan


def test_search_space_matches_supplementary_table_10():
    adm, full = get_params_optuna("Admission"), get_params_optuna("Full")
    assert adm == full
    assert (adm["max_depth_min"], adm["max_depth_max"]) == (1, 5)
    assert (adm["alpha_min"], adm["alpha_max"]) == (1e-8, 10)
    assert (adm["lambda_min"], adm["lambda_max"]) == (1e-8, 10)
    assert (adm["eta_min"], adm["eta_max"]) == (0.1, 1)
