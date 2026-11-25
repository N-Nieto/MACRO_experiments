# %%
"""
External validation of the Admission model in the eICU cardiogenic-shock cohort:
the model is tuned and trained on all CULPRIT-SHOCK patients of its cohort
(same search space as the main experiment) and applied once to eICU.

Requires the preprocessed eICU cohort (eICU_extract_CS_patients.py).
Outputs (output/eICU/Admission_model/):
    Admission_performance_CULPRIT_eICU.csv   metrics, eICU test and CULPRIT train
    Admission_eICU_predictions.csv           eICU prediction per patientunitstayid
    Admission_CULPRIT_model.json             fitted XGBoost model
"""
import sys
from pathlib import Path

import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.append(str(Path(__file__).resolve().parents[2]))
from lib.data_load_utils import load_CULPRIT_data, load_eICU
from lib.experiment_config import (
    INNER_N_SPLITS,
    RANDOM_STATE,
    THS_RANGE,
    get_model_data,
    get_params_optuna,
)
from lib.ml_utils import compute_results, get_inner_loop_optuna, results_to_df
from lib.paths import CULPRIT_DIR, DATA_DIR, OUTPUT_DIR
from lib.utils import ensure_dir

optuna.logging.set_verbosity(optuna.logging.WARNING)

# %%
model_name = "Admission"
save_dir = OUTPUT_DIR / "eICU" / f"{model_name}_model"
ensure_dir(save_dir)
# eICU data root (folder containing eicu-collaborative-research-database-2.0),
# same location as in eICU_extract_CS_patients.py
eicu_data_dir = DATA_DIR
# eICU label, one of lib.eICU_processing.EICU_OUTCOMES:
# "in_hospital_death" (submitted manuscript), "in_hospital_death_30d_icu",
# "in_hospital_death_30d_hospital", "icu_death"
eicu_outcome = "in_hospital_death"
# Same search space as the main experiment (lib/experiment_config.py)
params_optuna = get_params_optuna(model_name)

# %% Data
patient_info = load_CULPRIT_data(CULPRIT_DIR)
X, Y, patient_ids = get_model_data(patient_info, model_name)
# eICU patients aligned to the CULPRIT feature matrix (the Full model excludes
# patients who died within 24 h by default)
X_eicu, Y_test_eicu = load_eICU(eicu_data_dir, model_name, X, outcome=eicu_outcome)
print(f"eICU: {len(X_eicu)} patients, {int(Y_test_eicu.sum())} events")

# %% Train on CULPRIT-SHOCK, test on eICU
kf_inner = StratifiedKFold(n_splits=INNER_N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

print(f"Fitting {model_name} model on CULPRIT-SHOCK")
model = get_inner_loop_optuna(X, Y, kf_inner, params_optuna)

pred_train = model["model"].predict_proba(X)[:, 1]
pred_test = model["model"].predict_proba(X_eicu)[:, 1]

results = []
results = compute_results(
    1, f"{model_name} Test (eICU)", pred_test, Y_test_eicu, results, ths_range=THS_RANGE
)
results = compute_results(
    1, f"{model_name} Train (CULPRIT)", pred_train, Y, results, ths_range=THS_RANGE
)
results = results_to_df(results)
print(results[results["Thresholds"] == 0.5].loc[:, ["Model", "AUC", "Balanced ACC"]])

# %% Saving results
print("Saving Results")
results.to_csv(save_dir / f"{model_name}_performance_CULPRIT_eICU.csv")
pd.DataFrame(
    {"prediction": pred_test, "y_true": Y_test_eicu}, index=X_eicu.index
).to_csv(save_dir / f"{model_name}_eICU_predictions.csv")
model["model"].save_model(save_dir / f"{model_name}_CULPRIT_model.json")
print("Experiment done")

# %%
