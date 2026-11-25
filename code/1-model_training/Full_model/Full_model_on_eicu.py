# %%
"""
External validation of the Full model in the eICU cardiogenic-shock cohort:
the model is tuned and trained on all CULPRIT-SHOCK patients of its cohort
(same search space as the main experiment) and applied once to eICU.

Requires the preprocessed eICU cohort (eICU_extract_CS_patients.py).
The model is trained once and evaluated on every eICU outcome label
(lib.eICU_processing.EICU_OUTCOMES): in_hospital_death (submitted manuscript),
in_hospital_death_30d_icu (in-hospital death within 30 days of ICU admission,
closest to the 30-day mortality of CULPRIT-SHOCK), in_hospital_death_30d_hospital
and icu_death.

Outputs (output/eICU/Full_model/):
    Full_performance_CULPRIT_eICU.csv   metrics: CULPRIT train and eICU test
                                       per label ("Full Test (eICU, <label>)")
    Full_eICU_predictions.csv           eICU prediction and every label
                                       (y_<label>) per patientunitstayid
    Full_CULPRIT_model.json             fitted XGBoost model
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
from lib.eICU_processing import EICU_OUTCOMES
from lib.ml_utils import compute_results, get_inner_loop_optuna, results_to_df
from lib.paths import CULPRIT_DIR, DATA_DIR, OUTPUT_DIR
from lib.utils import ensure_dir

optuna.logging.set_verbosity(optuna.logging.WARNING)

# %%
model_name = "Full"
save_dir = OUTPUT_DIR / "eICU" / f"{model_name}_model"
ensure_dir(save_dir)
# eICU data root (folder containing eicu-collaborative-research-database-2.0),
# same location as in eICU_extract_CS_patients.py
eicu_data_dir = DATA_DIR
# eICU labels to evaluate (all of lib.eICU_processing.EICU_OUTCOMES)
eicu_outcomes = list(EICU_OUTCOMES)
# Same search space as the main experiment (lib/experiment_config.py)
params_optuna = get_params_optuna(model_name)

# %% Data
patient_info = load_CULPRIT_data(CULPRIT_DIR)
X, Y, patient_ids = get_model_data(patient_info, model_name)

# %% Train on CULPRIT-SHOCK, test on eICU
kf_inner = StratifiedKFold(n_splits=INNER_N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

print(f"Fitting {model_name} model on CULPRIT-SHOCK")
model = get_inner_loop_optuna(X, Y, kf_inner, params_optuna)

pred_train = model["model"].predict_proba(X)[:, 1]
results = []
results = compute_results(
    1, f"{model_name} Train (CULPRIT)", pred_train, Y, results, ths_range=THS_RANGE
)

# %% eICU. The eICU cohort keeps only patients with a known hospital discharge
# status (eICU_extract_CS_patients.py): in_hospital_death is known for all of
# them and every other label for a subset. The model is applied once to the
# cohort (aligned to the CULPRIT feature matrix; the Full model excludes
# patients who died within 24 h) and each label is evaluated on the patients
# where it is known.
X_eicu, _ = load_eICU(eicu_data_dir, model_name, X, outcome="in_hospital_death")
eicu_predictions = pd.DataFrame(
    {"prediction": model["model"].predict_proba(X_eicu)[:, 1]}, index=X_eicu.index
)
for outcome in eicu_outcomes:
    X_label, Y_label = load_eICU(eicu_data_dir, model_name, X, outcome=outcome)
    eicu_predictions[f"y_{outcome}"] = pd.Series(Y_label, index=X_label.index)
    pred_test = eicu_predictions.loc[X_label.index, "prediction"].to_numpy()
    results = compute_results(
        1, f"{model_name} Test (eICU, {outcome})", pred_test, Y_label, results, ths_range=THS_RANGE
    )

results = results_to_df(results)
print(results[results["Thresholds"] == 0.5].loc[:, ["Model", "AUC", "Balanced ACC"]].to_string())

# %% Saving results
print("Saving Results")
results.to_csv(save_dir / f"{model_name}_performance_CULPRIT_eICU.csv")
eicu_predictions.to_csv(save_dir / f"{model_name}_eICU_predictions.csv")
model["model"].save_model(save_dir / f"{model_name}_CULPRIT_model.json")
print("Experiment done")

# %%
