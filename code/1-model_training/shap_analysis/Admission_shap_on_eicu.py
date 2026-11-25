# %%
"""
SHAP values of the Admission model (trained on all CULPRIT-SHOCK patients)
for the eICU patients.

Outputs (output/shap/Admission_model/): eICU_shap_values.pkl (shap.Explanation)
"""
import pickle
import sys
from pathlib import Path

import optuna
import shap
from sklearn.model_selection import StratifiedKFold

sys.path.append(str(Path(__file__).resolve().parents[2]))
from lib.data_load_utils import load_CULPRIT_data, load_eICU
from lib.data_processing import naming_for_shap
from lib.experiment_config import (
    INNER_N_SPLITS,
    RANDOM_STATE,
    get_model_data,
    get_params_optuna,
)
from lib.ml_utils import compute_results, get_inner_loop_optuna, results_to_df
from lib.paths import CULPRIT_DIR, DATA_DIR, OUTPUT_DIR
from lib.utils import ensure_dir

optuna.logging.set_verbosity(optuna.logging.WARNING)

# %%
data_dir = CULPRIT_DIR  # see lib/paths.py
save_dir = OUTPUT_DIR / "shap" / "Admission_model"
ensure_dir(save_dir)
# eICU data root (folder containing eicu-collaborative-research-database-2.0),
# same location as in eICU_extract_CS_patients.py
eicu_data_dir = DATA_DIR
# eICU label, one of lib.eICU_processing.EICU_OUTCOMES:
# "in_hospital_death" (submitted manuscript), "in_hospital_death_30d_icu",
# "in_hospital_death_30d_hospital", "icu_death"
eicu_outcome = "in_hospital_death"
# Same search space as the main experiment (lib/experiment_config.py)
params_optuna = get_params_optuna("Admission")

# %% Data
patient_info = load_CULPRIT_data(data_dir)
X, Y, patient_ids = get_model_data(patient_info, "Admission")

# Load eICU with the CULPRIT column names, THEN rename both for SHAP
# (aligning after renaming left every eICU feature NaN)
X_eicu, Y_test_eicu = load_eICU(eicu_data_dir, "Admission", X, outcome=eicu_outcome)
X = naming_for_shap(data_dir, X)
X_eicu = naming_for_shap(data_dir, X_eicu)

# %%
kf_inner = StratifiedKFold(n_splits=INNER_N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

print("Fitting Admission model on CULPRIT")
admission_model = get_inner_loop_optuna(X, Y, kf_inner, params_optuna)

results_admission = []
pred_train = admission_model["model"].predict_proba(X)[:, 1]
results_admission = compute_results(
    1, "Admission Train (CULPRIT)", pred_train, Y, results_admission
)
pred_test = admission_model["model"].predict_proba(X_eicu)[:, 1]
results_admission = compute_results(
    1, "Admission Test (eICU)", pred_test, Y_test_eicu, results_admission
)
results_admission = results_to_df(results_admission)
print(results_admission.loc[:, ["Model", "AUC"]])

# %% SHAP values for the eICU patients (CULPRIT patients as background)
explainer = shap.TreeExplainer(admission_model["model"], X)
shap_values = explainer(X_eicu)
with open(save_dir / "eICU_shap_values.pkl", "wb") as fp:
    pickle.dump(shap_values, fp)

# %%
shap.plots.bar(shap_values, max_display=20)
# %%
shap.plots.beeswarm(shap_values, max_display=20)
# %%
