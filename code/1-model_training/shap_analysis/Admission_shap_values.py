# %%
"""
SHAP values of the Admission model, computed from the outer-fold models saved by
Admission_model/Admission_model_experiment.py (no refitting: the explained models are
the ones whose performance is reported). Every patient is explained by the
model of the fold where it was a test patient, once per CV repetition.

Outputs (output/shap/Admission_model/, file names as read by 2-plotting/plot_shap.ipynb):
    shap_values_Admission, shap_baselineAdmission, shap_dataAdmission   pickled arrays; values and
        baseline are the mean over the CV repetitions
    shap_loop_resultsAdmission   per repetition: [values, baseline, data]
    X_Admission.csv, Y_Admission.csv, predictions_Admission_shap.csv, model_Admission_shap_values.pkl (model of the last fold)
"""
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from lib.data_load_utils import load_CULPRIT_data
from lib.data_processing import naming_for_shap
from lib.experiment_config import (
    OUT_N_REPETITIONS,
    OUT_N_SPLITS,
    RANDOM_STATE,
    get_model_data,
)
from lib.paths import CULPRIT_DIR, OUTPUT_DIR
from lib.shap_utils import cross_validated_shap
from lib.utils import ensure_dir

# %%
model_name = "Admission"
data_dir = CULPRIT_DIR  # see lib/paths.py
experiment_dir = OUTPUT_DIR / f"{model_name}_model"
save_dir = OUTPUT_DIR / "shap" / f"{model_name}_model"
ensure_dir(save_dir)

# %% Data (same cohort, features and order as the experiment)
patient_info = load_CULPRIT_data(data_dir)
X, Y, patient_ids = get_model_data(patient_info, model_name)
saved_patient_ids = pd.read_csv(experiment_dir / f"{model_name}_patient_ID.csv", index_col=0)

# %% SHAP values from the saved fold models
shap_cv = cross_validated_shap(
    X, Y, patient_ids,
    models_dir=experiment_dir / "fold_models",
    model_name=model_name,
    n_splits=OUT_N_SPLITS,
    n_repetitions=OUT_N_REPETITIONS,
    random_state=RANDOM_STATE,
    saved_patient_ids=saved_patient_ids,
)
shap_data = X.to_numpy(dtype=float)
shap_values = np.nanmean(shap_cv["values"], axis=0)
shap_baseline = np.nanmean(shap_cv["base_values"], axis=0)
shap_loop = [
    [shap_cv["values"][r], shap_cv["base_values"][r], shap_data]
    for r in range(shap_cv["values"].shape[0])
]

# %% Saving
print("Saving")
for list_to_save, save_name in [
    [shap_values, "shap_values_Admission"],
    [shap_baseline, "shap_baselineAdmission"],
    [shap_data, "shap_dataAdmission"],
    [shap_loop, "shap_loop_resultsAdmission"],
]:
    with open(save_dir / save_name, "wb") as fp:  # Pickling
        pickle.dump(list_to_save, fp)

# Feature names for the plots
naming_for_shap(data_dir, X).to_csv(save_dir / "X_Admission.csv")
pd.Series(Y, index=X.index, name="fu_ce_death_le30d_yn").to_csv(save_dir / "Y_Admission.csv")
pd.DataFrame([list(p) for p in shap_cv["predictions"]]).T.to_csv(save_dir / "predictions_Admission_shap.csv")
joblib.dump(shap_cv["last_model"], save_dir / "model_Admission_shap_values.pkl")
print("Experiment done")
# %%
