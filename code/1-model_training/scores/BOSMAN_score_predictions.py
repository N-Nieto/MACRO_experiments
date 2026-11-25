# %%
"""
BOS,MA2 score in the eICU Cardiac-ICU cohort, evaluated on every eICU
outcome label (lib.eICU_processing.EICU_OUTCOMES), as the MACRO models.

Outputs (output/eICU/):
    BOSMAN_performance_eICU.csv   metrics per label ("BOSMAN (eICU, <label>)")
    BOSMAN_eICU_predictions.csv   prediction and one y_<label> column per label
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from lib.eICU_processing import EICU_OUTCOMES
from lib.ml_utils import compute_results, results_to_df
from lib.paths import EICU_PREPROCESSED_DIR, OUTPUT_DIR
from lib.utils import ensure_dir

eicu_root = str(EICU_PREPROCESSED_DIR) + "/"  # see lib/paths.py

patient_BOSMAN = pd.read_csv(eicu_root + "BOSMAN_info.csv", index_col=0)
patient_BOSMAN = patient_BOSMAN.dropna().set_index("patientunitstayid")

# Cardiac-ICU cohort and its labels, written by eICU_extract_CS_patients.py
outcomes_CICU = pd.read_csv(eicu_root + "outcomes_CICU.csv", index_col="patientunitstayid")
cicu_patients = patient_BOSMAN.index.intersection(outcomes_CICU.index)

y_pred = patient_BOSMAN.loc[cicu_patients, "BOSMAN_risk"] / 100
results_BOSMAN = []
predictions = pd.DataFrame({"prediction": y_pred})
for outcome in EICU_OUTCOMES:
    y_true = outcomes_CICU.loc[cicu_patients, outcome]
    known = y_true.notna()
    results_BOSMAN = compute_results(
        1, f"BOSMAN (eICU, {outcome})", y_pred[known].to_numpy(), y_true[known].astype(int).to_numpy(), results_BOSMAN
    )
    predictions[f"y_{outcome}"] = y_true

results_BOSMAN = results_to_df(results_BOSMAN)

print(results_BOSMAN)

# Saving (output/eICU/): metrics and per-patient predictions (for bootstrap CIs)
save_dir = OUTPUT_DIR / "eICU"
ensure_dir(save_dir)
results_BOSMAN.to_csv(save_dir / "BOSMAN_performance_eICU.csv")
predictions.rename_axis("patientunitstayid").to_csv(save_dir / "BOSMAN_eICU_predictions.csv")

# %%
