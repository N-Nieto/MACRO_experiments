# %%
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from lib.ml_utils import compute_results, results_to_df
from lib.paths import EICU_PREPROCESSED_DIR, OUTPUT_DIR
from lib.utils import ensure_dir

eicu_root = str(EICU_PREPROCESSED_DIR) + "/"  # see lib/paths.py

patient_BOSMAN = pd.read_csv(eicu_root + "BOSMAN_info.csv", index_col=0)
patient_BOSMAN = patient_BOSMAN.dropna()

# Cardiac-ICU cohort written by eICU_extract_CS_patients.py
# (patients_CICU.csv was not produced by any script)
patients_CICU = pd.read_csv(eicu_root + "outcomes_CICU.csv")
cicu_patients = patients_CICU["patientunitstayid"].to_list()

patient_BOSMAN_CICU = patient_BOSMAN[patient_BOSMAN["patientunitstayid"].isin(cicu_patients)]

y_true = patient_BOSMAN_CICU["hospitaldischargestatus"]
y_pred = (patient_BOSMAN_CICU["BOSMAN_risk"] / 100).to_numpy()
results_BOSMAN = []
# Compute metrics without removing any feature
results_BOSMAN = compute_results(1, "BOSMAN", y_pred, y_true, results_BOSMAN)

results_BOSMAN = results_to_df(results_BOSMAN)

print(results_BOSMAN)

# Saving (output/eICU/): metrics and per-patient predictions (for bootstrap CIs)
save_dir = OUTPUT_DIR / "eICU"
ensure_dir(save_dir)
results_BOSMAN.to_csv(save_dir / "BOSMAN_performance_eICU.csv")
pd.DataFrame(
    {"prediction": y_pred, "y_true": y_true.to_numpy()},
    index=pd.Index(patient_BOSMAN_CICU["patientunitstayid"], name="patientunitstayid"),
).to_csv(save_dir / "BOSMAN_eICU_predictions.csv")

# %%
