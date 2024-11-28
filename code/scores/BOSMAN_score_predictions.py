# %%
import pandas as pd
from lib.ml_utils import compute_results, results_to_df

eicu_root = "/home/nnieto/Nico/MODS_project/data/eicu-collaborative-research-database-2.0/preprocessed_MACRO/"          # noqa

patient_BOSMAN = pd.read_csv(eicu_root + "BOSMAN_info.csv", index_col=0)
patient_BOSMAN.dropna(inplace=True)

patients_CICU = pd.read_csv(eicu_root + "patients_CICU.csv", index_col=0)
cicu_patients = patients_CICU["patientunitstayid"].to_list()

patient_BOSMAN_CICU = patient_BOSMAN[patient_BOSMAN["patientunitstayid"].isin(cicu_patients)]                           # noqa

y_true = patient_BOSMAN_CICU["hospitaldischargestatus"]
y_pred = patient_BOSMAN_CICU["BOSMAN_risk"]/100
results_BOSMAN = []
# Compute metrics without removing any feature
results_BOSMAN = compute_results(1, "BOSMAN", y_pred, y_true, results_BOSMAN)

results_BOSMAN = results_to_df(results_BOSMAN)

print(results_BOSMAN)

# %%
