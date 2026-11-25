# %%
import os

import pandas as pd
from lib.eICU_data_loading import (
    load_eicu_diagnosis,
    load_eicu_mechanical_ventilation,
    load_eicu_patient_information,
)
from lib.eICU_features import first_day_minimum_vitals
from lib.eICU_processing import eICU_filter_CS_patients
from lib.paths import EICU_DIR, EICU_PREPROCESSED_DIR
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

root_dir = str(EICU_DIR) + "/"  # see lib/paths.py
# %%
# Load all the diagnosis
diagnosis = load_eicu_diagnosis(root_dir)

# Load important patient information
patients = load_eicu_patient_information(root_dir)
# Merge all the data
merge_data = pd.merge(patients, diagnosis, how="outer")

# Drop all the patients that don't have a endpoint
merge_data = merge_data.dropna(subset="hospitaldischargestatus")
del diagnosis, patients
# %%

# Filter the CS patients, include all not only the Cardiac ICU patients
merge_data = eICU_filter_CS_patients(merge_data, False)

# Remove duplicated patients
# prioritizing endpoint=1 over endpoint=0    

# Sort DataFrame by 'endpoint' in descending order
merge_data = merge_data.sort_values(by='hospitaldischargestatus',
                                    ascending=False)
# Keep only the first entry for each uniquepid
merge_data = merge_data.drop_duplicates(subset='uniquepid', keep='first')

# make sure that one stay for each patient is retained
assert merge_data["patientunitstayid"].nunique() == merge_data["uniquepid"].nunique()                 

# %%
# Mechanical ventilation
mechanical_ventilation = load_eicu_mechanical_ventilation(root_dir)
# Merge the features with the targets and information
merge_data = pd.merge(merge_data, mechanical_ventilation, how="outer")
merge_data = merge_data.dropna(subset="hospitaldischargestatus")

# If some patients don't have any of the past history fill with 0
merge_data["mechanical_ventilation"] = merge_data["mechanical_ventilation"].fillna(0)
del mechanical_ventilation
# %%
# Keep the important columnes
patient_BOSMAN = merge_data.loc[:, ["patientunitstayid", "age",
                                    "mechanical_ventilation",
                                    "hospitaldischargestatus"]]

patient_BOSMAN["age"] = patient_BOSMAN["age"].astype(int)
# Assine one point if the age is more or equal to 60 years olds
patient_BOSMAN["age"] = patient_BOSMAN["age"] >= 60
patient_BOSMAN["age"] = patient_BOSMAN["age"].astype(int)
selected_patients = merge_data["patientunitstayid"]
# Get the needed columns for faster processing
usecols = ["patientunitstayid", "labresultoffset", "labname", "labresult"]
# Load the lab data
lab = pd.read_csv(os.path.join(root_dir, "lab.csv.gz"), usecols=usecols)

# Keep only the selected patients for faster processing
lab = lab[lab["patientunitstayid"].isin(selected_patients)]

# Remove data before 24hs hour
lab = lab[lab['labresultoffset'] < 1440]

# %% Lab data
# Find the maximun anion gap in the first 24hs
max_anion_gap = lab[lab["labname"] == "anion gap"].groupby("patientunitstayid").max().reset_index()   
# Check which of the pass the condition
max_anion_gap["max_anion_bool"] = max_anion_gap["labresult"] >= 14
max_anion_gap["max_anion_bool"] = max_anion_gap["max_anion_bool"].astype(int)

# Find the BUN in the first 24hs
max_BUN = lab[lab["labname"] == "BUN"].groupby("patientunitstayid").max().reset_index()               
max_BUN["max_BUN_bool"] = max_BUN["labresult"] >= 25
max_BUN["max_BUN_bool"] = max_BUN["max_BUN_bool"].astype(int)

# Delete the lab to free space
del lab

# Include the new data in the score
max_anion_gap = max_anion_gap.loc[:, ["patientunitstayid", "max_anion_bool"]]
# Merge the features with the targets and information
patient_BOSMAN = pd.merge(patient_BOSMAN, max_anion_gap, how="outer")
patient_BOSMAN = patient_BOSMAN.dropna(subset="hospitaldischargestatus")

max_BUN = max_BUN.loc[:, ["patientunitstayid", "max_BUN_bool"]]
# Merge the features with the targets and information
patient_BOSMAN = pd.merge(patient_BOSMAN, max_BUN, how="outer")
patient_BOSMAN = patient_BOSMAN.dropna(subset="hospitaldischargestatus")
# %% Cardiac data: lowest systolic BP and SpO2 in the first 24 h from all
# eICU sources (invasive and non-invasive BP, monitor and charted SpO2), after
# removing artefacts (lib/eICU_features.first_day_minimum_vitals). Before, only
# invasive BP was used and monitor artefacts (e.g. SpO2 0 %) were kept.
selected_patients = patient_BOSMAN["patientunitstayid"].tolist()
heart_data = first_day_minimum_vitals(root_dir, selected_patients).reset_index()

# Compute the O2 saturation condition
min_sa02_bool = heart_data.loc[:, ["patientunitstayid", "min_spo2"]].dropna()
min_sa02_bool["min_sa02_bool"] = (min_sa02_bool["min_spo2"] < 88).astype(int)

# Compute the systolic pressure condition
min_sys_bool = heart_data.loc[:, ["patientunitstayid", "min_sbp"]].dropna()
min_sys_bool["min_sys_bool"] = (min_sys_bool["min_sbp"] < 80).astype(int)
# %% Add the information to the patients
min_sa02_bool = min_sa02_bool.loc[:, ["patientunitstayid", "min_sa02_bool"]]

patient_BOSMAN = pd.merge(patient_BOSMAN, min_sa02_bool, how="outer")
patient_BOSMAN = patient_BOSMAN.dropna(subset="hospitaldischargestatus")


min_sys_bool = min_sys_bool.loc[:, ["patientunitstayid", "min_sys_bool"]]


patient_BOSMAN = pd.merge(patient_BOSMAN, min_sys_bool, how="outer")
patient_BOSMAN = patient_BOSMAN.dropna(subset="hospitaldischargestatus")

# %%

BOSMAN_only_info = patient_BOSMAN.drop(columns=["patientunitstayid",
                                                "hospitaldischargestatus"])


# %%
patient_BOSMAN["BOSMAN_score"] = BOSMAN_only_info.sum(axis=1)


# Define a function to map scores to risks
def map_score_to_risk(score):
    risk_dict = {0: 0.5, 1: 1.4, 2: 3.9, 3: 10, 4: 23.5, 5: 46, 6: 70.2}
    return risk_dict.get(score, None)


# Apply the function to create the new column "risk"
patient_BOSMAN['BOSMAN_risk'] = patient_BOSMAN['BOSMAN_score'].apply(map_score_to_risk)       
# %%
# Save the dataframe
EICU_PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
patient_BOSMAN.to_csv(EICU_PREPROCESSED_DIR / "BOSMAN_info.csv")

# %%

patient_BOSMAN = patient_BOSMAN.dropna()
print(roc_auc_score(patient_BOSMAN["hospitaldischargestatus"],
                    patient_BOSMAN["BOSMAN_score"]))
print(balanced_accuracy_score(patient_BOSMAN["hospitaldischargestatus"],
                              round(patient_BOSMAN["BOSMAN_risk"]/100)))
# %%
