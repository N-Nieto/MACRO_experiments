# %%
import pandas as pd
import numpy as np

from lib.eICU_processing import eICU_admission_heart_function, eICU_filter_CS_patients                  # noqa
from lib.eICU_processing import admission_name_matching, full_model_name_matching                       # noqa
from lib.eICU_data_loading import load_eicu_history, eICU_load_physical_exam, load_eicu_diagnosis       # noqa
from lib.eICU_data_loading import load_eicu_patient_information, load_eicu_defibrillation               # noqa
from lib.eICU_data_loading import load_eicu_dyslipidemia, load_eicu_mechanical_ventilation              # noqa
from lib.eICU_data_loading import load_eicu_24hs_features, load_eicu_st_segmentation                    # noqa
from lib.eICU_data_loading import load_eicu_mechanical_support
root_dir = "/home/nnieto/Nico/MODS_project/data/eicu-collaborative-research-database-2.0/"              # noqa
# %%
# Load all the diagnosis
diagnosis = load_eicu_diagnosis(root_dir)

# Load important patient information
patients = load_eicu_patient_information(root_dir)
# Merge all the data
merge_data = pd.merge(patients, diagnosis, how="outer")

# Drop all the patients that don't have a endpoint
merge_data.dropna(subset="hospitaldischargestatus", inplace=True)
del diagnosis, patients
# %%

# Filter the CS patients, include all not only the Cardiac ICU patients
merge_data = eICU_filter_CS_patients(merge_data, True)

# Remove duplicated patients
# prioritizing endpoint=1 over endpoint=0      # noqa

# Sort DataFrame by 'endpoint' in descending order
merge_data = merge_data.sort_values(by='hospitaldischargestatus',
                                    ascending=False)
# Keep only the first entry for each uniquepid
merge_data = merge_data.drop_duplicates(subset='uniquepid', keep='first')

# make sure that one stay for each patient is retained
assert merge_data["patientunitstayid"].nunique() == merge_data["uniquepid"].nunique()                   # noqa
# %%

past_history = load_eicu_history(root_dir=root_dir)
past_history.fillna(value=0, inplace=True)

merge_data = pd.merge(merge_data, past_history, how="outer")
merge_data.dropna(subset="hospitaldischargestatus", inplace=True)

# If some patients don't have any of the past history fill with 0
for col in past_history.columns:
    merge_data[col].fillna(0, inplace=True)
del past_history

# %%
physicalExam = eICU_load_physical_exam(root_dir)

# Merge the features with the targets and information
merge_data = pd.merge(merge_data, physicalExam, how="outer")
merge_data.dropna(subset="hospitaldischargestatus", inplace=True)

# If some patients don't have any of the past history fill with 0
for col in physicalExam.columns:
    merge_data[col].fillna(0, inplace=True)
del physicalExam

# %%
# Resusitation within 24hs
resuscitation = load_eicu_defibrillation(root_dir)

# Merge the features with the targets and information
merge_data = pd.merge(merge_data, resuscitation, how="outer")
merge_data.dropna(subset="hospitaldischargestatus", inplace=True)

# If some patients don't have any of the past history fill with 0
merge_data["Resuscitation within 24hs"].fillna(0, inplace=True)
del resuscitation

# Dyslipidemia
dyslipidemia = load_eicu_dyslipidemia(root_dir,
                                      merge_data["patientunitstayid"])
# Merge the features with the targets and information
merge_data = pd.merge(merge_data, dyslipidemia, how="outer")
merge_data.dropna(subset="hospitaldischargestatus", inplace=True)

# If some patients don't have any of the past history fill with 0
merge_data["dyslipidemia"].fillna(0, inplace=True)
del dyslipidemia
# %%
# Mechanical ventilation
mechanical_ventilation = load_eicu_mechanical_ventilation(root_dir)
# Merge the features with the targets and information
merge_data = pd.merge(merge_data, mechanical_ventilation, how="outer")
merge_data.dropna(subset="hospitaldischargestatus", inplace=True)

# If some patients don't have any of the past history fill with 0
merge_data["mechanical_ventilation"].fillna(0, inplace=True)
del mechanical_ventilation
# %%
# ST elevation
st_elevation = load_eicu_st_segmentation(root_dir)
# Merge the features with the targets and information
merge_data = pd.merge(merge_data, st_elevation, how="outer")
merge_data.dropna(subset="hospitaldischargestatus", inplace=True)

# If some patients don't have any of the past history fill with 0
merge_data["ST_elevation"].fillna(0, inplace=True)
del st_elevation
# %%

plausible_range = [(20, 200, "heartrate"),
                   (20, 200, "systemicsystolic"),
                   (20, 200, "systemicdiastolic")]

HR = eICU_admission_heart_function(root_dir, plausible_range,
                                   time_before_cut_off=30)
merge_data = pd.merge(merge_data, HR, how="outer")
merge_data.dropna(subset="hospitaldischargestatus", inplace=True)

# %%
# Drop not longer needed columns
merge_data["gender"] = merge_data["gender"].astype(int)
merge_data["age"] = merge_data["age"].astype(int)
merge_data["p_rf_smoker_yn"] = np.nan

y_eicu = merge_data["hospitaldischargestatus"]

X_admission = merge_data.drop(["patientunitstayid", "icd9code",
                               "unittype", "uniquepid",
                               "hospitaldischargestatus"], axis=1)

X_admission = admission_name_matching(X_admission)
X_admission.to_csv("/home/nnieto/Nico/MODS_project/data/eicu-collaborative-research-database-2.0/preprocessed_MACRO/X_admission_CICU.csv")   # noqa
y_eicu.to_csv("/home/nnieto/Nico/MODS_project/data/eicu-collaborative-research-database-2.0/preprocessed_MACRO/y_CICU.csv")                  # noqa

# %% 24 features

# Lab
lab = load_eicu_24hs_features(root_dir)
merge_data = pd.merge(merge_data, lab, how="outer")
merge_data.dropna(subset="hospitaldischargestatus", inplace=True)

# Mechanical support
mechanical_support = load_eicu_mechanical_support(root_dir=root_dir)
merge_data = pd.merge(merge_data, mechanical_support, how="outer")
merge_data.dropna(subset="hospitaldischargestatus", inplace=True)
# %%
# Get only relevant features
Full_model_data = merge_data.drop(["patientunitstayid", "icd9code",
                                  "unittype", "uniquepid",
                                   "hospitaldischargestatus"], axis=1)

Full_model_data = full_model_name_matching(Full_model_data)
Full_model_data.to_csv("/home/nnieto/Nico/MODS_project/data/eicu-collaborative-research-database-2.0/preprocessed_MACRO/X_Full_CICU.csv")    # noqa

# %%
print("DONE!")
# %%
