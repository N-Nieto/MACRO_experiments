# %%
import pandas as pd
from lib.eICU_processing import eICU_admission_heart_function, eICU_filter_CS_patients             # noqa
from lib.eICU_data_loading import eICU_load_history, eICU_load_physical_exam, eICU_load_diagnosis                       # noqa
from lib.eICU_data_loading import eICU_load_patient_information, eICU_desfibrilation               # noqa
from lib.eICU_data_loading import eICU_dyslipidemia, eICU_mechanical_ventilation, eICU_st_segmentation                   # noqa
root_dir = "/home/nnieto/Nico/MODS_project/data/eicu-collaborative-research-database-2.0/" # noqa
# %%
# Load all the diagnosis
diagnosis = eICU_load_diagnosis(root_dir)
# Load important patient information
patients = eICU_load_patient_information(root_dir)
# Merge all the data
merge_data = pd.merge(patients, diagnosis, how="outer")

# Drop all the patients that don't have a endpoint
merge_data.dropna(subset="hospitaldischargestatus", inplace=True)
del diagnosis, patients
# %%
# Filter the CS patients, include all not only the Cardiac ICU patients
merge_data = eICU_filter_CS_patients(merge_data, False)

# Remove duplicated patients and week only one
# Function to keep one entry per uniquepid,
# prioritizing endpoint=1 over endpoint=0
# Sort DataFrame by 'endpoint' in descending order
merge_data = merge_data.sort_values(by='hospitaldischargestatus',
                                    ascending=False)
# Keep only the first entry for each uniquepid
merge_data = merge_data.drop_duplicates(subset='uniquepid', keep='first')

# make sure that one stay for each patient is retained
assert merge_data["patientunitstayid"].nunique() == merge_data["uniquepid"].nunique()           # noqa
# %%

past_history = eICU_load_history(root_dir=root_dir)
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

# Resusitation within 24hs
resusitation = eICU_desfibrilation(root_dir)

# Merge the features with the targets and information
merge_data = pd.merge(merge_data, resusitation, how="outer")
merge_data.dropna(subset="hospitaldischargestatus", inplace=True)

# If some patients don't have any of the past history fill with 0
merge_data["Resusitation within 24hs"].fillna(0, inplace=True)
del resusitation

# Dyslipidemia
dyslipidemia = eICU_dyslipidemia(root_dir, merge_data["patientunitstayid"])
# Merge the features with the targets and information
merge_data = pd.merge(merge_data, dyslipidemia, how="outer")
merge_data.dropna(subset="hospitaldischargestatus", inplace=True)

# If some patients don't have any of the past history fill with 0
merge_data["dyslipidemia"].fillna(0, inplace=True)
del dyslipidemia

# Mechanical ventilation
mechanical_ventilation = eICU_mechanical_ventilation(root_dir)
# Merge the features with the targets and information
merge_data = pd.merge(merge_data, mechanical_ventilation, how="outer")
merge_data.dropna(subset="hospitaldischargestatus", inplace=True)

# If some patients don't have any of the past history fill with 0
merge_data["mechanical_ventilation"].fillna(0, inplace=True)
del mechanical_ventilation

# ST elevation
st_elevation = eICU_st_segmentation(root_dir)
# Merge the features with the targets and information
merge_data = pd.merge(merge_data, st_elevation, how="outer")
merge_data.dropna(subset="hospitaldischargestatus", inplace=True)

# If some patients don't have any of the past history fill with 0
merge_data["ST_elevation"].fillna(0, inplace=True)
del st_elevation
# %%

HR = eICU_admission_heart_function(root_dir, merge_data["patientunitstayid"])
merge_data = pd.merge(merge_data, HR, how="outer")
merge_data.dropna(subset="hospitaldischargestatus", inplace=True)

# %%
# Drop not longer needed columns
merge_data.drop(["patientunitstayid", "icd9code",
                 "unittype", "uniquepid"], axis=1, inplace=True)
