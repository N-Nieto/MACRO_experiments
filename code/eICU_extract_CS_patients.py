# %%
import pandas as pd
from lib.eICU_data_loading import (
    eICU_load_physical_exam,
    load_eicu_24hs_features,
    load_eicu_defibrillation,
    load_eicu_diagnosis,
    load_eicu_dyslipidemia,
    load_eicu_history,
    load_eicu_mechanical_support,
    load_eicu_mechanical_ventilation,
    load_eicu_patient_information,
    load_eicu_st_segmentation,
)
from lib.paths import EICU_DIR, EICU_PREPROCESSED_DIR
from lib.eICU_processing import (
    EICU_OUTCOMES,
    admission_name_matching,
    eICU_admission_heart_function,
    eICU_filter_CS_patients,
    eicu_bnp_to_ntprobnp,
    eicu_outcomes,
    full_model_name_matching,
)

# Raw eICU tables and output folder (see lib/paths.py)
data_dir = str(EICU_DIR) + "/"
EICU_PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
save_dir = str(EICU_PREPROCESSED_DIR) + "/"
# eICU reports BNP, CULPRIT-SHOCK NT-proBNP. NT-proBNP proxy, one of
# lib.eICU_processing.EICU_BNP_OPTIONS: "ishihara2022" (published conversion
# with BMI and eGFR, default), "ratio" (x6.25), "as_is" (submitted manuscript),
# "missing"
bnp_handling = "ishihara2022"
# %%
# Load all the diagnosis
diagnosis = load_eicu_diagnosis(data_dir)

# Load important patient information
patients = load_eicu_patient_information(data_dir)
# Merge all the data
merge_data = pd.merge(patients, diagnosis, how="outer")

# Drop all the patients that don't have a endpoint
merge_data = merge_data.dropna(subset="hospitaldischargestatus")
del diagnosis, patients
# %%

# Filter the CS patients, include all not only the Cardiac ICU patients
merge_data = eICU_filter_CS_patients(merge_data, True)

# Remove duplicated patients
# prioritizing endpoint=1 over endpoint=0
# Sort DataFrame by 'endpoint' in descending order
merge_data = merge_data.sort_values(by="hospitaldischargestatus", ascending=False)
# Keep only the first entry for each uniquepid
merge_data = merge_data.drop_duplicates(subset="uniquepid", keep="first")

# make sure that one stay for each patient is retained
assert merge_data["patientunitstayid"].nunique() == merge_data["uniquepid"].nunique()
# %%

past_history = load_eicu_history(root_dir=data_dir)
past_history = past_history.fillna(value=0)

merge_data = pd.merge(merge_data, past_history, how="outer")
merge_data = merge_data.dropna(subset="hospitaldischargestatus")

# If some patients don't have any of the past history fill with 0
for col in past_history.columns:
    merge_data[col] = merge_data[col].fillna(0)
del past_history

# %%
physicalExam = eICU_load_physical_exam(data_dir)

# Merge the features with the targets and information
merge_data = pd.merge(merge_data, physicalExam, how="outer")
merge_data = merge_data.dropna(subset="hospitaldischargestatus")

# If some patients don't have any of the past history fill with 0
for col in physicalExam.columns:
    merge_data[col] = merge_data[col].fillna(0)
del physicalExam

# %%
# Resuscitation within 24hs
resuscitation = load_eicu_defibrillation(data_dir)

# Merge the features with the targets and information
merge_data = pd.merge(merge_data, resuscitation, how="outer")
merge_data = merge_data.dropna(subset="hospitaldischargestatus")

# If some patients don't have any of the past history fill with 0
merge_data["Resuscitation within 24hs"] = merge_data["Resuscitation within 24hs"].fillna(0)
del resuscitation

# Dyslipidemia
dyslipidemia = load_eicu_dyslipidemia(data_dir, merge_data["patientunitstayid"])
print(
    f"Dyslipidemia: {dyslipidemia['dyslipidemia'].sum()} flagged out of "
    f"{len(dyslipidemia)} patients with a pre-admission lipid lab "
    f"(cohort: {merge_data['patientunitstayid'].nunique()} patients)"
)
# Merge the features with the targets and information
merge_data = pd.merge(merge_data, dyslipidemia, how="outer")
merge_data = merge_data.dropna(subset="hospitaldischargestatus")

# If some patients don't have any of the past history fill with 0
merge_data["dyslipidemia"] = merge_data["dyslipidemia"].fillna(0)
del dyslipidemia
# %%
# Mechanical ventilation
mechanical_ventilation = load_eicu_mechanical_ventilation(data_dir)
# Merge the features with the targets and information
merge_data = pd.merge(merge_data, mechanical_ventilation, how="outer")
merge_data = merge_data.dropna(subset="hospitaldischargestatus")

# If some patients don't have any of the past history fill with 0
merge_data["mechanical_ventilation"] = merge_data["mechanical_ventilation"].fillna(0)
del mechanical_ventilation
# %%
# ST elevation
st_elevation = load_eicu_st_segmentation(data_dir)
# Merge the features with the targets and information
merge_data = pd.merge(merge_data, st_elevation, how="outer")
merge_data = merge_data.dropna(subset="hospitaldischargestatus")

# If some patients don't have any of the past history fill with 0
merge_data["ST_elevation"] = merge_data["ST_elevation"].fillna(0)
del st_elevation
# %%

plausible_range = [
    (20, 200, "heartrate"),
    (20, 200, "systemicsystolic"),
    (20, 200, "systemicdiastolic"),
]

HR = eICU_admission_heart_function(
    data_dir,
    plausible_range,
    selected_patients=merge_data["patientunitstayid"],
    time_before_cut_off=30,
)
merge_data = pd.merge(merge_data, HR, how="outer")
merge_data = merge_data.dropna(subset="hospitaldischargestatus")

# %%
merge_data["gender"] = merge_data["gender"].astype(int)
merge_data["age"] = merge_data["age"].astype(int)

# Outcome labels (all options, see EICU_OUTCOMES) and the early-death flag,
# indexed by patientunitstayid. The label used for validation is chosen
# later, in lib.data_load_utils.load_eICU(outcome=...).
outcomes = eicu_outcomes(merge_data)
print("eICU outcomes (events / patients with a known label):")
for name, description in EICU_OUTCOMES.items():
    n_events = int(outcomes[name].sum())
    n_known = int(outcomes[name].notna().sum())
    print(f"  {name}: {n_events} / {n_known}  ({description})")
print(
    "  died within 24 h of ICU admission (excluded for the Full model): "
    f"{int(outcomes['death_within_24h'].sum())}"
)

# Features are indexed by patientunitstayid, like the outcomes
X_admission = admission_name_matching(merge_data.set_index("patientunitstayid"))
# %%
X_admission.to_csv(save_dir + "X_Admission_CICU.csv")
outcomes.to_csv(save_dir + "outcomes_CICU.csv")

# %% 24 features

# Lab
lab = load_eicu_24hs_features(data_dir)
merge_data = pd.merge(merge_data, lab, how="outer")
merge_data = merge_data.dropna(subset="hospitaldischargestatus")

# Mechanical support
mechanical_support = load_eicu_mechanical_support(data_dir)
merge_data = pd.merge(merge_data, mechanical_support, how="outer")
merge_data = merge_data.dropna(subset="hospitaldischargestatus")
# No IABP record in the first 24 h means no mechanical support (0), as for the
# other binary features; it was left missing before
merge_data["hpe_proc_mechs_yn"] = merge_data["hpe_proc_mechs_yn"].fillna(0)

# NT-proBNP proxy from BNP (uses age, sex, height, weight and creatinine)
merge_data = eicu_bnp_to_ntprobnp(merge_data, method=bnp_handling)
# %%
# Full-model features for ALL patients; the patients who died within 24 h are
# excluded when loading (load_eICU, exclude_early_deaths), as in CULPRIT-SHOCK
Full_model_data = full_model_name_matching(merge_data.set_index("patientunitstayid"))
Full_model_data.to_csv(save_dir + "X_Full_CICU.csv")

# %%
print("DONE!")

# %%
