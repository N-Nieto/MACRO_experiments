from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from lib.experiment_definitions import get_features


def eICU_filter_CS_patients(data, CICU=True):

    if CICU:
        print("Only keeping patients admitted in the Cardiac ICU")
        data = data[data["unittype"] == "Cardiac ICU"]
    CS_patients = data[data["icd9code"] == str(785.51)]

    return CS_patients


def _read_csv_for_patients(
    file_path: str,
    usecols: list[str],
    selected_patients: pd.Series | Iterable[int],
    chunksize: int = 5_000_000,
) -> pd.DataFrame:
    """Read a (large) eICU table in chunks, keeping only the selected patients."""
    selected = set(pd.Series(list(selected_patients)).tolist())
    chunks = [
        chunk[chunk["patientunitstayid"].isin(selected)]
        for chunk in pd.read_csv(file_path, usecols=usecols, chunksize=chunksize)
    ]
    return pd.concat(chunks, ignore_index=True)


def eICU_admission_heart_function(
    root_dir: str,
    plausible_range: Sequence[tuple[float, float, str]],
    selected_patients: pd.Series | Iterable[int] | None = None,
    time_before_cut_off: int = 60,
    time_after_cut_off: int = -1440,
) -> pd.DataFrame:
    """
    Median heart rate and invasive systolic/diastolic pressure per patient in
    the window (time_after_cut_off, time_before_cut_off) minutes around ICU
    admission, after removing values outside ``plausible_range``.

    Reads the raw eICU ``vitalPeriodic.csv.gz``. Pass ``selected_patients``
    (e.g. the CS cohort) to read it in chunks and keep only those patients,
    which avoids loading the whole table (~146 M rows) into memory.
    """
    usecols = [
        "patientunitstayid",
        "observationoffset",
        "heartrate",
        "systemicsystolic",
        "systemicdiastolic",
    ]
    file_path = root_dir + "vitalPeriodic.csv.gz"
    if selected_patients is None:
        heart_data = pd.read_csv(file_path, usecols=usecols)
    else:
        heart_data = _read_csv_for_patients(file_path, usecols, selected_patients)
    # Remove the data after one Hour
    heart_data = heart_data[heart_data["observationoffset"] < time_before_cut_off]
    # Remove data older than one day
    heart_data = heart_data[heart_data["observationoffset"] > time_after_cut_off]
    # Drop timing column
    heart_data.drop(labels="observationoffset", axis=1, inplace=True)

    # Iterate over each range in plausible_range
    # to help removing outliers in the data
    for min_val, max_val, column in plausible_range:
        # Replace values outside the range with NaN for the specified column
        heart_data[column] = heart_data[column].where(
            (heart_data[column] >= min_val) & (heart_data[column] <= max_val),
            other=np.nan,
        )

    # Group the data by the median
    heart_data = heart_data.groupby("patientunitstayid").median().reset_index()

    return heart_data


def eICU_aperiodic_pressure(
    root_dir,
    plausible_range,
    selected_patients,
    time_before_cut_off=30,
    time_after_cut_off=-1440,
):

    usecols = [
        "patientunitstayid",
        "observationoffset",
        "noninvasivesystolic",
        "noninvasivediastolic",
    ]

    BP = pd.read_csv(
        root_dir + "vitalAperiodic.csv.gz",
        usecols=usecols,
    )

    selected_patients = selected_patients.tolist()
    # Keep only the CS patients for faster processing
    BP = BP[BP["patientunitstayid"].isin(selected_patients)]
    # Remove the data after one Hour
    BP = BP[BP["observationoffset"] < time_before_cut_off]
    # Remove data older than one day
    BP = BP[BP["observationoffset"] > time_after_cut_off]
    # Drop timing columne
    BP.drop(labels="observationoffset", axis=1, inplace=True)

    # Iterate over each range in plausible_range
    # to help removing outliers in the data
    for min_val, max_val, column in plausible_range:
        # Replace values outside the range with NaN for the specified column
        BP[column] = BP[column].where(
            (BP[column] >= min_val) & (BP[column] <= max_val), other=np.nan
        )

    # Group the data by the median
    BP = BP.groupby("patientunitstayid").median().reset_index()

    return BP


# General propose functions
def process_individual_features(
    data: pd.DataFrame, features_list: list, field: str
) -> pd.DataFrame:
    """
    Process individual features in the patient data.
    Return 1 where the feature is found

    Args:
        Data (pd.DataFrame): Patient data.
        features_list (List): List of features to process

    Returns:
        pd.DataFrame: Processed patient data.
    """

    # Create columns for each feature
    for feature in features_list:
        data[feature] = (data[field] == feature).astype(int)

    return data


def group_features(data: pd.DataFrame, features_group: dict) -> pd.DataFrame:
    """
    Group features in the patient data.

    Args:
        physical_exam (pd.DataFrame): Patient data.
        features_group (Dict): Dictionary of grouped features

    Returns:
        pd.DataFrame: Grouped patient data.
    """

    # Create new columns for each feature group
    for group, features in features_group.items():
        data[group] = (data[features].sum(axis=1) > 0).astype(int)
        data.drop(features, axis=1, inplace=True)

    return data


# Define a function to create new labname based on laboffset
def different_lactate_timing(row):
    if row["labname"] == "lactate":
        if row["labresultoffset"] <= 0:
            return "admission_lactate"
        elif 0 < row["labresultoffset"] <= 480:
            return "icu_lab_lact8hpci_x"
        elif 480 < row["labresultoffset"] <= 960:
            return "icu_lab_lact16hpci_x"
        elif 960 < row["labresultoffset"] <= 1440:
            return "icu_lab_lact24hpci_x"
    return row["labname"]


# eICU to CULPRIT unit Harmonize functions
def eicu_CPK_harmonization(cpk):
    return cpk / 0.14999  # Conversion factor: 1 U/L =  mmol/L


def eicu_Troponin_harmonization(troponin_t):
    return troponin_t * 1000  # Conversion factor: 1 ng/L = 1000 pg/L


def eicu_Creatinine_harmonization(creatinine):
    # Conversion factor: 1 microgram/dl = 0.00885 micromol/L
    creatinine = creatinine / 113.12
    creatinine = creatinine * 10000
    return creatinine


def eicul_ALAT_harmonization(alat):
    return alat / 60  # Conversion factor: 1 U/L = 0.01667 microkat/L


# eICU column name -> CULPRIT-SHOCK feature name (as used in experiment_definitions)
EICU_ADMISSION_NAME_MAPPING = {
    "gender": "had_dem_male_yn",
    "age": "had_dem_age_yr",
    "admissionheight": "had_pex_height_cm",
    "admissionweight": "had_pex_weight_kg",
    "Atrial fibrillation": "hpr_ecg_afib_y",
    "hypertension requiring treatment": "p_rf_aht_yn",
    "insulin dependent diabetes": "p_rf_dm_yn",
    "peripheral vascular disease": "p_mh_pad_yn",
    "sick sinus syndrome": "hpr_ecg_sinrhy_y",
    "Known coronary artery disease": "previous_heart_complications",
    "delusional": "had_sy_ams_yn",
    "extremities cold": "had_sy_extremity_yn",
    "Resuscitation within 24hs": "had_base_cpr24h_yn",
    "dyslipidemia": "p_rf_dyslip_yn",
    "mechanical_ventilation": "had_base_mechvent_yn",
    "ST_elevation": "hpr_ecg_stemi_yn",
    "heartrate": "hpr_hmdyn_hr_bpm",
    "systemicsystolic": "hpr_hmdyn_sbp_mmhg",
    "systemicdiastolic": "hpr_hmdyn_dbp_mmhg",
}

# The lactate columns (admission_lactate, icu_lab_lact8/16/24hpci_x) and
# hpe_proc_mechs_yn are already created with their CULPRIT-SHOCK names.
EICU_FULL_NAME_MAPPING = {
    **EICU_ADMISSION_NAME_MAPPING,
    "BNP": "pbnp",  # NOTE: eICU reports BNP, CULPRIT-SHOCK NT-proBNP
    "CPK": "icu_lab_ck_x",
    "troponin - T": "tnt",
    "creatinine": "creatine",
    "WBC x 1000": "white_cell_count",
    "Hct": "hematocrit",
    "CRP": "crp",
    "PT - INR": "icu_lab_inr_r",
    "glucose": "glucose",
    "ALT (SGPT)": "alat",
}


def _rename_and_order(
    data: pd.DataFrame, mapping: dict[str, str], exp_name: str
) -> pd.DataFrame:
    """
    Rename eICU columns to CULPRIT-SHOCK names and order them as in
    experiment_definitions.get_features(exp_name).

    Features of the experiment that do not exist in eICU are added as all-NaN
    columns; eICU columns that are not features of the experiment are dropped.
    The low-variance filter applied in CULPRIT-SHOCK is NOT applied here: align
    to a trained model with ``X_eicu.reindex(columns=X_culprit.columns)``
    (done in lib.data_load_utils.load_eICU).
    """
    features = get_features(exp_name)
    renamed = data.rename(columns=mapping)
    missing = [f for f in features if f not in renamed.columns]
    if missing:
        print(f"[{exp_name}] features not available in eICU (all NaN): {missing}")
    return renamed.reindex(columns=features)


def admission_name_matching(X_admission: pd.DataFrame) -> pd.DataFrame:
    """
    Rename eICU admission columns to CULPRIT-SHOCK names and return them in
    the order of the "Admission" experiment features.

    The feature list comes from lib.experiment_definitions, so no trained model
    has to be loaded.

    Args:
        X_admission (pd.DataFrame): eICU admission data with eICU column names.

    Returns:
        pd.DataFrame: Admission features with CULPRIT-SHOCK names.
    """
    return _rename_and_order(X_admission, EICU_ADMISSION_NAME_MAPPING, "Admission")


def full_model_name_matching(Full_model_data: pd.DataFrame) -> pd.DataFrame:
    """
    Rename eICU admission + 24 h columns to CULPRIT-SHOCK names and return them
    in the order of the "24hs" (Full model) experiment features.

    The feature list comes from lib.experiment_definitions, so no trained model
    has to be loaded.

    Args:
        Full_model_data (pd.DataFrame): eICU data with eICU column names.

    Returns:
        pd.DataFrame: Full-model features with CULPRIT-SHOCK names.
    """
    return _rename_and_order(Full_model_data, EICU_FULL_NAME_MAPPING, "24hs")
