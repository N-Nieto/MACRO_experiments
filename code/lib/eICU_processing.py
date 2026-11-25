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


MINUTES_PER_DAY = 1440

# Outcome labels available for the eICU validation cohort. eICU has no
# follow-up after hospital discharge, so 30-day mortality cannot be observed
# directly; the "_30d_" labels count in-hospital deaths within 30 days and
# treat patients discharged alive (or dying in hospital after day 30) as
# survivors.
EICU_OUTCOMES = {
    "in_hospital_death": "Died before hospital discharge (hospitaldischargestatus)",
    "in_hospital_death_30d_icu": "Died in hospital within 30 days of ICU admission",
    "in_hospital_death_30d_hospital": "Died in hospital within 30 days of hospital admission",
    "icu_death": "Died before ICU discharge (unitdischargestatus)",
}


def eicu_outcomes(patients: pd.DataFrame) -> pd.DataFrame:
    """
    Build all eICU outcome labels plus the early-death flag.

    Expects the columns returned by load_eicu_patient_information (statuses
    already mapped to 0/1). For patients who died in hospital, the hospital
    discharge offset is the time of death.

    Returns:
        pd.DataFrame indexed by patientunitstayid with one 0/1 column per key
        of EICU_OUTCOMES (NaN when the status is unknown), plus
        ``death_within_24h`` (died in hospital < 24 h after ICU admission,
        the equivalent of the 90 CULPRIT-SHOCK early deaths excluded from the
        Full model) and ``days_icu_admission_to_hospital_discharge``.
    """
    df = patients.set_index("patientunitstayid")
    if not df.index.is_unique:
        raise ValueError("eicu_outcomes expects one row per patientunitstayid")
    died = df["hospitaldischargestatus"]
    # Minutes from ICU admission / hospital admission to hospital discharge
    # (= time of death for patients who died in hospital)
    min_from_icu = df["hospitaldischargeoffset"]
    min_from_hosp = df["hospitaldischargeoffset"] - df["hospitaladmitoffset"]

    def died_within(minutes: pd.Series, limit: float, inclusive: bool) -> pd.Series:
        in_time = minutes <= limit if inclusive else minutes < limit
        label = ((died == 1) & in_time).astype(float)
        # Unknown status, or death with unknown time -> unknown label
        return label.mask(died.isna() | ((died == 1) & minutes.isna()))

    out = pd.DataFrame(index=df.index)
    out["in_hospital_death"] = died.astype(float)
    out["in_hospital_death_30d_icu"] = died_within(
        min_from_icu, 30 * MINUTES_PER_DAY, inclusive=True
    )
    out["in_hospital_death_30d_hospital"] = died_within(
        min_from_hosp, 30 * MINUTES_PER_DAY, inclusive=True
    )
    out["icu_death"] = df["unitdischargestatus"].astype(float)
    out["death_within_24h"] = died_within(
        min_from_icu, MINUTES_PER_DAY, inclusive=False
    )
    out["days_icu_admission_to_hospital_discharge"] = min_from_icu / MINUTES_PER_DAY
    return out


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
    heart_data = heart_data.drop(labels="observationoffset", axis=1)

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
    BP = BP.drop(labels="observationoffset", axis=1)

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
        data = data.drop(features, axis=1)

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


# eICU -> CULPRIT-SHOCK units (targets in lib.unit_harmonization.TARGET_UNITS).
# eICU units checked in output/data_checks/lab_units_report.md.
EICU_UNIT_FACTORS = {
    "CPK": 1.0,  # U/L -> U/L (was divided by 0.14999)
    "troponin - T": 1000.0,  # ng/mL -> ng/L (pg/mL)
    "creatinine": 88.42,  # mg/dL -> umol/L
    "ALT (SGPT)": 1 / 60,  # U/L -> ukat/L
    "CRP": 10.0,  # mg/dL -> mg/L (was not converted)
    # lactate (mmol/L), WBC x 1000 (K/uL), Hct (%), glucose (mg/dL), INR: same units
}

def harmonize_eicu_labs(lab: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the wide eICU lab table (one column per lab name) to the
    CULPRIT-SHOCK units. BNP is handled separately (eicu_bnp_to_ntprobnp).
    """
    lab = lab.copy()
    for name, factor in EICU_UNIT_FACTORS.items():
        if name in lab.columns:
            lab[name] = lab[name] * factor
    return lab


# eICU reports BNP, CULPRIT-SHOCK NT-proBNP (different peptides and scales).
# Options to use eICU BNP as a proxy of NT-proBNP:
#   "ishihara2022": log10 NT-proBNP = 1.21 + 1.03 log10 BNP - 0.009 BMI
#                   - 0.007 eGFR (Ishihara et al., Circ J 2022;86:2010-2018;
#                   eGFR with the Japanese equation used by the authors).
#                   The log base is not stated in the paper; log10 is the one
#                   that reproduces their medians (BNP 55.5 -> ~200 pg/mL).
#   "ratio":        NT-proBNP = 6.25 x BNP (median ratio in PARADIGM-HF,
#                   Rorth et al., Circ Heart Fail 2020;13:e006541)
#   "as_is":        BNP used unchanged as NT-proBNP (submitted manuscript)
#   "missing":      NT-proBNP not available in eICU
EICU_BNP_OPTIONS = ("ishihara2022", "ratio", "as_is", "missing")
BNP_NTPROBNP_RATIO = 6.25


def japanese_egfr(creatinine_mg_dl, age, female):
    """eGFR (mL/min/1.73 m2), Japanese equation (Matsuo et al., 2009)."""
    egfr = 194.0 * np.power(creatinine_mg_dl, -1.094) * np.power(age, -0.287)
    return np.where(female, egfr * 0.739, egfr)


def ishihara_ntprobnp(bnp, bmi, egfr):
    """NT-proBNP (pg/mL) from BNP (pg/mL), BMI (kg/m2) and eGFR (Ishihara 2022)."""
    return np.power(10.0, 1.21 + 1.03 * np.log10(bnp) - 0.009 * bmi - 0.007 * egfr)


def eicu_bnp_to_ntprobnp(data: pd.DataFrame, method: str = "ishihara2022") -> pd.DataFrame:
    """
    Replace the eICU "BNP" column by an NT-proBNP proxy (see EICU_BNP_OPTIONS).

    Expects the merged eICU data with "BNP" (pg/mL), "creatinine" (umol/L,
    after harmonize_eicu_labs), "age", "gender" (1 = male), "admissionheight"
    (cm) and "admissionweight" (kg). For "ishihara2022", a missing BMI or eGFR
    is replaced by the cohort median (count printed).
    """
    if method not in EICU_BNP_OPTIONS:
        raise ValueError(f"method must be one of {EICU_BNP_OPTIONS}, got {method!r}")
    data = data.copy()
    if "BNP" not in data.columns or method == "as_is":
        return data
    if method == "missing":
        return data.drop(columns="BNP")

    bnp = data["BNP"].where(data["BNP"] > 0)
    if method == "ratio":
        data["BNP"] = bnp * BNP_NTPROBNP_RATIO
        return data

    height_m = data["admissionheight"] / 100
    bmi = data["admissionweight"] / height_m**2
    bmi = bmi.where(bmi.between(12, 80))
    age = pd.to_numeric(data["age"], errors="coerce")
    creatinine_mg_dl = data["creatinine"] / 88.42
    egfr = pd.Series(
        japanese_egfr(creatinine_mg_dl, age, data["gender"] == 0), index=data.index
    )
    has_bnp = bnp.notna()
    for name, covariate in [("BMI", bmi), ("eGFR", egfr)]:
        n_missing = int((has_bnp & covariate.isna()).sum())
        if n_missing:
            print(
                f"[BNP -> NT-proBNP] {name} missing for {n_missing} patients "
                "with BNP: cohort median used"
            )
    bmi = bmi.fillna(bmi.median())
    egfr = egfr.fillna(egfr.median())
    data["BNP"] = ishihara_ntprobnp(bnp, bmi, egfr)
    return data


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
