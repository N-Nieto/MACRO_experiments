import numpy as np
import pandas as pd
from lib.experiment_definitions import get_features


def eICU_filter_CS_patients(data, CICU=True):

    if CICU:
        print("Only keeping patients admitted in the Cardiac ICU")
        data = data[data["unittype"] == "Cardiac ICU"]
    CS_patients = data[data["icd9code"] == str(785.51)]

    return CS_patients


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


# eICU column name -> CULPRIT-SHOCK feature name. The binary features and the
# admission vital signs are created with their CULPRIT-SHOCK names
# (lib.eICU_features), as are the lactate time points (load_eicu_24hs_features).
EICU_ADMISSION_NAME_MAPPING = {
    "gender": "had_dem_male_yn",
    "age": "had_dem_age_yr",
    "admissionheight": "had_pex_height_cm",
    "admissionweight": "had_pex_weight_kg",
}

EICU_FULL_NAME_MAPPING = {
    **EICU_ADMISSION_NAME_MAPPING,
    "BNP": "pbnp",  # NT-proBNP proxy, see eicu_bnp_to_ntprobnp
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

# Extra (non-feature) columns kept in X_Full_CICU.csv for load_eICU
EICU_EXTRA_COLUMNS = ["eicu_troponin_I"]


def _rename_and_order(
    data: pd.DataFrame, mapping: dict[str, str], exp_name: str, extra: list[str] | None = None
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
    extra = [c for c in (extra or []) if c in renamed.columns]
    return renamed.reindex(columns=features + extra)


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
    return _rename_and_order(Full_model_data, EICU_FULL_NAME_MAPPING, "24hs", EICU_EXTRA_COLUMNS)
