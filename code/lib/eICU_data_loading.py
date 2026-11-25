import os
from collections.abc import Iterable

import pandas as pd
from lib.eICU_features import read_table
from lib.eICU_processing import harmonize_eicu_labs


def load_eicu_diagnosis(root_dir: str) -> pd.DataFrame:
    """
    Load and preprocess patient diagnosis data from the eICU dataset.

    Args:
        root_dir (str): The root directory of the eICU dataset.

    Returns:
        pd.DataFrame: Processed patient diagnosis data.
    """
    # Load diagnosis data from file
    file_dir = os.path.join(root_dir, "diagnosis.csv.gz")
    diagnosis = pd.read_csv(file_dir, usecols=["patientunitstayid", "icd9code"])

    # Drop rows with missing icd9code values
    diagnosis = diagnosis.dropna(subset=["icd9code"])

    # Extract primary icd9code from comma-separated list
    diagnosis["icd9code"] = diagnosis["icd9code"].apply(lambda x: x.split(",")[0])

    return diagnosis


def load_eicu_patient_information(root_dir: str) -> pd.DataFrame:
    """
    Load and preprocess patient information data from the eICU dataset.

    Args:
        root_dir (str): The root directory of the eICU dataset.

    Returns:
        pd.DataFrame: Processed patient information data.
    """
    # Load patient information data from file
    file_dir = os.path.join(root_dir, "patient.csv.gz")
    patients = pd.read_csv(
        file_dir,
        usecols=[
            "patientunitstayid",
            "unittype",
            "uniquepid",
            "gender",
            "age",
            "admissionheight",
            "admissionweight",
            "hospitaldischargestatus",
            # Offsets (minutes from ICU admission) and ICU status, used to
            # build the alternative outcome labels (see eicu_outcomes)
            "hospitaladmitoffset",
            "hospitaldischargeoffset",
            "unitdischargeoffset",
            "unitdischargestatus",
            # Time of day of the ICU admission (calendar days 0-1, see
            # lib.eICU_features) and APACHE admission diagnosis
            "unitadmittime24",
            "apacheadmissiondx",
        ],
    )

    # Data preprocessing (assignments, not chained inplace calls, so that it
    # also works with pandas Copy-on-Write)
    patients["age"] = patients["age"].replace({"> 89": 90})

    patients["gender"] = patients["gender"].map({"Female": 0, "Male": 1})
    status = {"Alive": 0, "Expired": 1}
    patients["hospitaldischargestatus"] = patients["hospitaldischargestatus"].map(status)
    patients["unitdischargestatus"] = patients["unitdischargestatus"].map(status)

    # Data quality assertion
    assert patients["patientunitstayid"].nunique() == len(patients["patientunitstayid"])

    return patients


# Dyslipidemia cut-offs in mg/dL (the unit used by eICU for lipid labs).
# Triglycerides 150 mg/dL == 1.7 mmol/L.
DYSLIPIDEMIA_CUTOFFS_MG_DL = {
    "triglycerides": 150.0,  # dyslipidemia if above
    "HDL": 40.0,  # dyslipidemia if below
    "LDL": 190.0,  # dyslipidemia if above
}


def load_eicu_dyslipidemia(
    root_dir: str, selected_patients: pd.Series | Iterable[int]
) -> pd.DataFrame:
    """
    Load and diagnose dyslipidemia for selected patients from the eICU dataset.

    A patient is flagged when any lipid value measured BEFORE ICU admission
    meets one criterion: triglycerides > 150 mg/dL, HDL < 40 mg/dL or
    LDL > 190 mg/dL. Repeated measurements of the same analyte are averaged.
    An analyte that was never measured is NaN and never triggers a criterion.
    Patients without any pre-admission lipid value are not returned (the caller
    decides how to fill them).

    Args:
        root_dir (str): The root directory of the eICU dataset.
        selected_patients (pd.Series | Iterable[int]): patientunitstayid values
            to consider (e.g. merge_data["patientunitstayid"]).

    Returns:
        pd.DataFrame: Columns ["patientunitstayid", "dyslipidemia"] (0/1).
    """
    features_fat = list(DYSLIPIDEMIA_CUTOFFS_MG_DL)
    # Lipid labs of the selected patients measured before ICU admission
    lab = read_table(
        root_dir, "lab", ["patientunitstayid", "labresultoffset", "labname", "labresult"],
        selected_patients,
        row_filter=lambda c: (c["labresultoffset"] < 0) & c["labname"].isin(features_fat),
    )
    lab["labresult"] = pd.to_numeric(lab["labresult"], errors="coerce")

    # One column per analyte (mean of repeated values), NaN if never measured.
    # NOTE: the previous implementation averaged one-hot * value columns, which
    # mixed zeros from the other analytes into each mean, turned unmeasured
    # analytes into 0 (so "HDL < 40" fired whenever HDL was missing) and divided
    # triglycerides by the cholesterol factor (38.67 instead of 88.57).
    lipids = lab.pivot_table(
        index="patientunitstayid",
        columns="labname",
        values="labresult",
        aggfunc="mean",
    ).reindex(columns=features_fat)

    # Comparisons with NaN are False, so missing analytes never trigger
    dyslipidemia = (
        (lipids["triglycerides"] > DYSLIPIDEMIA_CUTOFFS_MG_DL["triglycerides"])
        | (lipids["HDL"] < DYSLIPIDEMIA_CUTOFFS_MG_DL["HDL"])
        | (lipids["LDL"] > DYSLIPIDEMIA_CUTOFFS_MG_DL["LDL"])
    )

    return dyslipidemia.astype(int).rename("dyslipidemia").reset_index()


def load_eicu_mechanical_ventilation(root_dir: str) -> pd.DataFrame:
    """
    Load and preprocess mechanical ventilation data from the eICU dataset.

    Args:
        root_dir (str): The root directory of the eICU dataset.

    Returns:
        pd.DataFrame: Processed mechanical ventilation data.
    """
    # Get the needed columns for faster processing
    usecols = ["patientunitstayid", "ventstartoffset"]
    # Load the respiratory care data
    vent = pd.read_csv(
        os.path.join(root_dir, "respiratoryCare.csv.gz"), usecols=usecols
    )

    # Set mechanical ventilation flag to 1 if it started before admission
    vent["mechanical_ventilation"] = vent["ventstartoffset"] < 60

    # Drop the ventstartoffset column as it's no longer needed
    vent = vent.drop("ventstartoffset", axis=1)

    # Convert boolean mechanical_ventilation column to integer
    vent["mechanical_ventilation"] = vent["mechanical_ventilation"].astype(int)

    # Group by patientunitstayid and take the maximum value
    # for mechanical_ventilation
    vent = vent.groupby("patientunitstayid").max().reset_index()

    return vent


# Lab names used for the Full model ("bedside glucose" fills missing "glucose";
# "troponin - I" is kept for the troponin-T proxy, see lib.data_load_utils.load_eICU)
EICU_LAB_NAMES = [
    "lactate", "BNP", "CPK", "troponin - T", "troponin - I", "creatinine",
    "WBC x 1000", "Hct", "CRP", "PT - INR", "glucose", "bedside glucose", "ALT (SGPT)",
]
# Labs from 24 h before to 24 h after ICU admission
LAB_WINDOW = (-1440, 1440)
# Lactate time points (offset in minutes, (lo, hi]; the admission value also
# includes lo). The admission lactate includes the first hour in the ICU:
# eICU patients usually reach the ICU after the PCI.
LACTATE_BINS = {
    "admission_lactate": (-1440, 60),
    "icu_lab_lact8hpci_x": (60, 480),
    "icu_lab_lact16hpci_x": (480, 960),
    "icu_lab_lact24hpci_x": (960, 1440),
}


def load_eicu_24hs_features(
    root_dir: str, selected_patients: pd.Series | Iterable[int] | None = None
) -> pd.DataFrame:
    """
    Mean value of each Full-model lab per patient (LAB_WINDOW), lactate per
    time point (LACTATE_BINS), converted to CULPRIT-SHOCK units. BNP stays BNP
    (pg/mL), see lib.eICU_processing.eicu_bnp_to_ntprobnp.

    Returns one row per patient (indexed by patientunitstayid) with eICU lab
    names as columns, the lactate columns with CULPRIT-SHOCK names, "glucose"
    filled with "bedside glucose" when no laboratory glucose was measured, and
    "eicu_troponin_I".
    """
    lab = read_table(
        root_dir, "lab", ["patientunitstayid", "labresultoffset", "labname", "labresult"],
        selected_patients,
        row_filter=lambda c: c["labname"].isin(EICU_LAB_NAMES)
        & c["labresultoffset"].between(*LAB_WINDOW),
    )
    lab["labresult"] = pd.to_numeric(lab["labresult"], errors="coerce")
    name = lab["labname"].copy()
    offset = lab["labresultoffset"]
    is_lactate = name == "lactate"
    for column, (lo, hi) in LACTATE_BINS.items():
        lower = offset >= lo if column == "admission_lactate" else offset > lo
        name[is_lactate & lower & (offset <= hi)] = column
    wide = lab.assign(labname=name).pivot_table(
        index="patientunitstayid", columns="labname", values="labresult", aggfunc="mean"
    )
    wide = wide.reindex(columns=[c for c in EICU_LAB_NAMES if c != "lactate"] + list(LACTATE_BINS))
    wide["glucose"] = wide["glucose"].fillna(wide["bedside glucose"])
    wide = wide.drop(columns="bedside glucose").rename(columns={"troponin - I": "eicu_troponin_I"})
    wide.columns.name = None
    # Convert to the CULPRIT-SHOCK units
    return harmonize_eicu_labs(wide)
