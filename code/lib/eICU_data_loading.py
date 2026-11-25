import os
from collections.abc import Iterable

import pandas as pd
from lib.eICU_processing import (
    different_lactate_timing,
    group_features,
    harmonize_eicu_labs,
    process_individual_features,
)


def load_eicu_history(root_dir: str) -> pd.DataFrame:
    """
    Load and preprocess patient past medicalhistory data
    from the eICU dataset.

    Args:
        root_dir (str): The root directory of the eICU dataset.

    Returns:
        pd.DataFrame: Processed patient past medical history data.
    """

    # Columns to load from the pastHistory.csv.gz file
    usecols = ["patientunitstayid", "pasthistoryvalue"]

    file_dir = os.path.join(root_dir, "pastHistory.csv.gz")
    # Load past medical history data
    past_history = pd.read_csv(file_dir, usecols=usecols)

    # List of features to filter
    features_history = [
        "atrial fibrillation - chronic",
        "atrial fibrillation - intermittent",
        "CABG - date unknown",
        "CABG - remote",
        "CABG - within 2 years",
        "CABG - within 5 years",
        "CABG - within 6 months",
        "hypertension requiring treatment",
        "insulin dependent diabetes",
        "MI - date unknown",
        "MI - remote",
        "MI - within 2 years",
        "MI - within 5 years",
        "MI - within 6 months",
        "peripheral vascular disease",
        "procedural coronary intervention - date unknown",
        "procedural coronary intervention - remote",
        "procedural coronary intervention - within 2 years",
        "procedural coronary intervention - within 5 years",
        "procedural coronary intervention - within 6 months",
        "sick sinus syndrome",
    ]

    # Filter rows based on useful information
    past_history = past_history[past_history["pasthistoryvalue"].isin(features_history)]

    # Create columns for each feature, indicating its presence
    past_history = process_individual_features(
        past_history, features_history, "pasthistoryvalue"
    )

    # Group different features into categories
    features_group = {
        "Known coronary artery disease": [
            "MI - date unknown",
            "MI - remote",
            "MI - within 2 years",
            "MI - within 5 years",
            "MI - within 6 months",
            "procedural coronary intervention - date unknown",
            "procedural coronary intervention - remote",
            "procedural coronary intervention - within 2 years",
            "procedural coronary intervention - within 5 years",
            "procedural coronary intervention - within 6 months",
            "CABG - date unknown",
            "CABG - remote",
            "CABG - within 2 years",
            "CABG - within 5 years",
            "CABG - within 6 months",
        ],
        "Atrial fibrillation": [
            "atrial fibrillation - chronic",
            "atrial fibrillation - intermittent",
        ],
    }

    # Create new columns for each feature group
    past_history = group_features(past_history, features_group)

    # Select final features
    selected_features = [
        "patientunitstayid",
        "Atrial fibrillation",
        "hypertension requiring treatment",
        "insulin dependent diabetes",
        "peripheral vascular disease",
        "sick sinus syndrome",
        "Known coronary artery disease",
    ]

    # Select and group by 'patientunitstayid'
    # and take the maximum value for each group
    past_history = past_history.loc[:, selected_features]
    past_history = past_history.groupby("patientunitstayid").max().reset_index()

    # Change sick sinus syndrom to match culprit
    past_history["sick sinus syndrome"] = past_history["sick sinus syndrome"].replace(
        {0: 1, 1: 0}
    )
    # Check for uniqueness of 'patientunitstayid'
    assert past_history["patientunitstayid"].nunique() == len(
        past_history["patientunitstayid"]
    )

    return past_history


def eICU_load_physical_exam(root_dir):

    usecols = ["patientunitstayid", "physicalexamvalue", "physicalexamoffset"]

    file_dir = os.path.join(root_dir, "physicalExam.csv.gz")
    physicalExam = pd.read_csv(file_dir, usecols=usecols)
    # Use only information obtained in the first hour
    physicalExam = physicalExam[physicalExam["physicalexamoffset"] <= 60]

    # Filter relevant features
    features_exam = [
        "patientunitstayid",
        "delusional",
        "extremities cold/dusky",
        "extremities cool",
    ]
    physicalExam = physicalExam[physicalExam["physicalexamvalue"].isin(features_exam)]

    # Pass from the diagnosis to a columne
    features_exam.remove("patientunitstayid")

    physicalExam = process_individual_features(
        physicalExam, features_exam, "physicalexamvalue"
    )

    physicalExam = physicalExam.drop("physicalexamvalue", axis=1)
    physicalExam = physicalExam.drop("physicalexamoffset", axis=1)

    # Group different features
    features_group = {
        "extremities cold": ["extremities cold/dusky", "extremities cool"]
    }

    # Iterate through features_group and create a new column for each group
    for group, features in features_group.items():
        # Add a new column where 1 indicates any of the features
        # in the group were presented and 0 otherwise
        physicalExam[group] = (physicalExam[features].sum(axis=1) > 0).astype(int)
        # Remove the individual features that are part of the group
        physicalExam = physicalExam.drop(features, axis=1)

    # Group by 'patientunitstayid' and take the maximum value for each group
    physicalExam = physicalExam.groupby("patientunitstayid").max().reset_index()

    # Check for uniqueness of 'patientunitstayid'
    assert (
        physicalExam["patientunitstayid"].nunique()
        == physicalExam["patientunitstayid"].__len__()
    )

    return physicalExam


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


def load_eicu_defibrillation(root_dir: str) -> pd.DataFrame:
    """
    Load and preprocess defibrillation treatment data from the eICU dataset.

    Args:
        root_dir (str): The root directory of the eICU dataset.

    Returns:
        pd.DataFrame: Processed defibrillation treatment data.
    """
    # Load treatment data from file
    file_dir = os.path.join(root_dir, "treatment.csv.gz")
    treatment = pd.read_csv(
        file_dir, usecols=["patientunitstayid", "treatmentoffset", "treatmentstring"]
    )

    # Keep only the information of the previous 24 hours
    treatment = treatment[
        (treatment["treatmentoffset"] < 60) & (treatment["treatmentoffset"] > -1440)
    ]

    # Define feature names
    feature_names = [
        "cardiovascular|arrhythmias|cardiac defibrillation",
        "cardiovascular|non-operative procedures|defibrillation",
    ]

    # Process individual features
    for feature in feature_names:
        treatment[feature] = (treatment["treatmentstring"] == feature).astype(int)

    # Group features
    features_group = {
        "Resuscitation within 24hs": [
            "cardiovascular|arrhythmias|cardiac defibrillation",
            "cardiovascular|non-operative procedures|defibrillation",
        ]
    }

    for group, features in features_group.items():
        # Add a new column where 1 indicates any of the features
        # in the group were presented and 0 otherwise
        treatment[group] = (treatment[features].sum(axis=1) > 0).astype(int)
        # Remove the individual features that are part of the group
        treatment = treatment.drop(features, axis=1)

    # Select relevant features
    selected_features = ["patientunitstayid", "Resuscitation within 24hs"]
    treatment = treatment[selected_features]

    # Group by 'patientunitstayid' and take the maximum value for each group
    treatment = treatment.groupby("patientunitstayid").max().reset_index()

    # Check for uniqueness of 'patientunitstayid'
    assert treatment["patientunitstayid"].nunique() == len(
        treatment["patientunitstayid"]
    )

    return treatment


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
    # Get the needed columns for faster processing
    usecols = ["patientunitstayid", "labresultoffset", "labname", "labresult"]
    # Load the lab data
    lab = pd.read_csv(os.path.join(root_dir, "lab.csv.gz"), usecols=usecols)

    # Keep only the selected patients for faster processing
    lab = lab[lab["patientunitstayid"].isin(selected_patients)]

    # Keep only values measured before ICU admission (offset in minutes)
    lab = lab[lab["labresultoffset"] < 0]

    # Keep only lipid labs
    features_fat = list(DYSLIPIDEMIA_CUTOFFS_MG_DL)
    lab = lab[lab["labname"].isin(features_fat)].copy()
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


def load_eicu_st_segmentation(root_dir: str) -> pd.DataFrame:
    """
    Load and preprocess ST segment elevation data from the eICU dataset.

    Args:
        root_dir (str): The root directory of the eICU dataset.

    Returns:
        pd.DataFrame: Processed ST segment elevation data.
    """
    # Get the needed columns for faster processing
    usecols = ["patientunitstayid", "diagnosisstring", "diagnosisoffset"]
    # Load the diagnosis data
    diagnosis = pd.read_csv(os.path.join(root_dir, "diagnosis.csv.gz"), usecols=usecols)

    # Filter for patients diagnosed with "with ST elevation"
    diagnosis = diagnosis[
        diagnosis["diagnosisstring"].str.contains("with ST elevation")
    ]

    # Keep only the patients' diagnoses up to 10 minutes after admission
    diagnosis["ST_elevation"] = diagnosis["diagnosisoffset"] < 60
    diagnosis["ST_elevation"] = diagnosis["ST_elevation"].astype(int)

    # Group the patients
    diagnosis = diagnosis.groupby("patientunitstayid").max().reset_index()

    # Keep only the important features
    ST_elevation = diagnosis.loc[:, ["patientunitstayid", "ST_elevation"]]

    return ST_elevation


def load_eicu_24hs_features(root_dir: str) -> pd.DataFrame:
    """
    Load and preprocess lab data for 24-hour features from the eICU dataset.
    Units are converted to CULPRIT-SHOCK units; BNP stays BNP (pg/mL), see
    lib.eICU_processing.eicu_bnp_to_ntprobnp.

    Args:
        root_dir (str): The root directory of the eICU dataset.

    Returns:
        pd.DataFrame: Processed lab data for 24-hour features.
    """
    # Get the needed columns for faster processing
    usecols = ["patientunitstayid", "labresultoffset", "labname", "labresult"]
    # Load the lab data
    lab = pd.read_csv(os.path.join(root_dir, "lab.csv.gz"), usecols=usecols)

    # Define features to include
    features_lab = [
        "lactate",
        "BNP",
        "CPK",
        "troponin - T",
        "creatinine",
        "WBC x 1000",
        "Hct",
        "CRP",
        "PT - INR",
        "glucose",
        "ALT (SGPT)",
        "-monos",
    ]

    # Filter lab data for selected features
    lab = lab[lab["labname"].isin(features_lab)]

    # Keep only lab results within the first 24 hours and one day before
    lab = lab[lab["labresultoffset"] <= 1440]
    lab = lab[lab["labresultoffset"] >= -1440]

    # Create binary columns indicating the presence of each feature
    for feature in features_lab:
        lab[feature] = (lab["labname"] == feature).astype(int)

    # Multiply each lab result by its corresponding binary column
    for col in features_lab:
        lab[col] = lab[col] * lab["labresult"]

    # Apply function to create new labname column
    lab["labname"] = lab.apply(different_lactate_timing, axis=1)

    # Group by patientunitstayid and labname, then calculate mean labresult
    lab = (
        lab.groupby(["patientunitstayid", "labname"])["labresult"].mean().reset_index()
    )

    # Pivot the table to have labname as columns
    lab = lab.pivot(
        index="patientunitstayid", columns="labname", values="labresult"
    ).reset_index()

    # Convert to the CULPRIT-SHOCK units
    lab = harmonize_eicu_labs(lab)

    return lab


def load_eicu_mechanical_support(root_dir: str) -> pd.DataFrame:
    """
    Load eICU data related to mechanical support.

    Args:
        root_dir (str): Root directory containing eICU data.

    Returns:
        pd.DataFrame: DataFrame containing mechanical support data.
    """
    # Define columns to use
    usecols = ["patientunitstayid", "treatmentoffset", "treatmentstring"]

    # Load treatment data
    treatment = pd.read_csv(os.path.join(root_dir, "treatment.csv.gz"), usecols=usecols)

    # Keep treatments within the first 24 hours (1440 minutes)
    treatment = treatment[treatment["treatmentoffset"] < 1440]

    # Define keywords to identify mechanical support
    keywords = ["intraaortic balloon pump"]

    # Function to check if any keyword is present in the treatment string
    def check_IABP(treatment_str):
        for keyword in keywords:
            if keyword in treatment_str:
                return True
        return False

    # Function to check if any keyword is present in the treatment string
    def check_IABP_removal(treatment_str):
        for keyword in keywords:
            if keyword in treatment_str:
                return False
        return True

    # Filter the dataframe based on keywords
    treatment = treatment[treatment["treatmentstring"].apply(check_IABP)]

    # Define keywords to identify mechanical support
    keywords = ["removal"]
    # Filter the dataframe based on keywords
    treatment = treatment[treatment["treatmentstring"].apply(check_IABP_removal)]

    # Create a new column indicating mechanical support presence
    treatment["hpe_proc_mechs_yn"] = 1

    # Select relevant columns
    treatment = treatment.loc[:, ["patientunitstayid", "hpe_proc_mechs_yn"]]

    # Group by patientunitstayid and keep only the maximum value
    treatment = treatment.groupby("patientunitstayid").max().reset_index()

    # Assert uniqueness of patientunitstayid
    assert treatment["patientunitstayid"].nunique() == len(
        treatment["patientunitstayid"]
    )

    return treatment


def load_eicu_diabetes_mellitus(root_dir: str) -> pd.DataFrame:
    """
    Load eICU diagnosis data related to diabetes mellitus.

    Args:
        root_dir (str): Root directory containing eICU data.

    Returns:
        pd.DataFrame: DataFrame containing diabetes mellitus patients' data.
    """
    # Construct file path for diagnosis data
    file_dir = os.path.join(root_dir, "diagnosis.csv.gz")

    # Load diagnosis data with relevant columns
    diagnosis = pd.read_csv(
        file_dir, usecols=["patientunitstayid", "icd9code", "diagnosisoffset"]
    )

    # Keep only the diagnosis before the admission time
    diagnosis = diagnosis[diagnosis["diagnosisoffset"] < 0]

    # Drop rows with missing icd9code values
    diagnosis = diagnosis.dropna(subset=["icd9code"])

    # Extract primary icd9code from comma-separated list
    diagnosis["icd9code"] = diagnosis["icd9code"].apply(lambda x: x.split(",")[0])

    # Filter patients diagnosed with diabetes mellitus (ICD-9 code: 250.00)
    diabetes_mellitus = diagnosis[diagnosis["icd9code"] == "250.00"]

    # Drop unnecessary columns
    diabetes_mellitus = diabetes_mellitus.drop(columns=["diagnosisoffset", "icd9code"])

    # Group by patientunitstayid and keep only the maximum value
    diabetes_mellitus = (
        diabetes_mellitus.groupby("patientunitstayid").max().reset_index()
    )

    return diabetes_mellitus
