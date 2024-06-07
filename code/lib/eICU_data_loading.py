import pandas as pd
import os
from typing import List
from lib.eICU_processing import process_individual_features, group_features
from lib.eICU_processing import diferent_lactate_timing, eicu_CPK_harmonization, eicu_Troponin_harmonization             # noqa
from lib.eICU_processing import eicu_Creatinine_harmonization, eicul_ALAT_harmonization     # noqa


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
        "sick sinus syndrome"
    ]

    # Filter rows based on useful information
    past_history = past_history[past_history["pasthistoryvalue"].isin(features_history)]                # noqa

    # Create columns for each feature, indicating its presence
    past_history = process_individual_features(past_history, features_history,
                                               "pasthistoryvalue")

    # Group different features into categories
    features_group = {
        'Known coronary artery disease': [
            'MI - date unknown',
            'MI - remote',
            'MI - within 2 years',
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
            "CABG - within 6 months"
        ],
        'Atrial fibrillation': [
            "atrial fibrillation - chronic",
            "atrial fibrillation - intermittent"
        ]
    }

    # Create new columns for each feature group
    past_history = group_features(past_history, features_group)

    # Select final features
    selected_features = [
        'patientunitstayid',
        'Atrial fibrillation',
        'hypertension requiring treatment',
        'insulin dependent diabetes',
        'peripheral vascular disease',
        'sick sinus syndrome',
        'Known coronary artery disease'
    ]

    # Select and group by 'patientunitstayid'
    # and take the maximum value for each group
    past_history = past_history.loc[:, selected_features]
    past_history = past_history.groupby("patientunitstayid").max().reset_index()                    # noqa

    # Change sick sinus syndrom to match culprit
    past_history["sick sinus syndrome"].replace({0: -1}, inplace=True)
    past_history["sick sinus syndrome"].replace({1: 0}, inplace=True)
    past_history["sick sinus syndrome"].replace({-1: 1}, inplace=True)
    # Check for uniqueness of 'patientunitstayid'
    assert past_history["patientunitstayid"].nunique() == len(past_history["patientunitstayid"])    # noqa

    return past_history


def eICU_load_physical_exam(root_dir):

    usecols = ["patientunitstayid", "physicalexamvalue", "physicalexamoffset"]

    file_dir = os.path.join(root_dir, "physicalExam.csv.gz")
    physicalExam = pd.read_csv(file_dir, usecols=usecols)
    # Use only information obtained in the first hour
    physicalExam = physicalExam[physicalExam['physicalexamoffset'] <= 60]

    # Filter relevant features
    features_exam = ["patientunitstayid", "delusional",
                     "extremities cold/dusky", "extremities cool"
                     ]
    physicalExam = physicalExam[physicalExam["physicalexamvalue"].isin(features_exam)]                  # noqa

    # Pass from the diagnosis to a columne
    features_exam.remove("patientunitstayid")

    physicalExam = process_individual_features(physicalExam, features_exam,
                                               "physicalexamvalue")

    physicalExam.drop("physicalexamvalue", axis=1, inplace=True)
    physicalExam.drop("physicalexamoffset", axis=1, inplace=True)

    # Group different features
    features_group = {'extremities cold':
                      ["extremities cold/dusky", "extremities cool"]
                      }

    # Iterate through features_group and create a new column for each group
    for group, features in features_group.items():
        # Add a new column where 1 indicates any of the features
        # in the group were presented and 0 otherwise
        physicalExam[group] = (physicalExam[features].sum(axis=1) > 0).astype(int)                  # noqa
        # Remove the individual features that are part of the group
        physicalExam.drop(features, axis=1, inplace=True)

    # Group by 'patientunitstayid' and take the maximum value for each group
    physicalExam = physicalExam.groupby("patientunitstayid").max().reset_index()                        # noqa

    # Check for uniqueness of 'patientunitstayid'
    assert physicalExam["patientunitstayid"].nunique() == physicalExam["patientunitstayid"].__len__()   # noqa

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
    diagnosis = pd.read_csv(file_dir, usecols=["patientunitstayid",
                                               "icd9code"])

    # Drop rows with missing icd9code values
    diagnosis = diagnosis.dropna(subset=["icd9code"])

    # Extract primary icd9code from comma-separated list
    diagnosis['icd9code'] = diagnosis['icd9code'].apply(lambda x: x.split(',')[0])          # noqa

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
    patients = pd.read_csv(file_dir, usecols=["patientunitstayid", "unittype",
                                              "uniquepid", "gender", "age",
                                              "admissionheight",
                                              "admissionweight",
                                              "hospitaldischargestatus"])

    # Data preprocessing
    patients["age"].replace({"> 89": 90}, inplace=True)

    patients["gender"].replace({"Female": 0, "Male": 1}, inplace=True)
    patients["hospitaldischargestatus"].replace({"Alive": 0, "Expired": 1},
                                                inplace=True)

    # Data quality assertion
    assert patients["patientunitstayid"].nunique() == len(patients["patientunitstayid"])    # noqa

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
    treatment = pd.read_csv(file_dir, usecols=["patientunitstayid",
                                               "treatmentoffset",
                                               "treatmentstring"])

    # Keep only the information of the previous 24 hours
    treatment = treatment[(treatment['treatmentoffset'] < 60) & (treatment['treatmentoffset'] > -1440)]  # noqa

    # Define feature names
    feature_names = ["cardiovascular|arrhythmias|cardiac defibrillation",
                     "cardiovascular|non-operative procedures|defibrillation"]

    # Process individual features
    for feature in feature_names:
        treatment[feature] = (treatment['treatmentstring'] == feature).astype(int)                      # noqa

    # Group features
    features_group = {'Resuscitation within 24hs':
                      ['cardiovascular|arrhythmias|cardiac defibrillation',
                       'cardiovascular|non-operative procedures|defibrillation']                        # noqa
                      }

    for group, features in features_group.items():
        # Add a new column where 1 indicates any of the features
        # in the group were presented and 0 otherwise
        treatment[group] = (treatment[features].sum(axis=1) > 0).astype(int)
        # Remove the individual features that are part of the group
        treatment.drop(features, axis=1, inplace=True)

    # Select relevant features
    selected_features = ['patientunitstayid', 'Resuscitation within 24hs']
    treatment = treatment[selected_features]

    # Group by 'patientunitstayid' and take the maximum value for each group
    treatment = treatment.groupby("patientunitstayid").max().reset_index()

    # Check for uniqueness of 'patientunitstayid'
    assert treatment["patientunitstayid"].nunique() == len(treatment["patientunitstayid"])              # noqa

    return treatment


def load_eicu_dyslipidemia(root_dir: str,
                           selected_patients: List[int]) -> pd.DataFrame:
    """
    Load and diagnose dyslipidemia for selected patients from the eICU dataset.

    Args:
        root_dir (str): The root directory of the eICU dataset.
        selected_patients (List[int]): List of patient IDs to consider.

    Returns:
        pd.DataFrame: Processed dyslipidemia diagnosis data.
    """
    # Get the needed columns for faster processing
    usecols = ["patientunitstayid", "labresultoffset", "labname", "labresult"]
    # Load the lab data
    lab = pd.read_csv(os.path.join(root_dir, "lab.csv.gz"), usecols=usecols)

    # Keep only the selected patients for faster processing
    lab = lab[lab["patientunitstayid"].isin(selected_patients)]

    # Remove data before afmission hour
    lab = lab[lab['labresultoffset'] < 0]

    # Define features related to lipid profile
    features_fat = ["triglycerides", "HDL", "LDL"]

    # Filter lab data for relevant features
    lab = lab[lab["labname"].isin(features_fat)]

    # Create binary columns indicating the presence of each lipid feature
    for feature in features_fat:
        lab[feature] = (lab['labname'] == feature).astype(int)

    # Multiply each lab result by its corresponding binary column
    for col in features_fat:
        lab[col] = lab[col] * lab["labresult"]

    # Harmonize units for triglycerides
    lab["triglycerides"] = lab["triglycerides"] / 38.67

    # Group if several values are present for the same patient
    lab = lab.groupby("patientunitstayid").mean().reset_index()

    # Define criteria for dyslipidemia
    var1 = lab["triglycerides"] > 1.7
    var2 = lab["HDL"] < 40
    var3 = lab["LDL"] > 190

    # If any of the criteria is presented, then diagnose dyslipidemia
    lab["dyslipidemia"] = var1 | var2 | var3

    # Transform binary diagnosis into 1s and 0s
    lab["dyslipidemia"] = lab["dyslipidemia"].astype(int)

    # Keep only relevant information
    lab = lab.loc[:, ["patientunitstayid", "dyslipidemia"]]

    return lab


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
    vent = pd.read_csv(os.path.join(root_dir, "respiratoryCare.csv.gz"),
                       usecols=usecols)

    # Set mechanical ventilation flag to 1 if it started before admission
    vent["mechanical_ventilation"] = vent["ventstartoffset"] < 60

    # Drop the ventstartoffset column as it's no longer needed
    vent.drop("ventstartoffset", axis=1, inplace=True)

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
    diagnosis = pd.read_csv(os.path.join(root_dir, "diagnosis.csv.gz"),
                            usecols=usecols)

    # Filter for patients diagnosed with "with ST elevation"
    diagnosis = diagnosis[diagnosis['diagnosisstring'].str.contains("with ST elevation")]       # noqa

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
    features_lab = ["lactate", "BNP", "CPK", "troponin - T", "creatinine",
                    "WBC x 1000", "Hct", "CRP",  "PT - INR", "glucose",
                    "ALT (SGPT)"]

    # Filter lab data for selected features
    lab = lab[lab["labname"].isin(features_lab)]

    # Keep only lab results within the first 24 hours and one day before
    lab = lab[lab['labresultoffset'] <= 1440]
    lab = lab[lab['labresultoffset'] >= -1440]

    # Create binary columns indicating the presence of each feature
    for feature in features_lab:
        lab[feature] = (lab['labname'] == feature).astype(int)

    # Multiply each lab result by its corresponding binary column
    for col in features_lab:
        lab[col] = lab[col] * lab["labresult"]

    # Apply function to create new labname column
    lab['labname'] = lab.apply(diferent_lactate_timing, axis=1)

    # Group by patientunitstayid and labname, then calculate mean labresult
    lab = lab.groupby(['patientunitstayid', 'labname'])['labresult'].mean().reset_index()           # noqa

    # Pivot the table to have labname as columns
    lab = lab.pivot(index='patientunitstayid', columns='labname',
                    values='labresult').reset_index()

    # Apply unit harmonization functions to specific columns
    lab['CPK'] = lab['CPK'].apply(eicu_CPK_harmonization)
    lab['troponin - T'] = lab['troponin - T'].apply(eicu_Troponin_harmonization)                    # noqa
    lab['creatinine'] = lab['creatinine'].apply(eicu_Creatinine_harmonization)                      # noqa
    lab['ALT (SGPT)'] = lab['ALT (SGPT)'].apply(eicul_ALAT_harmonization)

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
    treatment = pd.read_csv(os.path.join(root_dir, "treatment.csv.gz"),
                            usecols=usecols)

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
    treatment = treatment[treatment['treatmentstring'].apply(check_IABP)]

    # Define keywords to identify mechanical support
    keywords = ["removal"]
    # Filter the dataframe based on keywords
    treatment = treatment[treatment['treatmentstring'].apply(check_IABP_removal)]                   # noqa

    # Create a new column indicating mechanical support presence
    treatment["hpe_proc_mechs_yn"] = 1

    # Select relevant columns
    treatment = treatment.loc[:, ["patientunitstayid", "hpe_proc_mechs_yn"]]

    # Group by patientunitstayid and keep only the maximum value
    treatment = treatment.groupby("patientunitstayid").max().reset_index()

    # Assert uniqueness of patientunitstayid
    assert treatment["patientunitstayid"].nunique() == len(treatment["patientunitstayid"])          # noqa

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
    diagnosis = pd.read_csv(file_dir, usecols=["patientunitstayid", "icd9code",
                                               "diagnosisoffset"])

    # Keep only the diagnosis before the admission time
    diagnosis = diagnosis[diagnosis['diagnosisoffset'] < 0]

    # Drop rows with missing icd9code values
    diagnosis.dropna(subset=["icd9code"], inplace=True)

    # Extract primary icd9code from comma-separated list
    diagnosis['icd9code'] = diagnosis['icd9code'].apply(lambda x: x.split(',')[0])                                  # noqa

    # Filter patients diagnosed with diabetes mellitus (ICD-9 code: 250.00)
    diabetes_mellitus = diagnosis[diagnosis["icd9code"] == "250.00"]

    # Drop unnecessary columns
    diabetes_mellitus.drop(columns=["diagnosisoffset", "icd9code"],
                           inplace=True)

    # Group by patientunitstayid and keep only the maximum value
    diabetes_mellitus = diabetes_mellitus.groupby("patientunitstayid").max().reset_index()                          # noqa

    return diabetes_mellitus
