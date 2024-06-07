import pandas as pd
import numpy as np
import joblib
import os


def eICU_filter_CS_patients(data, CICU=True):

    if CICU:
        print("Only keeping patients admited in the Cardiac ICU")
        data = data[data["unittype"] == 'Cardiac ICU']
    CS_patients = data[data["icd9code"] == str(785.51)]      # noqa

    return CS_patients


def eICU_admission_heart_function(root_dir, plausible_range,
                                  selected_patients=None,
                                  time_before_cut_off=60,
                                  time_after_cut_off=-1440,
                                  prefilter=True):

    usecols = ["patientunitstayid", "observationoffset",
               "heartrate", "systemicsystolic", "systemicdiastolic"]
    # Load the pre-filtered data for faster processing
    if prefilter:
        heart_data = pd.read_csv(root_dir + "vitalPeriodic_selected_MACRO.csv.gz",                  # noqa
                                 usecols=usecols)
    else:
        # Load the raw data and keep the selected patients
        selected_patients = selected_patients.tolist()
        heart_data = pd.read_csv(root_dir + "vitalPeriodic_selected_MACRO.csv.gz",                  # noqa
                                 usecols=usecols)
        # Keep only the CS patients for faster processing
        heart_data = heart_data[heart_data["patientunitstayid"].isin(selected_patients)]            # noqa
    # Remove the data after one Hour
    heart_data = heart_data[heart_data['observationoffset'] < time_before_cut_off]                  # noqa
    # Remove data older than one day
    heart_data = heart_data[heart_data['observationoffset'] > time_after_cut_off]                   # noqa
    # Drop timing columne
    heart_data.drop(labels="observationoffset", axis=1, inplace=True)

    # Iterate over each range in plausible_range
    # to help removing outliers in the data
    for min_val, max_val, column in plausible_range:
        # Replace values outside the range with NaN for the specified column
        heart_data[column] = heart_data[column].where((heart_data[column] >= min_val) & (heart_data[column] <= max_val), other=np.nan)      # noqa

    # Group the data by the median
    heart_data = heart_data.groupby("patientunitstayid").median().reset_index()

    return heart_data


def eICU_aperiodic_preassure(root_dir, plausible_range,
                             selected_patients,
                             time_before_cut_off=30,
                             time_after_cut_off=-1440,
                             ):

    usecols = ["patientunitstayid", "observationoffset",
               "noninvasivesystolic", "noninvasivediastolic"]

    BP = pd.read_csv(root_dir + "vitalAperiodic.csv.gz",                  # noqa
                     usecols=usecols)

    selected_patients = selected_patients.tolist()
    # Keep only the CS patients for faster processing
    BP = BP[BP["patientunitstayid"].isin(selected_patients)]            # noqa
    # Remove the data after one Hour
    BP = BP[BP['observationoffset'] < time_before_cut_off]                  # noqa
    # Remove data older than one day
    BP = BP[BP['observationoffset'] > time_after_cut_off]                   # noqa
    # Drop timing columne
    BP.drop(labels="observationoffset", axis=1, inplace=True)

    # Iterate over each range in plausible_range
    # to help removing outliers in the data
    for min_val, max_val, column in plausible_range:
        # Replace values outside the range with NaN for the specified column
        BP[column] = BP[column].where((BP[column] >= min_val) & (BP[column] <= max_val), other=np.nan)      # noqa

    # Group the data by the median
    BP = BP.groupby("patientunitstayid").median().reset_index()

    return BP


# General propose functions
def process_individual_features(data: pd.DataFrame,
                                features_list: list,
                                field: str) -> pd.DataFrame:
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
def diferent_lactate_timing(row):
    if row['labname'] == 'lactate':
        if row['labresultoffset'] <= 0:
            return 'admission_lactate'
        elif 0 < row['labresultoffset'] <= 480:
            return 'icu_lab_lact8hpci_x'
        elif 480 < row['labresultoffset'] <= 960:
            return 'icu_lab_lact16hpci_x'
        elif 960 < row['labresultoffset'] <= 1440:
            return 'icu_lab_lact24hpci_x'
    return row['labname']


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


def admission_name_matching(X_admission: pd.DataFrame) -> pd.DataFrame:
    """
    Map column names from X_admission to a predefined set of column names
    and return the features corresponding to the feature names
    expected by the admission model.

    Args:
        X_admission (pd.DataFrame): Input DataFrame containing admission data.

    Returns:
        pd.DataFrame: DataFrame containing the admission data corrected.
    """
    # Define mapping from original column names to model feature names
    admission_colume_mapping = {'gender': "had_dem_male_yn",
                                'age': "had_dem_age_yr",
                                'admissionheight': 'had_pex_height_cm',
                                'admissionweight': 'had_pex_weight_kg',
                                'Atrial fibrillation': 'hpr_ecg_afib_y',
                                'hypertension requiring treatment': 'p_rf_aht_yn',                                      # noqa
                                'insulin dependent diabetes': 'p_rf_dm_yn',
                                'peripheral vascular disease': 'p_mh_pad_yn',
                                'sick sinus syndrome': 'hpr_ecg_sinrhy_y',
                                'Known coronary artery disease': 'previous_heart_complications',                        # noqa
                                'delusional': 'had_sy_ams_yn',
                                'extremities cold': 'had_sy_extremity_yn',
                                'Resuscitation within 24hs': 'had_base_cpr24h_yn',                                      # noqa
                                'dyslipidemia': 'p_rf_dyslip_yn',
                                'mechanical_ventilation': 'had_base_mechvent_yn',                                       # noqa
                                'ST_elevation': 'hpr_ecg_stemi_yn',
                                'heartrate': 'hpr_hmdyn_hr_bpm',
                                'systemicsystolic': 'hpr_hmdyn_sbp_mmhg',
                                'systemicdiastolic': 'hpr_hmdyn_dbp_mmhg'}

    # Rename columns according to the mapping
    X_admission.rename(columns=admission_colume_mapping, inplace=True)

    # Load admission model
    models_root = "/home/nnieto/Nico/MODS_project/CULPRIT_project/web_service/Synchon version/MACRO/macro/data/"        # noqa
    admission_model_path = os.path.join(models_root, "model_admission.pkl")
    Admission_model = joblib.load(admission_model_path)

    # Put the columns in the same order
    X_admission = X_admission[Admission_model.feature_names_in_]

    return X_admission


def full_model_name_matching(Full_model_data: pd.DataFrame) -> pd.DataFrame:
    """
    Map column names from Full_model_data to a predefined set of column names
    and return the features corresponding to the feature names
    expected by the full model.

    Args:
        Full_model_data (pd.DataFrame): Input DataFrame full model data.

    Returns:
        pd.DataFrame: DataFrame containing Full model corrected features.
    """
    # Define mapping from original column names to model feature names
    colume_mapping = {'gender': "had_dem_male_yn",
                      'age': "had_dem_age_yr",
                      'admissionheight': 'had_pex_height_cm',
                      'admissionweight': 'had_pex_weight_kg',
                      'Atrial fibrillation': 'hpr_ecg_afib_y',
                      'hypertension requiring treatment': 'p_rf_aht_yn',
                      'insulin dependent diabetes': 'p_rf_dm_yn',
                      'peripheral vascular disease': 'p_mh_pad_yn',
                      'sick sinus syndrome': 'hpr_ecg_sinrhy_y',
                      'Known coronary artery disease': 'previous_heart_complications',  # noqa
                      'delusional': 'had_sy_ams_yn',
                      'extremities cold': 'had_sy_extremity_yn',
                      'Resuscitation within 24hs': 'had_base_cpr24h_yn',
                      'dyslipidemia': 'p_rf_dyslip_yn',
                      'mechanical_ventilation': 'had_base_mechvent_yn',
                      'ST_elevation': 'hpr_ecg_stemi_yn',
                      'heartrate': 'hpr_hmdyn_hr_bpm',
                      'systemicsystolic': 'hpr_hmdyn_sbp_mmhg',
                      'systemicdiastolic': 'hpr_hmdyn_dbp_mmhg',
                      'BNP': "pbnp",
                      'CPK': "icu_lab_ck_x",
                      'troponin - T': 'tnt',
                      'creatinine': 'creatine',
                      'WBC x 1000': 'white_cell_count',
                      'Hct': 'hematocrit',
                      'CRP': 'crp',
                      'PT - INR': 'icu_lab_inr_r',
                      'glucose': 'glucose',
                      'ALT (SGPT)': 'alat'
                      }
    # Rename columns according to the mapping
    Full_model_data.rename(columns=colume_mapping, inplace=True)

    # Load full model
    models_root = "/home/nnieto/Nico/MODS_project/CULPRIT_project/web_service/Synchon version/MACRO/macro/data/"    # noqa
    full_model_path = os.path.join(models_root, "model_full.pkl")
    Full_model = joblib.load(full_model_path)

    # order the features
    Full_model_data = Full_model_data[Full_model.feature_names_in_]

    return Full_model_data
