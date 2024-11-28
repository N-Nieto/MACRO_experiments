import pandas as pd
import numpy as np
import os
import sys
from lib.unit_harmonization import ck_unit_harmonization, crp_unit_harmonization, creatine_unit_harmonization   # noqa
from lib.unit_harmonization import glucose_unit_harmonization, lactate_unit_harmonization                       # noqa
from lib.unit_harmonization import hematocrit_unit_harmonization, white_blood_count_unit_harmonization          # noqa
# Append project path for locating the data
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname((os.path.dirname((__file__)))))))               # noqa
sys.path.append(project_root)


def load_CULPRIT_data(data_dir: str) -> pd.DataFrame:
    '''
    Load CULPRIT study data, keep only the first day information, and harmonize units.
    # noqa
    Parameters:
        data_dir (str): The directory path where the data is located.

    Returns:
        pd.DataFrame: Processed and harmonized CULPRIT study data.
    '''

    # Load main data
    data = pd.read_excel(project_root + data_dir + "CULPRIT-data_20210407.xlsx",            # noqa
                         sheet_name=None)

    # Extract patient data
    patient_info = data["patient_data"]
    # Set patient ID as unique
    patient_info = create_unique_patient_ID(patient_info)
    # Include the combined variable in the patients information
    patient_info = create_previous_heart_complications(patient_info)
    # Create admission lactate
    patient_info = create_admission_lactate(patient_info)
    # Add Catecholamine Therapy information
    patient_info = add_catecholamine_therapy(data, patient_info)
    # Add Renal Replacement Therapy information
    patient_info = add_renal_replacement_therapy(data, patient_info)
    # Add Sepsis information
    patient_info = add_sepsis(data, patient_info)
    # Add Ventricular Fibrillation information
    patient_info = add_ventricular_fibrillation(data, patient_info)
    # Add stroke information
    patient_info = add_stroke(data, patient_info)
    # Add Resusitation within first 24hs
    patient_info = add_resusitation_24hs(patient_info)

    # Load Laboratory data
    lab_info = data["laboratory_data"]
    # Set patient ID as unique
    lab_info = create_unique_patient_ID(lab_info)

    # Keep the information for the first day.
    lab_info = lab_info[lab_info["icu_lab_day_text"] == 1]
    # The day information is not longer required
    lab_info.drop(columns="icu_lab_day_text", inplace=True)

    # Load clip data
    clip_info = data["clip"]
    # Change formating to match the rest of the variables
    clip_info.columns = clip_info.keys().str.lower()
    # Set patient ID as unique
    clip_info = create_unique_patient_ID(clip_info)

    # Get clip information only for TIME V1 == Admission
    clip_info = clip_info[clip_info["time"] == "V1"]
    # Time information is not longer required
    clip_info.drop(columns="time", inplace=True)

    # First merge patient and lab data
    data_final = pd.merge(patient_info, lab_info, on="patient_ID")
    # Add also the clip information
    data_final = data_final.merge(clip_info, on="patient_ID", how="left")

    # Calculate CLIP score from features
    data_final = calcule_CLIP_score_from_features(data_final)

    # Harmonize units
    data_final = ck_unit_harmonization(data_final)
    data_final = lactate_unit_harmonization(data_final)
    data_final = creatine_unit_harmonization(data_final)
    data_final = white_blood_count_unit_harmonization(data_final)
    data_final = hematocrit_unit_harmonization(data_final)
    data_final = crp_unit_harmonization(data_final)
    data_final = glucose_unit_harmonization(data_final)

    return data_final


def create_unique_patient_ID(data: pd.DataFrame) -> pd.DataFrame:
    """
    # noqa
    Combine the center ID and the patient ID to generate a unique ID.

    Parameters:
        data (pd.DataFrame): The input DataFrame containing center and patient IDs.

    Returns:
        pd.DataFrame: The DataFrame with a new column 'patient_ID' representing the unique ID.
    """
    data["patient_ID"] = data["cor_ctr_center_id"].astype(str) + "-" + data["cor_pt_caseno_id"].astype(str)     # noqa
    return data


def load_table(data_dir: str) -> pd.DataFrame:
    """
    Load a table from an Excel file.

    Parameters:
        data_dir (str): The directory containing the Excel file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    table = pd.read_excel(data_dir + "Shorten_table.xlsx",
                          sheet_name=None, index_col=0)
    return table['Sheet1']


def get_data_from_features(data: pd.DataFrame,
                           features_code: list) -> pd.DataFrame:
    """
    Extract features from patient data.

    Parameters:
        data (pd.DataFrame): The input DataFrame containing patient data.
        features_code (list): List of feature codes to extract.

    Returns:
        pd.DataFrame: DataFrame containing selected features.
    """
    return data.loc[:, features_code]


def create_previous_heart_complications(patient_info: pd.DataFrame
                                        ) -> pd.DataFrame:
    """
    Create a combined variable based on previous medical history.

    Parameters:
        patient_info (pd.DataFrame): DataFrame containing patient information.

    Returns:
        pd.DataFrame: DataFrame with 'previous_heart_complications'.
    """
    var1 = patient_info["p_mh_mi_yn"]
    var2 = patient_info["p_mh_pci_yn"]
    var3 = patient_info["p_mh_cabg_yn"]
    series_sum = var1 + var2 + var3
    patient_info["previous_heart_complications"] = (series_sum > 0).astype(int)
    return patient_info


def calcule_CLIP_score_from_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate CLIP score from features.

    Parameters:
        data (pd.DataFrame): The input DataFrame containing relevant features.

    Returns:
        pd.DataFrame: The DataFrame with a new column 'CLIP_Score'.
    """
    C = np.arcsinh(100 * data["cysc_s_1"])
    L = np.arcsinh(100 * data["lac"])
    Inter = np.arcsinh(100 * data["il_6"])
    P = np.arcsinh(100 * data["pbnp"])

    Linear_predictor = -15.8532036 + 0.06714669 * C + 1.0287073 * L + 0.2704829 * Inter + 0.1923877 * P     # noqa
    data["CLIP_Score"] = 100 * np.exp(Linear_predictor) / (1 + np.exp(Linear_predictor))                    # noqa

    return data


def create_admission_lactate(patient_info: pd.DataFrame) -> pd.DataFrame:
    """
    Create admission lactate variable.

    Parameters:
        patient_info (pd.DataFrame): DataFrame containing patient information.

    Returns:
        pd.DataFrame: DataFrame with a new column 'admission_lactate'.
    """
    patient_info["admission_lactate"] = patient_info['icu_lab_lactpopci_x'].combine_first(patient_info['icu_lab_lactprepci_x']) # noqa
    return patient_info


def get_admission_date(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract admission date from patient information.

    Parameters:
        data: The input data.

    Returns:
        pd.DataFrame: DataFrame with admission date.
    """
    patient_info = data["patient_data"]
    patient_info = create_unique_patient_ID(patient_info)

    admission_date = patient_info.loc[:, ["had_base_admis_date", "patient_ID"]]
    return admission_date


def add_catecholamine_therapy(data: pd.DataFrame,
                              patient_info: pd.DataFrame) -> pd.DataFrame:
    """
    Add catecholamine therapy information to patient data.

    Parameters:
        data: The input data.
        patient_info (pd.DataFrame): DataFrame containing patient information.
    # noqa
    Returns:
        pd.DataFrame: Updated patient_info DataFrame with 'cathecholamine_therapy' column.
    """
    cate = data["cathecholamine"]
    cate = create_unique_patient_ID(cate)

    cate = cate.loc[:, ["icu_catec_start_date", "patient_ID"]]
    admission_date = get_admission_date(data)

    cate['date_cate_start'] = pd.to_datetime(cate['icu_catec_start_date'])
    admission_date['date_admission'] = pd.to_datetime(admission_date['had_base_admis_date'])                            # noqa

    cate['cathecholamine_therapy'] = (cate['date_cate_start'] - admission_date['date_admission']).dt.days.isin([0, 1])  # noqa
    cate_df = cate.groupby('patient_ID')['cathecholamine_therapy'].any().reset_index()                                  # noqa
    cate_df.replace({True: 1, False: 0}, inplace=True)

    patient_info = pd.merge(patient_info, cate_df, on="patient_ID")

    return patient_info


def add_renal_replacement_therapy(data: pd.DataFrame,
                                  patient_X: pd.DataFrame) -> pd.DataFrame:
    """
    Add renal replacement therapy information to patient data.
    # noqa
    Parameters:
        data: The input data.
        patient_X (pd.DataFrame): DataFrame containing patient information.

    Returns:
        pd.DataFrame: Updated patient_X DataFrame with 'renal_replacement_therapy' column.
    """
    # Search where the renal_replacement_therapy is
    rrt = data["endpoint"]
    # Create a unique patient ID
    rrt = create_unique_patient_ID(rrt)
    # Keep only the starting date and the patient ID
    rrt = rrt.loc[:, ["icu_rrt_start_date", "patient_ID"]]
    # Get the admission date for the patients
    admission_date = get_admission_date(data)
    # Change
    rrt['date_rrt_start'] = pd.to_datetime(rrt['icu_rrt_start_date'])
    admission_date['date_admission'] = pd.to_datetime(admission_date['had_base_admis_date'])                            # noqa

    rrt['renal_replacement_therapy'] = (rrt['date_rrt_start'] - admission_date['date_admission']).dt.days.isin([0, 1])  # noqa
    rrt_df = rrt.groupby('patient_ID')['renal_replacement_therapy'].any().reset_index()                                 # noqa
    rrt_df.replace({True: 1, False: 0}, inplace=True)

    patient_X = patient_X.merge(rrt_df, on="patient_ID", how="left")

    return patient_X


def add_sepsis(data: pd.DataFrame, patient_info: pd.DataFrame) -> pd.DataFrame:
    """
    Add sepsis information to patient data.

    Parameters:
        data: The input data.
        patient_info (pd.DataFrame): DataFrame containing patient information.

    Returns:
        pd.DataFrame: Updated patient_info DataFrame with 'sepsis' column.
    """
    dates = data["patient_data"].loc[:, ["h_ev_sepsis_date", "had_base_admis_date", "patient_ID"]]          # noqa

    dates['date_sepsis_start'] = pd.to_datetime(dates['h_ev_sepsis_date'])
    dates['date_admission'] = pd.to_datetime(dates['had_base_admis_date'])

    dates['sepsis'] = (dates['date_sepsis_start'] - dates['date_admission']).dt.days.isin([0, 1])           # noqa
    sepsis_df = dates.groupby('patient_ID')['sepsis'].any().reset_index()
    sepsis_df.replace({True: 1, False: 0}, inplace=True)

    patient_info = pd.merge(patient_info, sepsis_df, on="patient_ID")

    return patient_info


def add_ventricular_fibrillation(data: pd.DataFrame,
                                 patient_info: pd.DataFrame) -> pd.DataFrame:
    """
    Add ventricular fibrillation information to patient data.
    # noqa
    Parameters:
        data: The input data.
        patient_info (pd.DataFrame): DataFrame containing patient information.

    Returns:
        pd.DataFrame: Updated patient_info DataFrame with 'ventricular_fibrillation' column.
    """
    dates = data["patient_data"].loc[:, ["h_ev_vfib_date", "had_base_admis_date", "patient_ID"]]        # noqa

    dates['date_vf_start'] = pd.to_datetime(dates['h_ev_vfib_date'])
    dates['date_admission'] = pd.to_datetime(dates['had_base_admis_date'])

    dates['ventricular_fibrillation'] = (dates['date_vf_start'] - dates['date_admission']).dt.days.isin([0, 1])         # noqa
    vf_df = dates.groupby('patient_ID')['ventricular_fibrillation'].any().reset_index()                                 # noqa
    vf_df.replace({True: 1, False: 0}, inplace=True)

    patient_info = pd.merge(patient_info, vf_df, on="patient_ID")

    return patient_info


def add_stroke(data: pd.DataFrame, patient_info: pd.DataFrame) -> pd.DataFrame:
    """
    Add stroke information to patient data.
    # noqa
    Parameters:
        data: The input data.
        patient_info (pd.DataFrame): DataFrame containing patient information.

    Returns:
        pd.DataFrame: Updated patient_info DataFrame with 'stroke' column.
    """
    dates = data["patient_data"].loc[:, ["h_ev_stroke_date", "had_base_admis_date", "patient_ID"]]      # noqa

    dates['date_stroke_start'] = pd.to_datetime(dates['h_ev_stroke_date'])
    dates['date_admission'] = pd.to_datetime(dates['had_base_admis_date'])

    dates['stroke'] = (dates['date_stroke_start'] - dates['date_admission']).dt.days.isin([0, 1])       # noqa
    stroke_df = dates.groupby('patient_ID')['stroke'].any().reset_index()
    stroke_df.replace({True: 1, False: 0}, inplace=True)

    patient_info = pd.merge(patient_info, stroke_df, on="patient_ID")

    return patient_info


def add_resusitation_24hs(patient_info: pd.DataFrame) -> pd.DataFrame:
    """
    Add resuscitation within 24 hours information to patient data.

    Parameters:
        patient_info (pd.DataFrame): DataFrame containing patient information.

    Returns:
        pd.DataFrame: Updated patient_info with 'resuscitation_24hs' column.
    """
    var1 = patient_info["had_base_cpr24h_yn"]
    var2 = patient_info["ventricular_fibrillation"]

    series_sum = var1 + var2
    patient_info["resuscitation_24hs"] = (series_sum > 0).astype(int)

    return patient_info


def load_eICU(features, exclude_smokers, X_CULPRIT):

    eicu_root = project_root + "data/eicu-collaborative-research-database-2.0/preprocessed_MACRO/"      # noqa
    X_eicu = pd.read_csv(eicu_root + "X_"+features+"_CICU_No_aperiodic.csv",
                         index_col=0)
    if exclude_smokers:
        X_eicu = X_eicu.drop(columns="p_rf_smoker_yn")

    Y_test_eicu = pd.read_csv(eicu_root + "y_CICU.csv", index_col=0)
    Y_test_eicu = Y_test_eicu.to_numpy()

    # Get Same naming
    X_eicu = pd.DataFrame(X_eicu, columns=X_CULPRIT.columns)

    return X_eicu, Y_test_eicu
