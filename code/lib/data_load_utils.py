import pandas as pd
import numpy as np
from lib.unit_harmonization import ck_unit_harmonization, crp_unit_harmonization, creatine_unit_harmonization   # noqa
from lib.unit_harmonization import glucose_unit_harmonization, lactate_unit_harmonization                       # noqa
from lib.unit_harmonization import hematocrit_unit_harmonization, white_blood_count_unit_harmonization          # noqa


def load_CULPRIT_data(data_dir):
    ''' Load data, keep only the first day information, Harmonize units
    '''
    # Load main data
    data = pd.read_excel(data_dir + "CULPRIT-data_20210407.xlsx",
                         sheet_name=None)

    # Extract patient data
    patient_info = data["patient_data"]
    # Set patient ID as unique
    patient_info = create_unique_patient_ID(patient_info)
    # Include the combined variable in the patients information
    patient_info = create_combined_variable(patient_info)
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


def create_unique_patient_ID(data):
    # Combine the center ID and the patient ID in that center
    # to generate a unique ID
    data["patient_ID"] = data["cor_ctr_center_id"].astype(str) + "-" + data["cor_pt_caseno_id"].astype(str) # noqa
    return data


def load_table(data_dir):
    table = pd.read_excel(data_dir + "Shorten_table.xlsx", sheet_name=None,
                          index_col=0)

    return table['Sheet1']


def get_data_from_features(data, features_code):
    # Get features from patient data
    return data.loc[:, features_code]


def create_combined_variable(patient_info):
    # Get Previous myocardial infarction
    var1 = patient_info["p_mh_mi_yn"]
    # Get Previous percutaneous coronary intervention (PCI)
    var2 = patient_info["p_mh_pci_yn"]
    # Get Previous coronary artery bypass grafting surgery
    var3 = patient_info["p_mh_cabg_yn"]
    # Sum the variables
    series_sum = var1 + var2 + var3
    # If some of the features is 1, the variable will be 1
    patient_info["combined_variable"] = (series_sum > 0).astype(int)
    return patient_info


def calcule_CLIP_score_from_features(data):
    C = np.arcsinh(100 * data["cysc_s_1"])
    L = np.arcsinh(100 * data["lac"])
    Inter = np.arcsinh(100 * data["il_6"])
    P = np.arcsinh(100 * data["pbnp"])

    Linear_predictor = - 15.8532036 + 0.06714669 * C + 1.0287073 * L + 0.2704829 * Inter + 0.1923877 * P        # noqa

    data["CLIP_Score"] = 100*np.exp(Linear_predictor) / (1 + np.exp(Linear_predictor))                          # noqa

    return data


# Create admission lactate
def create_admission_lactate(patient_info):
    patient_info["admission_lactate"] = patient_info['icu_lab_lactpopci_x'].combine_first(patient_info['icu_lab_lactprepci_x'])     # noqa
    return patient_info


def get_admission_date(data):
    # select the patient information
    patient_info = data["patient_data"]
    patient_info = create_unique_patient_ID(patient_info)

    admission_date = patient_info.loc[:, ["had_base_admis_date", "patient_ID"]]
    return admission_date


def add_catecholamine_therapy(data, patient_info):
    cate = data["cathecholamine"]
    cate = create_unique_patient_ID(cate)
    # Select the start  date of cathecholamine therapy
    cate = cate.loc[:, ["icu_catec_start_date", "patient_ID"]]
    # Get the patients admission date
    admission_date = get_admission_date(data)
    # Get numeric date
    cate['date_cate_start'] = pd.to_datetime(cate['icu_catec_start_date'])                                    # noqa
    # Get numeric date
    admission_date['date_admission'] = pd.to_datetime(admission_date['had_base_admis_date'])                                      # noqa
    # Check if the treatment started in the first 24hs
    cate['cathecholamine_therapy'] = (cate['date_cate_start'] - admission_date['date_admission']).dt.days.isin([0, 1])    # noqa
    # As several therapies can start at different time
    # check if any started at admission
    cate_df = cate.groupby('patient_ID')['cathecholamine_therapy'].any().reset_index()                                    # noqa    
    # Replace True and False
    cate_df.replace({True: 1, False: 0}, inplace=True)

    # Put the information in the patient data and return
    patient_info = pd.merge(patient_info, cate_df, on="patient_ID")

    return patient_info


def add_renal_replacement_therapy(data, patient_X):
    rrt = data["endpoint"]
    rrt = create_unique_patient_ID(rrt)
    # Select the start  date of replacement_therapy
    rrt = rrt.loc[:, ["icu_rrt_start_date", "patient_ID"]]
    # Get the patients admission date
    admission_date = get_admission_date(data)
    # Get numeric date
    rrt['date_rrt_start'] = pd.to_datetime(rrt['icu_rrt_start_date'])
    # Get numeric date
    admission_date['date_admission'] = pd.to_datetime(admission_date['had_base_admis_date'])                                        # noqa
    # Check if the treatment started in the first 24hs
    rrt['renal_replacement_therapy'] = (rrt['date_rrt_start'] - admission_date['date_admission']).dt.days.isin([0, 1])              # noqa
    # As several therapies can start at different time
    # check if any started at admission
    rrt_df = rrt.groupby('patient_ID')['renal_replacement_therapy'].any().reset_index()                                             # noqa    
    # Replace True and False
    rrt_df.replace({True: 1, False: 0}, inplace=True)
    # merge the results
    patient_X = patient_X.merge(rrt_df, on="patient_ID", how="left")

    return patient_X


def add_sepsis(data, patient_info):
    # Select the start  date of sepsis
    dates = data["patient_data"].loc[:, ["h_ev_sepsis_date",
                                         "had_base_admis_date", "patient_ID"]]
    # Get numeric date
    dates['date_sepsis_start'] = pd.to_datetime(dates['h_ev_sepsis_date'])                                      # noqa
    # Get numeric date
    dates['date_admission'] = pd.to_datetime(dates['had_base_admis_date'])                                      # noqa
    # Check if the treatment started in the first 24hs
    dates['sepsis'] = (dates['date_sepsis_start'] - dates['date_admission']).dt.days.isin([0, 1])               # noqa

    sepsis_df = dates.groupby('patient_ID')['sepsis'].any().reset_index()                                       # noqa    
    sepsis_df.replace({True: 1, False: 0}, inplace=True)

    sepsis_df['sepsis'].value_counts()
    patient_info = pd.merge(patient_info, sepsis_df, on="patient_ID")

    return patient_info


def add_ventricular_fibrillation(data, patient_info):
    # Select the start  date of ventricular_fibrillation
    dates = data["patient_data"].loc[:, ["h_ev_vfib_date",
                                         "had_base_admis_date", "patient_ID"]]
    # Get numeric date
    dates['date_vf_start'] = pd.to_datetime(dates['h_ev_vfib_date'])                                    # noqa
    # Get numeric date
    dates['date_admission'] = pd.to_datetime(dates['had_base_admis_date'])                                      # noqa
    # Check if the treatment started in the first 24hs
    dates['ventricular_fibrillation'] = (dates['date_vf_start'] - dates['date_admission']).dt.days.isin([0, 1])    # noqa

    vf_df = dates.groupby('patient_ID')['ventricular_fibrillation'].any().reset_index()                                    # noqa    
    vf_df.replace({True: 1, False: 0}, inplace=True)

    patient_info = pd.merge(patient_info, vf_df, on="patient_ID")
    return patient_info


def add_stroke(data, patient_info):
    # Select the start  date of stroke
    dates = data["patient_data"].loc[:, ["h_ev_stroke_date",
                                         "had_base_admis_date", "patient_ID"]]
    # Get numeric date
    dates['date_stroke_start'] = pd.to_datetime(dates['h_ev_stroke_date'])                                    # noqa
    # Get numeric date
    dates['date_admission'] = pd.to_datetime(dates['had_base_admis_date'])                                      # noqa
    # Check if the treatment started in the first 24hs
    dates['stroke'] = (dates['date_stroke_start'] - dates['date_admission']).dt.days.isin([0, 1])    # noqa

    stroke_df = dates.groupby('patient_ID')['stroke'].any().reset_index()                                    # noqa    
    stroke_df.replace({True: 1, False: 0}, inplace=True)

    stroke_df['stroke'].value_counts()
    patient_info = pd.merge(patient_info, stroke_df, on="patient_ID")
    return patient_info


def add_resusitation_24hs(patient_info):
    # Get Resuscitation within 24h before randomization
    var1 = patient_info["had_base_cpr24h_yn"]
    # Get ventricular_fibrillation
    var2 = patient_info["ventricular_fibrillation"]
    # Sum the variables
    series_sum = var1 + var2
    # If some of the features is 1, the variable will be 1
    patient_info["resusitation_24hs"] = (series_sum > 0).astype(int)
    return patient_info
