import numpy as np


def ck_unit_harmonization(data):
    # Define conversion factors for each unit
    unit_conversions = {
        1: 1.0,        # No conversion needed)
        2: 16.67,      # U/L to mmol/L (1 U = 0.01667 µkat)

    }
    # Harmonize the lab values
    data['icu_lab_ck_x'] = np.where(data['icu_lab_ck_x'] == 1, data['icu_lab_ck_x'], data['icu_lab_ck_x'] * unit_conversions[2])    # noqa
    data['ck_filled'] = data['ck'].fillna(data['icu_lab_ck_x'])

    return data


def lactate_unit_harmonization(data):
    # Define conversion factors for each unit
    unit_conversions = {
        1: 1.0,        # No conversion needed)
        2: 16.67,      # U/L to mmol/L (1 U = 0.01667 µkat)
        3: 0.2778      # mg/dL to mmol/L (1 mg/dL = 0.02778 mmol/L)
    }

    data['admission_lactate'] = np.where(data['icu_lab_lactunit_c_x'] == 1, data['admission_lactate'],                                                          # noqa
                                         np.where(data['icu_lab_lactunit_c_x'] == 2, data['admission_lactate'] * unit_conversions[2],                           # noqa
                                                  data['admission_lactate'] * unit_conversions[3]))                                             # noqa

    data['icu_lab_lact8hpci_x'] = np.where(data['icu_lab_lactunit_c_x'] == 1, data['icu_lab_lact8hpci_x'],                                                          # noqa
                                         np.where(data['icu_lab_lactunit_c_x'] == 2, data['icu_lab_lact8hpci_x'] * unit_conversions[2],                           # noqa
                                                  data['icu_lab_lact8hpci_x'] * unit_conversions[3]))                                      # noqa

    data['icu_lab_lact16hpci_x'] = np.where(data['icu_lab_lactunit_c_x'] == 1, data['icu_lab_lact16hpci_x'],                                                          # noqa
                                         np.where(data['icu_lab_lactunit_c_x'] == 2, data['icu_lab_lact16hpci_x'] * unit_conversions[2],                           # noqa
                                                  data['icu_lab_lact16hpci_x'] * unit_conversions[3]))                                             # noqa

    data['icu_lab_lact24hpci_x'] = np.where(data['icu_lab_lactunit_c_x'] == 1, data['icu_lab_lact24hpci_x'],                                                          # noqa
                                         np.where(data['icu_lab_lactunit_c_x'] == 2, data['icu_lab_lact24hpci_x'] * unit_conversions[2],                           # noqa
                                                  data['icu_lab_lact24hpci_x'] * unit_conversions[3]))                                             # noqa

    data['admission_lactate_filled'] = data["lac"].fillna(data["admission_lactate"])         # noqa

    return data


def creatine_unit_harmonization(data):

    # Define conversion factors for each unit
    unit_conversions = {
        1: 1.0,          # µmol/L (No conversion needed)
        2: 1000.0,       # nmol/mL to µmol/L (1 nmol/mL = 1000 µmol/L)
        3: 88.42         # mg/dL to µmol/L (1 mg/dL ≈ 88.42 µmol/L)
    }

    # Apply the unit conversions to the 'Creatine' column
    data['creatine'] = data.apply(lambda row: row['icu_lab_crea_x'] * unit_conversions[row['icu_lab_creaunit_c']], axis=1)  # noqa

    data['creatine_filled'] = data["cre"].fillna(data["creatine"])

    return data


def white_blood_count_unit_harmonization(data):

    # Define conversion factors for each unit
    unit_conversions = {
        1: 1.0,       # 10^9/L (No conversion needed)
        2: 1000.0,    # 10^6/mL to 10^9/L (1 10^6/mL = 1000 10^9/L)
        3: 1e-9       # Gpt/L to 10^9/L (1 Gpt/L = 1e-9 10^9/L)
    }

    # Apply the unit conversions to the 'Creatine' column
    data['white_cell_count'] = data.apply(lambda row: row['icu_lab_wct_x_x'] * unit_conversions[row['icu_lab_wbcunit_c']], axis=1)  # noqa

    return data


def hematocrit_unit_harmonization(data):

    unit_conversions = {
        1: 1.0,     # % (No conversion needed)
        2: 100.0    # L/L to % (1 L/L = 100%)
    }

    # Apply the unit conversions to the 'Hematocrit' column
    data['hematocrit'] = data.apply(lambda row: row['icu_lab_hct_x_x'] * unit_conversions[row['icu_lab_hctunit_c']], axis=1)  # noqa

    return data


def crp_unit_harmonization(data):
    # Define conversion factors for each unit
    unit_conversions = {
        1: 1.0,    # mg/dL (No conversion needed)
        2: 10.0    # mg/L to mg/dL (1 mg/L = 0.1 mg/dL)
    }

    # Apply the unit conversions to the 'CRP' column
    data['crp'] = data.apply(lambda row: row['icu_lab_crp_x'] * unit_conversions[row['icu_lab_crpunit_c']], axis=1)  # noqa

    return data


def glucose_unit_harmonization(data):

    # Define conversion factors for each unit
    unit_conversions = {
        1: 1.0,        # mg/dL (No conversion needed)
        2: 18.0182     # mmol/L to mg/dL (1 mmol/L = 18.0182 mg/dL)
    }

    # Apply the unit conversions to the 'Glucose' column
    data['glucose'] = data.apply(lambda row: row['icu_lab_glc_x'] * unit_conversions[row['icu_lab_glcunit_c']], axis=1)  # noqa
    data['glucose_filled'] = data["gluc"].fillna(data["glucose"])

    return data
