import numpy as np
import pandas as pd


def ck_unit_harmonization(data: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonize the lab values for Creatine Kinase (CK).

    Parameters:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with harmonized CK values.
    """
    # Define conversion factors for each unit
    unit_conversions = {
        1: 1.0,   # No conversion needed
        2: 16.67  # U/L to mmol/L (1 U = 0.01667 µkat)
    }

    # Harmonize the lab values for CK
    data['icu_lab_ck_x'] = np.where(data['icu_lab_ck_x'] == 1,
                                    data['icu_lab_ck_x'],
                                    data['icu_lab_ck_x'] * unit_conversions[2])
    data['ck_filled'] = data['ck'].fillna(data['icu_lab_ck_x'])

    return data


def lactate_unit_harmonization(data: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonize the lab values for Lactate.

    Parameters:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with harmonized Lactate values.
    """
    # Define conversion factors for each unit
    unit_conversions = {
        1: 1.0,     # No conversion needed
        2: 16.67,   # U/L to mmol/L (1 U = 0.01667 µkat)
        3: 0.2778   # mg/dL to mmol/L (1 mg/dL = 0.02778 mmol/L)
    }

    # Harmonize the lab values for Lactate
    data['admission_lactate'] = np.where(data['icu_lab_lactunit_c_x'] == 1, data['admission_lactate'],              # noqa
                                         np.where(data['icu_lab_lactunit_c_x'] == 2,                                # noqa
                                                  data['admission_lactate'] * unit_conversions[2],                  # noqa
                                                  data['admission_lactate'] * unit_conversions[3]))                 # noqa

    data['icu_lab_lact8hpci_x'] = np.where(data['icu_lab_lactunit_c_x'] == 1, data['icu_lab_lact8hpci_x'],          # noqa
                                           np.where(data['icu_lab_lactunit_c_x'] == 2,                              # noqa
                                                    data['icu_lab_lact8hpci_x'] * unit_conversions[2],              # noqa
                                                    data['icu_lab_lact8hpci_x'] * unit_conversions[3]))             # noqa

    data['icu_lab_lact16hpci_x'] = np.where(data['icu_lab_lactunit_c_x'] == 1, data['icu_lab_lact16hpci_x'],        # noqa
                                            np.where(data['icu_lab_lactunit_c_x'] == 2,                             # noqa
                                                     data['icu_lab_lact16hpci_x'] * unit_conversions[2],            # noqa
                                                     data['icu_lab_lact16hpci_x'] * unit_conversions[3]))           # noqa

    data['icu_lab_lact24hpci_x'] = np.where(data['icu_lab_lactunit_c_x'] == 1, data['icu_lab_lact24hpci_x'],        # noqa
                                            np.where(data['icu_lab_lactunit_c_x'] == 2,                             # noqa
                                                     data['icu_lab_lact24hpci_x'] * unit_conversions[2],            # noqa
                                                     data['icu_lab_lact24hpci_x'] * unit_conversions[3]))           # noqa

    data = lactate_in_biological_range(data, 'admission_lactate')
    data = lactate_in_biological_range(data, 'icu_lab_lact8hpci_x')
    data = lactate_in_biological_range(data, 'icu_lab_lact16hpci_x')
    data = lactate_in_biological_range(data, 'icu_lab_lact24hpci_x')

    return data


def lactate_in_biological_range(data, lactate_measure):

    # Puts the data in a biologically range
    data[lactate_measure].loc[data[lactate_measure] > 1000] /= 1000
    # Puts the data in a biologically range
    data[lactate_measure].loc[data[lactate_measure] > 25] /= 100

    return data


def creatine_unit_harmonization(data: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonize the lab values for Creatine.

    Parameters:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with harmonized Creatine values.
    """
    # Define conversion factors for each unit
    unit_conversions = {
        1: 1.0,        # µmol/L (No conversion needed)
        2: 1000.0,     # nmol/mL to µmol/L (1 nmol/mL = 1000 µmol/L)
        3: 88.42       # mg/dL to µmol/L (1 mg/dL ≈ 88.42 µmol/L)
    }

    # Apply the unit conversions to the 'Creatine' column
    data['creatine'] = data.apply(lambda row: row['icu_lab_crea_x'] * unit_conversions[row['icu_lab_creaunit_c']], axis=1)      # noqa
    data['creatine_filled'] = data["cre"].fillna(data["creatine"])

    return data


def white_blood_count_unit_harmonization(data: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonize the lab values for White Blood Count.

    Parameters:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with harmonized White Blood Count values.
    """

    # Apply the unit conversions to the 'White Cell Count' column
    wcc = data["icu_lab_wct_x_x"]
    # Puts the data in a biologically range
    wcc.loc[wcc > 1000] /= 1000
    # Puts the data in a biologically range
    wcc.loc[wcc > 100] /= 100

    data['white_cell_count'] = wcc

    return data


def hematocrit_unit_harmonization(data: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonize the lab values for Hematocrit.

    Parameters:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with harmonized Hematocrit values.
    """
    # Define conversion factors for each unit
    unit_conversions = {
        1: 1.0,    # % (No conversion needed)
        2: 100.0   # L/L to % (1 L/L = 100%)
    }

    # Apply the unit conversions to the 'Hematocrit' column
    data['hematocrit'] = data.apply(lambda row: row['icu_lab_hct_x_x'] * unit_conversions[row['icu_lab_hctunit_c']], axis=1)    # noqa
    # Correct values
    data['hematocrit'].loc[data['hematocrit'] > 100] /= 100
    # Puts the data in a biologically range
    data['hematocrit'].loc[1 > data['hematocrit']] *= 100

    return data


def crp_unit_harmonization(data: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonize the lab values for C-Reactive Protein (CRP).

    Parameters:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with harmonized CRP values.
    """
    # Define conversion factors for each unit
    unit_conversions = {
        1: 1.0,    # mg/dL (No conversion needed)
        2: 10.0    # mg/L to mg/dL (1 mg/L = 0.1 mg/dL)
    }

    # Apply the unit conversions to the 'CRP' column
    data['crp'] = data.apply(lambda row: row['icu_lab_crp_x'] * unit_conversions[row['icu_lab_crpunit_c']], axis=1)         # noqa

    return data


def glucose_unit_harmonization(data: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonize the lab values for Glucose.

    Parameters:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with harmonized Glucose values.
    """
    # Define conversion factors for each unit
    unit_conversions = {
        1: 1.0,       # mg/dL (No conversion needed)
        2: 18.0182    # mmol/L to mg/dL (1 mmol/L = 18.0182 mg/dL)
    }

    # Apply the unit conversions to the 'Glucose' column
    data['glucose'] = data.apply(lambda row: row['icu_lab_glc_x'] * unit_conversions[row['icu_lab_glcunit_c']], axis=1)     # noqa
    data['glucose_filled'] = data["gluc"].fillna(data["glucose"])

    return data
