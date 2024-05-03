import pandas as pd


def eICU_filter_CS_patients(data, CICU=True):

    if CICU:
        print("Only keeping patients admited in the Cardiac ICU")
        data = data[data["unittype"] == 'Cardiac ICU']
    CS_patients = data[data["icd9code"] == str(785.51)]      # noqa

    return CS_patients


def eICU_admission_heart_function(root_dir, selected_patients,
                                  time_before_cut_off=60,
                                  time_after_cut_off=-1440):
    selected_patients = selected_patients.tolist()
    usecols = ["patientunitstayid", "observationoffset",
               "heartrate", "systemicsystolic", "systemicdiastolic"]

    heart_data = pd.read_csv(root_dir + "vitalPeriodic.csv.gz",
                             usecols=usecols)
    # Keep only the CS patients for faster processing
    heart_data = heart_data[heart_data["patientunitstayid"].isin(selected_patients)]            # noqa
    # Remove the data after one Hour
    heart_data = heart_data[heart_data['observationoffset'] < time_before_cut_off]              # noqa
    # Remove data older than one day
    heart_data = heart_data[heart_data['observationoffset'] > time_after_cut_off]               # noqa
    # Drop timing columne
    heart_data.drop(labels="observationoffset", axis=1, inplace=True)
    # Group the data by the median
    heart_data = heart_data.groupby("patientunitstayid").median().reset_index()

    return heart_data
