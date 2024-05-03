import pandas as pd


def eICU_load_history(root_dir):

    usecols = ["patientunitstayid", "pasthistoryvalue"]

    past_history = pd.read_csv(root_dir + "pastHistory.csv.gz",
                               usecols=usecols)

    features_history = ["patientunitstayid",
                        "atrial fibrillation - chronic",
                        "atrial fibrillation - intermittent",
                        "CABG - date unknown",
                        "CABG - remote", "CABG - within 2 years",
                        "CABG - within 5 years", "CABG - within 6 months",
                        "hypertension requiring treatment",
                        "insulin dependent diabetes",
                        "MI - date unknown", "MI - remote",
                        "MI - within 2 years",
                        "MI - within 5 years",
                        "MI - within 6 months",
                        "peripheral vascular disease",
                        "procedural coronary intervention - date unknown",
                        "procedural coronary intervention - remote",
                        "procedural coronary intervention - within 2 years",
                        "procedural coronary intervention - within 5 years",
                        "procedural coronary intervention - within 6 months",
                        "sick sinus syndrome"]

    # Pre filter the rows with useful information
    #  to reduce the computation timing
    past_history = past_history[past_history["pasthistoryvalue"].isin(features_history)]            # noqa

    # Pass from the diagnosis to a columne
    features_history.remove("patientunitstayid")

    for feature in features_history:
        # Add a new column where 1 indicates the feature
        # was presented and 0 otherwise
        past_history[feature] = (past_history['pasthistoryvalue'] == feature).astype(int)           # noqa

    # Group different features
    features_group = {'Known coronary artery disease':
                      ['MI - date unknown',
                       'MI - remote',
                       'MI - within 2 years',
                       "MI - within 5 years", "MI - within 6 months",
                       "procedural coronary intervention - date unknown",
                       "procedural coronary intervention - remote",
                       "procedural coronary intervention - within 2 years",
                       "procedural coronary intervention - within 5 years",
                       "procedural coronary intervention - within 6 months",
                       "CABG - date unknown", "CABG - remote",
                       "CABG - within 2 years", "CABG - within 5 years",
                       "CABG - within 6 months"],
                      'Atrial fibrilation':
                      ["atrial fibrillation - chronic",
                       "atrial fibrillation - intermittent"]
                      }

    # Iterate through features_group and create a new column for each group
    for group, features in features_group.items():
        # Add a new column where 1 indicates any of the features
        # in the group were presented and 0 otherwise
        past_history[group] = (past_history[features].sum(axis=1) > 0).astype(int)                  # noqa
        # Remove the individual features that are part of the group
        past_history.drop(features, axis=1, inplace=True)

    selected_features = ['patientunitstayid',
                         'atrial fibrillation - chronic',
                         'hypertension requiring treatment',
                         'insulin dependent diabetes',
                         'peripheral vascular disease',
                         'sick sinus syndrome',
                         'Known coronary artery disease']

    past_history = past_history.loc[:, selected_features]

    past_history = past_history.groupby("patientunitstayid").max().reset_index()                        # noqa

    assert past_history["patientunitstayid"].nunique() == past_history["patientunitstayid"].__len__()   # noqa
    return past_history


def eICU_load_physical_exam(root_dir):

    usecols = ["patientunitstayid", "physicalexamvalue"]

    physicalExam = pd.read_csv(root_dir + "physicalExam.csv.gz",
                               usecols=usecols)

    features_exam = ["patientunitstayid", "delusional",
                     "extremities cold/dusky", "extremities cool"
                     ]
    physicalExam = physicalExam[physicalExam["physicalexamvalue"].isin(features_exam)]                  # noqa

    # Pass from the diagnosis to a columne
    features_exam.remove("patientunitstayid")

    for feature in features_exam:
        # Add a new column where 1 indicates the feature was
        # presented and 0 otherwise
        physicalExam[feature] = (physicalExam['physicalexamvalue'] == feature).astype(int)              # noqa

    physicalExam.drop("physicalexamvalue", axis=1, inplace=True)
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
    physicalExam = physicalExam.groupby("patientunitstayid").max().reset_index()                        # noqa

    assert physicalExam["patientunitstayid"].nunique() == physicalExam["patientunitstayid"].__len__()   # noqa

    return physicalExam


def eICU_load_diagnosis(root_dir):

    usecols = ["patientunitstayid", "icd9code"]

    diagnosis = pd.read_csv(root_dir + "diagnosis.csv.gz", usecols=usecols)

    diagnosis = diagnosis.dropna(subset="icd9code")

    diagnosis['icd9code'] = diagnosis['icd9code'].apply(lambda x: x.split(',')[0])                      # noqa

    return diagnosis


# Data loading
def eICU_load_patient_information(root_dir):
    usecols = ["patientunitstayid", "unittype", "uniquepid",
               "gender", "age", "admissionheight", "admissionweight",
               "hospitaldischargestatus"]
    patients = pd.read_csv(root_dir + "patient.csv.gz", usecols=usecols)
    patients["age"].replace({"> 89": 90}, inplace=True)

    patients["gender"].replace({"Female": 0, "Male": 1}, inplace=True)
    patients["hospitaldischargestatus"].replace({"Alive": 0, "Expired": 1},
                                                inplace=True)
    assert patients["patientunitstayid"].nunique() == patients["patientunitstayid"].__len__()           # noqa

    return patients


def eICU_desfibrilation(root_dir):
    usecols = ["patientunitstayid", "treatmentoffset", "treatmentstring"]
    treatment = pd.read_csv(root_dir + "treatment.csv.gz", usecols=usecols)

    # Keep only the information of the previos 24hs
    treatment = treatment[treatment['treatmentoffset'] < 0]
    treatment = treatment[treatment['treatmentoffset'] > -1440]

    feature_name = ["cardiovascular|arrhythmias|cardiac defibrillation",
                    "cardiovascular|non-operative procedures|defibrillation"]

    for feature in feature_name:
        # Add a new column where 1 indicates the feature
        # was presented and 0 otherwise
        treatment[feature] = (treatment['treatmentstring'] == feature).astype(int)                      # noqa

    # Group different features
    features_group = {'Resusitation within 24hs':
                      ['cardiovascular|arrhythmias|cardiac defibrillation',
                       'cardiovascular|non-operative procedures|defibrillation'
                       ]
                      }
    # Iterate through features_group and create a new column for each group
    for group, features in features_group.items():
        # Add a new column where 1 indicates any of the features
        # in the group were presented and 0 otherwise
        treatment[group] = (treatment[features].sum(axis=1) > 0).astype(int)                            # noqa
        # Remove the individual features that are part of the group
        treatment.drop(features, axis=1, inplace=True)

    selected_features = ['patientunitstayid',
                         'Resusitation within 24hs']

    treatment = treatment.loc[:, selected_features]
    treatment = treatment.groupby("patientunitstayid").max().reset_index()                              # noqa
    assert treatment["patientunitstayid"].nunique() == treatment["patientunitstayid"].__len__()         # noqa

    return treatment


def eICU_dyslipidemia(root_dir, selected_patients):
    # Get the needed columns for faster processing
    usecols = ["patientunitstayid", "labresultoffset",
               "labname", "labresult"]
    # Load the data
    lab = pd.read_csv(root_dir + "lab.csv.gz", usecols=usecols)
    # Keep only the CS patients for faster processing
    lab = lab[lab["patientunitstayid"].isin(selected_patients)]
    # Remove the data after one Hour
    lab = lab[lab['labresultoffset'] < 0]              # noqa

    # Group the data by the median
    features_fat = ["triglycerides", "HDL", "LDL"]

    lab = lab[lab["labname"].isin(features_fat)]           # noqa

    for feature in features_fat:
        # Add a new column where 1 indicates the feature was
        # presented and 0 otherwise
        lab[feature] = (lab['labname'] == feature).astype(int)      # noqa

    # Get each value in a new columne
    for col in features_fat:
        lab[col] = lab[col] * lab["labresult"]

    # Unit harmonization
    lab["triglycerides"] = lab["triglycerides"] / 38.67

    # Group if several values are present for the same patient
    lab = lab.groupby("patientunitstayid").mean().reset_index()

    # Different criteria for Dyslipidemia
    var1 = lab["triglycerides"] > 1.7

    var2 = lab["HDL"] < 40

    var3 = lab["LDL"] > 190

    # If any of the criteria is presented, then diagnose
    lab["dyslipidemia"] = var1+var2+var3
    # Transform in 1 and 0
    lab["dyslipidemia"] = lab["dyslipidemia"].astype(int)
    # keep only the relevant information
    lab = lab.loc[:, ["patientunitstayid", "dyslipidemia"]]

    return lab


def eICU_mechanical_ventilation(root_dir):

    usecols = ["patientunitstayid", "ventstartoffset"]
    vent = pd.read_csv(root_dir + "respiratoryCare.csv.gz", usecols=usecols)

    # If the mechanical ventilation started before
    # admission or in the first 10 minutes set as 1
    vent["mechanical_ventilation"] = vent["ventstartoffset"] < 10
    vent.drop("ventstartoffset", axis=1, inplace=True)

    vent["mechanical_ventilation"] = vent["mechanical_ventilation"].astype(int)

    vent = vent.groupby("patientunitstayid").max().reset_index()
    return vent


def eICU_st_segmentation(root_dir):

    usecols = ["patientunitstayid", "diagnosisstring", "diagnosisoffset"]
    diagnosis = pd.read_csv(root_dir + "diagnosis.csv.gz", usecols=usecols)

    diagnosis = diagnosis[diagnosis['diagnosisstring'].str.contains("with ST elevation")]       # noqa
    # keep only the patients diagnosis before admission
    diagnosis["ST_elevation"] = diagnosis["diagnosisoffset"] < 0
    diagnosis["ST_elevation"] = diagnosis["ST_elevation"].astype(int)
    # Group the patients
    diagnosis = diagnosis.groupby("patientunitstayid").max().reset_index()
    # keep the important features
    ST_elevation = diagnosis.loc[:, ["patientunitstayid", "ST_elevation"]]

    return ST_elevation
