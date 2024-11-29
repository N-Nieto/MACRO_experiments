# %%
import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold    # noqa
from sklearn.linear_model import LogisticRegressionCV

# Append project path for using the functions in lib
project_root = os.path.dirname(os.path.dirname(os.path.dirname((__file__))))               # noqa
sys.path.append(project_root+"/code/")

from lib.unit_harmonization import lactate_unit_harmonization                       # noqa
from lib.data_load_utils import create_unique_patient_ID, create_admission_lactate  # noqa
from lib.ml_utils import compute_results, results_to_df       # noqa


# %%
save_dir = project_root+"/output/"
# Load main data
data = pd.read_excel(project_root + "/data/CULPRIT-data_20210407.xlsx",      # noqa
                     sheet_name=None)

# Extract patient data
patient_info = data["patient_data"]
# Set patient ID as unique
patient_info = create_unique_patient_ID(patient_info)

# Create admission lactate
patient_info = create_admission_lactate(patient_info)

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

# Harmonize units
data_final = lactate_unit_harmonization(data_final)
# %%
missing_values = data_final[['had_dem_age_yr', 'had_sy_ams_yn', 'p_mh_mi_yn',
                             'hpr_echo_lvef_pct',
                             'admission_lactate', 'ckdepi']].isnull().any(axis=1)       # noqa

# %% Create the score
# Initialize a Series to store the scores
scores = pd.Series(np.nan, index=data_final.index)

# Age > 75 years
scores[~missing_values] = (data_final['had_dem_age_yr'] > 75).astype(int)

# Confusion at presentation
scores[~missing_values] += data_final['had_sy_ams_yn']

# Previous MI or CABG
scores[~missing_values] += data_final['p_mh_mi_yn']

# ACS aetiology / All patients are AMI, that is more severe thatn ACS
scores += 1

# LV EF < 40%
scores[~missing_values] += (data_final['hpr_echo_lvef_pct'] < 40).astype(int)

# Blood lactate
scores[~missing_values] += (data_final['admission_lactate'] > 4) + (data_final['admission_lactate'] > 2)                    # noqa

# eGFR CKD-EPI
scores[~missing_values] += (data_final['ckdepi'] < 30) + ((data_final['ckdepi'] >= 30) & (data_final['ckdepi'] <= 60))      # noqa

# Assign the calculated scores to the 'CardShock_Score' column
data_final['CardShock_Score'] = scores

# %% Classification using the score
data_score = data_final.dropna(subset="CardShock_Score")
# Removing patients that died in the first 24hs
data_score = data_score[data_score["fu_ce_Death_d"] != 0]

X = data_score.loc[:, ["CardShock_Score"]]
y = data_score.loc[:, ["patient_ID", "fu_ce_death_le30d_yn"]]
Y = y.iloc[:, 1].to_numpy()
# Set random state
random_state = 23

# Cross validation parameters
out_n_splits = 10
out_n_repetitions = 10

inner_n_splits = 3
kf_out = RepeatedStratifiedKFold(n_splits=out_n_splits,
                                 n_repeats=out_n_repetitions,
                                 random_state=random_state)

kf_inner = StratifiedKFold(n_splits=inner_n_splits,
                           shuffle=True,
                           random_state=random_state)
results_by_fold = []
score_clf = LogisticRegressionCV(cv=kf_inner)

# Outer loop

for i_fold, (train_index, test_index) in enumerate(kf_out.split(X, Y)):       # noqa
    print("FOLD: " + str(i_fold))

    # Patients used for train and internal XGB validation
    X_train_whole = X.iloc[train_index, :]
    Y_train_whole = Y[train_index]

    # Patients used to generete a prediction
    X_test = X.iloc[test_index, :]
    Y_test = Y[test_index]

    # impute train data, round for matching with the original distribution
    X_train_whole_imputed = X_train_whole
    # impute test data, round for matching with the original distribution
    X_test_imputed = X_test

    score_clf.fit(X=X_train_whole_imputed, y=Y_train_whole)

    imputed_train_proba = score_clf.predict_proba(X=X_train_whole_imputed)[:, 1]                    # noqa
    imputed_test_proba = score_clf.predict_proba(X=X_test_imputed)[:, 1]                    # noqa

    results_by_fold = compute_results(i_fold, "CardShock_Score_test", imputed_test_proba, Y_test, results_by_fold)                                           # noqa
    results_by_fold = compute_results(i_fold, "CardShock_Score_train", imputed_train_proba, Y_train_whole, results_by_fold)                                           # noqa

results_pt = results_to_df(results_by_fold)
# %%

# % Savng results
print("Saving Results")
results_pt.to_csv(save_dir+ "CardShock.csv")              # noqa

# %%
