# %%
import pandas as pd                     # noqa
import numpy as np

from CULPRIT_project.code.lib.ml_utils import load_CULPRIT_data, get_data_from_features
from CULPRIT_project.code.lib.ml_utils import compute_results_by_fold_and_percentage
from CULPRIT_project.code.lib.ml_utils import get_features, get_inner_loop_predictions_df
from sklearn.model_selection import RepeatedStratifiedKFold

# %%
data_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/CULPRIT_data/202302_Jung/" # noqa

# endpoint to use
endpoint_to_use = "fu_ce_death_le30d_yn"    # or "fu_ce_death_le365d_yn"

# Get different features depending on the model
# Get all data
patient_info, lab_info, clip_info = load_CULPRIT_data(data_dir)
# Set target
y = patient_info.loc[:, ["patient_ID", endpoint_to_use]]

# Extract the Admmission features
exp_name = "Admission"
admission_basic, lab_features, clip_features = get_features(exp_name)
X_final_admission, features_admission = get_data_from_features(patient_info,
                                                               lab_info,
                                                               clip_info,
                                                               admission_basic,        # noqa
                                                               lab_features,
                                                               clip_features)
X_admission = X_final_admission.drop(columns="patient_ID")

# Show the feature distribution
print("Admission features: " + str(X_admission.columns.nunique()))
Y = y.iloc[:, 1].to_numpy()
# %%

# Set random state
random_state = 23
# Validation percentage for XGBoost early stopping
val_percentage_xgb = 0.33

out_n_splits = 10
out_n_repetitions = 10
kf_out = RepeatedStratifiedKFold(n_splits=out_n_splits,
                                 n_repeats=out_n_repetitions,
                                 random_state=random_state)

inner_n_splits = 10
inner_n_repetitions = 1
kf_inner = RepeatedStratifiedKFold(n_splits=inner_n_splits,
                                   n_repeats=inner_n_repetitions,
                                   random_state=random_state)
metric = "error"
early_stopping_rounds = 250
reg_lambda = 0
reg_alpha = 0
params_admission = {
    'initial_n_estimators': 1000,
    'val_percentage': val_percentage_xgb,
    'eval_metric': metric,
    'random_state': random_state,
    'early_stopping_rounds': early_stopping_rounds,
    'reg_alpha': reg_alpha,
    'reg_lambda': reg_lambda,
}

thr = 0.5
n_participants = X_admission.shape[0]
n_features_24 = X_admission.shape[1]
# Initialize variables
model_24hs_cv_preds = np.ones(n_participants) * -1
random_rounds = 10
results_by_fold = []

feature_to_remove = ["hpr_ecg_afib_y",
                     "p_mh_pad_yn",
                     "combined_variable",
                     "had_base_cpr24h_yn",
                     "had_sy_extremity_yn",
                     #  "had_dem_male_yn",
                     "p_rf_aht_yn",
                     "hpr_ecg_stemi_yn",
                     "p_rf_smoker_yn",
                     "had_base_mechvent_yn",
                     "p_rf_dm_yn",
                     "hpr_ecg_sinrhy_y",
                     "had_sy_ams_yn",
                     "had_pex_weight_kg",
                     "hpr_hmdyn_sbp_mmhg",
                     "had_pex_height_cm",
                     "hpr_hmdyn_dbp_mmhg",
                     "p_rf_dyslip_yn",
                     "hpr_hmdyn_hr_bpm",
                     # "had_dem_age_yr"
                     ]
feature_to_remove = feature_to_remove[::-1]

# Outer loop
for i_fold, (train_index, test_index) in enumerate(kf_out.split(X_admission, Y)):       # noqa
    print("FOLD: " + str(i_fold))

    # Patients used for train and internal XGB validation
    Y_train_whole = Y[train_index]
    X_train_whole_admission = X_admission.iloc[train_index, :]

    # Patients used to generete a prediction
    Y_test = Y[test_index]
    X_test_admission = X_admission.iloc[test_index, :]

    print("Fitting Admission model")
    admision_model = get_inner_loop_predictions_df(X_train_whole_admission,
                                                   Y_train_whole,
                                                   kf_inner,
                                                   params_admission)

    pred_test_admission = admision_model["model"].predict_proba(X_test_admission)[:, 1]           # noqa

    results_by_fold = compute_results_by_fold_and_percentage(i_fold, "Admission", "None", pred_test_admission, Y_test, thr, results_by_fold)                 # noqa

    for feature_name in feature_to_remove:
        X_test_admission[feature_name] = np.nan

        pred_test_admission = admision_model["model"].predict_proba(X_test_admission)[:, 1]           # noqa
        # Compute metrics
        results_by_fold = compute_results_by_fold_and_percentage(i_fold, "Admission", feature_name, pred_test_admission, Y_test, thr, results_by_fold)                 # noqa

results_df = pd.DataFrame(results_by_fold, columns=["Fold",
                                                    "Model",
                                                    "Percenage_nan",
                                                    "Balanced ACC",
                                                    "AUC",
                                                    "F1",
                                                    "Specificity",
                                                    "Sensitivity",
                                                    "Precision",
                                                    "Recall"])

# %%

# % Savng results
print("Saving Results")
save_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/output/10x10/"
results_df.to_csv(save_dir+ "admission_missing_values_10foldx10rep_importace_inverse.csv")   # noqa
print("Experiment done!")

# %%
