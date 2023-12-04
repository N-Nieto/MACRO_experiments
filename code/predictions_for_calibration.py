# %%
import numpy as np
import pickle
from CULPRIT_project.code.lib.ml_utils import load_CULPRIT_data, get_data_from_features
from CULPRIT_project.code.lib.ml_utils import get_combine_model
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
patient_features, lab_features, clip_features = get_features(exp_name)
X_final_admission, features_admission = get_data_from_features(patient_info,
                                                               lab_info,
                                                               clip_info,
                                                               patient_features,        # noqa
                                                               lab_features,
                                                               clip_features)

# Extract all the 24hs available features
exp_name = "24hs"
patient_features, lab_features, clip_features = get_features(exp_name)

X_final_24, features_24 = get_data_from_features(patient_info, lab_info,
                                                 clip_info,
                                                 patient_features,
                                                 lab_features,
                                                 clip_features)


X_24 = X_final_24.drop(columns="patient_ID")
X_admission = X_final_admission.drop(columns="patient_ID")

# Show the feature distribution
print("Admission features: " + str(X_admission.columns.nunique()))
print("24hs features: " + str(X_24.columns.nunique()))
Y = y.iloc[:, 1].to_numpy()
# %%

# Set random state
random_state = 23
# Validation percentage for XGBoost early stopping
val_percentage_xgb = 0.33

out_n_splits = 10
out_n_repetitions = 1
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

early_stopping_rounds = 250
reg_lambda = 0
reg_alpha = 0
params_24hs = {
    'initial_n_estimators': 1000,
    'val_percentage': val_percentage_xgb,
    'eval_metric': metric,
    'random_state': random_state,
    'early_stopping_rounds': early_stopping_rounds,
    'reg_alpha': reg_alpha,
    'reg_lambda': reg_lambda,
}

early_stopping_rounds = 250
reg_lambda = 0
reg_alpha = 0
params_cascade = {
    'initial_n_estimators': 1000,
    'val_percentage': val_percentage_xgb,
    'eval_metric': metric,
    'random_state': random_state,
    'early_stopping_rounds': early_stopping_rounds,
    'reg_alpha': reg_alpha,
    'reg_lambda': reg_lambda,
}

# prediction threshold
thr = 0.5

n_participants = X_admission.shape[0]


# Initialize variables
model_admission_cv_preds = np.ones(n_participants) * -1
model_24hs_cv_preds = np.ones(n_participants) * -1
cascade_norisk_model_cv_preds = np.ones(n_participants) * -1
cascade_risk_model_cv_preds = np.ones(n_participants) * -1

# Outer loop
for i_fold, (train_index, test_index) in enumerate(kf_out.split(X_admission, Y)):       # noqa
    print("FOLD: " + str(i_fold))
    # Patients used for train and internal XGB validation
    X_train_whole_admission = X_admission.iloc[train_index, :]
    X_train_whole_24 = X_24.iloc[train_index, :]
    Y_train_whole = Y[train_index]

    # Patients used to generete a prediction
    X_test_admission = X_admission.iloc[test_index, :]
    X_test_24 = X_24.iloc[test_index, :]

    Y_test = Y[test_index]

    print("Fitting Admission model")
    admision_model = get_inner_loop_predictions_df(X_train_whole_admission,       # noqa
                                                   Y_train_whole,
                                                   kf_inner,
                                                   params_admission)
    print("Fitting 24hs model")
    model_24hs = get_inner_loop_predictions_df(X_train_whole_24,
                                               Y_train_whole,
                                               kf_inner,
                                               params_24hs)

    # Train cascade model
    # Get the admission predictions for features of the cascade
    predictions_admission = (admision_model["pred"] > thr).astype(int)
    # get the 24hs as target of the cascade model
    predictions_24model = (model_24hs["pred"] > thr).astype(int)

    # Concatenate the admission predictions and the admission data
    # for train the cascade model
    X_train_cascade = X_train_whole_admission.copy()
    X_train_cascade["Admission_prob"] = admision_model["pred"]

    # Filter those patients the Admission model predicted as No risk
    mask = predictions_admission == 0
    X_train_cascade_norisk = X_train_cascade[mask]
    y_train_cascade_norisk = predictions_24model[mask]

    print("Fitting Cascade Model for No Risk")
    # Get the predictions of the Admission model trained
    # with early stopping and retrained over all the train data
    model_cascade_norisk = get_inner_loop_predictions_df(X_train_cascade_norisk,    # noqa
                                                         y_train_cascade_norisk,    # noqa
                                                         kf_inner,
                                                         params_cascade)            # noqa

    # Filter those patients that the Admission model predicted as Risk
    mask = predictions_admission == 1
    X_train_cascade_risk = X_train_cascade[mask]
    y_train_cascade_risk = predictions_24model[mask]

    print("Fitting Cascade Model for Risk")
    # Get the predictions of the Admission model trained
    # with early stopping and retrained over all the train data
    model_cascade_risk = get_inner_loop_predictions_df(X_train_cascade_risk,        # noqa
                                                       y_train_cascade_risk,        # noqa
                                                       kf_inner,
                                                       params_cascade)

    # Get the admission test probabilities
    admission_test_pred = admision_model["model"].predict_proba(X_test_admission)[:, 1]   # noqa
    # Save the admission predictions with the test index
    model_admission_cv_preds[test_index] = admission_test_pred

    # Store 24hs predictions
    pred_test_24hs = model_24hs["model"].predict_proba(X_test_24)[:, 1]
    model_24hs_cv_preds[test_index] = pred_test_24hs  # noqa

    # Test cascade model
    # Get the admission predictions to separate the patiens
    # in the cascade model in the get_combined_model function
    test_predictions_admission = (admission_test_pred > thr).astype(int)                    # noqa

    # Concatenate the admission pred and the admission data
    # for train the cascade model
    X_test_cascade = X_test_admission.copy()
    X_test_cascade["Admission_prob"] = admission_test_pred

    # Store cascade predictions
    cascade_norisk_pred = model_cascade_norisk["model"].predict_proba(X_test_cascade)[:, 1] # noqa

    # Store cascade predictions
    cascade_risk_pred = model_cascade_risk["model"].predict_proba(X_test_cascade)[:, 1]     # noqa

    combined_proba = get_combine_model(admission_test_pred,
                                       cascade_norisk_pred,
                                       cascade_risk_pred,
                                       thr=thr)

    Combined_only_risk_cascade_proba = get_combine_model(admission_test_pred,                # noqa
                                                         None,
                                                         cascade_risk_pred,                  # noqa
                                                         thr=thr)

    Combined_only_norisk_cascade_proba = get_combine_model(admission_test_pred,              # noqa
                                                           cascade_norisk_pred,              # noqa
                                                           None,
                                                           thr=thr)
    cascade_norisk_model_cv_preds[test_index] = Combined_only_norisk_cascade_proba           # noqa   
    cascade_risk_model_cv_preds[test_index] = Combined_only_risk_cascade_proba

# %%

# %% Saving
save_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/output/calibration/"

save_list = [[model_admission_cv_preds, "admission_predictions_cv"],
             [model_24hs_cv_preds, "24hs_predictions_cv"],
             [cascade_norisk_model_cv_preds, "cascade_norisk_predictions_cv"],
             [cascade_risk_model_cv_preds, "cascade_risk_predictions_cv"],
             [Y, "true_labels"]
             ]

for list_to_save, save_name in save_list:
    with open(save_dir+save_name, "wb") as fp:   # Pickling
        pickle.dump(list_to_save, fp)
