# %%
import pandas as pd
import numpy as np

from lib.data_load_utils import load_CULPRIT_data, get_data_from_features
from lib.experiment_definitions import get_features
from lib.data_processing import remove_low_variance_features
from lib.ml_utils import get_inner_loop_predictions_df, compute_results_by_fold_and_percentage  # noqa
from sklearn.model_selection import RepeatedStratifiedKFold

# %%
data_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/CULPRIT_data/202302_Jung/" # noqa

removal_oder = "inverse"     # direct or inverse to the shap importance

# Minimun feature variance
variance_ths = 0.10
# Set random state
random_state = 23
random_permutation = [False]
random_permutation_number = 1

# Cross validation parameters
out_n_splits = 10
out_n_repetitions = 10
# Inner CV
inner_n_splits = 10
inner_n_repetitions = 1

# Model parameters
initial_n_estimators = 1000
metric = "error"
# Validation percentage for XGBoost early stopping
val_percentage_xgb = 0.33
early_stopping_rounds = 250
reg_lambda = 0
reg_alpha = 0

# Model Threshold
thr = 0.5
# number of thresholds used
ths_range = list(np.linspace(0, 1, 101))

# Data load and pre-processing
# endpoint to use
endpoint_to_use = "fu_ce_death_le30d_yn"    # or "fu_ce_death_le365d_yn"

# Get different features depending on the model
# Get all data
patient_info = load_CULPRIT_data(data_dir)

# Removing patients that died in the first 24hs
patient_info = patient_info[patient_info["fu_ce_Death_d"] != 0]

# Set target
y = patient_info.loc[:, ["patient_ID", endpoint_to_use]]
Y = y.iloc[:, 1].to_numpy()

# Extract the Admmission features
exp_name = "24hs_v2"
feature_24h = get_features(exp_name)
X_24 = get_data_from_features(patient_info, feature_24h)

# Remove low variance features
X_24 = remove_low_variance_features(X_24, variance_ths)

# Final data shape
n_participants, n_features = X_24.shape

# Show the feature distribution
print("24hs features: " + str(n_features))
# %%

kf_out = RepeatedStratifiedKFold(n_splits=out_n_splits,
                                 n_repeats=out_n_repetitions,
                                 random_state=random_state)


kf_inner = RepeatedStratifiedKFold(n_splits=inner_n_splits,
                                   n_repeats=inner_n_repetitions,
                                   random_state=random_state)

params_24hs = {
    'initial_n_estimators': 1000,
    'val_percentage': val_percentage_xgb,
    'eval_metric': metric,
    'random_state': random_state,
    'early_stopping_rounds': early_stopping_rounds,
    'reg_alpha': reg_alpha,
    'reg_lambda': reg_lambda,
}

# Initialize variables
results_by_fold = []

# Inverse order of importance
feature_to_remove = ["creatine",
                     "icu_lab_lact16hpci_x",
                     "icu_lab_lact8hpci_x",
                     "pbnp",
                     "icu_lab_lact24hpci_x",
                     "admission_lactate",
                     "crp",
                     "hematocrit",
                     "glucose",
                     "white_cell_count",
                     "icu_lab_inr_r",
                     "tnt",
                     "hpe_proc_mechs_yn",
                     "alat",
                     "icu_lab_ck_x",
                     ]

if removal_oder == "direct":
    feature_to_remove = feature_to_remove
elif removal_oder == "inverse":
    feature_to_remove = feature_to_remove[::-1]
else:
    RuntimeError("Removal order not valid")

# %%

# Outer loop
for i_fold, (train_index, test_index) in enumerate(kf_out.split(X_24, Y)):       # noqa
    print("FOLD: " + str(i_fold))

    # Patients used for train and internal XGB validation
    X_train_whole_24 = X_24.iloc[train_index, :]
    Y_train_whole = Y[train_index]

    # Patients used to generete a prediction
    X_test_24 = X_24.iloc[test_index, :]
    Y_test = Y[test_index]

    print("Fitting 24hs model")
    # Train the model with all the features
    model_24hs = get_inner_loop_predictions_df(X_train_whole_24,
                                               Y_train_whole,
                                               kf_inner,
                                               params_24hs)
    pred_test_24hs = model_24hs["model"].predict_proba(X_test_24)[:, 1]           # noqa
    # Compute metrics without removing any feature
    results_by_fold = compute_results_by_fold_and_percentage(i_fold, "24hs", 0, pred_test_24hs, Y_test, thr, results_by_fold)                 # noqa

    X_test_24_loop = X_test_24.copy()
    for i_features, feature_name in enumerate(feature_to_remove):
        # Remove feature from the test
        X_test_24_loop.loc[:, feature_name] = np.nan
        # Generate a prediction with the new test data
        pred_test_24hs = model_24hs["model"].predict_proba(X_test_24_loop)[:, 1]           # noqa
        # Compute metrics
        results_by_fold = compute_results_by_fold_and_percentage(i_fold, "24hs", i_features, pred_test_24hs, Y_test, thr, results_by_fold)                 # noqa

# Put the results in a DF to save
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

# % Saving results
print("Saving Results")
save_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/output/missing_values/v2/"                       # noqa
results_df.to_csv(save_dir + "missing_values_10foldx10rep_24hs_vs_" + removal_oder + "_importance.csv")     # noqa
print("Experiment done!")
# %%
