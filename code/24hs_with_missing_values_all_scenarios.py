
# %%
import pandas as pd
import numpy as np
from lib.data_load_utils import load_CULPRIT_data, get_data_from_features
from lib.experiment_definitions import get_features
from lib.data_processing import remove_low_variance_features, randomly_replace_with_nan     # noqa
from lib.data_processing import remove_random_features_fix_number
from lib.ml_utils import compute_results_by_fold_and_percentage, get_inner_loop_optuna      # noqa
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
# %%
data_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/CULPRIT_data/202302_Jung/" # noqa

# Minimun feature variance
variance_ths = 0.10
# Set random state
random_state = 23
random_rounds = 10


# Cross validation parameters
out_n_splits = 10
out_n_repetitions = 10

# Inner CV
inner_n_splits = 3

# Model parameters
num_boost_round = 10000
metric = "auc"
# Validation percentage for XGBoost early stopping
early_stopping_rounds = 100


# Model Threshold
thr = 0.5
# number of thresholds used
ths_range = list(np.linspace(0, 1, 101))

# Data load and pre-processing

# Get different features depending on the model
# Get all data
patient_info = load_CULPRIT_data(data_dir)

# Removing patients that died in the first 24hs
patient_info = patient_info[patient_info["fu_ce_Death_d"] != 0]

# Set target
# endpoint to use
endpoint_to_use = "fu_ce_death_le30d_yn"    # or "fu_ce_death_le365d_yn"
y = patient_info.loc[:, ["patient_ID", endpoint_to_use]]
Y = y.iloc[:, 1].to_numpy()

# Extract the 24hs features
exp_name = "Admission_v2"
feature_admission = get_features(exp_name)
X_Admission = get_data_from_features(patient_info, feature_admission)
# Remove low variance features
X_Admission = remove_low_variance_features(X_Admission, variance_ths)
feature_admission = X_Admission.columns

# Extract the 24hs features
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

kf_inner = StratifiedKFold(n_splits=inner_n_splits,
                           shuffle=True,
                           random_state=random_state)

porcentages = list(np.linspace(0, 1, 101))


# Inverse order of importance
direct_removal = ["creatine",
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
                  "icu_lab_ck_x"
                  ]

inverse_removal = direct_removal[::-1]

results_randomly = []
results_direct = []
results_inverse = []

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
    model_24hs = get_inner_loop_optuna(X_train_whole_24,
                                       Y_train_whole,
                                       kf_inner)

    # Store the prediction without removing any feature
    pred_test_24hs = model_24hs["model"].predict_proba(X_test_24)[:, 1]           # noqa
    results_direct = compute_results_by_fold_and_percentage(i_fold, "24hs", 0, pred_test_24hs, Y_test, thr, results_direct)                 # noqa
    results_inverse = compute_results_by_fold_and_percentage(i_fold, "24hs", 0, pred_test_24hs, Y_test, thr, results_inverse)                 # noqa
    results_randomly = compute_results_by_fold_and_percentage(i_fold, "24hs", 0, pred_test_24hs, Y_test, thr, results_randomly)                 # noqa

    # Direct importance removal
    X_test_24_loop = X_test_24.copy()
    for i_features, feature_name in enumerate(direct_removal):
        # Remove feature from the test
        X_test_24_loop.loc[:, feature_name] = np.nan
        # Generate a prediction with the new test data
        pred_test_24hs = model_24hs["model"].predict_proba(X_test_24_loop)[:, 1]           # noqa
        # Compute metrics
        results_direct = compute_results_by_fold_and_percentage(i_fold, "24hs", i_features, pred_test_24hs, Y_test, thr, results_direct)                 # noqa

    # Inverse importance removal
    X_test_24_loop = X_test_24.copy()
    for i_features, feature_name in enumerate(inverse_removal):
        # Remove feature from the test
        X_test_24_loop.loc[:, feature_name] = np.nan
        # Generate a prediction with the new test data
        pred_test_24hs = model_24hs["model"].predict_proba(X_test_24_loop)[:, 1]           # noqa
        # Compute metrics
        results_inverse = compute_results_by_fold_and_percentage(i_fold, "24hs", i_features, pred_test_24hs, Y_test, thr, results_inverse)                 # noqa

    # Randomly removal with probability
    X_test_24_loop = X_test_24.copy()
    for porcent in porcentages:
        # Randomply remove
        for n in range(random_rounds):
            # Store 24hs predictions
            X_test_24_maked = randomly_replace_with_nan(X_test_24_loop,
                                                        prob=porcent,
                                                        basic_features=feature_admission)         # noqa
            pred_test_24hs = model_24hs["model"].predict_proba(X_test_24_maked)[:, 1]           # noqa
            # Compute metrics
            results_randomly = compute_results_by_fold_and_percentage(i_fold, "24hs", porcent, pred_test_24hs, Y_test, thr, results_randomly)                 # noqa

    # Randomly removal with probability
    X_test_24_loop = X_test_24.copy()
    # Randomply remove
    for i_features, feature_name in enumerate(inverse_removal):
        # Store 24hs predictions
        X_test_24_maked = remove_random_features_fix_number(X_test_24_loop,
                                                            features_num=i_features,                   # noqa
                                                            basic_features=feature_admission)         # noqa
        pred_test_24hs = model_24hs["model"].predict_proba(X_test_24_maked)[:, 1]                     # noqa
        # Compute metrics
        results_randomly_fix = compute_results_by_fold_and_percentage(i_fold, "24hs", porcent, pred_test_24hs, Y_test, thr, results_randomly)                 # noqa


results_randomly_df = pd.DataFrame(results_randomly, columns=["Fold",
                                                              "Model",
                                                              "Percenage_nan",
                                                              "Balanced ACC",
                                                              "AUC",
                                                              "F1",
                                                              "Specificity",
                                                              "Sensitivity",
                                                              "Precision",
                                                              "Recall"])

results_randomly_fix_df = pd.DataFrame(results_randomly_fix,
                                       columns=["Fold",
                                                "Model",
                                                "Percenage_nan",
                                                "Balanced ACC",
                                                "AUC",
                                                "F1",
                                                "Specificity",
                                                "Sensitivity",
                                                "Precision",
                                                "Recall"])

results_direct_df = pd.DataFrame(results_direct, columns=["Fold",
                                                          "Model",
                                                          "Percenage_nan",
                                                          "Balanced ACC",
                                                          "AUC",
                                                          "F1",
                                                          "Specificity",
                                                          "Sensitivity",
                                                          "Precision",
                                                          "Recall"])

results_inverse_df = pd.DataFrame(results_inverse, columns=["Fold",
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
save_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/output/optuna/missing_values/"       # noqa
# results_randomly_df.to_csv(save_dir+ "missing_values_10foldx10repx10_shuffle.csv")              # noqa
results_randomly_fix_df.to_csv(save_dir+ "missing_values_10foldx10repx10_random_fix.csv")              # noqa

# results_direct_df.to_csv(save_dir+ "missing_values_10foldx10rep_direct.csv")                    # noqa
# results_inverse_df.to_csv(save_dir+ "missing_values_10foldx10rep_inverse.csv")                  # noqa

# %%
