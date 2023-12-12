# %%
import pandas as pd
import numpy as np
import random
from lib.data_load_utils import load_CULPRIT_data, get_data_from_features
from lib.experiment_definitions import get_features
from lib.data_processing import remove_low_variance_features
from lib.ml_utils import compute_results_by_fold, get_inner_loop_optuna
from lib.ml_utils import save_best_model_params, compute_results_several_ths            # noqa
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
# %%
data_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/CULPRIT_data/202302_Jung/" # noqa

# Minimun feature variance
variance_ths = 0.10
# Set random state
random_state = 23
random_permutation = [False]
random_permutation_number = 1


# Cross validation parameters
out_n_splits = 10
out_n_repetitions = 1

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
exp_name = "Admission_v2"
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


results_by_fold = []
results_estimators = []
results_by_ths = []
model_cv_preds = np.ones(n_participants) * -1
predictions_model = []
y_true_fold = []
# Outer loop
for rs in random_permutation:
    print("Randoms State: " + str(rs))
    for i_fold, (train_index, test_index) in enumerate(kf_out.split(X_24, Y)):       # noqa
        print("FOLD: " + str(i_fold))
        if rs:
            rnd_permutation_loop = random_permutation_number
        else:
            rnd_permutation_loop = 1

        for rpn in range(rnd_permutation_loop):
            print("Random Permutation Nº: " + str(rpn))
            # Patients used for train and internal XGB validation
            X_train_whole_24 = X_24.iloc[train_index, :]
            Y_train_whole = Y[train_index]

            if rs:
                random.shuffle(Y_train_whole)

            # Patients used to generete a prediction
            X_test_24 = X_24.iloc[test_index, :]
            Y_test = Y[test_index]

            print("Fitting 24hs model")
            model_24hs = get_inner_loop_optuna(X_train_whole_24,
                                               Y_train_whole,
                                               kf_inner,
                                               )
            # Store 24hs predictions
            pred_test_24hs = model_24hs["model"].predict_proba(X_test_24)[:, 1] # noqa
            pred_train_24hs = model_24hs["model"].predict_proba(X_train_whole_24)[:, 1]     # noqa                
            # Compute test metrics
            results_by_fold = compute_results_by_fold(i_fold,
                                                      "24hs",
                                                      rs,
                                                      rpn,
                                                      pred_test_24hs,
                                                      Y_test,
                                                      thr,
                                                      results_by_fold)                                                        # noqa
            # Compute train metrics
            results_by_fold = compute_results_by_fold(i_fold, "24hs_Train", rs, rpn, pred_train_24hs, Y_train_whole, thr, results_by_fold)                                                        # noqa

            results_estimators = save_best_model_params(i_fold, "24hs", model_24hs, rs, rpn, results_estimators)                                          # noqa

            if not rs:
                results_by_ths = compute_results_several_ths(i_fold, "24hs", pred_test_24hs, Y_test, ths_range, results_by_ths)                                                # noqa
                results_by_ths = compute_results_several_ths(i_fold, "24hs_train", pred_train_24hs, Y_train_whole, ths_range, results_by_ths)                                                # noqa

            # Store predictions
            predictions_model.append(pred_test_24hs)
            y_true_fold.append(Y_test)                                                        # noqa


results_df = pd.DataFrame(results_by_fold, columns=["Fold",
                                                    "Model",
                                                    "Random State",
                                                    "Random Permutation",
                                                    "Balanced ACC",
                                                    "AUC",
                                                    "F1",
                                                    "Specificity",
                                                    "Sensitivity"])

results_ths_df = pd.DataFrame(results_by_ths, columns=["Fold",
                                                       "Model",
                                                       "Threshold",
                                                       "Balanced ACC",
                                                       "AUC",
                                                       "F1",
                                                       "Specificity",
                                                       "Sensitivity"])

results_estimators = pd.DataFrame(results_estimators, columns=["Fold",
                                                               "Model",
                                                               "Random State",            # noqa
                                                               "Random Permutation",       # noqa
                                                               "Number of Estimators",      # noqa
                                                               "alpha",
                                                               "lambda",
                                                               "eta",
                                                               "max_depth"])


# # % Savng results
# print("Saving Results")
# save_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/output/optuna/10x10/"    # noqa
# results_df.to_csv(save_dir+ "metrics_10x10_true_and_random_labels_24hs_v2_strong.csv")   # noqa
# results_estimators.to_csv(save_dir+ "estimators_10x10_true_and_random_labels_24hs_strong.csv")   # noqa
# results_ths_df.to_csv(save_dir + "threshold_10x10_all_models_high_24hs_v2_strong.csv")   # noqa

# print("Saving Predictions")

# save_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/output/optuna/predictions/"               # noqa

# results_24hs = pd.DataFrame(predictions_model)
# results_24hs = results_24hs.T
# results_24hs.to_csv(save_dir+ "24hs_v2_predictions_for_AUC_strong.csv")   # noqa

# y_true_fold_24hs_df = pd.DataFrame(y_true_fold)
# y_true_fold_24hs_df = y_true_fold_24hs_df.T
# y_true_fold_24hs_df.to_csv(save_dir+ "24hs_v2_true_predictions_for_AUC_strong.csv")   # noqa

# print("Experiment done")

# %%
save_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/output/optuna/calibration/"               # noqa
results_24hs = pd.DataFrame(predictions_model)
results_24hs = results_24hs.T
results_24hs.to_csv(save_dir+ "Admission_v2_predictions_for_AUC.csv")   # noqa

y_true_fold_24hs_df = pd.DataFrame(y_true_fold)
y_true_fold_24hs_df = y_true_fold_24hs_df.T
y_true_fold_24hs_df.to_csv(save_dir+ "Admission_v2_true_for_AUC.csv")   # noqa
# %%
