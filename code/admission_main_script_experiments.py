# %%
import pandas as pd
import numpy as np
import random
from lib.data_load_utils import load_CULPRIT_data, get_data_from_features
from lib.data_processing import remove_low_variance_features
from lib.experiment_definitions import get_features
from lib.ml_utils import compute_results_by_fold, get_inner_loop_predictions_df
from sklearn.model_selection import RepeatedStratifiedKFold
from lib.ml_utils import compute_results_several_ths, save_best_model_params
# %%
# ##################### Parameters setting
data_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/CULPRIT_data/202302_Jung/" # noqa

# Minimun feature variance
variance_ths = 0.10
# Set random state
random_state = 23
random_permutation = [False, True]
random_permutation_number = 1

# Cross validation parameters
out_n_splits = 10
out_n_repetitions = 1
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

# ##################### Data Loading
# Get all data
patient_info = load_CULPRIT_data(data_dir)

# Set target
y = patient_info.loc[:, ["patient_ID", endpoint_to_use]]
Y = y.iloc[:, 1].to_numpy()

# Extract the Admmission features
exp_name = "Admission_v2"
feature_admission = get_features(exp_name)
X_admission = get_data_from_features(patient_info, feature_admission)

# Remove low variance features
X_admission = remove_low_variance_features(X_admission, variance_ths)

# Final data shape
n_participants = X_admission.shape[0]
admision_features = X_admission.shape[1]

# Show the feature distribution
print("Admission features: " + str(admision_features))
# %%
# ##################### Variables and objects declaration
kf_out = RepeatedStratifiedKFold(n_splits=out_n_splits,
                                 n_repeats=out_n_repetitions,
                                 random_state=random_state)


kf_inner = RepeatedStratifiedKFold(n_splits=inner_n_splits,
                                   n_repeats=inner_n_repetitions,
                                   random_state=random_state)

params_admission = {
    'initial_n_estimators': initial_n_estimators,
    'val_percentage': val_percentage_xgb,
    'eval_metric': metric,
    'random_state': random_state,
    'early_stopping_rounds': early_stopping_rounds,
    'reg_alpha': reg_alpha,
    'reg_lambda': reg_lambda,
}

# Initialize results variables
results_estimators = []
results_by_fold = []
results_by_ths = []

# ##################### Main loop
for rs in random_permutation:
    print("Randoms State: " + str(rs))
    for i_fold, (train_index, test_index) in enumerate(kf_out.split(X_admission, Y)):       # noqa
        print("FOLD: " + str(i_fold))
        if rs:
            rnd_permutation_loop = random_permutation_number
        else:
            rnd_permutation_loop = 1

        for rpn in range(rnd_permutation_loop):
            print("Random Permutation Nº: " + str(rpn))
            # Patients used for train and internal XGB validation
            X_train_whole_admission = X_admission.iloc[train_index, :]
            Y_train_whole = Y[train_index]

            if rs:
                random.shuffle(Y_train_whole)

            # Patients used to generete a prediction
            X_test_admission = X_admission.iloc[test_index, :]
            Y_test = Y[test_index]

            print("Fitting Admission model")
            admision_model = get_inner_loop_predictions_df(X_train_whole_admission,       # noqa
                                                           Y_train_whole,
                                                           kf_inner,
                                                           params_admission)

            # Get the admission test probabilities
            admission_test_pred = admision_model["model"].predict_proba(X_test_admission)[:, 1]   # noqa
            admission_train_pred = admision_model["model"].predict_proba(X_train_whole_admission)[:, 1]   # noqa
            # Compute test metrics
            results_by_fold = compute_results_by_fold(i_fold, "Admission", rs, rpn, admission_test_pred, Y_test, thr, results_by_fold)              # noqa
            # Compute train metrics
            results_by_fold = compute_results_by_fold(i_fold, "Admission_Train",rs, rpn, admission_train_pred, Y_train_whole, thr, results_by_fold)                                               # noqa
            results_estimators = save_best_model_params(i_fold, "Admission", admision_model, rs, rpn, results_estimators)                                 # noqa

            # compute the metrics with different thresholds
            # only when true labels are used
            if not rs:
                results_by_ths = compute_results_several_ths(i_fold, "Admission", admission_test_pred, Y_test, ths_range, results_by_ths)                                      # noqa  
                results_by_ths = compute_results_several_ths(i_fold, "Admission_train", admission_train_pred, Y_train_whole, ths_range, results_by_ths)                          # noqa                                    # noqa

# ##################### Acomodate results
# Put the final results in Dataframes
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
                                                               "Number of Estimators"])    # noqa    
# ##################### Savng results
print("Saving Results")
save_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/output/10x10/v2/"
results_df.to_csv(save_dir+ "metrics_10x10_true_and_random_labels_admission_v2.csv")   # noqa
results_estimators.to_csv(save_dir+ "estimators_10x10_true_and_random_labels_admission_v2.csv")   # noqa
results_ths_df.to_csv(save_dir + "threshold_10x10_all_models_high_admission_24_v2.csv")   # noqa
print("Experiment done")

# %%
