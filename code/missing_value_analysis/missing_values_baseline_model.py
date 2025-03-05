
# %%
import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
import optuna

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # noqa: E501 - Suppress long line length warning
sys.path.append(os.path.join(project_root, "code"))

from lib.data_load_utils import load_CULPRIT_data, get_data_from_features   # noqa
from lib.experiment_definitions import get_features                         # noqa
from lib.data_processing import remove_low_variance_features                # noqa
from lib.ml_utils import compute_results, get_inner_loop_optuna             # noqa
from lib.ml_utils import results_to_df                                      # noqa

optuna.logging.set_verbosity(optuna.logging.WARNING)
# %%
data_dir = "/data/CULPRIT/"
save_dir = "/output/missing_values/"
# Minimun feature variance
variance_ths = 0.10
# Set random state
random_state = 23

# Cross validation parameters
out_n_splits = 10
out_n_repetitions = 10

# Inner CV
inner_n_splits = 3

# Define the hyperparameters to tune
params_optuna = {
    "objective": "binary:logistic",
    'eval_metric': 'auc',
    'optuna_trials': 100,
    'random_state': random_state,
    'max_depth_min': 1,
    'max_depth_max': 5,
    'alpha_min': 1e-8,
    'alpha_max': 10,
    'lambda_min': 1e-8,
    'lambda_max': 100,
    'eta_min': 0.1,
    'eta_max': 1,
    "early_stopping_rounds": 100,
    "num_boost_round": 10000,
}

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
exp_name = "Admission"
feature_admission = get_features(exp_name)
X_Admission = get_data_from_features(patient_info, feature_admission)
# Remove low variance features
X = remove_low_variance_features(X_Admission, variance_ths)
feature_admission = X_Admission.columns


X = np.isnan(X).astype(int)

# Final data shape
n_participants, n_features = X.shape

# Show the feature distribution
print("24hs features: " + str(n_features))
# %%
kf_out = RepeatedStratifiedKFold(n_splits=out_n_splits,
                                 n_repeats=out_n_repetitions,
                                 random_state=random_state)

kf_inner = StratifiedKFold(n_splits=inner_n_splits,
                           shuffle=True,
                           random_state=random_state)


# Initialize variables
results_direct = []
results_training = []
predictions_full = []
y_true_loop = []

for i_fold, (train_index, test_index) in enumerate(kf_out.split(X, Y)):       # noqa
    print("FOLD: " + str(i_fold))

    # Patients used for train and internal XGB validation
    X_train_whole = X.iloc[train_index, :]
    Y_train_whole = Y[train_index]

    # Patients used to generete a prediction
    X_test = X.iloc[test_index, :]
    Y_test = Y[test_index]

    print("Fitting Full model")
    # Train the model with all the features
    full_model = get_inner_loop_optuna(X_train_whole,
                                       Y_train_whole,
                                       kf_inner,
                                       params_optuna)

    # Get probability
    pred_test = full_model["model"].predict_proba(X_test)[:, 1]

    pred_train = full_model["model"].predict_proba(X_train_whole)[:, 1]     # noqa                
    # Compute test metrics
    results_training = compute_results(i_fold, "Full on missing values Train",
                                       pred_train,
                                       Y_train_whole, results_training)

    # Compute metrics
    predictions_full.append(pred_test)
    y_true_loop.append(Y_test)                                                        # noqa

    # Compute metrics without removing any feature
    results_direct = compute_results(i_fold, "Full on missing values", pred_test, Y_test, results_direct)                 # noqa

# Create a dataframe to save
results_direct_df = results_to_df(results_direct)

# %%

# % Saving results
print("Saving Results")
save_dir = "output/missing_values/"

results_direct_df.to_csv(save_dir+ "Admission_base_missing.csv")                    # noqa

predictions_full = pd.DataFrame(predictions_full)
predictions_full = predictions_full.T
predictions_full.to_csv(save_dir+ "predictions_Admissionl_base_missing.csv")   # noqa

y_true_loop = pd.DataFrame(y_true_loop)
y_true_loop = y_true_loop.T
y_true_loop.to_csv(save_dir+ "y_true_Admission_base_missing.csv")   # noqa
