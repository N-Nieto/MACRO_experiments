# %%
import sys
import optuna
import timeit
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
optuna.logging.set_verbosity(optuna.logging.WARNING)

project_dir = Path().resolve().parents[1]
sys.path.append(str(project_dir / "code/"))
from lib.utils import ensure_dir    # noqa
from lib.data_load_utils import load_CULPRIT_data, get_data_from_features       # noqa
from lib.experiment_definitions import get_features, get_important_features     # noqa
from lib.data_processing import (                                               # noqa
    remove_low_variance_features,
    remove_random_features_fix_number,
)
from lib.ml_utils import (                                                      # noqa
    compute_results,
    get_inner_loop_optuna,
    results_to_df,
    save_best_model_params,
    estimator_to_df,
)

# %%
# Data is stored outside the project
data_dir = project_dir.parents[1] / "data" / "CULPRIT/"
save_dir = project_dir / "output" / "Admission_model/"
ensure_dir(save_dir)
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
    "eval_metric": "auc",
    "optuna_trials": 100,
    "random_state": random_state,
    "max_depth_min": 1,
    "max_depth_max": 10,
    "alpha_min": 1e-8,
    "alpha_max": 10,
    "lambda_min": 1e-8,
    "lambda_max": 100,
    "eta_min": 0.1,
    "eta_max": 1,
    "early_stopping_rounds": 100,
    "num_boost_round": 10000,
}
# %%
# number of thresholds used
ths_range = list(np.linspace(0, 1, 101))

# ########## Data load and pre-processing
# Get different features depending on the model
patient_info = load_CULPRIT_data(data_dir)

# Set target
# endpoint to use
endpoint_to_use = "fu_ce_death_le30d_yn"  # or "fu_ce_death_le365d_yn"
y = patient_info.loc[:, ["patient_ID", endpoint_to_use]]
Y = y.iloc[:, 1].to_numpy()

# Spot those patients that expired in the first 24hs
severity = patient_info["fu_ce_Death_d"] == 0
severity = severity.astype(int).to_numpy()

# Extract the Admission features
exp_name = "Admission"
features = get_features(exp_name)
X = get_data_from_features(patient_info, features)

# Remove low variance features
X = remove_low_variance_features(X, variance_ths)

# These features are not going to be removed in the missing value experiment
base_admission = ["had_dem_age_yr", "had_dem_male_yn"]

# Final data shape
n_participants, n_features = X.shape

# Show the feature distribution
print("Admission features: " + str(n_features))
print("Admission patients: " + str(n_participants))
# %%
kf_out = RepeatedStratifiedKFold(
    n_splits=out_n_splits, n_repeats=out_n_repetitions, random_state=random_state
)

kf_inner = StratifiedKFold(
    n_splits=inner_n_splits, shuffle=True, random_state=random_state
)

# Get features in the direct order of importance
direct_removal = get_important_features("Admission")

# Inverse order of importance
inverse_removal = direct_removal[::-1]

# Initialize variables
results_randomly = []
results_direct = []
results_inverse = []
results_training = []

results_estimators = []

predictions = []
y_true_loop = []
severity_loop = []

# Outer loop
print("Running experiment")
starttime = timeit.default_timer()
for i_fold, (train_index, test_index) in enumerate(kf_out.split(X, Y)):  # noqa
    print("FOLD: " + str(i_fold))

    # Patients used for train and internal XGB validation
    X_train_whole = X.iloc[train_index, :]
    Y_train_whole = Y[train_index]

    # Patients used to generete a prediction
    X_test = X.iloc[test_index, :]
    Y_test = Y[test_index]
    # Spot those patients that expired in the first 24 hs
    severity_test = severity[test_index]
    print("Fitting Admission model")
    # Train the model with all the features
    admission_model = get_inner_loop_optuna(
        X_train_whole, Y_train_whole, kf_inner, params_optuna
    )

    # Get probability
    pred_test = admission_model["model"].predict_proba(X_test)[:, 1]

    pred_train = admission_model["model"].predict_proba(X_train_whole)[:, 1]  # noqa
    # Compute test metrics
    results_training = compute_results(
        i_fold,
        "Admission Train",
        pred_train,
        Y_train_whole,
        results_training,
        ths_range=ths_range,
    )  # noqa

    # Compute train metrics
    results_estimators = save_best_model_params(
        i_fold, "Admission", admission_model, results_estimators
    )
    # Compute metrics
    predictions.append(pred_test)
    y_true_loop.append(Y_test)  # noqa
    severity_loop.append(severity_test)  # noqa

    # Compute metrics without removing any feature
    results_direct = compute_results(
        i_fold, "Admission", pred_test, Y_test, results_direct, ths_range=ths_range
    )  # noqa
    results_inverse = compute_results(
        i_fold, "Admission", pred_test, Y_test, results_inverse, ths_range=ths_range
    )  # noqa
    results_randomly = compute_results(
        i_fold, "Admission", pred_test, Y_test, results_randomly, ths_range=ths_range
    )  # noqa

    # Direct importance removal
    X_test_loop = X_test.copy()
    for i_features, feature_name in enumerate(direct_removal):
        # Remove feature from the test
        X_test_loop.loc[:, feature_name] = np.nan
        # Generate a prediction with the new test data (with one feature removed)
        pred_test = admission_model["model"].predict_proba(X_test_loop)[:, 1]  # noqa
        # Compute metrics
        results_direct = compute_results(
            i_fold,
            "Admission",
            pred_test,
            Y_test,
            results_direct,
            n_removed_features=i_features + 1,
            ths_range=ths_range,
        )

    # Inverse importance removal
    X_test_loop = X_test.copy()
    for i_features, feature_name in enumerate(inverse_removal):
        # Remove feature from the test
        X_test_loop.loc[:, feature_name] = np.nan
        # Generate a prediction with the new test data (with one feature removed)
        pred_test = admission_model["model"].predict_proba(X_test_loop)[:, 1]  # noqa
        # Compute metrics
        results_inverse = compute_results(
            i_fold,
            "Admission",
            pred_test,
            Y_test,
            results_inverse,
            n_removed_features=i_features + 1,
            ths_range=ths_range,
        )

    # Randomly removal with probability
    X_test_loop = X_test.copy()
    # Randomply remove
    for i_features, feature_name in enumerate(inverse_removal):
        # Remove data
        X_test_maked = remove_random_features_fix_number(
            X_test_loop,
            features_num=i_features + 1,
            basic_features=base_admission,
        )
        pred_test = admission_model["model"].predict_proba(X_test_maked)[:, 1]  # noqa
        # Compute metrics
        results_randomly = compute_results(
            i_fold,
            "Admission",
            pred_test,
            Y_test,
            results_randomly,
            n_removed_features=i_features + 1,
            ths_range=ths_range,
        )

# Create a dataframe to save
results_randomly_df = results_to_df(results_randomly)
results_direct_df = results_to_df(results_direct)
results_inverse_df = results_to_df(results_inverse)
results_training_df = results_to_df(results_training)

# Create a dataframe to the estimator
results_estimators = estimator_to_df(results_estimators)


# %%

# % Saving results
print("Saving Results")
results_randomly_df.to_csv(save_dir / "Admission_random_remove.csv")
results_direct_df.to_csv(save_dir / "Admission_direct_remove.csv")
results_inverse_df.to_csv(save_dir / "Admission_inverse_remove.csv")
results_training_df.to_csv(save_dir / "Admission_training.csv")

results_estimators.to_csv(save_dir / "Best_Admission_model_parameters.csv")

predictions = pd.DataFrame(predictions)
predictions = predictions.T
predictions.to_csv(save_dir / "Admission_predictions.csv")

y_true_loop = pd.DataFrame(y_true_loop)
y_true_loop = y_true_loop.T
y_true_loop.to_csv(save_dir / "Admission_y_true.csv")


severity_loop = pd.DataFrame(severity_loop)
severity_loop = severity_loop.T
severity_loop.to_csv(save_dir / "Admission_severity.csv")

# Create a experiment summary
experiment_setting = dict(
    data_dir=data_dir,
    save_dir=save_dir,
    random_state=random_state,
    out_n_splits=out_n_splits,
    endpoint_to_use=endpoint_to_use,
    out_n_repetitions=out_n_repetitions,
    inner_n_splits=inner_n_splits,
    variance_ths=variance_ths,
    ths_range=ths_range,
    params_optuna=params_optuna,
    n_participants=n_participants,
    n_features=n_features,
    experiment_timing=timeit.default_timer() - starttime,
)

# Save the experiment settings
experiment_setting_df = pd.DataFrame.from_dict(experiment_setting, orient="index")
experiment_setting_df.to_csv(
    save_dir / "experiment_settings_Full_model_strict_reg.csv"
)
print("Experiment finished")

# %%
