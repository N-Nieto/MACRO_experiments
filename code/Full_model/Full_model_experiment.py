# %%
import sys
import timeit
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

project_dir = Path.cwd().parents[1]
sys.path.append(str(project_dir / "code/"))
from lib.data_load_utils import get_data_from_features, load_CULPRIT_data
from lib.data_processing import (
    remove_low_variance_features,
    remove_random_features_fix_number,
)
from lib.experiment_definitions import get_features, get_important_features
from lib.ml_utils import (
    compute_results,
    estimator_to_df,
    get_inner_loop_optuna,
    results_to_df,
    save_best_model_params,
)
from lib.utils import ensure_dir

optuna.logging.set_verbosity(optuna.logging.WARNING)
# %%
# Data is stored outside the project
data_dir = project_dir.parents[1] / "Data MODS" / "data" / "CULPRIT/"
save_dir = project_dir / "output" / "Full_model/"
ensure_dir(save_dir)
# Fitted model of every outer fold (XGBoost JSON), for post-hoc analyses
models_dir = save_dir / "fold_models"
ensure_dir(models_dir)


# Minimun feature variance
variance_ths = 0.10
# Set random state
random_state = 23
# Random generator for the random feature-removal scenario
rng = np.random.default_rng(random_state)

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
    "max_depth_max": 5,
    "alpha_min": 1e-8,
    "alpha_max": 10,
    "lambda_min": 1e-8,
    "lambda_max": 100,
    "eta_min": 0.1,
    "eta_max": 1,
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
endpoint_to_use = "fu_ce_death_le30d_yn"  # or "fu_ce_death_le365d_yn"
y = patient_info.loc[:, ["patient_ID", endpoint_to_use]]
Y = y.iloc[:, 1].to_numpy()
patient_ids = y["patient_ID"].to_numpy()

# Extract the 24hs features
exp_name = "Admission"
feature_admission = get_features(exp_name)
X_Admission = get_data_from_features(patient_info, feature_admission)
# Remove low variance features
X_Admission = remove_low_variance_features(X_Admission, variance_ths)
feature_admission = X_Admission.columns

# Extract the 24hs features
exp_name = "24hs"
feature_24h = get_features(exp_name)
X = get_data_from_features(patient_info, feature_24h)

# Remove low variance features
X = remove_low_variance_features(X, variance_ths)

# Final data shape
n_participants, n_features = X.shape

# Show the feature distribution
print("24hs features: " + str(n_features))
# %%
kf_out = RepeatedStratifiedKFold(
    n_splits=out_n_splits, n_repeats=out_n_repetitions, random_state=random_state
)

kf_inner = StratifiedKFold(
    n_splits=inner_n_splits, shuffle=True, random_state=random_state
)

# Get features in the direct order of importance
direct_removal = get_important_features("Full")

# Inverse order of importance
inverse_removal = direct_removal[::-1]

# Initialize variables
results_randomly = []
results_direct = []
results_inverse = []
results_training = []

results_estimators = []

predictions_full = []
y_true_loop = []
patient_id_loop = []
# Time to run the experiment
print("Running experiment")
starttime = timeit.default_timer()
for i_fold, (train_index, test_index) in enumerate(kf_out.split(X, Y)):
    print("FOLD: " + str(i_fold))

    # Patients used for train and internal XGB validation
    X_train_whole = X.iloc[train_index, :]
    Y_train_whole = Y[train_index]

    # Patients used to generete a prediction
    X_test = X.iloc[test_index, :]
    Y_test = Y[test_index]

    print("Fitting Full model")
    # Train the model with all the features
    full_model = get_inner_loop_optuna(
        X_train_whole, Y_train_whole, kf_inner, params_optuna
    )
    # Save the fitted model and the test patients of this fold
    full_model["model"].save_model(models_dir / f"Full_fold_{i_fold:03d}.json")
    patient_id_loop.append(patient_ids[test_index])

    # Get probability
    pred_test = full_model["model"].predict_proba(X_test)[:, 1]

    pred_train = full_model["model"].predict_proba(X_train_whole)[:, 1]
    # Compute test metrics
    results_training = compute_results(
        i_fold,
        "Full Model Train",
        pred_train,
        Y_train_whole,
        results_training,
        ths_range=ths_range,
    )

    # Compute train metrics
    results_estimators = save_best_model_params(
        i_fold, "Full Model", full_model, results_estimators
    )
    # Compute metrics
    predictions_full.append(pred_test)
    y_true_loop.append(Y_test)

    # Compute metrics without removing any feature
    results_direct = compute_results(
        i_fold,
        "Full Model",
        pred_test,
        Y_test,
        results_direct,
        ths_range=ths_range,
    )
    results_inverse = compute_results(
        i_fold,
        "Full Model",
        pred_test,
        Y_test,
        results_inverse,
        ths_range=ths_range,
    )
    results_randomly = compute_results(
        i_fold,
        "Full Model",
        pred_test,
        Y_test,
        results_randomly,
        ths_range=ths_range,
    )

    # Direct importance removal
    X_test_loop = X_test.copy()
    for i_features, feature_name in enumerate(direct_removal):
        # Remove feature from the test
        X_test_loop.loc[:, feature_name] = np.nan
        # Generate a prediction with the new test data
        pred_test = full_model["model"].predict_proba(X_test_loop)[:, 1]
        # Compute metrics
        results_direct = compute_results(
            i_fold,
            "Full Model",
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
        # Generate a prediction with the new test data
        pred_test = full_model["model"].predict_proba(X_test_loop)[:, 1]
        # Compute metrics
        results_inverse = compute_results(
            i_fold,
            "Full Model",
            pred_test,
            Y_test,
            results_inverse,
            n_removed_features=i_features + 1,
            ths_range=ths_range,
        )

    # Random scenario: remove n randomly chosen features per patient
    # (n = 1..len(inverse_removal)), each time starting from the full test data
    for n_removed in range(1, len(inverse_removal) + 1):
        X_test_masked = remove_random_features_fix_number(
            X_test,
            features_num=n_removed,
            basic_features=feature_admission,
            rng=rng,
        )
        pred_test = full_model["model"].predict_proba(X_test_masked)[:, 1]
        # Compute metrics
        results_randomly = compute_results(
            i_fold,
            "Full Model",
            pred_test,
            Y_test,
            results_randomly,
            n_removed_features=n_removed,
            ths_range=ths_range,
        )

# Create a dataframe to save
results_randomly_df = results_to_df(results_randomly)
results_direct_df = results_to_df(results_direct)
results_inverse_df = results_to_df(results_inverse)
results_training_df = results_to_df(results_training)

results_estimators = estimator_to_df(results_estimators)


# %%

# % Saving results
print("Saving Results")
results_randomly_df.to_csv(save_dir / "Full_random_remove.csv")
results_direct_df.to_csv(save_dir / "Full_direct_remove.csv")
results_inverse_df.to_csv(save_dir / "Full_inverse_remove.csv")
results_training_df.to_csv(save_dir / "Full_training.csv")

results_estimators.to_csv(save_dir / "Best_Full_model_parameters.csv")

predictions_full = pd.DataFrame(predictions_full)
predictions_full = predictions_full.T
predictions_full.to_csv(save_dir / "Full_predictions.csv")

y_true_loop = pd.DataFrame(y_true_loop)
y_true_loop = y_true_loop.T
y_true_loop.to_csv(save_dir / "Full_y_true.csv")

# Patient ID of every prediction (same layout as Full_predictions.csv)
patient_id_loop = pd.DataFrame(patient_id_loop).T
patient_id_loop.to_csv(save_dir / "Full_patient_ID.csv")

# %%
# Create a experiment summary
experiment_setting = {
    "data_dir": data_dir,
    "save_dir": save_dir,
    "random_state": random_state,
    "out_n_splits": out_n_splits,
    "endpoint_to_use": endpoint_to_use,
    "out_n_repetitions": out_n_repetitions,
    "inner_n_splits": inner_n_splits,
    "variance_ths": variance_ths,
    "ths_range": ths_range,
    "params_optuna": params_optuna,
    "n_participants": n_participants,
    "n_features": n_features,
    "experiment_timing": timeit.default_timer() - starttime,
}

# Save the experiment settings
experiment_setting_df = pd.DataFrame.from_dict(experiment_setting, orient="index")
experiment_setting_df.to_csv(save_dir / "experiment_settings_Full_model.csv")
print("Experiment finished")

# %%
