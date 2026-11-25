# %%
import sys
import timeit
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.append(str(Path(__file__).resolve().parents[2]))
from lib.data_load_utils import load_CULPRIT_data
from lib.data_processing import remove_random_features_fix_number
from lib.experiment_config import (
    ENDPOINT,
    INNER_N_SPLITS,
    OUT_N_REPETITIONS,
    OUT_N_SPLITS,
    RANDOM_STATE,
    THS_RANGE,
    VARIANCE_THS,
    get_model_data,
    get_params_optuna,
)
from lib.experiment_definitions import get_important_features
from lib.ml_utils import (
    compute_results,
    estimator_to_df,
    get_inner_loop_optuna,
    results_to_df,
    save_best_model_params,
)
from lib.paths import CULPRIT_DIR, OUTPUT_DIR
from lib.utils import ensure_dir

# %%
# Data is stored outside the project (see lib/paths.py); settings shared by
# all experiments are in lib/experiment_config.py
data_dir = CULPRIT_DIR
save_dir = OUTPUT_DIR / "Admission_model"
ensure_dir(save_dir)
# Fitted model of every outer fold (XGBoost JSON), for post-hoc analyses
models_dir = save_dir / "fold_models"
ensure_dir(models_dir)
# Set random state
random_state = RANDOM_STATE
# Random generator for the random feature-removal scenario
rng = np.random.default_rng(random_state)

# Cross validation parameters
out_n_splits = OUT_N_SPLITS
out_n_repetitions = OUT_N_REPETITIONS
# Inner CV
inner_n_splits = INNER_N_SPLITS

# Hyperparameter search space (Supplementary Table 10)
params_optuna = get_params_optuna("Admission")

# %%
# number of thresholds used
ths_range = THS_RANGE

# ########## Data load and pre-processing
patient_info = load_CULPRIT_data(data_dir)

# Admission features of all patients (incl. those who died on day 0)
X, Y, patient_ids = get_model_data(patient_info, "Admission")

# Spot those patients that expired in the first 24hs
severity = (patient_info.loc[X.index, "fu_ce_Death_d"] == 0).astype(int).to_numpy()

# These features are not going to be removed in the missing value experiment
base_admission = ["had_dem_age_yr", "had_dem_male_yn"]

# Final data shape
n_participants, n_features = X.shape
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
patient_id_loop = []
severity_loop = []

# Outer loop
print("Running experiment")
starttime = timeit.default_timer()
for i_fold, (train_index, test_index) in enumerate(kf_out.split(X, Y)):
    print("FOLD: " + str(i_fold))

    # Patients used for train and internal XGB validation
    X_train_whole = X.iloc[train_index]
    Y_train_whole = Y[train_index]

    # Patients used to generate a prediction
    X_test = X.iloc[test_index]
    Y_test = Y[test_index]
    # Spot those patients that expired in the first 24 hs
    severity_test = severity[test_index]
    print("Fitting Admission model")
    # Train the model with all the features
    admission_model = get_inner_loop_optuna(
        X_train_whole, Y_train_whole, kf_inner, params_optuna
    )
    # Save the fitted model and the test patients of this fold
    admission_model["model"].save_model(models_dir / f"Admission_fold_{i_fold:03d}.json")
    patient_id_loop.append(patient_ids[test_index])

    # Get probability
    pred_test = admission_model["model"].predict_proba(X_test)[:, 1]

    pred_train = admission_model["model"].predict_proba(X_train_whole)[:, 1]
    # Compute test metrics
    results_training = compute_results(
        i_fold,
        "Admission Train",
        pred_train,
        Y_train_whole,
        results_training,
        ths_range=ths_range,
    )

    # Compute train metrics
    results_estimators = save_best_model_params(
        i_fold, "Admission", admission_model, results_estimators
    )
    # Compute metrics
    predictions.append(pred_test)
    y_true_loop.append(Y_test)
    severity_loop.append(severity_test)

    # Compute metrics without removing any feature
    results_direct = compute_results(
        i_fold, "Admission", pred_test, Y_test, results_direct, ths_range=ths_range
    )
    results_inverse = compute_results(
        i_fold, "Admission", pred_test, Y_test, results_inverse, ths_range=ths_range
    )
    results_randomly = compute_results(
        i_fold, "Admission", pred_test, Y_test, results_randomly, ths_range=ths_range
    )

    # Direct importance removal
    X_test_loop = X_test.copy()
    for i_features, feature_name in enumerate(direct_removal):
        # Remove feature from the test
        X_test_loop.loc[:, feature_name] = np.nan
        # Generate a prediction with the new test data (with one feature removed)
        pred_test = admission_model["model"].predict_proba(X_test_loop)[:, 1]
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
        pred_test = admission_model["model"].predict_proba(X_test_loop)[:, 1]
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

    # Random scenario: remove n randomly chosen features per patient
    # (n = 1..len(inverse_removal)), each time starting from the full test data
    for n_removed in range(1, len(inverse_removal) + 1):
        X_test_masked = remove_random_features_fix_number(
            X_test,
            features_num=n_removed,
            basic_features=base_admission,
            rng=rng,
        )
        pred_test = admission_model["model"].predict_proba(X_test_masked)[:, 1]
        # Compute metrics
        results_randomly = compute_results(
            i_fold,
            "Admission",
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

# Patient ID of every prediction (same layout as Admission_predictions.csv)
patient_id_loop = pd.DataFrame(patient_id_loop).T
patient_id_loop.to_csv(save_dir / "Admission_patient_ID.csv")


severity_loop = pd.DataFrame(severity_loop)
severity_loop = severity_loop.T
severity_loop.to_csv(save_dir / "Admission_severity.csv")

# Create a experiment summary
experiment_setting = {
    "data_dir": data_dir,
    "save_dir": save_dir,
    "random_state": random_state,
    "out_n_splits": out_n_splits,
    "endpoint_to_use": ENDPOINT,
    "out_n_repetitions": out_n_repetitions,
    "inner_n_splits": inner_n_splits,
    "variance_ths": VARIANCE_THS,
    "ths_range": ths_range,
    "params_optuna": params_optuna,
    "n_participants": n_participants,
    "n_features": n_features,
    "features": list(X.columns),
    "experiment_timing": timeit.default_timer() - starttime,
}

# Save the experiment settings
experiment_setting_df = pd.DataFrame.from_dict(experiment_setting, orient="index")
experiment_setting_df.to_csv(save_dir / "experiment_settings_Admission_model.csv")
print("Experiment finished")

# %%
