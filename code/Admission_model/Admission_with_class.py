
# %%
import numpy as np
from lib.data_load_utils import load_CULPRIT_data, get_data_from_features
from lib.experiment_definitions import get_features, get_important_features
from lib.data_processing import remove_low_variance_features
from lib.ml_utils import compute_results
from lib.xgb_optuna import XGBClassifier_optuna_es
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
# %%
data_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/CULPRIT_data/202302_Jung/" # noqa

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
    'max_depth_max': 10,
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

# Set target
# endpoint to use
endpoint_to_use = "fu_ce_death_le30d_yn"    # or "fu_ce_death_le365d_yn"
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

base_admission = ["had_dem_age_yr", "had_dem_male_yn"]

# Final data shape
n_participants, n_features = X.shape

# Show the feature distribution
print("Admission features: " + str(n_features))
# %%
kf_out = RepeatedStratifiedKFold(n_splits=out_n_splits,
                                 n_repeats=out_n_repetitions,
                                 random_state=random_state)

kf_inner = StratifiedKFold(n_splits=inner_n_splits,
                           shuffle=True,
                           random_state=random_state)

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

# %%
# Outer loop
for i_fold, (train_index, test_index) in enumerate(kf_out.split(X, Y)):       # noqa
    print("FOLD: " + str(i_fold))
    admission_model = XGBClassifier_optuna_es(kf_inner, params_optuna)

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
    admission_model = admission_model.fit(X_train_whole, Y_train_whole)

    # Get probability
    pred_test = admission_model.predict_proba(X_test)[:, 1]

    pred_train = admission_model.predict_proba(X_train_whole)[:, 1]     # noqa                
    # Compute test metrics
    results_training = compute_results(i_fold, "Admission Train", pred_train, Y_train_whole, results_training, ths_range=ths_range)                 # noqa

    # Compute metrics
    predictions.append(pred_test)
    y_true_loop.append(Y_test)                                                        # noqa
    severity_loop.append(severity_test)                                                        # noqa

    # Compute metrics without removing any feature
    results_direct = compute_results(i_fold, "Admission", pred_test, Y_test, results_direct, ths_range=ths_range)                 # noqa
    results_inverse = compute_results(i_fold, "Admission", pred_test, Y_test, results_inverse, ths_range=ths_range)                  # noqa
    results_randomly = compute_results(i_fold, "Admission", pred_test, Y_test, results_randomly, ths_range=ths_range)                  # noqa


# Create a dataframe to the estimator


# %%
