
# %%
import numpy as np
import pandas as pd
from lib.data_load_utils import load_CULPRIT_data, get_data_from_features
from lib.experiment_definitions import get_features
from lib.data_processing import remove_low_variance_features
from lib.ml_utils import compute_results, get_inner_loop_optuna, results_to_df       # noqa
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
import optuna
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
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
    'max_depth_min': 2,
    'max_depth_max': 12,
    'alpha_min': 1e-8,
    'alpha_max': 1.0,
    'lambda_min': 1e-8,
    'lambda_max': 1.0,
    'eta_min': 0.3,
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

# Extract the Admission features
exp_name = "Admission"
feature_24h = get_features(exp_name)
X = get_data_from_features(patient_info, feature_24h)

# Remove low variance features
X = remove_low_variance_features(X, variance_ths)

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


results_direct = []
results_training = []

results_estimators = []

predictions_full = []
predictions_MAPIE = []
y_true_loop = []

imp_mean = IterativeImputer(random_state=random_state)

# Outer loop
for i_fold, (train_index, test_index) in enumerate(kf_out.split(X, Y)):       # noqa
    print("FOLD: " + str(i_fold))

    # Patients used for train and internal XGB validation
    X_train_whole = X.iloc[train_index, :]
    Y_train_whole = Y[train_index]

    # Patients used to generete a prediction
    X_test = X.iloc[test_index, :]
    Y_test = Y[test_index]

    # impute train data
    X_train_whole = np.round(imp_mean.fit_transform(X_train_whole))
    # impute test data
    X_test_imputed = np.round(imp_mean.transform(X_test))

    print("Fitting Admission model")
    # Train the model with all the features
    full_model = get_inner_loop_optuna(X_train_whole,
                                       Y_train_whole,
                                       kf_inner,
                                       params_optuna)

    # Get probability
    pred_test = full_model["model"].predict_proba(X_test_imputed)[:, 1]
    # Compute metrics without removing any feature
    results_direct = compute_results(i_fold, "Admission_imp", pred_test, Y_test, results_direct, ths_range=ths_range)                 # noqa

    pred_train = full_model["model"].predict_proba(X_train_whole)[:, 1]     # noqa                
    # Compute test metrics
    results_training = compute_results(i_fold, "Admission Train _Imp", pred_train, Y_train_whole, results_training, ths_range=ths_range)                 # noqa

    # Compute metrics
    predictions_full.append(pred_test)
    y_true_loop.append(Y_test)

# Create a dataframe to save
results_direct_df = results_to_df(results_direct)
results_training_df = results_to_df(results_training)

# %%

# % Saving results
print("Saving Results")
save_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/output/optuna/imputed_data/"       # noqa
results_direct_df.to_csv(save_dir+ "Admission_imp_big_experiment_direct_remove2.csv")                    # noqa
results_training_df.to_csv(save_dir+ "Admission_big_experiment_training2.csv")                  # noqa


predictions_full = pd.DataFrame(predictions_full)
predictions_full = predictions_full.T
predictions_full.to_csv(save_dir+ "predictions_Admission_imp2.csv")   # noqa

y_true_loop = pd.DataFrame(y_true_loop)
y_true_loop = y_true_loop.T
y_true_loop.to_csv(save_dir+ "y_true_Admission_imp2.csv")   # noqa

# %%
