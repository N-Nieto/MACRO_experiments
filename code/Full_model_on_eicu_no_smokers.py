# %%
import joblib
import numpy as np
import pandas as pd
from lib.data_load_utils import load_CULPRIT_data, get_data_from_features
from lib.experiment_definitions import get_features
from lib.data_processing import remove_low_variance_features
from lib.ml_utils import compute_results, get_inner_loop_optuna, results_to_df       # noqa
from sklearn.model_selection import StratifiedKFold
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# %%
data_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/CULPRIT_data/202302_Jung/" # noqa

# Minimun feature variance
variance_ths = 0.10
# Set random state
random_state = 23

# Inner CV
inner_n_splits = 5

# Define the hyperparameters to tune
params_optuna = {
    "objective": "binary:logistic",
    'eval_metric': 'auc',
    'optuna_trials': 100,
    'random_state': 23,
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

# Extract the Admission features
exp_name = "24hs"
feature_24h = get_features(exp_name)
X = get_data_from_features(patient_info, feature_24h)
X = X.drop(columns="p_rf_smoker_yn")

# Remove low variance features
X = remove_low_variance_features(X, variance_ths)

# Final data shape
n_participants, n_features = X.shape

# Show the feature distribution
print("Full features: " + str(n_features))

eicu_root = "/home/nnieto/Nico/MODS_project/data/eicu-collaborative-research-database-2.0/preprocessed_MACRO/"          # noqa
X_eicu = pd.read_csv(eicu_root + "X_Full_CICU.csv", index_col=0)
X_eicu = X_eicu.drop(columns="p_rf_smoker_yn")

Y_test_eicu = pd.read_csv(eicu_root + "y_CICU.csv", index_col=0)
Y_test_eicu = Y_test_eicu.to_numpy()
# %%


kf_inner = StratifiedKFold(n_splits=inner_n_splits,
                           shuffle=True,
                           random_state=random_state)


results_full = []

print("Fitting Full model")
# Train the model with all the features
full_model = get_inner_loop_optuna(X,
                                   Y,
                                   kf_inner,
                                   params_optuna)

# Get probability
X_eicu = pd.DataFrame(X_eicu, columns=X.columns)
pred_test = full_model["model"].predict_proba(X_eicu)[:, 1]
# Compute metrics without removing any feature
results_full = compute_results(1, "Full Test (eICU)",
                               pred_test, Y_test_eicu, results_full)

pred_train = full_model["model"].predict_proba(X)[:, 1]

# Compute test metrics
results_full = compute_results(1, "Full Train (CULPRIT)",
                               pred_train, Y, results_full)

results_full = results_to_df(results_full)


# %%
# % Saving results
print("Saving Results")
save_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/output/review_1/eICU/full_model/"       # noqa
results_full.to_csv(save_dir+ "Full_performance_CULPRIT_eICU.csv")                    # noqa
# # Save the models in the web_service direction.
joblib.dump(full_model["model"], save_dir + 'Full_model_no_smokers.pkl')

predictions_full = pd.DataFrame(pred_train)
predictions_full = predictions_full.T
predictions_full.to_csv(save_dir+ "predictions_full_CULPRIT_no_smokers.csv")   # noqa

y_true_loop = pd.DataFrame(Y)
y_true_loop = y_true_loop.T
y_true_loop.to_csv(save_dir+ "y_true_CULPRIT_no_smokers.csv")   # noqa

predictions_full = pd.DataFrame(pred_test)
predictions_full = predictions_full.T
predictions_full.to_csv(save_dir+ "predictions_full_eICU_no_smokers.csv")   # noqa

y_true_loop = pd.DataFrame(Y_test_eicu)
y_true_loop = y_true_loop.T
y_true_loop.to_csv(save_dir+ "y_true_eICU_no_smokers.csv")   # noqa
# %%

# %%
results_full
# %%
