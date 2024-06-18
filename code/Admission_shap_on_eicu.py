
# %%
import shap

import joblib
import pandas as pd
from lib.data_load_utils import load_CULPRIT_data, get_data_from_features
from lib.experiment_definitions import get_features
from lib.data_processing import remove_low_variance_features, naming_for_shap
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
X = naming_for_shap(data_dir, X)

# Final data shape
n_participants, n_features = X.shape

# Show the feature distribution
print("Admission features: " + str(n_features))

eicu_root = "/home/nnieto/Nico/MODS_project/data/eicu-collaborative-research-database-2.0/preprocessed_MACRO/"          # noqa
X_eicu = pd.read_csv(eicu_root + "X_admission_CICU_No_aperiodic.csv",
                     index_col=0)

Y_test_eicu = pd.read_csv(eicu_root + "y_CICU.csv", index_col=0)
Y_test_eicu = Y_test_eicu.to_numpy()

# %%
kf_inner = StratifiedKFold(n_splits=inner_n_splits,
                           shuffle=True,
                           random_state=random_state)
# Initialize the results
results_admission = []

print("Fitting Admission model on CULPRIT")
# Train the model with all the features
admission_model = get_inner_loop_optuna(X,
                                        Y,
                                        kf_inner,
                                        params_optuna)


pred_train = admission_model["model"].predict_proba(X)[:, 1]
# Compute test metrics
results_admission = compute_results(1, "Admission Train (CULPRIT)",
                                    pred_train, Y, results_admission)

pred_test = admission_model["model"].predict_proba(X_eicu)[:, 1]
# Compute test metrics
results_admission = compute_results(1, "Admission Test (eICU)",
                                    pred_test, Y_test_eicu, results_admission)


# Create a dataframe to save
results_admission = results_to_df(results_admission)

# %%
results_admission
# %%
explainer = shap.TreeExplainer(admission_model["model"], X)
shap_values = explainer(X_eicu)

# %%
shap.plots.bar(shap_values, max_display=20)
# %%
shap.plots.beeswarm(shap_values, max_display=20)
