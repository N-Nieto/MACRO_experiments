
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
from sklearn.linear_model import LogisticRegressionCV

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
    'random_state': 23,
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

# Extract the Admission features
exp_name = "Admission"
feature_24h = get_features(exp_name)
X = get_data_from_features(patient_info, feature_24h)

# Remove low variance features
X = remove_low_variance_features(X, variance_ths)

X = X.drop(columns="p_rf_smoker_yn")
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

min_value = np.min(X, axis=0)

max_value = np.max(X, axis=0)
imp_mean = IterativeImputer(random_state=random_state, min_value=min_value,
                            max_value=max_value, max_iter=20)


LG_admission = LogisticRegressionCV()


results_admission = []

results_LG = []

results_estimators = []

predictions_admission = []
predictions_LG = []
y_true_loop = []

print("Fitting Admission model")
# Train the model with all the features
full_model = get_inner_loop_optuna(X,
                                   Y,
                                   kf_inner,
                                   params_optuna)

pred_train = full_model["model"].predict_proba(X)[:, 1]     # noqa                
# Compute test metrics
results_admission = compute_results(1, "Admission Train (CULPRIT)", pred_train, Y, results_admission)                 # noqa

print("Fitting Imputer")

X_imputed = imp_mean.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

print("Fitting LG model")

LG_admission.fit(X_imputed, Y)

pred_train_lg = LG_admission.predict_proba(X_imputed)[:, 1]

results_LG = compute_results(1, "LG train (CULPRIT)", pred_train_lg, Y, results_LG)                 # noqa


# %%
eicu_root = "/home/nnieto/Nico/MODS_project/data/eicu-collaborative-research-database-2.0/preprocessed_MACRO/"          # noqa
X_eicu = pd.read_csv(eicu_root + "X_admission_CICU_No_aperiodic.csv",
                     index_col=0)
X_eicu = X_eicu.drop(columns="p_rf_smoker_yn")

Y_test_eicu = pd.read_csv(eicu_root + "y_CICU.csv", index_col=0)
Y_test_eicu = Y_test_eicu.to_numpy()

# Get probability
X_eicu = pd.DataFrame(X_eicu, columns=X.columns)
pred_test = full_model["model"].predict_proba(X_eicu)[:, 1]
# Compute metrics without removing any feature
results_admission = compute_results(1, "Admission test (eICU) _no aperiodic", pred_test, Y_test_eicu, results_admission)                 # noqa


X_eicu_imputed = imp_mean.transform(X_eicu)
X_eicu_imputed = pd.DataFrame(X_eicu_imputed, columns=X.columns)

pred_test_lg = LG_admission.predict_proba(X_eicu_imputed)[:, 1]
results_LG = compute_results(1, "LG test (eicu) no aperiodic", pred_test_lg, Y_test_eicu, results_LG)                 # noqa

# %%

X_eicu = pd.read_csv(eicu_root + "X_admission_CICU_BP_merged_mean.csv",
                     index_col=0)

X_eicu = X_eicu.drop(columns="p_rf_smoker_yn")

# Get probability
X_eicu = pd.DataFrame(X_eicu, columns=X.columns)
pred_test = full_model["model"].predict_proba(X_eicu)[:, 1]
# Compute metrics without removing any feature
results_admission = compute_results(1, "Admission test (eICU)_merged_mean", pred_test, Y_test_eicu, results_admission)                 # noqa


X_eicu_imputed = imp_mean.transform(X_eicu)
X_eicu_imputed = pd.DataFrame(X_eicu_imputed, columns=X.columns)

pred_test_lg = LG_admission.predict_proba(X_eicu_imputed)[:, 1]
results_LG = compute_results(1, "LG test (eicu)_merged_mean", pred_test_lg, Y_test_eicu, results_LG)                 # noqa

# %%

X_eicu = pd.read_csv(eicu_root + "X_admission_CICU_nafilled_w_aperiodic.csv",
                     index_col=0)
X_eicu = X_eicu.drop(columns="p_rf_smoker_yn")


# Get probability
X_eicu = pd.DataFrame(X_eicu, columns=X.columns)
pred_test = full_model["model"].predict_proba(X_eicu)[:, 1]
# Compute metrics without removing any feature
results_admission = compute_results(1, "Admission test (eICU) _nafilled_w_periodic", pred_test, Y_test_eicu, results_admission)                 # noqa


X_eicu_imputed = imp_mean.transform(X_eicu)
X_eicu_imputed = pd.DataFrame(X_eicu_imputed, columns=X.columns)

pred_test_lg = LG_admission.predict_proba(X_eicu_imputed)[:, 1]
results_LG = compute_results(1, "LG test (eicu) _nafilled_w_periodic", pred_test_lg, Y_test_eicu, results_LG)                 # noqa
# %%
# Create a dataframe to save
results_admission = results_to_df(results_admission)
results_LG = results_to_df(results_LG)

# %%
results_admission
# %%
