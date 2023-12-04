# %%
import pandas as pd                     # noqa
import pickle
import joblib
import shap
import numpy as np
from lib.data_load_utils import load_CULPRIT_data, get_data_from_features
from lib.experiment_definitions import get_features
from lib.data_processing import remove_low_variance_features, naming_for_shap
from lib.ml_utils import get_inner_loop_optuna
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
# %%
# %%
data_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/CULPRIT_data/202302_Jung/" # noqa

# Minimun feature variance
variance_ths = 0.10
# Set random state
random_state = 23
random_permutation = [False]
random_permutation_number = 1

# Cross validation parameters
out_n_splits = 10
out_n_repetitions = 1
# Inner CV
inner_n_splits = 3

# Model Threshold
thr = 0.5
# number of thresholds used
ths_range = list(np.linspace(0, 1, 101))

# Data load and pre-processing
# endpoint to use
endpoint_to_use = "fu_ce_death_le30d_yn"    # or "fu_ce_death_le365d_yn"

# Get different features depending on the model
# Get all data
patient_info = load_CULPRIT_data(data_dir)

# Removing patients that died in the first 24hs
patient_info = patient_info[patient_info["fu_ce_Death_d"] != 0]

# Set target
y = patient_info.loc[:, ["patient_ID", endpoint_to_use]]
Y = y.iloc[:, 1].to_numpy()

# Extract the Admmission features
exp_name = "24hs_v2"
feature_24h = get_features(exp_name)
X_24 = get_data_from_features(patient_info, feature_24h)

# Remove low variance features
X_24 = remove_low_variance_features(X_24, variance_ths)

# Final data shape
n_participants = X_24.shape[0]
n_features = X_24.shape[1]

# Show the feature distribution
print("24hs features: " + str(n_features))

# %% Change naming for Shap

X_24 = naming_for_shap(data_dir, X_24)

# Show the feature distribution
print("24hs features: " + str(X_24.columns.nunique()))
# %%

kf_out = RepeatedStratifiedKFold(n_splits=out_n_splits,
                                 n_repeats=out_n_repetitions,
                                 random_state=random_state)


kf_inner = StratifiedKFold(n_splits=inner_n_splits,
                           shuffle=True,
                           random_state=random_state)


# Data shape
n_participants, n_features_24 = X_24.shape

# Initialize variables for shaply values fro 24hs
shap_24hs_values = np.ones([n_participants, n_features_24]) * -1
shap_24hs_baseline = np.ones(n_participants) * -1
shap_24hs_data = np.ones([n_participants, n_features_24]) * -1

# Outer loop
for i_fold, (train_index, test_index) in enumerate(kf_out.split(X_24, Y)):       # noqa
    print("FOLD: " + str(i_fold))
    # Patients used for train and internal XGB validation
    X_train_whole_24 = X_24.iloc[train_index, :]
    Y_train_whole = Y[train_index]

    # Patients used to generete a prediction
    X_test_24 = X_24.iloc[test_index, :]
    Y_test = Y[test_index]

    print("Fitting 24hs model")
    model_24hs = get_inner_loop_optuna(X_train_whole_24,
                                       Y_train_whole,
                                       kf_inner)

    # Initialize variables for shaply values from 24hs
    explainer = shap.Explainer(model_24hs["model"])
    shap_values = explainer(X_test_24)
    shap_24hs_values[test_index, :] = shap_values.values
    shap_24hs_baseline[test_index] = shap_values.base_values
    shap_24hs_data[test_index, :] = shap_values.data


# %% Saving
print("Saving")
save_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/output/optuna/shap_values/"              # noqa

save_list = [
             [shap_24hs_values, "shap_24hs_values"],
             [shap_24hs_baseline, "shap_24hs_baseline"],
             [shap_24hs_data, "shap_24hs_data"]]


for list_to_save, save_name in save_list:
    with open(save_dir+save_name, "wb") as fp:   # Pickling
        pickle.dump(list_to_save, fp)

X_24.to_csv(save_dir+"X_24hs_v2.csv")

joblib.dump(model_24hs["model"], save_dir + 'model_24hs_v2_shap_values.pkl')                     # noqa
print("Experiment done")
# %%
