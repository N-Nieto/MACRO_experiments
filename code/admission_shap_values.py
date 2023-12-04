# %%
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
data_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/CULPRIT_data/202302_Jung/" # noqa

# Minimun feature variance
variance_ths = 0.10
# Set random state
random_state = 23

# Cross validation parameters
out_n_splits = 10
out_n_repetitions = 1
# Inner CV
inner_n_splits = 3
inner_n_repetitions = 1

# Model Threshold
thr = 0.5
# number of thresholds used
ths_range = list(np.linspace(0, 1, 101))

# Data load and pre-processing

# Get different features depending on the model
# Get all data
patient_info = load_CULPRIT_data(data_dir)

# endpoint to use
endpoint_to_use = "fu_ce_death_le30d_yn"    # or "fu_ce_death_le365d_yn"
# Set target
y = patient_info.loc[:, ["patient_ID", endpoint_to_use]]
Y = y.iloc[:, 1].to_numpy()

# Extract the Admmission features
exp_name = "Admission_v2"
features = get_features(exp_name)
X = get_data_from_features(patient_info, features)

# Remove low variance features
X = remove_low_variance_features(X, variance_ths)
X = naming_for_shap(data_dir, X)

# Final data shape
n_participants = X.shape[0]
n_features = X.shape[1]

# Show the feature distribution
print("Admission features: " + str(n_features))

# %%

kf_out = RepeatedStratifiedKFold(n_splits=out_n_splits,
                                 n_repeats=out_n_repetitions,
                                 random_state=random_state)

kf_inner = StratifiedKFold(n_splits=inner_n_splits,
                           shuffle=True,
                           random_state=random_state)


# Data shape
n_participants, n_features = X.shape

# Initialize variables for shaply values fro 24hs
shap_values = np.ones([n_participants, n_features]) * -1
shap_baseline = np.ones(n_participants) * -1
shap_data = np.ones([n_participants, n_features]) * -1

# Outer loop
for i_fold, (train_index, test_index) in enumerate(kf_out.split(X, Y)):       # noqa
    print("FOLD: " + str(i_fold))
    # Patients used for train and internal XGB validation
    X_train_whole = X.iloc[train_index, :]
    Y_train_whole = Y[train_index]

    # Patients used to generete a prediction
    X_test = X.iloc[test_index, :]
    Y_test = Y[test_index]

    print("Fitting Admission model")
    model = get_inner_loop_optuna(X_train_whole,
                                  Y_train_whole,
                                  kf_inner)

    # Initialize variables for shaply values from 24hs
    explainer = shap.Explainer(model["model"])
    shap_values_loop = explainer(X_test)
    shap_values[test_index, :] = shap_values_loop.values
    shap_baseline[test_index] = shap_values_loop.base_values
    shap_data[test_index, :] = shap_values_loop.data

# %% Saving
print("Saving")
save_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/output/optuna/shap_values/"              # noqa

save_list = [
             [shap_values, "shap_values_"+exp_name],
             [shap_baseline, "shap_baseline"+exp_name],
             [shap_data, "shap_data"+exp_name]]


for list_to_save, save_name in save_list:
    with open(save_dir+save_name, "wb") as fp:   # Pickling
        pickle.dump(list_to_save, fp)

X.to_csv(save_dir+"X_"+exp_name+"_v2.csv")

joblib.dump(model["model"], save_dir + 'model_'+exp_name+'_shap_values.pkl')                     # noqa
print("Experiment done")
# %%
