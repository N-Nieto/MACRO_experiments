# %% imports
import joblib
import shap
import pickle
import pandas as pd
from xgboost import XGBClassifier
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname((__file__))))               # noqa
sys.path.append(project_root+"/code/")
from lib.data_load_utils import load_CULPRIT_data, get_data_from_features                   # noqa
from lib.data_processing import remove_low_variance_features                                # noqa
from lib.experiment_definitions import get_features                                         # noqa
# %% Load data
# Get the estimators
estimators_dir = project_root+"/output/optuna/big_experiment/"                     # noqa
estimators = pd.read_csv(estimators_dir + "Best_Admission_model_parameters.csv")            # noqa
# Get the estimators for the admission
estimators_admission = estimators[estimators["Model"] == "Admission"]
# Get only the estimators for the fit with No Shaffled data
estimators_admission = estimators_admission[estimators_admission["Random State"] == False]          # noqa

estimators = pd.read_csv(estimators_dir + "Best_Full_model_parameters_v3.csv")            # noqa
# Get the estimators for the 24hs model
estimators_Full = estimators[estimators["Model"] == "Full"]
# Get only the estimators for the fit with No Shaffled data
estimators_Full = estimators_Full[estimators_Full["Random State"] == False]                         # noqa

data_dir = "/data/CULPRIT/"
save_dir = project_root+"/final_models/"

# Minimun feature variance
variance_ths = 0.10
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

# Extract the 24hs features
exp_name = "Admission"
feature_admission = get_features(exp_name)
X_Admission = get_data_from_features(patient_info, feature_admission)
# Remove low variance features
X_Admission = remove_low_variance_features(X_Admission, variance_ths)


# Removing patients that died in the first 24hs
patient_info = patient_info[patient_info["fu_ce_Death_d"] != 0]

# Extract the 24hs features
exp_name = "24hs"
feature_24h = get_features(exp_name)
X_24 = get_data_from_features(patient_info, feature_24h)

# Remove low variance features
X_24 = remove_low_variance_features(X_24, variance_ths)

# Final data shape
n_participants, n_features = X_24.shape

# Show the feature distribution
print("24hs features: " + str(n_features))

# Get the median of the estimators for the admission
median_admission_estimators = round(estimators_admission["Number of Estimators"].median())          # noqa
median_admission_alpha = estimators_admission["alpha"].median()          # noqa
median_admission_lambda = estimators_admission["lambda"].median()          # noqa
median_admission_eta = estimators_admission["eta"].median()          # noqa
median_admission_max_depth = round(estimators_admission["max_depth"].median())          # noqa

# Get the median of the estimators for the Full
median_Full_estimators = round(estimators_Full["Number of Estimators"].median())          # noqa
median_Full_alpha = estimators_Full["alpha"].median()          # noqa
median_Full_lambda = estimators_Full["lambda"].median()          # noqa
median_Full_eta = estimators_Full["eta"].median()          # noqa
median_Full_max_depth = round(estimators_Full["max_depth"].median())          # noqa


# Create a model with the median of all the estimators
final_admission_model = XGBClassifier(n_estimators=median_admission_estimators,
                                      reg_alpha=median_admission_alpha,
                                      reg_lambda=median_admission_lambda,
                                      eta=median_admission_eta,
                                      max_depth=median_admission_max_depth,
                                      missing=-999,
                                      eval_metric="auc",
                                      random_state=23,
                                      verbosity=0,
                                      n_jobs=-1)

# Create a model with the median of all the estimators
final_24hs_model = XGBClassifier(n_estimators=median_Full_estimators,
                                 reg_alpha=median_Full_alpha,
                                 reg_lambda=median_Full_lambda,
                                 eta=median_Full_eta,
                                 max_depth=median_Full_max_depth,
                                 missing=-999,
                                 eval_metric="auc",
                                 random_state=23,
                                 verbosity=0,
                                 n_jobs=-1)

# Fit the models with the Whole dataset
final_admission_model.fit(X_Admission, Y)
final_24hs_model.fit(X_24, Y)

# Initialize variables for shaply values from Admission
explainer_admission = shap.Explainer(final_admission_model)
# Initialize variables for shaply values from 24hs
explainer_full = shap.Explainer(final_24hs_model)

# # Save the models in the web_service direction.
joblib.dump(final_admission_model, save_dir + 'Admission_model.pkl')
joblib.dump(final_24hs_model, save_dir + 'Full_model.pkl')

# %%

# Create the SHAP explainer
explainer_admission = shap.Explainer(final_admission_model)
explainer_Full = shap.Explainer(final_24hs_model)

# Save the explainer
joblib.dump(explainer_admission, save_dir + "Admission_explainer.pkl")
joblib.dump(explainer_Full, save_dir + "Full_explainer.pkl")

# %%
# Save the explainer
with open("explainer_admission.pkl", "wb") as f:
    pickle.dump(explainer_admission, f)

# %%
with open("explainer_admission.pkl", "wb") as f:
    explainer_admission.save(f)
# %%
