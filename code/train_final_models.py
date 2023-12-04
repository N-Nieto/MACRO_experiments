# %% imports
import joblib
import pandas as pd
from xgboost import XGBClassifier
from CULPRIT_project.code.lib.ml_utils import load_CULPRIT_data, get_features
from CULPRIT_project.code.lib.ml_utils import get_data_from_features

# %% Load data
# Get the estimators
estimators_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/output/10x10/"                     # noqa
estimators = pd.read_csv(estimators_dir + "estimators_10x10_true_and_random_labels.csv")            # noqa

# Get the estimators for the admission
estimators_admission = estimators[estimators["Model"] == "Admission"]
# Get only the estimators for the fit with No Shaffled data
estimators_admission = estimators_admission[estimators_admission["Random State"] == False]          # noqa

# Get the estimators for the 24hs model
estimators_24hs = estimators[estimators["Model"] == "24hs"]
# Get only the estimators for the fit with No Shaffled data
estimators_24hs = estimators_24hs[estimators_24hs["Random State"] == False]                         # noqa

# Load data
data_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/CULPRIT_data/202302_Jung/" # noqa

# target to use
endpoint_to_use = "fu_ce_death_le30d_yn"

# Get different features depending on the model
# Get all data
patient_info, lab_info, clip_info = load_CULPRIT_data(data_dir)
# Set target
y = patient_info.loc[:, ["patient_ID", endpoint_to_use]]

# Extract the Admmission features
exp_name = "Admission"
patient_features, lab_features, clip_features = get_features(exp_name)
X_final_admission, features_admission = get_data_from_features(patient_info,
                                                               lab_info,
                                                               clip_info,
                                                               patient_features,        # noqa
                                                               lab_features,
                                                               clip_features)
# Drop the Patients ID
X_admission = X_final_admission.drop(columns="patient_ID")


# Extract all the 24hs available features
exp_name = "24hs"
patient_features, lab_features, clip_features = get_features(exp_name)

X_final_24, features_24 = get_data_from_features(patient_info, lab_info,
                                                 clip_info,
                                                 patient_features,
                                                 lab_features,
                                                 clip_features)
# Drop the Patients ID
X_24 = X_final_24.drop(columns="patient_ID")
# get the target in the correct format
Y = y.iloc[:, 1].to_numpy()

# Get the median of the estimators for the admission
median_admission_estimators = round(estimators_admission["Number of Estimators"].median())          # noqa
# Get the median of the estimators for the 24hs
median_24hs_estimators = round(estimators_24hs["Number of Estimators"].median())                    # noqa


# Create a model with the median of all the estimators
final_admission_model = XGBClassifier(n_estimators=median_admission_estimators,
                                      n_jobs=-1,
                                      reg_alpha=0,
                                      reg_lambda=0,
                                      missing=-999,
                                      eval_metric="error",
                                      random_state=23,
                                      verbosity=0)

# Create a model with the median of all the estimators
final_24hs_model = XGBClassifier(n_estimators=median_24hs_estimators,
                                 n_jobs=-1,
                                 reg_alpha=0,
                                 reg_lambda=0,
                                 missing=-999,
                                 eval_metric="error",
                                 random_state=23,
                                 verbosity=0)
# Fit the models with the Whole dataset
final_admission_model.fit(X_admission, Y)
final_24hs_model.fit(X_24, Y)

# Save the models in the web_service direction.
save_dir = '/home/nnieto/Nico/MODS_project/CULPRIT_project/web_service/model_prediction/models/'    # noqa
joblib.dump(final_admission_model, save_dir + 'model_admission_final.pkl')
joblib.dump(final_24hs_model, save_dir + 'model_24hs_final.pkl')

# %%

data = pd.read_excel("/home/nnieto/Nico/MODS_project/CULPRIT_project/CULPRIT_data/202302_Jung/CULPRIT-data_20210407.xlsx",
                     sheet_name=None)

# Load patient data
patient_info = data["patient_data"]
# %%

patient_info["patient_ID"] = patient_info["cor_ctr_center_id"].astype(str) + "-" + patient_info["cor_pt_caseno_id"].astype(str) # noqa

patient_random = data["patient_data_random"]

# %%
patient_random["had_ie_random_grp_c"].value_counts()
# %%
patient_random["had_ie_random_time"].nunique()


# %%
# target to use
endpoint_to_use = "fu_ce_death_le30d_yn"

# Set target
y = patient_info.loc[:, ["patient_ID", endpoint_to_use]]

# %%

patient_info["random"] = patient_random["had_ie_random_grp_c"]
patient_info["target"] = y["fu_ce_death_le30d_yn"]

# %%

patients_filtered = patient_info[patient_info["random"]==2]
patients_filtered = patients_filtered[patients_filtered["fu_ce_death_le30d_yn"]==1]

# %%
patients_filtered["had_pex_bmi_kgm2"].nunique()
# %%
import matplotlib.pyplot as plt
import seaborn as sbn
sbn.swarmplot(data=patients_filtered, x="had_pex_bmi_kgm2",y="had_dem_age_yr")
# %%
