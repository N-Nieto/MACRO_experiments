# %%
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt

from lib.data_load_utils import load_CULPRIT_data, get_data_from_features
from lib.data_processing import remove_low_variance_features
from lib.experiment_definitions import get_features

data_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/CULPRIT_data/202302_Jung/" # noqa
# Minimun feature variance
variance_ths = 0.10
# Get different features depending on the model
# Get all data
patient_info = load_CULPRIT_data(data_dir)

# # Removing patients that died in the first 24hs

# patient_info = patient_info[patient_info["fu_ce_Death_d"] != 0]

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

eicu_root = "/home/nnieto/Nico/MODS_project/data/eicu-collaborative-research-database-2.0/preprocessed_MACRO/"          # noqa
X_eicu_full = pd.read_csv(eicu_root + "X_Full.csv", index_col=0)
X_eicu_full = pd.DataFrame(X_eicu_full, columns=X_24.columns)

X_eicu_admission = pd.read_csv(eicu_root + "X_admission.csv", index_col=0)
X_eicu_admission = pd.DataFrame(X_eicu_full, columns=X_Admission.columns)

Y_test_eicu = pd.read_csv(eicu_root + "y.csv", index_col=0)
Y_test_eicu = Y_test_eicu.to_numpy()
# Assuming your dataframe is called "data"
# Count the number of missing values in each column
missing_values_count = X_eicu_admission.isnull().sum()

# Calculate the variance of each column
variance = X_eicu_admission.var()

# Calculate the relative number of missing values
total_samples = len(X_eicu_admission)
relative_missing_values = missing_values_count / total_samples

# Create a new dataframe with the information
missing_data_summary = pd.DataFrame({
    'Missing Values': missing_values_count,
    'Variance': variance,
    'Relative Missing Values': relative_missing_values
})

print(missing_data_summary)

# %%
sbn.scatterplot(data=X_eicu_admission, x="had_pex_height_cm",
                y="had_pex_weight_kg")
sbn.scatterplot(data=X_Admission, x="had_pex_height_cm", y="had_pex_weight_kg")
plt.xlabel("Height [cm]")
plt.ylabel("Weight [Kg]")
plt.grid()
plt.xlim([0, max(X_eicu_admission["had_pex_height_cm"])+2])
plt.ylim([0, max(X_eicu_admission["had_pex_weight_kg"])+2])
plt.legend(["eICU", "CULPRIT"], title="Datasets",)
# %%
