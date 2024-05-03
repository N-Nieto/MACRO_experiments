# %%
import seaborn as sbn
import matplotlib.pyplot as plt
from lib.data_load_utils import load_CULPRIT_data, get_data_from_features
from lib.experiment_definitions import get_features
from lib.data_processing import remove_low_variance_features
# %%
data_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/CULPRIT_data/202302_Jung/" # noqa

# Minimun feature variance
variance_ths = 0.10


# Get different features depending on the model
# Get all data
patient_info = load_CULPRIT_data(data_dir)
# %%
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

# Add a new column to X containing the count
# of missing values for each participant
X['Missing_values'] = X.isna().sum(axis=1)


plt.figure(figsize=[10,5])
sbn.histplot(data=X, x="Missing_values", hue=Y, bins=19, stat="count")
plt.yscale("linear")
plt.legend(["Expired","Alive"])
plt.xlabel("Number of missing values per patient")
plt.ylabel("Number of patients")
plt.title("Admission Data")
# %%

# Extract the Admission features
exp_name = "24hs"
feature_24h = get_features(exp_name)
X = get_data_from_features(patient_info, feature_24h)

# Remove low variance features
X = remove_low_variance_features(X, variance_ths)

# Final data shape
n_participants, n_features = X.shape

# Show the feature distribution
print("Admission features: " + str(n_features))

# Add a new column to X containing the count
# of missing values for each participant
X['Missing_values'] = X.isna().sum(axis=1)

plt.figure(figsize=[10,5])
sbn.histplot(data=X, x="Missing_values", hue=Y, bins=19, stat="count")
plt.yscale("linear")
plt.legend(["Expired","Alive"])
plt.xlabel("Number of missing values per patient")
plt.ylabel("Number of patients")
plt.title("24hs Data")

# %%
