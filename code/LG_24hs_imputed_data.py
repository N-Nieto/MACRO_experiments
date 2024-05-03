# %%
from lib.data_load_utils import load_CULPRIT_data, get_data_from_features
from lib.experiment_definitions import get_features
from lib.data_processing import remove_low_variance_features, round_columns_with_different_digits   # noqa
from lib.ml_utils import compute_results, results_to_df       # noqa
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import seaborn as sbn
import numpy as np
import pandas as pd         # noqa
import  sklearn as skl      # noqa
import matplotlib.pyplot as plt

# %%
data_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/CULPRIT_data/202302_Jung/" # noqa

# Minimun feature variance
variance_ths = 0.10
# Set random state
random_state = 23


# Cross validation parameters
out_n_splits = 10
out_n_repetitions = 10

# ## Data load and pre-processing
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
exp_name = "24hs"
feature_24h = get_features(exp_name)
X = get_data_from_features(patient_info, feature_24h)

# Remove low variance features
X = remove_low_variance_features(X, variance_ths)

# Final data shape
n_participants, n_features = X.shape

# Show the feature distribution
print("24hs features: " + str(n_features))
# %%
kf_out = RepeatedStratifiedKFold(n_splits=out_n_splits,
                                 n_repeats=out_n_repetitions,
                                 random_state=random_state)

predictions_full = []
y_true_loop = []


imp_mean = IterativeImputer(random_state=0)
# score_clf = LogisticRegressionCV(max_iter=10000, cv=3,
#                                  random_state=random_state,
#                                  solver="saga", penalty="elasticnet",
#                                  l1_ratios=[0, 0.25, 0.5, 0.75, 1])

score_clf = LogisticRegressionCV()
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# score_clf = SVC(probability=True, kernel="sigmoid")
# score_clf = RandomForestClassifier(n_estimators=10)
results_by_fold = []

# Create a vector with 20 zeros
admission_round = np.zeros(20, dtype=int)

# Create the second part of the vector
round_24hs = np.array([3, 3, 3, 3, 1, 2, 1, 1, 2, 1, 1, 2, 4, 2, 0], dtype=int)

# Concatenate the two parts
rounded_digits = np.concatenate((admission_round, round_24hs))

# Outer loop

for i_fold, (train_index, test_index) in enumerate(kf_out.split(X, Y)):       # noqa
    print("FOLD: " + str(i_fold))

    # Patients used for train and internal XGB validation
    X_train_whole = X.iloc[train_index, :]
    Y_train_whole = Y[train_index]

    # Patients used to generete a prediction
    X_test = X.iloc[test_index, :]
    Y_test = Y[test_index]

    # impute train data, round for matching with the original distribution
    X_train_whole_imputed = round_columns_with_different_digits(imp_mean.fit_transform(X_train_whole), rounded_digits)
    # impute test data, round for matching with the original distribution
    X_test_imputed = round_columns_with_different_digits(imp_mean.transform(X_test), rounded_digits)

    score_clf.fit(X=X_train_whole_imputed, y=Y_train_whole)

    imputed_train_proba = score_clf.predict_proba(X=X_train_whole_imputed)[:, 1]                    # noqa
    imputed_test_proba = score_clf.predict_proba(X=X_test_imputed)[:, 1]                    # noqa

    results_by_fold = compute_results(i_fold, "LG_imputed_24hs_test", imputed_test_proba, Y_test, results_by_fold)                                           # noqa
    results_by_fold = compute_results(i_fold, "LG_imputed_24hs_train", imputed_train_proba, Y_train_whole, results_by_fold)                                           # noqa

    # Compute metrics
    predictions_full.append(imputed_test_proba)
    y_true_loop.append(Y_test)

results_pt = results_to_df(results_by_fold)

# %%
# # % Savng results
print("Saving Results")
save_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/output/optuna/imputed_data/"       # noqa
results_pt.to_csv(save_dir+ "LG_imp_data_24_hs.csv")              # noqa

predictions_full = pd.DataFrame(predictions_full)
predictions_full = predictions_full.T
predictions_full.to_csv(save_dir+ "predictions_24hs_LG_imp.csv")   # noqa

y_true_loop = pd.DataFrame(y_true_loop)
y_true_loop = y_true_loop.T
y_true_loop.to_csv(save_dir+ "y_true_24hs_LG_imp.csv")   # noqa

# %%
sbn.barplot(data=results_pt, y="Balanced ACC", x="Model")
results_pt[results_pt["Model"] == "LG_imputed_24hs_test"].mean()
# %%
metric_to_plot = "Balanced ACC"
fig, ax = plt.subplots(1, 1, figsize=[20, 10])

sbn.swarmplot(
    data=results_pt,
    x="Model", y=metric_to_plot,
    # order=models_to_plot,
    dodge=False, hue="Model", ax=ax,
    # palette=[[1, 0.1, 0.1],
    #                                           [0.1, 0.2, 0.2],
    #                                           [1, 0.1, 0.1],
    #                                           [0.1, 0.2, 0.2],
    #                                           ]
)

sbn.boxplot(
    data=results_pt, color="w", zorder=1,
    x="Model", y=metric_to_plot,
    # order=models_to_plot,
    dodge=True, ax=ax
)
plt.legend([])
plt.grid()

# %%
