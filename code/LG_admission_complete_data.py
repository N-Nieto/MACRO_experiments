# %%
from lib.data_load_utils import load_CULPRIT_data, get_data_from_features
from lib.experiment_definitions import get_features
from lib.data_processing import remove_low_variance_features
from lib.ml_utils import compute_results, results_to_df       # noqa
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold    # noqa
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

import pandas as pd
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

# Data load and pre-processing
# Get different features depending on the model
# Get all data
patient_info = load_CULPRIT_data(data_dir)

# Set target
# endpoint to use
endpoint_to_use = "fu_ce_death_le30d_yn"    # or "fu_ce_death_le365d_yn"
y = patient_info.loc[:, ["patient_ID", endpoint_to_use]]


# Extract the Admission features
exp_name = "Admission"
feature_24h = get_features(exp_name)
X = get_data_from_features(patient_info, feature_24h)

# Remove low variance features
X = remove_low_variance_features(X, variance_ths)
X.dropna(inplace=True)
y = y.loc[X.index]
Y = y.iloc[:, 1].to_numpy()
base_admission = ["had_dem_age_yr", "had_dem_male_yn"]

# Final data shape
n_participants, n_features = X.shape

# Show the feature distribution
print("Admission features: " + str(n_features))
print("Admission patients: " + str(n_participants))

# %%
kf_out = RepeatedStratifiedKFold(n_splits=out_n_splits,
                                 n_repeats=out_n_repetitions,
                                 random_state=random_state)

kf_inner = StratifiedKFold(n_splits=inner_n_splits,
                           shuffle=True,
                           random_state=random_state)
predictions_full = []
y_true_loop = []
results_by_fold = []
score_clf = LogisticRegressionCV(cv=kf_inner)

Scaler = StandardScaler()
# Outer loop

for i_fold, (train_index, test_index) in enumerate(kf_out.split(X, Y)):       # noqa
    print("FOLD: " + str(i_fold))

    # Patients used for train and internal XGB validation
    X_train_whole = X.iloc[train_index, :]
    Y_train_whole = Y[train_index]
    X_train_whole = Scaler.fit_transform(X_train_whole, Y_train_whole)
    # Patients used to generete a prediction
    X_test = X.iloc[test_index, :]
    X_test = Scaler.transform(X_test)
    Y_test = Y[test_index]

    score_clf.fit(X=X_train_whole, y=Y_train_whole)

    train_proba = score_clf.predict_proba(X=X_train_whole)[:, 1]                    # noqa
    test_proba = score_clf.predict_proba(X=X_test)[:, 1]                    # noqa

    results_by_fold = compute_results(i_fold, "LG_complete_admission_test", test_proba , Y_test, results_by_fold)                                           # noqa
    results_by_fold = compute_results(i_fold, "LG_complete_admission_train",train_proba , Y_train_whole, results_by_fold)                                           # noqa

    # Compute metrics
    predictions_full.append(test_proba)
    y_true_loop.append(Y_test)

results_pt = results_to_df(results_by_fold)

# %%
# % Saving results
print("Saving Results")
save_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/output/review_1/LG_compare/LG_with_scaler/"       # noqa
results_pt.to_csv(save_dir+ "LG_complete_data_admission.csv")              # noqa

predictions_full = pd.DataFrame(predictions_full)
predictions_full = predictions_full.T
predictions_full.to_csv(save_dir+ "predictions_Admission_complete.csv")   # noqa

y_true_loop = pd.DataFrame(y_true_loop)
y_true_loop = y_true_loop.T
y_true_loop.to_csv(save_dir+ "y_true_Admission_complete.csv")   # noqa
