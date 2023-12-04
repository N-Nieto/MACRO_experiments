# %%
import pandas as pd                     # noqa
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from CULPRIT_project.code.lib.ml_utils import load_CULPRIT_data, get_data_from_features
from CULPRIT_project.code.lib.ml_utils import compute_results_by_fold
from CULPRIT_project.code.lib.ml_utils import get_features, get_inner_loop_predictions_df
from sklearn.model_selection import RepeatedStratifiedKFold
# %%
data_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/CULPRIT_data/202302_Jung/" # noqa

# endpoint to use
endpoint_to_use = "fu_ce_death_le30d_yn"    # or "fu_ce_death_le365d_yn"

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

# Extract all the 24hs available features
exp_name = "24hs"
patient_features, lab_features, clip_features = get_features(exp_name)

X_final_24, features_24 = get_data_from_features(patient_info, lab_info,
                                                 clip_info,
                                                 patient_features,
                                                 lab_features,
                                                 clip_features)

# Extract all the IABP_SCORE available features
exp_name = "IABP_SCORE"
patient_features, lab_features, clip_features = get_features(exp_name)

X_final_IABP_SCORE, features_IABP_SCORE = get_data_from_features(patient_info,
                                                                 lab_info,
                                                                 clip_info,
                                                                 patient_features,          # noqa
                                                                 lab_features,
                                                                 clip_features)

# Extract all the SAPS_SCORE available features
exp_name = "SAPS_SCORE"
patient_features, lab_features, clip_features = get_features(exp_name)

X_final_SAPS_SCORE, features_SAPS_SCORE = get_data_from_features(patient_info,
                                                                 lab_info,
                                                                 clip_info,
                                                                 patient_features,          # noqa
                                                                 lab_features,
                                                                 clip_features)

# Extract all the SAPS_SCORE available features
exp_name = "CLIP_SCORE"
patient_features, lab_features, clip_features = get_features(exp_name)

X_final_CLIP_SCORE, features_CLIP_SCORE = get_data_from_features(patient_info,
                                                                 lab_info,
                                                                 clip_info,
                                                                 patient_features,          # noqa
                                                                 lab_features,
                                                                 clip_features)

X_24 = X_final_24.drop(columns="patient_ID")
X_admission = X_final_admission.drop(columns="patient_ID")

X_final_IABP_SCORE = X_final_IABP_SCORE.drop(columns="patient_ID")
X_final_SAPS_SCORE = X_final_SAPS_SCORE.drop(columns="patient_ID")
X_final_CLIP_SCORE = X_final_CLIP_SCORE.drop(columns="patient_ID")

# Show the feature distribution
print("Admission features: " + str(X_admission.columns.nunique()))
print("24hs features: " + str(X_24.columns.nunique()))
Y = y.iloc[:, 1].to_numpy()
# %%

# Set random state
random_state = 23
# Validation percentage for XGBoost early stopping
val_percentage_xgb = 0.33

out_n_splits = 10
out_n_repetitions = 10
kf_out = RepeatedStratifiedKFold(n_splits=out_n_splits,
                                 n_repeats=out_n_repetitions,
                                 random_state=random_state)

inner_n_splits = 10
inner_n_repetitions = 1
kf_inner = RepeatedStratifiedKFold(n_splits=inner_n_splits,
                                   n_repeats=inner_n_repetitions,
                                   random_state=random_state)
metric = "error"
early_stopping_rounds = 250
reg_lambda = 0
reg_alpha = 0
params_admission = {
    'initial_n_estimators': 1000,
    'val_percentage': val_percentage_xgb,
    'eval_metric': metric,
    'random_state': random_state,
    'early_stopping_rounds': early_stopping_rounds,
    'reg_alpha': reg_alpha,
    'reg_lambda': reg_lambda,
}

early_stopping_rounds = 250
reg_lambda = 0
reg_alpha = 0
params_24hs = {
    'initial_n_estimators': 1000,
    'val_percentage': val_percentage_xgb,
    'eval_metric': metric,
    'random_state': random_state,
    'early_stopping_rounds': early_stopping_rounds,
    'reg_alpha': reg_alpha,
    'reg_lambda': reg_lambda,
}

early_stopping_rounds = 250
reg_lambda = 0
reg_alpha = 0
params_cascade = {
    'initial_n_estimators': 1000,
    'val_percentage': val_percentage_xgb,
    'eval_metric': metric,
    'random_state': random_state,
    'early_stopping_rounds': early_stopping_rounds,
    'reg_alpha': reg_alpha,
    'reg_lambda': reg_lambda,
}

# Logistic regression
thr = 0.5
random_permutation = [True, False]
random_permutation_number = 1
score_clf = LogisticRegressionCV(cv=kf_inner)

n_participants = X_admission.shape[0]
admision_features = X_admission.shape[1]
n_features_24 = X_24.shape[1]
cascade_features = admision_features + 1

# Initialize variables
model_admission_cv_preds = np.ones(n_participants) * -1
model_24hs_cv_preds = np.ones(n_participants) * -1
cascade_norisk_model_cv_preds = np.ones(n_participants) * -1
cascade_risk_model_cv_preds = np.ones(n_participants) * -1

results_by_fold = []
results_by_ths = []
results_estimators = []

# number of thresholds used
ths_range = list(np.linspace(0, 1, 101))


def get_matching_cohors(X_score, X_admission, X_24, Y):
    # Get the features for the score
    X_score = X_score.dropna()
    X_admission_SCORE = X_admission.iloc[X_score.index]
    X_24_SCORE = X_24.iloc[X_score.index]
    Y_score = Y[X_score.index]
    return X_score, X_admission_SCORE, X_24_SCORE, Y_score


X_IABP_SCORE, X_admission_IABP, X_24_IABP, Y_IABP = get_matching_cohors(X_final_IABP_SCORE, X_admission, X_24, Y)   # noqa
X_SAPS_SCORE, X_admission_SAPS, X_24_SAPS, Y_SAPS = get_matching_cohors(X_final_SAPS_SCORE, X_admission, X_24, Y)   # noqa
X_CLIP_SCORE, X_admission_CLIP, X_24_CLIP, Y_CLIP = get_matching_cohors(X_final_CLIP_SCORE, X_admission, X_24, Y)   # noqa

rs = False
rpn = 0
# %%
# Outer loop

score_list = [["IABP", X_final_IABP_SCORE],
              ["SAPS", X_final_SAPS_SCORE],
              ["CLIP", X_final_CLIP_SCORE]]

for score_name, X_score in score_list:
    X_SCORE, X_admission_score, X_24_score, Y_score = get_matching_cohors(X_score, X_admission, X_24, Y)   # noqa
    results_by_fold = []

    for i_fold, (train_index, test_index) in enumerate(kf_out.split(X_SCORE, Y_score)):       # noqa
        print("FOLD: " + str(i_fold))

        # Patients used for train and internal XGB validation
        X_train_whole_admission = X_admission_score.iloc[train_index, :]
        X_train_whole_24 = X_24_score.iloc[train_index, :]
        X_train_SCORE = X_SCORE.iloc[train_index, :]
        Y_train_whole = Y_score[train_index]

        # Patients used to generete a prediction
        X_test_admission = X_admission_score.iloc[test_index, :]
        X_test_24 = X_24_score.iloc[test_index, :]
        X_test_SCORE = X_SCORE.iloc[test_index, :]

        Y_test = Y_score[test_index]

        print("Fitting Admission model")
        admision_model = get_inner_loop_predictions_df(X_train_whole_admission,       # noqa
                                                       Y_train_whole,
                                                       kf_inner,
                                                       params_admission)
        print("Fitting 24hs model")
        model_24hs = get_inner_loop_predictions_df(X_train_whole_24,
                                                   Y_train_whole,
                                                   kf_inner,
                                                   params_24hs)

        # Get the admission test probabilities
        admission_test_pred = admision_model["model"].predict_proba(X_test_admission)[:, 1]   # noqa

        # Store 24hs predictions
        pred_test_24hs = model_24hs["model"].predict_proba(X_test_24)[:, 1]

        print("Fitting logit using " + score_name + " score")
        score_clf.fit(X=X_train_SCORE, y=Y_train_whole)

        SCORE_proba = score_clf.predict_proba(X=X_test_SCORE)[:, 1]                    # noqa

        # Compute metrics
        results_by_fold = compute_results_by_fold(i_fold, "Admission",rs, rpn, admission_test_pred, Y_test, thr, results_by_fold)                                               # noqa
        results_by_fold = compute_results_by_fold(i_fold, "24hs", rs, rpn, pred_test_24hs, Y_test, thr, results_by_fold)                                                        # noqa
        results_by_fold = compute_results_by_fold(i_fold, "SCORE", rs, rpn, SCORE_proba, Y_test, thr, results_by_fold)                                           # noqa

    results_df = pd.DataFrame(results_by_fold, columns=["Fold",
                                                        "Model",
                                                        "Random State",
                                                        "Random Permutation",
                                                        "Balanced ACC",
                                                        "AUC",
                                                        "F1",
                                                        "Specificity",
                                                        "Sensitivity"])

    # % Savng results
    print("Saving Results")
    save_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/output/matching_cohors/"  # noqa
    results_df.to_csv(save_dir+ "metrics_10x10_matching_cohors_" +score_name+ ".csv")   # noqa
print("Experiment done")

# %%
