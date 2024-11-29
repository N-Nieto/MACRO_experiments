# %%
import pandas as pd
import random
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

# Append project path for using the functions in lib
project_root = os.path.dirname(os.path.dirname(os.path.dirname((__file__))))               # noqa
sys.path.append(project_root+"/code/")

from lib.data_load_utils import load_CULPRIT_data, get_data_from_features                   # noqa
from lib.experiment_definitions import get_features                                         # noqa
from lib.ml_utils import compute_results                                                    # noqa
# %%
data_dir = "/data/CULPRIT/"
save_dir = project_root+"/output/"
# Minimun feature variance
variance_ths = 0.10
# Set random state
random_state = 23
random_permutation = [False, True]
random_permutation_number = 1

# Cross validation parameters
out_n_splits = 10
out_n_repetitions = 10
# Inner CV
inner_n_splits = 3

# Model parameters
# Model Threshold
thr = 0.5

# Data load and pre-processing
# endpoint to use
endpoint_to_use = "fu_ce_death_le30d_yn"    # or "fu_ce_death_le365d_yn"

# Get different features depending on the model
# Get all data
patient_info = load_CULPRIT_data(data_dir)

# Set target
y = patient_info.loc[:, ["patient_ID", endpoint_to_use]]
Y = y.iloc[:, 1].to_numpy()

# %%
# Removing patients that died in the first 24hs
patient_info = patient_info[patient_info["fu_ce_Death_d"] != 0]

# Set target
y = patient_info.loc[:, ["patient_ID", endpoint_to_use]]
Y_loop = y.iloc[:, 1].to_numpy()

# Extract all the IABP_SCORE available features
exp_name = "IABP_SCORE"
features_IACP = get_features(exp_name)
X_IABP_SCORE = get_data_from_features(patient_info, features_IACP)

# Extract all the SAPS_SCORE available features
exp_name = "SAPS_SCORE"
features_SAPS = get_features(exp_name)
X_SAPS_SCORE = get_data_from_features(patient_info, features_SAPS)

# Extract all the SAPS_SCORE available features
exp_name = "CLIP_SCORE"
features_CLIP = get_features(exp_name)
X_CLIP_SCORE = get_data_from_features(patient_info, features_CLIP)

# %%
kf_out = RepeatedStratifiedKFold(n_splits=out_n_splits,
                                 n_repeats=out_n_repetitions,
                                 random_state=random_state)


kf_inner = StratifiedKFold(n_splits=inner_n_splits,
                           shuffle=True,
                           random_state=random_state)

score_clf = LogisticRegression()

results_by_fold = []

# Outer loop
for rs in random_permutation:
    print("Randoms State: " + str(rs))
    for i_fold, (train_index, test_index) in enumerate(kf_out.split(X_IABP_SCORE, Y_loop)):       # noqa
        print("FOLD: " + str(i_fold))
        if rs:
            rnd_permutation_loop = random_permutation_number
        else:
            rnd_permutation_loop = 1

        for rpn in range(rnd_permutation_loop):
            print("Random Permutation Nº: " + str(rpn))
            # Patients used for train and internal XGB validation
            X_train_SAPS_SCORE = X_SAPS_SCORE.iloc[train_index, :]
            X_train_IABP_SCORE = X_IABP_SCORE.iloc[train_index, :]
            X_train_CLIP_SCORE = X_CLIP_SCORE.iloc[train_index, :]
            Y_train_whole = Y[train_index]

            if rs:
                random.shuffle(Y_train_whole)

            # Patients used to generete a prediction
            X_test_SAPS_SCORE = X_SAPS_SCORE.iloc[test_index, :]
            X_test_IABP_SCORE = X_IABP_SCORE.iloc[test_index, :]
            X_test_CLIP_SCORE = X_CLIP_SCORE.iloc[test_index, :]

            Y_test = Y[test_index]

            print("Fitting logit using IABP score")
            # Train a loggistic model using the IABP score
            X_train_IABP_SCORE_drop = X_train_IABP_SCORE.dropna()
            Y_train_IABP = Y[X_train_IABP_SCORE_drop.index]
            if rs:
                random.shuffle(Y_train_IABP)
            score_clf.fit(X=X_train_IABP_SCORE_drop, y=Y_train_IABP)
            X_test_IABP_SCORE = X_test_IABP_SCORE.dropna()
            Y_test_IABP = Y[X_test_IABP_SCORE.index]

            IABP_SCORE_proba = score_clf.predict_proba(X=X_test_IABP_SCORE)[:, 1]                    # noqa

            print("Fitting logit using SAPS score")
            # Train a loggistic model using the SAPS score
            X_train_SAPS_SCORE_drop = X_train_SAPS_SCORE.dropna()
            Y_train_SAPS = Y[X_train_SAPS_SCORE_drop.index]
            if rs:
                random.shuffle(Y_train_SAPS)
            score_clf.fit(X=X_train_SAPS_SCORE_drop, y=Y_train_SAPS)
            X_test_SAPS_SCORE = X_test_SAPS_SCORE.dropna()
            Y_test_SAPS = Y[X_test_SAPS_SCORE.index]

            SAPS_SCORE_proba = score_clf.predict_proba(X=X_test_SAPS_SCORE)[:, 1]                    # noqa

            print("Fitting logit using CLIP score")
            # Train a loggistic model using the SAPS score
            X_train_CLIP_SCORE_drop = X_train_CLIP_SCORE.dropna()
            Y_train_CLIP = Y[X_train_CLIP_SCORE_drop.index]
            if rs:
                random.shuffle(Y_train_CLIP)
            score_clf.fit(X=X_train_CLIP_SCORE_drop, y=Y_train_CLIP)
            X_test_CLIP_SCORE = X_test_CLIP_SCORE.dropna()
            Y_test_CLIP = Y[X_test_CLIP_SCORE.index]

            CLIP_SCORE_proba = score_clf.predict_proba(X=X_test_CLIP_SCORE)[:, 1]                    # noqa

            # Store results
            results_by_fold = compute_results(i_fold, "SAPS_SCORE", rs, rpn, SAPS_SCORE_proba, Y_test_SAPS, thr, results_by_fold)                                           # noqa
            results_by_fold = compute_results(i_fold, "IABP_SCORE", rs, rpn, IABP_SCORE_proba, Y_test_IABP, thr, results_by_fold)                                           # noqa
            results_by_fold = compute_results(i_fold, "CLIP_SCORE", rs, rpn, CLIP_SCORE_proba, Y_test_CLIP, thr, results_by_fold)                                           # noqa


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
save_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/output/10x10/v2/"
results_df.to_csv(save_dir+ "metrics_10x10_true_and_random_labels_scores_v3.csv")   # noqa
print("Experiment done")

# %%
