# %%
import pandas as pd                     # noqa
import random
from sklearn.linear_model import LogisticRegressionCV
from lib.data_load_utils import load_CULPRIT_data, get_data_from_features
from lib.experiment_definitions import get_features
from sklearn.model_selection import RepeatedStratifiedKFold
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
out_n_repetitions = 10
# Inner CV
inner_n_splits = 10
inner_n_repetitions = 1

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

# Removing patients that died in the first 24hs
patient_info = patient_info[patient_info["fu_ce_Death_d"] != 0]

# Set target
y = patient_info.loc[:, ["patient_ID", endpoint_to_use]]
Y_loop = y.iloc[:, 1].to_numpy()

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

inner_n_splits = 10
inner_n_repetitions = 1
kf_inner = RepeatedStratifiedKFold(n_splits=inner_n_splits,
                                   n_repeats=inner_n_repetitions,
                                   random_state=random_state)

# Logistic regression
thr = 0.5
random_permutation = [False]
random_permutation_number = 1
score_clf = LogisticRegressionCV(cv=kf_inner)

results_SAPS = []
results_CLIP = []
y_true_fold = []
y_true_SAPS = []
y_true_CLIP = []

# Outer loop
for rs in random_permutation:
    print("Randoms State: " + str(rs))
    for i_fold, (train_index, test_index) in enumerate(kf_out.split(X_SAPS_SCORE, Y_loop)):       # noqa
        print("FOLD: " + str(i_fold))
        if rs:
            rnd_permutation_loop = random_permutation_number
        else:
            rnd_permutation_loop = 1

        for rpn in range(rnd_permutation_loop):
            print("Random Permutation Nº: " + str(rpn))
            # Patients used for train and internal XGB validation

            X_train_SAPS_SCORE = X_SAPS_SCORE.iloc[train_index, :]
            X_train_CLIP_SCORE = X_CLIP_SCORE.iloc[train_index, :]
            Y_train_whole = Y[train_index]

            if rs:
                random.shuffle(Y_train_whole)

            # Patients used to generete a prediction
            X_test_SAPS_SCORE = X_SAPS_SCORE.iloc[test_index, :]
            X_test_CLIP_SCORE = X_CLIP_SCORE.iloc[test_index, :]

            Y_test = Y[test_index]

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

            # Compute metrics
            results_SAPS.append(SAPS_SCORE_proba)                                              # noqa
            results_CLIP.append(CLIP_SCORE_proba)                                              # noqa

            y_true_SAPS.append(Y_test_SAPS)
            y_true_CLIP.append(Y_test_CLIP)

# %%
# % Savng results
print("Saving Results")
save_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/output/predictions/v2/"               # noqa

results_SAPS = pd.DataFrame(results_SAPS)
results_SAPS = results_SAPS.T
results_SAPS.to_csv(save_dir+ "SAPS_predictions_for_AUC.csv")   # noqa

results_CLIP = pd.DataFrame(results_CLIP)
results_CLIP = results_CLIP.T
results_CLIP.to_csv(save_dir+ "CLIP_predictions_for_AUC.csv")   # noqa

y_true_SAPS = pd.DataFrame(y_true_SAPS)
y_true_SAPS = y_true_SAPS.T
y_true_SAPS.to_csv(save_dir+ "y_true_SAPS_for_AUC.csv")   # noqa

y_true_CLIP = pd.DataFrame(y_true_CLIP)
y_true_CLIP = y_true_CLIP.T
y_true_CLIP.to_csv(save_dir+ "y_true_CLIP_for_AUC.csv")   # noqa

print("Experiment done")

# %%
