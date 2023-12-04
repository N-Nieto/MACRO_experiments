# %%
import pandas as pd
import numpy as np
import random
from lib.data_load_utils import load_CULPRIT_data, get_data_from_features
from lib.experiment_definitions import get_features
from lib.data_processing import remove_low_variance_features
from lib.ml_utils import get_inner_loop_predictions_df
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
initial_n_estimators = 1000
metric = "error"
# Validation percentage for XGBoost early stopping
val_percentage_xgb = 0.33
early_stopping_rounds = 250
reg_lambda = 0
reg_alpha = 0

# Model Threshold
thr = 0.5
# number of thresholds used
ths_range = list(np.linspace(0, 1, 101))

# Data load and pre-processing
# endpoint to use
endpoint_to_use = "fu_ce_death_le30d_yn"    # or "fu_ce_death_le365d_yn"

# Get different features depending on the model
# Get all data
patient_info = load_CULPRIT_data(data_dir)

# Removing patients that died in the first 24hs
patient_info = patient_info[patient_info["fu_ce_Death_d"] != 0]

# Set target
y = patient_info.loc[:, ["patient_ID", endpoint_to_use]]
Y = y.iloc[:, 1].to_numpy()

# Extract the Admmission features
exp_name = "24hs_v2"
feature_24h = get_features(exp_name)
X_24 = get_data_from_features(patient_info, feature_24h)

# Remove low variance features
X_24 = remove_low_variance_features(X_24, variance_ths)

# Final data shape
n_participants, n_features = X_24.shape

# Show the feature distribution
print("24hs features: " + str(n_features))
# %%

kf_out = RepeatedStratifiedKFold(n_splits=out_n_splits,
                                 n_repeats=out_n_repetitions,
                                 random_state=random_state)

inner_n_splits = 10
inner_n_repetitions = 1
kf_inner = RepeatedStratifiedKFold(n_splits=inner_n_splits,
                                   n_repeats=inner_n_repetitions,
                                   random_state=random_state)
params = {
    'initial_n_estimators': initial_n_estimators,
    'val_percentage': val_percentage_xgb,
    'eval_metric': metric,
    'random_state': random_state,
    'early_stopping_rounds': early_stopping_rounds,
    'reg_alpha': reg_alpha,
    'reg_lambda': reg_lambda,
}
# Logistic regression
thr = 0.5
random_permutation = [False]
random_permutation_number = 1

# Initialize variables
model_24hs_cv_preds = np.ones(n_participants) * -1

results_24hs = []
y_true_fold_24hs = []

# Outer loop
for rs in random_permutation:
    print("Randoms State: " + str(rs))
    for i_fold, (train_index, test_index) in enumerate(kf_out.split(X_24, Y)):       # noqa
        print("FOLD: " + str(i_fold))
        if rs:
            rnd_permutation_loop = random_permutation_number
        else:
            rnd_permutation_loop = 1

        for rpn in range(rnd_permutation_loop):
            print("Random Permutation Nº: " + str(rpn))
            # Patients used for train and internal XGB validation
            X_train_whole_24 = X_24.iloc[train_index, :]
            Y_train_whole = Y[train_index]

            if rs:
                random.shuffle(Y_train_whole)

            # Patients used to generete a prediction
            X_test_24 = X_24.iloc[test_index, :]
            Y_test = Y[test_index]

            print("Fitting 24hs model")
            model_24hs = get_inner_loop_predictions_df(X_train_whole_24,
                                                       Y_train_whole,
                                                       kf_inner,
                                                       params)
            # Store 24hs predictions
            pred_test_24hs = model_24hs["model"].predict_proba(X_test_24)[:, 1]

            # Compute metrics
            results_24hs.append(pred_test_24hs)
            y_true_fold_24hs.append(Y_test)                                                        # noqa


# %%
# % Saving results
print("Saving Results")
save_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/output/predictions/v2/"               # noqa

results_24hs = pd.DataFrame(results_24hs)
results_24hs = results_24hs.T
results_24hs.to_csv(save_dir+ "24hs_v2_predictions_for_AUC.csv")   # noqa

y_true_fold_24hs_df = pd.DataFrame(y_true_fold_24hs)
y_true_fold_24hs_df = y_true_fold_24hs_df.T
y_true_fold_24hs_df.to_csv(save_dir+ "24hs_v2_true_predictions_for_AUC.csv")   # noqa

print("Experiment done")

# %%
