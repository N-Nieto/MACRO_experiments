
# %%
import numpy as np
import pandas as pd
from lib.data_load_utils import load_CULPRIT_data, get_data_from_features
from lib.experiment_definitions import get_features, get_important_features
from lib.data_processing import remove_low_variance_features
from lib.data_processing import remove_random_features_fix_number
from lib.ml_utils import compute_results, get_inner_loop_optuna, results_to_df, save_best_model_params       # noqa
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
# %%
data_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/CULPRIT_data/202302_Jung/" # noqa

# Minimun feature variance
variance_ths = 0.10
# Set random state
random_state = 23
rs = False        # Random State
rpn = 0           # Random Permutation Number

# Cross validation parameters
out_n_splits = 10
out_n_repetitions = 10

# Inner CV
inner_n_splits = 3

# Define the hyperparameters to tune
params_optuna = {
    "objective": "binary:logistic",
    'eval_metric': 'auc',
    'optuna_trials': 100,
    'random_state': random_state,
    'max_depth_min': 2,
    'max_depth_max': 12,
    'alpha_min': 1e-8,
    'alpha_max': 1.0,
    'lambda_min': 1e-8,
    'lambda_max': 1.0,
    'eta_min': 0.3,
    'eta_max': 1,
    "early_stopping_rounds": 100,
    "num_boost_round": 10000,
}

# number of thresholds used
ths_range = list(np.linspace(0, 1, 101))

# Data load and pre-processing

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
feature_admission = X_Admission.columns

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

kf_inner = StratifiedKFold(n_splits=inner_n_splits,
                           shuffle=True,
                           random_state=random_state)

porcentages = list(np.linspace(0, 1, 101))


# Get features in the direct order of importance
direct_removal = get_important_features("Full")

# Inverse order of importance
inverse_removal = direct_removal[::-1]

# Initialize variables
results_randomly = []
results_direct = []
results_inverse = []
results_training = []

results_estimators = []

predictions_full = []
y_true_loop = []

for i_fold, (train_index, test_index) in enumerate(kf_out.split(X, Y)):       # noqa
    print("FOLD: " + str(i_fold))

    # Patients used for train and internal XGB validation
    X_train_whole = X.iloc[train_index, :]
    Y_train_whole = Y[train_index]

    # Patients used to generete a prediction
    X_test = X.iloc[test_index, :]
    Y_test = Y[test_index]

    print("Fitting Full model")
    # Train the model with all the features
    full_model = get_inner_loop_optuna(X_train_whole,
                                       Y_train_whole,
                                       kf_inner,
                                       params_optuna)

    # Get probability
    pred_test = full_model["model"].predict_proba(X_test)[:, 1]

    pred_train = full_model["model"].predict_proba(X_train_whole)[:, 1]     # noqa                
    # Compute test metrics
    results_training = compute_results(i_fold, "Full Train",
                                       pred_train,
                                       Y_train_whole, results_training,
                                       ths_range=ths_range,
                                       rs=rs, rpn=rpn)

    # Compute train metrics
    results_estimators = save_best_model_params(i_fold, "Full",
                                                full_model,
                                                results_estimators,
                                                rs=rs, rpn=rpn)
    # Compute metrics
    predictions_full.append(pred_test)
    y_true_loop.append(Y_test)                                                        # noqa

    # Compute metrics without removing any feature
    results_direct = compute_results(i_fold, "Full", pred_test, Y_test, results_direct, ths_range=ths_range, rs=rs, rpn=rpn)                 # noqa
    results_inverse = compute_results(i_fold, "Full", pred_test, Y_test, results_inverse, ths_range=ths_range, rs=rs, rpn=rpn)                  # noqa
    results_randomly = compute_results(i_fold, "Full", pred_test, Y_test, results_randomly, ths_range=ths_range, rs=rs, rpn=rpn)                  # noqa

    # Direct importance removal
    X_test_loop = X_test.copy()
    for i_features, feature_name in enumerate(direct_removal):
        # Remove feature from the test
        X_test_loop.loc[:, feature_name] = np.nan
        # Generate a prediction with the new test data
        pred_test = full_model["model"].predict_proba(X_test_loop)[:, 1]           # noqa
        # Compute metrics
        results_direct = compute_results(i_fold, "Full", pred_test, Y_test, results_direct, n_removed_features=i_features+1, ths_range=ths_range, rs=rs, rpn=rpn)                  # noqa

    # Inverse importance removal
    X_test_loop = X_test.copy()
    for i_features, feature_name in enumerate(inverse_removal):
        # Remove feature from the test
        X_test_loop.loc[:, feature_name] = np.nan
        # Generate a prediction with the new test data
        pred_test = full_model["model"].predict_proba(X_test_loop)[:, 1]           # noqa
        # Compute metrics
        results_inverse = compute_results(i_fold, "Full", pred_test, Y_test, results_inverse, n_removed_features=i_features+1, ths_range=ths_range, rs=rs, rpn=rpn)                # noqa

    # Randomly removal with probability
    X_test_loop = X_test.copy()
    # Randomply remove
    for i_features, feature_name in enumerate(inverse_removal):
        # Store 24hs predictions
        X_test_maked = remove_random_features_fix_number(X_test_loop,
                                                            features_num=i_features+1,                   # noqa
                                                            basic_features=feature_admission)         # noqa
        pred_test = full_model["model"].predict_proba(X_test_maked)[:, 1]                     # noqa
        # Compute metrics
        results_randomly = compute_results(i_fold, "Full", pred_test, Y_test, results_randomly, n_removed_features=i_features+1, ths_range=ths_range, rs=rs, rpn=rpn)                # noqa

# Create a dataframe to save
results_randomly_df = results_to_df(results_randomly)
results_direct_df = results_to_df(results_direct)
results_inverse_df = results_to_df(results_inverse)
results_training_df = results_to_df(results_training)

results_estimators = pd.DataFrame(results_estimators, columns=["Fold",
                                                               "Model",
                                                               "Random State",              # noqa
                                                               "Random Permutation",        # noqa
                                                               "Number of Estimators",      # noqa
                                                               "alpha",
                                                               "lambda",
                                                               "eta",
                                                               "max_depth"])    # noqa    


# %%

# % Savng results
print("Saving Results")
save_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/output/optuna/big_experiment/"       # noqa
results_randomly_df.to_csv(save_dir+ "Full_big_experiment_random_remove_v3.csv")              # noqa
results_direct_df.to_csv(save_dir+ "Full_big_experiment_direct_remove_v3.csv")                    # noqa
results_inverse_df.to_csv(save_dir+ "Full_big_experiment_inverse_remove_v3.csv")                  # noqa
results_training_df.to_csv(save_dir+ "Full_big_experiment_training_v3.csv")                  # noqa

results_estimators.to_csv(save_dir+ "Best_Full_model_parameters_v3.csv")   # noqa

predictions_full = pd.DataFrame(predictions_full)
predictions_full = predictions_full.T
predictions_full.to_csv(save_dir+ "predictions_Full_v3.csv")   # noqa

y_true_loop = pd.DataFrame(y_true_loop)
y_true_loop = y_true_loop.T
y_true_loop.to_csv(save_dir+ "y_true_Full_v3.csv")   # noqa

# %%
