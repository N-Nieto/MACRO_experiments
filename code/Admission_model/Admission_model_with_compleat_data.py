
# %%
import pandas as pd
from lib.data_load_utils import load_CULPRIT_data, get_data_from_features
from lib.experiment_definitions import get_features
from lib.data_processing import remove_low_variance_features
from lib.ml_utils import compute_results, get_inner_loop_optuna, results_to_df
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
import optuna
from sklearn.linear_model import LogisticRegressionCV
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

# Define the hyperparameters to tune
params_optuna = {
    "objective": "binary:logistic",
    'eval_metric': 'auc',
    'optuna_trials': 100,
    'random_state': random_state,
    'max_depth_min': 1,
    'max_depth_max': 5,
    'alpha_min': 1e-8,
    'alpha_max': 10,
    'lambda_min': 1e-8,
    'lambda_max': 100,
    'eta_min': 0.1,
    'eta_max': 1,
    "early_stopping_rounds": 100,
    "num_boost_round": 10000,
}
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

# Keep only complete data
X.dropna(inplace=True)
y = y.loc[X.index]
Y = y.iloc[:, 1].to_numpy()

# Final data shape
n_participants, n_features = X.shape

# Show the feature distribution
print("Admission features: " + str(n_features))
print("Nº patients: " + str(n_participants))

# %%
kf_out = RepeatedStratifiedKFold(n_splits=out_n_splits,
                                 n_repeats=out_n_repetitions,
                                 random_state=random_state)

kf_inner = StratifiedKFold(n_splits=inner_n_splits,
                           shuffle=True,
                           random_state=random_state)


# Initialize variables
results_direct = []
results_training = []

predictions = []
predictions_LG = []
y_true_loop = []

results_by_fold = []
score_clf = LogisticRegressionCV()

# Outer loop
for i_fold, (train_index, test_index) in enumerate(kf_out.split(X, Y)):       # noqa
    print("FOLD: " + str(i_fold))

    # Patients used for train and internal XGB validation
    X_train_whole = X.iloc[train_index, :]
    Y_train_whole = Y[train_index]

    # Patients used to generete a prediction
    X_test = X.iloc[test_index, :]
    Y_test = Y[test_index]

    print("Fitting Admission model")
    # Train the model with all the features
    admission_model = get_inner_loop_optuna(X_train_whole,
                                            Y_train_whole,
                                            kf_inner,
                                            params_optuna)

    # Get probability
    pred_test = admission_model["model"].predict_proba(X_test)[:, 1]

    pred_train = admission_model["model"].predict_proba(X_train_whole)[:, 1]     # noqa                
    # Compute test metrics
    results_training = compute_results(i_fold, "Admission Train complete", pred_train, Y_train_whole, results_training)                 # noqa

    # Compute metrics
    predictions.append(pred_test)
    y_true_loop.append(Y_test)                                                        # noqa

    # Compute metrics without removing any feature
    results_direct = compute_results(i_fold, "Admission complete", pred_test, Y_test, results_direct)                 # noqa

    print("Fitting LG model")
    score_clf.fit(X=X_train_whole, y=Y_train_whole)

    imputed_train_proba = score_clf.predict_proba(X=X_train_whole)[:, 1]                    # noqa
    imputed_test_proba = score_clf.predict_proba(X=X_test)[:, 1]                    # noqa
    predictions_LG.append(imputed_test_proba)

    results_by_fold = compute_results(i_fold, "LG_complete_admission_test", imputed_test_proba, Y_test, results_by_fold)                                           # noqa
    results_by_fold = compute_results(i_fold, "LG_imputed_admission_train", imputed_train_proba, Y_train_whole, results_by_fold)                                           # noqa

# Create a dataframe to save
results_direct_df = results_to_df(results_direct)
results_training_df = results_to_df(results_training)

results_LG = results_to_df(results_by_fold)

# %%

# % Saving results
print("Saving Results")
save_dir = "/home/nnieto/Nico/MODS_project/CULPRIT_project/output/review_1/LG_compare/"       # noqa
results_direct_df.to_csv(save_dir+ "Admission_complete.csv")                    # noqa
results_training_df.to_csv(save_dir+ "Admission_complete_traingin.csv")                  # noqa


predictions = pd.DataFrame(predictions)
predictions = predictions.T
predictions.to_csv(save_dir+ "predictions_Admission_complete.csv")   # noqa


results_LG.to_csv(save_dir+ "LG_Admission_complete.csv")              # noqa

results_LG = pd.DataFrame(predictions_LG)
results_LG = results_LG.T
results_LG.to_csv(save_dir+ "predictions_LG_Admission_complete.csv")   # noqa

y_true_loop = pd.DataFrame(y_true_loop)
y_true_loop = y_true_loop.T
y_true_loop.to_csv(save_dir+ "y_true_Admission_complete.csv")   # noqa

# %%
