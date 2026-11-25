# %%
"""
Permutation test of the Admission model: the outcome labels are shuffled and the
whole pipeline (nested CV with the same Optuna search space as the main
experiment) is re-run on the shuffled labels, giving the null distribution of
the cross-validated AUC.

Output: output/Admission_model/Admission_permutation_test.csv (one row per permutation
and outer fold; "Random Permutation Number" identifies the permutation) and
Admission_permutation_test_train.csv (training-set metrics).
"""
import sys
from pathlib import Path

import numpy as np
import optuna
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

sys.path.append(str(Path(__file__).resolve().parents[2]))
from lib.data_load_utils import load_CULPRIT_data
from lib.experiment_config import (
    INNER_N_SPLITS,
    N_PERMUTATIONS,
    OUT_N_SPLITS,
    RANDOM_STATE,
    get_model_data,
    get_params_optuna,
)
from lib.ml_utils import compute_results, get_inner_loop_optuna, results_to_df
from lib.paths import CULPRIT_DIR, OUTPUT_DIR
from lib.utils import ensure_dir

optuna.logging.set_verbosity(optuna.logging.WARNING)
# %%
model_name = "Admission"
save_dir = OUTPUT_DIR / f"{model_name}_model"
ensure_dir(save_dir)

# Number of label permutations (lib/experiment_config.py)
random_permutation_number = N_PERMUTATIONS
# One repetition of the outer 10-fold CV per permutation
out_n_repetitions = 1
# Same search space as the main experiment
params_optuna = get_params_optuna(model_name)
# Seeded generator: the permutations are reproducible
rng = np.random.default_rng(RANDOM_STATE)

# %% Data
patient_info = load_CULPRIT_data(CULPRIT_DIR)
X, Y, patient_ids = get_model_data(patient_info, model_name)

kf_out = RepeatedStratifiedKFold(
    n_splits=OUT_N_SPLITS, n_repeats=out_n_repetitions, random_state=RANDOM_STATE
)
kf_inner = StratifiedKFold(n_splits=INNER_N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# %% Permutations
results_pt = []
results_training = []
for rpn in range(random_permutation_number):
    print("Random permutation number: " + str(rpn))
    Y_permuted = rng.permutation(Y)
    for i_fold, (train_index, test_index) in enumerate(kf_out.split(X, Y_permuted)):
        X_train_whole = X.iloc[train_index]
        Y_train_whole = Y_permuted[train_index]
        X_test = X.iloc[test_index]
        Y_test = Y_permuted[test_index]

        model = get_inner_loop_optuna(X_train_whole, Y_train_whole, kf_inner, params_optuna)

        pred_test = model["model"].predict_proba(X_test)[:, 1]
        pred_train = model["model"].predict_proba(X_train_whole)[:, 1]
        results_pt = compute_results(
            i_fold, model_name, pred_test, Y_test, results_pt, rs=True, rpn=rpn
        )
        results_training = compute_results(
            i_fold, f"{model_name} Train", pred_train, Y_train_whole, results_training,
            rs=True, rpn=rpn,
        )

# %% Saving results
print("Saving Results")
results_to_df(results_pt).to_csv(save_dir / f"{model_name}_permutation_test.csv")
results_to_df(results_training).to_csv(save_dir / f"{model_name}_permutation_test_train.csv")
print("Experiment done")

# %%
