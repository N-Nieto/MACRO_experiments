# %%
"""
Baseline: how much can be predicted from the missingness pattern alone?
The features of each model are replaced by 0/1 missing-value indicators and
the same nested CV (folds, search space) as the main experiment is run.

Outputs (code/missing_value_analysis/output/missing_values/, see lib/paths.py):
<model>_base_missing.csv (metrics),
predictions_<model>_base_missing.csv, y_true_<model>_base_missing.csv,
patient_ID_<model>_base_missing.csv (one column per outer fold).
"""
import sys
from pathlib import Path

import optuna
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

sys.path.append(str(Path(__file__).resolve().parents[1]))
from lib.data_load_utils import load_CULPRIT_data
from lib.experiment_config import (
    INNER_N_SPLITS,
    OUT_N_REPETITIONS,
    OUT_N_SPLITS,
    RANDOM_STATE,
    get_model_data,
    get_params_optuna,
)
from lib.ml_utils import compute_results, get_inner_loop_optuna, results_to_df
from lib.paths import CULPRIT_DIR, MISSING_VALUES_DIR
from lib.utils import ensure_dir

optuna.logging.set_verbosity(optuna.logging.WARNING)
# %%
save_dir = MISSING_VALUES_DIR
ensure_dir(save_dir)

patient_info = load_CULPRIT_data(CULPRIT_DIR)

kf_out = RepeatedStratifiedKFold(
    n_splits=OUT_N_SPLITS, n_repeats=OUT_N_REPETITIONS, random_state=RANDOM_STATE
)
kf_inner = StratifiedKFold(n_splits=INNER_N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# %%
for model_name in ["Admission", "Full"]:
    X, Y, patient_ids = get_model_data(patient_info, model_name)
    # Missing-value indicators of the model's features
    X = X.isna().astype(int)
    params_optuna = get_params_optuna(model_name)

    results = []
    predictions = []
    y_true_loop = []
    patient_id_loop = []
    for i_fold, (train_index, test_index) in enumerate(kf_out.split(X, Y)):
        print(f"{model_name} FOLD: {i_fold}")
        X_train_whole = X.iloc[train_index]
        Y_train_whole = Y[train_index]
        X_test = X.iloc[test_index]
        Y_test = Y[test_index]

        model = get_inner_loop_optuna(X_train_whole, Y_train_whole, kf_inner, params_optuna)
        pred_test = model["model"].predict_proba(X_test)[:, 1]

        results = compute_results(
            i_fold, f"{model_name} on missing values", pred_test, Y_test, results
        )
        predictions.append(pred_test)
        y_true_loop.append(Y_test)
        patient_id_loop.append(patient_ids[test_index])

    # Saving results
    print("Saving Results")
    results_to_df(results).to_csv(save_dir / f"{model_name}_base_missing.csv")
    for values, prefix in [
        (predictions, "predictions"),
        (y_true_loop, "y_true"),
        (patient_id_loop, "patient_ID"),
    ]:
        pd.DataFrame([list(v) for v in values]).T.to_csv(
            save_dir / f"{prefix}_{model_name}_base_missing.csv"
        )
print("Experiment done")
# %%
