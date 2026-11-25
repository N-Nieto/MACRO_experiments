# %%
"""
Learning curve of the Admission model (Reviewer 3, comment 6).

Same cohort, features, search space and outer folds as Admission_model_experiment.py
(first repetition of the repeated 10-fold CV). In every outer fold the
training patients are subsampled (nested and stratified, 10% to 100%); the
hyperparameters are tuned with the inner CV of the subsample and the model is
fitted on it. AUC on the training subsample (in-sample) and on the complete
held-out fold (lib/learning_curve.py).

Outputs (output/Admission_model/):
    Admission_learning_curve.csv           one row per outer fold and training fraction
    Admission_learning_curve_summary.csv   mean and SD over the outer folds
"""
import sys
import timeit
from pathlib import Path

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.append(str(Path(__file__).resolve().parents[2]))
from lib.data_load_utils import load_CULPRIT_data
from lib.experiment_config import (
    INNER_N_SPLITS,
    OUT_N_SPLITS,
    RANDOM_STATE,
    get_model_data,
    get_params_optuna,
)
from lib.learning_curve import (
    LEARNING_CURVE_FRACTIONS,
    run_learning_curve,
    summarize_learning_curve,
)
from lib.paths import CULPRIT_DIR, OUTPUT_DIR
from lib.utils import ensure_dir

model_name = "Admission"
save_dir = OUTPUT_DIR / f"{model_name}_model"
ensure_dir(save_dir)

# %% Data (all patients)
patient_info = load_CULPRIT_data(CULPRIT_DIR)
X, Y, _ = get_model_data(patient_info, model_name)

# %% Learning curve
starttime = timeit.default_timer()
curve = run_learning_curve(
    X,
    Y,
    model_name,
    get_params_optuna(model_name),
    fractions=LEARNING_CURVE_FRACTIONS,
    n_splits=OUT_N_SPLITS,
    inner_n_splits=INNER_N_SPLITS,
    random_state=RANDOM_STATE,
)
print(f"Learning curve done in {(timeit.default_timer() - starttime) / 60:.1f} min")

# %% Saving
summary = summarize_learning_curve(curve)
print(summary.round(3).to_string(index=False))
curve.to_csv(save_dir / f"{model_name}_learning_curve.csv", index=False)
summary.to_csv(save_dir / f"{model_name}_learning_curve_summary.csv", index=False)

# %%
