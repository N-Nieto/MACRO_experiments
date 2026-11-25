# %%
"""
Final Admission and Full models for the web service: each model is trained on
all patients of its cohort with the median of the hyperparameters selected in
the outer folds of the main experiment (Best_<model>_model_parameters.csv).

The Admission model uses all patients (as in its cross-validation); the Full
model excludes patients who died within 24 h.

Outputs (output/final_models/): <model>_model.pkl and .json, <model>_explainer.pkl
and <model>_features.csv (column order expected by the model). Missing values
can be passed as NaN or -999.
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import shap
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parent))
from lib.data_load_utils import load_CULPRIT_data
from lib.experiment_config import RANDOM_STATE, get_model_data
from lib.paths import CULPRIT_DIR, OUTPUT_DIR
from lib.utils import ensure_dir

# %%
save_dir = OUTPUT_DIR / "final_models"
ensure_dir(save_dir)

patient_info = load_CULPRIT_data(CULPRIT_DIR)

# %%
for model_name in ["Admission", "Full"]:
    # Hyperparameters chosen in the outer folds (non-shuffled labels)
    estimators = pd.read_csv(
        OUTPUT_DIR / f"{model_name}_model" / f"Best_{model_name}_model_parameters.csv"
    )
    estimators = estimators[~estimators["Random State"].astype(bool)]
    if estimators.empty:
        raise ValueError(f"No hyperparameters found for the {model_name} model")

    X, Y, _ = get_model_data(patient_info, model_name)

    # Model with the median of the fold hyperparameters
    final_model = XGBClassifier(
        n_estimators=round(estimators["Number of Estimators"].median()),
        reg_alpha=estimators["alpha"].median(),
        reg_lambda=estimators["lambda"].median(),
        learning_rate=estimators["eta"].median(),
        max_depth=round(estimators["max_depth"].median()),
        missing=-999,  # NaN is treated as missing as well
        eval_metric="auc",
        random_state=RANDOM_STATE,
        verbosity=0,
        n_jobs=-1,
    )
    final_model.fit(X, Y)
    print(f"{model_name}: trained on {X.shape[0]} patients, {X.shape[1]} features")

    joblib.dump(final_model, save_dir / f"{model_name}_model.pkl")
    final_model.save_model(save_dir / f"{model_name}_model.json")
    joblib.dump(shap.Explainer(final_model), save_dir / f"{model_name}_explainer.pkl")
    pd.Series(X.columns, name="feature").to_csv(save_dir / f"{model_name}_features.csv")

print("Final models saved in " + str(save_dir))
# %%
