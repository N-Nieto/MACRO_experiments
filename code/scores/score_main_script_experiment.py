# %%
"""
Conventional risk scores on CULPRIT-SHOCK: SAPS II, IABP-SHOCK II, CLIP and
CardShock.

Each score is the single input of a logistic regression that predicts 30-day
mortality. The scores are evaluated with the same outer cross-validation
folds as the Full model (same cohort: patients alive after 24 h, same random
state), using in every fold the patients who have the score. Because the
folds are shared, the out-of-fold predictions can be compared with the Full
model patient by patient (analysis/bootstrap_ci.py).

Outputs (output/scores/):
    <score>.csv                metrics per fold (threshold 0.5); rows with
                               "Random State" True are the label-permutation
                               control (training labels shuffled)
    <score>_predictions.csv    out-of-fold predictions (one column per fold)
    <score>_y_true.csv, <score>_patient_ID.csv  (same layout)
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.append(str(Path(__file__).resolve().parents[1]))
from lib.data_load_utils import load_CULPRIT_data
from lib.experiment_config import (
    ENDPOINT,
    OUT_N_REPETITIONS,
    OUT_N_SPLITS,
    RANDOM_STATE,
    model_cohort,
)
from lib.ml_utils import compute_results, results_to_df, save_fold_predictions
from lib.paths import CULPRIT_DIR, OUTPUT_DIR
from lib.scores import cardshock_score
from lib.utils import ensure_dir

# %%
save_dir = OUTPUT_DIR / "scores"
ensure_dir(save_dir)

# Score name -> column of load_CULPRIT_data()
SCORES = {
    "SAPS": "icu_the_saps2score_num_x",  # SAPS II, as recorded in CULPRIT-SHOCK
    "IABP": "had_dem_iabpscore_c",  # IABP-SHOCK II, as recorded
    "CLIP": "CLIP_Score",  # computed in load_CULPRIT_data
    "CardShock": "CardShock_Score",  # computed below (lib/scores.py)
}
# Label-permutation control: number of refits with shuffled training labels
n_label_permutations = 1
# Decision threshold for the threshold-dependent metrics
thr = 0.5

# %% Data
patient_info = load_CULPRIT_data(CULPRIT_DIR)
patient_info["CardShock_Score"] = cardshock_score(patient_info)

# Same cohort and order as the Full model -> same outer folds
cohort = model_cohort(patient_info, "Full")
Y = cohort[ENDPOINT].to_numpy().astype(int)
patient_ids = cohort["patient_ID"].to_numpy()
for name, column in SCORES.items():
    print(f"{name}: available for {cohort[column].notna().sum()} of {len(cohort)} patients")

# %% Cross-validation
kf_out = RepeatedStratifiedKFold(
    n_splits=OUT_N_SPLITS, n_repeats=OUT_N_REPETITIONS, random_state=RANDOM_STATE
)
rng = np.random.default_rng(RANDOM_STATE)

results = {name: [] for name in SCORES}
oof = {name: {"pred": [], "y": [], "id": []} for name in SCORES}

for i_fold, (train_index, test_index) in enumerate(kf_out.split(cohort, Y)):
    print("FOLD: " + str(i_fold))
    for name, column in SCORES.items():
        x = cohort[column].to_numpy(dtype=float)
        # Patients of this fold with the score available
        train = train_index[~np.isnan(x[train_index])]
        test = test_index[~np.isnan(x[test_index])]

        clf = LogisticRegression().fit(x[train, None], Y[train])
        prob_test = clf.predict_proba(x[test, None])[:, 1]
        prob_train = clf.predict_proba(x[train, None])[:, 1]
        oof[name]["pred"].append(prob_test)
        oof[name]["y"].append(Y[test])
        oof[name]["id"].append(patient_ids[test])

        if len(np.unique(Y[test])) < 2:
            print(f"  {name}: only one class in test fold {i_fold}, metrics skipped")
            continue
        results[name] = compute_results(
            i_fold, f"{name}_Score_test", prob_test, Y[test], results[name], ths_range=thr
        )
        results[name] = compute_results(
            i_fold, f"{name}_Score_train", prob_train, Y[train], results[name], ths_range=thr
        )

        # Label-permutation control
        for rpn in range(n_label_permutations):
            clf_permuted = LogisticRegression().fit(x[train, None], rng.permutation(Y[train]))
            prob_permuted = clf_permuted.predict_proba(x[test, None])[:, 1]
            results[name] = compute_results(
                i_fold, f"{name}_Score_test", prob_permuted, Y[test], results[name],
                rs=True, rpn=rpn, ths_range=thr,
            )

# %% Saving results
print("Saving Results")
for name in SCORES:
    results_to_df(results[name]).to_csv(save_dir / f"{name}.csv")
    save_fold_predictions(save_dir, name, oof[name]["pred"], oof[name]["y"], oof[name]["id"])
print("Experiment done")

# %%
