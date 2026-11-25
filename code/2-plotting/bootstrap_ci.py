# %%
"""
95% confidence intervals of the AUC by patient-level bootstrap, paired AUC
differences between models, and permutation-test p-values.

Uses only the predictions saved by the other scripts (no model is refitted):
    CULPRIT-SHOCK CV  output/<model>_model/<model>_{predictions,y_true,patient_ID}.csv
                      output/scores/<score>_{predictions,y_true,patient_ID}.csv
                      code/1-model_training/missing_value_analysis/output/missing_values/{predictions,y_true,patient_ID}_<model>_base_missing.csv
    eICU              output/eICU/<model>_model/<model>_eICU_predictions.csv
                      output/eICU/BOSMAN_eICU_predictions.csv
                      (one entry per eICU label, from the y_<label> columns)
    permutation tests output/<model>_model/<model>_permutation_test.csv
Missing inputs are skipped with a message.

The CV AUC is the mean over CV repetitions of the AUC of the pooled
out-of-fold predictions; the CI resamples patients (lib/bootstrap.py). The
previous "CIs" (mean +/- 1.96 SD over the 100 test folds) are reported next
to it for comparison.

Outputs (output/analysis/, aggregate values only): auc_ci.csv,
auc_differences.csv, permutation_pvalues.csv, bootstrap_report.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from lib.bootstrap import (
    bootstrap_auc,
    bootstrap_auc_difference,
    fold_auc_summary,
    read_fold_predictions,
    single_test_set,
)
from lib.eICU_processing import EICU_OUTCOMES
from lib.experiment_config import OUT_N_SPLITS, QUICK, RANDOM_STATE
from lib.paths import MISSING_VALUES_DIR, OUTPUT_DIR
from lib.utils import ensure_dir

# %%
n_boot = 200 if QUICK else 2000
save_dir = OUTPUT_DIR / "analysis"
ensure_dir(save_dir)


def cv_files(folder: Path, name: str) -> list[Path]:
    return [folder / f"{name}_{s}.csv" for s in ("predictions", "y_true", "patient_ID")]


def missing_files(model: str) -> list[Path]:
    folder = MISSING_VALUES_DIR
    return [folder / f"{s}_{model}_base_missing.csv" for s in ("predictions", "y_true", "patient_ID")]


CV_SOURCES = {
    "Admission": cv_files(OUTPUT_DIR / "Admission_model", "Admission"),
    "Full": cv_files(OUTPUT_DIR / "Full_model", "Full"),
    "SAPS II": cv_files(OUTPUT_DIR / "scores", "SAPS"),
    "IABP-SHOCK II": cv_files(OUTPUT_DIR / "scores", "IABP"),
    "CLIP": cv_files(OUTPUT_DIR / "scores", "CLIP"),
    "CardShock": cv_files(OUTPUT_DIR / "scores", "CardShock"),
    "Admission (missingness only)": missing_files("Admission"),
    "Full (missingness only)": missing_files("Full"),
}
EICU_FILES = {
    "Admission": OUTPUT_DIR / "eICU" / "Admission_model" / "Admission_eICU_predictions.csv",
    "Full": OUTPUT_DIR / "eICU" / "Full_model" / "Full_eICU_predictions.csv",
    "BOS,MA2": OUTPUT_DIR / "eICU" / "BOSMAN_eICU_predictions.csv",
}
# One entry per model and eICU label, e.g. "Full (eICU, in_hospital_death)"
EICU_SOURCES = {
    f"{model} (eICU, {outcome})": (file, outcome)
    for outcome in EICU_OUTCOMES
    for model, file in EICU_FILES.items()
}
PAIRS = [
    ("Full", "Admission"),
    ("Full", "SAPS II"),
    ("Full", "IABP-SHOCK II"),
    ("Full", "CLIP"),
    ("Full", "CardShock"),
    ("Full", "Full (missingness only)"),
    ("Admission", "SAPS II"),
    ("Admission", "IABP-SHOCK II"),
    ("Admission", "CLIP"),
    ("Admission", "CardShock"),
    ("Admission", "Admission (missingness only)"),
]
for outcome in EICU_OUTCOMES:
    PAIRS += [
        (f"Full (eICU, {outcome})", f"BOS,MA2 (eICU, {outcome})"),
        (f"Admission (eICU, {outcome})", f"BOS,MA2 (eICU, {outcome})"),
        (f"Full (eICU, {outcome})", f"Admission (eICU, {outcome})"),
    ]

# %% Load predictions
predictions = {}
for name, files in CV_SOURCES.items():
    if all(f.exists() for f in files):
        predictions[name] = read_fold_predictions(*files, n_splits=OUT_N_SPLITS)
    else:
        print(f"skipped {name}: missing {[f.name for f in files if not f.exists()]}")
for name, (file, outcome) in EICU_SOURCES.items():
    label = f"y_{outcome}"
    if not file.exists():
        print(f"skipped {name}: missing {file.name}")
        continue
    d = pd.read_csv(file, index_col=0)
    if label not in d:
        print(f"skipped {name}: no {label} column in {file.name} (rerun the eICU scripts)")
        continue
    d = d.dropna(subset=[label])
    predictions[name] = single_test_set(
        d["prediction"].to_numpy(), d[label].astype(int).to_numpy(), d.index.to_numpy()
    )

# %% AUC with bootstrap CI
rows = []
for name, long in predictions.items():
    result = {"model": name, **bootstrap_auc(long, n_boot=n_boot, seed=RANDOM_STATE)}
    if long["fold"].nunique() > 1:
        folds = fold_auc_summary(long)
        result.update(
            folds,
            fold_CI_low=folds["fold_AUC_mean"] - 1.96 * folds["fold_AUC_sd"],
            fold_CI_high=folds["fold_AUC_mean"] + 1.96 * folds["fold_AUC_sd"],
            n_repetitions=long["repetition"].nunique(),
        )
    rows.append(result)
    print(f"{name}: AUC {result['AUC']:.3f} [{result['CI_low']:.3f}, {result['CI_high']:.3f}]")
auc_ci = pd.DataFrame(rows)

# %% Paired differences
rows = []
for a, b in PAIRS:
    if a in predictions and b in predictions:
        result = bootstrap_auc_difference(predictions[a], predictions[b], n_boot=n_boot, seed=RANDOM_STATE)
        rows.append({"model_a": a, "model_b": b, **result})
auc_diff = pd.DataFrame(rows)

# %% Permutation tests
rows = []
for model in ["Admission", "Full"]:
    file = OUTPUT_DIR / f"{model}_model" / f"{model}_permutation_test.csv"
    if not file.exists() or model not in predictions:
        print(f"skipped {model} permutation test")
        continue
    permuted = pd.read_csv(file, index_col=0)
    permuted = permuted[permuted["Model"] == model]
    permuted = permuted.drop_duplicates(subset=["Fold", "Random Permutation Number"], keep="first")
    null = permuted.groupby("Random Permutation Number")["AUC"].mean().to_numpy()
    observed = fold_auc_summary(predictions[model])["fold_AUC_mean"]
    rows.append(
        {
            "model": model,
            "observed_fold_AUC_mean": observed,
            "n_permutations": len(null),
            "null_AUC_mean": float(np.mean(null)),
            "null_AUC_max": float(np.max(null)),
            "p_value": (1 + int(np.sum(null >= observed))) / (1 + len(null)),
        }
    )
perm = pd.DataFrame(rows)

# %% Saving
auc_ci.to_csv(save_dir / "auc_ci.csv", index=False)
auc_diff.to_csv(save_dir / "auc_differences.csv", index=False)
perm.to_csv(save_dir / "permutation_pvalues.csv", index=False)


lines = [
    "# AUC confidence intervals (patient-level bootstrap)",
    "",
    f"{n_boot} bootstrap samples. CV AUC = mean over repetitions of the pooled out-of-fold AUC.",
    "'fold mean +/- 1.96 SD' is the interval reported so far (spread of the test folds).",
    "",
    "| model | patients | events | AUC [95% CI] | fold mean +/- 1.96 SD |",
    "|---|---|---|---|---|",
]
for _, r in auc_ci.iterrows():
    folds = (
        f"{r['fold_AUC_mean']:.3f} [{r['fold_CI_low']:.3f}, {r['fold_CI_high']:.3f}]"
        if "fold_AUC_mean" in r and pd.notna(r["fold_AUC_mean"]) else ""
    )
    lines.append(
        f"| {r['model']} | {r['n_patients']} | {r['n_events']} | "
        f"{r['AUC']:.3f} [{r['CI_low']:.3f}, {r['CI_high']:.3f}] | {folds} |"
    )
lines += ["", "## Paired AUC differences (same patients)", "",
          "| A | B | patients | AUC A | AUC B | A - B [95% CI] | p |", "|---|---|---|---|---|---|---|"]
for _, r in auc_diff.iterrows():
    lines.append(
        f"| {r['model_a']} | {r['model_b']} | {r['n_patients']} | {r['AUC_a']:.3f} | {r['AUC_b']:.3f} | "
        f"{r['difference']:+.3f} [{r['CI_low']:+.3f}, {r['CI_high']:+.3f}] | {r['p_value']:.3f} |"
    )
if len(perm):
    lines += ["", "## Permutation tests", "", "| model | observed | permutations | null mean | null max | p |",
              "|---|---|---|---|---|---|"]
    for _, r in perm.iterrows():
        lines.append(
            f"| {r['model']} | {r['observed_fold_AUC_mean']:.3f} | {r['n_permutations']} | "
            f"{r['null_AUC_mean']:.3f} | {r['null_AUC_max']:.3f} | {r['p_value']:.3f} |"
        )
(save_dir / "bootstrap_report.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# %%
