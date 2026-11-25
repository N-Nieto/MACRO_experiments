#!/usr/bin/env bash
# Full run of the second revision: tests, eICU cohort extraction, CULPRIT-SHOCK
# models and scores, eICU validation, SHAP, learning curves, final models,
# permutation tests and the analyses that use their outputs.
#
#   bash run_rev2.sh                  full run    -> output/rev2/
#   MACRO_QUICK=1 bash run_rev2.sh    quick run   -> output_quick/rev2/
#
# The output folder is lib/paths.OUTPUT_DIR; the data folder is lib/paths.DATA_DIR
# (set MACRO_DATA_DIR to use another one). Each step writes its log to
# <output>/logs/<step>.log and the run stops at the first failing step.
# The plotting notebooks (code/2-plotting/*.ipynb) are not run here; they read
# the results from the same output folder.
set -euo pipefail

cd "$(dirname "$0")"
# No plot windows: figures shown with plt.show() must not block the run
export MPLBACKEND=Agg

PY=(uv run --frozen python)
OUT=$("${PY[@]}" -c "import sys; sys.path.insert(0, 'code'); from lib.paths import OUTPUT_DIR; print(OUTPUT_DIR)")
LOGS="$OUT/logs"
mkdir -p "$OUT" "$LOGS"

# Run information (code version, settings) for this output folder
{
    echo "started: $(date '+%F %T')"
    echo "MACRO_QUICK=${MACRO_QUICK:-0}"
    echo "MACRO_DATA_DIR=${MACRO_DATA_DIR:-<default, lib/paths.py>}"
    echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "uncommitted changes:"
    git status --short 2>/dev/null || true
} > "$OUT/run_info.txt"

run() {
    local script="$1"
    local log="$LOGS/$(basename "$script" .py).log"
    local start=$SECONDS
    echo "[$(date '+%F %T')] $script"
    if ! "${PY[@]}" "$script" > "$log" 2>&1; then
        echo "FAILED: $script (log: $log)"
        tail -n 20 "$log"
        exit 1
    fi
    echo "    done in $(( (SECONDS - start) / 60 )) min"
}

echo "Output folder: $OUT"

# 0. Tests
echo "[$(date '+%F %T')] tests"
uv run --frozen pytest -q > "$LOGS/tests.log" 2>&1 || { echo "FAILED: tests (log: $LOGS/tests.log)"; tail -n 20 "$LOGS/tests.log"; exit 1; }

# 1. Preprocessing: eICU cohort, data checks, BOS,MA2 score
run code/0-preprocessing/eICU_extract_CS_patients.py
run code/0-preprocessing/check_lab_units.py
run code/0-preprocessing/compare_culprit_eicu.py
run code/0-preprocessing/BOSMA_score_experiment.py

# 2. CULPRIT-SHOCK: models (repeated nested CV), scores, missingness-only baseline
run code/1-model_training/Admission_model/Admission_model_experiment.py
run code/1-model_training/Full_model/Full_model_experiment.py
run code/1-model_training/scores/score_main_script_experiment.py
run code/1-model_training/missing_value_analysis/missing_values_baseline_model.py

# 3. SHAP values from the outer-fold models of step 2
run code/1-model_training/shap_analysis/Admission_shap_values.py
run code/1-model_training/shap_analysis/Full_shapy_values.py

# 4. eICU validation (all outcome labels) and BOS,MA2
run code/1-model_training/Admission_model/Admission_model_on_eicu.py
run code/1-model_training/Full_model/Full_model_on_eicu.py
run code/1-model_training/scores/BOSMAN_score_predictions.py
run code/1-model_training/shap_analysis/Admission_shap_on_eicu.py

# 5. Learning curves (Reviewer 3, comment 6)
run code/1-model_training/Admission_model/Admission_learning_curve.py
run code/1-model_training/Full_model/Full_learning_curve.py

# 6. Final models for the web service (hyperparameters from step 2)
run code/1-model_training/train_final_models.py

# 7. Permutation tests (the longest step)
run code/1-model_training/Admission_model/Admission_model_permutation_test.py
run code/1-model_training/Full_model/Full_model_experiment_permutation_test.py

# 8. Analyses: bootstrap CIs and permutation p-values, learning-curve figure
run code/2-plotting/bootstrap_ci.py
run code/2-plotting/plot_learning_curves.py

echo "finished: $(date '+%F %T')" >> "$OUT/run_info.txt"
echo "All steps done. Results in $OUT"
