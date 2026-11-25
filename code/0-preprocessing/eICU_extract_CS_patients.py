# %%
"""
eICU cardiogenic-shock cohort (Cardiac ICU, ICD-9 785.51, one stay per
patient) with the CULPRIT-SHOCK features, taken from every eICU table that
documents them (lib/eICU_features.py).

Outputs:
    <eICU>/preprocessed_MACRO/X_Admission_CICU.csv, X_Full_CICU.csv, outcomes_CICU.csv
    output/data_checks/eicu_feature_coverage.md   availability per feature and
        source, matched eICU strings (aggregate counts only)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from lib.eICU_data_loading import (
    load_eicu_24hs_features,
    load_eicu_diagnosis,
    load_eicu_dyslipidemia,
    load_eicu_patient_information,
)
from lib.eICU_features import eicu_coverage_report, extract_admission_flags
from lib.eICU_processing import (
    EICU_OUTCOMES,
    admission_name_matching,
    eicu_bnp_to_ntprobnp,
    eICU_filter_CS_patients,
    eicu_outcomes,
    full_model_name_matching,
)
from lib.experiment_definitions import get_features
from lib.paths import EICU_DIR, EICU_PREPROCESSED_DIR, OUTPUT_DIR
from lib.utils import ensure_dir

# Raw eICU tables and output folders (see lib/paths.py)
data_dir = str(EICU_DIR) + "/"
save_dir = EICU_PREPROCESSED_DIR
ensure_dir(save_dir)
report_dir = OUTPUT_DIR / "data_checks"
ensure_dir(report_dir)
# eICU reports BNP, CULPRIT-SHOCK NT-proBNP. NT-proBNP proxy, one of
# lib.eICU_processing.EICU_BNP_OPTIONS: "ishihara2022" (published conversion
# with BMI and eGFR, default), "ratio" (x6.25), "as_is" (submitted manuscript),
# "missing"
bnp_handling = "ishihara2022"

# %% Cohort: cardiogenic shock (ICD-9 785.51) in the Cardiac ICU
diagnosis = load_eicu_diagnosis(data_dir)
patients = load_eicu_patient_information(data_dir)
merge_data = pd.merge(patients, diagnosis, how="outer")
# Drop all the patients that don't have an endpoint
merge_data = merge_data.dropna(subset="hospitaldischargestatus")
del diagnosis, patients

merge_data = eICU_filter_CS_patients(merge_data, True)

# One stay per patient, prioritizing stays with endpoint = 1
merge_data = merge_data.sort_values(by="hospitaldischargestatus", ascending=False)
merge_data = merge_data.drop_duplicates(subset="uniquepid", keep="first")
assert merge_data["patientunitstayid"].nunique() == merge_data["uniquepid"].nunique()
cohort = merge_data.set_index("patientunitstayid")
print(f"eICU cardiogenic-shock cohort: {len(cohort)} patients")

# %% Binary features and admission vital signs from all eICU sources
flags, vitals, source_log = extract_admission_flags(data_dir, merge_data)

# Dyslipidemia also from lipid labs measured before admission
dyslipidemia = load_eicu_dyslipidemia(data_dir, cohort.index).set_index("patientunitstayid")
lipid_positive = dyslipidemia.index[dyslipidemia["dyslipidemia"] == 1]
source_log.add("p_rf_dyslip_yn", "lab (lipids before admission)", lipid_positive)
flags.loc[flags.index.isin(lipid_positive), "p_rf_dyslip_yn"] = 1

# %% Outcome labels (all options, see EICU_OUTCOMES) and the early-death flag
outcomes = eicu_outcomes(merge_data)
print("eICU outcomes (events / patients with a known label):")
for name, description in EICU_OUTCOMES.items():
    n_events = int(outcomes[name].sum())
    n_known = int(outcomes[name].notna().sum())
    print(f"  {name}: {n_events} / {n_known}  ({description})")
print(
    "  died within 24 h of ICU admission (excluded for the Full model): "
    f"{int(outcomes['death_within_24h'].sum())}"
)

# %% Admission features
demographics = cohort.loc[:, ["gender", "age", "admissionheight", "admissionweight"]].apply(
    pd.to_numeric, errors="coerce"
)
admission = pd.concat([demographics, flags, vitals], axis=1)
X_admission = admission_name_matching(admission)
X_admission.to_csv(save_dir / "X_Admission_CICU.csv")
outcomes.to_csv(save_dir / "outcomes_CICU.csv")

# %% Full-model (24 h) features
lab = load_eicu_24hs_features(data_dir, cohort.index).reindex(cohort.index)
full = pd.concat([admission, lab], axis=1)
# NT-proBNP proxy from BNP (uses age, sex, height, weight and creatinine)
full = eicu_bnp_to_ntprobnp(full, method=bnp_handling)
# Features for ALL patients; the patients who died within 24 h are excluded
# when loading (load_eICU, exclude_early_deaths), as in CULPRIT-SHOCK.
# X_Full_CICU.csv also keeps troponin I (eicu_troponin_I) for the troponin-T
# proxy of load_eICU.
X_full = full_model_name_matching(full)
X_full.to_csv(save_dir / "X_Full_CICU.csv")

# %% Coverage report (aggregate counts only)
report = eicu_coverage_report(
    X_full.reindex(columns=get_features("24hs") + ["eicu_troponin_I"]),
    source_log,
    get_features("24hs") + ["eicu_troponin_I"],
)
(report_dir / "eicu_feature_coverage.md").write_text(report)
print(f"Coverage report: {report_dir / 'eicu_feature_coverage.md'}")
print("DONE!")

# %%
