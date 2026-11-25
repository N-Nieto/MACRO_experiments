"""
Readable names of the model features, used for the SHAP analyses and the
figures. They replace the former data/CULPRIT/Shorten_table.xlsx (NAME ->
LABEL), which is no longer available.

Names follow the feature tables of the Supplementary Material; units are the
harmonized units of lib/unit_harmonization.TARGET_UNITS. XGBoost refuses
feature names with "[", "]" or "<", so units are written in parentheses.
"""

import pandas as pd

FEATURE_LABELS = {
    # Patient information
    "had_dem_age_yr": "Age (years)",
    "had_dem_male_yn": "Male sex",
    "had_pex_weight_kg": "Weight (kg)",
    "had_pex_height_cm": "Height (cm)",
    # Comorbidities
    "previous_heart_complications": "Known coronary artery disease",
    "p_mh_hfail_yn": "Previous congestive heart failure",
    "p_mh_stroke_yn": "Previous stroke",
    "p_mh_pad_yn": "Known peripheral artery disease",
    "p_mh_renf_yn": "Known renal insufficiency (GFR under 30 mL/min)",
    "p_mh_dial_yn": "Chronic dialysis",
    "p_rf_aht_yn": "Arterial hypertension",
    "p_rf_dyslip_yn": "Dyslipidemia",
    "p_rf_dm_yn": "Diabetes mellitus",
    # ECG
    "hpr_ecg_sinrhy_y": "Sinus rhythm",
    "hpr_ecg_afib_y": "Atrial fibrillation",
    "hpr_ecg_avblock3_y": "AV-block III",
    "hpr_ecg_stemi_yn": "ST-segment elevation",
    # Clinical shock characteristics
    "hpr_hmdyn_hr_bpm": "Heart rate (bpm)",
    "hpr_hmdyn_sbp_mmhg": "Systolic blood pressure (mmHg)",
    "hpr_hmdyn_dbp_mmhg": "Diastolic blood pressure (mmHg)",
    "had_sy_ams_yn": "Altered mental status",
    "had_sy_extremity_yn": "Cold, clammy skin and extremities",
    "had_base_mechvent_yn": "Mechanical ventilation",
    "had_base_cpr24h_yn": "Resuscitation within 24 h",
    # Organ perfusion (first 24 h)
    "admission_lactate": "Serum lactate, admission (mmol/L)",
    "icu_lab_lact8hpci_x": "Serum lactate, 8 h (mmol/L)",
    "icu_lab_lact16hpci_x": "Serum lactate, 16 h (mmol/L)",
    "icu_lab_lact24hpci_x": "Serum lactate, 24 h (mmol/L)",
    # Laboratory assessment
    "pbnp": "NT-proBNP (pg/mL)",
    "icu_lab_ck_x": "CK (U/L)",
    "tnt": "Troponin T (pg/mL)",
    "creatine": "Creatinine (µmol/L)",
    "white_cell_count": "White blood cell count (10^9/L)",
    "hematocrit": "Hematocrit (%)",
    "crp": "CRP (mg/L)",
    "icu_lab_inr_r": "INR",
    "glucose": "Glucose (mg/dL)",
    "alat": "ALAT (µkat/L)",
    # Treatment and events in the first 24 h
    "hpe_proc_mechs_yn": "Mechanical support",
    "renal_replacement_therapy": "Renal replacement therapy",
    "cathecholamine_therapy": "Catecholamine therapy",
    "sepsis": "Sepsis",
    "ventricular_fibrillation": "Ventricular fibrillation",
    "stroke": "Stroke",
}


def feature_label(feature: str) -> str:
    """Readable name of a feature code (the code itself if it has none)."""
    return FEATURE_LABELS.get(feature, feature)


def rename_features(data: pd.DataFrame) -> pd.DataFrame:
    """Copy of data with the feature codes replaced by their readable names."""
    return data.rename(columns=FEATURE_LABELS)
