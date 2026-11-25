"""lib.eICU_features on a hand-made mini eICU with known answers (strings as in eICU)."""
import pandas as pd
import pytest
from lib.eICU_data_loading import load_eicu_24hs_features
from lib.eICU_features import (
    calendar_day,
    eicu_coverage_report,
    extract_admission_flags,
    first_day_minimum_vitals,
)

PH = "notes/Progress Notes/Past History/Organ Systems"
PE = "notes/Progress Notes/Physical Exam/Physical Exam"
VS = f"{PE}/Constitutional/Vital Sign and Physiological Data"


def write(folder, name, rows, columns):
    pd.DataFrame(rows, columns=columns).to_csv(folder / f"{name}.csv.gz", index=False)


@pytest.fixture(scope="module")
def mini_eicu(tmp_path_factory):
    d = tmp_path_factory.mktemp("eicu")
    # Patient 1 admitted at 23:00 (offset 61 min = next calendar day), 2 at 08:00, 3 at 12:00
    patient = pd.DataFrame({
        "patientunitstayid": [1, 2, 3],
        "unitadmittime24": ["23:00:00", "08:00:00", "12:00:00"],
        "apacheadmissiondx": ["Shock, cardiogenic", "Cardiac arrest (with or without respiratory arrest)",
                              "Infarction, acute myocardial (MI)"],
    })
    write(d, "pastHistory", [
        (1, f"{PH}/Cardiovascular (R)/Congestive Heart Failure/CHF", "CHF"),
        (1, f"{PH}/Endocrine (R)/Non-Insulin Dependent Diabetes/medication dependent", "medication dependent"),
        (1, f"{PH}/Renal  (R)/Renal Failure/renal failure- not currently dialyzed", "renal failure- not currently dialyzed"),
        (1, f"{PH}/Cardiovascular (R)/Myocardial Infarction/MI - remote", "MI - remote"),
        (2, f"{PH}/Renal  (R)/Renal Failure/renal failure - hemodialysis", "renal failure - hemodialysis"),
        (2, f"{PH}/Neurologic/Strokes/stroke - date unknown", "stroke - date unknown"),
        (2, f"{PH}/Cardiovascular (R)/Arrhythmias/atrial fibrillation - chronic", "atrial fibrillation - chronic"),
        (3, f"{PH}/Pulmonary/COPD/COPD - severe", "COPD - severe"),
    ], ["patientunitstayid", "pasthistorypath", "pasthistoryvalue"])
    write(d, "diagnosis", [
        (1, 100, "cardiovascular|chest pain / ASHD|acute coronary syndrome|acute myocardial infarction (with ST elevation)|anterior wall"),
        (2, 100, "cardiovascular|chest pain / ASHD|acute coronary syndrome|acute myocardial infarction (no ST elevation)"),
        (3, 2000, "cardiovascular|chest pain / ASHD|acute coronary syndrome|acute myocardial infarction (with ST elevation)"),  # too late
        (1, 1000, "cardiovascular|shock / hypotension|sepsis|sepsis with single organ dysfunction- acute renal failure"),  # day 1
        (2, 3000, "cardiovascular|shock / hypotension|septic shock"),  # day 2 -> not an early event
        (3, 30, "infectious diseases|systemic/other infections|signs and symptoms of sepsis (SIRS)|due to non-infectious process without organ dysfunction"),
        (3, 60, "cardiovascular|arrhythmias|ventricular fibrillation"),
        (2, 200, "neurologic|disorders of vasculature|stroke|ischemic stroke"),
        (3, 100, "endocrine|glucose metabolism|diabetes mellitus|Type II"),
        (3, 100, "cardiovascular|chest pain / ASHD|hyperlipidemia"),
        (3, 100, "cardiovascular|vascular disorders|hypertension"),
        (1, 100, "pulmonary|pulmonary hypertension"),
        (2, 50, "cardiovascular|arrhythmias|AV block|complete heart block"),
    ], ["patientunitstayid", "diagnosisoffset", "diagnosisstring"])
    write(d, "treatment", [
        (1, -200, "cardiovascular|arrhythmias|cardiac defibrillation"),
        (1, 10, "pulmonary|ventilation and oxygenation|mechanical ventilation"),
        (2, 30, "pulmonary|ventilation and oxygenation|non-invasive ventilation"),
        (1, 300, "cardiovascular|ventricular dysfunction|intraaortic balloon pump"),
        (2, 300, "cardiovascular|ventricular dysfunction|intraaortic balloon pump|removal"),
        (3, 1000, "renal|dialysis|C V V H D"),
        (1, 3000, "renal|dialysis|hemodialysis"),  # 23:00 + 50 h -> day 2
        (2, 100, "cardiovascular|shock|vasopressors|norepinephrine > 0.1 micrograms/kg/min"),
    ], ["patientunitstayid", "treatmentoffset", "treatmentstring"])
    write(d, "infusionDrug", [
        (1, 30, "Norepinephrine (mcg/min)", "5"),
        (3, 30, "Dobutamine (mcg/kg/min)", "0"),  # rate 0 -> not running
        (3, 40, "Phenylephrine (mcg/min)", "50"),  # not a catecholamine
    ], ["patientunitstayid", "infusionoffset", "drugname", "drugrate"])
    write(d, "respiratoryCare", [(3, 30)], ["patientunitstayid", "ventstartoffset"])
    write(d, "apachePredVar", [(1, 0), (2, 1), (3, 0)], ["patientunitstayid", "diabetes"])
    write(d, "physicalExam", [
        (1, 10, f"{VS}/HR/HR Current", "HR Current", "110"),
        (1, 10, f"{VS}/HR Rhythm/dysrhythmia/regular", "regular", ""),
        (2, 10, f"{PE}/Neurologic/GCS/Eyes Score/3", "3", ""),
        (2, 10, f"{PE}/Neurologic/GCS/Verbal Score/4", "4", ""),
        (2, 10, f"{PE}/Neurologic/GCS/Motor Score/6", "6", ""),
        (3, 10, f"{PE}/Neurologic/GCS/Eyes Score/4", "4", ""),
        (3, 10, f"{PE}/Neurologic/GCS/Verbal Score/5", "5", ""),
        (3, 10, f"{PE}/Neurologic/GCS/Motor Score/6", "6", ""),
        (3, 10, f"{VS}/HR Rhythm/sinus", "sinus", ""),
        (3, 20, f"{PE}/Extremities/extremities cool", "extremities cool", ""),
        (1, 30, f"{PE}/Neurologic/Mental Status/confused", "confused", ""),
        (3, 500, f"{PE}/Neurologic/Mental Status/confused", "confused", ""),  # too late
    ], ["patientunitstayid", "physicalexamoffset", "physicalexampath", "physicalexamvalue", "physicalexamtext"])
    write(d, "vitalPeriodic", [
        (1, 5, 100, 90, 50, 95), (1, 25, 120, None, None, 0), (1, 200, 60, 60, 30, 85),
    ], ["patientunitstayid", "observationoffset", "heartrate", "systemicsystolic", "systemicdiastolic", "sao2"])
    write(d, "vitalAperiodic", [
        (1, 10, 100, 60), (2, 90, 85, 45),  # patient 2 only in the fallback window
    ], ["patientunitstayid", "observationoffset", "noninvasivesystolic", "noninvasivediastolic"])
    write(d, "nurseCharting", [
        (2, 100, "Heart Rate", "Heart Rate", "95"),
        (3, 15, "Non-Invasive BP", "Non-Invasive BP Systolic", "130"),
        (3, 15, "Non-Invasive BP", "Non-Invasive BP Diastolic", "70"),
        (3, 15, "Heart Rate", "Heart Rate", "999"),  # implausible
        (1, 20, "Glasgow coma score", "GCS Total", "15"),
    ], ["patientunitstayid", "nursingchartoffset", "nursingchartcelltypevallabel",
        "nursingchartcelltypevalname", "nursingchartvalue"])
    write(d, "lab", [
        (1, -30, "lactate", 5.0), (1, 30, "lactate", 7.0), (1, 300, "lactate", 4.0), (1, 1000, "lactate", 2.0),
        (2, 120, "glucose", 200.0), (2, 130, "bedside glucose", 300.0), (3, 100, "bedside glucose", 150.0),
        (3, 100, "troponin - I", 12.0), (1, 100, "CPK", 400.0), (1, 2000, "CPK", 5000.0),
    ], ["patientunitstayid", "labresultoffset", "labname", "labresult"])
    flags, vitals, log = extract_admission_flags(str(d), patient)
    return d, flags, vitals, log


def test_calendar_day():
    # admitted at 23:00: midnight is offset 60, the next midnight offset 1500
    s = calendar_day(pd.Series([-30, 30, 61, 1499, 1500]), pd.Series([23 * 60] * 5))
    assert s.tolist() == [0, 0, 1, 1, 2]


@pytest.mark.parametrize("feature, positives", [
    ("p_mh_hfail_yn", {1}),
    ("p_rf_dm_yn", {1, 2, 3}),  # pastHistory NIDDM, apachePredVar, diagnosis
    ("p_mh_renf_yn", {1, 2}),
    ("p_mh_dial_yn", {2}),  # "not currently dialyzed" excluded
    ("p_mh_stroke_yn", {2}),
    ("previous_heart_complications", {1}),
    ("p_rf_dyslip_yn", {3}),
    ("p_rf_aht_yn", {3}),  # pulmonary hypertension excluded
    ("hpr_ecg_stemi_yn", {1}),  # NSTEMI excluded, patient 3 documented too late
    ("hpr_ecg_afib_y", {2}),
    ("hpr_ecg_avblock3_y", {2}),
    ("had_base_cpr24h_yn", {1, 2}),  # defibrillation before admission; APACHE dx
    ("had_base_mechvent_yn", {1, 3}),  # treatment; respiratoryCare (non-invasive excluded)
    ("hpe_proc_mechs_yn", {1}),  # IABP removal excluded
    ("renal_replacement_therapy", {3}),  # patient 1 dialysis on day 2
    ("cathecholamine_therapy", {1, 2}),  # infusion; treatment (dobutamine rate 0, phenylephrine excluded)
    ("sepsis", {1}),  # day 1; patient 2 day 2; SIRS non-infectious excluded
    ("ventricular_fibrillation", {1, 3}),  # defibrillation on day 0 (before ICU admission); diagnosis
    ("stroke", {2}),
    ("had_sy_ams_yn", {1, 2}),  # confused (first hour); GCS 13
    ("had_sy_extremity_yn", {3}),
])
def test_binary_features(mini_eicu, feature, positives):
    _, flags, _, _ = mini_eicu
    assert set(flags.index[flags[feature] == 1]) == positives
    assert flags[feature].isin([0, 1]).all()


def test_sinus_rhythm(mini_eicu):
    _, flags, _, _ = mini_eicu
    # 1: documented dysrhythmia; 2: AF -> 0; 3: documented sinus
    assert flags["hpr_ecg_sinrhy_y"].to_dict() == {1: 0, 2: 0, 3: 1}


def test_admission_vitals(mini_eicu):
    _, _, vitals, _ = mini_eicu
    # patient 1: HR 100, 120 (vitalPeriodic) + 110 (exam) in the first 30 min; 60 at 200 min ignored
    assert vitals.loc[1, "hpr_hmdyn_hr_bpm"] == 110
    # SBP 90 (invasive) and 100 (non-invasive)
    assert vitals.loc[1, "hpr_hmdyn_sbp_mmhg"] == 95
    # patient 2: only fallback values (30 min - 2 h)
    assert vitals.loc[2, "hpr_hmdyn_hr_bpm"] == 95 and vitals.loc[2, "hpr_hmdyn_sbp_mmhg"] == 85
    # patient 3: nurse-charted BP; implausible HR discarded
    assert vitals.loc[3, "hpr_hmdyn_sbp_mmhg"] == 130 and vitals.loc[3, "hpr_hmdyn_dbp_mmhg"] == 70
    assert pd.isna(vitals.loc[3, "hpr_hmdyn_hr_bpm"])


def test_labs(mini_eicu):
    d, _, _, _ = mini_eicu
    lab = load_eicu_24hs_features(str(d), [1, 2, 3])
    assert lab.loc[1, "admission_lactate"] == 6.0  # -30 and +30 min
    assert lab.loc[1, "icu_lab_lact8hpci_x"] == 4.0 and lab.loc[1, "icu_lab_lact24hpci_x"] == 2.0
    assert lab.loc[2, "glucose"] == 200.0  # laboratory glucose preferred
    assert lab.loc[3, "glucose"] == 150.0  # bedside glucose otherwise
    assert lab.loc[3, "eicu_troponin_I"] == 12.0
    assert lab.loc[1, "CPK"] == 400.0  # value after 24 h ignored


def test_coverage_report_is_aggregate(mini_eicu):
    _, flags, vitals, log = mini_eicu
    report = eicu_coverage_report(pd.concat([flags, vitals], axis=1), log, ["sepsis", "hpr_hmdyn_hr_bpm", "pbnp"])
    assert "| sepsis | <10 | <10 |" in report
    assert "| pbnp | not extracted | |" in report
    assert "sepsis with single organ dysfunction" in report  # matched strings listed
    assert "patientunitstayid" not in report


def test_first_day_minimum_vitals(mini_eicu):
    d, _, _, _ = mini_eicu
    out = first_day_minimum_vitals(str(d), [1, 2, 3])
    assert out.loc[1, "min_sbp"] == 60  # invasive and non-invasive pooled
    assert out.loc[1, "min_spo2"] == 85  # SpO2 0 % is a monitor artefact
    assert out.loc[2, "min_sbp"] == 85 and out.loc[3, "min_sbp"] == 130  # non-invasive / charted
