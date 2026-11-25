"""
Synthetic eICU-like tables (random values, real eICU strings and columns) for
dry runs of the extraction and the pipeline without patient data.

    uv run python tests/synthetic_eicu.py <output folder> [n_patients]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PH = "notes/Progress Notes/Past History/Organ Systems"
PE = "notes/Progress Notes/Physical Exam/Physical Exam"
VS = f"{PE}/Constitutional/Vital Sign and Physiological Data"

HISTORY = [
    (f"{PH}/Cardiovascular (R)/Hypertension Requiring Treatment/hypertension requiring treatment", "hypertension requiring treatment"),
    (f"{PH}/Cardiovascular (R)/Myocardial Infarction/MI - remote", "MI - remote"),
    (f"{PH}/Cardiovascular (R)/Congestive Heart Failure/CHF", "CHF"),
    (f"{PH}/Endocrine (R)/Insulin Dependent Diabetes/insulin dependent diabetes", "insulin dependent diabetes"),
    (f"{PH}/Endocrine (R)/Non-Insulin Dependent Diabetes/medication dependent", "medication dependent"),
    (f"{PH}/Cardiovascular (R)/Arrhythmias/atrial fibrillation - chronic", "atrial fibrillation - chronic"),
    (f"{PH}/Cardiovascular (R)/Peripheral Vascular Disease/peripheral vascular disease", "peripheral vascular disease"),
    (f"{PH}/Renal  (R)/Renal Failure/renal failure- not currently dialyzed", "renal failure- not currently dialyzed"),
    (f"{PH}/Neurologic/Strokes/stroke - date unknown", "stroke - date unknown"),
    (f"{PH}/Pulmonary/COPD/COPD - severe", "COPD - severe"),
]
DIAGNOSES = [
    ("cardiovascular|chest pain / ASHD|acute coronary syndrome|acute myocardial infarction (with ST elevation)", (0, 300)),
    ("cardiovascular|chest pain / ASHD|acute coronary syndrome|acute myocardial infarction (no ST elevation)", (0, 300)),
    ("cardiovascular|arrhythmias|atrial fibrillation", (0, 300)),
    ("cardiovascular|cardiac arrest|cardiac arrest|un-witnessed", (0, 300)),
    ("cardiovascular|shock / hypotension|septic shock", (0, 2500)),
    ("cardiovascular|arrhythmias|ventricular fibrillation", (0, 2500)),
    ("neurologic|disorders of vasculature|stroke|ischemic stroke", (0, 2500)),
    ("endocrine|glucose metabolism|diabetes mellitus|Type II", (0, 1000)),
    ("cardiovascular|chest pain / ASHD|hyperlipidemia", (0, 1000)),
    ("pulmonary|respiratory failure", (0, 1000)),
]
TREATMENTS = [
    ("cardiovascular|arrhythmias|cardiac defibrillation", (-600, 60)),
    ("pulmonary|ventilation and oxygenation|mechanical ventilation", (-60, 120)),
    ("cardiovascular|ventricular dysfunction|intraaortic balloon pump", (0, 1400)),
    ("cardiovascular|ventricular dysfunction|intraaortic balloon pump|removal", (0, 3000)),
    ("renal|dialysis|C V V H D", (0, 3000)),
    ("cardiovascular|shock|vasopressors|norepinephrine > 0.1 micrograms/kg/min", (0, 1500)),
    ("cardiovascular|shock|inotropic agent|dobutamine 5-10 micrograms/kg/min", (0, 1500)),
]
LABS = {
    "lactate": ("mmol/L", 1, 12), "BNP": ("pg/mL", 50, 3000), "CPK": ("Units/L", 50, 5000),
    "troponin - T": ("ng/mL", 0.01, 10), "troponin - I": ("ng/mL", 0.01, 50),
    "creatinine": ("mg/dL", 0.5, 4), "WBC x 1000": ("K/mcL", 4, 25), "Hct": ("%", 25, 50),
    "CRP": ("mg/dL", 0.1, 30), "PT - INR": ("ratio", 0.9, 3), "glucose": ("mg/dL", 70, 400),
    "bedside glucose": ("mg/dL", 70, 400), "ALT (SGPT)": ("Units/L", 10, 900),
    "triglycerides": ("mg/dL", 60, 400), "HDL": ("mg/dL", 20, 70), "LDL": ("mg/dL", 50, 220),
    "anion gap": ("mmol/L", 5, 25), "BUN": ("mg/dL", 5, 80),
}


def write_synthetic_eicu(folder: Path, n: int = 150, seed: int = 1) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ids = np.arange(1001, 1001 + n)

    def save(name, df):
        df.to_csv(folder / f"{name}.csv.gz", index=False)

    def rows(k):
        return rng.choice(ids, k)

    save("patient", pd.DataFrame({
        "patientunitstayid": ids,
        "uniquepid": [f"00{i % (n - 5)}" for i in range(n)],  # a few patients with 2 stays
        "unittype": rng.choice(["Cardiac ICU", "MICU", "Cardiac ICU", "Cardiac ICU"], n),
        "gender": rng.choice(["Female", "Male"], n),
        "age": rng.choice(["55", "67", "> 89", "72", "80"], n),
        "admissionheight": rng.normal(172, 9, n).round(1),
        "admissionweight": rng.normal(82, 14, n).round(1),
        "hospitaldischargestatus": rng.choice(["Alive", "Expired"], n),
        "hospitaladmitoffset": -rng.integers(0, 3000, n),
        "hospitaldischargeoffset": rng.integers(60, 80000, n),
        "unitdischargeoffset": rng.integers(60, 20000, n),
        "unitdischargestatus": rng.choice(["Alive", "Expired"], n),
        "unitadmittime24": [f"{h:02d}:{m:02d}:00" for h, m in zip(rng.integers(0, 24, n), rng.integers(0, 60, n))],
        "apacheadmissiondx": rng.choice(["Shock, cardiogenic", "Infarction, acute myocardial (MI)",
                                         "Cardiac arrest (with or without respiratory arrest)"], n),
    }))
    dx: list[tuple] = [(i, "785.51, R57.0", "cardiovascular|shock / hypotension|shock|cardiogenic", 30) for i in ids]
    for _ in range(4 * n):
        string, (lo, hi) = DIAGNOSES[rng.integers(len(DIAGNOSES))]
        dx.append((int(rng.choice(ids)), np.nan, string, int(rng.integers(lo, hi))))
    save("diagnosis", pd.DataFrame(dx, columns=["patientunitstayid", "icd9code", "diagnosisstring", "diagnosisoffset"]))
    hist = [HISTORY[k] for k in rng.integers(len(HISTORY), size=3 * n)]
    save("pastHistory", pd.DataFrame({"patientunitstayid": rows(3 * n),
                                      "pasthistorypath": [h[0] for h in hist], "pasthistoryvalue": [h[1] for h in hist]}))
    tr = [TREATMENTS[k] for k in rng.integers(len(TREATMENTS), size=3 * n)]
    save("treatment", pd.DataFrame({"patientunitstayid": rows(3 * n),
                                    "treatmentoffset": [int(rng.integers(lo, hi)) for _, (lo, hi) in tr],
                                    "treatmentstring": [t[0] for t in tr]}))
    save("infusionDrug", pd.DataFrame({
        "patientunitstayid": rows(n), "infusionoffset": rng.integers(0, 2000, n),
        "drugname": rng.choice(["Norepinephrine (mcg/min)", "Dobutamine (mcg/kg/min)", "Phenylephrine (mcg/min)"], n),
        "drugrate": rng.choice(["0", "5", "10"], n),
    }))
    save("respiratoryCare", pd.DataFrame({"patientunitstayid": rows(n // 3), "ventstartoffset": rng.integers(-100, 500, n // 3)}))
    save("apachePredVar", pd.DataFrame({"patientunitstayid": ids, "diabetes": rng.integers(0, 2, n)}))
    pe = []
    for i in ids:
        off = int(rng.integers(-20, 40))
        pe.append((i, off, f"{VS}/HR/HR Current", "HR Current", str(int(rng.normal(95, 20)))))
        pe.append((i, off, f"{VS}/BP (systolic)/BP (systolic) Current", "BP (systolic) Current", str(int(rng.normal(95, 20)))))
        pe.append((i, off, f"{VS}/BP (diastolic)/BP (diastolic) Current", "BP (diastolic) Current", str(int(rng.normal(55, 10)))))
        pe.append((i, off, f"{VS}/HR Rhythm/" + rng.choice(["sinus", "dysrhythmia/regular"]), "x", ""))
        for part, top in [("Eyes", 4), ("Verbal", 5), ("Motor", 6)]:
            score = int(rng.integers(1, top + 1))
            pe.append((i, off, f"{PE}/Neurologic/GCS/{part} Score/{score}", str(score), ""))
        if rng.random() < 0.4:
            pe.append((i, off, f"{PE}/Extremities/extremities cool", "extremities cool", ""))
        if rng.random() < 0.2:
            pe.append((i, off, f"{PE}/Neurologic/Mental Status/delusional", "delusional", ""))
    save("physicalExam", pd.DataFrame(pe, columns=["patientunitstayid", "physicalexamoffset", "physicalexampath",
                                                   "physicalexamvalue", "physicalexamtext"]))
    vp = pd.DataFrame({"patientunitstayid": np.repeat(ids, 20), "observationoffset": np.tile(np.arange(-100, 100, 10), n),
                       "heartrate": rng.normal(95, 20, 20 * n).round(), "systemicsystolic": rng.normal(95, 25, 20 * n).round(),
                       "systemicdiastolic": rng.normal(55, 15, 20 * n).round(), "sao2": rng.normal(93, 5, 20 * n).round()})
    vp.loc[rng.random(len(vp)) < 0.6, ["heartrate", "systemicsystolic", "systemicdiastolic"]] = np.nan
    save("vitalPeriodic", vp)
    save("vitalAperiodic", pd.DataFrame({
        "patientunitstayid": np.repeat(ids, 5), "observationoffset": np.tile(np.arange(-20, 80, 20), n),
        "noninvasivesystolic": rng.normal(100, 20, 5 * n).round(), "noninvasivediastolic": rng.normal(60, 12, 5 * n).round()}))
    nc = []
    for i in rows(2 * n):
        off = int(rng.integers(-60, 120))
        nc.append((i, off, "Heart Rate", "Heart Rate", str(int(rng.normal(95, 20)))))
        nc.append((i, off, "Non-Invasive BP", "Non-Invasive BP Systolic", str(int(rng.normal(100, 20)))))
        nc.append((i, off, "Glasgow coma score", "GCS Total", rng.choice(["15", "12", "Unable to score due to medication"])))
    save("nurseCharting", pd.DataFrame(nc, columns=["patientunitstayid", "nursingchartoffset", "nursingchartcelltypevallabel",
                                                    "nursingchartcelltypevalname", "nursingchartvalue"]))
    lab = []
    for _ in range(25 * n):
        name = list(LABS)[rng.integers(len(LABS))]
        unit, lo, hi = LABS[name]
        lab.append((int(rng.choice(ids)), int(rng.integers(-3000, 2000)), name, round(float(rng.uniform(lo, hi)), 2), unit))
    save("lab", pd.DataFrame(lab, columns=["patientunitstayid", "labresultoffset", "labname", "labresult", "labmeasurenamesystem"]))
    return folder


if __name__ == "__main__":
    out = write_synthetic_eicu(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 150)
    print("synthetic eICU written to", out)
