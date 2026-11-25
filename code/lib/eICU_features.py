"""
CULPRIT-SHOCK features from the raw eICU tables, using every table that
documents a feature, so that as few values as possible are missing.

Sources per feature (patterns in the dictionaries below):

    comorbidities        pastHistory + diagnoses of the first 24 h (+ apachePredVar for diabetes)
    STEMI, AF, AV block  diagnoses documented in the first 6 h (+ AF history, physical-exam rhythm)
    sinus rhythm         physical-exam rhythm; otherwise 0 if AF / AV block III, else 1
    resuscitation        defibrillation/CPR treatments (24 h before to 1 h after ICU admission),
                         cardiac-arrest diagnoses (first 6 h), APACHE admission diagnosis
    ventilation          respiratoryCare + ventilation treatments (first hour)
    altered mental       physical exam (neurologic findings) or first GCS < 15
    status               (physical exam or nurse charting, first hour)
    cold extremities     physical exam (first hour)
    heart rate, BP       median of vitalPeriodic (invasive), vitalAperiodic (non-invasive),
                         nurse charting and physical-exam vitals, 24 h before to 30 min after
                         ICU admission; if none, 30 min to 2 h after admission
    catecholamines, RRT, events on the admission day or the next (calendar days 0-1, as
    sepsis, VF, stroke   in CULPRIT-SHOCK): treatments, infusions, diagnoses
    mechanical support   IABP, Impella/VAD, ECMO treatments (first 24 h)

Binary features are 1 if any source documents them and 0 otherwise (absence of
documentation is taken as absence, as before). Every match is logged, so the
matched eICU strings and the number of patients per source can be checked in
the coverage report (eicu_coverage_report).

Offsets are minutes from ICU admission (eICU convention).
"""

import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

MINUTES_PER_DAY = 1440
# Admission vital signs: as in the manuscript, then a fallback window
VITALS_WINDOW = (-1440, 30)
VITALS_FALLBACK_WINDOW = (30, 120)
# Physical exam / GCS at admission
EXAM_WINDOW = (-1440, 60)
# Diagnoses that describe the presentation (STEMI, arrest, rhythm)
PRESENTATION_DX_WINDOW = (-1440, 360)
# Chronic conditions documented as diagnoses
HISTORY_DX_WINDOW = (-1440, 1440)
# Resuscitation within 24 h before admission
RESUSCITATION_WINDOW = (-1440, 60)
# Ventilation at admission
VENTILATION_WINDOW = (-100_000, 60)
# Mechanical circulatory support in the first 24 h
SUPPORT_WINDOW = (-100_000, 1440)
# Plausible ranges of the admission vitals (values outside are discarded)
PLAUSIBLE = {"heart_rate": (20, 250), "sbp": (30, 250), "dbp": (10, 200)}
VITAL_FEATURES = {"heart_rate": "hpr_hmdyn_hr_bpm", "sbp": "hpr_hmdyn_sbp_mmhg", "dbp": "hpr_hmdyn_dbp_mmhg"}

# Regular expressions (case-insensitive): name -> (pattern, exclude pattern)
PAST_HISTORY_PATTERNS = {  # on "pasthistorypath/pasthistoryvalue"
    "previous_heart_complications": (
        r"Myocardial Infarction|/MI - |procedural coronary intervention|CABG|Coronary Artery Bypass",
        None,
    ),
    "p_mh_hfail_yn": (r"Congestive Heart Failure|/CHF", None),
    "p_mh_stroke_yn": (r"/Strokes?/|/stroke", None),
    "p_mh_pad_yn": (r"peripheral vascular disease|peripheral arter", None),
    # Known renal insufficiency (GFR < 30): renal failure, or baseline creatinine >= 2
    "p_mh_renf_yn": (r"renal failure|renal insufficiency.*creatinine (2-3|3-4|4-5|> ?[2-5])", None),
    "p_mh_dial_yn": (r"dialy", r"not currently"),
    "p_rf_aht_yn": (r"hypertension requiring treatment", None),
    "p_rf_dyslip_yn": (r"cholesterol|hyperlipid|dyslipid", None),
    "p_rf_dm_yn": (r"diabetes", r"insipidus"),
    "hpr_ecg_afib_y": (r"atrial fibrillation", None),
}

DIAGNOSIS_PATTERNS = {  # on diagnosisstring: name -> (pattern, exclude, window)
    # chronic conditions
    "previous_heart_complications": (
        r"s/p (CABG|PCI|coronary|stent|angioplasty)|old myocardial infarction|coronary artery disease\|known",
        None,
        "history",
    ),
    "p_mh_pad_yn": (r"peripheral vascular disease|peripheral arterial disease", None, "history"),
    "p_rf_aht_yn": (r"^cardiovascular\|[^|]+\|hypertension", r"pulmonary", "history"),
    "p_rf_dyslip_yn": (r"hyperlipidemia|hypercholesterolemia|dyslipidemia", None, "history"),
    "p_rf_dm_yn": (r"diabetes mellitus", None, "history"),
    # presentation
    "hpr_ecg_stemi_yn": (r"with ST elevation|\bSTEMI\b", r"no ST elevation|non-ST|NSTEMI", "presentation"),
    "hpr_ecg_afib_y": (r"atrial fibrillation", None, "presentation"),
    "hpr_ecg_avblock3_y": (
        r"third degree|3rd degree|complete heart block|AV block.*complete", None, "presentation"
    ),
    "had_base_cpr24h_yn": (r"cardiac arrest", None, "presentation"),
    # events on calendar days 0-1
    "sepsis": (r"sepsis|septic", r"non-infectious|\(SIRS\)$", "day01"),
    "ventricular_fibrillation": (r"ventricular fibrillation|\bVF\b|v-fib", None, "day01"),
    "stroke": (
        r"^neurologic\|.*(stroke|cerebrovascular|cerebral infarct|intracranial hemorrhage"
        r"|intracerebral hemorrhage|subarachnoid hemorrhage)",
        None,
        "day01",
    ),
}

CATECHOLAMINES = r"norepinephrine|epinephrine|dopamine|dobutamine|isoproterenol|levophed|adrenalin"

TREATMENT_PATTERNS = {  # on treatmentstring: name -> (pattern, exclude, window)
    "had_base_cpr24h_yn": (
        r"defibrillation|cardiopulmonary resuscitation|\bCPR\b", None, "resuscitation"
    ),
    "had_base_mechvent_yn": (
        r"mechanical ventilation|endotracheal|intubation", r"non-invasive|extubation", "ventilation"
    ),
    "hpe_proc_mechs_yn": (
        r"intraaortic balloon pump|ventricular assist|impella|\bECMO\b|extracorporeal",
        r"removal",
        "support",
    ),
    "renal_replacement_therapy": (
        r"dialysis|C V V H|CVVH|CRRT|hemofiltration|ultrafiltration|\bSLED\b", None, "day01"
    ),
    "cathecholamine_therapy": (CATECHOLAMINES, None, "day01"),
    "ventricular_fibrillation": (r"defibrillation", None, "day01"),
}

PHYSICAL_EXAM_PATTERNS = {  # on "physicalexampath/physicalexamvalue", first hour
    "had_sy_ams_yn": (
        r"Neurologic.*(delusional|confus|disorient|letharg|obtund|stupor|comatose|unresponsive|somnolen)",
        None,
    ),
    "had_sy_extremity_yn": (r"(extremit|skin).*(cold|cool|clammy|mottl|dusky|cyanot)", None),
}

# nurseCharting (MIT-LCP eicu-code, Nurse Charting Table Hierarchy)
NURSE_CHARTING_VITALS = {
    "Heart Rate": "heart_rate",
    "Non-Invasive BP Systolic": "sbp",
    "Non-Invasive BP Diastolic": "dbp",
    "Invasive BP Systolic": "sbp",
    "Invasive BP Diastolic": "dbp",
}
NURSE_CHARTING_LABELS = ["Heart Rate", "Non-Invasive BP", "Invasive BP", "Glasgow coma score"]
# physicalExam "Current" vitals (physicalexamtext holds the number)
PHYSICAL_EXAM_VITALS = {
    "HR Current": "heart_rate",
    "BP (systolic) Current": "sbp",
    "BP (diastolic) Current": "dbp",
}


@dataclass
class SourceLog:
    """Which source documented which feature, for the coverage report."""

    patients: dict[tuple[str, str], set] = field(default_factory=dict)
    strings: dict[tuple[str, str], pd.Series] = field(default_factory=dict)

    def add(self, feature: str, source: str, ids: Iterable, strings: Iterable | None = None):
        """Record the patients (and, optionally, the matched string of each row)."""
        ids = list(ids)
        key = (feature, source)
        self.patients.setdefault(key, set()).update(ids)
        if strings is not None and ids:
            pairs = pd.DataFrame({"id": ids, "string": list(strings)}).drop_duplicates()
            counts = pairs["string"].value_counts()  # patients per matched string
            previous = self.strings.get(key)
            self.strings[key] = counts if previous is None else previous.add(counts, fill_value=0)


def read_optional_table(
    root_dir: str | Path,
    table: str,
    usecols: list[str],
    patients: Iterable[int] | None = None,
    row_filter: Callable[[pd.DataFrame], pd.Series] | None = None,
    chunksize: int = 5_000_000,
) -> pd.DataFrame | None:
    """
    Read an eICU table (``<table>.csv.gz``) in chunks, keeping only the
    selected patients and the rows accepted by ``row_filter``. A missing file
    or column returns None (printed), so that a source can be skipped.
    """
    path = Path(root_dir) / f"{table}.csv.gz"
    if not path.exists():
        print(f"[eICU] {path.name} not found: source skipped")
        return None
    header = set(pd.read_csv(path, nrows=0).columns)
    missing = [c for c in usecols if c not in header]
    if missing:
        print(f"[eICU] {path.name}: columns {missing} not found: source skipped")
        return None
    selected = None if patients is None else set(pd.Series(list(patients)).tolist())
    chunks = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        if selected is not None:
            chunk = chunk[chunk["patientunitstayid"].isin(selected)]
        if row_filter is not None and len(chunk):
            chunk = chunk[row_filter(chunk)]
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)


def read_table(
    root_dir: str | Path,
    table: str,
    usecols: list[str],
    patients: Iterable[int] | None = None,
    row_filter: Callable[[pd.DataFrame], pd.Series] | None = None,
) -> pd.DataFrame:
    """Like read_optional_table, but the table and columns must exist."""
    df = read_optional_table(root_dir, table, usecols, patients, row_filter)
    if df is None:
        raise FileNotFoundError(f"{table}.csv.gz (columns {usecols}) is required")
    return df


def text_match(text: pd.Series, pattern: str, exclude: str | None = None) -> pd.Series:
    """Case-insensitive regex match of a text column (missing text never matches)."""
    text = text.fillna("").astype(str)
    with warnings.catch_warnings():  # groups in the patterns are only for alternation
        warnings.filterwarnings("ignore", "This pattern is interpreted as a regular expression")
        hit = text.str.contains(pattern, case=False, regex=True)
        if exclude:
            hit &= ~text.str.contains(exclude, case=False, regex=True)
    return hit


def calendar_day(offset: pd.Series, admit_minute: pd.Series) -> pd.Series:
    """
    Calendar day relative to the ICU admission day (0 = admission day), from
    the offset (minutes) and the admission time of day (minutes after midnight).
    """
    return pd.Series(np.floor((admit_minute + offset) / MINUTES_PER_DAY), index=offset.index)


def admit_minute_of_day(patients: pd.DataFrame) -> pd.Series:
    """Minutes after midnight of the ICU admission (patient.unitadmittime24), by patient."""
    t = pd.to_timedelta(patients.set_index("patientunitstayid")["unitadmittime24"].astype(str))
    return t.dt.total_seconds() / 60


def in_window(df: pd.DataFrame, offset_col: str, window: str, admit_minute: pd.Series) -> pd.Series:
    """Rows inside a named window (see the *_WINDOW constants; "day01" = calendar days 0-1)."""
    offset = df[offset_col]
    if window == "day01":
        minute = df["patientunitstayid"].map(admit_minute)
        return calendar_day(offset, minute).isin([0, 1])
    lo, hi = {
        "history": HISTORY_DX_WINDOW,
        "presentation": PRESENTATION_DX_WINDOW,
        "resuscitation": RESUSCITATION_WINDOW,
        "ventilation": VENTILATION_WINDOW,
        "support": SUPPORT_WINDOW,
        "exam": EXAM_WINDOW,
    }[window]
    return (offset > lo) & (offset <= hi)


def _flags(
    df: pd.DataFrame,
    text: pd.Series,
    patterns: dict,
    source: str,
    log: SourceLog,
    window_mask: Callable[[str], pd.Series] | None = None,
) -> dict[str, set]:
    """Patients matching each pattern (optionally within the pattern's window)."""
    out = {}
    for name, spec in patterns.items():
        pattern, exclude = spec[0], spec[1]
        hit = text_match(text, pattern, exclude)
        if window_mask is not None and len(spec) > 2:
            hit &= window_mask(spec[2])
        ids = df.loc[hit, "patientunitstayid"]
        log.add(name, source, ids, text[hit])
        out[name] = set(ids.tolist())
    return out


def history_flags(root_dir, patients: Iterable[int], log: SourceLog) -> dict[str, set]:
    """Comorbidities from pastHistory (any offset: history is pre-admission)."""
    df = read_table(root_dir, "pastHistory", ["patientunitstayid", "pasthistorypath", "pasthistoryvalue"], patients)
    text = df["pasthistorypath"].fillna("") + "/" + df["pasthistoryvalue"].fillna("").astype(str)
    return _flags(df, text, PAST_HISTORY_PATTERNS, "pastHistory", log)


def diagnosis_flags(root_dir, patients: Iterable[int], admit_minute: pd.Series, log: SourceLog) -> dict[str, set]:
    df = read_table(root_dir, "diagnosis", ["patientunitstayid", "diagnosisoffset", "diagnosisstring"], patients)
    return _flags(
        df, df["diagnosisstring"], DIAGNOSIS_PATTERNS, "diagnosis", log,
        window_mask=lambda w: in_window(df, "diagnosisoffset", w, admit_minute),
    )


def treatment_flags(root_dir, patients: Iterable[int], admit_minute: pd.Series, log: SourceLog) -> dict[str, set]:
    df = read_table(root_dir, "treatment", ["patientunitstayid", "treatmentoffset", "treatmentstring"], patients)
    return _flags(
        df, df["treatmentstring"], TREATMENT_PATTERNS, "treatment", log,
        window_mask=lambda w: in_window(df, "treatmentoffset", w, admit_minute),
    )


def infusion_flags(root_dir, patients: Iterable[int], admit_minute: pd.Series, log: SourceLog) -> dict[str, set]:
    """Catecholamine infusions (rate > 0 when the rate is numeric) on calendar days 0-1."""
    df = read_optional_table(
        root_dir, "infusionDrug", ["patientunitstayid", "infusionoffset", "drugname", "drugrate"],
        patients,
    )
    if df is None:
        return {}
    rate = pd.to_numeric(df["drugrate"], errors="coerce")
    running = rate.isna() | (rate > 0)
    window = in_window(df, "infusionoffset", "day01", admit_minute)
    spec = {"cathecholamine_therapy": (CATECHOLAMINES, None)}
    keep = running & window
    return _flags(df[keep], df.loc[keep, "drugname"], spec, "infusionDrug", log)


def ventilation_flags(root_dir, patients: Iterable[int], log: SourceLog) -> dict[str, set]:
    """Ventilation started before or within the first hour (respiratoryCare)."""
    df = read_optional_table(root_dir, "respiratoryCare", ["patientunitstayid", "ventstartoffset"], patients)
    if df is None:
        return {}
    ids = df.loc[df["ventstartoffset"] < 60, "patientunitstayid"]
    log.add("had_base_mechvent_yn", "respiratoryCare", ids)
    return {"had_base_mechvent_yn": set(ids.tolist())}


def apache_flags(root_dir, patients: Iterable[int], patient_table: pd.DataFrame, log: SourceLog) -> dict[str, set]:
    """Diabetes (apachePredVar) and cardiac arrest as APACHE admission diagnosis (patient)."""
    out: dict[str, set] = {}
    pred = read_optional_table(root_dir, "apachePredVar", ["patientunitstayid", "diabetes"], patients)
    if pred is not None:
        ids = pred.loc[pd.to_numeric(pred["diabetes"], errors="coerce") == 1, "patientunitstayid"]
        log.add("p_rf_dm_yn", "apachePredVar", ids)
        out["p_rf_dm_yn"] = set(ids.tolist())
    if "apacheadmissiondx" in patient_table:
        hit = text_match(patient_table["apacheadmissiondx"], r"cardiac arrest")
        ids = patient_table.loc[hit, "patientunitstayid"]
        log.add("had_base_cpr24h_yn", "patient.apacheadmissiondx", ids, patient_table.loc[hit, "apacheadmissiondx"])
        out["had_base_cpr24h_yn"] = set(ids.tolist())
    return out


def physical_exam(root_dir, patients: Iterable[int], log: SourceLog) -> tuple[dict[str, set], pd.DataFrame]:
    """
    Admission physical exam (first hour): altered mental status, cold
    extremities, rhythm (sinus / dysrhythmia), GCS and "Current" vitals.

    Returns the flags and a long table of measurements
    (patientunitstayid, offset, variable, value) with variables heart_rate,
    sbp, dbp, gcs and sinus (1/0).
    """
    df = read_table(
        root_dir, "physicalExam",
        ["patientunitstayid", "physicalexamoffset", "physicalexampath", "physicalexamvalue", "physicalexamtext"],
        patients,
        row_filter=lambda c: c["physicalexamoffset"].between(EXAM_WINDOW[0] + 1, VITALS_FALLBACK_WINDOW[1]),
    )
    exam = df[df["physicalexamoffset"] <= EXAM_WINDOW[1]]
    text = exam["physicalexampath"].fillna("") + "/" + exam["physicalexamvalue"].fillna("").astype(str)
    flags = _flags(exam, text, PHYSICAL_EXAM_PATTERNS, "physicalExam", log)

    rows = []
    # Vitals ("HR Current", ...), up to the end of the fallback window
    vit = df[df["physicalexamvalue"].isin(PHYSICAL_EXAM_VITALS)]
    rows.append(pd.DataFrame({
        "patientunitstayid": vit["patientunitstayid"],
        "offset": vit["physicalexamoffset"],
        "variable": vit["physicalexamvalue"].map(PHYSICAL_EXAM_VITALS),
        "value": pd.to_numeric(vit["physicalexamtext"], errors="coerce"),
        "source": "physicalExam",
    }))
    # GCS = eyes + verbal + motor of the same exam
    gcs = exam[exam["physicalexampath"].str.contains(r"/GCS/(?:Eyes|Verbal|Motor) Score/", na=False)]
    if len(gcs):
        comp = gcs.assign(
            part=gcs["physicalexampath"].str.extract(r"/GCS/(Eyes|Verbal|Motor) Score/")[0],
            score=pd.to_numeric(gcs["physicalexamvalue"], errors="coerce"),
        ).pivot_table(
            index=["patientunitstayid", "physicalexamoffset"], columns="part", values="score", aggfunc="min"
        )
        # total only for exams with the three components
        total = comp.reindex(columns=["Eyes", "Verbal", "Motor"]).dropna().sum(axis=1).reset_index(name="value")
        rows.append(pd.DataFrame({
            "patientunitstayid": total["patientunitstayid"], "offset": total["physicalexamoffset"],
            "variable": "gcs", "value": total["value"], "source": "physicalExam",
        }))
    # Rhythm
    rhythm = exam[exam["physicalexampath"].str.contains(r"/HR Rhythm/", na=False)]
    if len(rhythm):
        sinus = np.where(rhythm["physicalexampath"].str.contains(r"/HR Rhythm/sinus", case=False), 1.0,
                         np.where(rhythm["physicalexampath"].str.contains(r"/HR Rhythm/dysrhythmia", case=False), 0.0, np.nan))
        rows.append(pd.DataFrame({
            "patientunitstayid": rhythm["patientunitstayid"], "offset": rhythm["physicalexamoffset"],
            "variable": "sinus", "value": sinus, "source": "physicalExam",
        }))
    return flags, pd.concat(rows, ignore_index=True).dropna(subset=["value"])


def charted_vitals(root_dir, patients: Iterable[int]) -> pd.DataFrame:
    """
    Heart rate, systolic/diastolic BP and GCS from vitalPeriodic (invasive),
    vitalAperiodic (non-invasive) and nurseCharting, from 24 h before to the
    end of the fallback window, as a long table like physical_exam's.
    """
    lo, hi = VITALS_WINDOW[0], VITALS_FALLBACK_WINDOW[1]
    rows = []

    def window(c, col="observationoffset"):
        return (c[col] > lo) & (c[col] <= hi)

    vp = read_table(
        root_dir, "vitalPeriodic",
        ["patientunitstayid", "observationoffset", "heartrate", "systemicsystolic", "systemicdiastolic"],
        patients, row_filter=window,
    )
    for col, var in [("heartrate", "heart_rate"), ("systemicsystolic", "sbp"), ("systemicdiastolic", "dbp")]:
        rows.append(pd.DataFrame({
            "patientunitstayid": vp["patientunitstayid"], "offset": vp["observationoffset"],
            "variable": var, "value": vp[col], "source": "vitalPeriodic",
        }))
    va = read_optional_table(
        root_dir, "vitalAperiodic",
        ["patientunitstayid", "observationoffset", "noninvasivesystolic", "noninvasivediastolic"],
        patients, row_filter=window,
    )
    if va is not None:
        for col, var in [("noninvasivesystolic", "sbp"), ("noninvasivediastolic", "dbp")]:
            rows.append(pd.DataFrame({
                "patientunitstayid": va["patientunitstayid"], "offset": va["observationoffset"],
                "variable": var, "value": va[col], "source": "vitalAperiodic",
            }))
    nc = read_optional_table(
        root_dir, "nurseCharting",
        ["patientunitstayid", "nursingchartoffset", "nursingchartcelltypevallabel",
         "nursingchartcelltypevalname", "nursingchartvalue"],
        patients,
        row_filter=lambda c: window(c, "nursingchartoffset")
        & c["nursingchartcelltypevallabel"].isin(NURSE_CHARTING_LABELS),
    )
    if nc is not None:
        name = nc["nursingchartcelltypevalname"]
        variable = name.map(NURSE_CHARTING_VITALS).where(name != "GCS Total", "gcs")
        rows.append(pd.DataFrame({
            "patientunitstayid": nc["patientunitstayid"], "offset": nc["nursingchartoffset"],
            "variable": variable, "value": pd.to_numeric(nc["nursingchartvalue"], errors="coerce"),
            "source": "nurseCharting",
        }))
    out = pd.concat(rows, ignore_index=True)
    return out.dropna(subset=["variable", "value"])


def admission_vitals(measurements: pd.DataFrame, log: SourceLog) -> pd.DataFrame:
    """
    Median heart rate, systolic and diastolic BP per patient over all sources
    in VITALS_WINDOW; patients without any value there use VITALS_FALLBACK_WINDOW.
    Implausible values (PLAUSIBLE) are discarded first.
    """
    m = measurements[measurements["variable"].isin(list(PLAUSIBLE))].copy()
    lo_hi = m["variable"].map(PLAUSIBLE)
    m = m[[lo <= v <= hi for v, (lo, hi) in zip(m["value"], lo_hi)]]
    out = {}
    for var in PLAUSIBLE:
        v = m[m["variable"] == var]
        primary = v[(v["offset"] > VITALS_WINDOW[0]) & (v["offset"] <= VITALS_WINDOW[1])]
        fallback = v[(v["offset"] > VITALS_FALLBACK_WINDOW[0]) & (v["offset"] <= VITALS_FALLBACK_WINDOW[1])]
        value = primary.groupby("patientunitstayid")["value"].median()
        late = fallback.groupby("patientunitstayid")["value"].median()
        late = late[~late.index.isin(value.index)]
        feature = VITAL_FEATURES[var]
        for src, grp in primary.groupby("source"):
            log.add(feature, f"{src} (first 30 min)", grp["patientunitstayid"].unique())
        for src, grp in fallback[fallback["patientunitstayid"].isin(late.index)].groupby("source"):
            log.add(feature, f"{src} (30 min-2 h, fallback)", grp["patientunitstayid"].unique())
        out[feature] = pd.concat([value, late])
    return pd.DataFrame(out)


def first_value(measurements: pd.DataFrame, variable: str, window: tuple[int, int]) -> pd.Series:
    """Earliest value of a variable per patient within (lo, hi]."""
    v = measurements[(measurements["variable"] == variable)
                     & (measurements["offset"] > window[0]) & (measurements["offset"] <= window[1])]
    return v.sort_values("offset").groupby("patientunitstayid")["value"].first()


def extract_admission_flags(root_dir, patient_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, SourceLog]:
    """
    All binary features (admission and day 0-1) and the admission vitals for
    the patients of ``patient_table`` (needs patientunitstayid and
    unitadmittime24; apacheadmissiondx is used if present).

    Returns (flags, vitals, log): flags is 0/1 per CULPRIT-SHOCK feature,
    vitals has had_* vital-sign features; both indexed by patientunitstayid.
    """
    ids = patient_table["patientunitstayid"]
    admit_minute = admit_minute_of_day(patient_table)
    log = SourceLog()

    sources = [
        history_flags(root_dir, ids, log),
        diagnosis_flags(root_dir, ids, admit_minute, log),
        treatment_flags(root_dir, ids, admit_minute, log),
        infusion_flags(root_dir, ids, admit_minute, log),
        ventilation_flags(root_dir, ids, log),
        apache_flags(root_dir, ids, patient_table, log),
    ]
    exam_flags, exam_measurements = physical_exam(root_dir, ids, log)
    sources.append(exam_flags)
    measurements = pd.concat([exam_measurements, charted_vitals(root_dir, ids)], ignore_index=True)

    # GCS < 15 at admission -> altered mental status
    gcs = first_value(measurements, "gcs", EXAM_WINDOW)
    low_gcs = gcs[gcs < 15].index
    log.add("had_sy_ams_yn", "GCS < 15 (first hour)", low_gcs)
    sources.append({"had_sy_ams_yn": set(low_gcs.tolist())})

    names = sorted({name for src in sources for name in src})
    flags = pd.DataFrame(0, index=pd.Index(ids.to_numpy(), name="patientunitstayid"), columns=names)
    for src in sources:
        for name, patients in src.items():
            flags.loc[flags.index.isin(patients), name] = 1

    # Sinus rhythm: physical-exam rhythm if documented, otherwise sinus unless
    # atrial fibrillation or complete AV block is documented
    rhythm = first_value(measurements, "sinus", EXAM_WINDOW)
    no_sinus = (flags["hpr_ecg_afib_y"] == 1) | (flags["hpr_ecg_avblock3_y"] == 1)
    sinus = pd.Series(np.where(no_sinus, 0, 1), index=flags.index)
    documented = rhythm.reindex(flags.index)
    flags["hpr_ecg_sinrhy_y"] = documented.fillna(sinus).astype(int)
    log.add("hpr_ecg_sinrhy_y", "physicalExam rhythm", documented.dropna().index)
    assumed = documented.isna() & ~no_sinus
    log.add("hpr_ecg_sinrhy_y", "assumed (no AF/AV block documented)", assumed[assumed].index)

    vitals = admission_vitals(measurements, log).reindex(index=flags.index, columns=list(VITAL_FEATURES.values()))
    return flags, vitals, log


def eicu_coverage_report(X: pd.DataFrame, log: SourceLog, features: list[str], min_group: int = 10) -> str:
    """
    Aggregate report: availability of every feature in the eICU cohort, the
    number of patients documented by each source and the matched eICU strings
    (counts < min_group shown as "<min_group"). No patient-level values.
    """
    def count(n):
        return str(n) if n >= min_group or n == 0 else f"<{min_group}"

    lines = [f"# eICU feature coverage ({len(X)} patients)", "",
             "| feature | available | positive / median |", "|---|---|---|"]
    for f in features:
        if f not in X:
            lines.append(f"| {f} | not extracted | |")
            continue
        col = X[f]
        binary = col.dropna().isin([0, 1]).all()
        summary = count(int(col.sum())) if binary else (f"{col.median():.4g}" if col.notna().any() else "")
        lines.append(f"| {f} | {count(int(col.notna().sum()))} | {summary} |")
    lines += ["", "## Sources (patients documented per source)", "", "| feature | source | patients |", "|---|---|---|"]
    for (feature, source), patients in sorted(log.patients.items()):
        lines.append(f"| {feature} | {source} | {count(len(patients))} |")
    lines += ["", "## Matched eICU strings (patients)", ""]
    for (feature, source), counts in sorted(log.strings.items()):
        lines.append(f"### {feature} - {source}")
        for string, n in counts.sort_values(ascending=False).items():
            lines.append(f"- `{string}`: {count(int(n))}")
        lines.append("")
    return "\n".join(lines) + "\n"


def first_day_minimum_vitals(root_dir, patients: Iterable[int]) -> pd.DataFrame:
    """
    Lowest systolic BP and SpO2 per patient from 24 h before to 24 h after ICU
    admission (BOS,MA2 score), over invasive and non-invasive BP (vitalPeriodic,
    vitalAperiodic, nurseCharting) and monitor or charted SpO2. Implausible
    values (monitor artefacts) are discarded: SBP outside 30-250 mmHg, SpO2
    outside 50-100 %.

    Returns a DataFrame indexed by patientunitstayid with min_sbp and min_spo2.
    """
    lo, hi = -1440, 1440

    def window(c, col="observationoffset"):
        return (c[col] > lo) & (c[col] < hi)

    parts = []
    vp = read_table(root_dir, "vitalPeriodic",
                    ["patientunitstayid", "observationoffset", "systemicsystolic", "sao2"], patients, window)
    parts += [vp.assign(variable="sbp", value=vp["systemicsystolic"]),
              vp.assign(variable="spo2", value=vp["sao2"])]
    va = read_optional_table(root_dir, "vitalAperiodic",
                             ["patientunitstayid", "observationoffset", "noninvasivesystolic"], patients, window)
    if va is not None:
        parts.append(va.assign(variable="sbp", value=va["noninvasivesystolic"]))
    nc = read_optional_table(
        root_dir, "nurseCharting",
        ["patientunitstayid", "nursingchartoffset", "nursingchartcelltypevalname", "nursingchartvalue"],
        patients,
        lambda c: window(c, "nursingchartoffset") & c["nursingchartcelltypevalname"].isin(
            ["Non-Invasive BP Systolic", "Invasive BP Systolic", "O2 Saturation"]),
    )
    if nc is not None:
        variable = np.where(nc["nursingchartcelltypevalname"] == "O2 Saturation", "spo2", "sbp")
        parts.append(nc.assign(variable=variable, value=pd.to_numeric(nc["nursingchartvalue"], errors="coerce")))
    m = pd.concat([p.loc[:, ["patientunitstayid", "variable", "value"]] for p in parts], ignore_index=True)
    plausible = np.where(m["variable"] == "sbp", m["value"].between(30, 250), m["value"].between(50, 100))
    m = m[plausible]
    out = m.pivot_table(index="patientunitstayid", columns="variable", values="value", aggfunc="min")
    return out.reindex(columns=["sbp", "spo2"]).rename(columns={"sbp": "min_sbp", "spo2": "min_spo2"})
