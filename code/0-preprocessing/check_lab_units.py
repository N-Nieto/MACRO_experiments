"""
Aggregate-only report of laboratory units and value ranges in CULPRIT-SHOCK
and eICU, used to verify the unit harmonization (lib/unit_harmonization.py and
lib/eICU_processing.py).

Only counts and quantiles (5/25/50/75/95 %) are written; groups with fewer than
MIN_GROUP values report their count only. No patient-level values are written.

Run from this folder:
    uv run python check_lab_units.py [--culprit-dir PATH] [--eicu-dir PATH]
Output: output/rev2/data_checks/lab_units_report.md (lib/paths.OUTPUT_DIR)
"""

# %%
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_DIR / "code"))
from lib.data_load_utils import load_CULPRIT_data  # noqa: E402
from lib.experiment_definitions import get_features  # noqa: E402
from lib.paths import CULPRIT_DIR, EICU_DIR, OUTPUT_DIR  # noqa: E402
from lib.utils import ensure_dir  # noqa: E402

MIN_GROUP = 10
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]

# CULPRIT-SHOCK raw lab columns that feed the Full model (and the CLIP score)
CULPRIT_RAW_LABS = [
    "icu_lab_ck_x", "ck",
    "icu_lab_lactprepci_x", "icu_lab_lactpopci_x", "admission_lactate",
    "icu_lab_lact8hpci_x", "icu_lab_lact16hpci_x", "icu_lab_lact24hpci_x", "lac",
    "icu_lab_crea_x", "cre", "icu_lab_wct_x_x", "icu_lab_hct_x_x",
    "icu_lab_crp_x", "icu_lab_glc_x", "gluc",
    "pbnp", "tnt", "alat", "icu_lab_inr_r", "cysc_s_1", "il_6",
]
# eICU lab names used by the Full model (+ lipids and BOS,MA2 labs)
EICU_LABS = [
    "lactate", "BNP", "CPK", "troponin - T", "creatinine", "WBC x 1000", "Hct",
    "CRP", "PT - INR", "glucose", "ALT (SGPT)", "triglycerides", "HDL", "LDL",
    "anion gap", "BUN",
]
# Also list any eICU lab name that looks like an alternative of the above
EICU_SEARCH = r"bnp|crp|tropo|cpk|\bck\b|lact|creat|alt|ast|inr|gluc|hct|wbc|lipid|chol"


def describe(values: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    row: dict[str, float] = {"n": len(values)}
    if len(values) >= MIN_GROUP:
        row.update({f"q{int(q * 100)}": float(values.quantile(q)) for q in QUANTILES})
    return row


def table(rows: list[dict], index_names: list[str]) -> str:
    """Markdown table without extra dependencies."""
    if not rows:
        return "_(nothing to report)_\n"
    df = pd.DataFrame(rows)
    cols = index_names + ["n"] + [f"q{int(q * 100)}" for q in QUANTILES]
    df = df.reindex(columns=cols)
    fmt = lambda v: "" if pd.isna(v) else (f"{v:.4g}" if isinstance(v, float) else str(v))  # noqa: E731
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def culprit_section(culprit_dir: Path) -> str:
    out = ["## CULPRIT-SHOCK\n"]
    sheets = pd.read_excel(culprit_dir / "CULPRIT-data_20210407.xlsx", sheet_name=None, nrows=0)
    origin: dict[str, list[str]] = {}
    for sheet, df in sheets.items():
        for col in df.columns:
            origin.setdefault(str(col).lower(), []).append(sheet)

    raw = load_CULPRIT_data(culprit_dir, harmonize=False)
    unit_cols = [c for c in raw.columns if "unit" in c.lower()]

    out.append("### Unit-code columns (value counts)\n")
    for c in unit_cols:
        counts = raw[c].value_counts(dropna=False).to_dict()
        src = origin.get(re.sub(r"_[xy]$", "", c).lower(), ["?"])
        out.append(f"- `{c}` (sheet: {', '.join(src)}): {counts}\n")

    out.append("\n### Raw values per unit code\n")
    rows = []
    for ucol in unit_cols:
        stem = ucol.lower().split("unit")[0]
        value_cols = [
            c for c in raw.columns
            if c.lower().startswith(stem) and "unit" not in c.lower() and c != ucol
        ]
        if stem == "icu_lab_lact":
            value_cols.append("admission_lactate")
        for vcol in value_cols:
            for code, grp in raw.groupby(raw[ucol].astype("string").fillna("NA")):
                rows.append({"unit column": ucol, "value column": vcol, "code": code, **describe(grp[vcol])})
    out.append(table(rows, ["unit column", "value column", "code"]))

    out.append("\n### Lactate pooled over all time points, per unit code\n")
    lact_cols = [c for c in raw.columns if re.fullmatch(r"icu_lab_lact(prepci|popci|\d+hpci)_x", c)]
    code_col = "icu_lab_lactunit_c_x"
    if lact_cols and code_col in raw:
        long = raw[[code_col] + lact_cols].melt(id_vars=code_col, value_name="value")
        rows = [{"code": code, **describe(grp["value"])}
                for code, grp in long.groupby(long[code_col].astype("string").fillna("NA"))]
        out.append(table(rows, ["code"]))

    out.append("\n### Raw lab columns (all codes pooled)\n")
    rows = [{"column": c, "sheet": ", ".join(origin.get(c.lower(), ["derived/merged"])), **describe(raw[c])}
            for c in CULPRIT_RAW_LABS if c in raw.columns]
    missing = [c for c in CULPRIT_RAW_LABS if c not in raw.columns]
    out.append(table(rows, ["column", "sheet"]))
    if missing:
        out.append(f"\nNot found: {missing}\n")

    harmonized = load_CULPRIT_data(culprit_dir, harmonize=True)
    labs = [f for f in get_features("24hs") if f not in get_features("Admission")]
    out.append("\n### Model features after harmonization (Full model labs)\n")
    out.append(table([{"feature": f, **describe(harmonized[f])} for f in labs if f in harmonized], ["feature"]))
    out.append(f"\n`hpe_proc_mechs_yn` counts: {harmonized['hpe_proc_mechs_yn'].value_counts(dropna=False).to_dict()}\n")
    return "".join(out)


def eicu_section(eicu_dir: Path) -> str:
    out = ["## eICU\n"]
    outcomes_file = eicu_dir / "preprocessed_MACRO" / "outcomes_CICU.csv"
    cohort = None
    if outcomes_file.exists():
        cohort = set(pd.read_csv(outcomes_file, usecols=["patientunitstayid"])["patientunitstayid"])
        out.append(f"Restricted to the {len(cohort)} cohort patients in outcomes_CICU.csv, "
                   "labs from 24 h before to 24 h after ICU admission.\n\n")
    usecols = ["patientunitstayid", "labresultoffset", "labname", "labresult", "labmeasurenamesystem"]
    chunks = []
    for chunk in pd.read_csv(eicu_dir / "lab.csv.gz", usecols=usecols, chunksize=5_000_000):
        if cohort is not None:
            chunk = chunk[chunk["patientunitstayid"].isin(cohort)]
        chunks.append(chunk[chunk["labresultoffset"].between(-1440, 1440)])
    lab = pd.concat(chunks, ignore_index=True)

    found = sorted(n for n in lab["labname"].dropna().unique() if re.search(EICU_SEARCH, n, re.I))
    out.append("### Lab names matching the search (n results)\n")
    counts = lab["labname"].value_counts()
    out.append("".join(f"- `{n}`: {counts[n]}\n" for n in found))

    out.append("\n### Values per lab name and reported unit\n")
    rows = []
    for name in EICU_LABS:
        sub = lab[lab["labname"] == name]
        for unit, grp in sub.groupby(sub["labmeasurenamesystem"].fillna("NA")):
            rows.append({"labname": name, "unit": unit, **describe(grp["labresult"])})
    out.append(table(rows, ["labname", "unit"]))

    x_full = eicu_dir / "preprocessed_MACRO" / "X_Full_CICU.csv"
    if x_full.exists():
        X = pd.read_csv(x_full, index_col=0)
        labs = [f for f in get_features("24hs") if f not in get_features("Admission")]
        out.append("\n### Model features after harmonization (X_Full_CICU.csv)\n")
        out.append(table([{"feature": f, **describe(X[f])} for f in labs if f in X], ["feature"]))
    return "".join(out)


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--culprit-dir", type=Path, default=CULPRIT_DIR)
    parser.add_argument("--eicu-dir", type=Path, default=EICU_DIR)
    args = parser.parse_args()

    report = ["# Lab units report (aggregates only)\n\n",
              f"Groups with fewer than {MIN_GROUP} values show the count only.\n\n"]
    for name, func, path in [("CULPRIT-SHOCK", culprit_section, args.culprit_dir), ("eICU", eicu_section, args.eicu_dir)]:
        try:
            report.append(func(path))
        except Exception as err:  # keep the other section if one dataset is unavailable
            report.append(f"## {name}\n\nFailed: {type(err).__name__}: {err}\n")
        report.append("\n")

    save_dir = OUTPUT_DIR / "data_checks"
    ensure_dir(save_dir)
    (save_dir / "lab_units_report.md").write_text("".join(report))
    print("".join(report))
    print(f"Written to {save_dir / 'lab_units_report.md'}")
