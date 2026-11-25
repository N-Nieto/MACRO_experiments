"""
Unit harmonization of the CULPRIT-SHOCK laboratory values.

The meaning of each unit code was identified from the value distribution per
code (code/data_checks/check_lab_units.py -> output/data_checks/
lab_units_report.md). The eICU values are converted to the same target units
in lib.eICU_processing (EICU_UNIT_FACTORS).

Unit codes without a documented unit ("infer") are resolved for the whole code
group: among the candidate factors, the one that brings the group's median
closest (log scale) to the median of the values with a known unit is used
(the decision is printed). Codes not listed at all are set to NaN.
"""

import numpy as np
import pandas as pd

# Harmonized (target) unit of every laboratory feature of the Full model
TARGET_UNITS = {
    "admission_lactate": "mmol/L",
    "icu_lab_lact8hpci_x": "mmol/L",
    "icu_lab_lact16hpci_x": "mmol/L",
    "icu_lab_lact24hpci_x": "mmol/L",
    "pbnp": "pg/mL (NT-proBNP)",
    "icu_lab_ck_x": "U/L",
    "tnt": "ng/L (= pg/mL)",
    "creatine": "umol/L",
    "white_cell_count": "10^9/L",
    "hematocrit": "%",
    "crp": "mg/L",
    "icu_lab_inr_r": "ratio",
    "glucose": "mg/dL",
    "alat": "ukat/L",
}

# Unit code -> multiplicative factor to the target unit (NaN = unknown unit).
# Medians per code in the CULPRIT-SHOCK export are given for reference.
CK_FACTORS = {1: 60.0, 2: 1.0}  # 1: ukat/L (median 5.2), 2: U/L (median 288)
LACTATE_FACTORS = {
    1: 1.0,  # mmol/L (median 3-5)
    2: 1 / 9.008,  # mg/dL (median 33-43)
    3: "infer",  # undocumented (~10 patients)
}
LACTATE_CANDIDATES = (1.0, 1 / 9.008)  # mmol/L, mg/dL
CREATININE_FACTORS = {
    1: 1.0,  # umol/L (median 110)
    2: "infer",  # undocumented (1 patient; ~1.13 mg/dL in the dataset's own median)
    3: 88.42,  # mg/dL (median 1.29)
}
CREATININE_CANDIDATES = (1.0, 88.42, 1000.0)  # umol/L, mg/dL, mmol/L
HEMATOCRIT_FACTORS = {1: 1.0, 2: 100.0}  # 1: %, 2: L/L (median 0.43)
CRP_FACTORS = {1: 10.0, 2: 1.0}  # 1: mg/dL (median 0.81), 2: mg/L (median 6)
GLUCOSE_FACTORS = {1: 1.0, 2: 18.0182}  # 1: mg/dL (median 222), 2: mmol/L (median 12)


def resolve_inferred_factors(
    values: pd.Series,
    codes: pd.Series,
    factors: dict,
    candidates: tuple[float, ...],
    name: str,
) -> dict:
    """
    Replace the "infer" entries of ``factors``: for each such unit code, use the
    candidate factor that brings the median of its (positive) values closest,
    on the log scale, to the median of the values whose unit is known.
    """
    known = {code: f for code, f in factors.items() if f != "infer"}
    reference = (values * codes.map(known)).dropna()
    reference_median = reference[reference > 0].median()
    resolved = dict(known)
    for code, factor in factors.items():
        if factor != "infer":
            continue
        group = values[codes == code].dropna()
        group = group[group > 0]
        if group.empty or pd.isna(reference_median):
            resolved[code] = np.nan
            continue
        best = min(
            candidates,
            key=lambda c: abs(np.log(group.median() * c) - np.log(reference_median)),
        )
        print(
            f"[unit harmonization] {name}: unit code {code} ({len(group)} values) "
            f"-> factor {best:.4g} (median {group.median() * best:.3g} vs "
            f"{reference_median:.3g} for the known units)"
        )
        resolved[code] = best
    return resolved


def convert_by_unit_code(
    values: pd.Series,
    codes: pd.Series,
    factors: dict,
    name: str,
    candidates: tuple[float, ...] = (),
) -> pd.Series:
    """
    Multiply each value by the factor of its unit code. Codes marked "infer"
    are resolved with resolve_inferred_factors (needs ``candidates``). Values
    with a missing or unlisted code become NaN; their number is printed.
    """
    if "infer" in factors.values():
        factors = resolve_inferred_factors(values, codes, factors, candidates, name)
    factor = codes.map(factors)
    unknown = values.notna() & factor.isna()
    if unknown.any():
        bad_codes = sorted(codes[unknown].dropna().unique().tolist())
        print(
            f"[unit harmonization] {name}: {int(unknown.sum())} values with "
            f"unknown unit (codes {bad_codes}) set to NaN"
        )
    return values * factor


def ck_unit_harmonization(data: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonize Creatine Kinase (CK) to U/L using the unit code
    (the previous version compared the CK value, not the unit code, with 1).
    """
    data["icu_lab_ck_x"] = convert_by_unit_code(
        data["icu_lab_ck_x"], data["icu_lab_ckunit_c"], CK_FACTORS, "CK"
    )
    # Central-lab CK of the biomarker sub-study is in ukat/L (median 7)
    data["ck_filled"] = (data["ck"] * 60.0).fillna(data["icu_lab_ck_x"])

    return data


def lactate_unit_harmonization(data: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonize the lactate values (admission, 8 h, 16 h, 24 h) to mmol/L.
    The factor of undocumented unit codes is inferred once, from all time
    points pooled.
    """
    columns = [
        "admission_lactate",
        "icu_lab_lact8hpci_x",
        "icu_lab_lact16hpci_x",
        "icu_lab_lact24hpci_x",
    ]
    codes = data["icu_lab_lactunit_c_x"]
    pooled_values = pd.concat([data[col] for col in columns], ignore_index=True)
    pooled_codes = pd.concat([codes] * len(columns), ignore_index=True)
    factors = resolve_inferred_factors(
        pooled_values, pooled_codes, LACTATE_FACTORS, LACTATE_CANDIDATES, "lactate"
    )
    for col in columns:
        data[col] = convert_by_unit_code(data[col], codes, factors, col)
        data = lactate_in_biological_range(data, col)

    return data


def lactate_in_biological_range(data, lactate_measure):

    # Puts the data in a biologically range
    data.loc[data[lactate_measure] > 1000, lactate_measure] /= 1000
    # Puts the data in a biologically range
    data.loc[data[lactate_measure] > 25, lactate_measure] /= 100

    return data


def creatine_unit_harmonization(data: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonize creatinine to umol/L.
    """
    data["creatine"] = convert_by_unit_code(
        data["icu_lab_crea_x"],
        data["icu_lab_creaunit_c"],
        CREATININE_FACTORS,
        "creatinine",
        candidates=CREATININE_CANDIDATES,
    )
    # Central-lab creatinine of the biomarker sub-study is in umol/L (median 111)
    data["creatine_filled"] = data["cre"].fillna(data["creatine"])

    return data


def white_blood_count_unit_harmonization(data: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonize the white blood count to 10^9/L (values reported per uL or
    10^7/L are rescaled by magnitude; the result is in a plausible range).
    """
    wcc = data["icu_lab_wct_x_x"].copy()
    # Puts the data in a biologically range
    wcc[wcc > 1000] /= 1000
    # Puts the data in a biologically range
    wcc[wcc > 100] /= 100

    data["white_cell_count"] = wcc

    return data


def hematocrit_unit_harmonization(data: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonize hematocrit to %. Some code-2 (L/L) values are already in %, so the
    range corrections are kept.
    """
    data["hematocrit"] = convert_by_unit_code(
        data["icu_lab_hct_x_x"], data["icu_lab_hctunit_c"], HEMATOCRIT_FACTORS, "hematocrit"
    )
    # Correct values
    data.loc[data["hematocrit"] > 100, "hematocrit"] /= 100
    # Puts the data in a biologically range
    data.loc[data["hematocrit"] < 1, "hematocrit"] *= 100

    return data


def crp_unit_harmonization(data: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonize C-reactive protein to mg/L
    (the previous version multiplied the mg/L values by 10).
    """
    data["crp"] = convert_by_unit_code(
        data["icu_lab_crp_x"], data["icu_lab_crpunit_c"], CRP_FACTORS, "CRP"
    )

    return data


def glucose_unit_harmonization(data: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonize glucose to mg/dL.
    """
    data["glucose"] = convert_by_unit_code(
        data["icu_lab_glc_x"], data["icu_lab_glcunit_c"], GLUCOSE_FACTORS, "glucose"
    )
    # Central-lab glucose of the biomarker sub-study is in mmol/L (median 12)
    data["glucose_filled"] = (data["gluc"] * 18.0182).fillna(data["glucose"])

    return data
