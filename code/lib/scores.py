"""Conventional risk scores computed from the CULPRIT-SHOCK data."""

import numpy as np
import pandas as pd

# Columns of load_CULPRIT_data() used by the CardShock score
CARDSHOCK_COLUMNS = {
    "age": "had_dem_age_yr",
    "confusion": "had_sy_ams_yn",  # altered mental status at presentation
    "previous_mi": "p_mh_mi_yn",
    "previous_cabg": "p_mh_cabg_yn",
    "lvef": "hpr_echo_lvef_pct",
    "lactate": "admission_lactate",  # harmonized to mmol/L
    "egfr": "ckdepi",  # eGFR CKD-EPI, mL/min/1.73 m2
}


def cardshock_score(data: pd.DataFrame) -> pd.Series:
    """
    CardShock risk score (Harjola et al., Eur J Heart Fail 2015;
    Supplementary Table 7), 0-9 points:

    - age > 75 years: 1
    - confusion at presentation: 1
    - previous myocardial infarction or CABG: 1
    - ACS aetiology: 1 (all CULPRIT-SHOCK patients had an acute MI)
    - LVEF < 40%: 1
    - blood lactate: < 2 mmol/L 0, 2-4 mmol/L 1, > 4 mmol/L 2
    - eGFR (CKD-EPI): > 60 0, 30-60 1, < 30 mL/min/1.73 m2 2

    Returns NaN for patients with a missing component. "Previous MI or CABG"
    is known when at least one of the two is recorded as 1, or both as 0/1.
    """
    c = CARDSHOCK_COLUMNS
    age = data[c["age"]]
    confusion = data[c["confusion"]]
    history = data[[c["previous_mi"], c["previous_cabg"]]]
    lvef = data[c["lvef"]]
    lactate = data[c["lactate"]]
    egfr = data[c["egfr"]]

    previous_mi_or_cabg = (history == 1).any(axis=1)
    history_known = previous_mi_or_cabg | history.notna().all(axis=1)

    score = (
        (age > 75).astype(int)
        + (confusion == 1).astype(int)
        + previous_mi_or_cabg.astype(int)
        + 1  # ACS aetiology
        + (lvef < 40).astype(int)
        + pd.Series(np.select([lactate > 4, lactate >= 2], [2, 1], 0), index=data.index)
        + pd.Series(np.select([egfr < 30, egfr <= 60], [2, 1], 0), index=data.index)
    )
    complete = (
        age.notna()
        & confusion.notna()
        & history_known
        & lvef.notna()
        & lactate.notna()
        & egfr.notna()
    )
    return score.astype(float).where(complete)
