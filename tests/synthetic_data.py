"""Synthetic CULPRIT-SHOCK-like Excel file with the sheets/columns the loaders use.

Values are random; only the structure mimics the real export. Used by the tests
and for dry runs of scripts without patient data.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from lib.experiment_definitions import get_features

FILE_NAME = "CULPRIT-data_20210407.xlsx"


def write_synthetic_culprit(folder: Path, n: int = 80, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    ids = {"cor_ctr_center_id": rng.integers(1, 5, n), "cor_pt_caseno_id": np.arange(n)}
    admission = pd.Timestamp("2014-01-01") + pd.to_timedelta(rng.integers(0, 700, n), unit="D")

    def dates(p_event):
        d = admission + pd.to_timedelta(rng.integers(0, 4, n), unit="D")
        return pd.Series(d).where(rng.random(n) < p_event)

    patient = pd.DataFrame(ids)
    rare = {"p_mh_hfail_yn", "p_mh_stroke_yn", "p_mh_renf_yn", "p_mh_dial_yn", "hpr_ecg_avblock3_y"}
    for f in get_features("Admission"):
        if f == "previous_heart_complications":
            continue
        if f.endswith(("_yn", "_y")):
            patient[f] = (rng.random(n) < (0.03 if f in rare else 0.4)).astype(int)
        else:
            patient[f] = rng.normal(80, 15, n).round(1)
    for f in ["p_mh_mi_yn", "p_mh_pci_yn", "p_mh_cabg_yn"]:
        patient[f] = (rng.random(n) < 0.2).astype(int)
    patient["hpe_proc_mechs_yn"] = (rng.random(n) < 0.3).astype(int)
    for f in ["icu_lab_lactprepci_x", "icu_lab_lactpopci_x", "icu_lab_lact8hpci_x",
              "icu_lab_lact16hpci_x", "icu_lab_lact24hpci_x"]:
        patient[f] = rng.gamma(2, 2.5, n).round(2)
    # Lactate unit codes as in the real export: 1 = mmol/L, 2 = mg/dL, 3 = unknown
    lact_code = rng.choice([1, 2, 3], n, p=[0.94, 0.04, 0.02])
    lact_code[:3] = [1, 2, 3]
    patient["icu_lab_lactunit_c"] = lact_code
    for f in ["icu_lab_lactprepci_x", "icu_lab_lactpopci_x", "icu_lab_lact8hpci_x",
              "icu_lab_lact16hpci_x", "icu_lab_lact24hpci_x"]:
        patient[f] = np.where(lact_code == 2, patient[f] * 9.008, patient[f]).round(2)
    patient["icu_lab_wct_x"] = rng.normal(14, 4, n).round(1)
    patient["icu_lab_hct_x"] = rng.normal(38, 5, n).round(1)
    patient["had_base_admis_date"] = admission
    patient["h_ev_sepsis_date"] = dates(0.1)
    patient["h_ev_vfib_date"] = dates(0.15)
    patient["h_ev_stroke_date"] = dates(0.05)
    died = rng.random(n) < 0.47
    patient["fu_ce_death_le30d_yn"] = died.astype(int)
    patient["fu_ce_Death_d"] = np.where(died, rng.integers(0, 30, n), 30)
    patient["had_dem_iabpscore_c"] = rng.integers(0, 8, n)
    patient["icu_the_saps2score_num_x"] = rng.integers(20, 90, n)

    lab = pd.DataFrame(ids)
    lab["icu_lab_day_text"] = 1
    ck_u_l = rng.gamma(2, 400, n)
    lab["icu_lab_ckunit_c"] = np.where(rng.random(n) < 0.2, 1, 2)  # 1: ukat/L, 2: U/L
    lab.loc[:1, "icu_lab_ckunit_c"] = [1, 2]
    lab["icu_lab_ck_x"] = np.where(lab["icu_lab_ckunit_c"] == 1, ck_u_l / 60, ck_u_l).round(2)
    crea_umol = rng.normal(120, 40, n)
    lab["icu_lab_creaunit_c"] = np.where(rng.random(n) < 0.5, 1, 3)  # 1: umol/L, 3: mg/dL
    lab.loc[:2, "icu_lab_creaunit_c"] = [1, 3, 2]  # one code-2 patient (unknown unit)
    lab["icu_lab_crea_x"] = np.where(lab["icu_lab_creaunit_c"] == 3, crea_umol / 88.42, crea_umol).round(2)
    lab["icu_lab_wct_x"] = patient["icu_lab_wct_x"]
    lab["icu_lab_hct_x"] = patient["icu_lab_hct_x"]
    lab["icu_lab_hctunit_c"] = 1
    crp_mg_l = rng.gamma(2, 10, n)
    lab["icu_lab_crpunit_c"] = np.where(rng.random(n) < 0.3, 1, 2)  # 1: mg/dL, 2: mg/L
    lab.loc[:1, "icu_lab_crpunit_c"] = [1, 2]
    lab["icu_lab_crp_x"] = np.where(lab["icu_lab_crpunit_c"] == 1, crp_mg_l / 10, crp_mg_l).round(2)
    glc_mg_dl = rng.normal(180, 50, n)
    lab["icu_lab_glcunit_c"] = np.where(rng.random(n) < 0.4, 2, 1)  # 1: mg/dL, 2: mmol/L
    lab["icu_lab_glc_x"] = np.where(lab["icu_lab_glcunit_c"] == 2, glc_mg_dl / 18.0182, glc_mg_dl).round(2)
    lab["icu_lab_lactunit_c"] = 1
    lab["icu_lab_inr_r"] = rng.normal(1.3, 0.4, n).round(2)
    day2 = lab.copy()
    day2["icu_lab_day_text"] = 2  # must be dropped by the loader
    lab = pd.concat([lab, day2], ignore_index=True)

    clip = pd.DataFrame({k.upper(): v for k, v in ids.items()})
    clip["TIME"] = "V1"
    for col, (a, b) in {"CYSC_S_1": (2, 0.6), "LAC": (2, 2), "IL_6": (2, 100), "PBNP": (2, 2000), "CKDEPI": (4, 15),
                        "CK": (2, 400), "CRE": (2, 60), "GLUC": (2, 90), "TNT": (2, 2000), "ALAT": (2, 1)}.items():
        clip[col] = rng.gamma(a, b, n).round(2)

    # Therapy sheets as in the real export: one row per episode, several rows
    # per patient, not in the patient order (row-wise alignment with
    # patient_data is wrong). Catecholamine start: day 0-1 for ~85% of the
    # patients, otherwise later; second episodes always start later.
    first_day = rng.random(n) < 0.85
    start_day = np.where(first_day, rng.integers(0, 2, n), rng.integers(2, 6, n))
    cate = pd.DataFrame(ids)
    cate["icu_catec_start_date"] = admission + pd.to_timedelta(start_day, unit="D")
    second = pd.DataFrame(ids).iloc[: n // 2]
    second["icu_catec_start_date"] = (admission + pd.to_timedelta(rng.integers(3, 9, n), unit="D"))[: n // 2]
    cate = pd.concat([cate, second]).sample(frac=1, random_state=seed).reset_index(drop=True)
    # RRT: only ~60% of the patients appear in the endpoint sheet
    in_sheet = rng.random(n) < 0.6
    rrt_day = np.where(rng.random(n) < 0.25, rng.integers(0, 2, n), rng.integers(2, 10, n))
    endpoint = pd.DataFrame(ids)
    endpoint["icu_rrt_start_date"] = pd.Series(admission + pd.to_timedelta(rrt_day, unit="D")).where(
        rng.random(n) < 0.5
    )
    endpoint = endpoint[in_sheet].sample(frac=1, random_state=seed + 1).reset_index(drop=True)
    # Clinical variables used by the CardShock score
    patient["hpr_echo_lvef_pct"] = rng.normal(40, 12, n).round(0)

    path = Path(folder) / FILE_NAME
    with pd.ExcelWriter(path) as writer:
        for name, df in {"patient_data": patient, "laboratory_data": lab, "clip": clip,
                         "cathecholamine": cate, "endpoint": endpoint}.items():
            df.to_excel(writer, sheet_name=name, index=False)
    return path
