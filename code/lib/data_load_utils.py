from pathlib import Path

import numpy as np
import pandas as pd
from lib.eICU_processing import EICU_OUTCOMES
from lib.unit_harmonization import (
    ck_unit_harmonization,
    creatine_unit_harmonization,
    crp_unit_harmonization,
    glucose_unit_harmonization,
    hematocrit_unit_harmonization,
    lactate_unit_harmonization,
    white_blood_count_unit_harmonization,
)


def load_CULPRIT_data(data_dir: Path, harmonize: bool = True) -> pd.DataFrame:
    """
    Load CULPRIT study data, keep only the first day information, and harmonize units.

    Parameters:
        data_dir (str): The directory path where the data is located.
        harmonize (bool): Apply the unit harmonization (default). False returns
            the raw lab values, e.g. to inspect units (0-preprocessing/check_lab_units.py).

    Returns:
        pd.DataFrame: Processed and harmonized CULPRIT study data.
    """

    # Load main data
    data = pd.read_excel(
        data_dir / "CULPRIT-data_20210407.xlsx",
        sheet_name=None,
    )

    # Extract patient data
    patient_info = data["patient_data"]
    # Set patient ID as unique
    patient_info = create_unique_patient_ID(patient_info)
    # Include the combined variable in the patients information
    patient_info = create_previous_heart_complications(patient_info)
    # Create admission lactate
    patient_info = create_admission_lactate(patient_info)
    # Add Catecholamine Therapy information
    patient_info = add_catecholamine_therapy(data, patient_info)
    # Add Renal Replacement Therapy information
    patient_info = add_renal_replacement_therapy(data, patient_info)
    # Add Sepsis information
    patient_info = add_sepsis(data, patient_info)
    # Add Ventricular Fibrillation information
    patient_info = add_ventricular_fibrillation(data, patient_info)
    # Add stroke information
    patient_info = add_stroke(data, patient_info)
    # Add Resusitation within first 24hs
    patient_info = add_resusitation_24hs(patient_info)

    # Load Laboratory data
    lab_info = data["laboratory_data"]
    # Set patient ID as unique
    lab_info = create_unique_patient_ID(lab_info)

    # Keep the information for the first day.
    lab_info = lab_info[lab_info["icu_lab_day_text"] == 1]
    # The day information is not longer required
    lab_info = lab_info.drop(columns="icu_lab_day_text")

    # Load clip data
    clip_info = data["clip"]
    # Change formating to match the rest of the variables
    clip_info.columns = clip_info.keys().str.lower()
    # Set patient ID as unique
    clip_info = create_unique_patient_ID(clip_info)

    # Get clip information only for TIME V1 == Admission
    clip_info = clip_info[clip_info["time"] == "V1"]
    # Time information is not longer required
    clip_info = clip_info.drop(columns="time")

    # First merge patient and lab data
    data_final = pd.merge(patient_info, lab_info, on="patient_ID")
    # Add also the clip information
    data_final = data_final.merge(clip_info, on="patient_ID", how="left")

    # Calculate CLIP score from features
    data_final = calcule_CLIP_score_from_features(data_final)

    if not harmonize:
        return data_final

    # Harmonize units
    data_final = ck_unit_harmonization(data_final)
    data_final = lactate_unit_harmonization(data_final)
    data_final = creatine_unit_harmonization(data_final)
    data_final = white_blood_count_unit_harmonization(data_final)
    data_final = hematocrit_unit_harmonization(data_final)
    data_final = crp_unit_harmonization(data_final)
    data_final = glucose_unit_harmonization(data_final)

    return data_final


def create_unique_patient_ID(data: pd.DataFrame) -> pd.DataFrame:
    """
    #
    Combine the center ID and the patient ID to generate a unique ID.

    Parameters:
        data (pd.DataFrame): The input DataFrame containing center and patient IDs.

    Returns:
        pd.DataFrame: The DataFrame with a new column 'patient_ID' representing the unique ID.
    """
    data["patient_ID"] = (
        data["cor_ctr_center_id"].astype(str)
        + "-"
        + data["cor_pt_caseno_id"].astype(str)
    )
    return data


def get_data_from_features(data: pd.DataFrame, features_code: list) -> pd.DataFrame:
    """
    Extract features from patient data.

    Parameters:
        data (pd.DataFrame): The input DataFrame containing patient data.
        features_code (list): List of feature codes to extract.

    Returns:
        pd.DataFrame: DataFrame containing selected features.
    """
    return data.loc[:, features_code]


def create_previous_heart_complications(patient_info: pd.DataFrame) -> pd.DataFrame:
    """
    Create a combined variable based on previous medical history.

    Parameters:
        patient_info (pd.DataFrame): DataFrame containing patient information.

    Returns:
        pd.DataFrame: DataFrame with 'previous_heart_complications'.
    """
    var1 = patient_info["p_mh_mi_yn"]
    var2 = patient_info["p_mh_pci_yn"]
    var3 = patient_info["p_mh_cabg_yn"]
    series_sum = var1 + var2 + var3
    patient_info["previous_heart_complications"] = (series_sum > 0).astype(int)
    return patient_info


def calcule_CLIP_score_from_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate CLIP score from features.

    Parameters:
        data (pd.DataFrame): The input DataFrame containing relevant features.

    Returns:
        pd.DataFrame: The DataFrame with a new column 'CLIP_Score'.
    """
    C = np.arcsinh(100 * data["cysc_s_1"])
    L = np.arcsinh(100 * data["lac"])
    Inter = np.arcsinh(100 * data["il_6"])
    P = np.arcsinh(100 * data["pbnp"])

    Linear_predictor = (
        -15.8532036 + 0.06714669 * C + 1.0287073 * L + 0.2704829 * Inter + 0.1923877 * P
    )
    data["CLIP_Score"] = 100 * np.exp(Linear_predictor) / (1 + np.exp(Linear_predictor))

    return data


def create_admission_lactate(patient_info: pd.DataFrame) -> pd.DataFrame:
    """
    Create admission lactate variable.

    Parameters:
        patient_info (pd.DataFrame): DataFrame containing patient information.

    Returns:
        pd.DataFrame: DataFrame with a new column 'admission_lactate'.
    """
    # Pre-PCI lactate (the admission value); post-PCI only if pre-PCI is missing.
    # (The first submission used post-PCI first.)
    patient_info["admission_lactate"] = patient_info[
        "icu_lab_lactprepci_x"
    ].combine_first(patient_info["icu_lab_lactpopci_x"])
    return patient_info


def get_admission_date(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Extract admission date from patient information.

    Parameters:
        data: The Excel sheets of the CULPRIT-SHOCK export.

    Returns:
        pd.DataFrame: patient_ID and admission date (had_base_admis_date).
    """
    patient_info = create_unique_patient_ID(data["patient_data"])

    admission_date = patient_info.loc[:, ["had_base_admis_date", "patient_ID"]]
    return admission_date


def therapy_started_first_day(
    data: dict[str, pd.DataFrame], sheet: str, start_column: str, name: str
) -> pd.DataFrame:
    """
    Flag patients with a therapy that started on the admission day or the day
    after (day 0 or 1), from a sheet with one row per therapy episode.

    The start date of every episode is compared with the admission date of the
    same patient (matched on patient_ID). Previously the two columns were
    subtracted row by row although they come from different sheets, so the
    episodes were compared with the admission date of other patients.

    Returns:
        pd.DataFrame: one row per patient in the sheet, with patient_ID and
        ``name`` (1 if any episode started on day 0 or 1, else 0).
    """
    episodes = create_unique_patient_ID(data[sheet]).loc[:, ["patient_ID", start_column]]
    episodes = episodes.merge(get_admission_date(data), on="patient_ID", how="left")
    days = (
        pd.to_datetime(episodes[start_column])
        - pd.to_datetime(episodes["had_base_admis_date"])
    ).dt.days
    episodes[name] = days.isin([0, 1])
    flags = episodes.groupby("patient_ID")[name].any().astype(int).reset_index()
    return flags


def add_catecholamine_therapy(
    data: dict[str, pd.DataFrame], patient_info: pd.DataFrame
) -> pd.DataFrame:
    """
    Add catecholamine therapy (started on day 0 or 1) to patient data.

    As before, only patients present in the catecholamine sheet are kept
    (inner merge), so the cohort is unchanged.

    Parameters:
        data: The Excel sheets of the CULPRIT-SHOCK export.
        patient_info (pd.DataFrame): DataFrame containing patient information.

    Returns:
        pd.DataFrame: Updated patient_info DataFrame with 'cathecholamine_therapy' column.
    """
    cate_df = therapy_started_first_day(
        data, "cathecholamine", "icu_catec_start_date", "cathecholamine_therapy"
    )
    n_before = len(patient_info)
    patient_info = pd.merge(patient_info, cate_df, on="patient_ID")
    if len(patient_info) < n_before:
        print(
            f"{n_before - len(patient_info)} patients without a catecholamine record "
            "were excluded (cohort definition unchanged)"
        )

    return patient_info


def add_renal_replacement_therapy(
    data: dict[str, pd.DataFrame], patient_X: pd.DataFrame
) -> pd.DataFrame:
    """
    Add renal replacement therapy (started on day 0 or 1) to patient data.
    Patients not present in the endpoint sheet get NaN (missing), as before.

    Parameters:
        data: The Excel sheets of the CULPRIT-SHOCK export.
        patient_X (pd.DataFrame): DataFrame containing patient information.

    Returns:
        pd.DataFrame: Updated patient_X DataFrame with 'renal_replacement_therapy' column.
    """
    rrt_df = therapy_started_first_day(
        data, "endpoint", "icu_rrt_start_date", "renal_replacement_therapy"
    )
    patient_X = patient_X.merge(rrt_df, on="patient_ID", how="left")

    return patient_X


def add_sepsis(data: dict[str, pd.DataFrame], patient_info: pd.DataFrame) -> pd.DataFrame:
    """
    Add sepsis information to patient data.

    Parameters:
        data: The input data.
        patient_info (pd.DataFrame): DataFrame containing patient information.

    Returns:
        pd.DataFrame: Updated patient_info DataFrame with 'sepsis' column.
    """
    dates = data["patient_data"].loc[
        :, ["h_ev_sepsis_date", "had_base_admis_date", "patient_ID"]
    ]

    dates["date_sepsis_start"] = pd.to_datetime(dates["h_ev_sepsis_date"])
    dates["date_admission"] = pd.to_datetime(dates["had_base_admis_date"])

    dates["sepsis"] = (
        dates["date_sepsis_start"] - dates["date_admission"]
    ).dt.days.isin([0, 1])
    sepsis_df = dates.groupby("patient_ID")["sepsis"].any().reset_index()
    sepsis_df["sepsis"] = sepsis_df["sepsis"].astype(int)

    patient_info = pd.merge(patient_info, sepsis_df, on="patient_ID")

    return patient_info


def add_ventricular_fibrillation(
    data: dict[str, pd.DataFrame], patient_info: pd.DataFrame
) -> pd.DataFrame:
    """
    Add ventricular fibrillation information to patient data.

    Parameters:
        data: The input data.
        patient_info (pd.DataFrame): DataFrame containing patient information.

    Returns:
        pd.DataFrame: Updated patient_info DataFrame with 'ventricular_fibrillation' column.
    """
    dates = data["patient_data"].loc[
        :, ["h_ev_vfib_date", "had_base_admis_date", "patient_ID"]
    ]

    dates["date_vf_start"] = pd.to_datetime(dates["h_ev_vfib_date"])
    dates["date_admission"] = pd.to_datetime(dates["had_base_admis_date"])

    dates["ventricular_fibrillation"] = (
        dates["date_vf_start"] - dates["date_admission"]
    ).dt.days.isin([0, 1])
    vf_df = dates.groupby("patient_ID")["ventricular_fibrillation"].any().reset_index()
    vf_df["ventricular_fibrillation"] = vf_df["ventricular_fibrillation"].astype(int)

    patient_info = pd.merge(patient_info, vf_df, on="patient_ID")

    return patient_info


def add_stroke(data: dict[str, pd.DataFrame], patient_info: pd.DataFrame) -> pd.DataFrame:
    """
    Add stroke information to patient data.

    Parameters:
        data: The input data.
        patient_info (pd.DataFrame): DataFrame containing patient information.

    Returns:
        pd.DataFrame: Updated patient_info DataFrame with 'stroke' column.
    """
    dates = data["patient_data"].loc[
        :, ["h_ev_stroke_date", "had_base_admis_date", "patient_ID"]
    ]

    dates["date_stroke_start"] = pd.to_datetime(dates["h_ev_stroke_date"])
    dates["date_admission"] = pd.to_datetime(dates["had_base_admis_date"])

    dates["stroke"] = (
        dates["date_stroke_start"] - dates["date_admission"]
    ).dt.days.isin([0, 1])
    stroke_df = dates.groupby("patient_ID")["stroke"].any().reset_index()
    stroke_df["stroke"] = stroke_df["stroke"].astype(int)

    patient_info = pd.merge(patient_info, stroke_df, on="patient_ID")

    return patient_info


def add_resusitation_24hs(patient_info: pd.DataFrame) -> pd.DataFrame:
    """
    Add resuscitation within 24 hours information to patient data.

    Parameters:
        patient_info (pd.DataFrame): DataFrame containing patient information.

    Returns:
        pd.DataFrame: Updated patient_info with 'resuscitation_24hs' column.
    """
    var1 = patient_info["had_base_cpr24h_yn"]
    var2 = patient_info["ventricular_fibrillation"]

    series_sum = var1 + var2
    patient_info["resuscitation_24hs"] = (series_sum > 0).astype(int)

    return patient_info


def load_eICU(
    data_dir: Path,
    features: str,
    X_CULPRIT: pd.DataFrame,
    outcome: str = "in_hospital_death",
    exclude_early_deaths: bool | None = None,
    match_pbnp_to_culprit: bool = False,
    tni_as_tnt_proxy: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load the preprocessed eICU cohort (from eICU_extract_CS_patients.py) and
    align it with the CULPRIT-SHOCK feature matrix.

    Parameters
    ----------
    data_dir : Path
        Folder that contains ``eicu-collaborative-research-database-2.0``.
    features : str
        "Admission" or "Full".
    X_CULPRIT : pd.DataFrame
        CULPRIT-SHOCK features the model was trained on; eICU columns are
        reordered to match (missing ones are NaN).
    outcome : str
        Label to use, one of lib.eICU_processing.EICU_OUTCOMES:
        "in_hospital_death" (used in the submitted manuscript),
        "in_hospital_death_30d_icu", "in_hospital_death_30d_hospital" or
        "icu_death".
    exclude_early_deaths : bool | None
        Drop patients who died within 24 h of ICU admission. Default (None):
        True for "Full" (matching the CULPRIT-SHOCK Full-model cohort),
        False for "Admission".
    match_pbnp_to_culprit : bool
        Quantile-map the eICU NT-proBNP proxy onto the CULPRIT-SHOCK NT-proBNP
        distribution (sensitivity analysis; removes any true difference in
        level between the cohorts). Default False.
    tni_as_tnt_proxy : bool
        Full model only. eICU rarely reports troponin T (CULPRIT-SHOCK) but
        often troponin I. For patients without troponin T, troponin I is
        quantile-mapped onto the CULPRIT-SHOCK troponin-T distribution (rank
        preserving; the two assays have no common scale). Default True.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        X_eicu (indexed by patientunitstayid) and the 0/1 labels.
    """
    if features not in ("Admission", "Full"):
        raise ValueError(f"features must be 'Admission' or 'Full', got {features!r}")
    if outcome not in EICU_OUTCOMES:
        raise ValueError(f"outcome must be one of {list(EICU_OUTCOMES)}, got {outcome!r}")
    if exclude_early_deaths is None:
        exclude_early_deaths = features == "Full"

    eicu_root = (
        Path(data_dir) / "eicu-collaborative-research-database-2.0" / "preprocessed_MACRO"
    )
    X_eicu = pd.read_csv(
        eicu_root / f"X_{features}_CICU.csv", index_col="patientunitstayid"
    )
    outcomes = pd.read_csv(
        eicu_root / "outcomes_CICU.csv", index_col="patientunitstayid"
    ).loc[X_eicu.index]

    keep = outcomes[outcome].notna()
    if exclude_early_deaths:
        keep &= outcomes["death_within_24h"] != 1
    print(
        f"eICU {features}: {int(keep.sum())} patients, outcome '{outcome}' "
        f"({int(outcomes.loc[keep, outcome].sum())} events)"
        + (", early deaths excluded" if exclude_early_deaths else "")
    )

    X_eicu = X_eicu.loc[keep]
    if tni_as_tnt_proxy and "eicu_troponin_I" in X_eicu and "tnt" in X_CULPRIT and "tnt" in X_eicu:
        proxy = quantile_match(X_eicu["eicu_troponin_I"], X_CULPRIT["tnt"])
        fill = X_eicu["tnt"].isna() & proxy.notna()
        X_eicu.loc[fill, "tnt"] = proxy[fill]
        print(f"eICU troponin T: {int(X_eicu['tnt'].notna().sum() - fill.sum())} measured, "
              f"{int(fill.sum())} from troponin I (quantile-mapped)")

    # Align eICU feature columns with the CULPRIT-SHOCK training columns
    X_eicu = X_eicu.reindex(columns=X_CULPRIT.columns)

    if match_pbnp_to_culprit and "pbnp" in X_eicu and "pbnp" in X_CULPRIT:
        X_eicu["pbnp"] = quantile_match(X_eicu["pbnp"], X_CULPRIT["pbnp"])
    Y_eicu = outcomes.loc[keep, outcome].astype(int).to_numpy()

    return X_eicu, Y_eicu


def quantile_match(values: pd.Series, reference: pd.Series) -> pd.Series:
    """
    Map each value to the reference quantile at the value's (mid-)rank
    percentile, so that the result has the reference distribution. NaN stay NaN.
    """
    present = values.dropna()
    if present.empty or reference.dropna().empty:
        return values
    percentiles = (present.rank(method="average") - 0.5) / len(present)
    matched = np.quantile(reference.dropna().to_numpy(), percentiles.to_numpy())
    return values.astype(float).where(values.isna(), pd.Series(matched, index=present.index))
