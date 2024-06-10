from typing import List


def get_features(exp_name: str) -> List[str]:
    """
    Get the list of features based on the experiment name.

    Parameters:
        exp_name (str): The name of the experiment.

    Returns:
        List[str]: The list of features for the specified experiment.
    """
    features_list: List[str]

    if exp_name == "Admission":
        # Baseline Model features
        features_list = [
                        # Patient Information
                        "had_dem_age_yr",           # 1 Age
                        "had_dem_male_yn",          # 2 Gender
                        "had_pex_weight_kg",        # 3 Weight
                        "had_pex_height_cm",        # 4 Height

                        # Comorbilities
                        "previous_heart_complications", # 5  Combined variable: Previous myocardial infarction, Previous PCI, Previous CABG     # noqa
                        "p_mh_hfail_yn",            # 6  Previous congestive heart                                                          # noqa
                        "p_mh_stroke_yn",           # 7  Previous stroke
                        "p_mh_pad_yn",              # 8  Known peripheral artery disease                                                    # noqa
                        "p_mh_renf_yn",             # 9  Known renal insufficiency (GFR < 30 ml/min)                                        # noqa
                        "p_mh_dial_yn",             # 10 Chronic dialysis
                        "p_rf_smoker_yn",           # 11 Current smoking
                        "p_rf_aht_yn",              # 12 Arterial Hypertension
                        "p_rf_dyslip_yn",           # 13 Dyslipidemia
                        "p_rf_dm_yn",               # 14 Diabetes mellitus

                        # ECG
                        "hpr_ecg_sinrhy_y",         # 15 Sinus rhythm
                        "hpr_ecg_afib_y",           # 16 Atrial fibrillation
                        "hpr_ecg_avblock3_y",       # 17 AV-block III
                        "hpr_ecg_stemi_yn",         # 18 ST-segment elevation

                        # Clinical shock characteristics
                        "hpr_hmdyn_hr_bpm",         # 19 Heart rate [bpm]
                        "hpr_hmdyn_sbp_mmhg",       # 20 Systolic blood pressure                                                            # noqa
                        "hpr_hmdyn_dbp_mmhg",       # 21 Diastolic blood pressure                                                           # noqa
                        "had_sy_ams_yn",            # 22 Altered mental status
                        "had_sy_extremity_yn",      # 23 Cold, clammy skin and extremities                                                  # noqa
                        "had_base_mechvent_yn",     # 24 Mechanical ventilation
                        "had_base_cpr24h_yn"        # 25 Resuscitation within 24 hs before admission                                        # noqa
                        ]

    elif exp_name == "24hs":
        # Baseline Model features
        features_list = [
                        # Admission Features
                        # Patient Information
                        "had_dem_age_yr",           # 1 Age
                        "had_dem_male_yn",          # 2 Gender
                        "had_pex_weight_kg",        # 3 Weight
                        "had_pex_height_cm",        # 4 Height

                        # Comorbilities
                        "previous_heart_complications",        # 5  Combined variable: Previous myocardial infarction, Previous PCI, Previous CABG     # noqa
                        "p_mh_hfail_yn",            # 6  Previous congestive heart                                                          # noqa
                        "p_mh_stroke_yn",           # 7  Previous stroke
                        "p_mh_pad_yn",              # 8  Known peripheral artery disease                                                    # noqa
                        "p_mh_renf_yn",             # 9  Known renal insufficiency (GFR < 30 ml/min)                                        # noqa
                        "p_mh_dial_yn",             # 10 Chronic dialysis
                        "p_rf_smoker_yn",           # 11 Current smoking
                        "p_rf_aht_yn",              # 12 Arterial Hypertension
                        "p_rf_dyslip_yn",           # 13 Dyslipidemia
                        "p_rf_dm_yn",               # 14 Diabetes mellitus

                        # ECG
                        "hpr_ecg_sinrhy_y",         # 15 Sinus rhythm
                        "hpr_ecg_afib_y",           # 16 Atrial fibrillation
                        "hpr_ecg_avblock3_y",        # 17 AV-block III
                        "hpr_ecg_stemi_yn",         # 18 ST-segment elevation

                        # Clinical shock characteristics
                        "hpr_hmdyn_hr_bpm",         # 19 Heart rate [bpm]
                        "hpr_hmdyn_sbp_mmhg",       # 20 Systolic blood pressure                                                            # noqa
                        "hpr_hmdyn_dbp_mmhg",       # 21 Diastolic blood pressure                                                           # noqa    
                        "had_sy_ams_yn",            # 22 Altered mental status
                        "had_sy_extremity_yn",      # 23 Cold, clammy skin and extremities                                                  # noqa
                        "had_base_mechvent_yn",     # 24 Mechanical ventilation
                        "had_base_cpr24h_yn",        # 25 Resuscitation within 24 hs before admission                                        # noqa

                        # Added 24hs features
                        # Organ perfusion
                        "admission_lactate",        # 26 Admission Lactate: Pre-PCI or Post-PCI if Pre is missing                           # noqa
                        "icu_lab_lact8hpci_x",      # 27 Serum lactated measured 8hs after admission                                        # noqa
                        "icu_lab_lact16hpci_x",     # 28 Serum lactated measured 16hs after admission                                       # noqa
                        "icu_lab_lact24hpci_x",     # 29 Serum lactated measured 24hs after admission                                       # noqa    

                        # Laboratory assessment
                        "pbnp",                     # 30 Nt-pro BNP
                        "icu_lab_ck_x",             # 31 CK (Can be filled)
                        "tnt",                      # 32 Troponine V1 pg/ml
                        "creatine",                 # 33 Creatine
                        "white_cell_count",         # 34 White cell count
                        "hematocrit",               # 35 Hematocrit
                        "crp",                      # 36 CRP
                        "icu_lab_inr_r",            # 37 INR
                        "glucose",                  # 38 Glucose
                        "alat",                     # 39 ALAT

                        # Treatment modalities
                        "hpe_proc_mechs_yn",            # 40 Mechanical support
                        "renal_replacement_therapy",    # 41 Renal Replac. Ther
                        "cathecholamine_therapy",       # 42 Catecholamine thet
                        "sepsis",                       # 43 Sepsis
                        "ventricular_fibrillation",     # 44 Ventricular fib
                        "stroke",                       # 45 Stroke
                         ]

    elif exp_name == "Admission_less_features":
        # Baseline Model features
        features_list = [
                        # Patient Information
                        "had_dem_age_yr",           # 1 Age
                        "had_pex_weight_kg",        # 3 Weight
                        "had_pex_height_cm",        # 4 Height

                        # Comorbilities
                        "p_mh_hfail_yn",            # 6  Previous congestive heart                                                          # noqa
                        "p_mh_stroke_yn",           # 7  Previous stroke
                        "p_mh_renf_yn",             # 9  Known renal insufficiency (GFR < 30 ml/min)                                        # noqa
                        "p_mh_dial_yn",             # 10 Chronic dialysis
                        "p_rf_smoker_yn",           # 11 Current smoking
                        "p_rf_dyslip_yn",           # 13 Dyslipidemia
                        "p_rf_dm_yn",               # 14 Diabetes mellitus

                        # ECG
                        "hpr_ecg_sinrhy_y",         # 15 Sinus rhythm
                        "hpr_ecg_avblock3_y",       # 17 AV-block III

                        # Clinical shock characteristics
                        "hpr_hmdyn_hr_bpm",         # 19 Heart rate [bpm]
                        "hpr_hmdyn_sbp_mmhg",       # 20 Systolic blood pressure                                                            # noqa
                        "hpr_hmdyn_dbp_mmhg",       # 21 Diastolic blood pressure                                                           # noqa
                        "had_sy_ams_yn",            # 22 Altered mental status
                        "had_base_mechvent_yn",     # 24 Mechanical ventilation
                        ]
    elif exp_name == "CLIP_SCORE":
        features_list = ["CLIP_Score",
                         ]

    elif exp_name == "IABP_SCORE":
        features_list = ["had_dem_iabpscore_c"]

    elif exp_name == "SAPS_SCORE":
        features_list = ["icu_the_saps2score_num_x"]

    else:
        RuntimeError("Experiment not set")

    return features_list


def get_important_features(exp_name: str) -> List[str]:
    """
    Get the list of features with the most relevant features first.

    Parameters:
        exp_name (str): The name of the model.

    Returns:
        List[str]: The list of features.
    """
    features_list: List[str]

    if exp_name == "Admission":
        # Direct order of importance
        features_list = ["p_rf_dyslip_yn",
                         "hpr_hmdyn_hr_bpm",
                         "hpr_hmdyn_dbp_mmhg",
                         "had_sy_ams_yn",
                         "hpr_ecg_sinrhy_y",
                         "had_pex_height_cm",
                         "p_rf_smoker_yn",
                         "had_base_mechvent_yn",
                         "p_rf_dm_yn",
                         "had_pex_weight_kg",
                         "hpr_hmdyn_sbp_mmhg",
                         "p_mh_pad_yn",
                         "hpr_ecg_stemi_yn",
                         "previous_heart_complications",
                         "p_rf_aht_yn",
                         "had_sy_extremity_yn",
                         "had_base_cpr24h_yn",
                         "hpr_ecg_afib_y"
                         ]

    elif exp_name == "Full":
        # Direct order of importance
        features_list = ["creatine",
                         "icu_lab_lact16hpci_x",
                         "icu_lab_lact8hpci_x",
                         "pbnp",
                         "icu_lab_lact24hpci_x",
                         "admission_lactate",
                         "crp",
                         "hematocrit",
                         "glucose",
                         "white_cell_count",
                         "icu_lab_inr_r",
                         "tnt",
                         "hpe_proc_mechs_yn",
                         "alat",
                         "icu_lab_ck_x"
                         ]

    if exp_name == "Admission_less_features":
        # Direct order of importance
        features_list = ["p_rf_dyslip_yn",
                         "hpr_hmdyn_hr_bpm",
                         "hpr_hmdyn_dbp_mmhg",
                         "had_sy_ams_yn",
                         "hpr_ecg_sinrhy_y",
                         "had_pex_height_cm",
                         "p_rf_smoker_yn",
                         "had_base_mechvent_yn",
                         "p_rf_dm_yn",
                         "had_pex_weight_kg",
                         "hpr_hmdyn_sbp_mmhg",
                         ]
    else:
        RuntimeError("Experiment not set")

    return features_list
