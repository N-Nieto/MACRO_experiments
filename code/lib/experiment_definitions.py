def get_features(exp_name):

    if exp_name == "Admission":
        # Baseline Model features
        features_list = ["patient_ID",
                         "had_pex_weight_kg",
                         "had_pex_height_cm",
                         "had_base_mechvent_yn",
                         "had_base_cpr24h_yn",
                         "had_sy_ams_yn",
                         "had_sy_extremity_yn",
                         "combined_variable",
                         "p_mh_hfail_yn",
                         "p_mh_stroke_yn",
                         "p_mh_pad_yn",          # 10
                         "p_mh_renf_yn",
                         "p_mh_dial_yn",
                         "p_rf_smoker_yn",
                         "p_rf_aht_yn",
                         "p_rf_dyslip_yn",
                         "p_rf_dm_yn",
                         "hpr_ecg_sinrhy_y",
                         "hpr_ecg_afib_y",
                         "hpr_ecg_avblock3_y",
                         "hpr_ecg_stemi_yn",     # 20
                         "hpr_hmdyn_hr_bpm",
                         "hpr_hmdyn_sbp_mmhg",
                         "hpr_hmdyn_dbp_mmhg",
                         "had_dem_male_yn",
                         "had_dem_age_yr",       # 25
                         ]

    elif exp_name == "24hs":
        features_list = ["patient_ID",
                         "had_pex_weight_kg",
                         "had_pex_height_cm",
                         "had_base_mechvent_yn",
                         "had_base_cpr24h_yn",
                         "had_sy_ams_yn",
                         "had_sy_extremity_yn",
                         "combined_variable",
                         "p_mh_hfail_yn",
                         "p_mh_stroke_yn",
                         "p_mh_pad_yn",          # 10
                         "p_mh_renf_yn",
                         "p_mh_dial_yn",
                         "p_rf_smoker_yn",
                         "p_rf_aht_yn",
                         "p_rf_dyslip_yn",
                         "p_rf_dm_yn",
                         "hpr_ecg_sinrhy_y",
                         "hpr_ecg_afib_y",
                         "hpr_ecg_avblock3_y",
                         "hpr_ecg_stemi_yn",     # 20
                         "hpr_hmdyn_hr_bpm",
                         "hpr_hmdyn_sbp_mmhg",
                         "hpr_hmdyn_dbp_mmhg",
                         "had_dem_male_yn",
                         "had_dem_age_yr",       # 25    Admission features

                         "hpe_proc_mechs_c",     # 1     After 24hs
                         "hio_proc_mih_yn",
                         "h_ev_rrt_yn",
                         "icu_lab_lactprepci_x",
                         "icu_lab_lactpopci_x",
                         "icu_lab_lact8hpci_x",
                         "icu_lab_lact16hpci_x",
                         "icu_lab_lact24hpci_x",
                         "h_com_stroke_yn",
                         "h_com_cpr_yn",
                         "h_com_noreflow_yn",

                         "icu_lab_crea_x",
                         "icu_lab_hct_x",
                         "icu_lab_wct_x",
                         "icu_lab_crp_x",
                         "icu_lab_glc_x",
                         "icu_lab_ck_x",
                         "icu_lab_hstropi_x",

                         "hpo_proc_succottimi_yn",
                         "icu_lab_inr_r",
                         "alat",
                         "ckdepi",
                         "pbnp"
                         ]

    elif exp_name == "24hs_CLIP":
        features_list = [
                        "had_pex_weight_kg",
                        "had_pex_height_cm",
                        "had_base_mechvent_yn",
                        "had_base_cpr24h_yn",
                        "had_sy_ams_yn",
                        "had_sy_extremity_yn",
                        "combined_variable",
                        "p_mh_hfail_yn",
                        "p_mh_stroke_yn",
                        "p_mh_pad_yn",          # 10
                        "p_mh_renf_yn",
                        "p_mh_dial_yn",
                        "p_rf_smoker_yn",
                        "p_rf_aht_yn",
                        "p_rf_dyslip_yn",
                        "p_rf_dm_yn",
                        "hpr_ecg_sinrhy_y",
                        "hpr_ecg_afib_y",
                        "hpr_ecg_avblock3_y",
                        "hpr_ecg_stemi_yn",     # 20
                        "hpr_hmdyn_hr_bpm",
                        "hpr_hmdyn_sbp_mmhg",
                        "hpr_hmdyn_dbp_mmhg",
                        "had_dem_male_yn",
                        "had_dem_age_yr",       # 25    Admission features

                        "hpe_proc_mechs_c",     # 1     After 24hs
                        "hio_proc_mih_yn",
                        "h_ev_rrt_yn",
                        "icu_lab_lactprepci_x",
                        "icu_lab_lactpopci_x",
                        "icu_lab_lact8hpci_x",
                        "icu_lab_lact16hpci_x",
                        "icu_lab_lact24hpci_x",
                        "h_com_stroke_yn",
                        "h_com_cpr_yn",
                        "h_com_noreflow_yn",

                        "icu_lab_crea_x",
                        "icu_lab_hct_x",
                        "icu_lab_wct_x",
                        "icu_lab_crp_x",
                        "icu_lab_glc_x",
                        "icu_lab_ck_x",
                        "icu_lab_hstropi_x",

                        "hpo_proc_succottimi_yn",
                        "icu_lab_inr_r",
                        "alat",
                        "ckdepi",
                        "pbnp",
                        "cysc_s_1",
                        "il_6",
                         ]

    elif exp_name == "CLIP_SCORE":
        features_list = ["CLIP_Score",
                         ]

    elif exp_name == "IABP_SCORE":
        features_list = ["had_dem_iabpscore_c"]

    elif exp_name == "SAPS_SCORE":
        features_list = ["icu_the_saps2score_num_x"]

    if exp_name == "Admission_v2":
        # Baseline Model features
        features_list = [
                        # Patient Information
                        "had_dem_age_yr",           # 1 Age
                        "had_dem_male_yn",          # 2 Gender
                        "had_pex_weight_kg",        # 3 Weight
                        "had_pex_height_cm",        # 4 Height

                        # Comorbilities
                        "combined_variable",        # 5  Combined variable: Previous myocardial infarction, Previous PCI, Previous CABG     # noqa
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

    if exp_name == "24hs_v2":
        # Baseline Model features
        features_list = [
                        # Admission Features
                        # Patient Information
                        "had_dem_age_yr",           # 1 Age
                        "had_dem_male_yn",          # 2 Gender
                        "had_pex_weight_kg",        # 3 Weight
                        "had_pex_height_cm",        # 4 Height

                        # Comorbilities
                        "combined_variable",        # 5  Combined variable: Previous myocardial infarction, Previous PCI, Previous CABG     # noqa
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

    else:
        RuntimeError("Experiment not set")

    return features_list
