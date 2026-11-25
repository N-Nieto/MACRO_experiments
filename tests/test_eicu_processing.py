"""Tests for lib.eICU_processing helpers (synthetic data only)."""
import numpy as np
import pandas as pd
import pytest
from lib.eICU_processing import (
    EICU_ADMISSION_NAME_MAPPING,
    EICU_FULL_NAME_MAPPING,
    admission_name_matching,
    eICU_admission_heart_function,
    full_model_name_matching,
)
from lib.experiment_definitions import get_features


def _fake_eicu(columns):
    return pd.DataFrame(np.arange(3 * len(columns)).reshape(3, -1), columns=columns)


@pytest.mark.parametrize(
    "func, mapping, exp_name",
    [
        (admission_name_matching, EICU_ADMISSION_NAME_MAPPING, "Admission"),
        (full_model_name_matching, EICU_FULL_NAME_MAPPING, "24hs"),
    ],
)
def test_name_matching_follows_experiment_features(func, mapping, exp_name):
    extra = ["patientunitstayid_copy", "p_rf_smoker_yn"]  # must be dropped
    lact = ["admission_lactate", "icu_lab_lact8hpci_x", "hpe_proc_mechs_yn"]
    eicu = _fake_eicu(list(mapping) + extra + (lact if exp_name == "24hs" else []))
    out = func(eicu)

    assert list(out.columns) == get_features(exp_name)
    # every mapped eICU column keeps its values under the CULPRIT name
    for eicu_name, culprit_name in mapping.items():
        assert out[culprit_name].tolist() == eicu[eicu_name].tolist()
    # features that eICU does not have are all NaN
    unavailable = set(get_features(exp_name)) - set(mapping.values()) - set(lact)
    assert unavailable and out[list(unavailable)].isna().all().all()
    # the input is not modified in place
    assert list(eicu.columns)[: len(mapping)] == list(mapping)


def test_heart_function_raw_file_branch(tmp_path):
    vitals = pd.DataFrame(
        {
            "patientunitstayid": [1, 1, 1, 2, 3],
            "observationoffset": [-10, 10, 100, 0, 0],
            "heartrate": [100, 120, 50, 300, 80],  # 300 is implausible
            "systemicsystolic": [90, 110, 80, 85, 95],
            "systemicdiastolic": [50, 60, 40, 45, 55],
        }
    )
    vitals.to_csv(tmp_path / "vitalPeriodic.csv.gz", index=False)
    rng = [(20, 200, "heartrate"), (20, 200, "systemicsystolic"), (20, 200, "systemicdiastolic")]
    out = eICU_admission_heart_function(
        str(tmp_path) + "/", rng, selected_patients=pd.Series([1, 2]),
        time_before_cut_off=30,
    ).set_index("patientunitstayid")
    assert set(out.index) == {1, 2}  # patient 3 not selected
    assert out.loc[1, "heartrate"] == 110  # median of 100, 120 (offset 100 cut)
    assert out["heartrate"].isna().loc[2]  # implausible value removed


def test_heart_function_without_patient_filter(tmp_path):
    vitals = pd.DataFrame(
        {
            "patientunitstayid": [1, 2],
            "observationoffset": [0, 0],
            "heartrate": [90, 95],
            "systemicsystolic": [100, 105],
            "systemicdiastolic": [60, 65],
        }
    )
    vitals.to_csv(tmp_path / "vitalPeriodic.csv.gz", index=False)
    out = eICU_admission_heart_function(str(tmp_path) + "/", [])
    assert set(out["patientunitstayid"]) == {1, 2}


def test_eicu_lab_units():
    from lib.eICU_processing import harmonize_eicu_labs

    lab = pd.DataFrame({"CPK": [300.0], "CRP": [1.5], "creatinine": [1.0], "ALT (SGPT)": [60.0],
                        "troponin - T": [0.5], "BNP": [800.0], "lactate": [3.0]})
    out = harmonize_eicu_labs(lab)
    assert out.loc[0, "CPK"] == 300.0  # U/L
    assert out.loc[0, "CRP"] == 15.0  # mg/dL -> mg/L
    assert out.loc[0, "creatinine"] == 88.42
    assert out.loc[0, "ALT (SGPT)"] == pytest.approx(1.0)
    assert out.loc[0, "troponin - T"] == 500.0
    assert out.loc[0, "lactate"] == 3.0
    assert out.loc[0, "BNP"] == 800.0  # converted later, eicu_bnp_to_ntprobnp


def test_bnp_to_ntprobnp_options():
    from lib.eICU_processing import (
        eicu_bnp_to_ntprobnp,
        ishihara_ntprobnp,
        japanese_egfr,
    )

    # Ishihara 2022 reproduces the paper's medians (BNP 55.5 -> ~200 pg/mL)
    assert 150 < ishihara_ntprobnp(55.5, 23.4, 60.9) < 300
    assert japanese_egfr(1.0, 60, False) == pytest.approx(194 * 60**-0.287)
    df = pd.DataFrame({
        "BNP": [1000.0, np.nan, 500.0],
        "creatinine": [88.42, 100.0, np.nan],  # umol/L
        "age": [60, 70, 80], "gender": [1, 0, 1],
        "admissionheight": [170.0, 160.0, np.nan], "admissionweight": [70.0, 60.0, 80.0],
    })
    out = eicu_bnp_to_ntprobnp(df)
    egfr = 194 * 60**-0.287
    bmi = 70 / 1.7**2
    assert out.loc[0, "BNP"] == pytest.approx(10 ** (1.21 + 1.03 * 3 - 0.009 * bmi - 0.007 * egfr))
    assert out["BNP"].isna().loc[1]
    assert out["BNP"].gt(500).loc[2]  # missing covariates -> cohort medians
    assert eicu_bnp_to_ntprobnp(df, "ratio").loc[0, "BNP"] == 6250.0
    assert eicu_bnp_to_ntprobnp(df, "as_is").loc[0, "BNP"] == 1000.0
    assert "BNP" not in eicu_bnp_to_ntprobnp(df, "missing")
    with pytest.raises(ValueError):
        eicu_bnp_to_ntprobnp(df, "quantile")


def test_quantile_match():
    from lib.data_load_utils import quantile_match

    ref = pd.Series(np.arange(1, 1001, dtype=float))
    values = pd.Series([1.0, np.nan, 100.0, 10.0])
    out = quantile_match(values, ref)
    assert np.isnan(out[1])
    assert out[0] < out[3] < out[2]  # order kept
    np.testing.assert_allclose(out.dropna().sort_values(), np.quantile(ref, [1 / 6, 3 / 6, 5 / 6]))
