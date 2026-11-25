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
    assert np.isnan(out.loc[2, "heartrate"])  # implausible value removed


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
