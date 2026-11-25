"""Range corrections in lib.unit_harmonization must modify the data (pandas 2 and 3)."""
import numpy as np
import pandas as pd
import pytest
from lib.unit_harmonization import (
    hematocrit_unit_harmonization,
    lactate_in_biological_range,
    white_blood_count_unit_harmonization,
)


def test_lactate_biological_range():
    df = pd.DataFrame({"lac": [0.5, 30.0, 2000.0, 5.0, np.nan]})
    out = lactate_in_biological_range(df, "lac")
    np.testing.assert_allclose(out["lac"], [0.5, 0.3, 2.0, 5.0, np.nan])


def test_hematocrit_range():
    df = pd.DataFrame({"icu_lab_hct_x_x": [35.0, 0.4, 3500.0], "icu_lab_hctunit_c": [1, 1, 1]})
    out = hematocrit_unit_harmonization(df)
    np.testing.assert_allclose(out["hematocrit"], [35.0, 40.0, 35.0])


def test_white_cell_count_range():
    df = pd.DataFrame({"icu_lab_wct_x_x": [12.0, 12000.0, 150.0]})
    out = white_blood_count_unit_harmonization(df)
    np.testing.assert_allclose(out["white_cell_count"], [12.0, 12.0, 1.5])


# --- unit-code conversions (codes identified in output/data_checks/lab_units_report.md)
from lib.unit_harmonization import (  # noqa: E402
    ck_unit_harmonization,
    convert_by_unit_code,
    creatine_unit_harmonization,
    crp_unit_harmonization,
    glucose_unit_harmonization,
    lactate_unit_harmonization,
)


def test_convert_by_unit_code_unknown_codes_become_nan(capsys):
    values = pd.Series([1.0, 2.0, 3.0, np.nan])
    codes = pd.Series([1, 2, 9, 9])
    out = convert_by_unit_code(values, codes, {1: 10.0, 2: np.nan}, "x")
    np.testing.assert_allclose(out, [10.0, np.nan, np.nan, np.nan])
    assert "2 values with unknown unit" in capsys.readouterr().out


def test_ck_uses_unit_code_not_value():
    df = pd.DataFrame({"icu_lab_ck_x": [5.0, 300.0, 1.0], "icu_lab_ckunit_c": [1, 2, 2], "ck": [np.nan, 2.0, np.nan]})
    out = ck_unit_harmonization(df)
    np.testing.assert_allclose(out["icu_lab_ck_x"], [300.0, 300.0, 1.0])  # ukat/L * 60, U/L as is
    np.testing.assert_allclose(out["ck_filled"], [300.0, 120.0, 1.0])


def test_lactate_mg_dl_to_mmol_l_and_inferred_code():
    df = pd.DataFrame({
        "icu_lab_lactunit_c_x": [1, 2, 3, 3],
        "admission_lactate": [3.0, 90.08, 4.0, 36.0],
        "icu_lab_lact8hpci_x": [2.0, 45.04, 4.0, 27.0],
        "icu_lab_lact16hpci_x": [np.nan, 18.016, 4.0, 18.0],
        "icu_lab_lact24hpci_x": [1.0, 9.008, 4.0, 18.0],
    })
    out = lactate_unit_harmonization(df.copy())
    np.testing.assert_allclose(out["admission_lactate"][:2], [3.0, 10.0])
    np.testing.assert_allclose(out["icu_lab_lact8hpci_x"][:2], [2.0, 5.0])
    # code 3 pooled median 18 -> closer to the known median (~2.5) as mg/dL
    np.testing.assert_allclose(out["admission_lactate"][2:], [4.0 / 9.008, 36.0 / 9.008])
    # if code 3 looks like mmol/L it is kept as is
    df.loc[3, ["admission_lactate", "icu_lab_lact8hpci_x", "icu_lab_lact16hpci_x", "icu_lab_lact24hpci_x"]] = 3.0
    out = lactate_unit_harmonization(df.copy())
    np.testing.assert_allclose(out["admission_lactate"][2:], [4.0, 3.0])


@pytest.mark.parametrize("raw, expected", [(1.13, 1.13 * 88.42), (0.1, 100.0), (99.0, 99.0)])
def test_creatinine_undocumented_code_is_inferred(raw, expected):
    df = pd.DataFrame({
        "icu_lab_crea_x": [110.0, 90.0, 1.2, raw],
        "icu_lab_creaunit_c": [1, 1, 3, 2],
        "cre": [np.nan] * 4,
    })
    out = creatine_unit_harmonization(df)
    assert out["creatine"].iloc[3] == pytest.approx(expected)


def test_creatinine_crp_glucose():
    df = pd.DataFrame({
        "icu_lab_crea_x": [110.0, 1.0, 0.1], "icu_lab_creaunit_c": [1, 3, 2], "cre": [np.nan] * 3,
        "icu_lab_crp_x": [0.8, 8.0, np.nan], "icu_lab_crpunit_c": [1, 2, 2],
        "icu_lab_glc_x": [180.0, 10.0, np.nan], "icu_lab_glcunit_c": [1, 2, 1], "gluc": [np.nan, np.nan, 10.0],
    })
    out = glucose_unit_harmonization(crp_unit_harmonization(creatine_unit_harmonization(df)))
    np.testing.assert_allclose(out["creatine"], [110.0, 88.42, 100.0])  # code 2: 0.1 -> mmol/L
    np.testing.assert_allclose(out["crp"], [8.0, 8.0, np.nan])  # mg/L
    np.testing.assert_allclose(out["glucose"], [180.0, 180.182, np.nan])
    np.testing.assert_allclose(out["glucose_filled"], [180.0, 180.182, 180.182])
