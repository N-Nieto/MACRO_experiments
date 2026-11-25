"""Tests for eICU outcome labels (eicu_outcomes) and load_eICU (synthetic data)."""
import numpy as np
import pandas as pd
import pytest
from lib.data_load_utils import load_eICU
from lib.eICU_processing import EICU_OUTCOMES, eicu_outcomes

DAY = 1440


def _patients():
    # id, hosp status, unit status, hosp admit offset, hosp discharge offset
    rows = [
        (1, 0, 0, -100, 10 * DAY),  # survived
        (2, 1, 1, 0, 0.5 * DAY),  # died in ICU < 24 h
        (3, 1, 1, 0, 10 * DAY),  # died day 10
        (4, 1, 0, 0, 40 * DAY),  # died in hospital day 40 (after ICU)
        (5, 1, 0, -5 * DAY, 28 * DAY),  # day 28 from ICU, day 33 from hospital
        (6, np.nan, np.nan, 0, 3 * DAY),  # unknown status
        (7, 1, 1, 0, 1 * DAY),  # exactly 24 h -> not an early death
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "patientunitstayid",
            "hospitaldischargestatus",
            "unitdischargestatus",
            "hospitaladmitoffset",
            "hospitaldischargeoffset",
        ],
    )


def test_outcome_labels():
    out = eicu_outcomes(_patients())
    assert set(EICU_OUTCOMES) <= set(out.columns)
    col = lambda c: out[c].tolist()  # noqa: E731
    nan = np.nan
    np.testing.assert_array_equal(col("in_hospital_death"), [0, 1, 1, 1, 1, nan, 1])
    np.testing.assert_array_equal(col("in_hospital_death_30d_icu"), [0, 1, 1, 0, 1, nan, 1])
    np.testing.assert_array_equal(col("in_hospital_death_30d_hospital"), [0, 1, 1, 0, 0, nan, 1])
    np.testing.assert_array_equal(col("icu_death"), [0, 1, 1, 0, 0, nan, 1])
    np.testing.assert_array_equal(col("death_within_24h"), [0, 1, 0, 0, 0, nan, 0])


def test_duplicate_stays_are_rejected():
    df = pd.concat([_patients(), _patients().iloc[:1]])
    with pytest.raises(ValueError):
        eicu_outcomes(df)


@pytest.fixture
def eicu_dir(tmp_path):
    root = tmp_path / "eicu-collaborative-research-database-2.0" / "preprocessed_MACRO"
    root.mkdir(parents=True)
    outcomes = eicu_outcomes(_patients())
    outcomes.to_csv(root / "outcomes_CICU.csv")
    ids = outcomes.index
    for features in ("Admission", "Full"):
        X = pd.DataFrame(
            {"age": np.arange(len(ids)), "lactate": np.arange(len(ids)) * 2.0}, index=ids
        )
        # stored in a different row order than the outcomes on purpose
        X.iloc[::-1].to_csv(root / f"X_{features}_CICU.csv")
    return tmp_path


def test_load_eicu_aligns_by_patient_and_columns(eicu_dir):
    X_culprit = pd.DataFrame(columns=["lactate", "not_in_eicu", "age"])
    X, y = load_eICU(eicu_dir, "Admission", X_culprit)
    assert list(X.columns) == ["lactate", "not_in_eicu", "age"]
    assert X["not_in_eicu"].isna().all()
    # unknown status (patient 6) removed, early death (2) kept for Admission
    assert sorted(X.index) == [1, 2, 3, 4, 5, 7]
    expected = eicu_outcomes(_patients()).loc[X.index, "in_hospital_death"]
    np.testing.assert_array_equal(y, expected.to_numpy())


def test_full_excludes_early_deaths_by_default(eicu_dir):
    X_culprit = pd.DataFrame(columns=["age", "lactate"])
    X, _ = load_eICU(eicu_dir, "Full", X_culprit)
    assert 2 not in X.index
    X, _ = load_eICU(eicu_dir, "Full", X_culprit, exclude_early_deaths=False)
    assert 2 in X.index


def test_alternative_outcome_and_validation(eicu_dir):
    X_culprit = pd.DataFrame(columns=["age"])
    X, y = load_eICU(eicu_dir, "Admission", X_culprit, outcome="in_hospital_death_30d_icu")
    assert dict(zip(X.index, y))[4] == 0  # died on day 40 -> survivor at 30 d
    with pytest.raises(ValueError):
        load_eICU(eicu_dir, "Admission", X_culprit, outcome="30_day_mortality")
    with pytest.raises(ValueError):
        load_eICU(eicu_dir, "24hs", X_culprit)
