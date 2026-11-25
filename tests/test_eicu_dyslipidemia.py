"""Tests for lib.eICU_data_loading.load_eicu_dyslipidemia (synthetic data only)."""
import pandas as pd
from lib.eICU_data_loading import load_eicu_dyslipidemia


def _write_lab(tmp_path, rows):
    lab = pd.DataFrame(
        rows, columns=["patientunitstayid", "labresultoffset", "labname", "labresult"]
    )
    lab.to_csv(tmp_path / "lab.csv.gz", index=False)
    return str(tmp_path)


def _flags(df):
    return dict(zip(df["patientunitstayid"], df["dyslipidemia"]))


def test_normal_lipids_are_not_flagged(tmp_path):
    # Previously all three were flagged (zero-dilution + missing HDL counted as 0)
    root = _write_lab(
        tmp_path,
        [
            (1, -100, "triglycerides", 120.0),
            (1, -90, "HDL", 55.0),
            (2, -50, "triglycerides", 100.0),  # HDL never measured
            (3, -20, "LDL", 120.0),
            (3, -10, "HDL", 60.0),
        ],
    )
    assert _flags(load_eicu_dyslipidemia(root, [1, 2, 3])) == {1: 0, 2: 0, 3: 0}


def test_each_criterion_flags(tmp_path):
    root = _write_lab(
        tmp_path,
        [
            (4, -5, "triglycerides", 200.0),  # > 150 mg/dL
            (5, -5, "HDL", 35.0),  # < 40 mg/dL
            (6, -5, "LDL", 195.0),  # > 190 mg/dL
            (7, -5, "triglycerides", 140.0),  # borderline normal
        ],
    )
    assert _flags(load_eicu_dyslipidemia(root, [4, 5, 6, 7])) == {
        4: 1,
        5: 1,
        6: 1,
        7: 0,
    }


def test_post_admission_and_unselected_values_are_ignored(tmp_path):
    root = _write_lab(
        tmp_path,
        [
            (8, -5, "HDL", 60.0),
            (8, 30, "HDL", 10.0),  # after ICU admission -> ignored
            (9, -5, "HDL", 10.0),  # patient not selected -> ignored
        ],
    )
    assert _flags(load_eicu_dyslipidemia(root, [8])) == {8: 0}


def test_repeated_values_are_averaged(tmp_path):
    root = _write_lab(
        tmp_path,
        [(10, -50, "triglycerides", 100.0), (10, -10, "triglycerides", 220.0)],
    )
    # mean = 160 mg/dL > 150
    assert _flags(load_eicu_dyslipidemia(root, [10])) == {10: 1}
