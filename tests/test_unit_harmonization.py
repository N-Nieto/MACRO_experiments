"""Range corrections in lib.unit_harmonization must modify the data (pandas 2 and 3)."""
import numpy as np
import pandas as pd
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
