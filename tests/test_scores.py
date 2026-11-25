"""CardShock score (Supplementary Table 7)."""
import numpy as np
import pandas as pd
import pytest
from lib.scores import cardshock_score


def patient(**changes):
    base = {
        "had_dem_age_yr": 60, "had_sy_ams_yn": 0, "p_mh_mi_yn": 0, "p_mh_cabg_yn": 0,
        "hpr_echo_lvef_pct": 50, "admission_lactate": 1.0, "ckdepi": 90,
    }
    base.update(changes)
    return base


def score_of(**changes):
    return cardshock_score(pd.DataFrame([patient(**changes)])).iloc[0]


def test_minimum_is_the_acs_point():
    assert score_of() == 1


@pytest.mark.parametrize(
    "changes, points",
    [
        ({"had_dem_age_yr": 76}, 1), ({"had_dem_age_yr": 75}, 0),
        ({"had_sy_ams_yn": 1}, 1),
        ({"p_mh_mi_yn": 1}, 1), ({"p_mh_cabg_yn": 1}, 1), ({"p_mh_mi_yn": 1, "p_mh_cabg_yn": 1}, 1),
        ({"hpr_echo_lvef_pct": 39}, 1), ({"hpr_echo_lvef_pct": 40}, 0),
        ({"admission_lactate": 1.9}, 0), ({"admission_lactate": 2.0}, 1),
        ({"admission_lactate": 4.0}, 1), ({"admission_lactate": 4.1}, 2),
        ({"ckdepi": 61}, 0), ({"ckdepi": 60}, 1), ({"ckdepi": 30}, 1), ({"ckdepi": 29}, 2),
    ],
)
def test_component_points(changes, points):
    assert score_of(**changes) == 1 + points


def test_maximum_is_nine():
    assert score_of(had_dem_age_yr=80, had_sy_ams_yn=1, p_mh_mi_yn=1, hpr_echo_lvef_pct=20,
                    admission_lactate=8, ckdepi=20) == 9


@pytest.mark.parametrize("column", ["had_dem_age_yr", "had_sy_ams_yn", "hpr_echo_lvef_pct",
                                    "admission_lactate", "ckdepi"])
def test_missing_component_gives_missing_score(column):
    assert np.isnan(score_of(**{column: np.nan}))


def test_history_known_from_a_single_positive_item():
    assert score_of(p_mh_mi_yn=np.nan, p_mh_cabg_yn=1) == 2
    assert np.isnan(score_of(p_mh_mi_yn=np.nan, p_mh_cabg_yn=0))
