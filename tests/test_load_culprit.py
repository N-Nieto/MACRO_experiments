"""load_CULPRIT_data on a synthetic CULPRIT-SHOCK export."""
import pytest
from lib.data_load_utils import load_CULPRIT_data
from lib.experiment_definitions import get_features
from synthetic_data import write_synthetic_culprit


@pytest.fixture(scope="module")
def culprit_dir(tmp_path_factory):
    folder = tmp_path_factory.mktemp("culprit")
    write_synthetic_culprit(folder)
    return folder


def test_loader_builds_all_model_features(culprit_dir):
    data = load_CULPRIT_data(culprit_dir)
    assert len(data) == 80  # one row per patient (day-1 labs only)
    assert data["patient_ID"].is_unique
    missing = [f for f in get_features("24hs") + ["CLIP_Score"] if f not in data]
    assert not missing


def test_harmonize_switch_keeps_raw_values(culprit_dir):
    raw = load_CULPRIT_data(culprit_dir, harmonize=False)
    assert "creatine" not in raw  # created by the harmonization
    assert "icu_lab_crea_x" in raw and "icu_lab_creaunit_c" in raw


@pytest.mark.parametrize(
    "feature, code_col",
    [("icu_lab_ck_x", "icu_lab_ckunit_c"), ("crp", "icu_lab_crpunit_c"),
     ("glucose", "icu_lab_glcunit_c"), ("creatine", "icu_lab_creaunit_c"),
     ("admission_lactate", "icu_lab_lactunit_c_x")],
)
def test_harmonized_values_share_one_unit(culprit_dir, feature, code_col):
    # After harmonization, patients reported in different units must be on
    # the same scale (medians per unit code within a factor of 2)
    data = load_CULPRIT_data(culprit_dir)
    medians = data.groupby(code_col)[feature].median().dropna()
    medians = medians[data.groupby(code_col)[feature].count() >= 2]
    assert len(medians) >= 2
    assert medians.max() / medians.min() < 2
