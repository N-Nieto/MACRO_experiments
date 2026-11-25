"""load_CULPRIT_data on a synthetic CULPRIT-SHOCK export."""
import pandas as pd
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


def test_therapy_flags_use_the_patients_own_admission_date(culprit_dir):
    # Ground truth computed independently, per patient
    sheets = pd.read_excel(culprit_dir / "CULPRIT-data_20210407.xlsx", sheet_name=None)
    data = load_CULPRIT_data(culprit_dir).set_index("patient_ID")

    def patient_id(df):
        return df["cor_ctr_center_id"].astype(str) + "-" + df["cor_pt_caseno_id"].astype(str)

    admission = sheets["patient_data"].set_index(patient_id(sheets["patient_data"]))["had_base_admis_date"]
    for sheet, column, name in [
        ("cathecholamine", "icu_catec_start_date", "cathecholamine_therapy"),
        ("endpoint", "icu_rrt_start_date", "renal_replacement_therapy"),
    ]:
        episodes = sheets[sheet].assign(patient_ID=patient_id(sheets[sheet]))
        days = (pd.to_datetime(episodes[column]) - pd.to_datetime(admission.loc[episodes["patient_ID"]]).to_numpy()).dt.days
        expected = days.isin([0, 1]).groupby(episodes["patient_ID"]).any().astype(int)
        assert (data.loc[expected.index, name] == expected).all(), name
        assert 0 < expected.mean() < 1
    # Patients missing from the endpoint sheet stay missing (NaN), as before
    assert data["renal_replacement_therapy"].isna().sum() > 0


def test_admission_lactate_is_pre_pci_first():
    from lib.data_load_utils import create_admission_lactate
    df = pd.DataFrame({"icu_lab_lactprepci_x": [5.0, None, None], "icu_lab_lactpopci_x": [2.0, 3.0, None]})
    out = create_admission_lactate(df)["admission_lactate"]
    assert out.iloc[0] == 5.0 and out.iloc[1] == 3.0 and pd.isna(out.iloc[2])
