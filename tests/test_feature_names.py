"""lib.feature_names (readable feature names, replaces Shorten_table.xlsx)."""
import pandas as pd
from lib.data_processing import naming_for_shap
from lib.experiment_definitions import get_features
from lib.feature_names import FEATURE_LABELS, feature_label


def test_every_model_feature_has_a_unique_xgboost_safe_name():
    features = set(get_features("Admission")) | set(get_features("24hs"))
    assert features <= set(FEATURE_LABELS)
    labels = list(FEATURE_LABELS.values())
    assert len(labels) == len(set(labels))
    assert not any(c in label for label in labels for c in "[]<")


def test_naming_for_shap():
    X = pd.DataFrame({"had_dem_age_yr": [70.0], "p_rf_dyslip_yn": [1.0], "other": [0.0]})
    assert list(naming_for_shap(X).columns) == ["Age (years)", "Dyslipidemia", "other"]
    assert list(X.columns) == ["had_dem_age_yr", "p_rf_dyslip_yn", "other"]  # input unchanged
    assert feature_label("unknown_code") == "unknown_code"
