"""Tests for lib.experiment_definitions."""
import pytest
from lib.experiment_definitions import get_features, get_important_features


@pytest.mark.parametrize(
    "importance_name, feature_set, n_expected",
    [("Admission", "Admission", 17), ("Full", "24hs", 15), ("Admission_less_features", "Admission_less_features", 10)],
)
def test_important_features_are_valid(importance_name, feature_set, n_expected):
    ranked = get_important_features(importance_name)
    assert len(ranked) == n_expected
    assert len(set(ranked)) == len(ranked)  # no duplicates
    assert set(ranked) <= set(get_features(feature_set))


@pytest.mark.parametrize("func", [get_features, get_important_features])
def test_unknown_experiment_raises(func):
    with pytest.raises(RuntimeError):
        func("does_not_exist")
