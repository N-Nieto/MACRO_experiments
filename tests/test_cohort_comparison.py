"""lib.cohort_comparison (aggregate CULPRIT-SHOCK vs eICU comparison)."""
import numpy as np
import pandas as pd
import pytest
from lib.cohort_comparison import compare_cohorts, is_binary, standardized_difference, summarize, use_log_scale


def test_binary_and_log_detection():
    assert is_binary(pd.Series([0, 1, np.nan, 1]))
    assert not is_binary(pd.Series([0, 1, 2]))
    skewed = pd.Series(np.exp(np.random.default_rng(0).normal(5, 1.5, 500)))
    assert use_log_scale(skewed) and not use_log_scale(pd.Series(np.arange(50, 150)))
    assert not use_log_scale(pd.Series([-1.0, 5.0, 1000.0]))  # not strictly positive


def test_standardized_difference():
    a, b = pd.Series([0, 0, 1, 1]), pd.Series([0, 0, 0, 1])
    expected = (0.5 - 0.25) / np.sqrt((0.25 + 0.1875) / 2)
    assert standardized_difference(a, b, binary=True) == pytest.approx(expected)
    x = pd.Series([1.0, 2.0, 3.0, np.nan])
    assert standardized_difference(x, x, binary=False) == 0
    assert np.isnan(standardized_difference(x, pd.Series([1.0]), binary=False))


def test_summaries_are_aggregate_only():
    assert summarize(pd.Series([1, 0] * 10), binary=True) == "10 (50%)"
    assert summarize(pd.Series(range(5), dtype=float), binary=False) == "n<10"
    assert summarize(pd.Series([np.nan] * 3), binary=False) == "not available"
    t = compare_cohorts(pd.DataFrame({"a": range(20), "b": [0, 1] * 10}), pd.DataFrame({"b": [1] * 20}), ["a", "b"])
    assert t.set_index("feature").loc["a", "b_summary"] == "not available"
    assert t.set_index("feature").loc["b", "type"] == "binary"
