"""Tests for lib.data_processing.remove_random_features_fix_number."""
import numpy as np
import pandas as pd
import pytest
from lib.data_processing import remove_random_features_fix_number

BASIC = ["age", "sex"]


def _complete_df(n_rows=200, n_other=17):
    cols = BASIC + [f"f{i}" for i in range(n_other)]
    return pd.DataFrame(np.ones((n_rows, len(cols))), columns=cols)


@pytest.mark.parametrize("k", [1, 2, 5, 9, 17])
def test_exactly_k_features_removed_per_row(k):
    df = _complete_df()
    out = remove_random_features_fix_number(df, k, BASIC, np.random.default_rng(0))
    assert (out.isna().sum(axis=1) == k).all()
    assert out[BASIC].notna().all().all()  # basic features never removed


def test_input_not_modified_and_no_accumulation():
    # The old implementation wrote NaNs into its input; the experiment loop
    # reused that frame, so step k removed ~k(k+1)/2 features instead of k.
    df = _complete_df()
    rng = np.random.default_rng(0)
    for k in range(1, 8):
        out = remove_random_features_fix_number(df, k, BASIC, rng)
        assert (out.isna().sum(axis=1) == k).all()
    assert df.notna().all().all()


def test_reproducible_with_seed_and_varies_across_rows():
    df = _complete_df()
    a = remove_random_features_fix_number(df, 3, BASIC, np.random.default_rng(23))
    b = remove_random_features_fix_number(df, 3, BASIC, np.random.default_rng(23))
    pd.testing.assert_frame_equal(a, b)
    patterns = {tuple(row) for row in a.isna().to_numpy()}
    assert len(patterns) > 50  # each row gets its own random choice


def test_k_is_capped_and_basic_can_be_an_index():
    df = _complete_df(n_rows=10, n_other=4)
    out = remove_random_features_fix_number(df, 10, pd.Index(BASIC), np.random.default_rng(1))
    assert (out.isna().sum(axis=1) == 4).all()


def test_zero_removed_returns_copy():
    df = _complete_df(n_rows=5)
    out = remove_random_features_fix_number(df, 0, BASIC)
    pd.testing.assert_frame_equal(out, df)
    assert out is not df
