"""Aggregate comparison of feature distributions between two cohorts."""

import numpy as np
import pandas as pd


def is_binary(values: pd.Series) -> bool:
    """True for 0/1 features (missing values ignored)."""
    present = values.dropna()
    return len(present) > 0 and bool(present.isin([0, 1]).all())


def use_log_scale(*columns: pd.Series) -> bool:
    """Strictly positive, right-skewed features (e.g. CK, NT-proBNP) are compared on a log scale."""
    present = pd.concat([c.dropna() for c in columns])
    if present.empty or (present <= 0).any():
        return False
    return bool(present.quantile(0.95) > 5 * present.median())


def standardized_difference(a: pd.Series, b: pd.Series, binary: bool, log: bool = False) -> float:
    """
    Standardized mean difference (a - b) / pooled SD. For binary features the
    SD of a proportion is used; skewed features (log=True) are compared on
    log10. |SMD| > 0.1 is a common threshold for a relevant imbalance.
    """
    a, b = a.dropna().astype(float), b.dropna().astype(float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    if binary:
        p1, p2 = a.mean(), b.mean()
        sd = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
        diff = p1 - p2
    else:
        if log:
            a, b = a.apply(np.log10), b.apply(np.log10)
        sd = np.sqrt((a.var() + b.var()) / 2)
        diff = a.mean() - b.mean()
    return float(diff / sd) if sd > 0 else float("nan")


def summarize(values: pd.Series, binary: bool, min_group: int = 10) -> str:
    """'n positive (%)' or 'median [Q1, Q3]'; groups smaller than min_group are not described."""
    present = values.dropna()
    if len(present) < min_group:
        return f"n<{min_group}" if len(present) else "not available"
    if binary:
        return f"{int(present.sum())} ({100 * present.mean():.0f}%)"
    q1, q2, q3 = present.quantile([0.25, 0.5, 0.75])
    return f"{q2:.3g} [{q1:.3g}, {q3:.3g}]"


def compare_cohorts(
    a: pd.DataFrame, b: pd.DataFrame, features: list[str], min_group: int = 10
) -> pd.DataFrame:
    """
    One row per feature: availability (% missing), summary and standardized
    difference of cohort a vs cohort b. Aggregate values only.
    """
    rows = []
    for f in features:
        col_a = a[f] if f in a else pd.Series(dtype=float)
        col_b = b[f] if f in b else pd.Series(dtype=float)
        binary = is_binary(pd.concat([col_a, col_b]))
        log = not binary and use_log_scale(col_a, col_b)
        rows.append({
            "feature": f,
            "type": "binary" if binary else ("continuous (log)" if log else "continuous"),
            "a_missing_%": 100 * col_a.isna().mean() if len(col_a) else 100.0,
            "a_summary": summarize(col_a, binary, min_group),
            "b_missing_%": 100 * col_b.isna().mean() if len(col_b) else 100.0,
            "b_summary": summarize(col_b, binary, min_group),
            "SMD": standardized_difference(col_a, col_b, binary, log),
        })
    return pd.DataFrame(rows)
