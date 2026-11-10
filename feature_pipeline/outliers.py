"""
outliers.py - Phase 1: finding and neutralising outliers.

Two detection rules from the training kit:
    IQR rule  - outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]  (robust, works on skewed data)
    Z-score   - more than 3 standard deviations from the mean (assumes a bell shape)
Neutralisation: winsorization with numpy.clip (values are capped, no rows deleted).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def iqr_bounds(series: pd.Series, multiplier: float = config.IQR_MULTIPLIER) -> tuple[float, float]:
    """Lower and upper fence of the IQR rule."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return float(q1 - multiplier * iqr), float(q3 + multiplier * iqr)


def outlier_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Count outliers per column with both rules, plus skewness (vectorised)."""
    rows = []
    for col in columns:  # loops over a handful of COLUMNS, never over rows
        s = df[col].astype(float)
        low, high = iqr_bounds(s)
        z = (s - s.mean()) / s.std()
        rows.append({
            "column": col,
            "skewness": round(float(s.skew()), 3),
            "iqr_lower": round(low, 2),
            "iqr_upper": round(high, 2),
            "iqr_outliers": int(((s < low) | (s > high)).sum()),
            "zscore_outliers": int((z.abs() > config.ZSCORE_THRESHOLD).sum()),
            "max_value": round(float(s.max()), 2),
        })
    return pd.DataFrame(rows).set_index("column")


def winsorize(series: pd.Series, lower: float, upper: float) -> pd.Series:
    """Cap values at the given bounds with numpy.clip; every row is kept."""
    return pd.Series(np.clip(series.to_numpy(dtype=float), lower, upper), index=series.index)
