"""
collinearity.py - Phase 2: removing redundant (highly correlated) features.

Algorithm from the training kit:
    1. absolute Pearson correlation matrix
    2. keep only the upper triangle (each pair once, no diagonal)
    3. find pairs above the threshold (0.80)
    4. drop one feature from each pair

Step 4 rule: this dataset has NO target column, so "keep the one closer to the
target" cannot be used. Instead the feature with the higher AVERAGE correlation
to all other features is dropped - it is the more redundant of the two.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def find_collinear_drops(features: pd.DataFrame, threshold: float) -> tuple[list[str], pd.DataFrame]:
    """Return (columns to drop, table describing every flagged pair and the decision)."""
    corr = features.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack()
    pairs = pairs[pairs > threshold].sort_values(ascending=False)

    # Average correlation of each feature with every OTHER feature.
    avg_corr = (corr.sum() - 1) / (len(corr) - 1)

    dropped: list[str] = []
    rows = []
    for (a, b), value in pairs.items():  # loops over a few flagged PAIRS only
        if a in dropped or b in dropped:
            continue  # pair already resolved by an earlier drop
        drop = a if avg_corr[a] > avg_corr[b] else b
        dropped.append(drop)
        rows.append({
            "feature_a": a,
            "feature_b": b,
            "abs_corr": round(float(value), 4),
            "a_avg_corr": round(float(avg_corr[a]), 4),
            "b_avg_corr": round(float(avg_corr[b]), 4),
            "dropped": drop,
        })
    return dropped, pd.DataFrame(rows)
