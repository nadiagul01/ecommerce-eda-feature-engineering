"""
missing.py - Phase 1: measuring missing data and choosing a treatment.

Contains:
    decide_strategy()            - the Missing Data Decision Matrix as a function
    missing_report()             - % missing + chosen strategy for every column
    run_imputation_experiment()  - Mean vs Median vs KNN, scored against true values
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from . import config


# -----------------------------------------------------------------------------
# Decision matrix
# -----------------------------------------------------------------------------
def decide_strategy(missing_pct: float, column: str) -> str:
    """Return the treatment for one column.

    Possible results: "none", "structural_fill", "drop_rows", "stat_impute", "knn".
    """
    if missing_pct == 0:
        return "none"
    if column in config.STRUCTURAL_MISSING_FILL:
        # The blank has a known meaning (e.g. no coupon used) -> label it, don't estimate it.
        return "structural_fill"
    if missing_pct < config.DROP_ROWS_MAX_PCT:
        return "drop_rows"
    if missing_pct <= config.STAT_IMPUTE_MAX_PCT:
        return "stat_impute"
    return "knn"


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Missing count, missing % and the chosen strategy for every column."""
    counts = df.isna().sum()
    pct = (counts / len(df) * 100).round(2)
    report = pd.DataFrame({"missing_count": counts, "missing_pct": pct})
    report["strategy"] = [decide_strategy(p, c) for c, p in pct.items()]
    report.index.name = "column"
    return report.sort_values("missing_pct", ascending=False)


# -----------------------------------------------------------------------------
# Imputation experiment
# -----------------------------------------------------------------------------
def _knn_fill(numeric: pd.DataFrame, column: str) -> pd.Series:
    """KNN imputation on standardised numeric columns (scale matters for distance)."""
    scaler = StandardScaler().fit(numeric)  # NaNs are ignored while fitting
    filled = KNNImputer(n_neighbors=config.KNN_NEIGHBOURS).fit_transform(scaler.transform(numeric))
    restored = scaler.inverse_transform(filled)
    return pd.Series(restored[:, numeric.columns.get_loc(column)], index=numeric.index)


def run_imputation_experiment(
    df: pd.DataFrame,
    column: str = config.EXPERIMENT_COLUMN,
    share: float = config.EXPERIMENT_MISSING_SHARE,
    repeats: int = config.EXPERIMENT_REPEATS,
    seed: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, pd.Series], pd.Series]:
    """Hide `share` of a complete column at random, impute it, compare with the truth.

    KNN finds neighbours using the other numeric columns (Quantity, UnitPrice,
    TotalPrice). The hidden values are never visible to any method, so the
    comparison is fair.

    Returns:
        summary  - mean MAE / RMSE / std / KS over all repeats, one row per method
        example  - the filled column for each method from the first repeat (for plots)
        truth    - the original complete column
    """
    truth = df[column].astype(float)
    knn_inputs = [c for c in config.NUMERIC_COLUMNS if c in df.columns]
    rng = np.random.default_rng(seed)
    n_hide = int(round(len(df) * share))
    rows, example = [], {}

    for r in range(repeats):
        hidden_idx = rng.choice(df.index.to_numpy(), size=n_hide, replace=False)
        masked = truth.copy()
        masked.loc[hidden_idx] = np.nan
        observed = masked.dropna()

        numeric = df[knn_inputs].astype(float).copy()
        numeric[column] = masked
        filled = {
            "Mean": masked.fillna(observed.mean()),
            "Median": masked.fillna(observed.median()),
            "KNN (k=5)": masked.fillna(_knn_fill(numeric, column)),
        }
        for method, series in filled.items():
            error = series.loc[hidden_idx] - truth.loc[hidden_idx]
            rows.append({
                "method": method,
                "MAE_on_hidden": np.abs(error).mean(),
                "RMSE_on_hidden": np.sqrt((error ** 2).mean()),
                "std_after": series.std(),
                "KS_vs_true": ks_2samp(truth, series).statistic,
            })
        if r == 0:
            example = filled

    summary = pd.DataFrame(rows).groupby("method", sort=False).mean().round(3)
    truth_row = pd.DataFrame(
        {"MAE_on_hidden": 0.0, "RMSE_on_hidden": 0.0, "std_after": round(truth.std(), 3), "KS_vs_true": 0.0},
        index=pd.Index(["True values (reference)"], name="method"),
    )
    return pd.concat([truth_row, summary]), example, truth
