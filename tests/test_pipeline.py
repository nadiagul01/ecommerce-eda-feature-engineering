"""
test_pipeline.py - Automated checks that every Project 1 requirement holds.

Run from the project root with:  pytest
"""

import hashlib

import numpy as np
import pandas as pd
import pandera.errors
import pytest

from feature_pipeline import FeaturePipeline, config
from feature_pipeline.features import cart_features, date_features
from feature_pipeline.missing import decide_strategy, run_imputation_experiment
from feature_pipeline.outliers import iqr_bounds, outlier_summary

ENGINEERED = ["OrderMonth", "OrderDayOfWeek", "IsWeekend", "DaysSinceStart",
              "HasCoupon", "OtherItemsInCart", "QuantityShareOfCart"]


# =============================================================================
# Fixtures - load the data and fit the pipeline once for all tests
# =============================================================================
@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return pd.read_excel(config.RAW_DATA_PATH)


@pytest.fixture(scope="module")
def fitted(raw):
    pipe = FeaturePipeline()
    analysis, model = pipe.fit_transform(raw)
    return pipe, analysis, model


# =============================================================================
# Phase 1 - missing data
# =============================================================================
@pytest.mark.parametrize(
    "pct, column, expected",
    [
        (0.0, "Quantity", "none"),
        (25.75, "CouponCode", "structural_fill"),  # blank coupon = no coupon used
        (4.99, "X", "drop_rows"),
        (5.0, "X", "stat_impute"),
        (20.0, "X", "stat_impute"),
        (20.01, "X", "knn"),
    ],
)
def test_decision_matrix(pct, column, expected):
    assert decide_strategy(pct, column) == expected


def test_no_missing_values_left(fitted):
    _, analysis, model = fitted
    assert analysis.isna().sum().sum() == 0
    assert model.isna().sum().sum() == 0


def test_coupon_blanks_become_none_and_flag(raw, fitted):
    _, analysis, _ = fitted
    blanks = raw["CouponCode"].isna().to_numpy()
    assert (analysis.loc[blanks, "CouponCode"] == "NONE").all()
    assert (analysis["HasCoupon"].to_numpy() == (~blanks).astype(int)).all()


def test_no_rows_lost(raw, fitted):
    _, analysis, model = fitted
    assert len(analysis) == len(model) == len(raw)


def test_knn_beats_mean_in_experiment(raw):
    summary, _, _ = run_imputation_experiment(raw, repeats=5)
    assert summary.loc["KNN (k=5)", "MAE_on_hidden"] < summary.loc["Mean", "MAE_on_hidden"]


def test_knn_branch_fills_numeric_gaps(raw):
    """Hide 25% of UnitPrice (> 20%) and check the missing-data step picks KNN and fills
    every gap with values inside the observed range. The missing-data step is called
    directly because the input contract (correctly) rejects orders without a price."""
    df = raw.copy()
    rng = np.random.default_rng(config.RANDOM_STATE)
    hide = rng.choice(df.index.to_numpy(), size=300, replace=False)
    df.loc[hide, "UnitPrice"] = np.nan

    pipe = FeaturePipeline()
    out = pipe._handle_missing(df, fitting=True)

    assert pipe.strategy_["UnitPrice"] == "knn"
    assert out["UnitPrice"].notna().all()
    assert out["UnitPrice"].between(raw["UnitPrice"].min(), raw["UnitPrice"].max()).all()


# =============================================================================
# Phase 1 - outliers
# =============================================================================
def test_iqr_bounds_simple_case():
    # Q1 = 2.75, Q3 = 6.25 (pandas linear interpolation) -> IQR = 3.5
    low, high = iqr_bounds(pd.Series([1, 2, 3, 4, 5, 6, 7, 8]))
    assert (low, high) == (-2.5, 11.5)


def test_outlier_summary_matches_data(raw):
    summary = outlier_summary(raw, config.NUMERIC_COLUMNS)
    assert summary.loc[["Quantity", "UnitPrice", "ItemsInCart"], "iqr_outliers"].sum() == 0
    assert summary.loc["TotalPrice", "iqr_outliers"] == 8


def test_totalprice_capped_only_in_model_table(raw, fitted):
    pipe, analysis, model = fitted
    low, high = pipe.outlier_bounds_["TotalPrice"]
    assert model["TotalPrice"].between(low - 1e-9, high + 1e-9).all()
    assert analysis["TotalPrice"].max() == raw["TotalPrice"].max()  # real revenue kept


# =============================================================================
# Phase 2 - features, encoding, collinearity
# =============================================================================
def test_feature_formulas():
    dates = pd.Series(pd.to_datetime(["2024-01-06", "2024-01-08"]))  # Saturday, Monday
    d = date_features(dates, pd.Timestamp("2024-01-01"))
    assert d["IsWeekend"].tolist() == [1, 0]
    assert d["OrderDayOfWeek"].tolist() == [5, 0]
    assert d["DaysSinceStart"].tolist() == [5, 7]
    c = cart_features(pd.Series([2, 5]), pd.Series([8, 5]))
    assert c["OtherItemsInCart"].tolist() == [6, 0]
    assert c["QuantityShareOfCart"].tolist() == [0.25, 1.0]


def test_at_least_three_engineered_features_in_model(fitted):
    pipe, _, _ = fitted
    assert sum(f in pipe.feature_columns_ for f in ENGINEERED) >= 3


def test_all_model_features_numeric(fitted):
    pipe, _, model = fitted
    assert all(pd.api.types.is_numeric_dtype(model[c]) for c in pipe.feature_columns_)
    assert not any(" " in c for c in model.columns)


def test_reference_category_dropped(fitted):
    pipe, _, model = fitted
    for col, ref in pipe.reference_category_.items():
        assert f"{col}_{ref}".replace(" ", "_") not in model.columns


def test_no_collinear_pairs_remain(fitted):
    pipe, _, model = fitted
    corr = model[pipe.feature_columns_].corr().abs().to_numpy(copy=True)
    np.fill_diagonal(corr, 0)
    assert corr.max() <= config.COLLINEARITY_THRESHOLD


def test_identifiers_removed_from_model(fitted):
    _, _, model = fitted
    assert not set(config.IDENTIFIER_COLUMNS_TO_DROP) & set(model.columns)


# =============================================================================
# Phase 3 - contracts and consistency
# =============================================================================
def test_transform_matches_fit_transform(raw, fitted):
    """Same input -> identical output: no training-serving skew."""
    pipe, analysis, model = fitted
    analysis_2, model_2 = pipe.transform(raw)
    pd.testing.assert_frame_equal(model_2, model)
    pd.testing.assert_frame_equal(analysis_2, analysis)


def test_transform_handles_new_orders(raw, fitted):
    pipe, _, _ = fitted
    new = raw.head(3).copy()
    new["OrderID"] = ["ORD900001", "ORD900002", "ORD900003"]
    new.loc[new.index[0], "Product"] = "Headphones"      # category never seen in training
    new.loc[new.index[1], "Date"] = pd.Timestamp("2026-01-15")
    _, model = pipe.transform(new)
    assert len(model) == 3 and model.isna().sum().sum() == 0
    assert list(model.columns) == [config.ID_COLUMN, *pipe.feature_columns_]
    # The unseen product is encoded like the reference category: all Product_* dummies = 0
    assert model.filter(like="Product_").iloc[0].sum() == 0


def test_input_contract_reports_all_failures(raw):
    bad = raw.copy()
    bad.loc[0, "Quantity"] = 0            # breaks Quantity >= 1
    bad.loc[1, "OrderID"] = "BAD-ID"      # breaks the ORD123 pattern
    bad.loc[2, "TotalPrice"] = 1.0        # breaks TotalPrice = Quantity x UnitPrice
    with pytest.raises(pandera.errors.SchemaErrors) as err:
        FeaturePipeline().fit_transform(bad)
    cases = err.value.failure_cases
    assert {"Quantity", "OrderID"} <= set(cases["column"].dropna())
    assert cases["check"].astype(str).str.contains("total_equals_quantity_times_price").any()


def test_transform_before_fit_raises(raw):
    with pytest.raises(RuntimeError):
        FeaturePipeline().transform(raw)


def test_raw_file_not_modified(raw):
    before = hashlib.sha256(config.RAW_DATA_PATH.read_bytes()).hexdigest()
    FeaturePipeline().fit_transform(raw)
    assert hashlib.sha256(config.RAW_DATA_PATH.read_bytes()).hexdigest() == before
