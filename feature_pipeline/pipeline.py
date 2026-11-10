"""
pipeline.py - The Input -> Process -> Output pipeline.

    FeaturePipeline.fit_transform(df)  learns every statistic from the data
                                       (medians, IQR bounds, categories, dropped
                                       columns) and applies it.
    FeaturePipeline.transform(df)      re-applies exactly the SAME learned values
                                       to new data.

Learning once and re-using the stored values is what prevents
training-serving skew (Phase 3 of the training kit).

Two tables are produced:
    analysis table  - cleaned data + engineered features, original values kept
                      (for dashboards and business analysis)
    model table     - fully numeric, one-hot encoded, outliers capped,
                      collinear features removed (for machine learning)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from . import config
from .collinearity import find_collinear_drops
from .features import cart_features, coupon_flag, date_features
from .missing import missing_report
from .outliers import iqr_bounds, winsorize
from .schema import build_output_schema, raw_schema

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Cleans the e-commerce orders data and builds model-ready features."""

    def __init__(self) -> None:
        self.is_fitted = False
        # Phase 1 - missing data
        self.missing_report_: pd.DataFrame | None = None
        self.strategy_: dict[str, str] = {}
        self.medians_: dict[str, float] = {}
        self.modes_: dict[str, str] = {}
        self._knn_scaler: StandardScaler | None = None
        self._knn_imputer: KNNImputer | None = None
        self._knn_inputs: list[str] = []
        # Phase 1 - outliers
        self.outlier_bounds_: dict[str, tuple[float, float]] = {}
        self.outlier_counts_: dict[str, dict[str, int]] = {}
        # Phase 2 - features, encoding, collinearity
        self.start_date_: pd.Timestamp | None = None
        self.categories_: dict[str, list[str]] = {}
        self.reference_category_: dict[str, str] = {}
        self.collinear_dropped_: list[str] = []
        self.collinearity_table_: pd.DataFrame = pd.DataFrame()
        # Output
        self.feature_columns_: list[str] = []
        self.binary_columns_: set[str] = set()
        self.fill_values_: dict[str, float] = {}
        self.rows_in_: int = 0
        self.rows_dropped_: int = 0

    # =========================================================================
    # Public API
    # =========================================================================
    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Learn from `df` and return (analysis_table, model_table)."""
        return self._run(df, fitting=True)

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply the already-learned rules to new data."""
        if not self.is_fitted:
            raise RuntimeError("Call fit_transform() on the training data before transform().")
        return self._run(df, fitting=False)

    # =========================================================================
    # Orchestration
    # =========================================================================
    def _run(self, df: pd.DataFrame, fitting: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
        # ---------------- INPUT: validate the raw data contract
        df = raw_schema.validate(df.copy(), lazy=True)
        rows_in = len(df)

        # ---------------- PHASE 1: missing values
        coupon_raw = df["CouponCode"].copy()  # remember the original blanks for HasCoupon
        df = self._handle_missing(df, fitting)
        coupon_raw = coupon_raw.loc[df.index]

        # ---------------- PHASE 2: feature engineering
        if fitting:
            self.start_date_ = df[config.DATE_COLUMN].min()
        df = pd.concat(
            [
                df,
                date_features(df[config.DATE_COLUMN], self.start_date_),
                coupon_flag(coupon_raw),
                cart_features(df["Quantity"], df["ItemsInCart"]),
            ],
            axis=1,
        )
        analysis = df.drop(columns=config.IDENTIFIER_COLUMNS_TO_DROP).reset_index(drop=True)

        # ---------------- PHASE 1: outliers (model table only)
        model = self._cap_outliers(df.copy(), fitting)

        # ---------------- PHASE 2: encoding + collinearity
        model = self._encode(model, fitting)
        model = model.drop(columns=config.IDENTIFIER_COLUMNS_TO_DROP + [config.DATE_COLUMN])
        model = self._remove_collinear(model, fitting)

        # ---------------- OUTPUT: fixed column order + output contract
        model = self._finalise(model, fitting)
        schema = build_output_schema(self.feature_columns_, self.binary_columns_, self.outlier_bounds_)
        model = schema.validate(model, lazy=True).reset_index(drop=True)

        if fitting:
            self.rows_in_ = rows_in
            self.rows_dropped_ = rows_in - len(model)
            self.is_fitted = True
        logger.info("Pipeline done: %d rows in, %d rows out, %d model features.",
                    rows_in, len(model), len(self.feature_columns_))
        return analysis, model

    # =========================================================================
    # Phase 1 - missing data
    # =========================================================================
    def _handle_missing(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        if fitting:
            self.missing_report_ = missing_report(df)
            self.strategy_ = self.missing_report_["strategy"].to_dict()
            for col, row in self.missing_report_.iterrows():  # loops over COLUMNS only
                if row["strategy"] != "none":
                    logger.info("%-14s %6.2f%% missing -> %s", col, row["missing_pct"], row["strategy"])

        # 1) structural blanks -> explicit label (e.g. CouponCode -> "NONE")
        for col, label in config.STRUCTURAL_MISSING_FILL.items():
            df[col] = df[col].fillna(label)

        # 2) < 5% missing -> drop the affected rows
        drop_cols = [c for c, s in self.strategy_.items() if s == "drop_rows"]
        if drop_cols:
            before = len(df)
            df = df.dropna(subset=drop_cols).copy()
            if before - len(df):
                logger.info("Dropped %d rows with missing %s", before - len(df), drop_cols)

        # 3) 5-20% missing -> median (numeric) / most frequent value (text)
        for col in [c for c, s in self.strategy_.items() if s == "stat_impute"]:
            if pd.api.types.is_numeric_dtype(df[col]):
                if fitting:
                    self.medians_[col] = float(df[col].median())
                df[col] = df[col].fillna(self.medians_[col])
            else:
                if fitting:
                    self.modes_[col] = str(df[col].mode().iloc[0])
                df[col] = df[col].fillna(self.modes_[col])

        # 4) > 20% missing -> KNN on standardised numeric columns
        knn_cols = [c for c, s in self.strategy_.items() if s == "knn"]
        if knn_cols:
            df = self._knn_impute(df, knn_cols, fitting)

        # Safety net for NEW data: a column with no gaps during fit may have gaps later.
        for col in config.NUMERIC_COLUMNS:
            if df[col].isna().any():
                if col not in self.medians_:
                    raise ValueError(f"Unexpected missing values in '{col}' with no learned fill value.")
                logger.warning("%s: %d unexpected missing values filled with the learned median",
                               col, int(df[col].isna().sum()))
                df[col] = df[col].fillna(self.medians_[col])
        return df

    def _knn_impute(self, df: pd.DataFrame, knn_cols: list[str], fitting: bool) -> pd.DataFrame:
        numeric_knn = [c for c in knn_cols if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_knn) != len(knn_cols):
            raise ValueError(f"KNN imputation needs numeric columns, got: {knn_cols}")
        if fitting:
            self._knn_inputs = list(config.NUMERIC_COLUMNS)
            self._knn_scaler = StandardScaler().fit(df[self._knn_inputs])
            self._knn_imputer = KNNImputer(n_neighbors=config.KNN_NEIGHBOURS).fit(
                self._knn_scaler.transform(df[self._knn_inputs])
            )
            for col in knn_cols:
                self.medians_[col] = float(df[col].median())
        scaled = self._knn_scaler.transform(df[self._knn_inputs])
        restored = pd.DataFrame(
            self._knn_scaler.inverse_transform(self._knn_imputer.transform(scaled)),
            columns=self._knn_inputs, index=df.index,
        )
        for col in knn_cols:
            df[col] = df[col].fillna(restored[col])
        return df

    # =========================================================================
    # Phase 1 - outliers
    # =========================================================================
    def _cap_outliers(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        for col in config.OUTLIER_CAP_COLUMNS:
            if fitting:
                self.outlier_bounds_[col] = iqr_bounds(df[col])
            low, high = self.outlier_bounds_[col]
            below, above = int((df[col] < low).sum()), int((df[col] > high).sum())
            if fitting:
                self.outlier_counts_[col] = {"below": below, "above": above}
            df[col] = winsorize(df[col], low, high)
            if below or above:
                logger.info("%s capped to [%.2f, %.2f]: %d low, %d high", col, low, high, below, above)
        return df

    # =========================================================================
    # Phase 2 - encoding
    # =========================================================================
    def _encode(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        """One-hot encode text columns. The most frequent category of each column
        is the reference (all zeros), which avoids the dummy-variable trap."""
        for col in config.CATEGORICAL_COLUMNS:
            if fitting:
                counts = df[col].value_counts()
                self.categories_[col] = sorted(counts.index.tolist())
                self.reference_category_[col] = str(counts.index[0])
            cats = self.categories_[col]
            unseen = ~df[col].isin(cats)
            if unseen.any():
                logger.warning("%s: %d unseen categories -> treated as reference '%s'",
                               col, int(unseen.sum()), self.reference_category_[col])
            # Unseen values become NaN -> all-zero dummy row, i.e. the reference category.
            known = df[col].where(~unseen)
            dummies = pd.get_dummies(pd.Categorical(known, categories=cats), prefix=col, dtype=int)
            dummies.index = df.index
            dummies = dummies.drop(columns=f"{col}_{self.reference_category_[col]}")
            # "PaymentMethod_Credit Card" -> "PaymentMethod_Credit_Card" (no spaces in column names)
            dummies.columns = [c.replace(" ", "_") for c in dummies.columns]
            df = pd.concat([df.drop(columns=col), dummies], axis=1)
        return df

    # =========================================================================
    # Phase 2 - collinearity
    # =========================================================================
    def _remove_collinear(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        if fitting:
            features = df.drop(columns=config.ID_COLUMN)
            self.collinear_dropped_, self.collinearity_table_ = find_collinear_drops(
                features, config.COLLINEARITY_THRESHOLD
            )
            for _, r in self.collinearity_table_.iterrows():
                logger.info("Collinear pair %s / %s (|r|=%.3f) -> dropped %s",
                            r["feature_a"], r["feature_b"], r["abs_corr"], r["dropped"])
        return df.drop(columns=[c for c in self.collinear_dropped_ if c in df.columns])

    # =========================================================================
    # Output
    # =========================================================================
    def _finalise(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        if fitting:
            self.feature_columns_ = [c for c in df.columns if c != config.ID_COLUMN]
            self.binary_columns_ = {
                c for c in self.feature_columns_
                if set(pd.unique(df[c])) <= {0, 1} and c not in config.OUTLIER_CAP_COLUMNS
            }
        return df[[config.ID_COLUMN, *self.feature_columns_]]

    # =========================================================================
    # Persistence - every learned value, saved as readable JSON for review
    # =========================================================================
    def to_dict(self) -> dict:
        return {
            "rows_in": self.rows_in_,
            "rows_dropped": self.rows_dropped_,
            "missing_strategies": self.missing_report_.reset_index().to_dict(orient="records"),
            "structural_fill": config.STRUCTURAL_MISSING_FILL,
            "start_date": str(self.start_date_.date()),
            "outlier_bounds": {k: {"lower": v[0], "upper": v[1]} for k, v in self.outlier_bounds_.items()},
            "outlier_counts": self.outlier_counts_,
            "categories": self.categories_,
            "reference_category": self.reference_category_,
            "collinear_dropped": self.collinear_dropped_,
            "collinearity_pairs": self.collinearity_table_.to_dict(orient="records"),
            "feature_columns": self.feature_columns_,
        }

    def save_params(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
