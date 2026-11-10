"""
features.py - Phase 2: feature engineering.

Every function is vectorised (whole columns at once, no row-by-row loops).

Engineered features:
    OrderMonth            1-12, captures seasonality
    OrderDayOfWeek        0 = Monday ... 6 = Sunday
    IsWeekend             1 if the order was placed on Saturday or Sunday
    DaysSinceStart        days since the first order in the training data (trend)
    HasCoupon             1 if any coupon code was used
    OtherItemsInCart      items in the cart that are NOT this product
    QuantityShareOfCart   share of the cart taken by this product (0-1]
"""

from __future__ import annotations

import pandas as pd


def date_features(dates: pd.Series, start_date: pd.Timestamp) -> pd.DataFrame:
    """Calendar features from the order date."""
    return pd.DataFrame(
        {
            "OrderMonth": dates.dt.month.astype(int),
            "OrderDayOfWeek": dates.dt.day_of_week.astype(int),
            "IsWeekend": (dates.dt.day_of_week >= 5).astype(int),
            "DaysSinceStart": (dates - start_date).dt.days.astype(int),
        },
        index=dates.index,
    )


def cart_features(quantity: pd.Series, items_in_cart: pd.Series) -> pd.DataFrame:
    """Basket composition. The raw contract guarantees ItemsInCart >= Quantity >= 1."""
    return pd.DataFrame(
        {
            "OtherItemsInCart": (items_in_cart - quantity).astype(int),
            "QuantityShareOfCart": (quantity / items_in_cart).astype(float),
        },
        index=quantity.index,
    )


def coupon_flag(coupon_raw: pd.Series) -> pd.Series:
    """1 if a coupon was used (value present in the raw data), else 0."""
    return coupon_raw.notna().astype(int).rename("HasCoupon")
