"""
schema.py - Phase 3: data contracts with Pandera.

raw_schema           guards the INPUT  - column types, valid ranges, business rules
build_output_schema  guards the OUTPUT - no nulls, binary flags are 0/1, capped bounds

Both are validated with lazy=True, so ALL failures are reported in one error
instead of stopping at the first one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera import Check

from . import config

# -----------------------------------------------------------------------------
# Input contract (raw Excel data)
# -----------------------------------------------------------------------------
raw_schema = pa.DataFrameSchema(
    columns={
        "OrderID": pa.Column(str, Check.str_matches(r"^ORD\d+$"), unique=True),
        "Date": pa.Column("datetime64[ns]"),
        "CustomerID": pa.Column(str, Check.str_matches(r"^C\d+$")),
        "Product": pa.Column(str),
        "Quantity": pa.Column(int, Check.ge(1)),
        "UnitPrice": pa.Column(float, Check.gt(0)),
        "ShippingAddress": pa.Column(str),
        "PaymentMethod": pa.Column(str),
        "OrderStatus": pa.Column(str),
        "TrackingNumber": pa.Column(str),
        "ItemsInCart": pa.Column(int, Check.ge(1)),
        "CouponCode": pa.Column(str, nullable=True),  # blank = no coupon used
        "ReferralSource": pa.Column(str),
        "TotalPrice": pa.Column(float, Check.gt(0)),
    },
    checks=[
        # Business rule 1: the cart must contain at least this product's quantity.
        Check(lambda d: d["ItemsInCart"] >= d["Quantity"], name="items_in_cart_ge_quantity"),
        # Business rule 2: the order total must equal quantity x unit price.
        Check(
            lambda d: pd.Series(
                np.isclose(d["TotalPrice"], d["Quantity"] * d["UnitPrice"], atol=config.PRICE_TOLERANCE),
                index=d.index,
            ),
            name="total_equals_quantity_times_price",
        ),
    ],
    coerce=True,
    strict=False,
)


# -----------------------------------------------------------------------------
# Output contract (model-ready table)
# -----------------------------------------------------------------------------
def build_output_schema(
    feature_columns: list[str],
    binary_columns: set[str],
    bounds: dict[str, tuple[float, float]],
) -> pa.DataFrameSchema:
    """Contract for the final table: every feature present, numeric and non-null."""
    columns: dict[str, pa.Column] = {config.ID_COLUMN: pa.Column(str, unique=True, nullable=False)}
    for col in feature_columns:
        checks = []
        if col in binary_columns:
            checks.append(Check.isin([0, 1]))
        elif col in bounds:
            low, high = bounds[col]
            checks.append(Check.in_range(low - 1e-9, high + 1e-9))  # tolerance for float rounding
        elif col == "QuantityShareOfCart":
            checks.append(Check.in_range(0, 1, include_min=False))
        elif col in ("Quantity", "ItemsInCart"):
            checks.append(Check.ge(1))
        elif col == "OtherItemsInCart":
            checks.append(Check.ge(0))
        columns[col] = pa.Column(nullable=False, checks=checks)
    return pa.DataFrameSchema(columns, strict=True, ordered=False)
