# data/

| Folder | Contents | Edited by |
|---|---|---|
| `raw/` | `Dataset_for_Data_Analytics.xlsx` — the course dataset from the DecodeLabs intern portal (1,200 orders x 14 columns) | nobody — read-only input |
| `processed/` | outputs of `python main.py` | the pipeline only |

## Processed files

| File | What it is | Use it for |
|---|---|---|
| `orders_clean.csv` | cleaned data + engineered features, **original values kept** (1,200 x 18) | dashboards, business analysis |
| `orders_model_ready.csv` | fully numeric: one-hot encoded, `TotalPrice` capped, collinear features removed (1,200 x 31) | machine learning |
| `orders_model_ready.parquet` | same as above in Parquet (typed, compressed) | fast loading in Python |

## Raw columns

| Column | Type | Notes |
|---|---|---|
| OrderID | text | unique order key (`ORD200000` …) |
| Date | date | 2023-01-01 to 2025-06-30 |
| CustomerID | text | 1,189 unique — identifier, dropped from the model table |
| Product | text | 7 products |
| Quantity | integer | 1–5 |
| UnitPrice | decimal | 11.39–699.93 |
| ShippingAddress | text | always `<number> Main St` — dropped from the model table |
| PaymentMethod | text | 5 methods |
| OrderStatus | text | 5 statuses |
| TrackingNumber | text | unique per order — dropped from the model table |
| ItemsInCart | integer | 1–10, always ≥ Quantity |
| CouponCode | text | 25.75% blank = no coupon used |
| ReferralSource | text | 5 sources |
| TotalPrice | decimal | always equals Quantity x UnitPrice |
