# tests/

`test_pipeline.py` holds 25 automated checks, run with `pytest` from the project root
(and automatically on GitHub by `.github/workflows/tests.yml`).

| Group | What is checked |
|---|---|
| Phase 1 — missing data | decision-matrix thresholds, no missing values left, coupon blanks → `NONE` + `HasCoupon`, no rows lost, KNN beats Mean, KNN branch fills gaps |
| Phase 1 — outliers | IQR formula, outlier counts match the data, `TotalPrice` capped only in the model table |
| Phase 2 — features | feature formulas on hand-made examples, at least 3 engineered features kept, all model columns numeric, reference categories dropped, no pair above \|r\| 0.80, identifiers removed |
| Phase 3 — contracts | `transform()` equals `fit_transform()`, new orders with unseen categories handled, bad input rejected with every failure listed, raw file never modified |
