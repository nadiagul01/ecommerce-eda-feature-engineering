# EDA & Feature Engineering - Run Summary

_Generated automatically by `main.py`. Re-run the pipeline instead of editing this file._

- Raw data: **1200 rows x 14 columns**
- Analysis table: **1200 rows x 18 columns**
- Model-ready table: **1200 rows x 31 columns** (OrderID + 30 features), rows dropped: 0
- Missing values left: analysis **0**, model **0**

## 1. Missing values

| column | missing_count | missing_pct | strategy |
| --- | --- | --- | --- |
| CouponCode | 309 | 25.75 | structural_fill |

## 2. Imputation experiment (Mean vs Median vs KNN)

15% of `ItemsInCart` hidden at random, averaged over 20 repeats. Lower MAE/RMSE/KS is better; std closer to the true value means the spread is preserved.

| method | MAE_on_hidden | RMSE_on_hidden | std_after | KS_vs_true |
| --- | --- | --- | --- | --- |
| True values (reference) | 0.0 | 0.0 | 2.282 | 0.0 |
| Mean | 1.925 | 2.309 | 2.099 | 0.08 |
| Median | 1.95 | 2.379 | 2.106 | 0.079 |
| KNN (k=5) | 1.64 | 1.914 | 2.197 | 0.025 |

## 3. Outliers

| column | skewness | iqr_lower | iqr_upper | iqr_outliers | zscore_outliers | max_value |
| --- | --- | --- | --- | --- | --- | --- |
| Quantity | 0.028 | -1.0 | 7.0 | 0 | 0 | 5.0 |
| UnitPrice | -0.027 | -317.2 | 1024.83 | 0 | 0 | 699.93 |
| ItemsInCart | 0.001 | -0.5 | 11.5 | 0 | 0 | 10.0 |
| TotalPrice | 0.891 | -1341.41 | 3330.41 | 8 | 0 | 3456.4 |

## 4. Collinearity (|r| > 0.80)

| feature_a | feature_b | abs_corr | a_avg_corr | b_avg_corr | dropped |
| --- | --- | --- | --- | --- | --- |
| HasCoupon | CouponCode_NONE | 1.0 | 0.0735 | 0.0735 | CouponCode_NONE |
| OtherItemsInCart | QuantityShareOfCart | 0.8381 | 0.0712 | 0.0773 | QuantityShareOfCart |

## 5. Final model features

`Quantity`, `UnitPrice`, `ItemsInCart`, `TotalPrice`, `OrderMonth`, `OrderDayOfWeek`, `IsWeekend`, `DaysSinceStart`, `HasCoupon`, `OtherItemsInCart`, `Product_Chair`, `Product_Desk`, `Product_Laptop`, `Product_Monitor`, `Product_Phone`, `Product_Tablet`, `PaymentMethod_Cash`, `PaymentMethod_Credit_Card`, `PaymentMethod_Debit_Card`, `PaymentMethod_Gift_Card`, `OrderStatus_Delivered`, `OrderStatus_Pending`, `OrderStatus_Returned`, `OrderStatus_Shipped`, `CouponCode_SAVE10`, `CouponCode_WINTER15`, `ReferralSource_Email`, `ReferralSource_Facebook`, `ReferralSource_Google`, `ReferralSource_Referral`
