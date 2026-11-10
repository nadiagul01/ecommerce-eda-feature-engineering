"""
config.py - Central configuration for the whole project.

Every path, column name and decision threshold lives here, so a reviewer can
see all the rules of the pipeline in one place instead of hunting for
"magic numbers" inside the code.
"""

from pathlib import Path

# =============================================================================
# Paths (built with pathlib so they work on Windows, macOS and Linux)
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Dataset_for_Data_Analytics.xlsx"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# =============================================================================
# Column roles
# =============================================================================
ID_COLUMN = "OrderID"
DATE_COLUMN = "Date"

NUMERIC_COLUMNS = ["Quantity", "UnitPrice", "ItemsInCart", "TotalPrice"]
CATEGORICAL_COLUMNS = ["Product", "PaymentMethod", "OrderStatus", "CouponCode", "ReferralSource"]

# Identifiers carry no general pattern a model can learn from:
#   CustomerID      - 1,189 unique values for 1,200 orders
#   TrackingNumber  - unique per order
#   ShippingAddress - every address is "<number> Main St"; the house number is noise
IDENTIFIER_COLUMNS_TO_DROP = ["CustomerID", "TrackingNumber", "ShippingAddress"]

# =============================================================================
# Phase 1 - Missing data decision matrix (from the training kit)
#   < 5% missing      -> drop the affected rows
#   5% - 20% missing  -> statistical imputation (median)
#   > 20% missing     -> multi-dimensional estimation (KNN)
# =============================================================================
DROP_ROWS_MAX_PCT = 5.0
STAT_IMPUTE_MAX_PCT = 20.0
KNN_NEIGHBOURS = 5

# Structural missingness: the blank itself has a known business meaning.
# An empty CouponCode means "the customer did not use a coupon"; estimating a
# coupon with KNN would invent discounts that never happened. These columns are
# filled with an explicit label and a 0/1 flag is created instead.
STRUCTURAL_MISSING_FILL = {"CouponCode": "NONE"}
STRUCTURAL_MISSING_FLAG = {"CouponCode": "HasCoupon"}

# Imputation experiment: hide a share of a complete column on purpose, then
# measure how well each method recovers the TRUE hidden values.
EXPERIMENT_COLUMN = "ItemsInCart"
EXPERIMENT_MISSING_SHARE = 0.15  # inside the 5-20% band of the decision matrix
EXPERIMENT_REPEATS = 20          # repeat with different random masks for stable results

# =============================================================================
# Phase 1 - Outliers (IQR rule, values capped with numpy.clip)
# =============================================================================
IQR_MULTIPLIER = 1.5
ZSCORE_THRESHOLD = 3.0
# Only TotalPrice is capped in the model-ready table. Quantity, UnitPrice and
# ItemsInCart have zero IQR/Z-score outliers (checked in the notebook).
OUTLIER_CAP_COLUMNS = ["TotalPrice"]

# =============================================================================
# Phase 2 - Collinearity
# =============================================================================
COLLINEARITY_THRESHOLD = 0.80

# Tolerance used when checking TotalPrice == Quantity x UnitPrice
PRICE_TOLERANCE = 0.01

RANDOM_STATE = 42
