# feature_pipeline/

The Python package that holds all project logic. The notebook and `main.py` only call it.

| File | Training-kit phase | Responsibility |
|---|---|---|
| `config.py` | — | every path, column role and threshold in one place |
| `schema.py` | Phase 3 | Pandera contracts for the raw input and the final output |
| `missing.py` | Phase 1 | decision matrix, missing-value report, Mean vs Median vs KNN experiment |
| `outliers.py` | Phase 1 | IQR and Z-score detection, `numpy.clip` capping |
| `features.py` | Phase 2 | date, coupon and basket features (vectorised) |
| `collinearity.py` | Phase 2 | finds and removes features with \|r\| > 0.80 |
| `pipeline.py` | all | `FeaturePipeline` — runs every step in order; `fit_transform()` / `transform()` |
| `report.py` | — | figures and the auto-generated `reports/summary.md` |

Usage:

```python
import pandas as pd
from feature_pipeline import FeaturePipeline, config

raw = pd.read_excel(config.RAW_DATA_PATH)
pipe = FeaturePipeline()
analysis, model = pipe.fit_transform(raw)   # learn + apply
analysis_new, model_new = pipe.transform(new_orders)  # apply the same rules to new data
```
