# artifacts/

| File | Contents |
|---|---|
| `pipeline_params.json` | every value the pipeline learned during `fit_transform()`: missing-value strategies, IQR bounds, categories and reference categories, dropped collinear features, final feature list |

These stored values are what `transform()` re-uses on new data, so training and
serving always compute features the same way. The file is also a readable audit
record of every decision.
