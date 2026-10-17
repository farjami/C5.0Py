# c5own

A small, readable **C4.5/C5.0-style decision tree** (pure Python / NumPy / pandas / SciPy) with three things the common C5.0 ports lack in one place:

| feature | `C5Own` |
|---|---|
| class weights / sample weights | `class_weight=dict or "balanced"`, `fit(..., sample_weight=)` |
| misclassification-cost matrix | `cost_matrix=L` with `L[true, predicted]`, classes in `np.unique(y)` order; used for leaf labels **and** pruning |
| **feature (test) costs** | `feature_costs={"column": price}`, `cost_weight=w`; split score = gain ratio / (1 + cost/median cost)^w; pruning charges the test cost |
| numeric features | binary thresholds (C4.5 release-8 penalty) |
| categorical features | **multiway** splits (one branch per value) |
| pruning | pessimistic error with a Clopper-Pearson upper bound, `cf=0.25` |
| not implemented | boosting, winnowing, rule sets, fractional handling of missing values (median / `"__nan__"` category instead) |

It is **not** Quinlan's C5.0 and has not been validated against it. Name it "C5.0-style" or "C4.5-style" in papers.

## Install
```bash
pip install ./c5own                         # from the unzipped folder
# or, after you upload the folder to your own GitHub repository:
pip install git+https://github.com/<your-user>/c5own
# no install: set PYTHONPATH=src
```

## Use
```python
import pandas as pd, numpy as np
from c5own import C5Own

m = C5Own(min_cases=2, cf=0.25, max_depth=None).fit(X_train_df, y_train)      # X: DataFrame (numeric + category/object columns)
pred  = m.predict(X_test_df)
proba = m.predict_proba(X_test_df)           # columns follow m.classes_
print(m.export_text(max_depth=4)); print(m.n_leaves())
```
**Cost-sensitive + feature costs**
```python
L = np.array([[0, 100.], [400., 0]])          # L[true, pred]: missing class 1 costs 400, a false alarm 100  (classes sorted ascending)
m = C5Own(cost_matrix=L, feature_costs={"ct": 90, "xray": 20, "blood": 10}, cost_weight=1.0, min_cases=10).fit(X_df, y)
m.path_cost(X_test_df).mean()                 # mean money spent on tests along each patient's path
```
Run `python examples/quickstart.py`, `python examples/feature_costs_demo.py`, and `pytest tests/`.

Notes: leaf probabilities are smoothed counts (only for ranking/AUC); the predicted label is the leaf's cost-optimal label. Column names are used as keys of `feature_costs`. A feature is charged once per path.
Licence: **choose one yourself** before publishing (the code in this folder was written from the C4.5 description and does not copy c50py/c5tree/C5.0 code).
