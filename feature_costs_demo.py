"""Feature (test) costs: the same data, cost_weight = 0 (cost-blind) vs 1 (cost-aware). Costs are in arbitrary money units per feature."""
import numpy as np, pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from c5own import C5Own

X, y = make_classification(n_samples=6000, n_features=8, n_informative=6, n_redundant=2, n_classes=2, random_state=1, class_sep=1.2)
df = pd.DataFrame(X, columns=["history", "bp", "blood", "ultrasound", "xray", "ct", "mri", "biopsy"])
costs = {"history": 1, "bp": 2, "blood": 10, "ultrasound": 20, "xray": 20, "ct": 90, "mri": 150, "biopsy": 500}     # price of measuring each column
Xtr, Xte, ytr, yte = train_test_split(df, y, test_size=0.3, random_state=0, stratify=y)
L = np.array([[0, 100.0], [400.0, 0]])                                  # L[true, predicted]: a missed case (true 1 -> predicted 0) costs 400, a false alarm 100
def run(label, **kw):
    m = C5Own(cost_matrix=L, min_cases=10, **kw).fit(Xtr, ytr)
    pred = m.predict(Xte); mis = L[yte, pred].mean(); tests = float(m.path_cost(Xte).mean()) if kw.get("feature_costs") else float((m.features_used_(Xte) * np.array([costs[c] for c in df.columns])).sum(1).mean())
    print(f"{label:46s} accuracy={(pred == yte).mean():.3f}  test cost={tests:6.1f}  error cost={mis:6.1f}  TOTAL={tests + mis:6.1f}  leaves={m.n_leaves()}")

run("no feature costs (only the cost matrix)")
run("feature costs, cost_weight=0 (blind split choice,", feature_costs=costs, cost_weight=0.0)
run("feature costs, cost_weight=1", feature_costs=costs, cost_weight=1.0)
run("feature costs, cost_weight=3", feature_costs=costs, cost_weight=3.0)
print("note: with cost_weight=0 the split CHOICE ignores prices but pruning still charges them, so an expensive first split can be pruned away entirely.")
