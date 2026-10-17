"""Quick start: mixed numeric + categorical data, fit, predict, print the tree."""
import numpy as np, pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from c5own import C5Own

X, y = make_classification(n_samples=4000, n_features=8, n_informative=5, n_classes=3, random_state=0)
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
df["f0"] = pd.qcut(df["f0"], 4, labels=["q1", "q2", "q3", "q4"])       # a categorical column (multiway split)
df.loc[df.sample(frac=0.02, random_state=0).index, "f3"] = np.nan        # some missing values
Xtr, Xte, ytr, yte = train_test_split(df, y, test_size=0.3, random_state=0, stratify=y)

m = C5Own(min_cases=2, cf=0.25).fit(Xtr, ytr)                           # defaults: CF = 0.25 pessimistic pruning, min 2 cases per branch
print("accuracy:", (m.predict(Xte) == yte).mean(), "| leaves:", m.n_leaves())
print(m.predict_proba(Xte)[:3].round(3))
print(m.export_text(max_depth=3))
