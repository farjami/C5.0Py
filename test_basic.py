import numpy as np, pandas as pd
from c5own import C5Own


def _data(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n); df = pd.DataFrame({"a": rng.normal(y, 1.0), "b": rng.normal(0, 1, n), "c": rng.choice(["u", "v", "w"], n)})
    df.loc[y == 2, "c"] = "w"; df.loc[rng.random(n) < 0.03, "a"] = np.nan
    return df, y


def test_fit_predict_better_than_chance():
    df, y = _data(); m = C5Own().fit(df[:1000], y[:1000]); assert (m.predict(df[1000:]) == y[1000:]).mean() > 0.6


def test_proba_rows_sum_to_one():
    df, y = _data(); m = C5Own().fit(df, y); P = m.predict_proba(df); assert np.allclose(P.sum(1), 1) and P.shape == (len(df), 3)


def test_max_depth_and_pruning_shrink_tree():
    df, y = _data(); big = C5Own(prune=False).fit(df, y).n_leaves(); small = C5Own(prune=True).fit(df, y).n_leaves(); shallow = C5Own(max_depth=2, prune=False).fit(df, y).n_leaves()
    assert small <= big and shallow <= 4 * 3


def test_cost_matrix_changes_labels():
    df, y = _data(); L = np.zeros((3, 3)); L[2, :] = 50.0; L[2, 2] = 0; L[0, 1:] = 1; L[1, 0] = 1; L[1, 2] = 1
    p0 = C5Own().fit(df, y).predict(df); p1 = C5Own(cost_matrix=L).fit(df, y).predict(df); assert (p1 == 2).sum() >= (p0 == 2).sum()


def test_feature_costs_reduce_path_cost():
    df, y = _data(); fc = {"a": 1, "b": 1, "c": 1000}
    blind = C5Own(feature_costs=fc, cost_weight=0.0).fit(df, y).path_cost(df).mean(); aware = C5Own(feature_costs=fc, cost_weight=1.0).fit(df, y).path_cost(df).mean()
    assert aware <= blind + 1e-9


def test_unseen_category_and_missing_do_not_crash():
    df, y = _data(); m = C5Own().fit(df, y); t = df.head(5).copy(); t["c"] = "NEW"; t["a"] = np.nan; assert len(m.predict(t)) == 5
