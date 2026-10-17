"""c5own - a small, readable C4.5/C5.0-style decision tree with class weights, a misclassification-cost matrix and FEATURE (test) costs.

NOT Quinlan's C5.0: written from the published C4.5 description (gain ratio, average-gain filter, pessimistic Clopper-Pearson pruning). No boosting,
no winnowing, no rule sets. Numeric features: binary thresholds. Categorical features: multiway splits. Missing values: median / "__nan__" category.
"""
import math
import numpy as np
import pandas as pd
from scipy.stats import beta as _beta


def _is_cat(s):
    """categorical = category / object / string / bool dtype (anything that is not numeric)"""
    return str(s.dtype) == "category" or str(s.dtype) == "bool" or not pd.api.types.is_numeric_dtype(s)


def _ent_rows(c):
    s = c.sum(1, keepdims=True)
    p = c / np.maximum(s, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.where(p > 0, np.log2(p), 0.0)
    return -(p * lp).sum(1)


def _ent(c):
    return float(_ent_rows(np.asarray(c, float)[None, :])[0])


def upper_errors(N, E, cf=0.25):
    if N <= 0:
        return 0.0
    if E >= N:
        return float(N)
    return float(N * _beta.ppf(1.0 - cf, E + 1.0, N - E))


class C5Own:
    """sklearn-like. X: DataFrame (numeric columns = float/int, categorical = category/object/bool/string)."""

    def __init__(self, min_cases=2.0, cf=0.25, max_depth=None, prune=True, class_weight=None, cost_matrix=None, feature_costs=None,
                 cost_weight=1.0, shrink_m=0.0, max_thresholds=64, avg_gain=True):
        self.min_cases, self.cf, self.max_depth, self.prune = float(min_cases), cf, max_depth, prune
        self.class_weight, self.cost_matrix, self.feature_costs = class_weight, cost_matrix, feature_costs
        self.cost_weight, self.m, self.max_thr, self.avg_gain = cost_weight, shrink_m, max_thresholds, avg_gain

    # ---------------------------------------------------------------- encoding
    def _encode(self, X, fit):
        X = pd.DataFrame(X).reset_index(drop=True)
        if fit:
            self.cols_ = list(X.columns)
            self.iscat_ = [_is_cat(X[c]) for c in self.cols_]
            self.med_, self.cats_ = {}, {}
        num = np.zeros((len(X), len(self.cols_)))
        cat = np.zeros((len(X), len(self.cols_)), dtype=np.int64)
        for j, c in enumerate(self.cols_):
            if self.iscat_[j]:
                s = X[c].astype(object).where(X[c].notna(), "__nan__").astype(str)
                if fit:
                    self.cats_[j] = {v: i for i, v in enumerate(pd.unique(s))}
                cat[:, j] = s.map(self.cats_[j]).fillna(-1).astype(np.int64).values
            else:
                v = pd.to_numeric(X[c], errors="coerce").values.astype(float)
                if fit:
                    m = np.nanmedian(v); self.med_[j] = 0.0 if np.isnan(m) else m
                num[:, j] = np.where(np.isnan(v), self.med_[j], v)
        return num, cat

    # ---------------------------------------------------------------- fit
    def fit(self, X, y, sample_weight=None):
        num, cat = self._encode(X, True)
        y = np.asarray(y)
        self.classes_ = np.unique(y); K = self.K_ = len(self.classes_)
        yk = np.searchsorted(self.classes_, y)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float).copy()
        if self.class_weight == "balanced":
            cnt = np.bincount(yk, minlength=K); w *= (len(y) / (K * np.maximum(cnt, 1)))[yk]
        elif isinstance(self.class_weight, dict):
            w *= np.array([self.class_weight.get(c, 1.0) for c in self.classes_])[yk]
        self.L_ = None if self.cost_matrix is None else np.asarray(self.cost_matrix, float)
        self.num_, self.cat_, self.yk_, self.w_ = num, cat, yk, w
        self.Yw_ = np.zeros((len(y), K)); self.Yw_[np.arange(len(y)), yk] = w
        self.fcost_ = np.zeros(len(self.cols_))
        if self.feature_costs:
            for j, c in enumerate(self.cols_):
                self.fcost_[j] = float(self.feature_costs.get(c, 0.0))
        pos = self.fcost_[self.fcost_ > 0]
        self.cref_ = float(np.median(pos)) if len(pos) else 1.0
        base = self.Yw_.sum(0); self.prior_ = base / base.sum()
        self.tree_ = self._grow(np.arange(len(y)), 0, frozenset())
        if self.prune:
            self.tree_, _ = self._prune(self.tree_, frozenset())
        for a in ("num_", "cat_", "yk_", "w_", "Yw_"):
            setattr(self, a, None)
        return self

    def _label(self, cnt):
        if self.L_ is None:
            return int(np.argmax(cnt))
        n = cnt.sum(); p = (cnt + self.m * self.prior_) / (n + self.m) if n + self.m > 0 else self.prior_
        return int(np.argmin(p @ self.L_))

    def _leaf(self, cnt):
        return {"leaf": True, "cnt": cnt, "pred": self._label(cnt), "n": float(cnt.sum())}

    def _grow(self, idx, depth, used):
        cnt = self.Yw_[idx].sum(0); n = cnt.sum(); nd = self._leaf(cnt)
        if (self.max_depth is not None and depth >= self.max_depth) or (cnt > 0).sum() <= 1 or n < 2 * self.min_cases:
            return nd
        H0 = _ent(cnt); cands = []
        for j in range(len(self.cols_)):
            c = self._eval_feature(j, idx, cnt, n, H0)
            if c is not None:
                cands.append(c)
        if not cands:
            return nd
        keep = cands
        if self.avg_gain and len(cands) > 1:
            ag = np.mean([c["gain"] for c in cands]); keep = [c for c in cands if c["gain"] >= ag - 1e-12] or cands
        best, bs = None, -1.0
        for c in keep:
            if c["gain"] <= 1e-9:
                continue
            fc = 0.0 if c["j"] in used else self.fcost_[c["j"]]
            sc = c["gr"] / (1.0 + fc / self.cref_) ** self.cost_weight if self.feature_costs else c["gr"]
            if sc > bs:
                best, bs = c, sc
        if best is None:
            return nd
        j = best["j"]; nu = used | {j}
        if best["kind"] == "num":
            li = idx[self.num_[idx, j] <= best["thr"]]; ri = idx[self.num_[idx, j] > best["thr"]]
            nd.update(leaf=False, feat=j, kind="num", thr=best["thr"], kids={"l": self._grow(li, depth + 1, nu), "r": self._grow(ri, depth + 1, nu)})
        else:
            codes = self.cat_[idx, j]; kids = {}
            for v in np.unique(codes):
                kids[int(v)] = self._grow(idx[codes == v], depth + 1, nu)
            nd.update(leaf=False, feat=j, kind="cat", kids=kids)
        pop = {k: c["n"] for k, c in kids.items()} if best["kind"] == "cat" else {k: c["n"] for k, c in nd["kids"].items()}
        nd["major"] = max(pop, key=pop.get) if best["kind"] == "cat" else max(nd["kids"], key=lambda k: nd["kids"][k]["n"])
        return nd

    def _eval_feature(self, j, idx, cnt, n, H0):
        K, mc = self.K_, self.min_cases
        if self.iscat_[j]:
            codes = self.cat_[idx, j]
            nc = int(codes.max()) + 1
            tab = np.bincount(codes * K + self.yk_[idx], weights=self.w_[idx], minlength=nc * K).reshape(nc, K)
            nb = tab.sum(1)
            if (nb >= mc).sum() < 2:
                return None
            Hc = float((nb / n * _ent_rows(tab)).sum()); gain = H0 - Hc
            si = _ent(nb[nb > 0])
            return dict(j=j, kind="cat", gain=gain, gr=gain / si if si > 1e-9 else 0.0)
        v = self.num_[idx, j]
        order = np.argsort(v, kind="stable"); vs = v[order]
        pos = np.nonzero(vs[:-1] < vs[1:])[0]
        if len(pos) == 0:
            return None
        npos_all = len(pos)
        if len(pos) > self.max_thr:
            pos = pos[np.unique(np.linspace(0, len(pos) - 1, self.max_thr).astype(int))]
        cum = np.cumsum(self.Yw_[idx][order], axis=0)
        left = cum[pos]; nl = left.sum(1); right = cnt - left; nr = n - nl
        ok = (nl >= mc) & (nr >= mc)
        if not ok.any():
            return None
        gain = H0 - (nl / n) * _ent_rows(left) - (nr / n) * _ent_rows(right) - math.log2(max(npos_all, 2)) / n   # C4.5 release-8 penalty
        gain = np.where(ok, gain, -np.inf); b = int(np.argmax(gain))
        if not np.isfinite(gain[b]):
            return None
        p = np.array([nl[b], nr[b]]) / n
        si = _ent(p)
        return dict(j=j, kind="num", gain=float(gain[b]), gr=float(gain[b]) / si if si > 1e-9 else 0.0, thr=float((vs[pos[b]] + vs[pos[b] + 1]) / 2))

    # ---------------------------------------------------------------- pruning (bottom-up, pessimistic)
    def _leaf_pess(self, cnt, pred):
        N = cnt.sum(); E = N - cnt[pred]; U = upper_errors(N, E, self.cf)
        if self.L_ is None:
            return U + 0.0
        mis = float(cnt @ self.L_[:, pred]); avg = mis / E if E > 1e-9 else float(np.mean([self.L_[c, pred] for c in range(self.K_) if c != pred]))
        return mis + (U - E) * avg

    def _prune(self, nd, used):
        if nd["leaf"]:
            return nd, self._leaf_pess(nd["cnt"], nd["pred"])
        j = nd["feat"]; sub = 0.0
        for k in list(nd["kids"]):
            nd["kids"][k], c = self._prune(nd["kids"][k], used | {j}); sub += c
        if self.feature_costs:
            sub += nd["n"] * (0.0 if j in used else self.fcost_[j])
        leaf_c = self._leaf_pess(nd["cnt"], nd["pred"])
        slack = 0.1 if self.L_ is None else 0.0
        if leaf_c <= sub + slack:
            return self._leaf(nd["cnt"]), leaf_c
        return nd, sub

    # ---------------------------------------------------------------- predict
    def _route(self, nd, num, cat, idx, out, used, U):
        if nd["leaf"]:
            out[idx] = (nd["cnt"] + 0.5 / self.K_) / (nd["n"] + 0.5); return
        j = nd["feat"]; U[idx, j] = True
        if nd["kind"] == "num":
            m = num[idx, j] <= nd["thr"]
            self._route(nd["kids"]["l"], num, cat, idx[m], out, used, U); self._route(nd["kids"]["r"], num, cat, idx[~m], out, used, U)
        else:
            codes = cat[idx, j]; rest = np.ones(len(idx), bool)
            for k, ch in nd["kids"].items():
                m = codes == k
                if m.any():
                    self._route(ch, num, cat, idx[m], out, used, U); rest &= ~m
            if rest.any():
                self._route(nd["kids"][nd["major"]], num, cat, idx[rest], out, used, U)

    def predict_proba(self, X, return_features=False):
        num, cat = self._encode(X, False)
        out = np.zeros((len(num), self.K_)); U = np.zeros((len(num), len(self.cols_)), bool)
        self._route(self.tree_, num, cat, np.arange(len(num)), out, None, U)
        out = out / out.sum(1, keepdims=True)
        return (out, U) if return_features else out

    def predict(self, X):
        num, cat = self._encode(X, False)
        P = np.zeros((len(num), self.K_)); U = np.zeros((len(num), len(self.cols_)), bool)
        # label = leaf label (cost-aware when a cost matrix was given), not argmax of the smoothed proba
        pred = np.zeros(len(num), int)
        def rec(nd, idx):
            if nd["leaf"]:
                pred[idx] = nd["pred"]; return
            j = nd["feat"]
            if nd["kind"] == "num":
                m = num[idx, j] <= nd["thr"]; rec(nd["kids"]["l"], idx[m]); rec(nd["kids"]["r"], idx[~m])
            else:
                codes = cat[idx, j]; rest = np.ones(len(idx), bool)
                for k, ch in nd["kids"].items():
                    m = codes == k
                    if m.any():
                        rec(ch, idx[m]); rest &= ~m
                if rest.any():
                    rec(nd["kids"][nd["major"]], idx[rest])
        rec(self.tree_, np.arange(len(num)))
        return self.classes_[pred]

    def n_leaves(self):
        def c(nd): return 1 if nd["leaf"] else sum(c(k) for k in nd["kids"].values())
        return c(self.tree_)

    # ---------------------------------------------------------------- extras
    def path_cost(self, X):
        """Per-row cost of the tests actually used on the row's path (each feature charged once); needs feature_costs."""
        _, U = self.predict_proba(X, return_features=True)
        return (U * self.fcost_[None, :]).sum(1)

    def features_used_(self, X):
        """Boolean matrix (rows x columns): which columns were consulted on each row's path."""
        return self.predict_proba(X, return_features=True)[1]

    def export_text(self, max_depth=8):
        names = self.cols_; lines = []
        def rec(nd, d, prefix):
            if nd["leaf"]:
                lines.append(f"{prefix}-> class {self.classes_[nd['pred']]}  (n={nd['n']:.0f})"); return
            j = nd["feat"]
            if d >= max_depth:
                lines.append(f"{prefix}[{names[j]} ...] (depth limit of the printout)"); return
            if nd["kind"] == "num":
                lines.append(f"{prefix}{names[j]} <= {nd['thr']:.4g}"); rec(nd["kids"]["l"], d + 1, prefix + "|   ")
                lines.append(f"{prefix}{names[j]} >  {nd['thr']:.4g}"); rec(nd["kids"]["r"], d + 1, prefix + "|   ")
            else:
                inv = {v: k for k, v in self.cats_[j].items()}
                for k, ch in sorted(nd["kids"].items()):
                    lines.append(f"{prefix}{names[j]} = {inv.get(k, k)}"); rec(ch, d + 1, prefix + "|   ")
        rec(self.tree_, 0, ""); return "\n".join(lines)

