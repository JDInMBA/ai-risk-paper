"""
06b_ml_stress_days.py
Cross-sectional question: on AI-complex stress days, which firms suffer the worst losses?
Stress day d: AI_IDX return below its own rolling 500-day 5% quantile, known ex ante (t-1).
Target: firm i in the worst tercile of AI-universe returns on day d.
Features (as of d-1): rolling beta/correlation to AI index, volatility, momentum, drawdown,
idiosyncratic volatility, and (in the augmented model) the frontier-lab dependence score.
Train: stress days up to 2025-09-09 (P0-P1). Test: stress days in P2 (backlog era).
Also a rolling-origin scheme (train < year Y, test year Y) for 2024, 2025, 2026.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from common import *

r = pd.read_csv(OUT / "returns.csv", index_col=0, parse_dates=True)
px = load_prices(); ai = r["AI_IDX"]
thr = ai.rolling(500, min_periods=250).quantile(0.05).shift(1)
stress = ai.index[(ai < thr)]
firms = [c for c in r.columns if FLD.get(c, -1) >= 1]
feat = {}
for c in firms:
    x = r[c]; F = pd.DataFrame(index=r.index)
    F["beta_ai"] = x.rolling(120, min_periods=60).cov(ai) / ai.rolling(120, min_periods=60).var()
    F["corr_ai"] = x.rolling(120, min_periods=60).corr(ai)
    F["rv22"] = x.rolling(22, min_periods=18).std()*np.sqrt(252); F["rv120"] = x.rolling(120, min_periods=60).std()*np.sqrt(252)
    F["idio"] = np.sqrt(np.maximum(F.rv120**2 * (1 - F.corr_ai**2), 0))
    F["ret22"] = x.rolling(22, min_periods=18).sum(); F["ret120"] = x.rolling(120, min_periods=60).sum()
    lp = np.log(px[c].reindex(r.index)); F["dd252"] = np.exp(lp - lp.rolling(252, min_periods=60).max()) - 1
    F["fld"] = FLD[c]; F["fld3"] = int(FLD[c] == 3); F["fld2"] = int(FLD[c] == 2)
    feat[c] = F.shift(1)   # information as of d-1
rows = []
for d in stress:
    rd = r.loc[d, firms].dropna()
    if len(rd) < 15: continue
    cut = rd.quantile(1/3)
    for c in rd.index:
        f = feat[c].loc[d]
        if f[["beta_ai","rv22","dd252"]].isna().any(): continue
        rows.append(dict(date=d, ticker=c, y=int(rd[c] <= cut), ret=rd[c], **f.to_dict()))
D = pd.DataFrame(rows); D.to_csv(OUT / "stress_day_panel.csv", index=False)
print("stress days", D.date.nunique(), "obs", len(D))

BASE = ["beta_ai","corr_ai","rv22","rv120","idio","ret22","ret120","dd252"]
AUG = BASE + ["fld3","fld2","fld"]
def mk(kind):
    if kind == "logit": return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=2,
                              # Reproducibility settings; see 06_ml_tail_events.py.
                              deterministic=True, force_row_wise=True, n_jobs=1)
def evaluate(tr, te, tag):
    out = []; preds = {}
    for kind in ["logit","lgbm"]:
        for fs, nm in [(BASE,"market-only"),(AUG,"market + lab dependence")]:
            m = mk(kind).fit(tr[fs], tr.y); p = m.predict_proba(te[fs])[:,1]; preds[(kind,nm)] = p
            # day-block bootstrap, AUC level
            rl = np.random.default_rng(11); dys = te.date.unique(); lv = []
            for _ in range(1000):
                sel = rl.choice(dys, len(dys))
                ix = np.concatenate([np.where(te.date.values == s)[0] for s in sel])
                try: lv.append(roc_auc_score(te.y.values[ix], p[ix]))
                except ValueError: pass
            lv = np.array(lv)
            out.append(dict(split=tag, model=kind, features=nm, AUC=roc_auc_score(te.y, p),
                            AUC_lo=np.quantile(lv, .025), AUC_hi=np.quantile(lv, .975),
                            n_test=len(te), days=te.date.nunique()))
    # day-block bootstrap, AUC gain
    rng = np.random.default_rng(0); days = te.date.unique(); g = {k: [] for k in ["logit","lgbm"]}
    for _ in range(1000):
        sel = rng.choice(days, len(days)); idx = np.concatenate([np.where(te.date.values == s)[0] for s in sel])
        for k in g:
            try: g[k].append(roc_auc_score(te.y.values[idx], preds[(k,"market + lab dependence")][idx]) - roc_auc_score(te.y.values[idx], preds[(k,"market-only")][idx]))
            except ValueError: pass
    for k in g:
        v = np.array(g[k]); out.append(dict(split=tag, model=k, features="AUC gain (bootstrap)", AUC=v.mean(), lo=np.quantile(v,.025), hi=np.quantile(v,.975), p_le0=(v<=0).mean()))
    return out, preds
res = []
tr = D[D.date <= "2025-09-09"]; te = D[D.date >= "2025-09-10"]
o, preds = evaluate(tr, te, "Train P0-P1 / Test P2"); res += o
te = te.assign(p_aug=preds[("logit","market + lab dependence")], p_base=preds[("logit","market-only")])
te.to_csv(OUT / "stress_day_test_predictions.csv", index=False)
for Y in [2024, 2025, 2026]:
    o, _ = evaluate(D[D.date < f"{Y}-01-01"], D[(D.date >= f"{Y}-01-01") & (D.date <= f"{Y}-12-31")], f"Rolling origin test {Y}"); res += o
R = pd.DataFrame(res); R.to_csv(OUT / "ml_stress_day_results.csv", index=False)
pd.set_option("display.width", 200); print(R.round(4).to_string())
# standardised logit coefficients
m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(D[AUG], D.y)
print(pd.Series(m[-1].coef_[0], index=AUG).round(3))
