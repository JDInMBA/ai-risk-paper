"""
06_ml_tail_events.py
Out-of-sample prediction of next-day VaR breaches for AI-complex stocks.
Target  : 1{ R_{i,t+1:t+h} < q5_{i,t}(h) }, where q5 is the rolling 500-day 5% quantile of the
          firm's own h-day returns (h=1 is a next-day 95% HS-VaR breach). h in {1, 5, 10}.
Features: firm risk state, AI-complex system state (connectedness, bubble statistic,
          index drawdown), credit/macro state, and the frontier-lab dependence score.
Design  : expanding-window walk-forward, re-estimated every quarter, test 2024Q1-2026Q4.
Models  : logit (vol only), logit (all), random forest, LightGBM, LightGBM without the
          AI-system/lab features (ablation).  SHAP attribution for LightGBM.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb, shap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from common import *

r = pd.read_csv(OUT / "returns.csv", index_col=0, parse_dates=True)
px = load_prices(); fred = load_fred()
conn = pd.read_csv(OUT / "dy_total_connectedness.csv", index_col=0, parse_dates=True).iloc[:,0]
bs = pd.read_csv(OUT / "bsadf_sequences.csv", header=[0,1], index_col=0, parse_dates=True)
bsadf_ai = bs[("AI_IDX","bsadf")] - bs[("AI_IDX","cv95")]
bsadf_ai = bsadf_ai.shift(1).reindex(r.index, method="ffill")   # last *completed* week only

firms = [c for c in r.columns if FLD.get(c, -1) >= 1]
ai = r["AI_IDX"]
# system features, known at t
S = pd.DataFrame(index=r.index)
S["ai_rv22"] = ai.rolling(22).std()*np.sqrt(252)
S["ai_ret5"] = ai.rolling(5).sum()
S["ai_dd252"] = np.exp(ai.cumsum()) / np.exp(ai.cumsum()).rolling(252, min_periods=60).max() - 1
S["ai_minus_ctl_22"] = (ai - r["CTL_IDX"]).rolling(22).sum()
S["conn"] = conn.reindex(r.index); S["dconn22"] = S["conn"].diff(22)
S["bsadf_gap"] = bsadf_ai
S["vix"] = px["^VIX"].reindex(r.index); S["dvix5"] = S["vix"].diff(5)
S["hyg_ret5"] = r["HYG"].rolling(5).sum(); S["tlt_ret5"] = r["TLT"].rolling(5).sum()
S["ust10"] = fred["ust10"].reindex(r.index).ffill(); S["d10_5"] = S["ust10"].diff(5); S["d10_22"] = S["ust10"].diff(22)
S["conn_z"] = (S["conn"] - S["conn"].rolling(250, min_periods=120).mean()) / S["conn"].rolling(250, min_periods=120).std()
S["spy_rv22"] = r["SPY"].rolling(22).std()*np.sqrt(252)

def run(H):
    rows = []
    for c in firms:
        x = r[c]
        if x.notna().sum() < 300: continue
        F = pd.DataFrame(index=r.index)
        F["ret1"] = x; F["ret5"] = x.rolling(5, min_periods=4).sum(); F["ret22"] = x.rolling(22, min_periods=18).sum()
        F["rv5"] = x.rolling(5, min_periods=4).std()*np.sqrt(252); F["rv22"] = x.rolling(22, min_periods=18).std()*np.sqrt(252)
        F["rv66"] = x.rolling(66, min_periods=50).std()*np.sqrt(252); F["rv_ratio"] = F.rv5 / F.rv66
        F["semivol22"] = x.clip(upper=0).rolling(22, min_periods=18).apply(lambda v: np.sqrt((v**2).mean()), raw=True)*np.sqrt(252)
        lp = np.log(px[c].reindex(r.index))
        F["dd252"] = np.exp(lp - lp.rolling(252, min_periods=60).max()) - 1
        cov = x.rolling(66, min_periods=50).cov(ai); F["beta_ai66"] = cov / ai.rolling(66, min_periods=50).var()
        F["corr_ai66"] = x.rolling(66, min_periods=50).corr(ai)
        F["fld"] = FLD[c]
        var = -x.rolling(250, min_periods=200).quantile(0.05)
        F["hs_var"] = var
        xf = x.fillna(0)
        # quantiles on observed returns only
        xv = x.dropna()
        fwd = xv[::-1].rolling(H).sum()[::-1].shift(-1).reindex(x.index)   # R_{t+1:t+H}
        hret = xv.rolling(H).sum()
        q5 = hret.rolling(500, min_periods=250).quantile(0.05).reindex(x.index)
        F["y"] = (fwd < q5).astype(float)
        F.loc[fwd.isna() | q5.isna() | x.isna(), "y"] = np.nan
        F["ticker"] = c
        rows.append(F.join(S))
    P = pd.concat(rows).reset_index().rename(columns={"index":"date"})
    P = P[P.date >= "2022-01-01"].dropna(subset=["y"])
    P.to_parquet(OUT / "ml_panel.parquet") if False else None
    print("panel", P.shape, "breach rate", P.y.mean().round(4))

    FIRM_F = ["ret1","ret5","ret22","rv5","rv22","rv66","rv_ratio","semivol22","dd252","hs_var"]
    AI_F   = ["beta_ai66","corr_ai66","fld","ai_rv22","ai_ret5","ai_dd252","ai_minus_ctl_22","conn_z","dconn22","bsadf_gap"]
    MAC_F  = ["vix","dvix5","hyg_ret5","tlt_ret5","d10_5","d10_22","spy_rv22"]
    ALL = FIRM_F + AI_F + MAC_F; NOAI = FIRM_F + MAC_F
    P = P.dropna(subset=ALL)

    def models():
        return {
         "Logit (vol only)": (["rv22","rv_ratio","hs_var"], make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
         "Logit (all)":      (ALL, make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=0.5))),
         # n_jobs=1 fixes averaging order
         "Random forest":    (ALL, RandomForestClassifier(n_estimators=400, min_samples_leaf=200, max_features=0.4, n_jobs=1, random_state=1)),
         "LightGBM (all)":   (ALL, lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=300,
                                                      subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5, verbose=-1, random_state=1,
                                                      # reproducibility, not random_state
                                                      deterministic=True, force_row_wise=True, n_jobs=1)),
         "LightGBM (no AI/lab features)": (NOAI, lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=300,
                                                      subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5, verbose=-1, random_state=1,
                                                      # reproducibility, not random_state
                                                      deterministic=True, force_row_wise=True, n_jobs=1)),
        }

    qs = pd.period_range("2024Q1", "2026Q4", freq="Q")
    preds = []
    for q in qs:
        a, b = q.start_time, q.end_time
        tr = P[P.date < a - pd.Timedelta(days=2 + int(1.5*H))]; te = P[(P.date >= a) & (P.date <= b)]   # embargo >= horizon
        if te.empty: continue
        for nm, (cols, m) in models().items():
            m.fit(tr[cols], tr.y)
            p = m.predict_proba(te[cols])[:,1]
            preds.append(pd.DataFrame({"date": te.date.values, "ticker": te.ticker.values, "fld": te.fld.values,
                                       "y": te.y.values, "model": nm, "p": p}))
        print("H", H, "quarter", q, "train", len(tr), "test", len(te))
    PR = pd.concat(preds); PR.to_csv(OUT / f"ml_oos_predictions_h{H}.csv", index=False)

    def metrics(d):
        return pd.Series(dict(AUC=roc_auc_score(d.y, d.p), PR_AUC=average_precision_score(d.y, d.p),
                              Brier=brier_score_loss(d.y, d.p), LogLoss=log_loss(d.y, d.p.clip(1e-6,1-1e-6)),
                              N=len(d), breach_rate=d.y.mean()))
    tab = PR.groupby("model").apply(metrics); tab["sample"] = "All OOS 2024Q1-2026Q4"
    tab2 = PR[PR.date >= "2025-09-10"].groupby("model").apply(metrics); tab2["sample"] = "P2 backlog era"
    tab3 = PR[PR.fld == 3].groupby("model").apply(metrics); tab3["sample"] = "FLD=3 firms"
    M = pd.concat([tab, tab2, tab3]); M.to_csv(OUT / f"ml_metrics_h{H}.csv"); print(M.round(4).to_string())

    # day-block bootstrap, AUC gaps
    rng = np.random.default_rng(3)
    W = PR.pivot_table(index=["date","ticker"], columns="model", values="p").join(
        PR.drop_duplicates(["date","ticker"]).set_index(["date","ticker"])["y"])
    days = W.index.get_level_values(0).unique()
    diffs = {"LGBM(all) - LGBM(noAI)": ("LightGBM (all)","LightGBM (no AI/lab features)"),
             "LGBM(all) - Logit(vol)": ("LightGBM (all)","Logit (vol only)"),
             "LGBM(all) - Logit(all)": ("LightGBM (all)","Logit (all)")}
    bt = {k: [] for k in diffs}
    blocks = [days[i:i+20] for i in range(0, len(days), 20)]
    for _ in range(300):
        sel = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
        Wb = W.loc[sel]
        for k, (m1, m2) in diffs.items():
            bt[k].append(roc_auc_score(Wb.y, Wb[m1]) - roc_auc_score(Wb.y, Wb[m2]))
    bts = pd.DataFrame({k: dict(mean=np.mean(v), lo=np.quantile(v,.025), hi=np.quantile(v,.975), p_le0=np.mean(np.array(v)<=0)) for k, v in bt.items()}).T
    bts.to_csv(OUT / f"ml_auc_bootstrap_h{H}.csv"); print(bts.round(4))

    # calibration by decile
    L = PR[PR.model=="LightGBM (all)"].copy(); L["dec"] = pd.qcut(L.p, 10, labels=False, duplicates="drop")
    L.groupby("dec").agg(p_mean=("p","mean"), y_mean=("y","mean"), n=("y","size")).to_csv(OUT / f"ml_calibration_h{H}.csv")

    # SHAP, final model
    cols, m = models()["LightGBM (all)"]
    trf = P[P.date <= "2026-06-30"]; m.fit(trf[cols], trf.y)
    samp = P[P.date > "2025-09-09"].sample(min(6000, (P.date > "2025-09-09").sum()), random_state=0)
    sv = shap.TreeExplainer(m).shap_values(samp[cols])
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.Series(np.abs(sv).mean(0), index=cols).sort_values(ascending=False)
    imp.to_csv(OUT / f"ml_shap_importance_h{H}.csv"); print(imp.round(4))
    np.save(OUT / f"shap_values_h{H}.npy", sv); samp[cols].to_csv(OUT / f"shap_sample_h{H}.csv", index=False)

if __name__ == '__main__':
    import sys
    for H in [int(a) for a in sys.argv[1:]] or [1, 5, 10]:
        run(H)
