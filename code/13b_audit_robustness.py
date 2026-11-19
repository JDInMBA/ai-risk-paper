"""
13b_audit_robustness.py
 (a) Excess-tail-beta panel with group-specific tail-day intercepts (level shift on tail days).
 (b) Wild-cluster bootstrap (Rademacher, by event) for the pooled event-study regressions,
     because 15 clusters are too few for reliable cluster-robust t-tests.
 (c) Event study excluding confounded events (the S-1/A window overlaps the US mid-term
     elections; the DeepSeek day was a market-wide shock).
 (d) Bootstrap standard errors for the EVT (generalised Pareto) shape parameter, which is
     weakly identified with about 30 to 70 exceedances per period.
 (e) Stress-day table recomputed on an AI index that excludes the FLD-3 firms, so the days
     being tested are not selected partly on the returns of the firms under test.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy.stats import genpareto
from common import *
r = pd.read_csv(OUT / "returns.csv", index_col=0, parse_dates=True)
firms = [c for c in r.columns if c in FLD]
res = []
for p in ["P1 AI boom", "P2 backlog era"]:
    rp = period_slice(r, p); rows = []
    for c in firms:
        if rp[c].notna().sum() < 100: continue
        sysr = rp[[n for n in CORE_AI if n != c]].mean(axis=1); tl = (sysr <= sysr.quantile(0.05)).astype(int)
        rows.append(pd.DataFrame({"date": rp.index, "ticker": c, "fld": FLD[c], "r": rp[c]*100, "ai": sysr*100, "tl": tl}).dropna())
    D = pd.concat(rows)
    for g in range(4):
        D[f"tb{g}"] = D["ai"] * D["tl"] * (D["fld"] == g); D[f"ti{g}"] = D["tl"] * (D["fld"] == g)
    m = smf.ols("r ~ C(ticker) + C(ticker):ai + ti0 + ti1 + ti2 + ti3 + tb0 + tb1 + tb2 + tb3 - 1", D).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(D.date)[0]})
    tst = m.t_test("tb3 - tb1 = 0")
    for g in range(4):
        res.append(dict(period=p, term=f"FLD{g} excess tail beta", coef=m.params[f"tb{g}"], t=m.tvalues[f"tb{g}"], p=m.pvalues[f"tb{g}"]))
        res.append(dict(period=p, term=f"FLD{g} tail-day intercept", coef=m.params[f"ti{g}"], t=m.tvalues[f"ti{g}"], p=m.pvalues[f"ti{g}"]))
    res.append(dict(period=p, term="FLD3 - FLD1 excess tail beta", coef=float(np.ravel(tst.effect)[0]), t=float(np.ravel(tst.tvalue)[0]), p=float(np.ravel(tst.pvalue)[0])))
    # implied tail-day sensitivity
    Dt = D[D.tl == 1]; res.append(dict(period=p, term="mean tail-day return FLD3 / FLD1 / FLD2 / FLD0 (%)",
        coef=np.nan, t=np.nan, p=np.nan, note=" / ".join(f"{Dt[Dt.fld==g].r.mean():.2f}" for g in [3,1,2,0])))
A = pd.DataFrame(res); A.to_csv(OUT / "audit_tail_beta_with_intercepts.csv", index=False); print(A.round(3).to_string())

# (b) wild cluster bootstrap
ev = pd.read_csv(OUT / "event_study_firm.csv"); ev["absCAR"] = ev.CAR.abs(); ev["signedCAR"] = np.where(ev.exp_sign=="+", ev.CAR, -ev.CAR)
def wcb(formula, term, data, B=9999, seed=1):
    full = smf.ols(formula, data).fit(cov_type="cluster", cov_kwds={"groups": data.date})
    restricted = smf.ols(formula.replace(" + " + term, "").replace(term + " + ", ""), data).fit() if False else None
    # impose the null manually
    import patsy
    y, X = patsy.dmatrices(formula, data, return_type="dataframe")
    col = [c for c in X.columns if term in c][0]
    Xr = X.drop(columns=[col]); br = np.linalg.lstsq(Xr, y, rcond=None)[0]
    fit_r = (Xr @ br).values.ravel(); u_r = y.values.ravel() - fit_r
    cl = pd.factorize(data.date)[0]; G = cl.max() + 1; rng = np.random.default_rng(seed)
    t_obs = full.tvalues[col]; ts = []
    for _ in range(B):
        w = rng.choice([-1.0, 1.0], G)[cl]; yb = fit_r + u_r * w
        mb = __import__("statsmodels.api").api.OLS(yb, X).fit(cov_type="cluster", cov_kwds={"groups": cl}); ts.append(mb.tvalues[col])
    ts = np.array(ts); return full.params[col], t_obs, float(np.mean(np.abs(ts) >= abs(t_obs))), G
rows = []
for f_, lab in [("absCAR ~ C(fld) + sd", "abs CAR"), ("signedCAR ~ C(fld)", "signed CAR")]:
    b, t, p, G = wcb(f_, "C(fld)[T.3]", ev, B=1999)
    rows.append(dict(model=lab, term="FLD 3 vs FLD 0", coef=b, cluster_t=t, wild_cluster_p=p, clusters=G))
W = pd.DataFrame(rows); W.to_csv(OUT / "audit_event_wild_bootstrap.csv", index=False); print(W.round(3).to_string())

# (c) excluding confounded windows
CONFOUNDED = {"2026-11-03": "Anthropic S-1/A window overlaps the US mid-term elections",
              "2025-01-27": "DeepSeek shock was a market-wide repricing"}
samples = [("All 15 events", ev),
           ("Excluding the S-1/A window", ev[ev.date != "2026-11-03"]),
           ("Excluding both confounded events", ev[~ev.date.isin(CONFOUNDED)])]
rows = []
for lab, d in samples:
    for f_, mlab in [("absCAR ~ C(fld) + sd", "abs CAR"), ("signedCAR ~ C(fld)", "signed CAR")]:
        m = smf.ols(f_, d).fit(cov_type="cluster", cov_kwds={"groups": d.date})
        k = "C(fld)[T.3]"
        b, t, pw, G = wcb(f_, k, d, B=1999)
        rows.append(dict(sample=lab, model=mlab, events=d.date.nunique(), n=len(d),
                         fld3_coef=m.params[k], cluster_p=m.pvalues[k], wild_cluster_p=pw))
E = pd.DataFrame(rows); E.to_csv(OUT / "audit_event_drop_confounded.csv", index=False)
print("\n(c) event study excluding confounded windows"); print(E.round(4).to_string())

# (d) bootstrap GPD shape
def pot_fit(L, u_q=0.90, pq=0.99):
    u = L.quantile(u_q); exc = L[L > u] - u
    xi, _, beta = genpareto.fit(exc, floc=0)
    n, nu = len(L), len(exc)
    var = u + beta/xi * (((n/nu)*(1-pq))**(-xi) - 1)
    return xi, var*100, nu
rng = np.random.default_rng(5)
ai = r["AI_IDX"]; ctl = r["CTL_IDX"]
rows = []
for p in PERIODS:
    # paired bootstrap on days
    P_ = pd.concat([period_slice(ai, p).rename("AI_IDX"), period_slice(ctl, p).rename("CTL_IDX"),
                    period_slice(r["SPY"], p).rename("SPY")], axis=1).dropna()
    base = {nm: pot_fit(-P_[nm]) for nm in P_.columns}
    bs = {nm: [] for nm in P_.columns}; ratio = []
    for _ in range(1000):
        Pb = P_.iloc[rng.integers(0, len(P_), len(P_))]
        try:
            f = {nm: pot_fit(-Pb[nm]) for nm in P_.columns}
            for nm in P_.columns: bs[nm].append(f[nm])
            ratio.append(f["AI_IDX"][1] / f["CTL_IDX"][1])
        except Exception: pass
    for nm in P_.columns:
        xi, var, nu = base[nm]
        xb = np.array([b[0] for b in bs[nm]]); vb = np.array([b[1] for b in bs[nm]])
        rows.append(dict(period=p, series=nm, n_exc=nu, xi=xi, xi_se=xb.std(ddof=1),
                         xi_lo=np.quantile(xb, .025), xi_hi=np.quantile(xb, .975),
                         xi_gt0_frac=float((xb > 0).mean()),
                         VaR99=var, VaR99_lo=np.quantile(vb, .025), VaR99_hi=np.quantile(vb, .975)))
    rr = np.array(ratio)
    rows.append(dict(period=p, series="AI/CTL VaR99 ratio", n_exc=np.nan,
                     VaR99=base["AI_IDX"][1]/base["CTL_IDX"][1],
                     VaR99_lo=np.quantile(rr, .025), VaR99_hi=np.quantile(rr, .975)))
X = pd.DataFrame(rows); X.to_csv(OUT / "audit_evt_shape.csv", index=False)
print("\n(d) GPD shape and 99% VaR with paired-bootstrap intervals"); print(X.round(3).to_string())

# (e) FLD3-purged index
names = [c for c in r.columns if c in FLD]
FLD3 = [c for c in names if FLD[c] == 3]
purged = [n for n in CORE_AI if FLD.get(n) != 3]        # drops ORCL and AMD from the index
ai_purged = r[purged].mean(axis=1)
rows = []
for lab, idx in [("AI index (as published)", r["AI_IDX"]), ("AI index excluding FLD-3 firms", ai_purged)]:
    i2 = period_slice(idx, "P2 backlog era"); worst = i2.nsmallest(10)
    wins = 0
    for d in worst.index:
        row = r.loc[d]
        means = {k: row[[c for c in names if FLD[c] == k and pd.notna(row.get(c))]].mean()*100 for k in range(4)}
        if means[3] == min(means.values()): wins += 1
        rows.append(dict(index=lab, date=d.date(), idx_ret=i2[d]*100, **{f"fld{k}": means[k] for k in range(4)}))
    rows.append(dict(index=lab, date="FLD-3 worst on n of 10 days", idx_ret=np.nan,
                     fld0=np.nan, fld1=np.nan, fld2=np.nan, fld3=wins))
S = pd.DataFrame(rows); S.to_csv(OUT / "audit_stress_days_purged.csv", index=False)
print("\n(e) stress days, published vs FLD3-purged index"); print(S.round(2).to_string())
