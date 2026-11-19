"""
07_var_backtest.py
One-day 99% and 95% VaR (and 97.5% ES) for four portfolios, out of sample 2024-01-02..2026-11-05:
  AI_IDX (core AI complex), FLD3 basket (lab-dependent names), FLD2 basket (hyperscalers/
  diversified suppliers), CTL_IDX (control firms).
Models: historical simulation (250d), RiskMetrics EWMA (normal), GARCH(1,1)-t, filtered HS
(GARCH-filtered bootstrap), and a linear quantile regression on lagged system-state variables
(QR-state), re-estimated quarterly on an expanding window.
Backtests: Kupiec POF, Christoffersen independence and conditional coverage, Basel traffic light
(last 250 days, 99%), tick-loss with Diebold-Mariano test vs GARCH-t, McNeil-Frey ES test.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy.stats import chi2, norm, t as tdist
from arch import arch_model
from common import *

r = pd.read_csv(OUT / "returns.csv", index_col=0, parse_dates=True)
px = load_prices()
conn = pd.read_csv(OUT / "dy_total_connectedness.csv", index_col=0, parse_dates=True).iloc[:,0]
bs = pd.read_csv(OUT / "bsadf_sequences.csv", header=[0,1], index_col=0, parse_dates=True)
bsgap = (bs[("AI_IDX","bsadf")] - bs[("AI_IDX","cv95")]).shift(1).reindex(r.index, method="ffill")

# baskets follow the FLD scores in common.py
FLD3 = sorted(t for t, f in FLD.items() if f == 3); FLD2 = sorted(t for t, f in FLD.items() if f == 2)
FLD3_FIXED = sorted(t for t in FLD3 if r[t].loc[:"2022-11-29"].notna().any())   # traded before P1
ports = {"AI_IDX": r["AI_IDX"], "FLD3_basket": r[FLD3].mean(axis=1, skipna=True),
         "FLD3_fixed": r[FLD3_FIXED].mean(axis=1),   # constant membership (robustness)
         "FLD2_basket": r[FLD2].mean(axis=1), "CTL_IDX": r["CTL_IDX"]}
TEST0 = "2024-01-02"

def state(x):
    """Parsimonious CAViaR-style state vector: own volatility, last absolute return, VIX and
    AI-system connectedness (z-score; zero before it is defined)."""
    S = pd.DataFrame(index=r.index)
    S["rv22"] = x.rolling(22).std(); S["absr"] = x.abs()
    S["vix"] = px["^VIX"].reindex(r.index) / 100
    cz = (conn - conn.rolling(250, min_periods=120).mean()) / conn.rolling(250, min_periods=120).std()
    S["conn_z"] = cz.reindex(r.index).fillna(0)
    return S

def kupiec(I, p):
    n, x = len(I), I.sum(); ph = x / n
    if x == 0: return -2*n*np.log(1-p), 1 - chi2.cdf(-2*n*np.log(1-p), 1)
    lr = -2*((n-x)*np.log(1-p) + x*np.log(p) - (n-x)*np.log(1-ph) - x*np.log(ph))
    return lr, 1 - chi2.cdf(lr, 1)
def christoffersen(I):
    I = np.asarray(I, int); a, b = I[:-1], I[1:]
    n00 = ((a==0)&(b==0)).sum(); n01 = ((a==0)&(b==1)).sum(); n10 = ((a==1)&(b==0)).sum(); n11 = ((a==1)&(b==1)).sum()
    p01 = n01/max(n00+n01,1); p11 = n11/max(n10+n11,1); p = (n01+n11)/max(n00+n01+n10+n11,1)
    L = lambda q, k0, k1: (k0*np.log(1-q) if k0 and q<1 else 0) + (k1*np.log(q) if k1 and q>0 else 0)
    lr = -2*(L(p, n00+n10, n01+n11) - L(p01, n00, n01) - L(p11, n10, n11))
    return lr, 1 - chi2.cdf(lr, 1)
def tick(y, q, a): return ((y < q).astype(float) - a) * (q - y) * -1 if False else (a - (y < q)) * (y - q)
def dm(l1, l2):
    d = (l1 - l2).dropna(); T = len(d); h = int(T**(1/3))
    g = [d.cov(d.shift(k)) if k else d.var() for k in range(h+1)]
    v = g[0] + 2*sum((1-k/(h+1))*g[k] for k in range(1, h+1))
    s = d.mean() / np.sqrt(v / T); return s, 2*(1-norm.cdf(abs(s)))

results, series_out = [], {}
for pname, x in ports.items():
    x = x.dropna(); S = state(x).reindex(x.index)
    test_idx = x.loc[TEST0:].index
    F = {f"{m}_{a}": pd.Series(np.nan, index=test_idx) for m in ["HS","EWMA","GARCH-t","FHS","QR-state"] for a in ["99","95"]}
    ES = {f"{m}": pd.Series(np.nan, index=test_idx) for m in ["GARCH-t","FHS"]}
    V975 = {f"{m}": pd.Series(np.nan, index=test_idx) for m in ["GARCH-t","FHS"]}
    SIG = pd.Series(np.nan, index=test_idx)
    # HS and EWMA
    lam = 0.94; ew = x.pow(2).ewm(alpha=1-lam, adjust=False).mean()
    for a, al in [("99", .01), ("95", .05)]:
        F[f"HS_{a}"] = x.rolling(250).quantile(al).shift(1).reindex(test_idx)
        F[f"EWMA_{a}"] = (norm.ppf(al) * np.sqrt(ew)).shift(1).reindex(test_idx)
    # GARCH-t and FHS, 21-day refit
    pos0 = x.index.get_loc(test_idx[0])
    for k in range(pos0, len(x), 21):
        win = x.iloc[max(0, k-1000):k] * 100
        am = arch_model(win, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
        mu, om, al1, be = am.params["mu"], am.params["omega"], am.params["alpha[1]"], am.params["beta[1]"]; nu = am.params["nu"]
        z = (am.resid / am.conditional_volatility).dropna().values
        s2 = am.conditional_volatility.iloc[-1]**2; eps = am.resid.iloc[-1]
        for j in range(k, min(k+21, len(x))):
            s2 = om + al1*eps**2 + be*s2; sig = np.sqrt(s2)
            d = x.index[j]
            for a, al in [("99", .01), ("95", .05)]:
                qt = tdist.ppf(al, nu) * np.sqrt((nu-2)/nu)
                F[f"GARCH-t_{a}"][d] = (mu + sig*qt) / 100
                F[f"FHS_{a}"][d] = (mu + sig*np.quantile(z, al)) / 100
            # ES at 97.5%, FRTB level
            SIG[d] = sig / 100
            V975["GARCH-t"][d] = (mu + sig*tdist.ppf(.025, nu)*np.sqrt((nu-2)/nu)) / 100
            V975["FHS"][d] = (mu + sig*np.quantile(z, .025)) / 100
            qt = tdist.ppf(.025, nu); es_t = -(tdist.pdf(qt, nu)/.025)*((nu+qt**2)/(nu-1))*np.sqrt((nu-2)/nu)
            ES["GARCH-t"][d] = (mu + sig*es_t) / 100
            zq = np.quantile(z, .025); ES["FHS"][d] = (mu + sig*z[z <= zq].mean()) / 100
            eps = x.iloc[j]*100 - mu
    # QR-state, quarterly refit
    Z = S.shift(1)
    for q in pd.period_range("2024Q1", "2026Q4", freq="Q"):
        te = test_idx[(test_idx >= q.start_time) & (test_idx <= q.end_time)]
        if len(te) == 0: continue
        trn = pd.concat([x, Z], axis=1).loc[:q.start_time - pd.Timedelta(days=1)].dropna()
        X = sm.add_constant(trn.iloc[:,1:]); Xte = sm.add_constant(Z.loc[te], has_constant="add")
        for a, al in [("99", .01), ("95", .05)]:
            m = sm.QuantReg(trn.iloc[:,0], X).fit(q=al, max_iter=5000)
            F[f"QR-state_{a}"].loc[te] = m.predict(Xte.dropna()).reindex(te)
    y = x.loc[test_idx]
    for mname in ["HS","EWMA","GARCH-t","FHS","QR-state"]:
        for a, al in [("99", .01), ("95", .05)]:
            v = F[f"{mname}_{a}"]; ok = v.notna(); I = (y[ok] < v[ok]).astype(int)
            lrk, pk = kupiec(I, al); lri, pi = christoffersen(I)
            lrc = lrk + lri; pc = 1 - chi2.cdf(lrc, 2)
            last = I.iloc[-250:].sum() if a == "99" else np.nan
            zone = ("green" if last <= 4 else "yellow" if last <= 9 else "red") if a == "99" else ""
            results.append(dict(portfolio=pname, model=mname, level=a, N=int(ok.sum()), exceed=int(I.sum()),
                                rate=I.mean(), kupiec_p=pk, ind_p=pi, cc_p=pc, last250=last, basel=zone,
                                tickloss=float(tick(y[ok], v[ok], al).mean()*1e4)))
            series_out[(pname, f"{mname}_{a}")] = v
    # DM tests against GARCH-t
    for mname in ["HS","EWMA","FHS","QR-state"]:
        l1 = tick(y, F[f"{mname}_95"], .05); l2 = tick(y, F["GARCH-t_95"], .05)
        s, p = dm(l1, l2)
        results.append(dict(portfolio=pname, model=f"DM {mname} vs GARCH-t (95%)", level="95", dm_stat=s, dm_p=p))
    # McNeil-Frey ES test, one-sided
    for mname in ["GARCH-t","FHS"]:
        es, v975 = ES[mname], V975[mname]
        br = y < v975; resid = (y[br] - es[br]) / SIG[br]
        n = int(br.sum())
        tst = resid.mean() / (resid.std(ddof=1)/np.sqrt(n)) if n > 2 else np.nan
        # bootstrap p-value, skewed residuals
        rb = np.random.default_rng(7); c = resid - resid.mean(); bt = []
        for _ in range(5000):
            sm_ = rb.choice(c.values, n); bt.append(sm_.mean() / (sm_.std(ddof=1)/np.sqrt(n)))
        p_boot = float(np.mean(np.array(bt) <= tst)) if n > 2 else np.nan
        results.append(dict(portfolio=pname, model=f"ES97.5 McNeil-Frey {mname}", level="97.5", exceed=n, rate=n/len(y),
                            mf_mean=float(resid.mean()), mf_t=float(tst), mf_p=float(tdist.cdf(tst, n-1)) if n>2 else np.nan, mf_p_boot=p_boot))
    print(pname, "done")
R = pd.DataFrame(results); R.to_csv(OUT / "var_backtests.csv", index=False)
pd.DataFrame(series_out).to_csv(OUT / "var_forecasts.csv")
pd.concat(ports, axis=1).to_csv(OUT / "portfolio_returns.csv")
pd.set_option("display.width", 250)
print(R[R.level.isin(["99","95"]) & R.model.isin(["HS","EWMA","GARCH-t","FHS","QR-state"])][["portfolio","model","level","exceed","rate","kupiec_p","ind_p","cc_p","last250","basel","tickloss"]].round(3).to_string())
print(R[R.model.str.startswith(("DM","ES"))].dropna(axis=1, how="all").round(3).to_string())
