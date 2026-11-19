"""
03_tail_risk.py
System-level and firm-level tail-risk measures for the AI infrastructure complex.
 (a) Rolling Diebold-Yilmaz (2012) total connectedness, generalised FEVD, VAR(1), H=10
 (b) Delta-CoVaR (Adrian & Brunnermeier 2016) by quantile regression, per sub-period
 (c) Marginal Expected Shortfall (Acharya et al. 2017) w.r.t. AI-complex index
 (d) Peaks-over-threshold EVT (GPD) VaR/ES of AI index vs control index
 (e) Lower-tail dependence (empirical) between firms and the AI index
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy.stats import genpareto
from common import *

px = load_prices(); r = log_returns(px).iloc[1:]
fred = load_fred()
ai = ai_index_returns(r); ctl = r[CONTROL].mean(axis=1)
r["AI_IDX"] = ai; r["CTL_IDX"] = ctl
r.to_csv(OUT / "returns.csv")

# (a) Diebold-Yilmaz connectedness
def gfevd_spillover(X, p=1, H=10):
    X = X - X.mean(0)
    T, k = X.shape
    Y = X[p:]; Z = np.hstack([X[p-1-j:T-1-j] for j in range(p)])
    Z = np.hstack([np.ones((len(Z),1)), Z])
    B = np.linalg.lstsq(Z, Y, rcond=None)[0]
    U = Y - Z @ B; S = U.T @ U / (len(U) - Z.shape[1])
    A = [B[1+j*k:1+(j+1)*k].T for j in range(p)]
    Phi = [np.eye(k)]
    for h in range(1, H):
        Phi.append(sum(A[j] @ Phi[h-1-j] for j in range(min(p, h))))
    sig = np.diag(S)
    num = np.zeros((k,k)); den = np.zeros(k)
    for P in Phi:
        PS = P @ S
        num += (PS ** 2) / sig[None, :]
        den += np.diag(PS @ P.T)
    theta = num / den[:, None]
    theta = theta / theta.sum(1, keepdims=True)
    total = (theta.sum() - np.trace(theta)) / k * 100
    to_ = (theta.sum(0) - np.diag(theta)) * 100
    from_ = (theta.sum(1) - np.diag(theta)) * 100
    return total, to_ - from_, theta

R = r[CORE_AI].dropna()
W = 250
tot, net = [], []
for i in range(W, len(R)+1):
    t_, n_, _ = gfevd_spillover(R.values[i-W:i])
    tot.append((R.index[i-1], t_)); net.append(pd.Series(n_, index=CORE_AI, name=R.index[i-1]))
conn = pd.Series(dict(tot)).rename("total_connectedness")
netdf = pd.DataFrame(net)
conn.to_csv(OUT / "dy_total_connectedness.csv"); netdf.to_csv(OUT / "dy_net_directional.csv")

# static tables by period
static = {}
for p in PERIODS:
    Xp = period_slice(R, p).values
    t_, n_, th = gfevd_spillover(Xp)
    static[p] = dict(total=t_, **{f"net_{c}": v for c, v in zip(CORE_AI, n_)})
pd.DataFrame(static).T.to_csv(OUT / "dy_static_by_period.csv")

# extended system, P2 only
EXT = CORE_AI + ["CRWV","NBIS","IREN","APLD"]
Rx = period_slice(r[EXT].dropna(), "P2 backlog era")
t_, n_, th = gfevd_spillover(Rx.values)
pd.DataFrame({"net": n_}, index=EXT).assign(total=t_).to_csv(OUT / "dy_extended_P2.csv")

# ---------- state variables ----------
st = pd.DataFrame(index=r.index)
st["dvix"] = px["^VIX"].reindex(r.index).diff()
st["hyg"] = r["HYG"]; st["tlt"] = r["TLT"]; st["spy"] = r["SPY"]
st["d10"] = fred["ust10"].reindex(r.index).ffill().diff()
st["dterm"] = (fred["ust10"] - fred["ust2"]).reindex(r.index).ffill().diff()
st = st.shift(1)  # lagged states
st.to_csv(OUT / "state_vars.csv")

# (b) Delta-CoVaR, (c) MES
def dcovar(x_sys, x_i, M, q=0.05):
    df = pd.concat([x_sys.rename("sys"), x_i.rename("i"), M], axis=1).dropna()
    if len(df) < 150: return np.nan, np.nan, len(df)
    Xi = sm.add_constant(df[M.columns])
    # state-conditional firm VaR
    vq = sm.QuantReg(df["i"], Xi).fit(q=q).predict(Xi)
    vm = sm.QuantReg(df["i"], Xi).fit(q=0.5).predict(Xi)
    Xs = sm.add_constant(df[["i"] + list(M.columns)])
    mod = sm.QuantReg(df["sys"], Xs).fit(q=q)
    b = mod.params["i"]
    return float(b * (vq - vm).mean()) * 100, float(mod.pvalues["i"]), len(df)

universe = [c for c in r.columns if c in FLD and c not in ("SFTBY",)] + ["SFTBY"]
rows = []
M = st[["dvix","hyg","d10","dterm","spy"]]
for p in PERIODS:
    rp = period_slice(r, p)
    for c in universe:
        if c not in rp or rp[c].notna().sum() < 120: continue
        # system excludes the firm
        sys_names = [n for n in CORE_AI if n != c]
        sysr = rp[sys_names].mean(axis=1)
        d, pv, n = dcovar(sysr, rp[c], period_slice(M, p))
        # MES on worst 5% days
        both = pd.concat([sysr, rp[c]], axis=1).dropna()
        thr = both.iloc[:,0].quantile(0.05)
        mes = both.loc[both.iloc[:,0] <= thr].iloc[:,1].mean() * 100
        # downside and full beta
        lo = both.iloc[:,0] < both.iloc[:,0].median()
        beta = np.polyfit(both.iloc[:,0], both.iloc[:,1], 1)[0]
        dbeta = np.polyfit(both.iloc[:,0][lo], both.iloc[:,1][lo], 1)[0]
        # lower-tail dependence, 5%
        u = both.rank(pct=True)
        ltd = ((u.iloc[:,0] <= .05) & (u.iloc[:,1] <= .05)).mean() / .05
        var5 = -rp[c].quantile(0.05) * 100
        es5 = -rp[c][rp[c] <= rp[c].quantile(0.05)].mean() * 100
        rows.append(dict(period=p, ticker=c, fld=FLD.get(c), dcovar=d, dcovar_p=pv, mes=mes,
                         beta=beta, down_beta=dbeta, ltd=ltd, var5=var5, es5=es5, n=n))
tr = pd.DataFrame(rows); tr.to_csv(OUT / "firm_tail_measures.csv", index=False)

# (d) EVT peaks-over-threshold
def pot(x, u_q=0.90, p=0.99):
    L = -x.dropna()
    u = L.quantile(u_q); exc = L[L > u] - u
    xi, loc, beta = genpareto.fit(exc, floc=0)
    n, nu = len(L), len(exc)
    var = u + beta/xi * (((n/nu)*(1-p))**(-xi) - 1)
    es = (var + beta - xi*u) / (1 - xi)
    return dict(xi=xi, beta=beta, u=u, n_exc=nu, VaR99=var*100, ES99=es*100)
evt = []
for p in PERIODS:
    for nm, s in [("AI_IDX", ai), ("CTL_IDX", ctl), ("SPY", r["SPY"])]:
        evt.append(dict(period=p, series=nm, **pot(period_slice(s, p))))
pd.DataFrame(evt).to_csv(OUT / "evt_pot.csv", index=False)

print(pd.DataFrame(static).T[["total"]])
print(pd.DataFrame(evt).round(3))
print(tr.groupby(["period","fld"])[["dcovar","mes","down_beta","ltd","es5"]].mean().round(3))
