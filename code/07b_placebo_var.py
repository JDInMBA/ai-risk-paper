"""07b_placebo_var.py
also fail Kupiec at 95%? Reuses 07_var_backtest.py's model code verbatim."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy.stats import chi2, norm, t as tdist
from arch import arch_model
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

r = pd.read_csv(OUT/"returns.csv", index_col=0, parse_dates=True)
px = load_prices()
conn = pd.read_csv(OUT/"dy_total_connectedness.csv", index_col=0, parse_dates=True).iloc[:,0]

FLD3_FIXED = ["ORCL","AMD","SFTBY"]   # FLD-3 firms trading before P1
assert set(FLD3_FIXED) <= {t for t, f in FLD.items() if f == 3}

def p1_risk(x, mkt):
    """Pre-period beta on the AI index and 5% expected shortfall, on the overlap."""
    d = pd.concat([x, mkt], axis=1).dropna()
    xi, mi = d.iloc[:,0], d.iloc[:,1]
    return np.cov(xi, mi)[0,1] / np.var(mi), -xi[xi <= xi.quantile(0.05)].mean()

def choose_placebos(k=3):
    """Pick the placebo baskets by rule rather than by hand, so the choice reproduces.

    The pool is the AI firms with no material frontier-lab contract (FLD 1) that
    traded through P1. The matched basket takes the k firms closest to the fixed
    FLD-3 basket on P1 beta and P1 expected shortfall, each scaled by the target
    value so the two dimensions count equally. The high-beta basket takes the k
    firms with the largest P1 beta."""
    a, b = PERIODS["P1 AI boom"]; p1 = r.loc[a:b]
    tb, te = p1_risk(p1[FLD3_FIXED].mean(axis=1), p1["AI_IDX"])
    pool = [t for t, f in FLD.items() if f == 1 and t in p1.columns and p1[t].notna().sum() > 400]
    d = pd.DataFrame([(t, *p1_risk(p1[t], p1["AI_IDX"])) for t in pool],
                     columns=["ticker","beta","es5"])
    d["dist"] = (d.beta - tb).abs()/tb + (d.es5 - te).abs()/te
    matched = list(d.nsmallest(k, "dist").ticker)
    highbeta = list(d.nlargest(k, "beta").ticker)
    d.sort_values("dist").to_csv(OUT / "placebo_selection.csv", index=False)
    print(f"target P1 beta {tb:.3f} ES {te:.3f}; matched {matched}; high-beta {highbeta}", flush=True)
    return matched, highbeta, d

MATCHED, HIGHBETA, POOL = choose_placebos()
ports = {
 "FLD3_fixed (paper)":                      r[FLD3_FIXED].mean(axis=1),
 f"PLACEBO matched ({','.join(MATCHED)})":  r[MATCHED].mean(axis=1),
 f"PLACEBO high-beta ({','.join(HIGHBETA)})": r[HIGHBETA].mean(axis=1),
 "CTL_IDX (paper)":                         r["CTL_IDX"],
}
TEST0="2024-01-02"
def state(x):
    S=pd.DataFrame(index=r.index); S["rv22"]=x.rolling(22).std(); S["absr"]=x.abs()
    S["vix"]=px["^VIX"].reindex(r.index)/100
    cz=(conn-conn.rolling(250,min_periods=120).mean())/conn.rolling(250,min_periods=120).std()
    S["conn_z"]=cz.reindex(r.index).fillna(0); return S
def kupiec(I,p):
    n,x=len(I),I.sum(); ph=x/n
    if x==0: return -2*n*np.log(1-p), 1-chi2.cdf(-2*n*np.log(1-p),1)
    lr=-2*((n-x)*np.log(1-p)+x*np.log(p)-(n-x)*np.log(1-ph)-x*np.log(ph))
    return lr, 1-chi2.cdf(lr,1)

out=[]
for pname,x in ports.items():
    x=x.dropna(); S=state(x).reindex(x.index); test_idx=x.loc[TEST0:].index
    F={f"{m}_{a}":pd.Series(np.nan,index=test_idx) for m in ["HS","EWMA","GARCH-t","FHS","QR-state"] for a in ["99","95"]}
    lam=0.94; ew=x.pow(2).ewm(alpha=1-lam,adjust=False).mean()
    for a,al in [("99",.01),("95",.05)]:
        F[f"HS_{a}"]=x.rolling(250).quantile(al).shift(1).reindex(test_idx)
        F[f"EWMA_{a}"]=(norm.ppf(al)*np.sqrt(ew)).shift(1).reindex(test_idx)
    pos0=x.index.get_loc(test_idx[0])
    for k in range(pos0,len(x),21):
        win=x.iloc[max(0,k-1000):k]*100
        am=arch_model(win,mean="Constant",vol="GARCH",p=1,q=1,dist="t").fit(disp="off")
        mu,om,al1,be=am.params["mu"],am.params["omega"],am.params["alpha[1]"],am.params["beta[1]"]; nu=am.params["nu"]
        z=(am.resid/am.conditional_volatility).dropna().values
        s2=am.conditional_volatility.iloc[-1]**2; eps=am.resid.iloc[-1]
        for j in range(k,min(k+21,len(x))):
            s2=om+al1*eps**2+be*s2; sig=np.sqrt(s2); d=x.index[j]
            for a,al in [("99",.01),("95",.05)]:
                qt=tdist.ppf(al,nu)*np.sqrt((nu-2)/nu)
                F[f"GARCH-t_{a}"][d]=(mu+sig*qt)/100
                F[f"FHS_{a}"][d]=(mu+sig*np.quantile(z,al))/100
            eps=x.iloc[j]*100-mu
    Z=S.shift(1)
    for q in pd.period_range("2024Q1","2026Q4",freq="Q"):
        te=test_idx[(test_idx>=q.start_time)&(test_idx<=q.end_time)]
        if len(te)==0: continue
        trn=pd.concat([x,Z],axis=1).loc[:q.start_time-pd.Timedelta(days=1)].dropna()
        X=sm.add_constant(trn.iloc[:,1:]); Xte=sm.add_constant(Z.loc[te],has_constant="add")
        for a,al in [("99",.01),("95",.05)]:
            m=sm.QuantReg(trn.iloc[:,0],X).fit(q=al,max_iter=5000)
            F[f"QR-state_{a}"].loc[te]=m.predict(Xte.dropna()).reindex(te)
    y=x.loc[test_idx]
    for mn in ["HS","EWMA","GARCH-t","FHS","QR-state"]:
        for a,al in [("95",.05),("99",.01)]:
            v=F[f"{mn}_{a}"]; ok=v.notna(); I=(y[ok]<v[ok]).astype(int)
            _,pk=kupiec(I,al)
            out.append(dict(portfolio=pname,model=mn,level=a,rate=I.mean(),kupiec_p=pk,fail=pk<0.05))
    print("done",pname,flush=True)
R=pd.DataFrame(out)
R.to_csv(OUT / "placebo_var_backtests.csv", index=False)
for a in ["95","99"]:
    print(f"\n===== {a}% VaR: exceedance rate (Kupiec p) — * = rejected =====")
    t=R[R.level==a].pivot_table(index="portfolio",columns="model",values="rate")
    pv=R[R.level==a].pivot_table(index="portfolio",columns="model",values="kupiec_p")
    disp=t.copy().astype(object)
    for i in t.index:
        for c in t.columns:
            disp.loc[i,c]=f"{t.loc[i,c]*100:5.2f}% ({pv.loc[i,c]:.3f}){'*' if pv.loc[i,c]<0.05 else ' '}"
    print(disp.reindex(list(ports)).to_string())
    print("  models failed:", R[(R.level==a)].groupby("portfolio").fail.sum().reindex(list(ports)).to_dict())

# FLD-3 against random AI baskets
NDRAW = 500
rng = np.random.default_rng(20261118)
pool = sorted(POOL.ticker)
rows = []
def hs_ewma_rates(x):
    x = x.dropna(); idx = x.loc[TEST0:].index
    ew = x.pow(2).ewm(alpha=0.06, adjust=False).mean()
    hs = x.rolling(250).quantile(0.05).shift(1).reindex(idx)
    ewv = (norm.ppf(0.05) * np.sqrt(ew)).shift(1).reindex(idx)
    y = x.loc[idx]
    return float((y < hs).mean()), float((y < ewv).mean())

tgt_hs, tgt_ew = hs_ewma_rates(r[FLD3_FIXED].mean(axis=1))
for i in range(NDRAW):
    pick = list(rng.choice(pool, size=3, replace=False))
    h, e = hs_ewma_rates(r[pick].mean(axis=1))
    rows.append(dict(draw=i, tickers="|".join(sorted(pick)), hs_rate=h, ewma_rate=e))
RB = pd.DataFrame(rows)
RB.to_csv(OUT / "placebo_random_baskets.csv", index=False)
pct = lambda v, col: float((RB[col] < v).mean() * 100)
summary = pd.DataFrame([
    dict(model="HS 95%",   fld3_rate=tgt_hs, random_median=RB.hs_rate.median(),
         random_p95=RB.hs_rate.quantile(0.95), percentile_of_fld3=pct(tgt_hs, "hs_rate")),
    dict(model="EWMA 95%", fld3_rate=tgt_ew, random_median=RB.ewma_rate.median(),
         random_p95=RB.ewma_rate.quantile(0.95), percentile_of_fld3=pct(tgt_ew, "ewma_rate")),
])
summary.to_csv(OUT / "placebo_random_summary.csv", index=False)
print(f"\n===== FLD-3 basket against {NDRAW} random FLD-1 baskets (95% VaR breach rate) =====")
print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
