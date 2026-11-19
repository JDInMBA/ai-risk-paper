"""
05c_tail_beta_panel.py
Panel test of 'excess tail beta'. For firm i on day t:
  r_it = a_i + b_i * AI_t + sum_g d_g * AI_t * Tail_t * 1{FLD_i = g} + e_it
where b_i is a firm-specific normal-day beta to the AI index (excluding firm i), Tail_t = 1 when
the AI index is in its worst 5% of days within the sub-period. d_g measures how much MORE a group
co-moves with the AI complex on tail days than its own normal-day beta implies.
Estimated separately for P1 and P2; standard errors clustered by date. Also a difference
(P2 - P1) for groups present in both eras.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from common import *
r = pd.read_csv(OUT / "returns.csv", index_col=0, parse_dates=True)
firms = [c for c in r.columns if c in FLD]
out = []
for p in ["P1 AI boom", "P2 backlog era"]:
    rp = period_slice(r, p); rows = []
    for c in firms:
        if rp[c].notna().sum() < 100: continue
        sysr = rp[[n for n in CORE_AI if n != c]].mean(axis=1)
        tail = (sysr <= sysr.quantile(0.05)).astype(int)
        rows.append(pd.DataFrame({"date": rp.index, "ticker": c, "fld": FLD[c], "r": rp[c]*100, "ai": sysr*100, "tl": tail}).dropna())
    D = pd.concat(rows)
    for g in range(4): D[f"tb{g}"] = D["ai"] * D["tl"] * (D["fld"] == g)
    m = smf.ols("r ~ C(ticker) + C(ticker):ai + tb0 + tb1 + tb2 + tb3 - 1", D).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(D.date)[0]})
    tst = m.t_test("tb3 - tb1 = 0")
    for g in range(4):
        out.append(dict(period=p, group=f"FLD{g}", excess_tail_beta=m.params[f"tb{g}"], t=m.tvalues[f"tb{g}"], p=m.pvalues[f"tb{g}"],
                        n_firms=D[D.fld==g].ticker.nunique(), N=len(D)))
    out.append(dict(period=p, group="FLD3 - FLD1", excess_tail_beta=float(np.ravel(tst.effect)[0]), t=float(np.ravel(tst.tvalue)[0]), p=float(np.ravel(tst.pvalue)[0])))
    # leave-one-out, FLD3 firms
    if p.startswith("P2"):
        for c in D[D.fld==3].ticker.unique():
            mm = smf.ols("r ~ C(ticker) + C(ticker):ai + tb0 + tb1 + tb2 + tb3 - 1", D[D.ticker!=c]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(D[D.ticker!=c].date)[0]})
            out.append(dict(period=p, group=f"FLD3 (drop {c})", excess_tail_beta=mm.params.tb3, t=mm.tvalues.tb3, p=mm.pvalues.tb3))
R = pd.DataFrame(out); R.to_csv(OUT / "tail_beta_panel.csv", index=False); print(R.round(3).to_string())
