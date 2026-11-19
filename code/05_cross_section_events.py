"""
05_cross_section_events.py
 (a) Cross-sectional tests: does frontier-lab dependence (FLD) predict tail outcomes in the
     backlog era (P2) after controlling for pre-determined risk (P1 beta, P1 ES, size, leverage)?
 (b) Event study around frontier-lab news (market model, SPY), CAR[-1,+1] by FLD group.
 (c) Stress-day returns (largest AI-complex down days in P2).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from common import *

r = pd.read_csv(OUT / "returns.csv", index_col=0, parse_dates=True)
px = load_prices()
tm = pd.read_csv(OUT / "firm_tail_measures.csv")
facts = pd.read_csv(DATA / "sec_facts_long.csv", parse_dates=["end"])

# firm characteristics at P1 end
def last_val(t, concept, before="2025-09-09"):
    d = facts[(facts.ticker==t)&(facts.concept==concept)&(facts.end<=before)&(facts.start.isna())&(facts.unit=="USD")]
    return d.sort_values(["end","filed"]).val.iloc[-1] if len(d) else np.nan
sh = facts[facts.concept=="WeightedAverageNumberOfDilutedSharesOutstanding"]
chars = []
for t in tm.ticker.unique():
    A = last_val(t,"Assets"); L = last_val(t,"Liabilities")
    if pd.isna(L) and pd.notna(A): L = A - last_val(t,"StockholdersEquity")
    lev = L / A if pd.notna(A) and pd.notna(L) else np.nan
    s = sh[(sh.ticker==t)&(sh.end<="2025-09-09")].sort_values(["end","filed"]).val
    if not len(s): s = sh[(sh.ticker==t)].sort_values(["end","filed"]).val  # firms listed after P1 (CRWV, CBRS)
    mcap = px[t].loc[:"2025-09-09"].dropna().iloc[-1]*s.iloc[-1]/1e9 if len(s) and px[t].loc[:"2025-09-09"].notna().any() else np.nan
    chars.append(dict(ticker=t, lev=lev, mcap_bn=mcap))
chars = pd.DataFrame(chars)

p1 = tm[tm.period=="P1 AI boom"].set_index("ticker")[["beta","es5","down_beta","mes"]].add_suffix("_P1")
p2 = tm[tm.period=="P2 backlog era"].set_index("ticker")
cs = p2.join(p1, how="left").join(chars.set_index("ticker"))
# P2 drawdown and return
r2 = period_slice(r, "P2 backlog era")
cum = r2.cumsum()
cs["maxdd_P2"] = [((np.exp(cum[t]) / np.exp(cum[t]).cummax()) - 1).min()*100 if t in cum else np.nan for t in cs.index]
cs["ret_P2"] = [(np.exp(r2[t].sum())-1)*100 if t in r2 else np.nan for t in cs.index]
cs["fld3"] = (cs.fld==3).astype(int); cs["fld2"] = (cs.fld==2).astype(int); cs["fld1"] = (cs.fld==1).astype(int)
cs["logmcap"] = np.log(cs.mcap_bn)
cs["vol_P2"] = [r2[t].std()*np.sqrt(252)*100 if t in r2 else np.nan for t in cs.index]
cs.to_csv(OUT / "cross_section_P2.csv")

specs = {
 "MES (P2)":        "mes ~ fld3 + fld2 + fld1",
 "MES (P2) ctrl":   "mes ~ fld3 + fld2 + fld1 + beta_P1 + logmcap + lev",
 "ES5 (P2) ctrl":   "es5 ~ fld3 + fld2 + fld1 + es5_P1 + logmcap + lev",
 "DownBeta ctrl":   "down_beta ~ fld3 + fld2 + fld1 + down_beta_P1 + logmcap + lev",
 "MaxDD ctrl":      "maxdd_P2 ~ fld3 + fld2 + fld1 + beta_P1 + logmcap + lev",
 "LTD ctrl":        "ltd ~ fld3 + fld2 + fld1 + beta_P1 + logmcap + lev",
 "MES contemp":     "mes ~ fld3 + fld2 + fld1 + beta + vol_P2 + logmcap",
 "ES5 contemp":     "es5 ~ fld3 + fld2 + fld1 + vol_P2 + logmcap",
 "LTD contemp":     "ltd ~ fld3 + fld2 + fld1 + vol_P2 + logmcap",
 "MaxDD contemp":   "maxdd_P2 ~ fld3 + fld2 + fld1 + beta + vol_P2 + logmcap",
}
tabs = []
for nm, f in specs.items():
    d = cs.dropna(subset=[c for c in ["beta_P1","logmcap","lev","es5_P1","down_beta_P1","vol_P2","beta"] if c in f])
    m = smf.ols(f, data=d).fit(cov_type="HC3")
    row = {"model": nm, "N": int(m.nobs), "R2": m.rsquared}
    for k in m.params.index:
        row[k] = m.params[k]; row[k+"_t"] = m.tvalues[k]
    # FLD3 against FLD1
    w = m.t_test("fld3 - fld1 = 0"); row["fld3_minus_fld1"] = float(np.ravel(w.effect)[0]); row["fld3_minus_fld1_p"] = float(np.ravel(w.pvalue)[0])
    tabs.append(row)
reg = pd.DataFrame(tabs); reg.to_csv(OUT / "cs_regressions.csv", index=False)
print(reg[["model","N","R2","fld3","fld3_t","fld2","fld2_t","fld1","fld1_t","fld3_minus_fld1","fld3_minus_fld1_p"]].round(3).to_string())

# (b) event study
EVENTS = [  # date (first trading day of market reaction), label, sign expected for lab-linked suppliers
 ("2025-01-22","Stargate JV announced (OpenAI/SoftBank/Oracle)","+"),
 ("2025-01-27","DeepSeek R1 efficiency shock","-"),
 ("2025-09-10","Oracle RPO jumps to $455bn (OpenAI contract)","+"),
 ("2025-09-22","Nvidia-OpenAI $100bn letter of intent","+"),
 ("2025-10-06","AMD-OpenAI 6 GW supply + warrants","+"),
 ("2025-10-13","Broadcom-OpenAI 10 GW custom accelerators","+"),
 ("2025-10-28","OpenAI recapitalisation; Microsoft stake","+"),
 ("2025-11-03","OpenAI-AWS $38bn agreement","+"),
 ("2025-11-18","Microsoft/Nvidia investment in Anthropic","+"),
 ("2026-05-14","Cerebras IPO prices above range","+"),
 ("2026-05-28","Anthropic Series H at $965bn","+"),
 ("2026-06-08","OpenAI confidential IPO filing reported","+"),
 ("2026-10-08","OpenAI run-rate revenue below reports","-"),
 ("2026-10-26","Anthropic public S-1","+"),
 ("2026-11-03","Anthropic S-1/A: $895bn commitments","-"),
]
names = [c for c in r.columns if c in FLD]
idx = r.index
ar_rows = []
for d, lab, sgn in EVENTS:
    t0 = idx.searchsorted(pd.Timestamp(d))
    est = r.iloc[max(0,t0-280):t0-30]
    for c in names:
        e = est[[c,"SPY"]].dropna()
        if len(e) < 120: continue
        b, a = np.polyfit(e["SPY"], e[c], 1)
        win = r.iloc[t0-1:t0+2][[c,"SPY"]]
        if win[c].isna().any(): continue
        ar = win[c] - (a + b*win["SPY"])
        sig = (e[c] - (a + b*e["SPY"])).std()
        ar_rows.append(dict(date=d, event=lab, exp_sign=sgn, ticker=c, fld=FLD[c], CAR=ar.sum()*100, AR0=ar.iloc[1]*100, sd=sig*100))
ev = pd.DataFrame(ar_rows); ev.to_csv(OUT / "event_study_firm.csv", index=False)
g = ev.groupby(["date","event","fld"]).CAR.mean().unstack()
g.to_csv(OUT / "event_study_by_fld.csv"); print(g.round(2).to_string())
# pooled absolute-CAR test
ev["absCAR"] = ev.CAR.abs(); ev["scaledCAR"] = ev.CAR / (ev.sd*np.sqrt(3))
ev["signedCAR"] = np.where(ev.exp_sign=="+", ev.CAR, -ev.CAR)
m1 = smf.ols("signedCAR ~ C(fld)", data=ev).fit(cov_type="cluster", cov_kwds={"groups": ev.date})
m2 = smf.ols("absCAR ~ C(fld) + sd", data=ev).fit(cov_type="cluster", cov_kwds={"groups": ev.date})
with open(OUT / "event_pooled.txt","w") as f: f.write(m1.summary().as_text()+"\n\n"+m2.summary().as_text())
print(m1.params.round(3), m1.pvalues.round(3)); print(m2.params.round(3), m2.pvalues.round(3))

# (c) P2 stress days
ai2 = period_slice(r["AI_IDX"], "P2 backlog era")
worst = ai2.nsmallest(10)
sd_rows = []
for d in worst.index:
    row = r.loc[d]
    for k in [0,1,2,3]:
        sel = [c for c in names if FLD[c]==k and pd.notna(row.get(c))]
        sd_rows.append(dict(date=d.date(), ai_idx=row["AI_IDX"]*100, fld=k, mean_ret=row[sel].mean()*100, n=len(sel)))
sdd = pd.DataFrame(sd_rows).pivot_table(index=["date","ai_idx"], columns="fld", values="mean_ret")
sdd.to_csv(OUT / "stress_days_P2.csv"); print(sdd.round(2))
