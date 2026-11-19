"""09_ipo_cohort.py - post-IPO risk profile of AI-infrastructure listings (ARM, CRWV, CBRS)."""
import numpy as np, pandas as pd
from common import *
px = load_prices(); r = pd.read_csv(OUT / "returns.csv", index_col=0, parse_dates=True)
IPO = {"ARM": ("2023-09-14", 51.0), "CRWV": ("2025-03-28", 40.0), "CBRS": ("2026-05-14", 185.0)}
rows = []
for t, (d, p0) in IPO.items():
    s = px[t].dropna().loc[d:]; x = r[t].dropna().loc[s.index[1]:]
    ai = r["AI_IDX"].reindex(x.index)
    peak = s.cummax(); dd = (s / peak - 1).min()
    rows.append(dict(ticker=t, ipo_date=d, offer=p0, first_close=s.iloc[0], first_day_ret=(s.iloc[0]/p0-1)*100,
                     peak_vs_offer=(s.max()/p0-1)*100, last=s.iloc[-1], ret_offer_to_now=(s.iloc[-1]/p0-1)*100,
                     ret_firstclose_to_now=(s.iloc[-1]/s.iloc[0]-1)*100, max_drawdown=dd*100,
                     ann_vol=x.std()*np.sqrt(252)*100, beta_ai=np.cov(x, ai)[0,1]/ai.var(), days=len(s)))
T = pd.DataFrame(rows); T.to_csv(OUT / "ipo_cohort.csv", index=False); print(T.round(2).to_string())
