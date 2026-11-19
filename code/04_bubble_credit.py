"""
04_bubble_credit.py
 (a) PSY (Phillips, Shi & Yu 2015) GSADF / BSADF date-stamping on weekly log prices.
     DF regressions (lag 0) computed in closed form from cumulative sums; Monte-Carlo
     critical values under a unit-root null with drift ~ T^-1 (PSY recommended null).
 (b) Merton (1974) distance-to-default for levered AI-infrastructure firms, quarterly,
     using SEC XBRL debt (short-term + 0.5 x long-term, KMV default point) and
     252-day equity volatility; solved by the iterative KMV procedure.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import norm
from common import *

px = load_prices()

def bsadf_seq(y, r0):
    y = np.asarray(y, float); T = len(y) - 1
    dy = np.diff(y); yl = y[:-1]
    # cumulative sums for windows
    c = lambda v: np.concatenate([[0], np.cumsum(v)])
    S1, Sx, Sy, Sxx, Sxy, Syy = c(np.ones(T)), c(yl), c(dy), c(yl*yl), c(yl*dy), c(dy*dy)
    w0 = int(np.floor(r0 * T))
    bsadf = np.full(T, np.nan)
    for b in range(w0, T+1):
        a = np.arange(0, b - w0 + 1)
        n = S1[b]-S1[a]; sx = Sx[b]-Sx[a]; sy = Sy[b]-Sy[a]
        sxx = Sxx[b]-Sxx[a]; sxy = Sxy[b]-Sxy[a]; syy = Syy[b]-Syy[a]
        den = n*sxx - sx*sx
        beta = (n*sxy - sx*sy) / den
        alpha = (sy - beta*sx) / n
        rss = syy - alpha*sy - beta*sxy
        s2 = rss / (n - 2)
        se = np.sqrt(s2 * n / den)
        bsadf[b-1] = np.nanmax(beta / se)
    return bsadf

def mc_cv(T, r0, reps=499, seed=7):
    rng = np.random.default_rng(seed)
    sups, seqs = [], []
    for _ in range(reps):
        e = rng.standard_normal(T+1)
        y = np.cumsum(e) + np.arange(T+1) * T**-1
        s = bsadf_seq(y, r0); sups.append(np.nanmax(s)); seqs.append(s)
    seqs = np.array(seqs)
    return np.quantile(sups, [.90,.95,.99]), np.nanquantile(seqs, .95, axis=0)

wk = px.resample("W-FRI").last()
ai_w = np.exp(np.log(px[CORE_AI]).diff().mean(axis=1).fillna(0).cumsum()).resample("W-FRI").last()
series = {"AI_IDX": ai_w, "SPY": wk["SPY"], "NVDA": wk["NVDA"], "ORCL": wk["ORCL"], "AMD": wk["AMD"],
          "AVGO": wk["AVGO"], "MU": wk["MU"], "CRWV": wk["CRWV"], "AI_rel_SPY": ai_w / wk["SPY"]}
res, stamps = [], {}
for nm, s in series.items():
    s = np.log(s.dropna()); T = len(s) - 1
    r0 = 0.01 + 1.8 / np.sqrt(T)
    b = bsadf_seq(s.values, r0)
    cv, cvseq = mc_cv(T, r0)
    bs = pd.Series(b, index=s.index[1:])
    stamps[nm] = pd.DataFrame({"bsadf": bs, "cv95": cvseq}, index=s.index[1:])
    flag = (bs > cvseq)
    # episodes at least log(T) weeks
    run, eps, start = 0, [], None
    for d, f in flag.items():
        if f: run += 1; start = start or d
        else:
            if run >= np.log(T): eps.append((start.date(), prev.date()))
            run, start = 0, None
        prev = d
    if run >= np.log(T): eps.append((start.date(), prev.date()))
    res.append(dict(series=nm, T=T, gsadf=np.nanmax(b), cv90=cv[0], cv95=cv[1], cv99=cv[2],
                    last_bsadf=bs.iloc[-1], last_cv95=cvseq[-1], episodes="; ".join(f"{a}→{c}" for a, c in eps)))
    print(nm, "done")
pd.DataFrame(res).to_csv(OUT / "gsadf_results.csv", index=False)
pd.concat(stamps, axis=1).to_csv(OUT / "bsadf_sequences.csv")
print(pd.DataFrame(res).drop(columns="episodes").round(2)); print(pd.DataFrame(res)[["series","episodes"]].to_string())

# (b) Merton distance-to-default
facts = pd.read_csv(DATA / "sec_facts_long.csv", parse_dates=["end"])
facts = facts[facts.unit == "USD"]
def latest_series(t, concepts, instant=True):
    d = facts[(facts.ticker == t) & (facts.concept.isin(concepts))]
    if not instant: d = d[d.start.isna()]
    d = d.sort_values(["end","filed"]).drop_duplicates(["end","concept"], keep="last")
    return d.pivot(index="end", columns="concept", values="val")
shares = facts[(facts.concept=="WeightedAverageNumberOfDilutedSharesOutstanding")]
sh_all = pd.read_csv(DATA / "sec_facts_long.csv", parse_dates=["end"])
sh_all = sh_all[sh_all.concept=="WeightedAverageNumberOfDilutedSharesOutstanding"]

def merton_dd(E, sE, D, r=0.04, T=1.0):
    V, sV = E + D, sE * E / (E + D)
    for _ in range(200):
        d1 = (np.log(V/D) + (r + .5*sV**2)*T) / (sV*np.sqrt(T)); d2 = d1 - sV*np.sqrt(T)
        Vn = (E + D*np.exp(-r*T)*norm.cdf(d2)) / norm.cdf(d1)
        sVn = sE * E / (Vn * norm.cdf(d1))
        if abs(Vn-V) < 1e-6*V and abs(sVn-sV) < 1e-6: break
        V, sV = Vn, sVn
    dd = (np.log(V/D) + (r - .5*sV**2)*T) / (sV*np.sqrt(T))
    return dd, norm.cdf(-dd)

DD_FIRMS = ["ORCL","CRWV","MSFT","AMZN","GOOGL","META","AMD","NVDA","AVGO","APLD","IREN","CORZ","CIFR","WULF","DLR","EQIX","SMCI","DELL","VST","CEG"]
rv = np.log(px).diff().rolling(252, min_periods=120).std() * np.sqrt(252)
ust = load_fred()["ust2"].ffill() / 100
out = []
for t in DD_FIRMS:
    bs = latest_series(t, ["LongTermDebtNoncurrent","LongTermDebt","LongTermDebtCurrent","DebtCurrent","NotesPayableCurrent","LongTermNotesAndLoans","DebtLongtermAndShorttermCombinedAmount","DebtInstrumentCarryingAmount","LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths","FinanceLeaseLiability","OperatingLeaseLiability"])
    sh = sh_all[sh_all.ticker==t].sort_values(["end","filed"]).drop_duplicates("end", keep="last").set_index("end")["val"]
    for end, row in bs.iterrows():
        if end < pd.Timestamp("2021-06-30"): continue
        g = lambda k: row.get(k, np.nan)
        cur = next((g(k) for k in ["DebtCurrent","NotesPayableCurrent","LongTermDebtCurrent","LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths"] if pd.notna(g(k))), 0.0)
        lt = next((g(k) for k in ["LongTermDebtNoncurrent","LongTermNotesAndLoans"] if pd.notna(g(k))), np.nan)
        if pd.isna(lt):
            tot = next((g(k) for k in ["DebtLongtermAndShorttermCombinedAmount","LongTermDebt","DebtInstrumentCarryingAmount"] if pd.notna(g(k))), np.nan)
            lt = tot - cur if pd.notna(tot) else 0.0
        st_debt = cur
        lease = np.nansum([g("FinanceLeaseLiability")])
        D = st_debt + 0.5 * (lt + lease)
        if not np.isfinite(D) or D <= 0: continue
        s_ = sh[sh.index <= end]
        if s_.empty or t not in px: continue
        p_ = px[t].loc[:end].dropna()
        if p_.empty or (end - p_.index[-1]).days > 10: continue
        E = p_.iloc[-1] * s_.iloc[-1]; sE = rv[t].loc[:end].dropna()
        if sE.empty: continue
        rf = ust.loc[:end].dropna().iloc[-1]
        dd, pd_ = merton_dd(E, sE.iloc[-1], D, r=rf)
        out.append(dict(ticker=t, date=end.date(), equity_bn=E/1e9, debt_point_bn=D/1e9, sigmaE=sE.iloc[-1], DD=dd, PD=pd_, lev=D/(E+D)))
ddf = pd.DataFrame(out).sort_values(["ticker","date"]); ddf.to_csv(OUT / "merton_dd.csv", index=False)
print(ddf.groupby("ticker").tail(2).round(3).to_string())
