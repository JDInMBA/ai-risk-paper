"""Shared helpers: data loading, groups, periods, exposure scores."""
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA, FIG, OUT = ROOT / "data", ROOT / "figures", ROOT / "results"
FIG.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)

CORE_AI = ["NVDA","AMD","AVGO","TSM","MU","MSFT","AMZN","GOOGL","META","ORCL","DELL","SMCI","ANET","VRT","CEG"]
CONTROL = ["KO","PG","JNJ","WMT","PEP","MCD","BRK-B","JPM","XOM","UNH","HD","LLY","COST","V","MA","CVX","MRK","ABBV","T","VZ"]

PERIODS = {  # sub-samples used throughout
    "P0 pre-ChatGPT":       ("2021-01-04", "2022-11-29"),
    "P1 AI boom":           ("2022-11-30", "2025-09-09"),
    "P2 backlog era":       ("2025-09-10", "2026-11-05"),
}

# FLD score, hand-coded (Table A1)
# 3 = direct lab contracts
# 2 = material but diversified
# 1 = AI capex cycle
# 0 = control firms
FLD = {
 "ORCL":3,"CRWV":3,"CBRS":3,"AMD":3,"SFTBY":3,
 "MSFT":2,"GOOGL":2,"AMZN":2,"AVGO":2,"NVDA":2,
 "TSM":1,"MU":1,"MRVL":1,"INTC":1,"QCOM":1,"ARM":1,"ASML":1,"AMAT":1,"LRCX":1,"META":1,
 "NBIS":1,"APLD":1,"IREN":1,"CORZ":1,"CIFR":1,"WULF":1,"EQIX":1,"DLR":1,
 "DELL":1,"SMCI":1,"ANET":1,"HPE":1,"CSCO":1,"VRT":1,"WDC":1,"STX":1,
 "CEG":1,"VST":1,"GEV":1,"ETN":1,"TLN":1,"PLTR":1,"SNOW":1,"IBM":1,"NOW":1,"CRM":1,"ADBE":1,
 "AAPL":1,"TSLA":1,
}
FLD.update({t:0 for t in CONTROL})

def load_prices():
    px = pd.read_csv(DATA / "prices.csv", index_col=0, parse_dates=True).sort_index()
    return px

def log_returns(px):
    """Daily log returns on NYSE trading days only (rows where SPY did not trade are dropped)."""
    px = px[px["SPY"].notna()]
    return np.log(px).diff()

def load_fred():
    f = pd.read_csv(DATA / "fred.csv", index_col=0, parse_dates=True).sort_index()
    return f

def ai_index_returns(r, names=CORE_AI):
    return r[names].mean(axis=1, skipna=True)

def period_slice(s, p):
    a, b = PERIODS[p]; return s.loc[a:b]
