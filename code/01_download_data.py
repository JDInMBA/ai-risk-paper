"""
01_download_data.py
Downloads daily adjusted prices (Yahoo Finance via yfinance) and credit/macro
series (FRED) used in the study. Output: data/prices.csv, data/fred.csv
"""
import io, time, requests, pandas as pd, yfinance as yf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"; DATA.mkdir(exist_ok=True)
START, END = "2021-01-01", "2026-11-06"

UNIVERSE = {
    # group: tickers
    "semis":        ["NVDA","AMD","AVGO","TSM","MU","MRVL","INTC","QCOM","ARM","ASML","AMAT","LRCX","CBRS"],
    "hyperscalers": ["MSFT","AMZN","GOOGL","META","ORCL"],
    "neocloud_dc":  ["CRWV","NBIS","APLD","IREN","CORZ","CIFR","WULF","EQIX","DLR"],
    "hardware":     ["DELL","SMCI","ANET","HPE","CSCO","VRT","WDC","STX"],
    "power":        ["CEG","VST","GEV","ETN","TLN"],
    "ai_software":  ["PLTR","SNOW","IBM","NOW","CRM","ADBE"],
    "investor":     ["SFTBY"],
    "megacap_other":["AAPL","TSLA"],
    "control":      ["KO","PG","JNJ","WMT","PEP","MCD","BRK-B","JPM","XOM","UNH","HD","LLY","COST","V","MA","CVX","MRK","ABBV","T","VZ"],
    "etf":          ["SPY","QQQ","RSP","XLP","XLU","XLV","SMH","HYG","LQD","TLT","^VIX"],
}

import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FLD
_firms = {t for k, v in UNIVERSE.items() if k != "etf" for t in v}
assert _firms == set(FLD), f"universe and FLD disagree: {_firms ^ set(FLD)}"   # keep the two in step

def get_prices():
    tickers = sorted({t for v in UNIVERSE.values() for t in v})
    frames = []
    for i in range(0, len(tickers), 15):
        chunk = tickers[i:i+15]
        for attempt in range(4):
            try:
                d = yf.download(chunk, start=START, end=END, auto_adjust=True,
                                progress=False, threads=False)["Close"]
                frames.append(d); break
            except Exception as e:
                print("retry", chunk, e); time.sleep(5*(attempt+1))
        time.sleep(2)
    px = pd.concat(frames, axis=1)
    px = px.loc[:, ~px.columns.duplicated()]
    px.index.name = "date"
    px.to_csv(DATA / "prices.csv")
    meta = pd.DataFrame([(t, g) for g, v in UNIVERSE.items() for t in v], columns=["ticker","group"])
    meta.to_csv(DATA / "universe.csv", index=False)
    print("prices", px.shape, px.index.min(), px.index.max())
    print("missing columns:", [t for t in tickers if t not in px.columns or px[t].isna().all()])

FRED = {"BAMLH0A0HYM2":"hy_oas","BAMLC0A0CM":"ig_oas","BAMLC0A4CBBB":"bbb_oas",
        "VIXCLS":"vix","DGS10":"ust10","DGS2":"ust2","NFCI":"nfci"}

def get_fred():
    """FRED csv endpoint; fetched with curl over HTTP/1.1 (more robust behind proxies).
    Note: ICE BofA OAS series on FRED are limited to roughly the last three years."""
    import subprocess
    out = []
    for sid, name in FRED.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={START}"
        txt = ""
        for attempt in range(6):
            res = subprocess.run(["curl", "-sS", "--http1.1", "-m", "60", url], capture_output=True, text=True)
            if res.returncode == 0 and res.stdout.startswith("observation_date"):
                txt = res.stdout; break
            time.sleep(3 * (attempt + 1))
        s = pd.read_csv(io.StringIO(txt), na_values=".")
        s.columns = ["date", name]; s["date"] = pd.to_datetime(s["date"])
        out.append(s.set_index("date"))
    f = pd.concat(out, axis=1).sort_index()
    f.to_csv(DATA / "fred.csv")
    print("fred", f.shape, f.dropna(how="all").index.min(), f.dropna(how="all").index.max())

if __name__ == "__main__":
    import sys
    if "--fred-only" not in sys.argv: get_prices()
    get_fred()
