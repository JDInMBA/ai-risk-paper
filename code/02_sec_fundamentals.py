"""
02_sec_fundamentals.py
Pulls quarterly/annual XBRL facts from SEC EDGAR 'companyfacts' for AI-complex
firms: remaining performance obligations (RPO), capex, operating cash flow,
revenue, debt and liabilities. Output: data/sec_facts_long.csv
"""
import json, time, requests, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "data"
UA = {"User-Agent": "AcademicResearch research-bot@example.org"}  # SEC requires a UA string

# universe from common.py
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FLD
TICKERS = sorted(FLD)
CONCEPTS = ["RevenueRemainingPerformanceObligation","PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsToAcquireProductiveAssets","NetCashProvidedByUsedInOperatingActivities",
            "RevenueFromContractWithCustomerExcludingAssessedTax","Revenues","LongTermDebt",
            "LongTermDebtNoncurrent","LongTermDebtCurrent","DebtCurrent","Liabilities","LiabilitiesCurrent",
            "Assets","StockholdersEquity","OperatingLeaseLiability","FinanceLeaseLiability",
            "CashAndCashEquivalentsAtCarryingValue","WeightedAverageNumberOfDilutedSharesOutstanding",
            "LongTermNotesAndLoans","NotesPayableCurrent","DebtLongtermAndShorttermCombinedAmount",
            "DebtInstrumentCarryingAmount","LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths",
            "PropertyPlantAndEquipmentNet","OperatingIncomeLoss"]

def main():
    tick = json.load(open(DATA / "company_tickers.json"))
    cik = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in tick.values()}
    rows = []
    for t in TICKERS:
        if t not in cik: print("no CIK", t); continue
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik[t]}.json"
        for a in range(4):
            try:
                r = requests.get(url, headers=UA, timeout=60); r.raise_for_status(); facts = r.json(); break
            except Exception as e:
                print("retry", t, e); time.sleep(3*(a+1)); facts = None
        if not facts: continue
        ns = facts.get("facts", {})
        g = dict(ns.get("ifrs-full", {}))      # foreign private issuers filing 20-F
        g.update(ns.get("us-gaap", {}))        # us-gaap wins where a firm reports both
        for c in CONCEPTS:
            if c not in g: continue
            for unit, obs in g[c]["units"].items():
                for o in obs:
                    rows.append(dict(ticker=t, concept=c, unit=unit, start=o.get("start"), end=o["end"],
                                     val=o["val"], fy=o.get("fy"), fp=o.get("fp"), form=o.get("form"),
                                     filed=o.get("filed"), frame=o.get("frame")))
        time.sleep(0.25)
        print(t, "ok")
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "sec_facts_long.csv", index=False)
    print(df.shape)

if __name__ == "__main__":
    main()
