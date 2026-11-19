"""
08_commitment_stress.py
Commitment-network stress test: how much listed-supplier equity value is exposed to the two
frontier labs through (i) take-or-pay / committed compute contracts and (ii) equity stakes.
Anthropic amounts are from its Form S-1/A (filed 2 Nov 2026). OpenAI amounts are from
supplier filings and widely reported announcements; status column records legal weight.
First-round loss_i = sum_l [ h * C_li * m ] + stake_li * V_l * delta,
reported as % of the supplier's market capitalisation on 5 Nov 2026.
Also: frontier-lab funding-gap metrics computed from the S-1/A (ALM-style ratios).
"""
import numpy as np, pandas as pd
from common import *

E = pd.DataFrame([
 # lab, supplier, $bn, status
 ("Anthropic","GOOGL",121.3,"S-1/A (non-cancelable)","CSP hosting Apr-2026..Jul-2033; further TPU est. $125-150bn excluded"),
 ("Anthropic","AMZN",110.0,"S-1/A (non-cancelable)","May-2026..Apr-2036"),
 ("Anthropic","MSFT",44.9,"S-1/A (non-cancelable)","Nov-2026..Jun-2033"),
 ("Anthropic","AVGO",125.2,"S-1/A (equipment lease, partly assigned)","2026 Lease Arrangement rent over 5 yrs; portion assigned to financial partner"),
 ("OpenAI","ORCL",300.0,"Press reports (not named in Oracle filings)","~5-yr cloud contract from 2027"),
 ("OpenAI","MSFT",250.0,"Company announcement (Oct-2025)","incremental Azure commitment"),
 ("OpenAI","AMZN",138.0,"Supplier disclosure / reports","$38bn (Nov-2025) + reported $100bn expansion"),
 ("OpenAI","CRWV",22.4,"Supplier filing","maximum contract value"),
 ("OpenAI","CBRS",20.0,"Supplier filing (approx.)","~750 MW inference capacity through 2028"),
 ("OpenAI","AVGO",350.0,"Analyst estimate (terms undisclosed)","10 GW custom accelerators - extended scenario only"),
 ("OpenAI","AMD",90.0,"Analyst estimate (6 GW)","extended scenario only; plus warrant dilution"),
], columns=["lab","supplier","commit_bn","status","note"])
E["base_case"] = ~E.status.str.startswith("Analyst")
STAKES = pd.DataFrame([("OpenAI","MSFT",0.23,"~27% as-converted at Oct-2025 recap, diluted by Mar-2026 round (approx.)")],
                      columns=["lab","holder","stake","note"])
LAB_VAL = {"OpenAI": 852.0, "Anthropic": 965.0}   # last primary rounds: Mar-2026 and May-2026 ($bn)

px = load_prices(); facts = pd.read_csv(DATA / "sec_facts_long.csv")
sh = facts[facts.concept=="WeightedAverageNumberOfDilutedSharesOutstanding"].sort_values(["end","filed"])
# Cerebras shares: 10-Q, Jun 2026
SHARES_OVERRIDE = {"CBRS": 228.2e6}
mcap = {}
for t in E.supplier.unique():
    s = pd.Series([SHARES_OVERRIDE[t]]) if t in SHARES_OVERRIDE else sh[sh.ticker==t].val
    mcap[t] = px[t].dropna().iloc[-1] * s.iloc[-1] / 1e9 if len(s) else np.nan
mc = pd.Series(mcap, name="mcap_bn_2026_11_05")
E = E.merge(mc, left_on="supplier", right_index=True)
E["commit_to_mcap_pct"] = E.commit_bn / E.mcap_bn_2026_11_05 * 100
E.to_csv(OUT / "commitment_network.csv", index=False)
print(E[["lab","supplier","commit_bn","status","mcap_bn_2026_11_05","commit_to_mcap_pct"]].round(1).to_string())

rows = []
grid = [("Mild", .10, .30), ("Severe", .25, .50), ("Extreme", .50, .70)]
for scen, h, d in grid:
    for m in [.15, .30, .50]:
        for case in ["base","extended"]:
            Ec = E if case == "extended" else E[E.base_case]
            for t in mc.index:
                c = Ec[Ec.supplier==t]
                loss = (h * c.commit_bn * m).sum()
                st = STAKES[STAKES.holder==t]
                loss += sum(sv.stake * LAB_VAL[sv.lab] * d for sv in st.itertuples())
                rows.append(dict(scenario=scen, haircut=h, lab_markdown=d, margin=m, case=case, ticker=t,
                                 loss_bn=loss, loss_pct_mcap=loss / mc[t] * 100))
S = pd.DataFrame(rows); S.to_csv(OUT / "stress_results.csv", index=False)
piv = S[(S.margin==.30)].pivot_table(index="ticker", columns=["case","scenario"], values="loss_pct_mcap")
print(piv.round(1).to_string())
# aggregate loss against market cap
agg = S.groupby(["scenario","margin","case"]).loss_bn.sum().unstack("case") / mc.sum() * 100
agg.to_csv(OUT / "stress_aggregate.csv"); print(agg.round(2))

# Anthropic funding gap
A = dict(rev_9m26=33.242, rev_9m25=2.545, rev_fy25=4.585, opinc_9m26=-1.260, ocf_9m26=6.569,
         cash=95.6, sti=36.3, commit_total=895.4, commit_pre2031=447.5, commit_12m=69.8, rpo=13.8,
         rpo_12m_share=0.75, raised=141.1,
         # cancelable share, from the S-1/A
         commit_cancelable=213.8)
A["rev_ttm"] = A["rev_fy25"] - A["rev_9m25"] + A["rev_9m26"]
A["commit_noncancelable"] = A["commit_total"] - A["commit_cancelable"]
ratios = {
 "TTM revenue ($bn)": A["rev_ttm"],
 "Committed spend / TTM revenue (x)": A["commit_total"]/A["rev_ttm"],
 "Non-cancelable commitments / TTM revenue (x)": A["commit_noncancelable"]/A["rev_ttm"],
 "Cancelable share of commitments (%)": A["commit_cancelable"]/A["commit_total"]*100,
 "Commitments due <2031 / TTM revenue (x)": A["commit_pre2031"]/A["rev_ttm"],
 "Next-12m commitments / TTM revenue (x)": A["commit_12m"]/A["rev_ttm"],
 "Liquidity (cash+STI) / next-12m commitments (x)": (A["cash"]+A["sti"])/A["commit_12m"],
 "Contracted revenue (RPO) / committed spend (%)": A["rpo"]/A["commit_total"]*100,
 "RPO due in 12m / next-12m commitments (%)": A["rpo"]*A["rpo_12m_share"]/A["commit_12m"]*100,
}
R = pd.Series(ratios).round(3); R.to_csv(OUT / "anthropic_alm_ratios.csv"); print(R)
# years of revenue growth needed
avg_annual = (A["commit_pre2031"] - A["commit_12m"]) / 3.25
q3 = 33.242 - 0  # placeholder
