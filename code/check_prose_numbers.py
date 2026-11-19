"""
Check the numbers quoted in the manuscript prose against results/.

Tables and figures are filled from results/ when the manuscript is built, so they
cannot drift. Sentences are written by hand and can. This recomputes the quantities
the prose quotes and reports whether the manuscript still contains them.

    python code/check_prose_numbers.py
"""
import re, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
MS = ROOT / "drafts" / "02_Financial_Innovation" / "Manuscript_Financial_Innovation.md"
if not MS.exists():
    sys.exit(f"build the manuscript first: {MS} not found")
TEXT = MS.read_text().split("# References")[0]

checks, failed = [], 0
def want(label, snippet, value=None):
    """Assert the prose contains `snippet`, which was computed from results/."""
    global failed
    hit = snippet in TEXT
    failed += not hit
    checks.append((hit, label, snippet if value is None else f"{snippet}   (from results: {value})"))

csv = lambda n, **kw: pd.read_csv(OUT / n, **kw)

# tail measures by FLD
tb = csv("audit_tail_beta_with_intercepts.csv")
row = tb[(tb.period == "P2 backlog era") & tb.term.str.startswith("mean tail-day")].note.iloc[0]
f3, f1, f2, f0 = [float(x) for x in row.split(" / ")]
want("mean tail-day return, FLD 3", f"{abs(f3):.1f}%", f3)
want("mean tail-day return, FLD 2", f"{abs(f2):.1f}%", f2)

# connectedness
dy = csv("dy_static_by_period.csv", index_col=0)["total"]
want("connectedness P0", f"{dy.iloc[0]:.0f}%", dy.iloc[0])
want("connectedness P2", f"{dy.iloc[-1]:.0f}%", dy.iloc[-1])

# VaR backtests at 95%
vb = csv("var_backtests.csv")
b = vb[(vb.portfolio == "FLD3_basket") & (vb.level == 95)].rate * 100
want("FLD-3 basket 95% exceedance range, low", f"{b.min():.1f}%", b.min())
want("FLD-3 basket 95% exceedance range, high", f"{b.max():.1f}%", b.max())

# placebo and random baskets
pv = csv("placebo_var_backtests.csv")
fails = pv[pv.level == 95].groupby("portfolio").fail.sum()
m = [p for p in fails.index if p.startswith("PLACEBO matched")][0]
want("matched placebo rejections at 95%", f"{int(fails[m])} of 5")
rs = csv("placebo_random_summary.csv").set_index("model")
want("FLD-3 percentile among random baskets", f"{rs.loc['HS 95%','percentile_of_fld3']:.0f}rd percentile",
     rs.loc["HS 95%", "percentile_of_fld3"])

# permutation p-values
rb = csv("cs_robustness.csv")
perm = rb[rb["drop"].str.startswith("PERM")].set_index("spec")["t"]
for spec, lab in (("MES", "MES"), ("ES5", "ES"), ("LTD", "tail dependence")):
    want(f"permutation p, {lab}", f"{perm[spec]:.3f}", perm[spec])

# machine learning
for h in (1, 5, 10):
    mm = csv(f"ml_metrics_h{h}.csv")
    mm = mm[mm["sample"].str.startswith("All")]
    if h == 1:
        for model in ("Logit (vol only)", "Logit (all)", "Random forest", "LightGBM (all)",
                      "LightGBM (no AI/lab features)"):
            v = mm[mm.model == model].AUC.iloc[0]
            want(f"h=1 AUC, {model}", f"{v:.3f}", v)
sd = csv("ml_stress_day_results.csv")
sd = sd[(sd.split == "Train P0-P1 / Test P2") & (sd.features == "market-only")]
want("stress-day logit AUC", f"{sd[sd.model=='logit'].AUC.iloc[0]:.3f}")
want("stress-day LightGBM AUC", f"{sd[sd.model=='lgbm'].AUC.iloc[0]:.3f}")

# commitment stress
st = csv("stress_results.csv")
ex = st[(st.margin == 0.30) & (st.case == "base") & (st.scenario == "Extreme")].set_index("ticker")["loss_pct_mcap"]
trio = ex.reindex(["ORCL", "CRWV", "CBRS"])
want("extreme-scenario loss, low end", f"{trio.min():.0f}%", trio.min())
want("extreme-scenario loss, high end", f"{trio.max():.0f}%", trio.max())

# report
for hit, label, detail in checks:
    print(f"[{'PASS ' if hit else 'CHECK'}] {label}: {detail}")
print(f"\n{len(checks) - failed} of {len(checks)} quoted values found in the prose")
print("PROSE MATCHES RESULTS" if not failed else "CHECK THE LINES ABOVE")
