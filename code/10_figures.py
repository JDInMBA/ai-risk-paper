"""10_figures.py - all figures used in the paper (PNG, 300 dpi)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from common import *

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ORD = {0: "#86b6ef", 1: "#3987e5", 2: "#1c5cab", 3: "#0d366b"}
FLDLAB = {0: "FLD 0 (control)", 1: "FLD 1 (indirect AI)", 2: "FLD 2 (diversified lab partner)", 3: "FLD 3 (lab-dependent)"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight", "lines.linewidth": 1.6})
def save(fig, name):
    # captions come from the manuscript
    for ax in fig.axes:
        if ax.get_title(loc="left").startswith("Figure"): ax.set_title("", loc="left")
    if fig._suptitle is not None and fig._suptitle.get_text().startswith("Figure"): fig._suptitle.set_text("")
    fig.savefig(FIG / name, facecolor="white"); plt.close(fig); print("saved", name)
def shade(ax):
    cols = ["#f4f4f2", "#ffffff", "#eef4fc"]
    for (p, (a, b)), c in zip(PERIODS.items(), cols):
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=c, zorder=0, lw=0)
r = pd.read_csv(OUT / "returns.csv", index_col=0, parse_dates=True)

# F1 cumulative performance
fig, ax = plt.subplots(figsize=(7.2, 3.4)); shade(ax)
for nm, lab, c in [("AI_IDX","AI-complex index (15 firms, EW)",CAT[0]),("CTL_IDX","Control index (20 firms, EW)",CAT[1]),("SPY","S&P 500 ETF",INK2)]:
    s = np.exp(r[nm].fillna(0).cumsum()); ax.plot(s.index, s, color=c, lw=1.6 if nm!="SPY" else 1.1, label=lab)
    off = {"AI_IDX": 0, "CTL_IDX": 6, "SPY": -6}[nm]
    ax.annotate(f"{s.iloc[-1]:.2f}", (s.index[-1], s.iloc[-1]), xytext=(4, off), textcoords="offset points", color=INK2, fontsize=7.5, va="center")
ax.set_yscale("log"); ax.set_ylabel("Growth of $1 (log scale)")
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator
ax.yaxis.set_major_locator(FixedLocator([0.75,1,1.5,2,3,4,6,8])); ax.yaxis.set_major_formatter(FormatStrFormatter("%.2g")); ax.yaxis.set_minor_locator(NullLocator()); ax.set_ylim(0.75, 9)
for d, t in [("2022-11-30","ChatGPT"),("2025-01-27","DeepSeek"),("2025-09-10","Oracle RPO\n$455bn"),("2026-10-26","Anthropic\nS-1")]:
    ax.axvline(pd.Timestamp(d), color=INK2, lw=0.6, ls=":"); ax.text(pd.Timestamp(d), 0.8, t, fontsize=7, color=INK2, ha="center", va="bottom")
ax.legend(loc="upper left", fontsize=7.5); ax.set_xlim(r.index[0], r.index[-1] + pd.Timedelta(days=90))
ax.set_title("Figure 1. AI-complex vs control equities, Jan 2021 - Nov 2026 (shading: P0, P1, P2)", loc="left", fontsize=9.5)
save(fig, "fig01_cumulative.png")

# F2 connectedness
conn = pd.read_csv(OUT / "dy_total_connectedness.csv", index_col=0, parse_dates=True).iloc[:,0]
fig, ax = plt.subplots(figsize=(7.2, 3.0)); shade(ax)
ax.plot(conn.index, conn, color=CAT[0]); ax.set_ylabel("Total connectedness (%)")
st = pd.read_csv(OUT / "dy_static_by_period.csv", index_col=0)
for (p,(a,b)) in PERIODS.items():
    ax.text(pd.Timestamp(a)+(pd.Timestamp(b)-pd.Timestamp(a))/2, conn.min()-1, f"{p.split()[0]}: {st.loc[p,'total']:.1f}%", ha="center", fontsize=7.5, color=INK2)
ax.set_title("Figure 2. Diebold-Yilmaz total connectedness of the core AI complex (250-day rolling, VAR(1), H=10)", loc="left", fontsize=9.5)
save(fig, "fig02_connectedness.png")

# F3 tail measures by FLD
tm = pd.read_csv(OUT / "firm_tail_measures.csv")
g = tm.groupby(["period","fld"])[["mes","es5","ltd"]].mean()
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.8))
for ax, (col, lab) in zip(axs, [("mes","MES, % (more negative = worse)"),("es5","ES 95%, % daily"),("ltd","Lower-tail dependence (5%)")]):
    w = 0.2
    for k in range(4):
        vals = [g.loc[(p,k), col] if (p,k) in g.index else np.nan for p in PERIODS]
        ax.bar(np.arange(3)+(k-1.5)*w, vals, w*0.92, color=ORD[k], label=FLDLAB[k])
    ax.set_xticks(range(3)); ax.set_xticklabels([p.split()[0] for p in PERIODS]); ax.set_title(lab, fontsize=8.5); ax.axhline(0, color=INK2, lw=0.6)
axs[0].legend(fontsize=6.5, loc="lower left")
fig.suptitle("Figure 3. Firm tail-risk measures by frontier-lab dependence (FLD) and sub-period", x=0.01, ha="left", fontsize=9.5)
fig.tight_layout(); save(fig, "fig03_tail_by_fld.png")

# F4 event study heatmap
ev = pd.read_csv(OUT / "event_study_by_fld.csv")
lab = [f"{d}  {e}" for d, e in zip(ev.date, ev.event)]
M = ev[["0","1","2","3"]].values
fig, ax = plt.subplots(figsize=(7.2, 4.6)); lim = np.nanmax(np.abs(M))
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
im = ax.imshow(M, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
for i in range(M.shape[0]):
    for j in range(4): ax.text(j, i, f"{M[i,j]:+.1f}", ha="center", va="center", fontsize=7, color=INK)
ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab, fontsize=7); ax.set_xticks(range(4)); ax.set_xticklabels([FLDLAB[k].replace(" (","\n(") for k in range(4)], fontsize=7)
ax.grid(False); cb = fig.colorbar(im, ax=ax, shrink=0.6); cb.set_label("Mean CAR[-1,+1], %")
ax.set_title("Figure 4. Market-model CARs around frontier-lab news, by FLD group", loc="left", fontsize=9.5)
save(fig, "fig04_event_heatmap.png")

# F5 stress days
sd = pd.read_csv(OUT / "stress_days_P2.csv")
fig, ax = plt.subplots(figsize=(7.2, 3.0)); w = 0.2; x = np.arange(len(sd))
for k in range(4): ax.bar(x+(k-1.5)*w, sd[str(k)], w*0.92, color=ORD[k], label=FLDLAB[k])
ax.set_xticks(x); ax.set_xticklabels([f"{d}\n(AI {a:.1f}%)" for d, a in zip(sd.date, sd.ai_idx)], fontsize=6.5)
ax.axhline(0, color=INK2, lw=0.6); ax.set_ylabel("Mean daily return, %"); ax.legend(fontsize=6.5, ncol=4, loc="lower left")
ax.set_title("Figure 5. Ten worst AI-complex days in the backlog era (P2): returns by FLD group", loc="left", fontsize=9.5)
save(fig, "fig05_stress_days.png")

# F6 VaR exceedance rates
vb = pd.read_csv(OUT / "var_backtests.csv"); vb = vb[vb.model.isin(["HS","EWMA","GARCH-t","FHS","QR-state"])]
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
ports = ["CTL_IDX","FLD2_basket","AI_IDX","FLD3_basket"]
for ax, lev, nom in [(axs[0],"95",5),(axs[1],"99",1)]:
    d = vb[vb.level.astype(float)==float(lev)]
    for i, m in enumerate(["HS","EWMA","GARCH-t","FHS","QR-state"]):
        dd = d[d.model==m].set_index("portfolio").reindex(ports)
        ax.scatter(dd.rate*100, np.arange(4)+(i-2)*0.12, s=26, color=CAT[i], label=m, zorder=3, edgecolor="white", linewidth=0.8)
        rej = dd.kupiec_p < 0.05
        ax.scatter((dd.rate*100)[rej], (np.arange(4)+(i-2)*0.12)[rej.values], s=70, facecolor="none", edgecolor=CAT[i], lw=0.9, zorder=2)
    ax.axvline(nom, color=INK, lw=0.9, ls="--"); ax.set_title(f"{lev}% VaR (nominal {nom}%)", fontsize=8.5); ax.set_xlabel("Exceedance rate, %")
axs[0].set_yticks(range(4)); axs[0].set_yticklabels(["Control index","FLD2 basket","AI-complex index","FLD3 basket"])
axs[1].legend(fontsize=6.5, loc="lower right")
fig.suptitle("Figure 6. Out-of-sample VaR exceedances, Jan 2024 - Nov 2026 (ring = Kupiec rejects at 5%)", x=0.01, ha="left", fontsize=9.5)
fig.tight_layout(); save(fig, "fig06_var_backtest.png")

# F7 FLD3 returns and VaR
vf = pd.read_csv(OUT / "var_forecasts.csv", header=[0,1], index_col=0, parse_dates=True)
pr = pd.read_csv(OUT / "portfolio_returns.csv", index_col=0, parse_dates=True)
y = pr["FLD3_basket"].loc["2024-01-02":]
fig, ax = plt.subplots(figsize=(7.2, 3.0))
ax.bar(y.index, y*100, width=1.0, color="#b9b8b2", label="FLD3 basket daily return")
ax.plot(vf.index, vf[("FLD3_basket","GARCH-t_95")]*100, color=CAT[0], lw=1.1, label="GARCH-t 95% VaR")
ax.plot(vf.index, vf[("FLD3_basket","HS_95")]*100, color=CAT[1], lw=1.1, label="HS 95% VaR")
br = y < vf[("FLD3_basket","GARCH-t_95")].reindex(y.index)
ax.scatter(y.index[br], y[br]*100, s=9, color=CAT[7], zorder=3, label="GARCH-t exceedance")
ax.set_ylabel("%"); ax.legend(fontsize=6.5, ncol=4, loc="lower left")
ax.set_title("Figure 7. Lab-dependent (FLD3) basket: returns, 95% VaR forecasts and exceedances", loc="left", fontsize=9.5)
save(fig, "fig07_fld3_var_path.png")

# F8 ML AUC by horizon
rows = []
for h in [1,5,10]:
    p = OUT / f"ml_metrics_h{h}.csv"
    if p.exists():
        m = pd.read_csv(p); m["h"] = h; rows.append(m)
if rows:
    M = pd.concat(rows); M = M[M["sample"].str.startswith("All")]
    models = ["Logit (vol only)","Logit (all)","Random forest","LightGBM (no AI/lab features)","LightGBM (all)"]
    fig, ax = plt.subplots(figsize=(7.2, 2.9)); w = 0.15; hs = sorted(M.h.unique())
    for i, mname in enumerate(models):
        v = [M[(M.h==h)&(M.model==mname)].AUC.values[0] for h in hs]
        ax.bar(np.arange(len(hs))+(i-2)*w, v, w*0.92, color=CAT[i], label=mname)
    ax.axhline(0.5, color=INK, lw=0.9, ls="--"); ax.set_ylim(0.4, 0.65)
    ax.set_xticks(range(len(hs))); ax.set_xticklabels([f"h = {h} day" + ("s" if h>1 else "") for h in hs]); ax.set_ylabel("Out-of-sample AUC")
    ax.legend(fontsize=6.5, ncol=3, loc="upper right")
    ax.set_title("Figure 8. Timing tail events: walk-forward AUC, 2024Q1-2026Q4 (dashed = no skill)", loc="left", fontsize=9.5)
    save(fig, "fig08_ml_auc.png")

# F9 stress-day ROC
from sklearn.metrics import roc_curve, roc_auc_score
te = pd.read_csv(OUT / "stress_day_test_predictions.csv")
fig, ax = plt.subplots(figsize=(3.8, 3.4))
for col, lab, c in [("p_base","Market-only logit",CAT[0]),("p_aug","Market + lab dependence",CAT[1])]:
    f, t, _ = roc_curve(te.y, te[col]); ax.plot(f, t, color=c, label=f"{lab} (AUC {roc_auc_score(te.y, te[col]):.3f})")
ax.plot([0,1],[0,1], color=INK2, lw=0.8, ls="--"); ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate"); ax.legend(fontsize=6.5, loc="lower right")
ax.set_title("Figure 9. Who falls hardest on\nAI stress days? (test: P2, 18 days)", loc="left", fontsize=9)
save(fig, "fig09_stress_roc.png")

# F10 SHAP importance h=1
sp = OUT / "ml_shap_importance_h1.csv"
if sp.exists():
    s = pd.read_csv(sp, index_col=0).iloc[:,0].sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.barh(s.index, s.values, color=CAT[0], height=0.7); ax.set_xlabel("Mean |SHAP| (log-odds)")
    ax.set_title("Figure 10. LightGBM feature attribution (h = 1, P2 sample)", loc="left", fontsize=9.5)
    save(fig, "fig10_shap.png")

# F11 commitments and stress loss
E = pd.read_csv(OUT / "commitment_network.csv"); S = pd.read_csv(OUT / "stress_results.csv")
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2))
agg = E[E.base_case].groupby(["supplier","lab"]).commit_to_mcap_pct.sum().unstack().fillna(0)
agg = agg.loc[agg.sum(1).sort_values().index]
axs[0].barh(agg.index, agg["Anthropic"], color=CAT[0], height=0.65, label="Anthropic (S-1/A)")
axs[0].barh(agg.index, agg["OpenAI"], left=agg["Anthropic"], color=CAT[1], height=0.65, label="OpenAI (filings/reports)")
axs[0].set_xlabel("Lab commitments, % of market cap"); axs[0].legend(fontsize=6.5, loc="lower right"); axs[0].set_title("Contracted exposure", fontsize=8.5)
s = S[(S.case=="base")&(S.margin==.30)].pivot_table(index="ticker", columns="scenario", values="loss_pct_mcap").reindex(agg.index)
for i, sc in enumerate(["Mild","Severe","Extreme"]):
    axs[1].barh(np.arange(len(s))+(i-1)*0.26, s[sc], 0.24, color=ORD[i+1], label=sc)
axs[1].set_yticks(range(len(s))); axs[1].set_yticklabels(s.index); axs[1].set_xlabel("First-round value loss, % of market cap"); axs[1].legend(fontsize=6.5, loc="lower right")
axs[1].set_title("Stress loss (margin 30%)", fontsize=8.5)
fig.suptitle("Figure 11. Frontier-lab commitment network: exposure and first-round stress losses (5 Nov 2026)", x=0.01, ha="left", fontsize=9.5)
fig.tight_layout(); save(fig, "fig11_commitment_stress.png")

# F12 Anthropic funding gap
fig, ax = plt.subplots(figsize=(6.2, 3.0))
items = [("TTM revenue", 35.282, CAT[0]), ("Cash + short-term inv.", 131.9, CAT[2]), ("Contracted revenue (RPO)", 13.8, CAT[0]),
         ("Commitments due next 12m", 69.8, CAT[1]), ("Commitments due before 2031", 447.5, CAT[1]), ("Total committed spend", 895.4, CAT[1])]
for i, (k, v, c) in enumerate(items):
    ax.barh(i, v, color=c, height=0.62); ax.text(v + 8, i, f"${v:,.1f}bn", va="center", fontsize=7.5, color=INK)
ax.set_yticks(range(len(items))); ax.set_yticklabels([k for k, _, _ in items]); ax.invert_yaxis(); ax.set_xlim(0, 1000); ax.set_xlabel("$bn (as of 30 Sep 2026)")
ax.set_title("Figure 12. Anthropic: obligations vs resources, from the Form S-1/A", loc="left", fontsize=9.5)
save(fig, "fig12_anthropic_alm.png")

# F13 RPO growth
facts = pd.read_csv(DATA / "sec_facts_long.csv", parse_dates=["end"])
rp = facts[(facts.concept=="RevenueRemainingPerformanceObligation")&(facts.end>="2023-01-01")].sort_values(["end","filed"]).drop_duplicates(["ticker","end"], keep="last")
fig, ax = plt.subplots(figsize=(7.2, 3.0))
for i, t in enumerate(["MSFT","ORCL","GOOGL","AVGO","CRWV"]):
    d = rp[rp.ticker==t].set_index("end").val/1e9
    d = d[d.index >= "2023-06-01"]; ax.plot(d.index, d, marker="o", ms=3.5, color=CAT[i], label=t)
    ax.annotate(f"{t} ${d.iloc[-1]:,.0f}bn", (d.index[-1], d.iloc[-1]), xytext=(4,0), textcoords="offset points", fontsize=7, color=INK2, va="center")
ax.set_ylabel("Remaining performance obligations, $bn"); ax.legend(fontsize=7, loc="upper left"); ax.set_xlim(None, pd.Timestamp("2027-03-01"))
ax.set_title("Figure 13. Contracted backlog (RPO) of AI-infrastructure suppliers, from XBRL filings", loc="left", fontsize=9.5)
save(fig, "fig13_rpo.png")

# F14 Merton DD
dd = pd.read_csv(OUT / "merton_dd.csv", parse_dates=["date"])
fig, ax = plt.subplots(figsize=(7.2, 3.0))
for i, t in enumerate(["ORCL","CRWV","IREN","CORZ","MSFT"]):
    d = dd[(dd.ticker==t)&(dd.date>="2023-01-01")]; ax.plot(d.date, d.DD, marker="o", ms=3, color=CAT[i], label=t)
ax.axhline(2, color=INK2, lw=0.7, ls=":"); ax.set_ylabel("Distance to default (sd)"); ax.legend(fontsize=7, ncol=5, loc="upper right"); ax.set_ylim(0, 22)
ax.set_title("Figure 14. Merton (KMV) distance to default, quarterly, from XBRL debt and equity volatility", loc="left", fontsize=9.5)
save(fig, "fig14_merton_dd.png")

# F15 BSADF
bs = pd.read_csv(OUT / "bsadf_sequences.csv", header=[0,1], index_col=0, parse_dates=True)
fig, ax = plt.subplots(figsize=(7.2, 2.8))
ax.plot(bs.index, bs[("AI_rel_SPY","bsadf")], color=CAT[0], lw=1.2, label="BSADF: AI index / S&P 500")
ax.plot(bs.index, bs[("AI_rel_SPY","cv95")], color=INK, lw=0.9, ls="--", label="95% critical value (Monte Carlo)")
ax.legend(fontsize=7, loc="lower left"); ax.set_ylabel("BSADF statistic")
ax.set_title("Figure 15. PSY date-stamping of explosive behaviour in AI-complex relative prices (weekly)", loc="left", fontsize=9.5)
save(fig, "fig15_bsadf.png")

# F16 AI IPO cohort
px = load_prices()
IPO = {"ARM": ("2023-09-14", 51.0), "CRWV": ("2025-03-28", 40.0), "CBRS": ("2026-05-14", 185.0)}
fig, ax = plt.subplots(figsize=(7.2, 3.0))
for i, (t, (d, p0)) in enumerate(IPO.items()):
    s = px[t].dropna().loc[d:] / p0 * 100; days = np.arange(len(s))
    ax.plot(days, s.values, color=CAT[i], label=f"{t} (IPO {d}, ${p0:.0f})")
    ax.annotate(f"{t} {s.iloc[-1]:.0f}", (days[-1], s.iloc[-1]), xytext=(4,0), textcoords="offset points", fontsize=7, color=INK2, va="center")
ax.axhline(100, color=INK, lw=0.8, ls="--"); ax.set_xlabel("Trading days since IPO"); ax.set_ylabel("Price, IPO offer = 100"); ax.set_yscale("log")
ax.legend(fontsize=7, loc="upper right")
ax.set_title("Figure 16. AI-infrastructure IPO cohort: price relative to offer price", loc="left", fontsize=9.5)
save(fig, "fig16_ipo_cohort.png")
