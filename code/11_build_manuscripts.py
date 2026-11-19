"""
11_build_manuscripts.py
Assembles the manuscript variants from drafts/src/*.md, fills every table from results/*.csv,
renumbers figures by order of first citation, and exports DOCX (pandoc) and PDF (LibreOffice).
"""
import re, subprocess, shutil, numpy as np, pandas as pd, statsmodels.formula.api as smf
from pathlib import Path
from common import ROOT, OUT, FIG, FLD
SRC = ROOT / "drafts" / "src"; DR = ROOT / "drafts"

TITLE = "Lab-dependence risk in the AI build-out: backlog concentration, tail co-movement and the limits of VaR as frontier AI labs go public"
RUNNING = "Lab-dependence risk in the AI build-out"
KEYWORDS = ["Artificial intelligence", "Wrong-way risk", "Tail risk", "Value at risk", "Initial public offerings", "Machine learning"]
JEL = "G12, G14, G24, G32, C58"
AUTHOR = "Jay Raviraj Deore"
AFFIL = ("Finance, Symbiosis Institute of Management Studies (SIMS), "
         "Symbiosis International (Deemed University), Pune, India")
EMAIL = "jay.deore2027@sims.edu"
BIO = ("Jay Raviraj Deore is an MBA (Finance) student at the Symbiosis Institute of Management Studies, "
       "Symbiosis International (Deemed University), Pune, having previously completed a Bachelor of "
       "Engineering in Electronics and Telecommunication, with honours in artificial intelligence and "
       "machine learning, at the International Institute of Information Technology. Research interests "
       "cover financial risk, machine learning and financial technology.")
ABSTRACT = ("Frontier AI laboratories have committed to spend far more than they earn, and listed suppliers carry those commitments as contracted backlog. "
 "Using daily data on 69 US-traded firms from January 2021 to November 2026, a hand-coded measure of frontier-lab dependence (FLD) built from SEC filings, "
 "and the contract terms in Anthropic's 2026 registration statement, we study its effect on tail risk. In the backlog era that began in "
 "September 2025, the most lab-dependent firms (Oracle, CoreWeave, Cerebras, AMD and SoftBank) had the largest marginal expected shortfall (-7.1% against -2.9% "
 "for hyperscalers) and the strongest reactions to lab news. The extra risk works through exposure to the AI factor. It disappears once contemporaneous beta and "
 "volatility are controlled for, and panel estimates show no excess tail beta. Average connectedness of the AI complex fell from 81% to 65% while its 99% EVT VaR "
 "relative to non-AI equities doubled. All five VaR models tested under-cover a lab-dependent basket at the 95% level (7.0% to 10.8% exceedances; three of five "
 "with membership held fixed) but not a non-AI control portfolio. Placebo baskets of AI firms without lab contracts, matched on pre-period risk, are rejected at comparable rates and the lab-dependent basket sits at the 83rd percentile of 500 random AI baskets, so the "
 "under-coverage is a property of AI-factor exposure rather than of lab dependence. Machine-learning models using connectedness, bubble and lab features do not time tail events better than volatility alone (AUC at most 0.59), and lab "
 "dependence adds nothing to a market-based ranking of stress-day losers (AUC 0.82). A contract-level stress test puts first-round losses at 13% to 16% of market "
 "value for Oracle, CoreWeave and Cerebras, far below their realised drawdowns of 65% to 73%. We draw implications for frontier-lab IPOs and risk-model validation.")
HIGHLIGHTS = ["Lab-dependent AI suppliers became the riskiest AI stocks after September 2025",
              "Their extra tail risk runs through AI-factor beta, not a separate crash channel",
              "Standard VaR models under-cover AI suppliers generally, not just lab-dependent ones",
              "Machine learning cannot time AI tail events; market beta ranks stress-day losers",
              "Anthropic's commitments are 25 times revenue; contracted revenue covers 1.5%"]
CAPTIONS = {
 "fig01_cumulative.png": "Cumulative growth of $1 in the AI, control and market indices. Equal-weighted AI-complex index (15 firms), control index (20 non-AI firms) and the S&P 500 ETF, January 2021 to November 2026. Shading marks P0 (pre-ChatGPT), P1 (AI boom) and P2 (backlog era).",
 "fig02_connectedness.png": "Rolling total connectedness of the AI complex. Diebold-Yilmaz total connectedness of the 15 core AI-complex firms, 250-day rolling window, VAR(1), generalised FEVD at H = 10. Labels give full-period estimates.",
 "fig03_tail_by_fld.png": "Tail risk by frontier-lab dependence group and sub-period. Mean marginal expected shortfall, 95% expected shortfall and lower-tail dependence by FLD group.",
 "fig04_event_heatmap.png": "Abnormal returns around frontier-lab events by FLD group. Mean market-model CAR[-1,+1] (%) across 15 events.",
 "fig05_stress_days.png": "Returns by FLD group on the worst AI-complex days of P2. Mean daily return on the ten worst days. The AI-index return is shown under each date.",
 "fig06_var_backtest.png": "Out-of-sample VaR exceedance rates by portfolio and model. January 2024 to November 2026. Rings mark rejection by Kupiec's test at 5%; the dashed line is the nominal rate.",
 "fig07_fld3_var_path.png": "FLD-3 basket returns against 95% VaR forecasts. GARCH-t and historical-simulation forecasts; red dots are GARCH-t exceedances.",
 "fig08_ml_auc.png": "Out-of-sample AUC of tail-event timing models by horizon. Walk-forward 2024Q1 to 2026Q4. The dashed line marks no skill.",
 "fig09_stress_roc.png": "ROC curves for ranking stress-day losers. Predicting the worst-third losers on the 18 P2 stress days, from models trained on P0 and P1 stress days.",
 "fig10_shap.png": "Mean absolute SHAP values for the next-day VaR-breach model. The 15 most important features, LightGBM, P2 sample.",
 "fig11_commitment_stress.png": "Frontier-lab commitments and first-round stress losses by supplier. Left: commitments as a percentage of supplier market value on 5 November 2026 (base case). Right: first-round stress losses with a 30% value-loss rate.",
 "fig12_anthropic_alm.png": "Anthropic's resources and obligations at 30 September 2026. From its Form S-1/A.",
 "fig13_rpo.png": "Remaining performance obligations of AI-infrastructure suppliers, quarterly. From SEC XBRL filings.",
 "fig14_merton_dd.png": "Merton (KMV) distance to default, quarterly. From XBRL debt and 252-day equity volatility. The dotted line marks DD = 2.",
 "fig15_bsadf.png": "PSY backward SADF sequence for the AI complex against the market. AI-complex index relative to the S&P 500, weekly, with its 95% Monte Carlo critical value.",
 "fig16_ipo_cohort.png": "Price relative to IPO offer price for three AI listings. Offer = 100, log scale, for Arm, CoreWeave and Cerebras.",
}
def f(x, d=2): return "" if pd.isna(x) else f"{x:.{d}f}"
def mdtable(df, title, note=None):
    cols = list(df.columns)
    lines = [f"*{title}*", "", "| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join([":--"] + ["--:"]*(len(cols)-1)) + "|"]
    for _, row in df.iterrows(): lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    if note: lines += ["", f"Note: {note}"]
    return "\n".join(lines) + "\n"

def build_tables():
    T = {}
    short = {"P0 pre-ChatGPT":"P0","P1 AI boom":"P1","P2 backlog era":"P2"}
    T["gap"] = mdtable(pd.DataFrame([
        ["AI exposure and valuation","Babina et al. (2024); Eisfeldt et al. (2023)","Returns and growth of AI-exposed firms","Risk transfer through lab contracts"],
        ["AI bubbles","Basele et al. (2025); Pastor and Veronesi (2009)","Explosive price episodes","Tail co-movement and model coverage"],
        ["Systemic risk","Adrian and Brunnermeier (2016); Acharya et al. (2017); Diebold and Yilmaz (2014)","Bank tail spillovers","Application to a contract network outside banking"],
        ["Supply-chain links","Cohen and Frazzini (2008); Hertzel et al. (2008); Patatoukas (2012)","Customer news and distress spillovers","Backlogs larger than supplier equity; risk-model tests"],
        ["Circular AI financing","IMF (2026); Solanki (2026)","Deal mapping, qualitative wrong-way risk","Market-based estimates and stress test from primary filings"],
        ["Risk models and ML","Kupiec (1995); Christoffersen (1998); Gu et al. (2020)","VaR backtests; ML return prediction","Coverage and predictability for lab-dependent portfolios"],
    ], columns=["Strand","Representative studies","What is measured","Gap addressed here"]), "Table 1. Position of this study in the literature")
    E = pd.read_csv(OUT/"commitment_network.csv")
    e = E[["lab","supplier","commit_bn","status","mcap_bn_2026_11_05","commit_to_mcap_pct"]].copy()
    e.columns = ["Lab","Supplier","Commitment ($bn)","Source / legal status","Market value 5 Nov 2026 ($bn)","Commitment / market value (%)"]
    for c in ["Commitment ($bn)","Market value 5 Nov 2026 ($bn)","Commitment / market value (%)"]: e[c] = e[c].map(lambda v: f"{v:.1f}")
    MCAP_NOTE = (" Market values use the latest weighted-average diluted share count in SEC XBRL filings, the only"
                 " share series in our data; against point-in-time shares outstanding this differs by about 2% to 5%"
                 " for the large-cap suppliers, which does not affect the ranking or any conclusion. Cerebras is the"
                 " exception, because its weighted-average count straddles the May 2026 IPO, so we use the 228.2"
                 " million class A, B and N shares outstanding in its June 2026 10-Q.")
    T["network"] = mdtable(e, "Table 2. Frontier-lab commitments to listed suppliers",
        "Anthropic amounts are minimum commitments from its Form S-1/A (2 November 2026); the Broadcom figure is rent under an equipment lease of which a portion was assigned to a financial partner. OpenAI amounts come from supplier filings, company announcements or press reports; rows marked 'Analyst estimate' enter only the extended stress scenario." + MCAP_NOTE)
    A = pd.read_csv(OUT/"anthropic_alm_ratios.csv", index_col=0).iloc[:,0]
    T["alm"] = mdtable(pd.DataFrame({"Ratio": A.index, "Value": [f"{v:,.2f}" for v in A.values]}), "Table 3. Asset-liability ratios for Anthropic (30 September 2026)",
        "Computed from the Form S-1/A. TTM revenue = FY2025 revenue - 9M2025 revenue + 9M2026 revenue. RPO is remaining performance obligations (contracted customer revenue); 75% of it is expected within 12 months. The filing discloses that about $213.8bn of the $895.4bn of contractual obligations is subject to exercisable termination rights, leaving $681.6bn non-cancelable; the cancelable share calibrates the haircut in the Section 7.8 stress test.")
    st = pd.read_csv(OUT/"dy_static_by_period.csv", index_col=0)["total"]
    ev = pd.read_csv(OUT/"evt_pot.csv")
    bo = pd.read_csv(OUT/"audit_evt_shape.csv")
    rows = []
    for p in st.index:
        g = ev[ev.period==p].set_index("series"); b = bo[bo.period==p].set_index("series")
        ai_ci = f"[{b.loc['AI_IDX','VaR99_lo']:.2f}, {b.loc['AI_IDX','VaR99_hi']:.2f}]"
        rt_ci = f"[{b.loc['AI/CTL VaR99 ratio','VaR99_lo']:.2f}, {b.loc['AI/CTL VaR99 ratio','VaR99_hi']:.2f}]"
        xi_s = f"{b.loc['AI_IDX','xi']:.2f} ({b.loc['AI_IDX','xi_se']:.2f})"
        rows.append([short[p], f(st[p],1), f(g.loc["AI_IDX","VaR99"]), ai_ci, f(g.loc["AI_IDX","ES99"]),
                     f(g.loc["CTL_IDX","VaR99"]), f(g.loc["SPY","VaR99"]),
                     f(g.loc["AI_IDX","VaR99"]/g.loc["CTL_IDX","VaR99"]), rt_ci, xi_s,
                     str(int(b.loc["AI_IDX","n_exc"]))])
    T["system"] = mdtable(pd.DataFrame(rows, columns=["Period","Total connectedness (%)","AI VaR99 (%)","AI VaR99 95% CI","AI ES99 (%)","Control VaR99 (%)","S&P 500 VaR99 (%)","AI/control VaR99","Ratio 95% CI","AI GPD shape xi (se)","Exceedances"]),
        "Table 4. System connectedness and EVT tail risk by period",
        "Connectedness: Diebold-Yilmaz generalised FEVD, VAR(1), H = 10, 15 core firms. VaR and ES: one-day, peaks over the 90th-percentile loss threshold, generalised Pareto fit. Intervals and shape standard errors come from 1,000 paired bootstrap resamples of trading days, which preserves the dependence between the AI and control series and so gives the ratio a tighter interval than either level. With 30 to 70 exceedances per period the shape parameter is weakly identified, so the levels should be read with their intervals and the ratio treated as the better-determined quantity.")
    tm = pd.read_csv(OUT/"firm_tail_measures.csv")
    g = tm.groupby(["period","fld"]).agg(N=("ticker","size"), beta=("beta","mean"), down_beta=("down_beta","mean"), mes=("mes","mean"), dcovar=("dcovar","mean"), ltd=("ltd","mean"), var5=("var5","mean"), es5=("es5","mean")).reset_index()
    g["fld"] = g.fld.map(lambda k: f"FLD {k}"); g["period"] = g.period.map(short)
    for c in ["beta","down_beta","mes","dcovar","ltd","var5","es5"]: g[c] = g[c].map(f)
    g.columns = ["Period","Group","N","Beta","Downside beta","MES (%)","Delta-CoVaR (%)","Tail dependence","VaR 95% (%)","ES 95% (%)"]
    T["tail"] = mdtable(g, "Table 5. Firm tail measures by FLD group and period (group means)", "System = equal-weighted core AI index excluding the firm. MES: mean firm return on the system's worst 5% of days. Delta-CoVaR at q = 0.05 with lagged state variables. Tail dependence: empirical lower-tail coefficient at 5%.")
    reg = pd.read_csv(OUT/"cs_regressions.csv").set_index("model")
    rb = pd.read_csv(OUT/"cs_robustness.csv"); perm = rb[rb["drop"].str.startswith("PERM")].set_index("spec")["t"]
    sel = [("MES (P2) ctrl","MES","P1 controls"),("ES5 (P2) ctrl","ES 95%","P1 controls"),("LTD ctrl","Tail dep.","P1 controls"),("MES contemp","MES","P2 controls"),("ES5 contemp","ES 95%","P2 controls"),("LTD contemp","Tail dep.","P2 controls")]
    rows = []
    for var in ["fld3","fld2","fld1"]:
        rows.append([var.upper().replace("FLD","FLD ")] + [f"{reg.loc[m,var]:.2f}" for m,_,_ in sel])
    rows.append(["FLD 3 - FLD 1, p-value"] + [f(reg.loc[m,"fld3_minus_fld1_p"],3) for m,_,_ in sel])
    pm = {"MES (P2) ctrl":"MES","ES5 (P2) ctrl":"ES5","LTD ctrl":"LTD"}
    rows.append(["Permutation p (FLD 3)"] + [f(perm[pm[m]],3) if m in pm else "" for m,_,_ in sel])
    rows.append(["N"] + [str(int(reg.loc[m,"N"])) for m,_,_ in sel]); rows.append(["R-squared"] + [f(reg.loc[m,"R2"]) for m,_,_ in sel])
    T["cs"] = mdtable(pd.DataFrame(rows, columns=["Dependent variable (P2)"] + [f"{a}, {b}" for _,a,b in sel]),
        "Table 6. Cross-sectional regressions of P2 tail measures on FLD groups",
        "Coefficients only; control group FLD 0 is the omitted category. P1 controls: P1 beta (P1 ES for the ES model), log market value and leverage at the end of P1; the controls require SEC balance-sheet data and a P1 price history, which nine firms lack: CoreWeave and Cerebras listed after P1; SoftBank is an unsponsored ADR with no SEC filings; ASML and TSMC report under IFRS without a comparable diluted-share concept; and Berkshire Hathaway, Visa, Verizon and ExxonMobil are missing one of the two inputs. N falls from 69 to 60, and FLD 3 contains only AMD and Oracle. P2 controls: contemporaneous P2 beta and annualised volatility, and log market value. We do not report HC3 t-statistics for the FLD-3 dummy: with two observations on the dummy and six regressors the two residuals nearly cancel, so the HC3 standard error (0.71 for MES) falls below the classical one (1.11) and overstates precision. Inference is the permutation p-value, which reshuffles the FLD-3 label among AI firms over 2,000 draws and is unaffected by that problem. The FLD 3 - FLD 1 p-value is reported for completeness and shares the HC3 limitation.")
    pb = pd.read_csv(OUT/"tail_beta_panel.csv")
    pb["excess_tail_beta"] = pb.excess_tail_beta.map(f); pb["t"] = pb.t.map(f); pb["p"] = pb.p.map(lambda v: f(v,3))
    pb["n_firms"] = pb.n_firms.map(lambda v: "" if pd.isna(v) else str(int(v)))
    pb["period"] = pb.period.map(short); pb = pb[["period","group","excess_tail_beta","t","p","n_firms"]]; pb.columns = ["Period","Group","Excess tail beta","t","p","Firms"]
    T["panel"] = mdtable(pb, "Table 7. Excess tail beta by FLD group (daily panel)", "r_it = a_i + b_i AI_t + sum_g d_g AI_t x Tail_t x 1{FLD = g}; firm-specific intercepts and betas; tail days are the system's worst 5% within the period; standard errors clustered by date. 'Drop' rows re-estimate the FLD-3 coefficient leaving out one firm. With group-specific tail-day intercepts added, the P2 FLD-3 coefficient is -0.66 (p = 0.081) and FLD 3 - FLD 1 is -0.76 (p = 0.002).")
    evf = pd.read_csv(OUT/"event_study_firm.csv"); evf["absCAR"] = evf.CAR.abs(); evf["signedCAR"] = np.where(evf.exp_sign=="+", evf.CAR, -evf.CAR)
    m1 = smf.ols("signedCAR ~ C(fld)", evf).fit(cov_type="cluster", cov_kwds={"groups": evf.date})
    m2 = smf.ols("absCAR ~ C(fld) + sd", evf).fit(cov_type="cluster", cov_kwds={"groups": evf.date})
    rows = []
    for k, lab in [("Intercept","Intercept (FLD 0)"),("C(fld)[T.1]","FLD 1"),("C(fld)[T.2]","FLD 2"),("C(fld)[T.3]","FLD 3"),("sd","Residual volatility")]:
        rows.append([lab, f"{m1.params[k]:.2f} ({m1.pvalues[k]:.3f})" if k in m1.params else "", f"{m2.params[k]:.2f} ({m2.pvalues[k]:.3f})"])
    W = pd.read_csv(OUT/"audit_event_wild_bootstrap.csv").set_index("model")
    rows.append(["FLD 3, wild-cluster bootstrap p", f(W.loc["signed CAR","wild_cluster_p"],3), f(W.loc["abs CAR","wild_cluster_p"],3)])
    rows.append(["Observations", str(int(m1.nobs)), str(int(m2.nobs))]); rows.append(["Events", "15", "15"])
    T["event"] = mdtable(pd.DataFrame(rows, columns=["","Signed CAR[-1,+1] (%)","Absolute CAR[-1,+1] (%)"]), "Table 8. Pooled event-study regressions",
        "p-values in parentheses, standard errors clustered by event. Signed CAR multiplies the CAR by -1 for events expected to be negative for lab suppliers (DeepSeek, OpenAI revenue report, S-1/A commitments).")
    vb_all = pd.read_csv(OUT/"var_backtests.csv"); vb = vb_all[vb_all.model.isin(["HS","EWMA","GARCH-t","FHS","QR-state"])]
    rows = []
    for port, lab in [("FLD3_basket","FLD-3 basket"),("FLD3_fixed","FLD-3 basket, fixed members"),("AI_IDX","AI-complex index"),("FLD2_basket","FLD-2 basket"),("CTL_IDX","Control index")]:
        for m in ["HS","EWMA","GARCH-t","FHS","QR-state"]:
            a = vb[(vb.portfolio==port)&(vb.model==m)&(vb.level.astype(float)==95)].iloc[0]; b = vb[(vb.portfolio==port)&(vb.model==m)&(vb.level.astype(float)==99)].iloc[0]
            rows.append([lab, m, f(a.rate*100), f(a.kupiec_p,3), f(a.cc_p,3), f(b.rate*100), f(b.kupiec_p,3), f(b.cc_p,3), b.basel])
    T["var"] = mdtable(pd.DataFrame(rows, columns=["Portfolio","Model","95% rate (%)","Kupiec p","CC p","99% rate (%)","Kupiec p ","CC p ","Basel zone (99%, last 250 d)"]),
        "Table 9. Out-of-sample VaR backtests, 2 January 2024 to 5 November 2026", "About 715 forecasts per model. Nominal rates are 5% and 1%. CC: Christoffersen conditional coverage. HS: historical simulation (250 days); EWMA: RiskMetrics, lambda = 0.94; GARCH-t and FHS re-estimated every 21 days on 1,000 observations; QR-state: quantile regression on lagged state variables re-estimated quarterly. The fixed-member FLD-3 basket holds Oracle, AMD and SoftBank throughout.")
    es = vb_all[vb_all.model.str.startswith("ES97.5")].copy()
    es["Portfolio"] = es.portfolio.map({"FLD3_basket":"FLD-3 basket","FLD3_fixed":"FLD-3 basket, fixed members","AI_IDX":"AI-complex index","FLD2_basket":"FLD-2 basket","CTL_IDX":"Control index"})
    es["Model"] = es.model.str.replace("ES97.5 McNeil-Frey ", "")
    es = es[["Portfolio","Model","exceed","rate","mf_mean","mf_t","mf_p_boot"]]
    es["rate"] = (es.rate*100).map(f); es["mf_mean"] = es.mf_mean.map(lambda v: f(v,3)); es["mf_t"] = es.mf_t.map(f); es["mf_p_boot"] = es.mf_p_boot.map(lambda v: f(v,3)); es["exceed"] = es.exceed.astype(int)
    es.columns = ["Portfolio","Model","VaR97.5 breaches","Breach rate (%)","Mean std. residual","t","Bootstrap p (one-sided)"]
    T["var"] += "\n" + mdtable(es, "Table 9 (continued). McNeil-Frey backtest of 97.5% expected shortfall",
        "Residual = (return - forecast ES) / forecast volatility on days when the loss exceeds the model's 97.5% VaR (nominal breach rate 2.5%). A negative mean means realised tail losses were worse than forecast; the p-value tests that alternative (5,000 bootstrap draws).")
    # placebo baskets
    pl = pd.read_csv(OUT/"placebo_var_backtests.csv")
    # labels follow 07b's selection
    def plab(name):
        members = re.search(r"\(([^)]*)\)", name)
        members = ", ".join(members.group(1).split(",")) if members else ""
        # this name has no member list
        if name.startswith("FLD3_fixed"):       return "FLD-3 basket, fixed members"
        if name.startswith("PLACEBO matched"):  return f"Placebo, matched on P1 risk ({members})"
        if name.startswith("PLACEBO high-beta"): return f"Placebo, high P1 beta ({members})"
        return "Control index (20 non-AI firms)"
    PLAB = {n: plab(n) for n in pl.portfolio.unique()}
    rows = []
    for port, lab in PLAB.items():
        r95 = pl[(pl.portfolio==port)&(pl.level==95)].set_index("model")
        r99 = pl[(pl.portfolio==port)&(pl.level==99)].set_index("model")
        cells = [lab]
        for m in ["HS","EWMA","GARCH-t","FHS","QR-state"]:
            cells.append(f"{r95.loc[m,'rate']*100:.2f} ({r95.loc[m,'kupiec_p']:.3f}){'*' if r95.loc[m,'fail'] else ''}")
        cells.append(f"{int(r95['fail'].sum())} of 5")
        cells.append(f"{int(r99['fail'].sum())} of 5")
        rows.append(cells)
    T["placebo"] = mdtable(pd.DataFrame(rows, columns=["Portfolio","HS","EWMA","GARCH-t","FHS","QR-state","Rejected at 95%","Rejected at 99%"]),
        "Table 10. Placebo VaR backtests: non-lab-dependent AI baskets, 2 January 2024 to 5 November 2026",
        "95% exceedance rate (%) with Kupiec p-value in parentheses; * marks rejection at 5%. The last two columns count rejections across the five models at each level. The matched placebo holds the three FLD-1 (non-lab-dependent) AI firms closest to the fixed-member FLD-3 basket on P1 beta and P1 expected shortfall; the high-beta placebo holds the three FLD-1 firms with the highest P1 beta. Models, estimation windows and forecast dates are identical to Table 9.")
    rows = []
    for h in [1,5,10]:
        M = pd.read_csv(OUT/f"ml_metrics_h{h}.csv")
        for mname in ["Logit (vol only)","Logit (all)","Random forest","LightGBM (no AI/lab features)","LightGBM (all)"]:
            g_ = lambda s: M[(M.model==mname)&(M["sample"].str.startswith(s))].iloc[0]
            a = g_("All"); rows.append([f"h = {h}", mname, f(a.AUC,3), f(a.PR_AUC,3), f(a.Brier,4), f(g_("P2").AUC,3), f(g_("FLD").AUC,3)])
    T["ml"] = mdtable(pd.DataFrame(rows, columns=["Horizon","Model","AUC","PR-AUC","Brier","AUC, P2 only","AUC, FLD-3 firms"]), "Table 11. Walk-forward prediction of firm tail events, 2024Q1 to 2026Q4",
        "Target: next h-day return below the firm's rolling 500-day 5th percentile of h-day returns. Quarterly re-estimation on an expanding window with an embargo of at least 1.5h days. Base rates are 6.8% (h = 1), 7.4% (h = 5) and 7.5% (h = 10).")
    S = pd.read_csv(OUT/"ml_stress_day_results.csv")
    rows = []
    for _, r_ in S.iterrows():
        if r_.features.startswith("AUC gain"): rows.append([r_.split, r_.model, "Gain from lab features", f(r_.AUC,4), f"[{r_.lo:.4f}, {r_.hi:.4f}]"])
        else:
            ci = f"[{r_.AUC_lo:.3f}, {r_.AUC_hi:.3f}]" if pd.notna(r_.get("AUC_lo")) else ""
            rows.append([r_.split, r_.model, r_.features, f(r_.AUC,3), ci])
    T["stressml"] = mdtable(pd.DataFrame(rows, columns=["Split","Model","Features","AUC","95% CI (day-block bootstrap)"]), "Table 12. Predicting the worst-third losers on AI stress days",
        "Stress days: AI-index return below its rolling 500-day 5th percentile (ex ante). Features measured the previous day. Intervals resample whole trading days, of which the P2 test set has only 18, so the AUC level is bracketed widely even though the gain from the lab features is pinned down tightly.")
    ST = pd.read_csv(OUT/"stress_results.csv")
    p_ = ST[(ST.margin==0.30)].pivot_table(index="ticker", columns=["case","scenario"], values="loss_pct_mcap")
    p_ = p_.sort_values(("base","Extreme"), ascending=False)
    out = pd.DataFrame({"Supplier": p_.index})
    for case in ["base","extended"]:
        for sc in ["Mild","Severe","Extreme"]: out[f"{case.capitalize()}: {sc}"] = p_[(case,sc)].map(lambda v: f(v,1)).values
    agg = pd.read_csv(OUT/"stress_aggregate.csv")
    T["stress"] = mdtable(out, "Table 13. First-round stress losses (% of market value, 5 November 2026; value-loss rate m = 0.30)",
        f"Mild: 10% of commitments cancelled, lab valuations -30%; severe: 25%, -50%; extreme: 50%, -70%. The severe haircut matches the 23.9% of Anthropic's contractual obligations that its S-1/A discloses as subject to exercisable termination rights ($213.8bn of $895.4bn); the extreme case assumes renegotiation reaching contracts with no termination right. Microsoft includes an approximate 23% stake in OpenAI valued at the $852bn March 2026 round. Aggregate loss across the seven suppliers in the extreme scenario: {agg[(agg.scenario=='Extreme')&(agg.margin==0.30)].base.iloc[0]:.1f}% (base) to {agg[(agg.scenario=='Extreme')&(agg.margin==0.50)].extended.iloc[0]:.1f}% (extended, m = 0.50) of combined market value.")
    I = pd.read_csv(OUT/"ipo_cohort.csv")
    I = I[["ticker","ipo_date","offer","first_day_ret","peak_vs_offer","ret_offer_to_now","max_drawdown","ann_vol","beta_ai","days"]]
    for c in ["first_day_ret","peak_vs_offer","ret_offer_to_now","max_drawdown","ann_vol"]: I[c] = I[c].map(lambda v: f(v,1))
    I["beta_ai"] = I.beta_ai.map(f); I["offer"] = I.offer.map(lambda v: f"{v:.0f}")
    I.columns = ["Ticker","IPO date","Offer ($)","First-day return (%)","Peak vs offer (%)","Offer to 5 Nov 2026 (%)","Max drawdown (%)","Annualised vol (%)","Beta to AI index","Trading days"]
    T["ipo"] = mdtable(I, "Table 14. Post-IPO risk profile of AI-infrastructure listings")
    T["fld"] = mdtable(pd.DataFrame([
        ["3","Oracle (ORCL)","Reported ~$300bn OpenAI cloud contract; RPO $664bn (Aug 2026)","Press reports; 10-Q RPO"],
        ["3","CoreWeave (CRWV)","OpenAI contracts up to $22.4bn; Nvidia equity; high leverage","10-K/10-Q"],
        ["3","Cerebras (CBRS)","~750 MW OpenAI inference agreement; high customer concentration","S-1; 10-Q"],
        ["3","AMD (AMD)","6 GW OpenAI supply deal with warrants; Anthropic hardware supply agreement","8-K; Anthropic S-1/A"],
        ["3","SoftBank (SFTBY)","Large OpenAI equity investor and Stargate partner","Company announcements"],
        ["2","Microsoft (MSFT)","OpenAI stake; $250bn Azure commitment; Anthropic $44.9bn","Company announcement; S-1/A"],
        ["2","Alphabet (GOOGL)","Anthropic investor; $121.3bn take-or-pay commitment","S-1/A"],
        ["2","Amazon (AMZN)","Anthropic notes and preferred; $110bn commitment; OpenAI AWS deal","S-1/A; announcements"],
        ["2","Broadcom (AVGO)","Anthropic equipment lease (~$125bn rent); OpenAI 10 GW accelerators","S-1/A; announcement"],
        ["2","Nvidia (NVDA)","Investments in OpenAI and Anthropic; supplier to both","Announcements"],
        ["1","39 firms","Semiconductors, equipment, servers, networking, data centres, power, AI software, AAPL, TSLA, META","Industry classification"],
        ["0","20 firms","KO, PG, JNJ, WMT, PEP, MCD, BRK-B, JPM, XOM, UNH, HD, LLY, COST, V, MA, CVX, MRK, ABBV, T, VZ","Control group"],
    ], columns=["FLD","Firm","Basis for the score","Source"]), "Table A1. Frontier-lab dependence (FLD) coding")
    evg = pd.read_csv(OUT/"event_study_by_fld.csv")
    evg = evg.rename(columns={"date":"Date (day 0)","event":"Event","0":"FLD 0","1":"FLD 1","2":"FLD 2","3":"FLD 3"})
    for c in ["FLD 0","FLD 1","FLD 2","FLD 3"]: evg[c] = evg[c].map(f)
    T["events"] = mdtable(evg, "Table A2. Frontier-lab events and mean CAR[-1,+1] (%) by FLD group", "The 3 November 2026 window overlaps with the US mid-term elections.")
    G = pd.read_csv(OUT/"gsadf_results.csv")
    G = G[["series","T","gsadf","cv95","cv99","last_bsadf","last_cv95","episodes"]].fillna("")
    for c in ["gsadf","cv95","cv99","last_bsadf","last_cv95"]: G[c] = G[c].map(f)
    G["episodes"] = G.episodes.str.replace("→"," to ")
    G.columns = ["Series","Weeks","GSADF","CV 95%","CV 99%","Last BSADF","Last CV 95%","Date-stamped episodes"]
    D = pd.read_csv(OUT/"merton_dd.csv"); D = D[D.ticker.isin(["ORCL","CRWV","IREN","CIFR","CORZ","MSFT","AMZN","META"])].groupby("ticker").tail(1)
    D = D[["ticker","date","equity_bn","debt_point_bn","sigmaE","DD","PD"]]
    D["equity_bn"] = D.equity_bn.map(lambda v: f"{v:.1f}"); D["debt_point_bn"] = D.debt_point_bn.map(f); D["sigmaE"] = D.sigmaE.map(f); D["DD"] = D.DD.map(f); D["PD"] = D.PD.map(lambda v: f"{v*100:.1f}%")
    D.columns = ["Ticker","Balance-sheet date","Equity ($bn)","Default point ($bn)","Equity vol","DD","Merton PD"]
    T["gsadf_dd"] = mdtable(G, "Table A3. PSY bubble tests (weekly log prices) and latest Merton distance to default",
        "Critical values from 499 Monte Carlo replications. Episodes last at least log(T) weeks.") + "\n" + mdtable(D, "Table A3 (continued). Latest Merton distance to default")
    return T

def renumber_figures(md):
    order = []
    for m in re.finditer(r"!\[\]\((fig\d+_[^)]+)\)|Figure (\d+)", md):
        key = m.group(1) or f"fig{int(m.group(2)):02d}"
        k2 = key[:5]
        if k2 not in order: order.append(k2)
    newnum = {k: i+1 for i, k in enumerate(order)}
    md = re.sub(r"Figure (\d+)", lambda m: f"Figure {newnum.get(f'fig{int(m.group(1)):02d}', m.group(1))}", md)
    def img(m):
        fn = m.group(1); n = newnum[fn[:5]]
        # pandoc resolves via --resource-path
        return f"![Figure {n}. {CAPTIONS[fn]}]({fn}){{width=6.3in}}"
    md = re.sub(r"!\[\]\((fig\d+_[^)]+)\)", img, md)
    return md, newnum

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
def clean_dashes(md):
    body, sep, refs = md.partition("# References")
    body = re.sub(rf"({MONTHS})–({MONTHS})", r"\1 to \2", body)
    body = body.replace("—", ", ").replace("–", "-")
    return body + sep + refs

def read(name): return (SRC / name).read_text()

def fi_title_page():
    """Separate title page for Financial Innovation: author details, Short Bio and Declarations."""
    decl = read("12_declarations_springer.md").split("# Declarations", 1)[1]
    return "\n\n".join([
        f"# {TITLE}", f"{AUTHOR}^1^",
        f"^1^ {AFFIL}", f"Corresponding author: {AUTHOR}, {EMAIL}",
        "## Author's information (Short Bio)", BIO,
        "# Declarations" + decl,
    ])

def assemble(variant, T):
    # FI reviews double-anonymous
    anonymous = variant in ("jor","iimb","vikalpa","fi")
    parts = []
    fm = [f"# {TITLE}", ""]
    if not anonymous:
        fm += [f"{AUTHOR}^1^", "", f"^1^ {AFFIL}. Corresponding author: {EMAIL}", ""]
    if variant == "iimb":
        fm += ["## Highlights", ""] + [f"- {h}" for h in HIGHLIGHTS] + [""]
    fm += ["## Abstract", "", ABSTRACT, "", f"Keywords: {'; '.join(KEYWORDS)}", "", f"JEL classification: {JEL}", ""]
    if variant == "vikalpa": fm += ["## Executive summary", "", vik_trim("00_exec_summary.md", read("00_exec_summary.md")), ""]
    if variant == "report": fm += ["## Executive summary", "", read("00_exec_summary.md"), ""]
    parts.append("\n".join(fm))
    secs = ["01_intro_short.md" if variant=="vikalpa" else "01_intro.md", "02_literature_short.md" if variant=="vikalpa" else "02_literature.md","03_background.md","04_hypotheses_data.md",
            "06_methods_short.md" if variant=="vikalpa" else "06_methods.md",
            "07_results.md","08_implications.md","09_robustness.md"]
    for s_ in secs:
        txt = read(s_)
        if variant == "vikalpa": txt = vik_trim(s_, txt)
        if variant != "iimb" and variant != "report" and s_ == "08_implications.md":
            txt = re.sub(r"## 8\.4 Relevance for investors outside the US.*", "", txt, flags=re.S)
        parts.append(txt)
    if variant == "fi":
        parts.append(read("12_declarations_springer.md").split("# Declarations", 1)[0].rstrip())
        parts.append(read("12b_additional_files.md"))
    if variant == "report":
        parts.append(read("12_declarations_report.md"))
    parts.append(read("10_references.md"))
    parts.append(read("11_appendix.md"))
    if variant == "report": parts.append(read("13_report_appendix.md"))
    md = "\n\n".join(parts)
    for k, v in T.items(): md = md.replace("{{TAB:" + k + "}}", v)
    md, _ = renumber_figures(md)
    if variant == "vikalpa": md = md.replace("{{TAB:extra}}", "")
    return clean_dashes(md)

def vik_trim(name, txt):
    """Vikalpa: <= 6,000 words. Drop credit-risk subsection and the IPO-cohort table detail, shorten literature."""
    if name == "07_results.md":
        txt = re.sub(r"## 7\.7 Credit risk.*?(?=## 7\.8)", "", txt, flags=re.S)
        txt = txt.replace("## 7.8 Commitment", "## 7.7 Commitment").replace("## 7.9 The AI IPO", "## 7.8 The AI IPO")
        txt = re.sub(r"\n!\[\]\(fig10_shap\.png\)\n", "\n", txt); txt = re.sub(r" The SHAP attribution \(Figure 10\)[^.]*\.", "", txt)
        txt = re.sub(r"\n!\[\]\(fig15_bsadf\.png\)\n", "\n", txt)
        txt = re.sub(r"\n!\[\]\(fig07_fld3_var_path\.png\)\n", "\n", txt); txt = txt.replace(" Figure 7 plots the FLD-3 basket's daily returns against its GARCH-t and HS forecasts.", "")
    if name == "07_results.md":
        txt = re.sub(r" In the extended P2 system CoreWeave is.*?\(-?−?23\.4\)\.", "", txt)
        txt = txt.replace(" The FLD score was coded with information available in 2026 and applied to earlier stress days, which should favour it; that it still adds nothing strengthens this conclusion.", "")
        txt = re.sub(r"Three details explain where the models go wrong\..*?slow to absorb\.", "Breaches do not cluster (Christoffersen's independence test never rejects), and the McNeil-Frey test does not reject 97.5% ES for the FLD-3 basket, so the models miss how often large losses occur rather than how large they are once they occur.", txt, flags=re.S)
        txt = re.sub(r"Recent AI-infrastructure listings show what public investors buy \(Table 13, Figure 16\)\..*?first day of trading\.", "Arm, CoreWeave and Cerebras had first-day returns of 0% to 68%, drawdowns of 62% to 76% from their post-IPO peaks, annualised volatility of 75% to 112% and AI-index betas of 1.2 to 1.7 (Table 13, Figure 16). They behaved as levered AI-factor claims from the first day of trading.", txt, flags=re.S)
    if name == "07_results.md":
        txt = re.sub(r"Oracle's RPO disclosure produced a mean FLD-3 CAR of \+11\.1%.*?produced -?−?6\.3%\. ", "Oracle's RPO disclosure produced a mean FLD-3 CAR of +11.1%. ", txt, flags=re.S)
        txt = re.sub(r"Their equity carried a claim on the labs' expected growth.*?at once\.", "Their equity carried a claim on the labs' expected growth, so a stress test built only from signed contracts understates exposure.", txt, flags=re.S)
    if name == "07_results.md":
        # practitioner variants: conclusion only
        txt = re.sub(r"We base inference on the permutation test rather than on the regression standard errors,.*?gives p-values of 0\.067 for MES, 0\.168 for ES and 0\.102 for tail dependence\. None of the three is significant at 5%; MES is significant at 10%\.",
            "Inference rests on a permutation test that reshuffles the FLD-3 label among AI firms, because only AMD and Oracle carry the FLD-3 dummy in this sample and regression standard errors on a two-observation dummy overstate precision. The permutation p-values are 0.067 for MES, 0.168 for ES and 0.102 for tail dependence, so none is significant at 5% and only MES is significant at 10%.", txt, flags=re.S)
        # compress prose, keep the table
        txt = re.sub(r"The comparison that identifies H4 is not the control index,.*?control index still passes at 95%\.",
            "The control index differs from the lab-dependent basket in sector, beta and volatility at once, so it cannot separate lab dependence from AI-factor exposure. Table 10 therefore repeats the exercise on two baskets of AI firms with no material lab contract, one matched to the fixed-member FLD-3 basket on P1 beta and expected shortfall and one holding the highest-beta FLD-1 firms, using identical models and forecast dates. Both placebos fail: the matched basket for 1 of 5 models at 95% and 3 of 5 at 99%, the high-beta basket for 2 of 5 and 4 of 5, the latter worse than the FLD-3 basket's 3 of 5, while the control index still passes at 95%. Against 500 random FLD-1 baskets the FLD-3 breach rate sits at the 83rd percentile, inside the AI distribution rather than outside it.", txt, flags=re.S)
        txt = re.sub(r"We therefore accept H4 as a statement about AI-infrastructure equity.*?not the counterparty\.",
            "We therefore accept H4 as a statement about AI-infrastructure equity and reject the stronger reading that lab dependence is what the models miss. Standard VaR models under-covered AI suppliers whether or not they held frontier-lab contracts, while covering non-AI equities adequately, so the failure tracks the AI factor's volatility regime. For a risk manager the unit of validation is the AI factor exposure, not the counterparty.", txt, flags=re.S)
    if name == "00_exec_summary.md":
        # Vikalpa repeats its setup
        paras = [p for p in txt.split("\n\n") if p.strip()]
        lead = ("OpenAI and Anthropic, the two leading frontier AI laboratories, are moving towards public listing, and both have "
                "signed multi-year compute contracts with listed cloud, chip and data-centre companies that book them as backlog. "
                "Anthropic's registration statement shows the asymmetry: about $895 billion of committed spend, $682 billion of it "
                "non-cancelable, against trailing revenue of about $35 billion and contracted customer revenue of $13.8 billion. "
                "Commitments are 25 times revenue. Customers can cut their spending quickly; the lab cannot. For the suppliers the "
                "backlog is therefore a loan-like exposure to a customer whose ability to pay depends on the same AI demand that "
                "drives the suppliers' own share prices, which risk managers call wrong-way risk. We examine whether this shows up "
                "in market data, whether standard risk models capture it and whether machine learning can predict it, using 69 "
                "US-traded firms from January 2021 to November 2026.")
        keep = [p for p in paras if p.startswith("Four results matter") or p.startswith("The implications are direct")]
        txt = "\n\n".join([lead] + keep)
    if name == "03_background.md":
        txt = re.sub(r"The commitments are not uniformly unconditional, and the filing says so\..*?carry no termination right\.",
            "The commitments are not uniformly unconditional. The filing discloses that about $213.8 billion, roughly 24% of the total, sits in arrangements subject to exercisable termination rights, leaving $681.6 billion non-cancelable, or 19 times trailing revenue. The wrong-way exposure should be read against that non-cancelable balance, and the cancelable share is what calibrates the haircut in our stress test.", txt, flags=re.S)
    if name == "04_hypotheses_data.md":
        txt = re.sub(r" We use an ordinal score rather than a dollar exposure because.*?Section 7\.8\.", " An ordinal score is used because several OpenAI contract values are not disclosed in filings.", txt, flags=re.S)
        txt = re.sub(r" We test this against two comparisons: non-AI control firms,.*?AI-factor exposure\.", " We test this against both non-AI control firms and placebo baskets of AI firms without material lab contracts matched on pre-period risk, which tells us whether any shortfall is specific to lab dependence.", txt, flags=re.S)
    if name == "09_robustness.md" and "Fourth, the universe" in txt:
        txt = re.sub(r"Fourth, the universe was chosen in 2026.*?with that in mind\.", "Fourth, the universe was chosen in 2026 with hindsight, SoftBank enters through an over-the-counter ADR, and several specifications are reported, so p-values near 0.05 should be read with care.", txt, flags=re.S)
    if name == "08_implications.md":
        txt = re.sub(r"## 8\.3 For suppliers, lenders and regulators\n\n.*?(?=## 8\.4|$)", "## 8.3 For suppliers, lenders and regulators\n\nSuppliers should treat lab backlog as a credit exposure with wrong-way characteristics and disclose the share of RPO owed by any single customer above 10% of RPO. Supervisors should not rely on correlation-based dashboards alone: average connectedness fell in P2 while tail risk relative to the market doubled.\n\n", txt, flags=re.S)
    if name == "01_intro.md":
        txt = re.sub(r"\n!\[\]\(fig13_rpo\.png\)\n", "\n", txt); txt = txt.replace(" (Figure 13)", "")
    if name == "09_robustness.md":
        txt = re.sub(r"Merton distances to default use book debt.*?estimate\.\n", "", txt, flags=re.S)
        txt = re.sub(r"Several checks are reported in the text:.*?Three further limits apply\.", "Three limits apply.", txt, flags=re.S)
    if name == "03_background.md":
        txt = re.sub(r"## 3\.3 Timeline.*", "", txt, flags=re.S)
    if name == "07_results.md":
        txt = re.sub(r"The bubble tests tell a related story.*?not a run-up\.\n", "", txt, flags=re.S)
    return txt

REFDOC = DR / "reference.docx"
CSS = DR / "print.css"
CSS.write_text("""
body{font-family:'Times New Roman',Times,'Liberation Serif',serif;font-size:11pt;line-height:1.5;max-width:none;margin:0;color:#000}
h1{font-size:14pt;margin:18pt 0 6pt} h2{font-size:12pt;margin:14pt 0 4pt} h3{font-size:11pt}
h1:first-of-type{font-size:16pt;line-height:1.25}
p{text-align:justify;margin:0 0 7pt}
table{border-collapse:collapse;width:100%;font-size:8.5pt;line-height:1.25;margin:4pt 0 4pt;page-break-inside:auto}
th{border-top:1px solid #000;border-bottom:1px solid #000;padding:2pt 4pt;text-align:right;font-weight:bold}
td{padding:1.5pt 4pt;text-align:right;vertical-align:top} th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
tbody tr:last-child td{border-bottom:1px solid #000}
tr{page-break-inside:avoid}
figure{margin:8pt 0;text-align:center;page-break-inside:avoid} figure img{max-width:100%} figcaption{font-size:9.5pt;font-style:italic;text-align:left}
math{font-size:105%}
""")
def make_refdoc():
    from docx import Document
    from docx.shared import Pt
    from docx.enum.text import WD_LINE_SPACING
    data = subprocess.run(["pandoc", "--print-default-data-file", "reference.docx"], capture_output=True).stdout
    REFDOC.write_bytes(data); d = Document(REFDOC)
    for st in d.styles:
        try:
            if st.type == 1:
                st.font.name = "Times New Roman"
                if st.name in ("Normal","Body Text","First Paragraph","Compact","Abstract","Bibliography"): st.font.size = Pt(12)
                if st.name.startswith("Heading"): st.font.size = Pt(13 if st.name.endswith("1") else 12); st.font.bold = True
                if st.name in ("Body Text","First Paragraph","Abstract","Bibliography"):
                    st.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
                if st.name == "Title": st.font.size = Pt(16)
                try:
                    from docx.shared import RGBColor; st.font.color.rgb = RGBColor(0,0,0)
                except Exception: pass
        except Exception: pass
    d.save(REFDOC)

def fi_format(docx, line_numbers=True):
    """Double spacing, line numbers, page numbers."""
    from docx import Document
    from docx.enum.text import WD_LINE_SPACING, WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    d = Document(str(docx))
    for st in d.styles:
        try:
            if st.type == 1 and st.name in ("Normal","Body Text","First Paragraph","Compact","Abstract","Bibliography"):
                st.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
        except Exception:
            pass
    for sec in d.sections:
        sectPr = sec._sectPr
        if line_numbers and sectPr.find(qn("w:lnNumType")) is None:
            ln = OxmlElement("w:lnNumType")
            ln.set(qn("w:countBy"), "1"); ln.set(qn("w:restart"), "continuous"); ln.set(qn("w:distance"), "360")
            # schema order: after pgMar
            anchor = sectPr.find(qn("w:cols"))
            sectPr.insert(list(sectPr).index(anchor), ln) if anchor is not None else sectPr.append(ln)
        par = sec.footer.paragraphs[0]; par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if not par.runs:
            run = par.add_run()
            for tag, attr, text in (("w:fldChar", "begin", None), ("w:instrText", None, "PAGE"), ("w:fldChar", "end", None)):
                el = OxmlElement(tag)
                if attr: el.set(qn("w:fldCharType"), attr)
                if text is not None: el.set(qn("xml:space"), "preserve"); el.text = text
                run._r.append(el)
    d.save(str(docx))

def export(md, outdir, stem):
    outdir.mkdir(parents=True, exist_ok=True)
    mdp = outdir / f"{stem}.md"; mdp.write_text(md)
    docx = outdir / f"{stem}.docx"
    subprocess.run(["pandoc", str(mdp), "-o", str(docx), "--reference-doc", str(REFDOC), "--resource-path", str(FIG)], check=True)
    html = outdir / f"{stem}.html"
    subprocess.run(["pandoc", str(mdp), "-s", "--mathml", "--embed-resources", "--resource-path", str(FIG), "-c", str(CSS),
                    "--metadata", f"pagetitle={RUNNING}", "-o", str(html)], check=True)
    subprocess.run(["node", str(Path(__file__).with_name("html2pdf.js")), str(html), str(outdir / f"{stem}.pdf")], check=True, timeout=300)
    html.unlink()
    body = md.split("# References")[0]
    body = re.sub(r"\|.*\|", " ", body); body = re.sub(r"!\[.*?\]\(.*?\)\{.*?\}", " ", body)
    return len(re.findall(r"\b\w[\w'.%-]*\b", body))

if __name__ == "__main__":
    make_refdoc(); T = build_tables()
    targets = {"report": (DR/"01_Full_Research_Report", "AI_Lab_Dependence_Risk_Full_Research_Report"),
               "fi":     (DR/"02_Financial_Innovation", "Manuscript_Financial_Innovation"),
               "jor":    (DR/"03_Journal_of_Risk", "Manuscript_Journal_of_Risk_ANONYMISED"),
               "iimb":   (DR/"04_IIMB_Management_Review", "Manuscript_IIMB_Management_Review_ANONYMISED"),
               "vikalpa":(DR/"05_Vikalpa", "Manuscript_Vikalpa_ANONYMISED")}
    import sys
    which = sys.argv[1:] or list(targets)
    for v in which:
        od, stem = targets[v]
        wc = export(assemble(v, T), od, stem)
        if v == "fi":
            fi_format(od / f"{stem}.docx")
            export(fi_title_page(), od, "Title_Page_Financial_Innovation")
            fi_format(od / "Title_Page_Financial_Innovation.docx", line_numbers=False)
        print(v, "words in main text (excl. tables, references):", wc)
