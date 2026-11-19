"""05b_robustness.py - leave-one-out and permutation tests for the FLD3 cross-sectional effect."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from common import *
cs = pd.read_csv(OUT / "cross_section_P2.csv", index_col=0)
specs = {"MES": "mes ~ fld3 + fld2 + fld1 + beta_P1 + logmcap + lev",
         "ES5": "es5 ~ fld3 + fld2 + fld1 + es5_P1 + logmcap + lev",
         "LTD": "ltd ~ fld3 + fld2 + fld1 + beta_P1 + logmcap + lev"}
rows = []
for nm, f in specs.items():
    d = cs.dropna(subset=["beta_P1","logmcap","lev","es5_P1"])
    base = smf.ols(f, d).fit(cov_type="HC3")
    rows.append(dict(spec=nm, drop="none", coef=base.params.fld3, t=base.tvalues.fld3))
    for t in d.index[d.fld3 == 1]:
        dd_ = d.drop(t)
        m = smf.ols(f, dd_).fit(cov_type="HC3")
        # HC3 undefined at leverage 1
        n3 = int(dd_.fld3.sum())
        tval = m.tvalues.fld3 if n3 > 1 else np.nan
        rows.append(dict(spec=nm, drop=t, coef=m.params.fld3, t=tval,
                         note="SE undefined (single FLD3 obs, leverage=1)" if n3 <= 1 else ""))
    # permutation over AI firms
    rng = np.random.default_rng(11); ai = d[d.fld >= 1].index; k = int(d.fld3.sum()); perm = []
    for _ in range(2000):
        dd = d.copy(); pick = rng.choice(ai, k, replace=False)
        dd["fld3"] = dd.index.isin(pick).astype(int)
        dd["fld1"] = ((dd.fld >= 1) & (dd.fld3 == 0) & (dd.fld2 == 0)).astype(int)
        dd.loc[pick, "fld2"] = 0
        perm.append(smf.ols(f, dd).fit().params.fld3)
    perm = np.array(perm); obs = base.params.fld3
    p = (np.abs(perm) >= abs(obs)).mean()
    rows.append(dict(spec=nm, drop="PERMUTATION p-value (two-sided, 2000 draws)", coef=obs, t=p))
R = pd.DataFrame(rows); R.to_csv(OUT / "cs_robustness.csv", index=False); print(R.round(3).to_string())
