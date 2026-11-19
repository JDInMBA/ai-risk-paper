"""
13_audit_checks.py - independent checks of the core estimators against reference
implementations or known answers. Each check prints PASS/FAIL with the numbers compared.
"""
import warnings; warnings.filterwarnings("ignore")
import importlib.util, numpy as np, pandas as pd, statsmodels.api as sm
from scipy.stats import t as tdist, norm, chi2
from arch import arch_model
def load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py",""), name)
    src = open(name).read()
    return src
rng = np.random.default_rng(0); out = []
def chk(name, ok, detail): out.append((name, "PASS" if ok else "FAIL", detail)); print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

# 1. GFEVD vs orthogonalised FEVD
# anchor on code, not comments
_src3 = open("03_tail_risk.py").read()
ns = {}; exec(_src3.split("px = load_prices()")[0] +
              "def gfevd_spillover" + _src3.split("def gfevd_spillover")[1].split("R = r[CORE_AI]")[0], ns)
k, T = 4, 20000; A = np.diag([0.3,-0.2,0.1,0.25]); A[0,1] = 0.2; A[2,3] = -0.15
X = np.zeros((T,k))
for t in range(1,T): X[t] = A @ X[t-1] + rng.standard_normal(k) * np.array([1,2,0.5,1.5])
tot, net, th = ns["gfevd_spillover"](X, p=1, H=10)
from statsmodels.tsa.api import VAR
fe = VAR(X).fit(1).fevd(10).decomp[:, -1, :]
chk("Diebold-Yilmaz GFEVD vs statsmodels FEVD (diagonal Sigma)", np.allclose(th, fe, atol=0.01), f"max abs diff {np.abs(th-fe).max():.4f}")

# 2. rolling DF vs statsmodels
ns2 = {"np": np}; exec("def bsadf_seq" + open("04_bubble_credit.py").read().split("def bsadf_seq")[1].split("def mc_cv")[0], ns2)
y = np.cumsum(rng.standard_normal(300)); r0 = 0.2
bs = ns2["bsadf_seq"](y, r0)
w0 = int(np.floor(r0*299)); b = 250
best = -np.inf
for a in range(0, b - w0 + 1):
    dy = np.diff(y)[a:b]; yl = y[:-1][a:b]
    m = sm.OLS(dy, sm.add_constant(yl)).fit(); best = max(best, m.tvalues[1])
chk("BSADF closed form vs statsmodels OLS sup-t", abs(best - bs[b-1]) < 1e-8, f"{best:.6f} vs {bs[b-1]:.6f}")

# 3. EVT VaR vs true quantile
from scipy.stats import genpareto
x = tdist.rvs(4, size=200000, random_state=1) * 0.01
L = -x; u = np.quantile(L, .90); exc = L[L>u]-u; xi, _, beta = genpareto.fit(exc, floc=0)
var = u + beta/xi*(((len(L)/len(exc))*0.01)**(-xi)-1); true = -tdist.ppf(0.01, 4)*0.01
chk("EVT POT 99% VaR on t(4) sample vs true quantile", abs(var/true-1) < 0.02, f"{var:.5f} vs {true:.5f}, xi={xi:.3f} (theory 0.25)")

# 4. Merton KMV iteration
ns4 = {"np": np, "norm": norm}; exec("def merton_dd" + open("04_bubble_credit.py").read().split("def merton_dd")[1].split("DD_FIRMS")[0], ns4)
E, sE, D, rr = 100.0, 0.6, 80.0, 0.04
dd, pdv = ns4["merton_dd"](E, sE, D, r=rr)
# solve the Merton equations
from scipy.optimize import fsolve
def eqs(z):
    V, s = z; d1 = (np.log(V/D)+(rr+.5*s*s))/s; d2 = d1-s
    return [V*norm.cdf(d1)-D*np.exp(-rr)*norm.cdf(d2)-E, norm.cdf(d1)*V*s/E - sE]
V, s = fsolve(eqs, [E+D, sE*E/(E+D)]); dd2 = (np.log(V/D)+(rr-.5*s*s))/s
chk("Merton DD iterative vs direct solution", abs(dd-dd2) < 1e-3, f"{dd:.4f} vs {dd2:.4f}")

# 5. Kupiec and Christoffersen
ns5 = {"np": np, "chi2": chi2}; src7 = open("07_var_backtest.py").read()
exec("def kupiec" + src7.split("def kupiec")[1].split("def tick")[0], ns5)
I = np.array([0]*95 + [1]*5); lr, p = ns5["kupiec"](pd.Series(I), 0.05)
chk("Kupiec LR = 0 when rate equals nominal", abs(lr) < 1e-10, f"LR={lr:.2e}")
I2 = np.array([0]*90 + [1]*10); lr2, _ = ns5["kupiec"](pd.Series(I2), 0.05)
hand = -2*(90*np.log(.95)+10*np.log(.05) - 90*np.log(.9) - 10*np.log(.1))
chk("Kupiec LR hand computation (10/100 at 5%)", abs(lr2-hand) < 1e-10, f"{lr2:.4f} vs {hand:.4f}")

# 6. forward-target alignment
x = pd.Series(np.arange(1, 21, dtype=float)); H = 3
fwd = x[::-1].rolling(H).sum()[::-1].shift(-1)
chk("Forward H-day target uses t+1..t+H only", fwd.iloc[0] == 9 and fwd.last_valid_index() == len(x)-H-1, f"fwd[0]={fwd.iloc[0]} (expect 9), last valid index {fwd.last_valid_index()} (expect {len(x)-H-1})")

# 7. GARCH one-step recursion
y = pd.Series(tdist.rvs(6, size=1200, random_state=3))
am = arch_model(y[:1000], mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
mu, om, a1, b1 = am.params[["mu","omega","alpha[1]","beta[1]"]]
s2 = om + a1*am.resid.iloc[-1]**2 + b1*am.conditional_volatility.iloc[-1]**2
f = am.forecast(horizon=1).variance.iloc[-1,0]
chk("GARCH one-step variance recursion vs arch forecast", abs(s2-f) < 1e-8, f"{s2:.6f} vs {f:.6f}")

# 8. Student-t ES formula
nu, a = 5.0, 0.025; sc = np.sqrt((nu-2)/nu); q = tdist.ppf(a, nu)
es_formula = -(tdist.pdf(q, nu)/a)*((nu+q**2)/(nu-1))*sc
z = tdist.rvs(nu, size=4_000_000, random_state=5)*sc; es_mc = z[z <= np.quantile(z, a)].mean()
chk("Standardised-t ES formula vs Monte Carlo", abs(es_formula-es_mc) < 0.01, f"{es_formula:.4f} vs {es_mc:.4f}")
pd.DataFrame(out, columns=["check","result","detail"]).to_csv("../results/audit_unit_checks.csv", index=False)
