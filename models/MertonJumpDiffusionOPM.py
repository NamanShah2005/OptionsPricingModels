import os
import bisect
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from scipy.stats  import norm
from scipy.optimize import minimize, brentq
import QuantLib as ql

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0. PATHS & FOLDERS
# ─────────────────────────────────────────────────────────────────────────────
output_folder = "graphs"
os.makedirs(output_folder, exist_ok=True)

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "aapl_2016_2020.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, low_memory=False)
df.columns = [c.strip().strip("[]") for c in df.columns]
df["QUOTE_DATE"]  = df["QUOTE_DATE"].str.strip()
df["EXPIRE_DATE"] = df["EXPIRE_DATE"].str.strip()

available_dates = sorted(df["QUOTE_DATE"].unique())

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATE INPUT
# ─────────────────────────────────────────────────────────────────────────────
while True:
    print(f"\nAvailable: {available_dates[0]}  →  {available_dates[-1]}")
    QUOTE_DATE = input("Enter a quote date (YYYY-MM-DD): ").strip()
    if QUOTE_DATE not in available_dates:
        idx   = bisect.bisect_left(available_dates, QUOTE_DATE)
        hints = available_dates[max(0, idx - 2) : idx + 3]
        print(f"  ⚠  '{QUOTE_DATE}' not found. Nearby dates: {hints}")
        continue
    break

df_day = df[df["QUOTE_DATE"] == QUOTE_DATE].copy()
df_day["C_IV"] = pd.to_numeric(df_day["C_IV"], errors="coerce")
spot = float(df_day["UNDERLYING_LAST"].iloc[0])
_yr, _mo, _dy = [int(x) for x in QUOTE_DATE.split("-")]

print(f"\nQuote date : {QUOTE_DATE}")
print(f"Spot price : {spot:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. RATE TABLES  (FRED 1-yr CMT  +  AAPL dividend yield)
# ─────────────────────────────────────────────────────────────────────────────
_TREASURY_1Y = {
    2015: 0.0065, 2016: 0.0059, 2017: 0.0130,
    2018: 0.0238, 2019: 0.0224, 2020: 0.0016, 2021: 0.0005,
}
_AAPL_DIV = {
    2015: 0.0171, 2016: 0.0200, 2017: 0.0158,
    2018: 0.0147, 2019: 0.0127, 2020: 0.0077, 2021: 0.0065,
}

def _interp(table, year, month):
    t    = year + (month - 1) / 12.0
    y_lo = int(t);  y_hi = y_lo + 1
    vals = list(table.values())
    r_lo = table.get(y_lo, vals[0]);  r_hi = table.get(y_hi, vals[-1])
    return r_lo + (t - y_lo) * (r_hi - r_lo)

r_f = _interp(_TREASURY_1Y, _yr, _mo)
q   = _interp(_AAPL_DIV,    _yr, _mo)

print(f"\nRisk-free rate : {r_f*100:.4f}%")
print(f"Dividend yield : {q*100:.4f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 4. QUANTLIB  –  VOL SURFACE
# ─────────────────────────────────────────────────────────────────────────────
day_count        = ql.Actual365Fixed()
calendar         = ql.UnitedStates(ql.UnitedStates.Settlement)
calculation_date = ql.Date(_dy, _mo, _yr)
ql.Settings.instance().evaluationDate = calculation_date

flat_ts     = ql.YieldTermStructureHandle(
                  ql.FlatForward(calculation_date, r_f, day_count))
dividend_ts = ql.YieldTermStructureHandle(
                  ql.FlatForward(calculation_date, q,   day_count))

def str_to_ql(s):
    yr, mo, dy = [int(x) for x in s.split("-")]
    return ql.Date(dy, mo, yr)

raw_expiries = sorted(df_day["EXPIRE_DATE"].unique())
valid_expiries = []
for exp_str in raw_expiries:
    ql_date = str_to_ql(exp_str)
    if ql_date <= calculation_date:
        continue
    sub = (df_day[df_day["EXPIRE_DATE"] == exp_str]
           .copy()
           .pipe(lambda d: d.dropna(subset=["C_IV"]))
           .pipe(lambda d: d[d["C_IV"] > 0])
           .pipe(lambda d: d[(d["STRIKE"] >= spot * 0.80) &
                             (d["STRIKE"] <= spot * 1.20)])
           .sort_values("STRIKE")
           .reset_index(drop=True))
    if len(sub) >= 3:
        valid_expiries.append((exp_str, ql_date, sub))

expiration_dates = [v[1] for v in valid_expiries]
all_strike_sets  = [set(v[2]["STRIKE"].values) for v in valid_expiries]
common_strikes   = sorted(set.intersection(*all_strike_sets))
if len(common_strikes) < 3:
    common_strikes = sorted(set.union(*all_strike_sets))
    common_strikes = [s for s in common_strikes
                      if spot * 0.80 <= s <= spot * 1.20]

print(f"\nExpiries in vol surface : {len(expiration_dates)}")
print(f"Common strikes          : {len(common_strikes)}")

implied_vols_matrix = ql.Matrix(len(common_strikes), len(expiration_dates))
for j, (_, _, sub) in enumerate(valid_expiries):
    iv_by_K = sub.set_index("STRIKE")["C_IV"]
    ks = iv_by_K.index.values.astype(float)
    vs = iv_by_K.values.astype(float)
    for i, K in enumerate(common_strikes):
        implied_vols_matrix[i][j] = float(np.interp(K, ks, vs))

bvs = ql.BlackVarianceSurface(
    calculation_date, calendar,
    expiration_dates, common_strikes,
    implied_vols_matrix, day_count)
bvs.setInterpolation("bicubic")

dte_years = [(ql_date - calculation_date) / 365.0
             for _, ql_date, _ in valid_expiries]

# ─────────────────────────────────────────────────────────────────────────────
# 5. CORE PRICING  —  Black-Scholes building block
# ─────────────────────────────────────────────────────────────────────────────
def bs_call(S, K, r, q, T, sigma):
    """
    Standard Black-Scholes-Merton call price with continuous dividends.
    Equation (1) of Merton 1973.
    """
    if sigma <= 0 or T <= 0:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    sq = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sq
    d2 = d1 - sq
    return (S * np.exp(-q * T) * norm.cdf(d1)
            - K * np.exp(-r * T) * norm.cdf(d2))

# ─────────────────────────────────────────────────────────────────────────────
# 6. MERTON (1976) CLOSED-FORM SERIES  —  FAITHFUL IMPLEMENTATION
# ─────────────────────────────────────────────────────────────────────────────
_MAX_TERMS  = 50
_EPS_WEIGHT = 1e-12

def merton_call(S, K, r, q, T, sigma, lam, mu_J, sigma_J):
    kappa_bar  = np.exp(mu_J + 0.5 * sigma_J**2) - 1.0   # E[J] - 1
    lam_prime  = lam * (1.0 + kappa_bar)                  # λ' = λ·E[J]
    lam_drift  = lam * kappa_bar                           # drift correction

    lam_prime_T = lam_prime * T                            # λ'T  (Poisson mean)

    price  = 0.0
    log_w  = -lam_prime_T                                  # log w_0

    for n in range(_MAX_TERMS):
        w = np.exp(log_w)
        if w < _EPS_WEIGHT and n > 0:
            break

        r_n     = r - lam_drift + n * (mu_J + 0.5 * sigma_J**2) / T
        sigma_n = np.sqrt(max(sigma**2 + n * sigma_J**2 / T, 1e-14))

        price  += w * bs_call(S, K, r_n, q, T, sigma_n)

        log_w  += np.log(max(lam_prime_T, 1e-300)) - np.log(n + 1)

    return price

# ─────────────────────────────────────────────────────────────────────────────
# 7. IMPLIED VOL INVERSION
# ─────────────────────────────────────────────────────────────────────────────
def merton_implied_vol(S, K, r, q, T, sigma, lam, mu_J, sigma_J,
                       tol=1e-6, max_iter=200):
    """
    Invert BS to find the implied vol of the Merton price.
    Uses Brent's method on a bracketed interval [1e-6, 5].
    tol=1e-6 is sufficient for calibration; use 1e-8 for final reporting.
    """
    mdl = merton_call(S, K, r, q, T, sigma, lam, mu_J, sigma_J)
    intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    if mdl <= intrinsic + 1e-9:
        return np.nan
    try:
        return brentq(
            lambda v: bs_call(S, K, r, q, T, v) - mdl,
            1e-6, 5.0, xtol=tol, maxiter=max_iter)
    except ValueError:
        return np.nan

# ─────────────────────────────────────────────────────────────────────────────
# 8. ANALYTICAL GREEKS  —  FAITHFUL MERTON DERIVATION
# ─────────────────────────────────────────────────────────────────────────────
def _bs_all(S, K, r, q, T, sigma):
    """
    Returns (price, delta, gamma, vega_raw, theta, rho) for one BS term.
    vega_raw = ∂BS/∂σ_n  (w.r.t. the PER-TERM vol, before chain rule).
    """
    if sigma <= 0 or T <= 0:
        price = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        return price, 0.0, 0.0, 0.0, 0.0, 0.0
    sq   = sigma * np.sqrt(T)
    d1   = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sq
    d2   = d1 - sq
    Nd1  = norm.cdf(d1);   Nd2 = norm.cdf(d2)
    nd1  = norm.pdf(d1)
    eqT  = np.exp(-q * T); erT = np.exp(-r * T)

    price = S * eqT * Nd1 - K * erT * Nd2
    delta = eqT * Nd1
    gamma = eqT * nd1 / (S * sq)
    vega  = S * eqT * nd1 * np.sqrt(T)          # ∂BS/∂σ_n  (raw, per-term)
    theta = ((-S * eqT * nd1 * sigma / (2 * np.sqrt(T)))
             - r * K * erT * Nd2
             + q * S * eqT * Nd1) / 365.0       # per calendar day
    rho   = K * T * erT * Nd2 / 100.0           # per 1 bp move in r
    return price, delta, gamma, vega, theta, rho


def merton_greeks(S, K, r, q, T, sigma, lam, mu_J, sigma_J):
    kappa_bar   = np.exp(mu_J + 0.5 * sigma_J**2) - 1.0
    lam_prime   = lam * (1.0 + kappa_bar)
    lam_drift   = lam * kappa_bar
    lam_prime_T = lam_prime * T

    price = delta = gamma = theta = rho = 0.0
    vega_sigma = vega_sigma_J = 0.0
    log_w = -lam_prime_T

    for n in range(_MAX_TERMS):
        w = np.exp(log_w)
        if w < _EPS_WEIGHT and n > 0:
            break

        r_n        = r - lam_drift + n * (mu_J + 0.5 * sigma_J**2) / T
        var_n      = max(sigma**2 + n * sigma_J**2 / T, 1e-14)
        sigma_n    = np.sqrt(var_n)

        p, d, g, v_raw, th, rh = _bs_all(S, K, r_n, q, T, sigma_n)

        # chain rule:  ∂σ_n/∂σ   = σ / σ_n
        dsigma_n__dsigma   = sigma / sigma_n
        # chain rule:  ∂σ_n/∂σ_J = (n·σ_J/T) / σ_n
        dsigma_n__dsigma_J = (n * sigma_J / T) / sigma_n if sigma_n > 0 else 0.0

        price      += w * p
        delta      += w * d
        gamma      += w * g
        theta      += w * th
        rho        += w * rh
        vega_sigma   += w * v_raw * dsigma_n__dsigma
        vega_sigma_J += w * v_raw * dsigma_n__dsigma_J

        log_w += np.log(max(lam_prime_T, 1e-300)) - np.log(n + 1)

    return dict(
        price=price,
        delta=delta,
        gamma=gamma,
        vega_sigma=vega_sigma,
        vega_sigma_J=vega_sigma_J,
        theta=theta,
        rho=rho,
    )

# ─────────────────────────────────────────────────────────────────────────────
# 9. CALIBRATION  —  IV-RMSE  with vectorised objective + multi-start
# ─────────────────────────────────────────────────────────────────────────────
one_yr_idx = int(np.argmin(np.abs(np.array(dte_years) - 1.0)))
one_yr_T   = dte_years[one_yr_idx]
exp_str_1y, _, cal_sub = valid_expiries[one_yr_idx]
T_cal = one_yr_T
S, r  = spot, r_f

# Pre-extract arrays once — avoids Pandas overhead in the objective loop
K_arr_cal  = cal_sub["STRIKE"].values.astype(float)
iv_mkt_cal = cal_sub["C_IV"].values.astype(float)

def ivrmse(params):
    sigma, lam, mu_J, sigma_J = params
    sq_errs = []
    for K, iv_mkt in zip(K_arr_cal, iv_mkt_cal):
        iv_mdl = merton_implied_vol(
            S, K, r, q, T_cal, sigma, lam, mu_J, sigma_J, tol=1e-6)
        if not np.isnan(iv_mdl):
            sq_errs.append((iv_mdl - iv_mkt) ** 2)
    return np.mean(sq_errs) if sq_errs else 1e6

atm_iv = float(np.interp(S, K_arr_cal, iv_mkt_cal))

bounds = [
    (0.01, 2.0),    # σ  diffusion vol
    (0.00, 10.0),   # λ  jump intensity
    (-1.0,  1.0),   # μ_J log-jump mean
    (0.01,  2.0),   # σ_J log-jump vol
]

np.random.seed(42)
seeds = [[atm_iv * 0.8, 0.5, -0.10, 0.15]]
for _ in range(4):
    seeds.append([
        np.random.uniform(0.05, 0.60),
        np.random.uniform(0.00, 3.00),
        np.random.uniform(-0.40, 0.10),
        np.random.uniform(0.05, 0.60),
    ])

best_res, best_val = None, np.inf
print("\nCalibrating (multi-start IV-RMSE) — 5 starts …")

for i, x0 in enumerate(seeds):
    res = minimize(
        ivrmse, x0,
        method  = "L-BFGS-B",
        bounds  = bounds,
        options = {"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-10},
    )
    if res.fun < best_val:
        best_val, best_res = res.fun, res
    print(f"  start {i+1}/5  →  IV-RMSE = {np.sqrt(res.fun)*100:.4f}%")

sigma_c, lam_c, mu_J_c, sigma_J_c = best_res.x
kappa_bar_c = np.exp(mu_J_c + 0.5 * sigma_J_c**2) - 1.0

print(f"\n{'─'*60}")
print("Calibrated Merton Parameters (IV-RMSE optimised)")
print(f"{'─'*60}")
print(f"  σ   (diffusion vol)  = {sigma_c:.6f}  ({sigma_c*100:.3f}%)")
print(f"  λ   (jump intensity) = {lam_c:.6f}  jumps / yr")
print(f"  μ_J (log-jump mean)  = {mu_J_c:.6f}")
print(f"  σ_J (log-jump vol)   = {sigma_J_c:.6f}  ({sigma_J_c*100:.3f}%)")
print(f"  κ̄   (exp. jump ret)  = {kappa_bar_c:.6f}")
print(f"  IV-RMSE              = {np.sqrt(best_val)*100:.4f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 10. RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────

atm_strike_idx = int(np.argmin(np.abs(K_arr_cal - S)))
atm_strike_val = K_arr_cal[atm_strike_idx]

print(f"\n{'Strike':>10} {'Mkt IV%':>10} {'Mdl IV%':>10} {'ΔIV (bp)':>11} "
      f"{'Mkt $':>10} {'Mdl $':>10} {'Δ Price%':>11}")
print("═" * 82)

iv_errors, price_errors = [], []

for _, row in cal_sub.iterrows():
    K       = float(row["STRIKE"])
    iv_mkt  = float(row["C_IV"])
    mkt_px  = bs_call(S, K, r, q, T_cal, iv_mkt)
    mdl_px  = merton_call(S, K, r, q, T_cal,
                           sigma_c, lam_c, mu_J_c, sigma_J_c)
    iv_mdl  = merton_implied_vol(S, K, r, q, T_cal,
                                  sigma_c, lam_c, mu_J_c, sigma_J_c,
                                  tol=1e-8)

    div_iv  = (iv_mdl - iv_mkt) * 1e4 if not np.isnan(iv_mdl) else np.nan
    div_px  = (mdl_px / mkt_px - 1) * 100 if mkt_px > 0 else np.nan

    tag = "←ATM" if K == atm_strike_val else ""

    print(f"{K:>10.1f} {iv_mkt*100:>10.4f} "
          f"{(iv_mdl*100 if not np.isnan(iv_mdl) else float('nan')):>10.4f} "
          f"{(div_iv if not np.isnan(div_iv) else 0.0):>+11.2f} "
          f"{mkt_px:>10.5f} {mdl_px:>10.5f} "
          f"{(div_px if not np.isnan(div_px) else 0.0):>+11.4f}  {tag}")

    if not np.isnan(div_iv):
        iv_errors.append(abs(div_iv))
    if not np.isnan(div_px):
        price_errors.append(abs(div_px))

print("─" * 82)
print(f"  Avg |ΔIV|  = {np.mean(iv_errors):.2f} bp   |   "
      f"Max |ΔIV|  = {np.max(iv_errors):.2f} bp")
print(f"  Avg |ΔPx|  = {np.mean(price_errors):.4f}%  |   "
      f"Max |ΔPx|  = {np.max(price_errors):.4f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 11. ATM GREEKS TABLE
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("ATM Merton Greeks")
print(f"{'─'*60}")
g = merton_greeks(S, S, r, q, T_cal,
                  sigma_c, lam_c, mu_J_c, sigma_J_c)
print(f"  Price       = ${g['price']:>10.5f}")
print(f"  Delta  (Δ)  = {g['delta']:>10.6f}   ∂C/∂S")
print(f"  Gamma  (Γ)  = {g['gamma']:>10.6f}   ∂²C/∂S²")
print(f"  Vega σ      = {g['vega_sigma']:>10.6f}   ∂C/∂σ  (diffusion vol)")
print(f"  Vega σ_J    = {g['vega_sigma_J']:>10.6f}   ∂C/∂σ_J (jump vol)")
print(f"  Theta  (Θ)  = {g['theta']:>10.6f}   ∂C/∂t  (per calendar day)")
print(f"  Rho    (ρ)  = {g['rho']:>10.6f}   ∂C/∂r  (per 1 bp)")

# ─────────────────────────────────────────────────────────────────────────────
# 12. PLOTS
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"       : "serif",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "grid.alpha"        : 0.35,
})

strikes_grid = np.linspace(common_strikes[0], common_strikes[-1], 300)

# ── 12a. 2-D smile fit ──────────────────────────────────────────────────────
market_ivs = [bvs.blackVol(one_yr_T, float(k)) for k in strikes_grid]
merton_ivs = [merton_implied_vol(S, k, r, q, T_cal,
                                  sigma_c, lam_c, mu_J_c, sigma_J_c)
              for k in strikes_grid]

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1]})
ax, ax_res = axes

ax.plot(strikes_grid, market_ivs, color="steelblue", lw=1.8,
        label=f"Market surface  (T ≈ {one_yr_T:.2f} yr)")
ax.plot(strikes_grid, merton_ivs, color="darkorange", lw=1.8, ls="--",
        label="Merton IV (calibrated)")
ax.scatter(cal_sub["STRIKE"], cal_sub["C_IV"],
           s=50, color="steelblue", zorder=5, label="Market quotes")
ax.axvline(S, color="grey", lw=0.9, ls=":", label=f"Spot = {S:.2f}")
ax.set_ylabel("Implied Volatility", fontsize=11)
ax.set_title(
    f"AAPL Merton Jump-Diffusion  –  {QUOTE_DATE}\n"
    f"σ={sigma_c:.3f}  λ={lam_c:.3f}  μ_J={mu_J_c:.3f}  "
    f"σ_J={sigma_J_c:.3f}  |  IV-RMSE={np.sqrt(best_val)*100:.4f}%",
    fontsize=11)
ax.legend(fontsize=9)
ax.grid(True)

res_bps = [(mv - smv) * 1e4 if not np.isnan(mv) else np.nan
           for mv, smv in zip(merton_ivs, market_ivs)]
ax_res.plot(strikes_grid, res_bps, color="firebrick", lw=1.2)
ax_res.axhline(0, color="black", lw=0.8)
ax_res.set_ylabel("Residual (bp)", fontsize=9)
ax_res.set_xlabel("Strike", fontsize=11)
ax_res.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "aapl_merton_smile_v2.png"), dpi=300)

# ── 12b. Multi-expiry smile grid ────────────────────────────────────────────
sfc_K_min = float(common_strikes[0])
sfc_K_max = float(common_strikes[-1])

n_exp = len(valid_expiries)
ncols = min(4, n_exp)
nrows = (n_exp + ncols - 1) // ncols
fig2, axs = plt.subplots(nrows, ncols,
                          figsize=(4 * ncols, 3.5 * nrows))

if n_exp == 1:
    axs_flat = [axs]
elif nrows == 1:
    axs_flat = list(axs)
else:
    axs_flat = list(np.array(axs).flatten())

for idx_e, (exp_str, ql_date, sub) in enumerate(valid_expiries):
    T_e  = dte_years[idx_e]
    ax_e = axs_flat[idx_e]

    k_lo     = max(float(sub["STRIKE"].min()), sfc_K_min)
    k_hi     = min(float(sub["STRIKE"].max()), sfc_K_max)
    ks_plot  = np.linspace(k_lo, k_hi, 200)

    miv         = [merton_implied_vol(S, k, r, q, T_e,
                                       sigma_c, lam_c, mu_J_c, sigma_J_c)
                   for k in ks_plot]
    mkt_iv_sfc  = [bvs.blackVol(T_e, float(k)) for k in ks_plot]

    ax_e.plot(ks_plot, mkt_iv_sfc, color="steelblue",  lw=1.5, label="Market")
    ax_e.plot(ks_plot, miv,        color="darkorange",  lw=1.5, ls="--",
              label="Merton")
    ax_e.scatter(sub["STRIKE"], sub["C_IV"],
                 s=18, color="steelblue", zorder=4)
    ax_e.set_title(f"{exp_str}  (T={T_e:.2f}yr)", fontsize=8)
    ax_e.set_xlabel("K", fontsize=7)
    ax_e.set_ylabel("IV",fontsize=7)
    ax_e.legend(fontsize=6)
    ax_e.grid(True, alpha=0.3)

for idx_e in range(n_exp, len(axs_flat)):
    axs_flat[idx_e].set_visible(False)

fig2.suptitle(
    f"AAPL – Merton JDM smile across all expiries  ({QUOTE_DATE})",
    fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "aapl_merton_all_expiries.png"), dpi=300)

# ── 12c. 3-D vol surface ────────────────────────────────────────────────────
t_min_plot   = dte_years[0]  * 1.01
t_max_plot   = dte_years[-1] * 0.99
plot_years   = np.linspace(t_min_plot, t_max_plot, 35)
plot_strikes = np.linspace(common_strikes[0], common_strikes[-1], 45)
X, Y = np.meshgrid(plot_strikes, plot_years)
Z = np.array([bvs.blackVol(float(y), float(x))
              for xr, yr in zip(X, Y)
              for x, y   in zip(xr, yr)]).reshape(X.shape)

fig3d = plt.figure(figsize=(12, 7))
ax3d  = fig3d.add_subplot(111, projection="3d")
surf  = ax3d.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=cm.coolwarm, linewidth=0.08, alpha=0.9)
fig3d.colorbar(surf, shrink=0.45, aspect=8, pad=0.05)
ax3d.set_xlabel("Strike",    labelpad=10)
ax3d.set_ylabel("Time (yrs)",labelpad=10)
ax3d.set_zlabel("Implied Vol",labelpad=10)
ax3d.set_title(f"AAPL Implied Volatility Surface  –  {QUOTE_DATE}", pad=15)
plt.savefig(os.path.join(output_folder, "aapl_merton_surface_3d.png"), dpi=300)

# ── 12d. Greeks term structure ───────────────────────────────────────────────
T_grid = np.linspace(0.05, 2.0, 80)

greek_names = ["delta", "gamma", "vega_sigma", "theta"]
greek_vals  = {gn: [] for gn in greek_names}

for T_g in T_grid:
    g_t = merton_greeks(S, S, r, q, T_g,
                        sigma_c, lam_c, mu_J_c, sigma_J_c)
    for gn in greek_names:
        greek_vals[gn].append(g_t[gn])

labels = {
    "delta"     : "Delta (∂C/∂S)",
    "gamma"     : "Gamma (∂²C/∂S²)",
    "vega_sigma": "Vega σ (∂C/∂σ)",
    "theta"     : "Theta (∂C/∂t, /day)",
}
colors = ["steelblue", "darkorange", "seagreen", "firebrick"]

fig_g, axs_g = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
for ax_g, gn, col in zip(axs_g.flatten(), greek_names, colors):
    ax_g.plot(T_grid, greek_vals[gn], color=col, lw=1.8)
    ax_g.axvline(T_cal, color="grey", ls=":", lw=0.9,
                 label=f"Calib T={T_cal:.2f}")
    ax_g.set_title(labels[gn], fontsize=10)
    ax_g.set_xlabel("T (yrs)")
    ax_g.grid(True, alpha=0.35)
    ax_g.legend(fontsize=8)

fig_g.suptitle(
    f"ATM Merton Greeks vs Maturity  –  {QUOTE_DATE}", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "aapl_merton_greeks.png"), dpi=300)

plt.show()
print("\nAll graphs saved to ./graphs/")