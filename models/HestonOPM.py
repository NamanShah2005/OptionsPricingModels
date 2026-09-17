import bisect
import os

import QuantLib as ql
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

# ─────────────────────────────────────────────────────────────────────────────
# 0. DIRECTORY MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────
output_folder = os.path.join(os.path.dirname(__file__), "..", "graphs")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD AAPL OPTIONS DATA
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "aapl_2016_2020.csv")

df = pd.read_csv(CSV_PATH, low_memory=False)
df.columns = [c.strip().strip("[]") for c in df.columns]
df["QUOTE_DATE"]  = df["QUOTE_DATE"].str.strip()
df["EXPIRE_DATE"] = df["EXPIRE_DATE"].str.strip()

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATE INPUT WITH VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
available_dates = sorted(df["QUOTE_DATE"].unique())

while True:
    print("\nAvailable date range: {} to {}".format(available_dates[0], available_dates[-1]))
    QUOTE_DATE = input("Enter a quote date (YYYY-MM-DD): ").strip()

    if QUOTE_DATE not in available_dates:
        print(f"  ⚠  Date '{QUOTE_DATE}' not found in the dataset.")
        print("  Closest available dates:")
        idx = bisect.bisect_left(available_dates, QUOTE_DATE)
        hints = available_dates[max(0, idx - 2) : idx + 3]
        for h in hints:
            print(f"    • {h}")
        print("  Please try again.")
        continue
    break

df_day = df[df["QUOTE_DATE"] == QUOTE_DATE].copy()
df_day["C_IV"] = pd.to_numeric(df_day["C_IV"], errors="coerce")

spot = float(df_day["UNDERLYING_LAST"].iloc[0])
print(f"\nQuote date : {QUOTE_DATE}")
print(f"Spot price : {spot}")

_yr, _mo, _dy = [int(x) for x in QUOTE_DATE.split("-")]

# ─────────────────────────────────────────────────────────────────────────────
# 3. DATE-DRIVEN RATES (FRED 1-yr CMT  +  AAPL dividend yield)
# ─────────────────────────────────────────────────────────────────────────────
_TREASURY_1Y = {
    2015: 0.0065,
    2016: 0.0059,
    2017: 0.0130,
    2018: 0.0238,
    2019: 0.0224,
    2020: 0.0016,
    2021: 0.0005,
}

_AAPL_DIV_YIELD = {
    2015: 0.0171,
    2016: 0.0200,
    2017: 0.0158,
    2018: 0.0147,
    2019: 0.0127,
    2020: 0.0077,
    2021: 0.0065,
}

def _interp_rate(table: dict, year: int, month: int) -> float:
    t    = year + (month - 1) / 12.0
    y_lo = int(t)
    y_hi = y_lo + 1
    r_lo = table.get(y_lo, list(table.values())[0])
    r_hi = table.get(y_hi, list(table.values())[-1])
    frac = t - y_lo
    return r_lo + frac * (r_hi - r_lo)

risk_free_rate = _interp_rate(_TREASURY_1Y,    _yr, _mo)
dividend_rate  = _interp_rate(_AAPL_DIV_YIELD, _yr, _mo)

print(f"\n--- Rates for {QUOTE_DATE} ---")
print(f"Risk-free rate (1-yr Treasury, interpolated) : {risk_free_rate:.4f}  ({risk_free_rate*100:.2f}%)")
print(f"Dividend yield (AAPL, interpolated)          : {dividend_rate:.4f}  ({dividend_rate*100:.2f}%)")

day_count        = ql.Actual365Fixed()
calendar         = ql.UnitedStates(ql.UnitedStates.Settlement)
calculation_date = ql.Date(_dy, _mo, _yr)
ql.Settings.instance().evaluationDate = calculation_date

flat_ts     = ql.YieldTermStructureHandle(
    ql.FlatForward(calculation_date, risk_free_rate, day_count))
dividend_ts = ql.YieldTermStructureHandle(
    ql.FlatForward(calculation_date, dividend_rate, day_count))

# ─────────────────────────────────────────────────────────────────────────────
# 4. BUILD IMPLIED VOL SURFACE  (used for smile/3D plots only)
# ─────────────────────────────────────────────────────────────────────────────
def str_to_ql_date(s):
    yr, mo, dy = [int(x) for x in s.split("-")]
    return ql.Date(dy, mo, yr)

raw_expiries = sorted(df_day["EXPIRE_DATE"].unique())

valid_expiries = []
for exp_str in raw_expiries:
    ql_date = str_to_ql_date(exp_str)
    if ql_date <= calculation_date:
        continue
    sub = df_day[df_day["EXPIRE_DATE"] == exp_str].copy()
    sub = sub.dropna(subset=["C_IV"])
    sub = sub[sub["C_IV"] > 0]
    sub = sub[(sub["STRIKE"] >= spot * 0.80) & (sub["STRIKE"] <= spot * 1.20)]
    sub = sub.sort_values("STRIKE").reset_index(drop=True)
    if len(sub) >= 3:
        valid_expiries.append((exp_str, ql_date, sub))

expiration_dates = [v[1] for v in valid_expiries]
print(f"\nExpiries used in vol surface : {len(expiration_dates)}")

all_strike_sets = [set(v[2]["STRIKE"].values) for v in valid_expiries]
common_strikes  = sorted(set.intersection(*all_strike_sets))
if len(common_strikes) < 3:
    common_strikes = sorted(set.union(*all_strike_sets))
    common_strikes = [s for s in common_strikes if spot * 0.80 <= s <= spot * 1.20]

print(f"Common strikes               : {common_strikes}")

implied_vols_matrix = ql.Matrix(len(common_strikes), len(expiration_dates))
for j, (exp_str, ql_date, sub) in enumerate(valid_expiries):
    iv_by_strike = sub.set_index("STRIKE")["C_IV"]
    ks = iv_by_strike.index.values.astype(float)
    vs = iv_by_strike.values.astype(float)
    for i, K in enumerate(common_strikes):
        implied_vols_matrix[i][j] = float(np.interp(K, ks, vs))

black_var_surface = ql.BlackVarianceSurface(
    calculation_date, calendar,
    expiration_dates, common_strikes,
    implied_vols_matrix, day_count)
black_var_surface.setInterpolation("bicubic")

# ─────────────────────────────────────────────────────────────────────────────
# 5. SELECT 1-YEAR EXPIRY FOR CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────
dte_years  = [(ql_date - calculation_date) / 365.0 for _, ql_date, _ in valid_expiries]
one_yr_idx = int(np.argmin(np.abs(np.array(dte_years) - 1.0)))
one_yr_T   = dte_years[one_yr_idx]

cal_exp_str, cal_ql_date, cal_sub = valid_expiries[one_yr_idx]
t_days = cal_ql_date - calculation_date

strikes_grid    = np.linspace(common_strikes[0], common_strikes[-1], 200)
implied_vols_1y = [black_var_surface.blackVol(one_yr_T, float(s)) for s in strikes_grid]

# ─────────────────────────────────────────────────────────────────────────────
# 6. PLOT A — 2-D MARKET IMPLIED VOL SMILE  (1-year expiry)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(strikes_grid, implied_vols_1y, label=f"Black Surface (T≈{one_yr_T:.2f} yr)")
ax.plot(cal_sub["STRIKE"].values, cal_sub["C_IV"].values, "o", label="Market IVs")
ax.set_xlabel("Strike", size=12)
ax.set_ylabel("Implied Vol", size=12)
ax.set_title(f"AAPL Implied Vol Smile – {QUOTE_DATE} → expiry {cal_exp_str}")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "aapl_heston_market_smile.png"), dpi=300)

# ─────────────────────────────────────────────────────────────────────────────
# 7. PLOT B — 3-D IMPLIED VOL SURFACE
# ─────────────────────────────────────────────────────────────────────────────
t_min = dte_years[0]  * 1.01
t_max = dte_years[-1] * 0.99

plot_years   = np.linspace(t_min, t_max, 30)
plot_strikes = np.linspace(common_strikes[0], common_strikes[-1], 40)

X, Y = np.meshgrid(plot_strikes, plot_years)
Z = np.array([
    black_var_surface.blackVol(float(y), float(x))
    for xr, yr in zip(X, Y)
    for x, y in zip(xr, yr)
]).reshape(len(X), len(X[0]))

fig3d = plt.figure(figsize=(11, 7))
ax3d  = fig3d.add_subplot(111, projection="3d")
surf  = ax3d.plot_surface(X, Y, Z, rstride=1, cstride=1,
                          cmap=cm.coolwarm, linewidth=0.1)
fig3d.colorbar(surf, shrink=0.5, aspect=5)
ax3d.set_xlabel("Strike")
ax3d.set_ylabel("Time (yrs)")
ax3d.set_zlabel("Implied Vol")
ax3d.set_title(f"AAPL Implied Volatility Surface – {QUOTE_DATE}")
plt.savefig(os.path.join(output_folder, "aapl_heston_vol_surface_3d.png"), dpi=300)

# ─────────────────────────────────────────────────────────────────────────────
# 8. BLACK-SCHOLES CALL PRICER  (recovers true market prices from IVs)
# ─────────────────────────────────────────────────────────────────────────────
def bs_call(S, K, r, q, T, sigma):
    if sigma <= 0 or T <= 0:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# ─────────────────────────────────────────────────────────────────────────────
# 9. HESTON (1993) MODEL CALIBRATION 
# ─────────────────────────────────────────────────────────────────────────────
atm_iv_guess = float(np.interp(spot,
                               cal_sub["STRIKE"].values,
                               cal_sub["C_IV"].values))
v0_init    = atm_iv_guess ** 2
theta_init = v0_init
kappa_init = 1.5
sigma_init = 0.4
rho_init   = -0.7 # negative leverage effect

process = ql.HestonProcess(
    flat_ts, dividend_ts,
    ql.QuoteHandle(ql.SimpleQuote(spot)),
    v0_init, kappa_init, theta_init, sigma_init, rho_init)

model  = ql.HestonModel(process)
engine = ql.AnalyticHestonEngine(model)

heston_helpers = []
for _, row in cal_sub.iterrows():
    K  = float(row["STRIKE"])
    iv = float(row["C_IV"])
    helper = ql.HestonModelHelper(
        ql.Period(int(t_days), ql.Days),
        calendar, spot, K,
        ql.QuoteHandle(ql.SimpleQuote(iv)),
        flat_ts, dividend_ts)
    helper.setPricingEngine(engine)
    heston_helpers.append(helper)

lm = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
model.calibrate(
    heston_helpers, lm,
    ql.EndCriteria(1000, 100, 1e-8, 1e-8, 1e-8))

theta_c, kappa_c, sigma_c, rho_c, v0_c = model.params()

print(f"\nCalibrated Heston Parameters  (Heston 1993):")
print(f"  v0    = {v0_c:.6f}  (initial variance;      √v0 = {np.sqrt(v0_c):.4f})")
print(f"  kappa = {kappa_c:.6f}  (mean-reversion speed)")
print(f"  theta = {theta_c:.6f}  (long-run variance;    √θ  = {np.sqrt(theta_c):.4f})")
print(f"  sigma = {sigma_c:.6f}  (vol-of-vol)")
print(f"  rho   = {rho_c:.6f}  (spot-vol correlation)")

feller = 2.0 * kappa_c * theta_c
print(f"\nFeller condition:  2κθ = {feller:.6f},  σ² = {sigma_c**2:.6f}  "
      f"→  {'✓ satisfied (variance stays positive)' if feller > sigma_c**2 else '✗ violated (variance may hit zero)'}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. PRICE EUROPEAN CALLS UNDER CALIBRATED HESTON MODEL
# ─────────────────────────────────────────────────────────────────────────────
def heston_call_price(K, expiry_date, model):
    """Price a European call at strike K using the calibrated Heston model."""
    payoff   = ql.PlainVanillaPayoff(ql.Option.Call, K)
    exercise = ql.EuropeanExercise(expiry_date)
    option   = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(ql.AnalyticHestonEngine(model))
    return option.NPV()

# ─────────────────────────────────────────────────────────────────────────────
# 11. RESULTS TABLE  (1-year expiry only)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nCalibration expiry : {cal_exp_str}  (T ≈ {one_yr_T:.3f} yr)")
print(f"\n{'Strike':>10} {'Market Call':>14} {'Heston Call':>13} "
      f"{'Rel Err (%)':>14} {'Market IV':>11}")
print("=" * 68)

avg_err    = 0.0
heston_ivs = []

for _, row in cal_sub.iterrows():
    K  = float(row["STRIKE"])
    iv = float(row["C_IV"])

    mkt_price = bs_call(spot, K, risk_free_rate, dividend_rate, one_yr_T, iv)
    mdl_price = heston_call_price(K, cal_ql_date, model)

    rel_err = (mdl_price / mkt_price - 1.0) * 100.0 if mkt_price > 0 else float("nan")
    avg_err += abs(rel_err)

    # Back-solve Heston implied vol for the smile plot
    try:
        heston_iv = brentq(
            lambda v: bs_call(spot, K, risk_free_rate, dividend_rate, one_yr_T, v) - mdl_price,
            1e-6, 5.0, xtol=1e-8, maxiter=200)
    except ValueError:
        heston_iv = float("nan")
    heston_ivs.append(heston_iv)

    print(f"{K:>10.2f} {mkt_price:>14.5f} {mdl_price:>13.5f} "
          f"{rel_err:>14.7f} {iv:>11.6f}")

print("-" * 68)
avg_err /= len(cal_sub)
print(f"Average Abs Error (%)  : {avg_err:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 12. PLOT C — HESTON IMPLIED VOL SMILE  (overlaid on market)
# ─────────────────────────────────────────────────────────────────────────────
valid_ks  = [float(cal_sub.iloc[i]["STRIKE"])
             for i in range(len(cal_sub)) if not np.isnan(heston_ivs[i])]
valid_ivs = [v for v in heston_ivs if not np.isnan(v)]

fig2, ax2 = plt.subplots(figsize=(9, 5))
ax2.plot(strikes_grid, implied_vols_1y, color="steelblue",
         label=f"Market IV Surface (T≈{one_yr_T:.2f} yr)")
ax2.plot(cal_sub["STRIKE"].values, cal_sub["C_IV"].values,
         "o", color="steelblue", label="Market IVs (raw)")
ax2.plot(valid_ks, valid_ivs, "x--", color="darkorange",
         label="Heston Implied Vol (calibrated)", linewidth=1.8, markersize=8)
ax2.set_xlabel("Strike", size=12)
ax2.set_ylabel("Implied Vol", size=12)
ax2.set_title(f"AAPL Implied Vol Smile vs Heston Model – {QUOTE_DATE}")
ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "aapl_heston_smile_comparison.png"), dpi=300)

plt.show()
print("\nDone. Graphs saved to:", os.path.abspath(output_folder))