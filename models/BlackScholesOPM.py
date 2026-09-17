import bisect
import os

import QuantLib as ql
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

# ─────────────────────────────────────────────────────────────────────────────
# 0. ENSURE DIRECTORY EXISTS
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
# 3. QUANTLIB SETTINGS  ←  DATE-DRIVEN RATES
# ─────────────────────────────────────────────────────────────────────────────

# 1-year US Treasury constant-maturity yield, annual averages (source: FRED H.15)
_TREASURY_1Y = {
    2015: 0.0065,
    2016: 0.0059,
    2017: 0.0130,
    2018: 0.0238,
    2019: 0.0224,
    2020: 0.0016,
    2021: 0.0005,
}

def _interp_rate(table: dict, year: int, month: int) -> float:
    """Linearly interpolate between adjacent annual entries."""
    t    = year + (month - 1) / 12.0
    y_lo = int(t)
    y_hi = y_lo + 1
    r_lo = table.get(y_lo, list(table.values())[0])
    r_hi = table.get(y_hi, list(table.values())[-1])
    frac = t - y_lo
    return r_lo + frac * (r_hi - r_lo)

risk_free_rate = _interp_rate(_TREASURY_1Y, _yr, _mo)

print(f"\n--- Rates for {QUOTE_DATE} ---")
print(f"Risk-free rate (1-yr Treasury, interpolated) : {risk_free_rate:.4f}  ({risk_free_rate*100:.2f}%)")
print(f"Dividend yield                               : 0.0000  (excluded — Black-Scholes 1973 assumes no dividends)")

day_count        = ql.Actual365Fixed()
calendar         = ql.UnitedStates(ql.UnitedStates.Settlement)
calculation_date = ql.Date(_dy, _mo, _yr)
ql.Settings.instance().evaluationDate = calculation_date

# ─────────────────────────────────────────────────────────────────────────────
# 4. BUILD VOL SURFACE
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
# 5. HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
S = spot
r = risk_free_rate
# No dividend parameter exists in this implementation — Black-Scholes (1973).

def bs_call(S, K, r, T, sigma):
    if sigma <= 0 or T <= 0:
        return max(S - K * np.exp(-r * T), 0.0)   # intrinsic value at expiry
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_vega(S, K, r, T, sigma):
    if sigma <= 0 or T <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def calibrate_flat_vol(sub, T):
    atm_iv = float(np.interp(S, sub["STRIKE"].values, sub["C_IV"].values))

    vegas = np.array([
        bs_vega(S, float(row["STRIKE"]), r, T, atm_iv)
        for _, row in sub.iterrows()
    ])
    vega_weights = (vegas / vegas.sum()
                    if vegas.sum() > 0
                    else np.ones(len(vegas)) / len(vegas))

    def calibration_error(sigma):
        if sigma <= 0:
            return 1e10
        sq_rel_errors = []
        for (_, row), w in zip(sub.iterrows(), vega_weights):
            K   = float(row["STRIKE"])
            iv  = float(row["C_IV"])
            mkt = bs_call(S, K, r, T, iv)
            mdl = bs_call(S, K, r, T, sigma)
            if mkt > 0:
                sq_rel_errors.append(w * (mdl / mkt - 1.0) ** 2)
        return np.sum(sq_rel_errors)

    result = minimize_scalar(calibration_error, bounds=(0.01, 2.0), method="bounded")
    return result.x, atm_iv, vega_weights

# ─────────────────────────────────────────────────────────────────────────────
# 6. MULTI-EXPIRY CALIBRATION  
# ─────────────────────────────────────────────────────────────────────────────
dte_years = [(ql_date - calculation_date) / 365.0
             for _, ql_date, _ in valid_expiries]

term_structure = []   # list of (exp_str, T, sigma, atm_iv, avg_err, sub, vega_weights)

for (exp_str, ql_date, sub), T in zip(valid_expiries, dte_years):
    sigma, atm_iv, vw = calibrate_flat_vol(sub, T)

    avg_err = 0.0
    for (_, row), w in zip(sub.iterrows(), vw):
        K   = float(row["STRIKE"])
        iv  = float(row["C_IV"])
        mkt = bs_call(S, K, r, T, iv)
        mdl = bs_call(S, K, r, T, sigma)
        if mkt > 0:
            avg_err += abs(mdl / mkt - 1.0)
    avg_err = avg_err * 100.0 / len(sub)

    term_structure.append((exp_str, T, sigma, atm_iv, avg_err, sub, vw))

# Overall weighted average error
total_strikes   = sum(len(ts[5]) for ts in term_structure)
overall_avg_err = sum(ts[4] * len(ts[5]) for ts in term_structure) / total_strikes

# Find the 1-year expiry for display
one_yr_idx = int(np.argmin(np.abs(np.array([ts[1] for ts in term_structure]) - 1.0)))
ts_1y = term_structure[one_yr_idx]

print(f"\n{'Expiry':>12} {'T (yrs)':>9} {'ATM IV':>9} {'BS σ':>9}")
print("=" * 44)
print(f"{ts_1y[0]:>12} {ts_1y[1]:>9.3f} {ts_1y[3]:>9.4f} {ts_1y[2]:>9.4f}")
print("=" * 44)
print(f"  (Showing 1-year expiry only; all {len(term_structure)} expiries used in calculations)")

# ─────────────────────────────────────────────────────────────────────────────
# 7. DETAILED PER-STRIKE TABLE  (1-year expiry only)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'─'*80}")
print(f"PER-STRIKE BREAKDOWN - 1-YEAR EXPIRY ({ts_1y[0]})")
print(f"{'─'*80}")

exp_str, T, sigma, atm_iv, avg_err, sub, vw = ts_1y
print(f"\n  Expiry: {exp_str}  |  T = {T:.3f} yr  |  σ = {sigma:.4f}  |  ATM IV = {atm_iv:.4f}")
print(f"  {'Strike':>10} {'Mkt Price':>12} {'BS Price':>12} {'Rel Err%':>12} {'Vega Wt':>10}")
print(f"  {'─'*58}")
for (_, row), w in zip(sub.iterrows(), vw):
    K   = float(row["STRIKE"])
    iv  = float(row["C_IV"])
    mkt = bs_call(S, K, r, T, iv)
    mdl = bs_call(S, K, r, T, sigma)
    rel_err = (mdl / mkt - 1.0) * 100.0 if mkt > 0 else float("nan")
    print(f"  {K:>10.2f} {mkt:>12.5f} {mdl:>12.5f} {rel_err:>12.7f} {w:>10.6f}")
print(f"  {'─'*58}")
print(f"  Average Abs Error (%)                              : {avg_err:.3f}")
print(f"  Overall vega-weighted avg abs error (all {len(term_structure)} expiries) : {overall_avg_err:.3f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 8. PLOT A — TERM STRUCTURE OF CALIBRATED FLAT VOLS
# ─────────────────────────────────────────────────────────────────────────────
ts_T       = [ts[1] for ts in term_structure]
ts_sigma   = [ts[2] for ts in term_structure]
ts_atm_iv  = [ts[3] for ts in term_structure]

fig_ts, ax_ts = plt.subplots(figsize=(9, 5))
ax_ts.plot(ts_T, ts_atm_iv,  "o--", color="steelblue",  label="ATM Market IV",         linewidth=1.4)
ax_ts.plot(ts_T, ts_sigma,   "s-",  color="firebrick",  label="BS Calibrated Flat Vol", linewidth=1.8)
ax_ts.set_xlabel("Time to Expiry (yrs)", size=12)
ax_ts.set_ylabel("Volatility", size=12)
ax_ts.set_title(f"BS Flat Vol Term Structure — {QUOTE_DATE}")
ax_ts.legend(fontsize=10)
ax_ts.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "aapl_bs_vol_term_structure.png"), dpi=300)

# ─────────────────────────────────────────────────────────────────────────────
# 9. PLOT B — SMILE FIT ACROSS ALL EXPIRIES  (grid of subplots)
# ─────────────────────────────────────────────────────────────────────────────
n_expiries = len(term_structure)
n_cols     = 3
n_rows     = (n_expiries + n_cols - 1) // n_cols

fig_grid, axes = plt.subplots(n_rows, n_cols,
                               figsize=(5.5 * n_cols, 4 * n_rows),
                               constrained_layout=True)
axes_flat = axes.flatten() if n_expiries > 1 else [axes]

strikes_grid = np.linspace(common_strikes[0], common_strikes[-1], 200)

for idx, (exp_str, T, sigma, atm_iv, avg_err, sub, vw) in enumerate(term_structure):
    ax = axes_flat[idx]
    mkt_iv_curve = [black_var_surface.blackVol(T, float(k)) for k in strikes_grid]

    ax.plot(strikes_grid, mkt_iv_curve, color="steelblue",
            label=f"Market IV (T≈{T:.2f}yr)")
    ax.axhline(sigma, color="firebrick", linestyle="--", linewidth=1.4,
               label=f"BS σ={sigma:.4f}")
    ax.axhline(atm_iv, color="darkorange", linestyle=":", linewidth=1.1,
               label=f"ATM IV={atm_iv:.4f}")
    ax.plot(sub["STRIKE"].values, sub["C_IV"].values,
            "o", color="steelblue", markersize=4)
    ax.axvline(S, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.set_title(f"{exp_str}  |  Err={avg_err:.2f}%", fontsize=9)
    ax.set_xlabel("Strike", fontsize=8)
    ax.set_ylabel("Implied Vol", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

# Hide any unused subplot panels
for idx in range(n_expiries, len(axes_flat)):
    axes_flat[idx].set_visible(False)

fig_grid.suptitle(f"BS Flat Vol Smile Fit — All Expiries — {QUOTE_DATE}",
                  fontsize=13, fontweight="bold")
plt.savefig(os.path.join(output_folder, "aapl_bs_iv_smile_all_expiries.png"), dpi=300)

# ─────────────────────────────────────────────────────────────────────────────
# 10. PLOT C — 3-D VOL SURFACE
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
ax3d.set_title(f"AAPL Implied Volatility Surface — {QUOTE_DATE}")
plt.savefig(os.path.join(output_folder, "aapl_bs_vol_surface_3d.png"), dpi=300)

# ─────────────────────────────────────────────────────────────────────────────
# 11. PLOT D — ORIGINAL SINGLE-EXPIRY SMILE  (1-year slice, kept for reference)
# ─────────────────────────────────────────────────────────────────────────────
exp_str_1y, T_1y, sigma_1y, atm_iv_1y, avg_err_1y, sub_1y, vw_1y = ts_1y

implied_vols_1y = [black_var_surface.blackVol(T_1y, float(k)) for k in strikes_grid]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(strikes_grid, implied_vols_1y, color="steelblue",
        label=f"Market IV Surface (T≈{T_1y:.2f} yr)")
ax.axhline(sigma_1y, color="firebrick", linestyle="--",
           label=f"BS Flat Vol σ = {sigma_1y:.4f}  (vega-weighted)")
ax.axhline(atm_iv_1y, color="darkorange", linestyle=":", linewidth=1.2,
           label=f"ATM Market IV = {atm_iv_1y:.4f}")
ax.plot(sub_1y["STRIKE"].values, sub_1y["C_IV"].values,
        "o", color="steelblue", label="Market IVs (calibration points)")
ax.axvline(S, color="gray", linestyle=":", linewidth=0.8, alpha=0.6,
           label=f"Spot S = {S:.2f}")
ax.set_xlabel("Strike", size=12)
ax.set_ylabel("Implied Vol", size=12)
ax.set_title(f"AAPL Implied Vol Smile vs BS Flat Vol — {QUOTE_DATE} → {exp_str_1y}")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "aapl_bs_iv_smile.png"), dpi=300)

plt.show()
print("\nDone. Graphs saved to:", os.path.abspath(output_folder))