#!/usr/bin/env python
"""Diagnose the slow residual tail of Option B (scale=0.5, stable gain).

Option B's tau_fit grows ~N^1.8 (R^2 drops) -> the E_gap decay is two-timescale:
a FAST feedforward phase + a SLOW residual tail. This script decomposes it:

  knee_frac = E_gap(t_pk + ~2.5*T_FF) / peak   -> how much the fast phase removed
  tau_fast  = log-fit of the early decay  [t_pk, t_pk+2.5]
  tau_slow  = log-fit of the late decay   [t_pk+5, end, while E_gap>floor]

Then compares tau_slow to the BASELINE tau (normalized gain) at the same N:
if tau_slow ~ baseline_tau, the residual is the LOW-GAIN FEEDBACK (O(N^2)) cleaning
up what the feedforward did not deliver. Reads existing runs (no re-run):
  optionB_runs/  (B, scale=0.5)   and   gain_runs/  (baseline + A, normalized gain).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = os.path.dirname(os.path.abspath(__file__))
B_RUNS = os.path.join(EXP, "optionB_runs")
GAIN_RUNS = os.path.join(EXP, "gain_runs")
T0 = 5.0
T_FF = 1.0
NS = [24, 40, 50]


def load_egap(path):
    if not os.path.exists(path):
        return None, None
    d = pd.read_csv(path)
    d = d[d["timestamp"] >= T0]
    return d["timestamp"].to_numpy(float), d["E_gap"].to_numpy(float)


def logfit_tau(t, e, t_lo, t_hi, floor):
    m = (t >= t_lo) & (t <= t_hi) & (e > floor) & np.isfinite(e)
    if m.sum() < 5:
        return np.nan, np.nan
    coef = np.polyfit(t[m], np.log(e[m]), 1)
    tau = -1.0 / coef[0] if coef[0] < 0 else np.nan
    pred = np.polyval(coef, t[m])
    ss_res = float(np.sum((np.log(e[m]) - pred) ** 2))
    ss_tot = float(np.sum((np.log(e[m]) - np.mean(np.log(e[m]))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(tau), float(r2)


def at_time(t, e, tt):
    i = int(np.argmin(np.abs(t - tt)))
    return float(e[i])


print(f"{'N':>3} {'peak':>7} {'knee_frac':>9} {'tau_fast':>8} {'(R2)':>6} {'tau_slow':>8} {'(R2)':>6} {'base_tau':>8} {'slow/base':>9}")
fig, ax = plt.subplots(figsize=(9.5, 6))
for n in NS:
    t, e = load_egap(os.path.join(B_RUNS, f"dual_pulse_N{n}", "target_telemetry.csv"))
    if t is None:
        continue
    i_pk = int(np.nanargmax(e))
    t_pk, peak = t[i_pk], e[i_pk]
    knee = at_time(t, e, t_pk + 2.5 * T_FF)
    knee_frac = knee / peak if peak > 0 else np.nan
    tau_fast, r2f = logfit_tau(t, e, t_pk, t_pk + 2.5 * T_FF, 1e-4)
    tau_slow, r2s = logfit_tau(t, e, t_pk + 5.0, t[-1], 5e-4)

    # baseline (normalized gain) tau for reference
    tb, eb = load_egap(os.path.join(GAIN_RUNS, f"baseline_N{n}", "target_telemetry.csv"))
    base_tau = np.nan
    if tb is not None:
        ib = int(np.nanargmax(eb))
        base_tau, _ = logfit_tau(tb, eb, tb[ib], tb[-1], 5e-4)

    ratio = tau_slow / base_tau if (np.isfinite(tau_slow) and np.isfinite(base_tau) and base_tau > 0) else np.nan
    print(f"{n:>3} {peak:>7.3f} {knee_frac:>9.3f} {tau_fast:>8.2f} {r2f:>6.2f} "
          f"{tau_slow:>8.2f} {r2s:>6.2f} {base_tau:>8.2f} {ratio:>9.2f}")

    ax.plot(t - t_pk, e, label=f"Option B (scale .5) N={n}")

# reference traces at N=50
for variant, d in (("baseline", "baseline"), ("A", "dual_pulse")):
    tb, eb = load_egap(os.path.join(GAIN_RUNS, f"{d}_N50", "target_telemetry.csv"))
    if tb is not None:
        ib = int(np.nanargmax(eb))
        ax.plot(tb - tb[ib], eb, "--", alpha=0.6, label=f"{variant} N=50 (ref)")

ax.set_yscale("log")
ax.set_xlim(0, 40)
ax.set_xlabel("time since fault peak [s]")
ax.set_ylabel("E_gap (log)")
ax.set_title("Option B (scale 0.5): fast feedforward phase + slow residual tail?\n"
             "knee = where fast phase ends; slow tail vs baseline = is the low-gain feedback cleaning residual")
ax.grid(True, which="both", ls=":", alpha=0.5)
ax.legend(fontsize=8)
fig.tight_layout()
out = os.path.join(EXP, "diag_optionB_residual.png")
fig.savefig(out, dpi=130)
print(f"\nsaved {out}")
