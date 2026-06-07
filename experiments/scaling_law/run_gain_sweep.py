#!/usr/bin/env python
"""Gain-normalization test: does the stability-speed-N trade-off behave as predicted?

The normalized e_tau injects an effective-gain factor ~N near equilibrium. With
FIXED K_E_TAU this (a) gives the fast O(N) relaxation but (b) destabilizes the
high-spatial-frequency modes at large N (the N=50 limit cycle). NORMALIZING the
gain to hold the effective gain constant -- K_E_TAU = GAIN_PRODUCT / N -- should:

  P1 (stability):  kill the N=50 limit cycle (E_gap recovers; low late-time std).
  P2 (baseline):   slow the baseline back toward O(N^2) (the price of stability).
  P3 (overlay):    keep dual_pulse fast AND stable even at low gain -> the overlay
                   ESCAPES the trade-off (feedforward, not high-gain diffusion).

This runs only the NORMALIZED-gain runs (N x {baseline,dual_pulse}); the FIXED-gain
baseline comes from the existing runs/ directory (K_E_TAU=25), read for comparison.

GAIN_PRODUCT default 250 = 25 * 10  (so K_E_TAU(N=10) == the default 25).

Usage:
    python experiments/scaling_law/run_gain_sweep.py
"""
import os
import sys
import subprocess

import numpy as np
import pandas as pd

_THIS = os.path.abspath(__file__)
EXP_DIR = os.path.dirname(_THIS)
REPO_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))
MAIN_PY = os.path.join(REPO_ROOT, "main.py")
FIXED_RUNS = os.path.join(EXP_DIR, "runs")            # existing K_E_TAU=25 runs
NORM_RUNS = os.path.join(EXP_DIR, "gain_runs")        # new normalized-gain runs
RESULTS_CSV = os.path.join(EXP_DIR, "gain_results.csv")

METHODS = ["baseline", "dual_pulse"]
N_VALUES = [int(x) for x in os.environ.get("GAIN_N_VALUES", "24,40,50").split(",") if x.strip()]
GAIN_PRODUCT = float(os.environ.get("GAIN_PRODUCT", "250"))  # K_E_TAU = GAIN_PRODUCT / N
T0 = 5.0
RECOVERY_BUDGET = float(os.environ.get("GAIN_RECOVERY_BUDGET", "180"))  # normalized gain is slower
TAIL_FLOOR_FRAC = 0.05
LATE_WIN = 20.0  # s at end of run used for the stability (oscillation) metric


def victim_node_id(n, seed=0):
    return 2 + ((n // 2 + seed) % n)


def metrics_from_csv(tgt):
    if not os.path.exists(tgt):
        return {}
    df = pd.read_csv(tgt)
    full = df.copy()
    df = df[df["timestamp"] >= T0].reset_index(drop=True)
    if df.empty:
        return {}
    t = df["timestamp"].to_numpy(float)
    e = df["E_gap"].to_numpy(float)
    i_pk = int(np.nanargmax(e))
    e_pk = e[i_pk]
    floor = TAIL_FLOOR_FRAC * e_pk
    mask = (np.arange(e.size) >= i_pk) & (e > floor) & np.isfinite(e)
    tau, r2 = np.nan, np.nan
    if mask.sum() >= 5:
        tt, ee = t[mask], e[mask]
        coef = np.polyfit(tt, np.log(ee), 1)
        if coef[0] < 0:
            tau = -1.0 / coef[0]
        pred = np.polyval(coef, tt)
        ss_res = float(np.sum((np.log(ee) - pred) ** 2))
        ss_tot = float(np.sum((np.log(ee) - np.mean(np.log(ee))) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    # stability metric: std of E_gap over the last LATE_WIN seconds (settled -> ~0)
    t_end = float(full["timestamp"].max())
    late = full[full["timestamp"] >= t_end - LATE_WIN]["E_gap"].to_numpy(float)
    late_std = float(np.std(late)) if late.size else np.nan
    return {"tau_fit": float(tau), "tau_fit_r2": float(r2),
            "egap_final": float(e[-1]), "egap_late_std": late_std}


def run_normalized(method, n):
    k_e_tau = GAIN_PRODUCT / n
    victim = victim_node_id(n)
    duration = T0 + RECOVERY_BUDGET
    run_dir = os.path.join(NORM_RUNS, f"{method}_N{n}")
    os.makedirs(run_dir, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "PYTHONIOENCODING": "utf-8",
        "PROPAGATION_METHOD": method,
        "PROPAGATION_K_PROP": "0.0",
        "NUM_AGENTS": str(n),
        "SIM_DURATION": str(duration),
        "K_E_TAU": f"{k_e_tau:.6f}",
        "VIS_OPEN_BROWSER": "False",
        "SKIP_TELEMETRY_PLOTS": "True",
        "INIT_ANGLES_EQUIDISTANT": "True",
        "INIT_RADIUS_RANGE": "0.0",
        "TARGET_MOTION_SPEED_XY": "0.0",
        "DETERMINISTIC_FAILURE_ENABLE": "True",
        "DETERMINISTIC_FAILURE_AGENT_ID": str(victim),
        "DETERMINISTIC_FAILURE_TIME_T0": str(T0),
        "DETERMINISTIC_FAILURE_OFF_TIME": "-1.0",
        "RUNS_SUMMARY_CSV_PATH": os.path.join(run_dir, "runs_summary.csv"),
    })
    print(f"  -> normalized {method} N={n}  (K_E_TAU={k_e_tau:.2f}) ...", flush=True)
    proc = subprocess.run([sys.executable, MAIN_PY], cwd=run_dir, env=env, capture_output=True, text=True)
    tgt = os.path.join(run_dir, "target_telemetry.csv")
    if not os.path.exists(tgt):
        print(f"     FAILED (rc={proc.returncode})\n{(proc.stderr or '')[-800:]}")
        return None
    m = metrics_from_csv(tgt)
    m.update({"method": method, "N": n, "gain": "normalized", "k_e_tau": k_e_tau})
    print(f"     tau_fit={m.get('tau_fit'):.2f}  R2={m.get('tau_fit_r2'):.2f}  "
          f"egap_final={m.get('egap_final'):.4f}  late_std={m.get('egap_late_std'):.4f}")
    return m


def main():
    os.makedirs(NORM_RUNS, exist_ok=True)
    print(f"Gain-normalization test: N={N_VALUES}, GAIN_PRODUCT={GAIN_PRODUCT} "
          f"(normalized K_E_TAU=GAIN_PRODUCT/N), fixed baseline K_E_TAU=25\n")
    rows = []
    # normalized runs
    for n in N_VALUES:
        for method in METHODS:
            r = run_normalized(method, n)
            if r:
                rows.append(r)
                pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
    # fixed-gain comparison from existing runs/
    for n in N_VALUES:
        for method in METHODS:
            tgt = os.path.join(FIXED_RUNS, f"{method}_N{n}_s0", "target_telemetry.csv")
            m = metrics_from_csv(tgt)
            if m:
                m.update({"method": method, "N": n, "gain": "fixed", "k_e_tau": 25.0})
                rows.append(m)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)

    print("\n=== COMPARISON (fixed K_E_TAU=25  vs  normalized K_E_TAU=250/N) ===")
    print(f"{'method':11s} {'N':>3} {'gain':>11} {'K_E_TAU':>8} {'tau_fit':>8} {'R2':>5} "
          f"{'egap_final':>10} {'late_std':>9}  stability")
    for (method, n), g in df.groupby(["method", "N"]):
        for _, r in g.sort_values("gain").iterrows():
            stable = "OSCILLATING/UNSTABLE" if (r["egap_late_std"] > 0.02 or r["egap_final"] > 0.05) else "settled"
            print(f"{method:11s} {n:>3} {r['gain']:>11} {r['k_e_tau']:>8.2f} "
                  f"{r['tau_fit']:>8.2f} {r['tau_fit_r2']:>5.2f} {r['egap_final']:>10.4f} "
                  f"{r['egap_late_std']:>9.4f}  {stable}")

    # baseline exponent under normalized gain (P2: does it revert toward O(N^2)?)
    for method in METHODS:
        sub = df[(df.method == method) & (df.gain == "normalized")]
        sub = sub[(sub.tau_fit_r2 >= 0.85) & np.isfinite(sub.tau_fit)]
        if len(sub) >= 2:
            p = np.polyfit(np.log(sub["N"].to_numpy(float)), np.log(sub["tau_fit"].to_numpy(float)), 1)[0]
            print(f"\n{method} (normalized gain): tau ~ N^{p:.2f}")
    print(f"\nWrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
