#!/usr/bin/env python
"""Re-run the baseline (stable gain) at large N with an ADEQUATE budget.

The earlier large-N baselines (N=75/100) had a budget shorter than their true
O(N^2) relaxation, so their tau_fit was underestimated (the N^1.36 artifact).
Here the budget scales with the expected tau (~3.5x) so the slow tail is captured
and tau is measured cleanly. Combined with the clean gain_runs N=24/40/50, this
gives a fully-MEASURED baseline O(N^2) up to N=100 for the scaling-law figure.

Slow by nature (measuring something O(N^2) is O(N^2)); run in background.

Usage:
    python experiments/scaling_law/run_baseline_longbudget.py
    # $env:BL_NS="60,75,100"
"""
import os
import sys
import subprocess
import math

import numpy as np
import pandas as pd

_THIS = os.path.abspath(__file__)
EXP_DIR = os.path.dirname(_THIS)
REPO_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))
MAIN_PY = os.path.join(REPO_ROOT, "main.py")
RUNS = os.path.join(EXP_DIR, "baseline_long_runs")
RESULTS = os.path.join(EXP_DIR, "baseline_long_results.csv")

T0 = 5.0
GAIN_PRODUCT = 250.0
TAU50 = 85.35  # measured baseline tau at N=50 (stable gain)
TAIL_FLOOR_FRAC = 0.05
LATE_WIN = 20.0
NS = [int(x) for x in os.environ.get("BL_NS", "75,100").split(",") if x.strip()]


def tau_est(n):
    return TAU50 * (n / 50.0) ** 2


def budget(n):
    return max(200.0, 3.5 * tau_est(n))


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
    t_end = float(full["timestamp"].max())
    late = full[full["timestamp"] >= t_end - LATE_WIN]["E_gap"].to_numpy(float)
    late_std = float(np.std(late)) if late.size else np.nan
    return {"tau_fit": float(tau), "tau_fit_r2": float(r2),
            "egap_final": float(e[-1]), "egap_late_std": late_std}


def run(n):
    k = GAIN_PRODUCT / n
    b = budget(n)
    victim = 2 + (n // 2)
    run_dir = os.path.join(RUNS, f"baseline_N{n}")
    os.makedirs(run_dir, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "PYTHONIOENCODING": "utf-8",
        "PROPAGATION_METHOD": "baseline",
        "NUM_AGENTS": str(n),
        "SIM_DURATION": str(T0 + b),
        "K_E_TAU": f"{k:.6f}",
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
    print(f"  -> baseline N={n}  (K_E_TAU={k:.2f}, budget={b:.0f}s = ~3.5x tau_est {tau_est(n):.0f}) ...", flush=True)
    proc = subprocess.run([sys.executable, MAIN_PY], cwd=run_dir, env=env, capture_output=True, text=True)
    tgt = os.path.join(run_dir, "target_telemetry.csv")
    if not os.path.exists(tgt):
        print(f"     FAILED (rc={proc.returncode})\n{(proc.stderr or '')[-800:]}")
        return None
    m = metrics_from_csv(tgt)
    m.update({"N": n, "budget": b})
    print(f"     tau_fit={m.get('tau_fit'):.2f}  R2={m.get('tau_fit_r2'):.2f}  late_std={m.get('egap_late_std'):.4f}")
    return m


def main():
    os.makedirs(RUNS, exist_ok=True)
    print(f"Baseline re-run (adequate budget): N={NS}\n")
    rows = []
    for n in NS:
        r = run(n)
        if r:
            rows.append(r)
            pd.DataFrame(rows).to_csv(RESULTS, index=False)
    # combine with clean gain_runs N=24/40/50 for the O(N^2) fit
    GAIN = os.path.join(EXP_DIR, "gain_runs")
    for n in (24, 40, 50):
        m = metrics_from_csv(os.path.join(GAIN, f"baseline_N{n}", "target_telemetry.csv"))
        if m and not any(r["N"] == n for r in rows):
            m.update({"N": n, "budget": float("nan")})
            rows.append(m)
    df = pd.DataFrame(rows).sort_values("N")
    df.to_csv(RESULTS, index=False)
    print("\n=== baseline tau (clean budgets) ===")
    print(df[["N", "tau_fit", "tau_fit_r2", "budget"]].to_string(index=False))
    good = df[df["tau_fit_r2"] >= 0.9]
    if len(good) >= 2:
        p = np.polyfit(np.log(good["N"].to_numpy(float)), np.log(good["tau_fit"].to_numpy(float)), 1)[0]
        print(f"\nbaseline tau ~ N^{p:.2f}  (esperado ~2.0)")
    print(f"\nWrote {RESULTS}")


if __name__ == "__main__":
    main()
