#!/usr/bin/env python
"""Option B test: does direct feedforward let the overlay ESCAPE the trilemma?

At STABLE (normalized) gain K_E_TAU=250/N -- where Option A gave only a modest,
shrinking advantage and the baseline is O(N^2) -- run the dual_pulse overlay with
the Option-B integration (DUAL_PULSE_INTEGRATION=B): a direct tangential feedforward
(time constant T_FF, gain-independent) executes the redistribution, while the
low-gain feedback sees a neighbour-shift cancelling bias and only fixes residual.

Prediction if Option B escapes the trilemma:
  - STABLE (no limit cycle; low late_std, egap_final ~ 0), like Option A at low gain.
  - FAST and ~N-INDEPENDENT (tau_fit ~ a few * T_FF, not growing like baseline's O(N^2)).
  - advantage_B = tau_baseline / tau_overlayB  GROWS with N (or stays large), unlike
    Option A whose advantage shrank (1.67 -> 1.30 -> 1.14).

Baseline and Option A (both normalized gain) are read from gain_runs/ (already computed).

Usage:
    python experiments/scaling_law/run_optionB_test.py
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
GAIN_RUNS = os.path.join(EXP_DIR, "gain_runs")          # baseline + Option A (normalized gain)
_TAG = os.environ.get("OPTB_TAG", "")
_SUF = ("_" + _TAG) if _TAG else ""
B_RUNS = os.path.join(EXP_DIR, "optionB_runs" + _SUF)
RESULTS_CSV = os.path.join(EXP_DIR, "optionB_results" + _SUF + ".csv")

N_VALUES = [int(x) for x in os.environ.get("OPTB_N_VALUES", "24,40,50").split(",") if x.strip()]
GAIN_PRODUCT = float(os.environ.get("GAIN_PRODUCT", "250"))   # K_E_TAU = GAIN_PRODUCT / N (stable)
T_FF = float(os.environ.get("OPTB_T_FF", "1.0"))
T0 = 5.0
RECOVERY_BUDGET = float(os.environ.get("OPTB_RECOVERY_BUDGET", "120"))
TAIL_FLOOR_FRAC = 0.05
LATE_WIN = 20.0


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
    t_end = float(full["timestamp"].max())
    late = full[full["timestamp"] >= t_end - LATE_WIN]["E_gap"].to_numpy(float)
    late_std = float(np.std(late)) if late.size else np.nan
    return {"tau_fit": float(tau), "tau_fit_r2": float(r2),
            "egap_final": float(e[-1]), "egap_late_std": late_std}


def run_optionB(n):
    k_e_tau = GAIN_PRODUCT / n
    victim = victim_node_id(n)
    duration = T0 + RECOVERY_BUDGET
    run_dir = os.path.join(B_RUNS, f"dual_pulse_N{n}")
    os.makedirs(run_dir, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "PYTHONIOENCODING": "utf-8",
        "PROPAGATION_METHOD": "dual_pulse",
        "PROPAGATION_K_PROP": "0.0",
        "NUM_AGENTS": str(n),
        "SIM_DURATION": str(duration),
        "K_E_TAU": f"{k_e_tau:.6f}",
        "DUAL_PULSE_INTEGRATION": os.environ.get("OPTB_INTEGRATION", "B"),
        "DUAL_PULSE_T_FF": f"{T_FF:.4f}",
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
    print(f"  -> Option B  dual_pulse N={n}  (K_E_TAU={k_e_tau:.2f}, T_FF={T_FF}) ...", flush=True)
    proc = subprocess.run([sys.executable, MAIN_PY], cwd=run_dir, env=env, capture_output=True, text=True)
    tgt = os.path.join(run_dir, "target_telemetry.csv")
    if not os.path.exists(tgt):
        print(f"     FAILED (rc={proc.returncode})\n{(proc.stderr or '')[-1000:]}")
        return None
    m = metrics_from_csv(tgt)
    m.update({"N": n, "variant": "B", "k_e_tau": k_e_tau})
    print(f"     tau_fit={m.get('tau_fit'):.2f}  R2={m.get('tau_fit_r2'):.2f}  "
          f"egap_final={m.get('egap_final'):.4f}  late_std={m.get('egap_late_std'):.4f}")
    return m


def main():
    os.makedirs(B_RUNS, exist_ok=True)
    print(f"Option B test: N={N_VALUES}, normalized gain K_E_TAU={GAIN_PRODUCT}/N, T_FF={T_FF}\n")
    rows = []
    for n in N_VALUES:
        r = run_optionB(n)
        if r:
            rows.append(r)
        # baseline + Option A from gain_runs (normalized gain)
        for variant, dirname in (("baseline", "baseline"), ("A", "dual_pulse")):
            tgt = os.path.join(GAIN_RUNS, f"{dirname}_N{n}", "target_telemetry.csv")
            m = metrics_from_csv(tgt)
            if m:
                m.update({"N": n, "variant": variant, "k_e_tau": GAIN_PRODUCT / n})
                rows.append(m)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)

    print("\n=== STABLE-GAIN COMPARISON (baseline vs Option A vs Option B) ===")
    print(f"{'N':>3} {'variant':>8} {'tau_fit':>8} {'R2':>5} {'egap_final':>10} {'late_std':>9}  {'advantage':>9}  stability")
    for n in N_VALUES:
        sub = df[df.N == n]
        base = sub[sub.variant == "baseline"]
        base_tau = float(base["tau_fit"].iloc[0]) if len(base) else float("nan")
        for variant in ("baseline", "A", "B"):
            r = sub[sub.variant == variant]
            if not len(r):
                continue
            r = r.iloc[0]
            adv = base_tau / r["tau_fit"] if (variant != "baseline" and np.isfinite(r["tau_fit"]) and r["tau_fit"] > 0) else float("nan")
            stable = "UNSTABLE" if (r["egap_late_std"] > 0.02 or r["egap_final"] > 0.05) else "settled"
            print(f"{n:>3} {variant:>8} {r['tau_fit']:>8.2f} {r['tau_fit_r2']:>5.2f} "
                  f"{r['egap_final']:>10.4f} {r['egap_late_std']:>9.4f}  {adv:>9.2f}  {stable}")
    print(f"\nWrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
