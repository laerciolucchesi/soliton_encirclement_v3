#!/usr/bin/env python
"""Agility-axis experiment: overlay advantage vs UAV responsiveness (the Pe axis).

Fixes the swarm size N and sweeps VM_TAU_XY -- the UAV's first-order tracking lag,
i.e. the "sluggishness" knob (smaller = snappier UAV). For each agility it measures
the slow-mode relaxation time tau_fit for baseline vs dual_pulse, and the overlay
advantage = tau_baseline / tau_overlay.

HYPOTHESIS (to be TESTED, not assumed): a snappier UAV is more information-limited,
so the overlay should help more (advantage grows as VM_TAU_XY shrinks). But the
relationship might instead be flat (if both methods' relaxation scales with the
actuation time, the ratio cancels) or even NON-monotonic (a very snappy UAV lets
the baseline diffuse so fast that the overlay's own discrete pulse latency becomes
the bottleneck, shrinking or inverting the advantage). The experiment decides --
this is exactly the Péclet (information-limited vs actuation-limited) boundary.

Usage:
    python experiments/scaling_law/run_agility_sweep.py
    # overrides: $env:AGILITY_TAU_VALUES="0.1,0.3,0.6,1.0,2.0"; $env:AGILITY_N="24"
"""

import os
import sys
import subprocess

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS = os.path.abspath(__file__)
EXP_DIR = os.path.dirname(_THIS)
REPO_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))
MAIN_PY = os.path.join(REPO_ROOT, "main.py")
RUNS_DIR = os.path.join(EXP_DIR, "agility_runs")
RESULTS_CSV = os.path.join(EXP_DIR, "agility_results.csv")

METHODS = ["baseline", "dual_pulse"]
N_FIXED = int(os.environ.get("AGILITY_N", "24"))
TAU_VALUES = [float(x) for x in os.environ.get("AGILITY_TAU_VALUES", "0.2,0.5,1.0,2.0").split(",") if x.strip()]
T0 = float(os.environ.get("SCALING_T0", "5.0"))
RECOVERY_BUDGET = float(os.environ.get("AGILITY_RECOVERY_BUDGET", "100"))
TAIL_FLOOR_FRAC = 0.05


def victim_node_id(n, seed=0):
    return 2 + ((n // 2 + seed) % n)


def tau_fit_from_csv(tgt):
    """Slow-mode relaxation time = exp-fit time constant of the E_gap decay tail."""
    df = pd.read_csv(tgt)
    df = df[df["timestamp"] >= T0].reset_index(drop=True)
    if df.empty:
        return (np.nan, np.nan, np.nan)
    t = df["timestamp"].to_numpy(float)
    e = df["E_gap"].to_numpy(float)
    i_pk = int(np.nanargmax(e))
    e_pk = e[i_pk]
    floor = TAIL_FLOOR_FRAC * e_pk
    mask = (np.arange(e.size) >= i_pk) & (e > floor) & np.isfinite(e)
    if mask.sum() < 5:
        return (np.nan, np.nan, float(e_pk))
    tt, ee = t[mask], e[mask]
    coef = np.polyfit(tt, np.log(ee), 1)
    slope = coef[0]
    tau = (-1.0 / slope) if slope < 0 else np.nan
    pred = np.polyval(coef, tt)
    ss_res = float(np.sum((np.log(ee) - pred) ** 2))
    ss_tot = float(np.sum((np.log(ee) - np.mean(np.log(ee))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return (float(tau), float(r2), float(e_pk))


def run_one(method, tau_xy):
    victim = victim_node_id(N_FIXED)
    duration = T0 + RECOVERY_BUDGET
    run_name = f"{method}_N{N_FIXED}_tau{tau_xy:g}"
    run_dir = os.path.join(RUNS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "PYTHONIOENCODING": "utf-8",
        "PROPAGATION_METHOD": method,
        "PROPAGATION_K_PROP": "0.0",
        "NUM_AGENTS": str(N_FIXED),
        "SIM_DURATION": str(duration),
        "VIS_OPEN_BROWSER": "False",
        "SKIP_TELEMETRY_PLOTS": "True",
        "INIT_ANGLES_EQUIDISTANT": "True",
        "INIT_RADIUS_RANGE": "0.0",
        "TARGET_MOTION_SPEED_XY": "0.0",
        "VM_TAU_XY": str(tau_xy),                       # the agility knob
        "DETERMINISTIC_FAILURE_ENABLE": "True",
        "DETERMINISTIC_FAILURE_AGENT_ID": str(victim),
        "DETERMINISTIC_FAILURE_TIME_T0": str(T0),
        "DETERMINISTIC_FAILURE_OFF_TIME": "-1.0",
        "RUNS_SUMMARY_CSV_PATH": os.path.join(run_dir, "runs_summary.csv"),
    })
    print(f"  -> {run_name} ...", flush=True)
    proc = subprocess.run([sys.executable, MAIN_PY], cwd=run_dir, env=env, capture_output=True, text=True)
    tgt = os.path.join(run_dir, "target_telemetry.csv")
    if not os.path.exists(tgt):
        print(f"     FAILED (rc={proc.returncode})")
        print((proc.stderr or "")[-1000:])
        return None
    tau, r2, epk = tau_fit_from_csv(tgt)
    print(f"     tau_fit={tau:.3f}  R2={r2:.3f}  egap_peak={epk:.3f}")
    return {"method": method, "N": N_FIXED, "tau_xy": tau_xy,
            "tau_fit": tau, "tau_fit_r2": r2, "egap_peak": epk}


def main():
    os.makedirs(RUNS_DIR, exist_ok=True)
    print(f"Agility sweep: N={N_FIXED}, VM_TAU_XY in {TAU_VALUES}, "
          f"t0={T0}s, recovery_budget={RECOVERY_BUDGET}s\n")
    rows = []
    for method in METHODS:
        for tau in TAU_VALUES:
            r = run_one(method, tau)
            if r:
                rows.append(r)
                pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    if not rows:
        print("\nNo successful runs.")
        return
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="tau_xy", columns="method", values="tau_fit")
    if {"baseline", "dual_pulse"}.issubset(piv.columns):
        piv["advantage"] = piv["baseline"] / piv["dual_pulse"]
    print("\n" + piv.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for method, g in df.groupby("method"):
        gg = g.sort_values("tau_xy")
        axes[0].plot(gg["tau_xy"], gg["tau_fit"], "o-", label=method)
    axes[0].set_xlabel("VM_TAU_XY  (UAV lag, s)  -- smaller = snappier")
    axes[0].set_ylabel("tau_fit  (slow-mode relaxation, s)")
    axes[0].set_title(f"Relaxation vs UAV agility (N={N_FIXED})")
    axes[0].grid(True, ls=":", alpha=0.5)
    axes[0].legend()
    if "advantage" in piv.columns:
        axes[1].plot(piv.index, piv["advantage"], "o-", color="purple")
        axes[1].axhline(1.0, color="gray", ls="--", alpha=0.6)
        axes[1].set_xlabel("VM_TAU_XY  (UAV lag, s)  -- smaller = snappier")
        axes[1].set_ylabel("overlay advantage  (tau_baseline / tau_overlay)")
        axes[1].set_title("Does a snappier UAV widen the overlay advantage?")
        axes[1].grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    out = os.path.join(EXP_DIR, "agility_advantage.png")
    fig.savefig(out, dpi=130)
    print(f"\nSaved {out}\nWrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
