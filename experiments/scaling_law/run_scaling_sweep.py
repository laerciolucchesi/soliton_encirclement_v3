#!/usr/bin/env python
"""Scaling-law experiment: post-failure stabilization time vs swarm size N.

DE-RISKING EXPERIMENT for the thesis spine
------------------------------------------
Central hypothesis (CS / distributed-systems framing): the baseline local
controller re-stabilizes uniform ring spacing in O(N^2) rounds (information
diffuses through nearest-neighbour coupling), while the dual_pulse dissemination
overlay re-stabilizes in O(N) rounds (information crosses the ring ballistically,
matching the diameter lower bound). If true, a log-log plot of stabilization
time vs N shows two different slopes: baseline ~2, dual_pulse ~1.

To measure this cleanly we inject EXACTLY ONE deterministic crash at a fixed
time t0 (no random Poisson stream) and time how long the swarm takes to re-reach
uniform spacing, as a function of N, for baseline vs dual_pulse.

Stabilization signal: G_max from target_telemetry.csv = max(gap / ideal_gap)
over alive agents. At uniform spacing G_max -> 1. We report the first time after
t0 at which G_max <= GMAX_TOL continuously for SETTLE_WINDOW seconds. G_max is
used (rather than the RMS E_gap) because a single bad gap keeps a consistent
peak (~2x ideal) across all N, so the perturbation magnitude does not get
RMS-diluted as N grows -- a more N-robust settling signal.

Isolation: each run executes `python main.py` with cwd set to its own output
directory, so all CSV outputs land there and the repo root stays clean. main.py
resolves imports via its own __file__, so a foreign cwd is fine.

Usage (PowerShell, from repo root):
    python experiments/scaling_law/run_scaling_sweep.py
    python experiments/scaling_law/plot_scaling.py

Grid is overridable by env var without editing this file, e.g.:
    $env:SCALING_N_VALUES="8,12,16,24,32"; $env:SCALING_RECOVERY_BUDGET="120"
    python experiments/scaling_law/run_scaling_sweep.py
"""

import os
import sys
import subprocess

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
_THIS = os.path.abspath(__file__)
EXP_DIR = os.path.dirname(_THIS)
REPO_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))
MAIN_PY = os.path.join(REPO_ROOT, "main.py")
RUNS_DIR = os.path.join(EXP_DIR, "runs")
RESULTS_CSV = os.path.join(EXP_DIR, "scaling_results.csv")

# --------------------------------------------------------------------------
# Experiment grid (env-overridable so the file stays the committed default)
# --------------------------------------------------------------------------
def _env_list_int(name, default):
    raw = os.environ.get(name, default)
    return [int(x) for x in raw.split(",") if x.strip()]


METHODS = [m.strip() for m in os.environ.get("SCALING_METHODS", "baseline,dual_pulse").split(",") if m.strip()]
# SMOKE default: small + short so the pipeline validates in a few minutes.
# Real campaign: e.g. SCALING_N_VALUES="10,15,20,30,50,75,100" and raise budget.
N_VALUES = _env_list_int("SCALING_N_VALUES", "8,12,16")
SEEDS = _env_list_int("SCALING_SEEDS", "0")          # each "seed" = a distinct victim node (symmetry check)
T0 = float(os.environ.get("SCALING_T0", "5.0"))      # crash time (s); > warmup (1.0) and after initial settle
RECOVERY_BUDGET = float(os.environ.get("SCALING_RECOVERY_BUDGET", "90"))  # s of sim after t0 to allow settling

# Stabilization tolerances
GMAX_TOL = float(os.environ.get("SCALING_GMAX_TOL", "1.05"))   # worst gap within 5% of ideal
EGAP_TOL = float(os.environ.get("SCALING_EGAP_TOL", "0.05"))   # RMS normalized gap error
SETTLE_WINDOW = float(os.environ.get("SCALING_SETTLE_WINDOW", "3.0"))  # s tolerance must hold continuously


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def victim_node_id(n_agents: int, seed: int) -> int:
    """Pick a mid-ring agent (node ids are 2..n_agents+1; target=0, adversary=1).

    `seed` rotates the victim around the ring for a symmetry/robustness check.
    """
    return 2 + ((n_agents // 2 + seed) % n_agents)


def settling_time(t, y, tol, window, dt, leq=True):
    """First t_s such that (y <= tol) [or >= tol] holds continuously for `window`."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size == 0:
        return None
    W = max(int(round(window / max(dt, 1e-12))), 1)
    ok = (y <= tol) if leq else (y >= tol)
    ok = ok & np.isfinite(y)
    if ok.size < W:
        return None
    s = np.convolve(ok.astype(int), np.ones(W, dtype=int), mode="valid")
    idx = np.where(s == W)[0]
    if idx.size == 0:
        return None
    return float(t[int(idx[0])])


def run_one(method: str, n_agents: int, seed: int):
    victim = victim_node_id(n_agents, seed)
    duration = T0 + RECOVERY_BUDGET
    run_name = f"{method}_N{n_agents}_s{seed}"
    run_dir = os.path.join(RUNS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    env = dict(os.environ)
    env.update({
        "PYTHONIOENCODING": "utf-8",
        "PROPAGATION_METHOD": method,
        "PROPAGATION_K_PROP": "0.0",
        "NUM_AGENTS": str(n_agents),
        "SIM_DURATION": str(duration),
        "VIS_OPEN_BROWSER": "False",
        "SKIP_TELEMETRY_PLOTS": "True",
        "INIT_ANGLES_EQUIDISTANT": "True",
        "INIT_RADIUS_RANGE": "0.0",
        "TARGET_MOTION_SPEED_XY": "0.0",
        "EXPERIMENT_SEED": str(seed),
        "DETERMINISTIC_FAILURE_ENABLE": "True",
        "DETERMINISTIC_FAILURE_AGENT_ID": str(victim),
        "DETERMINISTIC_FAILURE_TIME_T0": str(T0),
        "DETERMINISTIC_FAILURE_OFF_TIME": "-1.0",   # permanent crash -> single SAIDA event
        "RUNS_SUMMARY_CSV_PATH": os.path.join(run_dir, "runs_summary.csv"),
    })

    print(f"  -> {run_name}  (victim node {victim}, duration {duration:.0f}s) ...", flush=True)
    proc = subprocess.run(
        [sys.executable, MAIN_PY],
        cwd=run_dir, env=env, capture_output=True, text=True,
    )

    tgt = os.path.join(run_dir, "target_telemetry.csv")
    if not os.path.exists(tgt):
        print(f"     FAILED: no target_telemetry.csv (rc={proc.returncode})")
        print("     --- stdout tail ---\n" + (proc.stdout or "")[-1500:])
        print("     --- stderr tail ---\n" + (proc.stderr or "")[-1500:])
        return None

    df = pd.read_csv(tgt)
    df = df[df["timestamp"] >= T0]
    if df.empty:
        print("     FAILED: no post-t0 rows in target_telemetry.csv")
        return None

    t = df["timestamp"].to_numpy(dtype=float)
    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.01
    ts_g = settling_time(t, df["G_max"].to_numpy(float), GMAX_TOL, SETTLE_WINDOW, dt, leq=True)
    ts_e = settling_time(t, df["E_gap"].to_numpy(float), EGAP_TOL, SETTLE_WINDOW, dt, leq=True)

    # Message overhead (dual_pulse only)
    n_msgs = np.nan
    msg_csv = os.path.join(run_dir, "dual_pulse_messages.csv")
    if os.path.exists(msg_csv):
        try:
            n_msgs = int(pd.read_csv(msg_csv)["pulse_payloads_broadcast"].sum())
        except Exception:
            pass

    res = {
        "method": method, "N": n_agents, "seed": seed, "victim_node": victim,
        "t0": T0, "sim_duration": duration,
        "t_stab_gmax": (ts_g - T0) if ts_g is not None else np.nan,
        "t_stab_egap": (ts_e - T0) if ts_e is not None else np.nan,
        "settled_gmax": bool(ts_g is not None),
        "gmax_peak": float(np.nanmax(df["G_max"].to_numpy(float))),
        "egap_final": float(df["E_gap"].to_numpy(float)[-1]),
        "n_pulse_payloads": n_msgs,
    }
    print(
        f"     t_stab(Gmax)={res['t_stab_gmax']!s:>8}  "
        f"t_stab(Egap)={res['t_stab_egap']!s:>8}  "
        f"peakGmax={res['gmax_peak']:.2f}  msgs={n_msgs}"
    )
    return res


def main():
    os.makedirs(RUNS_DIR, exist_ok=True)
    print("Scaling-law sweep")
    print(f"  methods = {METHODS}")
    print(f"  N       = {N_VALUES}")
    print(f"  seeds   = {SEEDS}  (each = a rotated victim node)")
    print(f"  t0={T0}s  recovery_budget={RECOVERY_BUDGET}s  "
          f"Gmax_tol={GMAX_TOL}  settle_window={SETTLE_WINDOW}s")
    print()

    rows = []
    for method in METHODS:
        for n in N_VALUES:
            for s in SEEDS:
                r = run_one(method, n, s)
                if r:
                    rows.append(r)
                    pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)  # incremental save

    if rows:
        print(f"\nWrote {RESULTS_CSV}  ({len(rows)} runs).")
        print("Now: python experiments/scaling_law/plot_scaling.py")
    else:
        print("\nNo successful runs -- check the stdout/stderr tails above.")


if __name__ == "__main__":
    main()
