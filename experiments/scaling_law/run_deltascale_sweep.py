#!/usr/bin/env python
"""Mechanism test: is the snappy-UAV degradation of the overlay tuning or fundamental?

At a fixed swarm size N and two UAV agilities (VM_TAU_XY in {0.2, 0.5}), sweep the
overlay's feedforward gain DUAL_PULSE_DELTA_SCALE in {0.5, 0.75, 1.0}. delta_scale
is the fraction of the analytically-computed redistribution shift that the overlay
feeds forward; 0.5 leaves half the work to the slow local controller (the source of
the two-timescale decay seen at small VM_TAU_XY).

INTERPRETATION:
  - if the overlay advantage RISES with delta_scale at the snappy UAV (tau=0.2),
    the agility-axis "downturn" is a TUNING artifact -> the right fix is to make
    delta_scale a function of the UAV agility (path B: dimensionless gains).
  - if it does NOT rise, the downturn is more FUNDAMENTAL (discrete pulse-latency
    floor N*CONTROL_PERIOD, or residual-consumption dynamics).

Baseline (delta_scale-independent) is run once per tau as the reference for the
advantage = tau_baseline / tau_overlay.

Usage:
    python experiments/scaling_law/run_deltascale_sweep.py
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
RUNS_DIR = os.path.join(EXP_DIR, "deltascale_runs")
RESULTS_CSV = os.path.join(EXP_DIR, "deltascale_results.csv")

N_FIXED = int(os.environ.get("AGILITY_N", "24"))
TAU_VALUES = [float(x) for x in os.environ.get("DS_TAU_VALUES", "0.2,0.5").split(",") if x.strip()]
SCALE_VALUES = [float(x) for x in os.environ.get("DS_SCALE_VALUES", "0.5,0.75,1.0").split(",") if x.strip()]
T0 = float(os.environ.get("SCALING_T0", "5.0"))
RECOVERY_BUDGET = float(os.environ.get("AGILITY_RECOVERY_BUDGET", "100"))
SUSTAIN = 2.0
TAIL_FLOOR_FRAC = 0.05


def victim_node_id(n, seed=0):
    return 2 + ((n // 2 + seed) % n)


def sustained_cross(t, y, thr, window, dt):
    W = max(int(round(window / max(dt, 1e-12))), 1)
    ok = (y <= thr) & np.isfinite(y)
    if ok.size < W:
        return None
    s = np.convolve(ok.astype(int), np.ones(W, dtype=int), mode="valid")
    idx = np.where(s == W)[0]
    return float(t[int(idx[0])]) if idx.size else None


def metrics_from_csv(tgt):
    df = pd.read_csv(tgt)
    df = df[df["timestamp"] >= T0].reset_index(drop=True)
    if df.empty:
        return {}
    t = df["timestamp"].to_numpy(float)
    e = df["E_gap"].to_numpy(float)
    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.01
    i_pk = int(np.nanargmax(e))
    e_pk = e[i_pk]
    # exp-fit tau on the decay tail
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
    tc = sustained_cross(t, e, 0.05 * e_pk, SUSTAIN, dt)
    t05 = (tc - T0) if tc is not None else np.nan
    return {"egap_peak": float(e_pk), "tau_fit": float(tau),
            "tau_fit_r2": float(r2), "t_05pct": float(t05)}


def run(method, tau_xy, scale=None):
    victim = victim_node_id(N_FIXED)
    duration = T0 + RECOVERY_BUDGET
    tag = f"{method}_N{N_FIXED}_tau{tau_xy:g}" + (f"_scale{scale:g}" if scale is not None else "")
    run_dir = os.path.join(RUNS_DIR, tag)
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
        "VM_TAU_XY": str(tau_xy),
        "DETERMINISTIC_FAILURE_ENABLE": "True",
        "DETERMINISTIC_FAILURE_AGENT_ID": str(victim),
        "DETERMINISTIC_FAILURE_TIME_T0": str(T0),
        "DETERMINISTIC_FAILURE_OFF_TIME": "-1.0",
        "RUNS_SUMMARY_CSV_PATH": os.path.join(run_dir, "runs_summary.csv"),
    })
    if scale is not None:
        env["DUAL_PULSE_DELTA_SCALE"] = str(scale)
    print(f"  -> {tag} ...", flush=True)
    proc = subprocess.run([sys.executable, MAIN_PY], cwd=run_dir, env=env, capture_output=True, text=True)
    tgt = os.path.join(run_dir, "target_telemetry.csv")
    if not os.path.exists(tgt):
        print(f"     FAILED (rc={proc.returncode})\n{(proc.stderr or '')[-800:]}")
        return None
    m = metrics_from_csv(tgt)
    m.update({"method": method, "tau_xy": tau_xy,
              "delta_scale": (scale if scale is not None else np.nan)})
    print(f"     tau_fit={m.get('tau_fit'):.3f}  R2={m.get('tau_fit_r2'):.3f}  t_05pct={m.get('t_05pct'):.3f}")
    return m


def main():
    os.makedirs(RUNS_DIR, exist_ok=True)
    print(f"delta_scale mechanism test: N={N_FIXED}, tau in {TAU_VALUES}, scale in {SCALE_VALUES}\n")
    rows = []
    base = {}  # (tau) -> baseline metrics
    for tau in TAU_VALUES:
        r = run("baseline", tau)
        if r:
            rows.append(r)
            base[tau] = r
        for scale in SCALE_VALUES:
            r = run("dual_pulse", tau, scale)
            if r:
                rows.append(r)
        pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    df = pd.DataFrame(rows)
    # advantage per (tau, scale)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for tau in TAU_VALUES:
        b = base.get(tau, {})
        dp = df[(df.method == "dual_pulse") & (df.tau_xy == tau)].sort_values("delta_scale")
        if dp.empty or not b:
            continue
        adv_tau = b["tau_fit"] / dp["tau_fit"]
        adv_t05 = b["t_05pct"] / dp["t_05pct"]
        axes[0].plot(dp["delta_scale"], adv_tau, "o-", label=f"tau_xy={tau}")
        axes[1].plot(dp["delta_scale"], adv_t05, "o-", label=f"tau_xy={tau}")
        print(f"\ntau_xy={tau}: baseline tau_fit={b['tau_fit']:.2f}, t_05pct={b['t_05pct']:.2f}")
        for _, r in dp.iterrows():
            print(f"   scale={r['delta_scale']:.2f}: overlay tau_fit={r['tau_fit']:.2f} (R2={r['tau_fit_r2']:.2f}) "
                  f"t_05pct={r['t_05pct']:.2f}  adv_tau={b['tau_fit']/r['tau_fit']:.2f}  adv_t05={b['t_05pct']/r['t_05pct']:.2f}")
    for ax, ttl in zip(axes, ["advantage (tau_fit)", "advantage (t_05pct, robust)"]):
        ax.axhline(1.0, color="gray", ls="--", alpha=0.6)
        ax.set_xlabel("DUAL_PULSE_DELTA_SCALE (feedforward gain)")
        ax.set_ylabel("overlay advantage")
        ax.set_title(ttl)
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend()
    fig.suptitle(f"Is the snappy-UAV downturn tuning or fundamental? (N={N_FIXED})")
    fig.tight_layout()
    out = os.path.join(EXP_DIR, "deltascale_advantage.png")
    fig.savefig(out, dpi=130)
    print(f"\nSaved {out}\nWrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
