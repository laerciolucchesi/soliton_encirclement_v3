#!/usr/bin/env python
"""Generate the TTL>=N coverage data for fig23.

Single permanent fault, B2, measures the REDISTRIBUTION COVERAGE = fraction of
alive agents that ever received a non-zero dual_pulse_target (i.e. completed
their event: both counter-propagating pulses arrived). Swept over N for a
FIXED TTL=50 (the old default) vs an ADEQUATE TTL=3N.

If TTL truncates pulse circulation before the long-way pulse completes the ring,
distant agents never complete -> coverage collapses as N grows past ~TTL.
Expected (CLAUDE.md / H3): TTL=50 collapses coverage at N=100; TTL=3N stays ~1.0.

Short budget (~20 s after the fault) — coverage is decided in the first seconds
(pulses traverse the ring in ~diameter x tick).
"""
import os
import sys
import subprocess

import numpy as np
import pandas as pd

EXP = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(EXP))
MAIN = os.path.join(REPO, "main.py")
RUNS = os.path.join(EXP, "ttl_cov_runs")
T0 = 5.0
BUDGET = 20.0
N_VALUES = [40, 50, 75, 100]


def run_one(n, ttl, tag):
    run_dir = os.path.join(RUNS, tag)
    os.makedirs(run_dir, exist_ok=True)
    for fn in ("agent_telemetry.csv", "events.csv"):
        p = os.path.join(run_dir, fn)
        if os.path.exists(p):
            os.remove(p)
    env = dict(os.environ)
    env.update({
        "PYTHONIOENCODING": "utf-8", "PROPAGATION_METHOD": "dual_pulse",
        "PROPAGATION_K_PROP": "0.0", "NUM_AGENTS": str(n),
        "SIM_DURATION": str(T0 + BUDGET), "K_E_TAU": f"{250.0 / n:.6f}",
        "INIT_ANGLES_EQUIDISTANT": "True", "INIT_RADIUS_RANGE": "0.0",
        "TARGET_MOTION_SPEED_XY": "0.0", "VIS_OPEN_BROWSER": "False",
        "SKIP_TELEMETRY_PLOTS": "True",
        "DETERMINISTIC_FAILURE_ENABLE": "True",
        "DETERMINISTIC_FAILURE_AGENT_ID": str(2 + n // 2),
        "DETERMINISTIC_FAILURE_TIME_T0": str(T0),
        "DETERMINISTIC_FAILURE_OFF_TIME": "-1.0",
        "DUAL_PULSE_INTEGRATION": "B2", "DUAL_PULSE_DELTA_SCALE": "1.0",
        "DUAL_PULSE_T_FF": "1.0", "DUAL_PULSE_TTL_HOPS": str(ttl),
        "RUNS_SUMMARY_CSV_PATH": os.path.join(run_dir, "runs_summary.csv"),
    })
    subprocess.run([sys.executable, MAIN], cwd=run_dir, env=env,
                   capture_output=True, text=True)
    at = os.path.join(run_dir, "agent_telemetry.csv")
    if not os.path.exists(at):
        return float("nan"), 0, 0
    df = pd.read_csv(at)
    col = "dual_pulse_target" if "dual_pulse_target" in df.columns else "dual_pulse_shift"
    per = df.groupby("node_id")[col].apply(lambda s: float(np.nanmax(np.abs(s))))
    got = int((per > 1e-6).sum())
    total = int(per.size)
    try:
        os.remove(at)  # big file
    except OSError:
        pass
    return (got / total if total else float("nan")), total, got


def main():
    os.makedirs(RUNS, exist_ok=True)
    rows = []
    for n in N_VALUES:
        for mode, ttl in (("TTL=50 (fixo)", 50), ("TTL=3N (adequado)", 3 * n)):
            frac, total, got = run_one(n, ttl, f"N{n}_ttl{ttl}")
            print(f"  N={n:3d}  {mode:18s} TTL={ttl:3d}  "
                  f"cobertura={frac:5.1%}  ({got}/{total})")
            rows.append({"N": n, "mode": mode, "ttl": ttl,
                         "coverage_frac": frac, "agents_total": total,
                         "agents_with_shift": got})
    out = os.path.join(EXP, "ttl_coverage_results.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {os.path.relpath(out, EXP)}")


if __name__ == "__main__":
    main()
