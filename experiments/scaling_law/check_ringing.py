#!/usr/bin/env python
"""Hard numeric test for overshoot/ringing in the dual_pulse agility runs.

If a snappy UAV overshoots the new equilibrium, E_gap dips toward 0 as the node
crosses the correct spacing, then RISES again as it swings past. So: find the
global minimum of E_gap after the post-fault peak, then the max E_gap AFTER that
minimum. rebound = (max after min) / (min). rebound >> 1 == clear overshoot.
A clean exponential decay never rebounds (rebound ~ 1).
"""
import os
import numpy as np
import pandas as pd

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(EXP_DIR, "agility_runs")
T0 = 5.0
WIN = 25.0  # s after t0 to inspect

print(f"{'run':28s} {'min_Egap':>10s} {'max_after_min':>14s} {'rebound':>9s}")
for method in ("dual_pulse", "baseline"):
    for tau in (0.2, 0.5, 1.0, 2.0):
        tgt = os.path.join(RUNS, f"{method}_N24_tau{tau:g}", "target_telemetry.csv")
        if not os.path.exists(tgt):
            continue
        df = pd.read_csv(tgt)
        df = df[(df["timestamp"] >= T0) & (df["timestamp"] <= T0 + WIN)]
        e = df["E_gap"].to_numpy(float)
        i_pk = int(np.nanargmax(e))
        tail = e[i_pk:]
        i_min = int(np.nanargmin(tail))
        e_min = float(tail[i_min])
        after = tail[i_min:]
        e_max_after = float(np.nanmax(after)) if after.size else e_min
        rebound = e_max_after / e_min if e_min > 1e-9 else float("nan")
        flag = "  <-- RINGING" if rebound > 1.5 else ""
        print(f"{method}_tau{tau:g}".ljust(28) +
              f" {e_min:10.4f} {e_max_after:14.4f} {rebound:9.2f}{flag}")
