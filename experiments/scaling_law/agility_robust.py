#!/usr/bin/env python
"""Re-analyze the agility runs with a FIT-ROBUST relaxation metric.

tau_fit assumes a single exponential, which fits poorly (low R^2) when the snappy
UAV produces a two-timescale decay (fast feedforward + slow residual). So we also
compute t_05pct = time (from t0) for E_gap to fall to 5% of its post-fault peak and
stay there 2 s -- no exponential assumption. We then compare the overlay advantage
computed both ways, to see whether the non-monotonic hump is real or a fit artifact.
"""
import os
import numpy as np
import pandas as pd

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(EXP_DIR, "agility_runs")
T0 = 5.0
SUSTAIN = 2.0
TAUS = [0.2, 0.5, 1.0, 2.0]


def sustained_cross(t, y, thr, window, dt):
    W = max(int(round(window / max(dt, 1e-12))), 1)
    ok = (y <= thr) & np.isfinite(y)
    if ok.size < W:
        return None
    s = np.convolve(ok.astype(int), np.ones(W, dtype=int), mode="valid")
    idx = np.where(s == W)[0]
    return float(t[int(idx[0])]) if idx.size else None


rows = []
for method in ("baseline", "dual_pulse"):
    for tau in TAUS:
        tgt = os.path.join(RUNS, f"{method}_N24_tau{tau:g}", "target_telemetry.csv")
        if not os.path.exists(tgt):
            continue
        df = pd.read_csv(tgt)
        df = df[df["timestamp"] >= T0].reset_index(drop=True)
        t = df["timestamp"].to_numpy(float)
        e = df["E_gap"].to_numpy(float)
        dt = float(np.median(np.diff(t))) if t.size > 1 else 0.01
        peak = float(np.nanmax(e))
        tc = sustained_cross(t, e, 0.05 * peak, SUSTAIN, dt)
        rows.append({"method": method, "tau_xy": tau,
                     "t_05pct": (tc - T0) if tc is not None else np.nan})

df = pd.DataFrame(rows)
piv = df.pivot_table(index="tau_xy", columns="method", values="t_05pct")
piv["advantage_t05"] = piv["baseline"] / piv["dual_pulse"]
print("Fit-robust metric  t_05pct (time to 5% of peak, sustained 2s):\n")
print(piv.to_string())
