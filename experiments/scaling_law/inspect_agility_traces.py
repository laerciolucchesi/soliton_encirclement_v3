#!/usr/bin/env python
"""Diagnostic: overlay E_gap(t) decay for dual_pulse across UAV agility.

Tests whether the poor exponential fits (low R^2) at small VM_TAU_XY are caused by
overshoot/ringing (a snappy actuator + aggressive feedforward overshooting the new
equilibrium) rather than a clean exponential relaxation. On a log-y axis a clean
exponential decay is a straight line; ringing shows as wiggles / non-monotonic dips.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(EXP_DIR, "agility_runs")
T0 = 5.0
TAUS = [0.2, 0.5, 1.0, 2.0]

fig, ax = plt.subplots(figsize=(9, 5.6))
for tau in TAUS:
    tgt = os.path.join(RUNS, f"dual_pulse_N24_tau{tau:g}", "target_telemetry.csv")
    if not os.path.exists(tgt):
        continue
    df = pd.read_csv(tgt)
    df = df[df["timestamp"] >= T0]
    t = df["timestamp"].to_numpy(float) - T0
    e = df["E_gap"].to_numpy(float)
    m = t <= 30
    ax.plot(t[m], e[m], label=f"dual_pulse  tau_xy={tau}")

tgt = os.path.join(RUNS, "baseline_N24_tau1", "target_telemetry.csv")
if os.path.exists(tgt):
    df = pd.read_csv(tgt)
    df = df[df["timestamp"] >= T0]
    t = df["timestamp"].to_numpy(float) - T0
    e = df["E_gap"].to_numpy(float)
    m = t <= 30
    ax.plot(t[m], e[m], "k--", alpha=0.5, label="baseline tau_xy=1 (ref)")

ax.set_yscale("log")
ax.set_xlabel("time since fault [s]")
ax.set_ylabel("E_gap (log scale)")
ax.set_title("E_gap(t) after fault -- dual_pulse across UAV agility (N=24)\n"
             "straight line = clean exponential; wiggles/dips = overshoot/ringing")
ax.grid(True, which="both", ls=":", alpha=0.5)
ax.legend()
fig.tight_layout()
out = os.path.join(EXP_DIR, "agility_traces.png")
fig.savefig(out, dpi=130)
print("saved", out)
