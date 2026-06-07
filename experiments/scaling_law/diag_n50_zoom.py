#!/usr/bin/env python
"""Zoom: distinguish a smooth control-loop oscillation (gain instability) from
jagged neighbor flapping (hysteresis) in baseline N=50 after the fault.

Smooth sinusoid in e_tau + velocity => loop instability (effective gain too high).
Jagged steps / discontinuities => neighbor identity flapping (hysteresis vs gap).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = os.path.dirname(os.path.abspath(__file__))
f = os.path.join(EXP, "runs", "baseline_N50_s0", "agent_telemetry.csv")
d = pd.read_csv(f)
col = "e_tau_real" if "e_tau_real" in d.columns else "e_tau"
ids = sorted(d["node_id"].unique())
pick = ids[:3]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
for nid in pick:
    di = d[(d["node_id"] == nid) & (d["timestamp"] >= 3) & (d["timestamp"] <= 18)]
    ax1.plot(di["timestamp"], di[col], label=f"node {nid}", lw=1.0)
    ax2.plot(di["timestamp"], di["velocity_norm"], label=f"node {nid}", lw=1.0)
ax1.axvline(5.0, color="k", ls=":", alpha=0.5)
ax2.axvline(5.0, color="k", ls=":", alpha=0.5)
ax1.set_ylabel(col)
ax1.set_title("baseline N=50, zoom around fault (t0=5): smooth sinusoid=gain instability; jagged=flapping")
ax2.set_ylabel("velocity_norm [m/s]")
ax2.set_xlabel("t [s]")
for ax in (ax1, ax2):
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=9)
fig.tight_layout()
out = os.path.join(EXP, "diag_n50_zoom.png")
fig.savefig(out, dpi=140)
print("saved", out)

# numeric: is it oscillatory? estimate dominant period from zero-crossings of node 2's e_tau after t0
di = d[(d["node_id"] == ids[0]) & (d["timestamp"] >= 5.5)]
y = di[col].to_numpy(float)
t = di["timestamp"].to_numpy(float)
sign = np.sign(y - np.median(y))
crossings = np.where(np.diff(sign) != 0)[0]
if len(crossings) >= 2:
    periods = 2 * np.diff(t[crossings])
    print(f"node {ids[0]}: ~{len(crossings)} zero-crossings after t0, "
          f"median oscillation period ~ {np.median(periods):.2f} s")
print(f"post-t0 e_tau std (node {ids[0]}): {np.std(y):.4f}")
