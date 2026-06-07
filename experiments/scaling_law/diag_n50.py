#!/usr/bin/env python
"""Diagnose the N=50 breakdown: did the swarm equilibrate, or did the ring order
break (neighbor flapping)? Compares E_gap(t) across N and looks at per-agent e_tau
jitter (a flapping signature) at N=50.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(EXP, "runs")


def tgt(method, n):
    return os.path.join(RUNS, f"{method}_N{n}_s0", "target_telemetry.csv")


def agt(method, n):
    return os.path.join(RUNS, f"{method}_N{n}_s0", "agent_telemetry.csv")


fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: full-run E_gap for N=24, 40, 50 (did N=50 ever equilibrate?)
for n in (24, 40, 50):
    f = tgt("baseline", n)
    if not os.path.exists(f):
        continue
    d = pd.read_csv(f)
    axes[0].plot(d["timestamp"], d["E_gap"], label=f"baseline N={n}", lw=1.0)
axes[0].axvline(5.0, color="k", ls=":", alpha=0.5, label="t0 (fault)")
axes[0].set_yscale("log")
axes[0].set_xlabel("t [s]")
axes[0].set_ylabel("E_gap (log)")
axes[0].set_title("E_gap(t) full run -- does N=50 equilibrate?")
axes[0].grid(True, which="both", ls=":", alpha=0.5)
axes[0].legend(fontsize=8)

# Panel B: per-agent e_tau jitter at N=50 (high-frequency = neighbor flapping)
f = agt("baseline", 50)
if os.path.exists(f):
    d = pd.read_csv(f)
    col = "e_tau_real" if "e_tau_real" in d.columns else "e_tau"
    ids = sorted(d["node_id"].unique())
    pick = ids[:: max(1, len(ids) // 5)][:5]
    for nid in pick:
        di = d[d["node_id"] == nid]
        axes[1].plot(di["timestamp"], di[col], label=f"node {nid}", lw=0.7)
    axes[1].set_xlim(0, 20)
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel(col)
    axes[1].set_title("baseline N=50: per-agent e_tau (jitter => neighbor flapping)")
    axes[1].grid(True, ls=":", alpha=0.5)
    axes[1].legend(fontsize=8)

fig.tight_layout()
out = os.path.join(EXP, "diag_n50.png")
fig.savefig(out, dpi=130)
print("saved", out)

# Quantitative: did each N equilibrate before/after the fault?
print(f"\n{'N':>4} {'Egap_min':>10} {'Egap_max':>10} {'Egap_final':>11} {'Egap@t0-':>10}")
for n in (8, 24, 40, 50):
    f = tgt("baseline", n)
    if not os.path.exists(f):
        continue
    d = pd.read_csv(f)
    pre = d[d["timestamp"] < 5.0]["E_gap"]
    pre_min = float(pre.min()) if len(pre) else float("nan")
    print(f"{n:>4} {d['E_gap'].min():>10.4f} {d['E_gap'].max():>10.4f} "
          f"{d['E_gap'].iloc[-1]:>11.4f} {pre_min:>10.4f}")
