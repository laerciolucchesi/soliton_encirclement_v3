#!/usr/bin/env python
"""Diagnose the N>=75 break: is it the TTL truncating pulse circulation?

If DUAL_PULSE_TTL_HOPS=50 kills pulses before they complete the ring (a node needs
the long-way pulse, up to ~N-1 hops), then at N=75 only the agents within ~TTL hops
of the originator should COMPLETE their event (get a delta_D / shift), and the rest
stay with zero/incomplete shift -> incomplete redistribution.

Coverage = fraction of alive agents that ever got a non-zero dual_pulse_target.
Also counts dual_pulse completion events in events.csv. Compares N=50 (worked) vs
N=75 / N=100 (broke), from largeN_runs/.
"""
import os
import numpy as np
import pandas as pd

EXP = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(EXP, "largeN_runs")


def coverage(run_dir, n):
    out = {"N": n}
    # per-agent shift coverage (agent_telemetry: dual_pulse_target)
    at = os.path.join(run_dir, "agent_telemetry.csv")
    if os.path.exists(at):
        df = pd.read_csv(at)
        col = "dual_pulse_target" if "dual_pulse_target" in df.columns else "dual_pulse_shift"
        per = df.groupby("node_id")[col].apply(lambda s: float(np.nanmax(np.abs(s))))
        got = int((per > 1e-6).sum())
        out["agents_total"] = int(per.size)
        out["agents_with_shift"] = got
        out["coverage_frac"] = got / per.size if per.size else float("nan")
    # completion events (events.csv)
    ev = os.path.join(run_dir, "events.csv")
    if os.path.exists(ev):
        e = pd.read_csv(ev)
        et = e["event_type"].astype(str)
        out["n_completed"] = int(et.str.contains("dual_pulse_event_completed").sum())
        out["n_self_shift"] = int(et.str.contains("dual_pulse_self_shift").sum())
        # max hop reached (proxy for TTL truncation)
        for c in ("h_CCW", "h_CW"):
            if c in e.columns:
                vals = pd.to_numeric(e[c], errors="coerce")
                out[f"max_{c}"] = float(np.nanmax(vals)) if vals.notna().any() else np.nan
    return out


print(f"{'N':>4} {'agents':>7} {'with_shift':>11} {'coverage':>9} {'n_completed':>12} {'n_self':>7} {'max_hCCW':>9} {'max_hCW':>9}")
for n in (50, 75, 100):
    rd = os.path.join(RUNS, f"B2_N{n}_s0")
    if not os.path.isdir(rd):
        continue
    c = coverage(rd, n)
    print(f"{c.get('N'):>4} {c.get('agents_total','?'):>7} {c.get('agents_with_shift','?'):>11} "
          f"{c.get('coverage_frac', float('nan')):>9.3f} {c.get('n_completed','?'):>12} "
          f"{c.get('n_self_shift','?'):>7} {c.get('max_h_CCW', float('nan')):>9.0f} "
          f"{c.get('max_h_CW', float('nan')):>9.0f}")
print("\nTTL atual = 50. Se a coverage cai em N=75/100 e max_hop satura ~50 -> TTL confirmado.")
