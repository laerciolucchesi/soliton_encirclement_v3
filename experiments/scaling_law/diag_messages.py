#!/usr/bin/env python
"""Message-complexity of the dual_pulse overlay (the CS cost metric).

Counts the pulse payloads emitted per fault (summed across agents from
dual_pulse_messages.csv) vs N, and fits the scaling. The pulse protocol is the
SAME for Option A/B/B2 (only the integration of the computed shift differs), so
this measures the dissemination overhead of the overlay regardless of integration.

Uses runs where the pulses COMPLETE the ring (TTL >= N):
  N<=50  from optionB_runs_B2scale1/  (B2, TTL=50, complete since N<=50)
  N>50   from largeN_runs/B2_N*_s0/   (B2, TTL=3N, complete)

NOTE on dp_shift overhead (Option B/B2): the shift_remaining scalar rides on the
AgentState that every agent already broadcasts EVERY tick, so it is a constant
BYTE overhead per existing broadcast, NOT extra messages. The pulse payloads below
are the event-triggered overhead specific to a fault.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = os.path.dirname(os.path.abspath(__file__))

SOURCES = []
for n in (24, 40, 50):
    SOURCES.append((n, os.path.join(EXP, "optionB_runs_B2scale1", f"dual_pulse_N{n}", "dual_pulse_messages.csv")))
for n in (75, 100):
    for s in (0, 1):
        SOURCES.append((n, os.path.join(EXP, "largeN_runs", f"B2_N{n}_s{s}", "dual_pulse_messages.csv")))

rows = []
for n, p in SOURCES:
    if not os.path.exists(p):
        continue
    df = pd.read_csv(p)
    total = int(df["pulse_payloads_broadcast"].sum())
    n_agents = int(df["node_id"].nunique())
    rows.append({"N": n, "total_payloads": total, "n_agents": n_agents,
                 "per_agent": total / n_agents if n_agents else float("nan")})

res = pd.DataFrame(rows)
# average over seeds at the same N (smooths spurious-event noise at large N)
res = res.groupby("N", as_index=False).agg(
    total_payloads=("total_payloads", "mean"),
    per_agent=("per_agent", "mean"),
).sort_values("N").reset_index(drop=True)
print(res.to_string(index=False))

N = res["N"].to_numpy(float)
y = res["total_payloads"].to_numpy(float)
p, b = np.polyfit(np.log(N), np.log(y), 1)
print(f"\ntotal pulse payloads por falha ~ N^{p:.2f}   (esperado ~1.0 = O(N))")
print(f"payloads POR AGENTE (medio): {res['per_agent'].mean():.2f}   (constante = O(1) por agente)")

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(N, y, s=55, zorder=3, color="tab:blue", label="B2 (medido)")
xs = np.linspace(N.min(), N.max(), 50)
ax.plot(xs, np.exp(b) * xs ** p, "--", color="tab:blue", label=f"ajuste ~ N^{p:.2f}")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("N (tamanho do enxame)")
ax.set_ylabel("payloads de pulso por falha")
ax.set_title("Complexidade de mensagens do overlay: O(N) por falha\n"
             "(payloads por agente ~ constante)")
ax.grid(True, which="both", ls=":", alpha=0.5)
ax.legend()
fig.tight_layout()
out = os.path.join(EXP, "message_complexity.png")
fig.savefig(out, dpi=130)
print(f"\nsaved {out}")
