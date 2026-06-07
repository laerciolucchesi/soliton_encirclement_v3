#!/usr/bin/env python
"""Extra advisor figures that show the DISTRIBUTED ALGORITHM (Cap. 3), not just
the performance:

  fig7_kymograph.png    space-time map of the counter-propagating pulses filling
                        the ring with each node's redistribution shift (N=50).
  fig8_anel_snapshots.png  polar before/during/after snapshots of the swarm
                        closing the gap left by a failed node (small N for clarity).
  fig9_mensagens.png    message complexity: ~4 payloads/agent (O(1)) and total
                        O(N) -- the speed costs no communication blow-up.

Reads figure_runs/B2_N*/{agent_telemetry.csv, events.csv, dual_pulse_messages.csv}.
A small-N B2 run (N=10) is generated on demand for the ring snapshots.
"""
import os
import sys
import subprocess

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))
MAIN_PY = os.path.join(REPO_ROOT, "main.py")
FIG_RUNS = os.path.join(EXP_DIR, "figure_runs")
OUT_DIR = os.path.join(EXP_DIR, "figures")
T0 = 5.0
GAIN_PRODUCT = 250.0

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "legend.fontsize": 9.5, "figure.dpi": 150})


def run_b2(n, budget=20.0):
    """Generate a clean B2 run for size n (used for the small-N ring demo)."""
    run_dir = os.path.join(FIG_RUNS, f"B2_N{n}")
    tgt = os.path.join(run_dir, "agent_telemetry.csv")
    if os.path.exists(tgt):
        return run_dir
    os.makedirs(run_dir, exist_ok=True)
    k = GAIN_PRODUCT / n
    env = dict(os.environ)
    env.update({
        "PYTHONIOENCODING": "utf-8", "PROPAGATION_METHOD": "dual_pulse",
        "PROPAGATION_K_PROP": "0.0", "NUM_AGENTS": str(n),
        "SIM_DURATION": str(T0 + budget), "K_E_TAU": f"{k:.6f}",
        "VIS_OPEN_BROWSER": "False", "SKIP_TELEMETRY_PLOTS": "True",
        "INIT_ANGLES_EQUIDISTANT": "True", "INIT_RADIUS_RANGE": "0.0",
        "TARGET_MOTION_SPEED_XY": "0.0", "DETERMINISTIC_FAILURE_ENABLE": "True",
        "DETERMINISTIC_FAILURE_AGENT_ID": str(2 + n // 2),
        "DETERMINISTIC_FAILURE_TIME_T0": str(T0),
        "DETERMINISTIC_FAILURE_OFF_TIME": "-1.0",
        "DUAL_PULSE_INTEGRATION": "B2", "DUAL_PULSE_DELTA_SCALE": "1.0",
        "DUAL_PULSE_T_FF": "1.0", "DUAL_PULSE_TTL_HOPS": str(3 * n),
        "RUNS_SUMMARY_CSV_PATH": os.path.join(run_dir, "runs_summary.csv"),
    })
    print(f"  [gen] small-N B2 run N={n} ...", flush=True)
    subprocess.run([sys.executable, MAIN_PY], cwd=run_dir, env=env,
                   capture_output=True, text=True)
    return run_dir


def failed_node(run_dir):
    ev = pd.read_csv(os.path.join(run_dir, "events.csv"))
    f = ev[ev.event_type == "failure_start"]
    return int(f.iloc[0]["node_id"]) if len(f) else None


# ---------------------------------------------------------------------------
def fig7_kymograph(n=50):
    run_dir = os.path.join(FIG_RUNS, f"B2_N{n}")
    at = pd.read_csv(os.path.join(run_dir, "agent_telemetry.csv"))
    fail = failed_node(run_dir)
    # signed ring position relative to the failed node, in [-n/2, n/2)
    def signed_pos(nid):
        d = (nid - fail)
        return ((d + n // 2) % n) - n // 2
    win = at[(at.timestamp >= T0 - 0.1) & (at.timestamp <= T0 + 1.2)].copy()
    win["pos"] = win["node_id"].map(signed_pos)
    grid = win.pivot_table(index="pos", columns="timestamp",
                           values="dual_pulse_target", aggfunc="last").sort_index()
    times = grid.columns.to_numpy(float)
    poss = grid.index.to_numpy(float)
    Z = grid.to_numpy(float)
    vmax = np.nanmax(np.abs(Z)) or 1.0
    fig, ax = plt.subplots(figsize=(9, 6.2))
    im = ax.pcolormesh(times - T0, poss, Z, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       shading="nearest")
    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.text(0.02, 0.5, "nó que falhou", rotation=90, va="center", fontsize=9,
            color="k", alpha=0.7)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("deslocamento angular a aplicar  $\\delta_D$  [rad]")
    ax.set_xlabel("tempo desde a falha  [s]")
    ax.set_ylabel("posição do nó no anel  (hops a partir do nó que falhou)")
    ax.set_title(f"Kymograph: pulsos contra-propagantes preenchem o anel (N={n})\n"
                 "cada nó descobre seu $\\delta_D$ em ~0,3 s ≈ (diâmetro × tick)")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig7_kymograph.png")
    fig.savefig(out); plt.close(fig)
    return out


def fig8_ring(n=10):
    run_dir = run_b2(n)
    at = pd.read_csv(os.path.join(run_dir, "agent_telemetry.csv"))
    fail = failed_node(run_dir)
    snaps = [(T0 - 0.2, "antes da falha"),
             (T0 + 0.3, "logo após (lacuna aberta)"),
             (T0 + 1.5, "redistribuindo"),
             (at.timestamp.max() - 0.5, "reacomodado")]
    fig, axes = plt.subplots(1, len(snaps), figsize=(4.2 * len(snaps), 4.6),
                             subplot_kw={"projection": "polar"})
    for ax, (tt, lab) in zip(axes, snaps):
        idx = (at.timestamp - tt).abs()
        t_near = at.loc[idx.idxmin(), "timestamp"]
        rows = at[np.isclose(at.timestamp, t_near)]
        ang = rows["theta_rel"].to_numpy(float)
        ids = rows["node_id"].to_numpy(int)
        dead = t_near >= T0 - 1e-6  # before the failure instant, every node is alive
        alive = (ids != fail) if dead else np.ones(ids.size, bool)
        # polygon connecting consecutive (by angle) live nodes: a long edge marks
        # the open gap; it shrinks to a regular polygon once redistributed.
        a = np.sort(ang[alive] % (2 * np.pi))
        a_closed = np.append(a, a[0] + 2 * np.pi)
        ax.plot(a_closed, np.ones(a_closed.size), color="royalblue", lw=1.2,
                alpha=0.35, zorder=2)
        ax.scatter(ang[alive], np.ones(alive.sum()), s=130, color="royalblue",
                   zorder=3, edgecolor="k", linewidth=0.5)
        if (~alive).any():
            ax.scatter(ang[~alive], np.ones((~alive).sum()), s=170, color="red",
                       marker="x", zorder=4, linewidth=2.5, label="nó que falhou")
        ax.set_title(f"t = {t_near - T0:+.1f} s\n{lab}", fontsize=10)
        ax.set_rticks([]); ax.set_ylim(0, 1.25)
        ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
        ax.set_xticklabels([]); ax.grid(alpha=0.3)
    fig.suptitle(f"Reacomodação do anel ao redor do alvo após falha de 1 nó (N={n})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.join(OUT_DIR, "fig8_anel_snapshots.png")
    fig.savefig(out); plt.close(fig)
    return out


def fig9_messages(ns=(24, 40, 50, 75, 100)):
    rows = []
    for n in ns:
        p = os.path.join(FIG_RUNS, f"B2_N{n}", "dual_pulse_messages.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        tot = int(d["pulse_payloads_broadcast"].sum())
        rows.append({"N": n, "total": tot, "per_agent": tot / n,
                     "mean_nonzero": d.loc[d.pulse_payloads_broadcast > 0,
                                           "pulse_payloads_broadcast"].mean()})
    df = pd.DataFrame(rows)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))
    # total vs N (O(N))
    axL.plot(df.N, df.total, "o-", color="darkorange", lw=2, ms=9, label="total medido")
    n0, t0 = df.N.iloc[0], df.total.iloc[0]
    axL.plot(df.N, t0 * df.N / n0, ":", color="black", alpha=0.6, label="referência $\\propto N$")
    for _, r in df.iterrows():
        axL.annotate(f"{int(r.total)}", (r.N, r.total), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=9)
    axL.set_xlabel("tamanho do enxame  $N$")
    axL.set_ylabel("total de payloads de pulso por evento")
    axL.set_title("(a) Custo total de comunicação  $O(N)$")
    axL.grid(True, ls=":", alpha=0.5); axL.legend(loc="upper left")
    # per-agent (O(1))
    axR.bar(df.N.astype(str), df.per_agent, color="mediumseagreen", width=0.55)
    axR.axhline(4, color="k", ls="--", alpha=0.6, label="≈ 4 / agente")
    for i, v in enumerate(df.per_agent):
        axR.text(i, v + 0.08, f"{v:.1f}", ha="center", fontsize=9)
    axR.set_ylim(0, 6)
    axR.set_xlabel("tamanho do enxame  $N$")
    axR.set_ylabel("payloads por agente")
    axR.set_title("(b) Custo por agente  $O(1)$  (= 2 direções × 2 repetições)")
    axR.grid(True, axis="y", ls=":", alpha=0.5); axR.legend(loc="upper right")
    fig.suptitle("Complexidade de mensagens: rápido E barato (ótimo pelo diâmetro)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(OUT_DIR, "fig9_mensagens.png")
    fig.savefig(out); plt.close(fig)
    print(df.to_string(index=False))
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    outs = [fig7_kymograph(50), fig8_ring(10), fig9_messages()]
    print("\nWrote:")
    for o in outs:
        print("  " + os.path.relpath(o, EXP_DIR))


if __name__ == "__main__":
    main()
