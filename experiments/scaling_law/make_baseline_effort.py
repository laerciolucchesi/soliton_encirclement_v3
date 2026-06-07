#!/usr/bin/env python
"""fig12 -- control effort over time with ONLY the baseline acting (no overlay).

Complements fig11: with the overlay, the baseline is nearly silent and a sharp
~2 s feedforward burst does the redistribution. Here, with the baseline ALONE,
the local controller must drive the whole redistribution itself -> a LOW but
LONG, diffusive effort spread over the full O(N^2) relaxation. The shape (low &
drawn-out, lasting longer with N) is the control-effort face of the O(N^2) law.

Effort metric = swarm-mean |v_tangential| = K_TAU * |u| * r  (m/s).

The baseline agent_telemetry.csv files are huge (per-tick, up to ~3.4 GB), so
they are streamed in CHUNKS and accumulated into time bins (bounded memory).
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(EXP_DIR, "figures", "fig12_esforco_baseline.png")
K_TAU = 0.2
R = 20.0
T0 = 5.0
T_FF = 1.0     # feedforward time constant (overlay)
VCAP = 10.0    # VM_MAX_SPEED_XY cap on the feedforward
BIN = 0.25  # s
CHUNK = 2_000_000

BASE = {
    24: os.path.join(EXP_DIR, "gain_runs", "baseline_N24", "agent_telemetry.csv"),
    50: os.path.join(EXP_DIR, "gain_runs", "baseline_N50", "agent_telemetry.csv"),
    75: os.path.join(EXP_DIR, "baseline_long_runs", "baseline_N75", "agent_telemetry.csv"),
    100: os.path.join(EXP_DIR, "baseline_long_runs", "baseline_N100", "agent_telemetry.csv"),
}
B2 = {n: os.path.join(EXP_DIR, "figure_runs", f"B2_N{n}", "agent_telemetry.csv")
      for n in (50, 75, 100)}

C_BASE = "firebrick"
C_OVER = "royalblue"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "legend.fontsize": 9.5, "figure.dpi": 150})


CACHE = os.path.join(EXP_DIR, "figures", "_cache")


def effort_profile(path, include_ff):
    """Stream the (possibly huge) telemetry, return (t_bin, mean|v_tangential|).

    The TOTAL tangential effort of the run = K_TAU*u*r (local controller) PLUS, when
    include_ff, the overlay feedforward (shift_remaining/T_FF)*r (capped). For a
    baseline-only run dual_pulse_shift==0, so include_ff is a no-op there.
    """
    cols = ["timestamp", "u", "dual_pulse_shift"] if include_ff else ["timestamp", "u"]
    parts = []
    for chunk in pd.read_csv(path, usecols=cols, chunksize=CHUNK):
        t = chunk["timestamp"].to_numpy(float) - T0
        v = K_TAU * chunk["u"].to_numpy(float) * R
        if include_ff:
            v = v + np.clip(chunk["dual_pulse_shift"].to_numpy(float) / T_FF * R,
                            -VCAP, VCAP)
        d = pd.DataFrame({"tb": np.round(t / BIN) * BIN, "v": np.abs(v)})
        d = d[d.tb >= 0.0]
        if len(d):
            parts.append(d.groupby("tb")["v"].agg(["sum", "count"]))
    if not parts:
        return np.array([]), np.array([])
    g = pd.concat(parts).groupby(level=0).sum()
    g["mean"] = g["sum"] / g["count"]
    g = g.sort_index()
    return g.index.to_numpy(float), g["mean"].to_numpy(float)


def cached_profile(key, path, include_ff):
    """Compute-or-load: the streamed profile is tiny once binned, so cache it to
    avoid re-scanning the multi-GB telemetry on every tweak."""
    os.makedirs(CACHE, exist_ok=True)
    cf = os.path.join(CACHE, f"effort_{key}_bin{BIN}.csv")
    if os.path.exists(cf):
        d = pd.read_csv(cf)
        return d["tb"].to_numpy(float), d["mean"].to_numpy(float)
    t, v = effort_profile(path, include_ff)
    pd.DataFrame({"tb": t, "mean": v}).to_csv(cf, index=False)
    return t, v


def main():
    print("Computing baseline-only effort profiles (streaming)...")
    base_prof = {}
    for n, p in BASE.items():
        if os.path.exists(p):
            t, v = cached_profile(f"baseline_N{n}", p, include_ff=False)
            base_prof[n] = (t, v)
            print(f"  baseline N={n}: peak |v|={np.nanmax(v):.3f} m/s, "
                  f"duracao ~{t[v > 0.05 * np.nanmax(v)].max():.0f}s")
    print("Computing B2 TOTAL effort (local controller + feedforward) for contrast...")
    b2_prof = {}
    for n, p in B2.items():
        if os.path.exists(p):
            t, v = cached_profile(f"B2tot_N{n}", p, include_ff=True)
            b2_prof[n] = (t, v)
            print(f"  B2 N={n}: peak |v|={np.nanmax(v):.3f} m/s (total da config. B2)")

    cols = {n: cm.viridis(i / max(1, len(base_prof) - 1))
            for i, n in enumerate(sorted(base_prof))}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # --- LEFT: baseline-only, all N ---
    for n in sorted(base_prof):
        t, v = base_prof[n]
        v = np.where(v > 5e-4, v, np.nan)
        axL.plot(t, v, color=cols[n], lw=2, label=f"N={n}")
    axL.set_xscale("log"); axL.set_yscale("log")
    axL.set_xlim(0.1, 1000)
    axL.set_xlabel("tempo desde a falha  [s]  (escala log)")
    axL.set_ylabel("esforço tangencial médio  |v|  [m/s]  (log)")
    axL.set_title("(a) Só o baseline: pico cada vez mais BAIXO e cauda mais LONGA com N")
    axL.grid(True, which="both", ls=":", alpha=0.5)
    axL.legend(loc="lower left", title="baseline (sem overlay)")

    # --- RIGHT: baseline vs overlay at N=50 (matches fig11's decomposition) ---
    Ncmp = 50  # same N as fig11 -> apples-to-apples with the overlay graphs
    tb, vb = base_prof[Ncmp]
    to, vo = b2_prof[Ncmp]
    dur_b = tb[vb > 0.05 * np.nanmax(vb)].max()
    vb_m = np.where(vb > 5e-4, vb, np.nan)
    vo_m = np.where(vo > 5e-4, vo, np.nan)
    axR.plot(tb, vb_m, color=C_BASE, lw=2.2, label=f"config. BASELINE (sem overlay)  ~{dur_b:.0f} s")
    axR.plot(to, vo_m, color=C_OVER, lw=2.2, label="config. B2 (com overlay)  burst ~2 s")
    axR.set_xscale("log"); axR.set_yscale("log")
    axR.set_xlim(0.1, 1000)
    axR.set_xlabel("tempo desde a falha  [s]  (escala log)")
    axR.set_ylabel("esforço tangencial TOTAL médio  |v|  [m/s]  (log)")
    axR.set_title(f"(b) Esforço total por configuração (N={Ncmp}): "
                  "burst forte e curto  vs  fraco e longo")
    axR.grid(True, which="both", ls=":", alpha=0.5)
    axR.legend(loc="upper right")

    fig.suptitle("Esforço de controle ao longo do tempo: o baseline sozinho trabalha "
                 "pouco por vez, mas por muito tempo",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT, dpi=150)
    print(f"\nWrote {os.path.relpath(OUT, EXP_DIR)}  (contraste em N={Ncmp})")


if __name__ == "__main__":
    main()
