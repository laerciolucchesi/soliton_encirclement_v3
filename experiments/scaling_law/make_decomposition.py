#!/usr/bin/env python
"""Temporal decomposition: how much of the tangential control comes from the
LOCAL controller (baseline channel) vs the OVERLAY feedforward, over time -- and
when the overlay hands control back to the baseline.

Both contributions are reconstructed EXACTLY from the logged agent telemetry:
    v_baseline = K_TAU * u * r          (compute_tangential_velocity, protocol_agent:635)
    v_overlay  = (shift_remaining / T_FF) * r, capped at VM_MAX_SPEED_XY   (:1202)
and the logged column `dual_pulse_shift` IS get_shift_remaining() (:1187).

Reads figure_runs/B2_N50/agent_telemetry.csv.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(EXP_DIR, "figure_runs", "B2_N50")
OUT = os.path.join(EXP_DIR, "figures", "fig11_decomposicao_temporal.png")

# constants (config_param.py) and the B2-run settings
K_TAU = 0.2
R = 20.0
T_FF = 1.0
VCAP = 10.0
SLEEP = 0.01
T0 = 5.0

C_BASE = "firebrick"
C_OVER = "royalblue"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "legend.fontsize": 9.5, "figure.dpi": 150})


def main():
    df = pd.read_csv(os.path.join(RUN, "agent_telemetry.csv"))
    df = df[df.timestamp >= T0 - 0.3].copy()
    df["t"] = df.timestamp - T0
    df["v_base"] = K_TAU * df["u"].astype(float) * R
    df["v_over"] = (df["dual_pulse_shift"].astype(float) / T_FF * R).clip(-VCAP, VCAP)
    df["a_base"] = df["v_base"].abs()
    df["a_over"] = df["v_over"].abs()

    # swarm-mean tangential effort per time bin
    post = df[df.t >= 0].copy()
    post["tb"] = (post["t"] / 0.1).round() * 0.1
    agg = post.groupby("tb").agg(over=("a_over", "mean"),
                                 base=("a_base", "mean")).reset_index()
    agg["total"] = agg.over + agg.base
    # handover = first crossover AFTER the overlay's peak (ignore the t~0 transient,
    # before the pulses complete and the feedforward has even started).
    t_peak = float(agg.loc[agg.over.idxmax(), "tb"])
    aft = agg[agg.tb > t_peak]
    cross = aft[aft.base > aft.over]
    t_hand = float(cross["tb"].iloc[0]) if len(cross) else np.nan
    # shift essentially consumed: swarm-max |shift_remaining| drops below SLEEP.
    # NOTE: in B2 the overlay has NO hard sleep gate (that is Option A's
    # apply_shift_to_gaps); the feedforward decays continuously. This is just a
    # "shift drained" marker, not a switch.
    srt = post.groupby("tb")["dual_pulse_shift"].agg(lambda s: s.abs().max())
    asleep = srt.index[srt.to_numpy() >= SLEEP]
    t_sleep = float(asleep.max()) if len(asleep) else np.nan
    # share of total effort (guard the t~0 floor where both are ~0)
    floor = 0.008
    sh = agg[agg.total > floor].copy()
    sh["share_over"] = 100.0 * sh.over / sh.total
    sh["share_base"] = 100.0 * sh.base / sh.total

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # --- LEFT: swarm-mean effort magnitudes ---
    axL.plot(agg.tb, agg.over, color=C_OVER, lw=2.2, label="overlay (feedforward)")
    axL.plot(agg.tb, agg.base, color=C_BASE, lw=2.2, label="baseline (controlador local)")
    if np.isfinite(t_sleep):
        axL.axvline(t_sleep, color="k", ls="--", lw=1.2, alpha=0.7)
        axL.annotate(f"shift praticamente consumido\n(|shift|<{SLEEP} rad) ≈ {t_sleep:.1f} s",
                     (t_sleep, axL.get_ylim()[1]), xytext=(8, -30),
                     textcoords="offset points", fontsize=9.5, fontweight="bold")
    axL.set_xlim(0, 8)
    axL.set_xlabel("tempo desde a falha  [s]")
    axL.set_ylabel("esforço tangencial médio do enxame  |v|  [m/s]")
    axL.set_title("(a) Esforço de controle (média do enxame)")
    axL.grid(True, ls=":", alpha=0.5)
    axL.legend(loc="center right")

    # --- RIGHT: share of total effort ---
    axR.stackplot(sh.tb, sh.share_over, sh.share_base,
                  colors=[C_OVER, C_BASE], alpha=0.55,
                  labels=["overlay", "baseline"])
    axR.plot(sh.tb, sh.share_over, color=C_OVER, lw=2)
    axR.axhline(50, color="0.4", ls=":", lw=1)
    if np.isfinite(t_hand):
        axR.axvline(t_hand, color="k", ls="--", lw=1.2, alpha=0.8)
        axR.annotate(f"handover ≈ {t_hand:.1f} s\n(baseline assume)",
                     (t_hand, 50), xytext=(8, 6), textcoords="offset points",
                     fontsize=9.5, fontweight="bold")
    axR.set_xlim(0, 4.5); axR.set_ylim(0, 100)
    axR.set_xlabel("tempo desde a falha  [s]")
    axR.set_ylabel("participação no esforço de controle  [%]")
    axR.set_title("(b) Quem está no comando (% do esforço)")
    axR.legend(loc="upper right")

    fig.suptitle("Quem controla ao longo do tempo: o overlay redistribui rápido, "
                 "depois entrega ao baseline para a manutenção",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT, dpi=150)
    print(f"overlay peak at t={t_peak:.2f}s   handover(share<50%)={t_hand:.2f}s   "
          f"shift consumed (|shift|<{SLEEP})={t_sleep:.2f}s")
    print(f"Wrote {os.path.relpath(OUT, EXP_DIR)}")


if __name__ == "__main__":
    main()
