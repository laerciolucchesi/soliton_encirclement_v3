#!/usr/bin/env python
"""Advisor-facing figure package for the scaling-law result.

Reads figure_data.csv (written by gen_figure_data.py) + the per-run
target_telemetry.csv files, and renders a consistent set of figures into
experiments/scaling_law/figures/:

  fig1_lei_de_escala.png    log-log tau vs N: baseline ~N^2 vs B2 flat (THE figure)
  fig2_speedup.png          speedup = tau_base / tau_B2 vs N (grows ~N^2)
  fig3_recuperacao_baseline.png  E_gap(t) all N, baseline (recovery slows with N)
  fig4_recuperacao_B2.png        E_gap(t) all N, B2 (all collapse to ~2 s)
  fig5_comparacao_N100.png       baseline vs B2 at N=100 (the "money shot")
  fig6_painel_resumo.png         2x2 summary panel (single shareable image)

All labels in pt-BR. Consistent palette: baseline = firebrick, B2 = royalblue;
per-N curves use a viridis gradient.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(EXP_DIR, "figure_data.csv")
OUT_DIR = os.path.join(EXP_DIR, "figures")
T0 = 5.0

C_BASE = "firebrick"
C_B2 = "royalblue"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9.5,
    "figure.dpi": 150,
})


def load():
    df = pd.read_csv(DATA_CSV)
    return df


def fit_loglog(N, y):
    N = np.asarray(N, float)
    y = np.asarray(y, float)
    m = np.isfinite(N) & np.isfinite(y) & (N > 0) & (y > 0)
    if m.sum() < 2:
        return None, None
    p, b = np.polyfit(np.log(N[m]), np.log(y[m]), 1)
    return float(p), float(b)


def load_trace(path):
    d = pd.read_csv(path)
    t = d["timestamp"].to_numpy(float)
    e = d["E_gap"].to_numpy(float)
    keep = t >= T0
    return t[keep] - T0, e[keep]


def n_colors(ns):
    return {n: cm.viridis(i / max(1, len(ns) - 1)) for i, n in enumerate(sorted(ns))}


# ----------------------------------------------------------------------------
def fig1_scaling(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    for method, color, label in (("baseline", C_BASE, "baseline (controlador local)"),
                                 ("B2", C_B2, "overlay 2-DOF (dual-pulse + feedforward)")):
        g = df[df.method == method].sort_values("N")
        N, y = g["N"].to_numpy(float), g["tau_fit"].to_numpy(float)
        ax.scatter(N, y, s=70, color=color, zorder=4, label=label)
        p, b = fit_loglog(N, y)
        xs = np.linspace(N.min(), N.max(), 80)
        if p is not None:
            ax.plot(xs, np.exp(b) * xs ** p, "--", color=color, alpha=0.9, zorder=3,
                    label=f"   ajuste: $\\tau \\sim N^{{{p:.2f}}}$")
    # reference guides anchored at the smallest-N baseline point
    g0 = df[df.method == "baseline"].sort_values("N").iloc[0]
    n0, y0 = float(g0["N"]), float(g0["tau_fit"])
    xs = np.linspace(df["N"].min(), df["N"].max(), 80)
    ax.plot(xs, y0 * (xs / n0) ** 2.0, ":", color="black", alpha=0.45, label="guia $\\sim N^2$")
    ax.plot(xs, y0 * (xs / n0) ** 1.0, ":", color="gray", alpha=0.55, label="guia $\\sim N^1$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([24, 40, 50, 75, 100])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("tamanho do enxame  $N$  (escala log)")
    ax.set_ylabel("tempo de reacomodação  $\\tau$  [s]  (escala log)")
    ax.set_title("Lei de escala: tempo de reacomodação após falha de 1 nó\n"
                 "baseline difusivo $\\Theta(N^2)$  vs  overlay com $\\tau$ ~plano")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.95)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig1_lei_de_escala.png")
    fig.savefig(out); plt.close(fig)
    return out


def fig2_speedup(df):
    base = df[df.method == "baseline"].set_index("N")["tau_fit"]
    b2 = df[df.method == "B2"].set_index("N")["tau_fit"]
    Ns = sorted(set(base.index) & set(b2.index))
    sp = [base[n] / b2[n] for n in Ns]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(Ns, sp, "o-", color="seagreen", lw=2, ms=9, zorder=4, label="speedup medido")
    for n, s in zip(Ns, sp):
        ax.annotate(f"{s:.0f}×", (n, s), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=10, color="seagreen", fontweight="bold")
    # N^2 reference anchored at the first point
    n0, s0 = Ns[0], sp[0]
    xs = np.linspace(min(Ns), max(Ns), 80)
    ax.plot(xs, s0 * (xs / n0) ** 2.0, ":", color="black", alpha=0.5, label="referência $\\sim N^2$")
    ax.set_yscale("log")
    ax.set_xlabel("tamanho do enxame  $N$")
    ax.set_ylabel("aceleração  $\\tau_{baseline}\\,/\\,\\tau_{overlay}$  (escala log)")
    ax.set_title("Vantagem do overlay cresce ~$N^2$\n"
                 "(baseline $\\propto N^2$, overlay ~constante)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig2_speedup.png")
    fig.savefig(out); plt.close(fig)
    return out


def _recovery(ax, df, method, logx, xmax, title):
    g = df[df.method == method].sort_values("N")
    cols = n_colors(g["N"].astype(int).tolist())
    for _, r in g.iterrows():
        n = int(r["N"])
        t, e = load_trace(r["telemetry_path"])
        if e.size:
            i_pk = int(np.nanargmax(e))          # start at the peak: show the DECAY,
            t, e = t[i_pk:], e[i_pk:]            # drop the crash rising-edge transient
        e = np.where(e > 1e-6, e, np.nan)
        ax.plot(t, e, color=cols[n], lw=1.8,
                label=f"N={n}  ($\\tau$={r['tau_fit']:.1f} s)")
    if logx:
        ax.set_xscale("log")
    ax.set_yscale("log")
    if xmax:
        ax.set_xlim(left=(0.1 if logx else 0), right=xmax)
    ax.set_xlabel("tempo desde a falha  [s]" + ("  (escala log)" if logx else ""))
    ax.set_ylabel("erro global de espaçamento  $E_{gap}$  (log)")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.legend(fontsize=8.5, ncol=1, loc="lower left")


def fig3_baseline_recovery(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    _recovery(ax, df, "baseline", logx=True, xmax=None,
              title="Recuperação do baseline: quanto maior $N$, mais lenta\n"
                    "(decaimento de $E_{gap}$ após a falha)")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig3_recuperacao_baseline.png")
    fig.savefig(out); plt.close(fig)
    return out


def fig4_b2_recovery(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    _recovery(ax, df, "B2", logx=False, xmax=15.0,
              title="Recuperação do overlay: ~mesma duração para todo $N$\n"
                    "(as curvas colapsam — $\\tau$ independente de $N$)")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig4_recuperacao_B2.png")
    fig.savefig(out); plt.close(fig)
    return out


def fig5_moneyshot(df, N=100):
    row_b = df[(df.method == "baseline") & (df.N == N)]
    row_2 = df[(df.method == "B2") & (df.N == N)]
    if row_b.empty or row_2.empty:
        return None
    tb, eb = load_trace(row_b.iloc[0]["telemetry_path"])
    t2, e2 = load_trace(row_2.iloc[0]["telemetry_path"])
    eb = np.where(eb > 1e-6, eb, np.nan); e2 = np.where(e2 > 1e-6, e2, np.nan)
    tau_b = float(row_b.iloc[0]["tau_fit"]); tau_2 = float(row_2.iloc[0]["tau_fit"])
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax in (axL, axR):
        ax.plot(tb, eb, color=C_BASE, lw=2, label=f"baseline ($\\tau$={tau_b:.0f} s)")
        ax.plot(t2, e2, color=C_B2, lw=2, label=f"overlay 2-DOF ($\\tau$={tau_2:.1f} s)")
        ax.set_yscale("log")
        ax.set_xlabel("tempo desde a falha  [s]")
        ax.set_ylabel("$E_{gap}$  (log)")
        ax.grid(True, which="both", ls=":", alpha=0.45)
        ax.legend(loc="upper right")
    axL.set_title(f"N={N}: visão completa")
    axL.set_xlim(0, max(np.nanmax(tb), 1))
    axR.set_title(f"N={N}: zoom nos primeiros 15 s")
    axR.set_xlim(0, 15)
    fig.suptitle(f"Mesma falha, mesmo controlador: ~{tau_b/tau_2:.0f}× mais rápido (N={N})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(OUT_DIR, "fig5_comparacao_N100.png")
    fig.savefig(out); plt.close(fig)
    return out


def fig6_panel(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    # (0,0) scaling
    ax = axes[0, 0]
    for method, color, lab in (("baseline", C_BASE, "baseline"),
                               ("B2", C_B2, "overlay 2-DOF")):
        g = df[df.method == method].sort_values("N")
        N, y = g["N"].to_numpy(float), g["tau_fit"].to_numpy(float)
        ax.scatter(N, y, s=55, color=color, zorder=4, label=lab)
        p, b = fit_loglog(N, y)
        xs = np.linspace(N.min(), N.max(), 80)
        if p is not None:
            ax.plot(xs, np.exp(b) * xs ** p, "--", color=color, alpha=0.9,
                    label=f"$N^{{{p:.2f}}}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([24, 40, 50, 75, 100]); ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("$N$"); ax.set_ylabel("$\\tau$ [s]")
    ax.set_title("(a) Lei de escala")
    ax.grid(True, which="both", ls=":", alpha=0.5); ax.legend(fontsize=8.5)
    # (0,1) speedup
    ax = axes[0, 1]
    base = df[df.method == "baseline"].set_index("N")["tau_fit"]
    b2 = df[df.method == "B2"].set_index("N")["tau_fit"]
    Ns = sorted(set(base.index) & set(b2.index)); sp = [base[n] / b2[n] for n in Ns]
    ax.plot(Ns, sp, "o-", color="seagreen", lw=2, ms=8)
    for n, s in zip(Ns, sp):
        ax.annotate(f"{s:.0f}×", (n, s), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9, color="seagreen", fontweight="bold")
    ax.set_yscale("log"); ax.set_xlabel("$N$")
    ax.set_ylabel("$\\tau_{base}/\\tau_{overlay}$")
    ax.set_title("(b) Aceleração (~$N^2$)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    # (1,0) baseline recovery
    _recovery(axes[1, 0], df, "baseline", logx=True, xmax=None,
              title="(c) Recuperação baseline (lenta, cresce com $N$)")
    # (1,1) B2 recovery
    _recovery(axes[1, 1], df, "B2", logx=False, xmax=15.0,
              title="(d) Recuperação overlay (colapsa, ~plana)")
    fig.suptitle("Escape do trilema estabilidade–velocidade–escala  "
                 "(falha única, ganho estável $K=250/N$)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(OUT_DIR, "fig6_painel_resumo.png")
    fig.savefig(out); plt.close(fig)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load()
    print("figure_data.csv:")
    print(df[["method", "N", "tau_fit", "tau_fit_r2"]].to_string(index=False))
    outs = [
        fig1_scaling(df),
        fig2_speedup(df),
        fig3_baseline_recovery(df),
        fig4_b2_recovery(df),
        fig5_moneyshot(df, N=100),
        fig6_panel(df),
    ]
    print("\nWrote:")
    for o in outs:
        if o:
            print("  " + os.path.relpath(o, EXP_DIR))


if __name__ == "__main__":
    main()
