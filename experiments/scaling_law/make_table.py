#!/usr/bin/env python
"""Render the tau / R^2 summary table (per algorithm, per N) as a slide-ready PNG,
consistent with the figure package. Reads figure_data.csv."""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(EXP_DIR, "figure_data.csv")
OUT = os.path.join(EXP_DIR, "figures", "fig10_tabela.png")

C_BASE = "firebrick"
C_B2 = "royalblue"


def fit(N, y):
    return float(np.polyfit(np.log(N), np.log(y), 1)[0])


def main():
    df = pd.read_csv(DATA_CSV)
    b = df[df.method == "baseline"].set_index("N").sort_index()
    o = df[df.method == "B2"].set_index("N").sort_index()
    Ns = sorted(b.index)

    cols = ["N", "τ baseline [s]", "R²", "τ overlay [s]", "R²", "aceleração"]
    rows = []
    for n in Ns:
        sp = b.loc[n, "tau_fit"] / o.loc[n, "tau_fit"]
        rows.append([
            f"{n}",
            f"{b.loc[n, 'tau_fit']:.1f}",
            f"{b.loc[n, 'tau_fit_r2']:.3f}",
            f"{o.loc[n, 'tau_fit']:.2f}",
            f"{o.loc[n, 'tau_fit_r2']:.3f}",
            f"{sp:.1f}×",
        ])
    p_b = fit(np.array(Ns, float), b["tau_fit"].to_numpy(float))
    p_o = fit(np.array(Ns, float), o["tau_fit"].to_numpy(float))

    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    ax.axis("off")
    tab = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(12)
    tab.scale(1, 1.7)

    ncol = len(cols)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("0.8")
        if r == 0:  # header
            cell.set_facecolor("0.93")
            cell.set_text_props(fontweight="bold")
            if c in (1, 2):
                cell.set_text_props(color=C_BASE, fontweight="bold")
            elif c in (3, 4):
                cell.set_text_props(color=C_B2, fontweight="bold")
        else:
            if c in (1, 2):
                cell.set_facecolor("#fdeeee")
            elif c in (3, 4):
                cell.set_facecolor("#eef2fb")
            elif c == 5:
                cell.set_facecolor("#eef7ee")
                cell.set_text_props(fontweight="bold", color="seagreen")

    ax.set_title("Tempo de reacomodação τ e qualidade do ajuste (R²) por algoritmo e N\n"
                 "falha de 1 nó · ganho estável K=250/N",
                 fontsize=12, fontweight="bold", pad=14)
    fig.text(0.5, 0.045,
             f"ajuste log-log:  baseline  τ ~ N^{p_b:.2f}  (≈ Θ(N²))      "
             f"overlay 2-DOF  τ ~ N^{p_o:.2f}  (plano)",
             ha="center", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT, dpi=150)
    print(f"Wrote {os.path.relpath(OUT, EXP_DIR)}")
    print(f"baseline p={p_b:.3f}  overlay p={p_o:.3f}")


if __name__ == "__main__":
    main()
