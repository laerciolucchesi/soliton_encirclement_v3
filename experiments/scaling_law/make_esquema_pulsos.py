#!/usr/bin/env python
"""Esquema didático (não é simulação) dos dois pulsos contra-propagantes que
correm o anel quando um drone cai. Substitui o kymograph (fig7) no relatório
dos orientadores por uma imagem mais intuitiva. Saída: figures/esquema_pulsos.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures",
                   "esquema_pulsos.png")
C_B2 = "royalblue"
C_BASE = "firebrick"

N = 9
ang = np.linspace(90, 90 - 360, N, endpoint=False)  # graus, sentido horário
ang_rad = np.deg2rad(ang)
R = 1.0
xs, ys = R * np.cos(ang_rad), R * np.sin(ang_rad)
fail = 0  # nó do topo "cai"

fig, ax = plt.subplots(figsize=(7.6, 7.2))
ax.set_aspect("equal"); ax.axis("off")
ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.55, 1.7)

# anel de referência
th = np.linspace(0, 2 * np.pi, 400)
ax.plot(R * np.cos(th), R * np.sin(th), color="0.8", lw=1.2, zorder=1)

# nós
for i in range(N):
    if i == fail:
        ax.scatter(xs[i], ys[i], s=420, marker="X", color=C_BASE, zorder=4,
                   linewidth=2.5)
        ax.annotate("drone que caiu", (xs[i], ys[i]), textcoords="offset points",
                    xytext=(0, 20), ha="center", fontsize=11, fontweight="bold",
                    color=C_BASE)
    else:
        ax.scatter(xs[i], ys[i], s=300, color=C_B2, edgecolor="k",
                   linewidth=0.6, zorder=4)

# dois pulsos contra-propagantes saindo da vizinhança da falha
def arc_arrow(i0, i1, rad, color, label, lab_xy):
    a0 = ang_rad[i0]; a1 = ang_rad[i1]
    p0 = (1.22 * np.cos(a0), 1.22 * np.sin(a0))
    p1 = (1.22 * np.cos(a1), 1.22 * np.sin(a1))
    arr = FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}",
                          arrowstyle="-|>", mutation_scale=22, lw=3,
                          color=color, zorder=5)
    ax.add_patch(arr)
    ax.text(*lab_xy, label, fontsize=11, fontweight="bold", color=color,
            ha="center")

# horário: do vizinho 1 ao 4 ; anti-horário: do vizinho N-1 ao N-4
arc_arrow(1, 4, -0.32, "#1f9d55", "pulso  →\n(sentido horário)", (1.18, 0.55))
arc_arrow(N - 1, N - 4, 0.32, "#b8651b", "←  pulso\n(anti‑horário)", (-1.18, 0.55))

# nota central
ax.text(0, -0.05,
        "cada drone que recebe\nos DOIS pulsos já sabe\nquanto precisa se deslocar",
        ha="center", va="center", fontsize=12, color="0.15",
        bbox=dict(boxstyle="round,pad=0.5", fc="#eef2fb", ec="0.7"))

# setinhas de deslocamento em alguns nós (ilustrativo)
for i in (3, 6):
    a = ang_rad[i]
    tx, ty = -np.sin(a), np.cos(a)  # tangente
    s = 0.16 * (1 if i == 3 else -1)
    ax.annotate("", (xs[i] + s * tx, ys[i] + s * ty), (xs[i], ys[i]),
                arrowprops=dict(arrowstyle="-|>", color="0.35", lw=2), zorder=6)

ax.set_title("Os dois pulsos contra‑propagantes preenchem o anel\n"
             "(esquema da ideia — não é uma simulação)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("Wrote", os.path.relpath(OUT, os.path.dirname(os.path.abspath(__file__))))
