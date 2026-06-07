#!/usr/bin/env python
"""Render a slide-ready 'equation card' for the E_gap metric (matplotlib mathtext,
no LaTeX install needed). Matches the figure-package style."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(EXP_DIR, "figures", "fig13_equacao_Egap.png")

plt.rcParams.update({"font.size": 13, "mathtext.fontset": "cm"})

fig = plt.figure(figsize=(10.5, 6.2))
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")

# title
ax.text(0.5, 0.93, "Métrica principal — erro de espaçamento angular",
        ha="center", va="center", fontsize=16, fontweight="bold")

# main equation (in a soft rounded box)
EQ = (r"$E_{gap} \;=\; \sqrt{\,\frac{1}{M}\sum_{i=1}^{M}"
      r"\left(\frac{\Delta\theta_i}{\Delta\theta_{ideal}}-1\right)^{2}}"
      r"\qquad \Delta\theta_{ideal}=\frac{2\pi}{M}$")
ax.text(0.5, 0.66, EQ, ha="center", va="center", fontsize=26,
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#eef2fb",
                  edgecolor="royalblue", linewidth=1.5))

# definitions
DEFS = "\n".join([
    r"$M$  = nº de agentes vivos no instante",
    r"$\theta_i$  = posição angular do agente $i$ ao redor do alvo, ordenada em $[0,2\pi)$",
    r"$\Delta\theta_i$  = folga angular até o vizinho seguinte (cíclica),  $\sum_i \Delta\theta_i = 2\pi$",
    r"$\Delta\theta_{ideal}=2\pi/M$  = folga se a distribuição fosse perfeitamente uniforme",
])
ax.text(0.5, 0.33, DEFS, ha="center", va="center", fontsize=13.5, linespacing=1.7)

# interpretation
ax.text(0.5, 0.075,
        r"Desvio RMS das folgas em relação à uniforme — adimensional;  "
        r"$E_{gap}=0 \;\Leftrightarrow\;$ equidistância perfeita.",
        ha="center", va="center", fontsize=13, style="italic", color="0.25")

fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Wrote {os.path.relpath(OUT, EXP_DIR)}")
