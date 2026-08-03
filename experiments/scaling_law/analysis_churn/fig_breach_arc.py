#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figura do vao ABSOLUTO no pico: o numero comunicavel, com o p e o IC honestos.

T2 (trava): d_arco (metros) e alpha_total (normalizado) sao O MESMO EFEITO em duas
normalizacoes. Os DOIS sao marginais (p = 0,040 e 0,064). A figura mostra os dois
lado a lado; reportar so o de p menor seria SELECAO DE NORMALIZACAO.

Escopo da figura: taxas 6/12/24 (Pi_2' 0,80-3,20). A taxa 48 esta FORA -- foi
declarada nao identificavel para a analise por evento (mediana de W_e no piso).

Uso:
    python experiments/scaling_law/analysis_churn/fig_breach_arc.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
R_ENC = 20.0
USABLE = (6.0, 12.0, 24.0)


def cluster_ci(y, groups):
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    y, groups = y[ok], np.asarray(groups)[ok]
    gs = np.unique(groups)
    n, G = y.size, gs.size
    mu = y.mean()
    u = y - mu
    se = np.sqrt(sum(u[groups == g].sum() ** 2 for g in gs) / n ** 2 * (G / (G - 1.0)))
    tc = stats.t.ppf(0.975, G - 1)
    p = 2 * (1 - stats.t.cdf(abs(mu / se), G - 1)) if se > 0 else np.nan
    return mu, se, mu - tc * se, mu + tc * se, p, n


def main():
    p = os.path.join(HERE, "gmax_events_paired.csv")
    if not os.path.exists(p):
        sys.exit("gmax_events_paired.csv ausente -- rode aggregate_gmax.py antes.")
    d = pd.read_csv(p)
    d = d[d.rate_total.isin(USABLE)].copy()
    d["d_arco"] = d["arco_base"] - d["arco_b2"]
    grp = (d.rate_total.astype(str) + "_" + d.seed.astype(str)).to_numpy()

    mu, se, lo, hi, pv, n = cluster_ci(d["d_arco"], grp)
    print(f"d_arco agregado = {mu:+.5f} m [{lo:+.5f}, {hi:+.5f}] p={pv:.5f} n={n}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.2, 4.8))

    # ---- A: distribuicao das DIFERENCAS PAREADAS --------------------------
    # O objeto honesto e' a diferenca pareada, nao as duas marginais lado a lado:
    # as medianas marginais vao na direcao OPOSTA a da media pareada, e mostrar so
    # elas (ou so a media) escolhe a estatistica que agrada.
    dd = d["d_arco"].to_numpy(float)
    med, mean = float(np.median(dd)), float(np.mean(dd))
    frac_pos = float((dd > 0).mean())
    p90 = float(np.quantile(dd, 0.9))
    mean_no_top = float(dd[dd < p90].mean())
    lim = float(np.quantile(np.abs(dd), 0.98))
    axA.hist(dd, bins=np.linspace(-lim, lim, 51), color="0.7", edgecolor="0.45")
    axA.axvline(0.0, color="0.25", ls="--", lw=1.5)
    axA.axvline(med, color="seagreen", lw=2.0, label=f"mediana {med:+.3f} m")
    axA.axvline(mean, color="darkorange", lw=2.0, label=f"media {mean:+.3f} m")
    axA.axvline(mean_no_top, color="firebrick", lw=1.6, ls=":",
                label=f"media sem o decil sup. {mean_no_top:+.3f} m")
    axA.set_xlabel("d_arco = arco(baseline) − arco(B2), MESMO evento [m]")
    axA.set_ylabel("eventos")
    axA.set_title(f"A) diferenca pareada: media e mediana DISCORDAM em magnitude\n"
                  f"{frac_pos:.1%} dos eventos com baseline pior "
                  f"(moeda = 50%)", fontweight="bold")
    axA.legend(fontsize=8); axA.grid(alpha=0.3)

    # ---- B: diferenca pareada por taxa + agregado -------------------------
    labs, mus, los, his, ns = [], [], [], [], []
    for r in USABLE:
        s = d[d.rate_total == r]
        g = s.seed.astype(str).to_numpy()
        m_, _, l_, h_, _, n_ = cluster_ci(s["d_arco"], g)
        labs.append(f"{r:g}/min"); mus.append(m_); los.append(l_); his.append(h_); ns.append(n_)
    labs.append("agregado"); mus.append(mu); los.append(lo); his.append(hi); ns.append(n)
    y = np.arange(len(labs))
    axB.errorbar(mus, y, xerr=[np.array(mus) - np.array(los),
                               np.array(his) - np.array(mus)],
                 fmt="o", color="seagreen", capsize=4, lw=1.8)
    axB.axvline(0.0, ls="--", color="0.35", lw=1.3)
    axB.set_yticks(y); axB.set_yticklabels([f"{a}\n(n={b})" for a, b in zip(labs, ns)])
    axB.invert_yaxis()
    axB.set_xlabel("d_arco = arco(baseline) − arco(B2), MESMO evento [m]")
    axB.set_title("B) diferenca pareada por evento\n(media; barras = IC95 agrupado por rodada)",
                  fontweight="bold")
    axB.grid(alpha=0.3, axis="x")
    axB.annotate(f"agregado: {mu:+.3f} m  IC95 [{lo:+.3f}, {hi:+.3f}]  p = {pv:.3f}",
                 (0.5, 0.06), xycoords="axes fraction", ha="center", fontsize=8.5,
                 color="0.25")

    fig.suptitle("Vao ABSOLUTO da brecha: efeito marginal e concentrado na cauda",
                 fontsize=12, fontweight="bold")
    med_run = d.groupby(["rate_total", "seed"])["d_arco"].median()
    foot = (
        "MESMO EFEITO, DUAS NORMALIZACOES — os dois sao MARGINAIS e vao juntos:\n"
        f"   em metros (esta figura):   d_arco = {mu:+.3f} m,  p = {pv:.3f}\n"
        "   normalizado por 2*pi/M:    alpha_total = +0,0247,  p = 0,064\n"
        "A MEDIA NAO E' O EVENTO TIPICO — tres qualificacoes que precisam andar com o numero:\n"
        f"   mediana pareada = {med:+.3f} m (contra media {mean:+.3f} m);  "
        f"{frac_pos:.1%} dos eventos com baseline pior\n"
        f"   removendo o decil superior a media INVERTE DE SINAL: {mean_no_top:+.3f} m\n"
        f"   mediana por rodada: {float(med_run.median()):+.3f} m, "
        f"{int((med_run > 0).sum())}/{len(med_run)} rodadas positivas\n"
        "   as medianas MARGINAIS vao na direcao oposta (11,19 m baseline vs 11,30 m B2)\n"
        "ESCOPO: taxas 6/12/24 (Pi_2' de 0,80 a 3,20); N=24, tau_xy=1 s, T_off=8 s, dt=0,05 s,\n"
        "8 sementes pareadas, regime NAO saturado (sat_frac=0). A taxa 48 esta FORA: declarada\n"
        "nao identificavel para a analise por evento (mediana de W_e no piso de 0,6 s).")
    fig.text(0.5, -0.10, foot, ha="center", va="top", fontsize=8, color="0.25",
             family="monospace")
    fig.tight_layout(rect=(0, 0.02, 1, 0.90))
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"fig_vao_absoluto.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"gravado: {os.path.basename(out)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
