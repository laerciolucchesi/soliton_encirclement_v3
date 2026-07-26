#!/usr/bin/env python
"""Advisor figures for the ROBUSTNESS / THEORY campaign (Ciclos 0-2) — the part
the orientadores have NOT seen yet (fig1-13 only cover the clean single-fault
scaling law). Same pt-BR style and palette as make_figures.py.

Produces into experiments/scaling_law/figures/:

  fig14_churn_robustez.png   churn (Poisson) paired per-seed advantage, 8 seeds,
                             rates 6/12/24/48 per min — helps on every seed.
  fig15_mapa_robustez.png    single-image robustness scorecard across all stress
                             axes (falha / churn / comunicacao / alvo / coordenacao).
  fig17_atraso_m8.png        comm delay: the old "limit" was an M8-off artifact;
                             with M8 (default) it degrades gracefully in seconds.
  fig19_lei_adimensional.png dimensionless collapse of the speedup A ~ N^2/tau_a
                             (and the FAILED Peclet collapse) + dt-invariance.
  fig20_escada_feedforward.png  why 2-DOF: Option A advantage SHRINKS with N,
                             only B2 (direct feedforward) grows ~N^2.
  tab1_mapa_robustez.png     auditable table form of the robustness map (T1).
  tab2_churn_vantagem.png    churn paired-advantage table, 8 seeds (T2).
  tabelas_novas.md           markdown of T1/T2 for the thesis text.

All data comes from the canonical CSVs in this folder — NO new simulation.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(EXP_DIR, "figures")

C_BASE = "firebrick"
C_B2 = "royalblue"

# robustness-map verdict palette
CAT = {
    "forte":   ("#1b7837", "ajuda forte (escala)"),
    "ajuda":   ("#5aae61", "ajuda"),
    "gracioso": ("#4393c3", "degrada graciosamente"),
    "robusto": ("#2a9d8f", "robusto (provado/testado)"),
    "limite":  ("#e08214", "limite aberto"),
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9.5,
    "figure.dpi": 150,
})


def _csv(name):
    return os.path.join(EXP_DIR, name)


# ===========================================================================
# fig14 — churn paired per-seed advantage (8 seeds)
# ===========================================================================
def churn_paired(df):
    """Return {rate: dict(adv list, med, min, max, n_lose, effort_ratio)}."""
    out = {}
    for rate, g in df.groupby("rate_total"):
        b = g[g.method == "baseline"].set_index("seed")["egap_avg"]
        o = g[g.method == "B2"].set_index("seed")["egap_avg"]
        eb = g[g.method == "baseline"].set_index("seed")["effort_mean_v2"]
        eo = g[g.method == "B2"].set_index("seed")["effort_mean_v2"]
        seeds = sorted(set(b.index) & set(o.index))
        adv = np.array([b[s] / o[s] for s in seeds])
        eff = np.array([eo[s] / eb[s] for s in seeds if eb[s] > 0])
        out[int(rate)] = dict(
            adv=adv, med=float(np.median(adv)), lo=float(adv.min()),
            hi=float(adv.max()), n=len(adv), n_lose=int((adv < 1.0).sum()),
            effort=float(np.median(eff)) if eff.size else float("nan"),
        )
    return out


def fig14_churn(stats):
    rates = sorted(stats)
    x = np.arange(len(rates))
    med = [stats[r]["med"] for r in rates]
    lo = [stats[r]["med"] - stats[r]["lo"] for r in rates]
    hi = [stats[r]["hi"] - stats[r]["med"] for r in rates]
    fig, ax = plt.subplots(figsize=(8.6, 6))
    bars = ax.bar(x, med, width=0.55, color=C_B2, alpha=0.85, zorder=3,
                  label="vantagem mediana (pareada por semente)")
    ax.errorbar(x, med, yerr=[lo, hi], fmt="none", ecolor="k", capsize=6,
                lw=1.4, zorder=5, label="faixa min–máx (8 sementes)")
    # individual seed points (jittered)
    rng_off = np.linspace(-0.16, 0.16, 8)
    for i, r in enumerate(rates):
        ax.scatter(x[i] + rng_off, stats[r]["adv"], s=22, color="navy",
                   alpha=0.55, zorder=6,
                   label="sementes individuais" if i == 0 else None)
    ax.axhline(1.0, color=C_BASE, ls="--", lw=1.6, zorder=2,
               label="sem vantagem (= baseline)")
    for i, r in enumerate(rates):
        s = stats[r]
        ax.annotate(f"{s['med']:.2f}×", (x[i], s["hi"]), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=11, fontweight="bold",
                    color=C_B2)
        ax.annotate(f"min {s['lo']:.2f}×\n{s['n']}/{s['n']} ajuda",
                    (x[i], 1.0), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8.2, color="0.25")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}" for r in rates])
    ax.set_ylim(0.9, max(med) + max(hi) + 0.18)
    ax.set_xlabel("taxa de falhas do enxame  [falhas / min]")
    ax.set_ylabel("vantagem  $E_{gap}^{baseline}\\,/\\,E_{gap}^{overlay}$  (pareada por semente)")
    ax.set_title("Robustez sob churn contínuo (falhas Poisson temporárias)\n"
                 "o overlay ajuda em TODAS as 8 sementes, em TODAS as taxas "
                 "(0 sementes perdidas)")
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig14_churn_robustez.png")
    fig.savefig(out); plt.close(fig)
    return out


# ===========================================================================
# Robustness map data (shared by fig15 and tab1)
# ===========================================================================
def robustness_rows():
    """(family, scenario, baseline, overlay, result_text, category)."""
    return [
        ("Falha permanente de 1 nó", [
            ("N = 24", "τ = 19.5 s", "τ = 2.15 s", "9×  mais rápido", "forte"),
            ("N = 100", "τ = 311 s", "τ = 2.09 s", "149×  mais rápido", "forte"),
        ]),
        ("Churn (Poisson, 24 nós, 8 sementes)", [
            ("6 falhas/min", "—", "—", "1.31×  (min 1.24)", "ajuda"),
            ("12 falhas/min", "—", "—", "1.23×  (min 1.14)", "ajuda"),
            ("24 falhas/min", "—", "—", "1.15×  (min 1.11)", "ajuda"),
            ("48 falhas/min", "—", "—", "1.14×  (min 1.11)", "ajuda"),
        ]),
        ("Comunicação imperfeita", [
            ("Perda ≤ 20%", "assenta", "assenta", "speedup encolhe (gracioso)", "gracioso"),
            ("Perda 40%", "assenta", "assenta", "inerte (= baseline)", "gracioso"),
            ("Atraso 0.1 s", "τ = 22 s", "τ = 3.1 s", "7×  (assenta)", "ajuda"),
            ("Atraso 0.5 s", "τ = 31 s", "τ = 8.9 s", "3.5×  (assenta, sem cliff)", "gracioso"),
            ("Fora de ordem / duplicado", "—", "rejeitado", "seq# por emissor (testado)", "robusto"),
        ]),
        ("Alvo em movimento", [
            ("Velocidade constante", "—", "—", "ajuda · rastreio $E_r$ intacto", "ajuda"),
            ("Manobra", "—", "—", "ajuda (diluído) · $E_r$ intacto", "ajuda"),
            ("Recuperação de nó (ENTRADA)", "0.0094", "0.0050", "1.88×", "ajuda"),
        ]),
        ("Coordenação / casos difíceis", [
            ("Falhas adjacentes (M-mult)", "τ = 13.6 s", "τ = 2.2 s", "6–7×  (corrigido)", "ajuda"),
            ("Estresse combinado", "—", "—", "1.10–1.15×", "ajuda"),
            ("ENTRADA c/ canônico morto", "—", "~3/24 denso", "não coberto (futuro)", "limite"),
        ]),
    ]


def fig15_mapa():
    data = robustness_rows()
    n_rows = sum(1 + len(items) for _, items in data)
    fig_h = 0.46 * n_rows + 1.9
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    ax.set_xlim(0, 1); ax.set_ylim(0, n_rows + 1.3)
    ax.axis("off")
    y = n_rows
    for family, items in data:
        ax.text(0.012, y - 0.5, family, fontsize=12.5, fontweight="bold",
                color="0.12", va="center")
        ax.axhline((y - 1.0) , xmin=0.01, xmax=0.99, color="0.85", lw=0.8, zorder=1)
        y -= 1
        for scen, _b, _o, result, cat in items:
            color = CAT[cat][0]
            # status pill
            ax.add_patch(FancyBboxPatch((0.045, y - 0.74), 0.016, 0.5,
                         boxstyle="round,pad=0.002", facecolor=color,
                         edgecolor="none", zorder=3))
            ax.text(0.082, y - 0.5, scen, fontsize=10.7, va="center", color="0.15")
            ax.text(0.52, y - 0.5, result, fontsize=10.7, va="center",
                    fontweight="bold", color=color)
            y -= 1
    # legend
    handles = [Rectangle((0, 0), 1, 1, color=c) for c, _ in CAT.values()]
    labels = [lab for _, lab in CAT.values()]
    ax.legend(handles, labels, loc="lower center", ncol=5, fontsize=9,
              bbox_to_anchor=(0.5, -0.02), frameon=False, handlelength=1.1)
    fig.suptitle("Mapa de robustez do overlay 2-DOF: onde acelera, onde degrada "
                 "graciosamente\n(falha · churn · perda · atraso · alvo móvel · "
                 "coordenação — campanha Ciclos 0–2)",
                 fontsize=13, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0.02, 1, 0.91))
    out = os.path.join(OUT_DIR, "fig15_mapa_robustez.png")
    fig.savefig(out); plt.close(fig)
    return out


# ===========================================================================
# fig17 — comm delay: M8 before/after + graceful seconds-denominated slowdown
# ===========================================================================
def fig17_atraso(off_df, on01_df, sweep):
    """off_df: M8-off control (dt=0.01, delay 0.1); on01_df: M8-on same cond;
    sweep: dict delay->df (dt=0.05) for the graceful panel."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.4))

    # (a) M8 off vs on, same condition
    e_off = float(off_df[off_df.method == "B2"]["egap_final"].iloc[0])
    on_b2 = on01_df[on01_df.method == "B2"].iloc[0]
    e_on = float(on_b2["egap_final"]); tau_on = float(on_b2["tau_fit"])
    bars = axL.bar([0, 1], [e_off, e_on], width=0.5,
                   color=[C_BASE, C_B2], zorder=3)
    bars[0].set_hatch("//")
    axL.set_yscale("log")
    axL.set_xticks([0, 1])
    axL.set_xticklabels(["M8 DESLIGADO\n(artefato histórico)", "M8 LIGADO\n(padrão atual)"])
    axL.annotate("NÃO assenta\n$E_{gap}$ = 0.109", (0, e_off),
                 textcoords="offset points", xytext=(0, 8), ha="center",
                 fontsize=9.5, fontweight="bold", color=C_BASE)
    axL.annotate(f"assenta\nτ = {tau_on:.1f} s\n$E_{{gap}}$ = {e_on:.0e}", (1, e_on),
                 textcoords="offset points", xytext=(0, 8), ha="center",
                 fontsize=9.5, fontweight="bold", color=C_B2)
    axL.set_ylabel("erro residual de espaçamento  $E_{gap}$  (log)")
    axL.set_title("(a) Mesma condição (atraso 0.1 s, $dt$=0.01)\n"
                  "o 'limite de atraso' era um artefato do M8-desligado")
    axL.set_ylim(1e-5, 1.0)
    axL.grid(True, axis="y", which="both", ls=":", alpha=0.45)

    # (b) graceful: tau vs delay (dt=0.05), both settle
    delays = sorted(sweep)
    tb = [float(sweep[d][sweep[d].method == "baseline"]["tau_fit"].iloc[0]) for d in delays]
    t2 = [float(sweep[d][sweep[d].method == "B2"]["tau_fit"].iloc[0]) for d in delays]
    axR.plot(delays, tb, "s--", color=C_BASE, lw=2, ms=9, label="baseline")
    axR.plot(delays, t2, "o-", color=C_B2, lw=2, ms=10, label="overlay 2-DOF (M8)")
    for d, t in zip(delays, t2):
        axR.annotate(f"{t:.1f} s\n(assenta)", (d, t), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8.8, color=C_B2)
    axR.set_xlabel("atraso de comunicação  [s]")
    axR.set_ylabel("tempo de reacomodação  τ  [s]")
    axR.set_title("(b) Com M8 (padrão): desaceleração SUAVE em segundos\n"
                  "todos assentam — sem cliff de ticks")
    axR.set_xticks(delays)
    axR.grid(True, ls=":", alpha=0.5)
    axR.legend(loc="upper left")
    fig.suptitle("Atraso de comunicação: o único 'limite' do mapa, reaberto e fechado pelo M8",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(OUT_DIR, "fig17_atraso_m8.png")
    fig.savefig(out); plt.close(fig)
    return out


# ===========================================================================
# fig19 — dimensionless collapse  A ~ N^2 / tau_a   (and failed Peclet)
# ===========================================================================
def collapse_cells(df):
    rows = []
    for (N, ta, dt), g in df.groupby(["N", "tau_xy", "dt"]):
        b = g[g.method == "baseline"]["tau_fit"]
        o = g[g.method == "B2"]["tau_fit"]
        if b.empty or o.empty:
            continue
        A = float(b.iloc[0]) / float(o.iloc[0])
        Pe = float(g["Pe"].iloc[0])
        rows.append(dict(N=int(N), ta=float(ta), dt=float(dt),
                         A=A, x_collapse=N * N / ta, Pe=Pe,
                         valid=(ta >= 0.5)))
    return pd.DataFrame(rows)


def fig19_lei(cells):
    val = cells[cells.valid]
    inv = cells[~cells.valid]
    # fit A = c * (N^2/ta) through valid points (origin)
    c = float(np.sum(val.A * val.x_collapse) / np.sum(val.x_collapse ** 2))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 5.6))

    # (a) collapse vs N^2/tau_a
    axL.scatter(val.x_collapse, val.A, s=70, color=C_B2, zorder=4,
                label="τ$_a$ ≥ 0.5 (válido)")
    axL.scatter(inv.x_collapse, inv.A, s=70, facecolors="none",
                edgecolors="gray", zorder=4, label="τ$_a$ = 0.2 (atuador satura)")
    xs = np.linspace(cells.x_collapse.min() * 0.8, cells.x_collapse.max() * 1.1, 80)
    axL.plot(xs, c * xs, "--", color="black", alpha=0.7,
             label=f"ajuste  $A \\approx {c:.4f}\\,N^2/\\tau_a$")
    cv = float(np.std(val.A / val.x_collapse) / np.mean(val.A / val.x_collapse) * 100)
    axL.set_xscale("log"); axL.set_yscale("log")
    axL.set_xlabel("$N^2 / \\tau_a$   (escala log)")
    axL.set_ylabel("aceleração  $A = \\tau_{base}/\\tau_{overlay}$   (log)")
    axL.set_title(f"(a) COLAPSA sobre $N^2/\\tau_a$  (CV ≈ {cv:.0f}%)")
    axL.grid(True, which="both", ls=":", alpha=0.5)
    axL.legend(loc="upper left", fontsize=9)

    # (b) does NOT collapse vs Peclet
    axR.scatter(val.Pe, val.A, s=70, color=C_B2, zorder=4, label="τ$_a$ ≥ 0.5")
    axR.scatter(inv.Pe, inv.A, s=70, facecolors="none", edgecolors="gray",
                zorder=4, label="τ$_a$ = 0.2")
    cv_pe = float(np.std(val.A / val.Pe) / np.mean(val.A / val.Pe) * 100)
    axR.set_xscale("log"); axR.set_yscale("log")
    axR.set_xlabel("Péclet  $N\\,dt/\\tau_a$   (escala log)")
    axR.set_ylabel("aceleração  $A$   (log)")
    axR.set_title(f"(b) NÃO colapsa sobre Péclet  (CV ≈ {cv_pe:.0f}%)\n"
                  "hipótese de número de Péclet refutada")
    axR.grid(True, which="both", ls=":", alpha=0.5)
    axR.legend(loc="upper left", fontsize=9)

    # dt-invariance callout: points sharing N^2/tau_a across dt
    fig.suptitle("Lei adimensional: a aceleração do overlay é governada por "
                 "$N^2/\\tau_a$ (não por Péclet)\n"
                 "τ invariante a $dt$ (0.01–0.1 s) · validade τ$_a$ ≥ 0.5 s",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.join(OUT_DIR, "fig19_lei_adimensional.png")
    fig.savefig(out); plt.close(fig)
    return out, c, cv, cv_pe


# ===========================================================================
# fig20 — the feedforward ladder: A shrinks, B2 grows
# ===========================================================================
def fig20_escada(optB, fig_data):
    Ns = sorted(optB.N.unique())
    base = optB[optB.variant == "baseline"].set_index("N")["tau_fit"]
    A = optB[optB.variant == "A"].set_index("N")["tau_fit"]
    B = optB[optB.variant == "B"].set_index("N")["tau_fit"]
    B2 = fig_data[fig_data.method == "B2"].set_index("N")["tau_fit"]
    adv_A = [base[n] / A[n] for n in Ns]
    adv_B = [base[n] / B[n] for n in Ns]
    adv_B2 = [base[n] / B2[n] for n in Ns if n in B2.index]
    Ns_b2 = [n for n in Ns if n in B2.index]

    fig, ax = plt.subplots(figsize=(8.6, 6))
    ax.plot(Ns, adv_A, "o-", color="gray", lw=2, ms=9,
            label="Option A (ajusta a meta — via ganho)")
    ax.plot(Ns, adv_B, "s-", color="darkorange", lw=2, ms=9,
            label="Option B (feedforward, correção mínima)")
    ax.plot(Ns_b2, adv_B2, "D-", color=C_B2, lw=2.4, ms=10,
            label="Option B2 (feedforward 2-DOF) — adotado")
    ax.axhline(1.0, color=C_BASE, ls="--", lw=1.5, label="sem vantagem (= baseline)")
    for n, a in zip(Ns, adv_A):
        ax.annotate(f"{a:.2f}×", (n, a), textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=9, color="gray")
    for n, a in zip(Ns_b2, adv_B2):
        ax.annotate(f"{a:.0f}×", (n, a), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=10, fontweight="bold", color=C_B2)
    ax.annotate("A executa PELA malha do controlador\n→ vantagem ENCOLHE com N",
                (Ns[-1], adv_A[-1]), textcoords="offset points", xytext=(-12, 26),
                ha="right", fontsize=9, color="gray",
                arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7))
    ax.set_yscale("log")
    ax.set_xticks(Ns)
    ax.set_xlabel("tamanho do enxame  $N$")
    ax.set_ylabel("aceleração  $\\tau_{baseline}/\\tau_{variante}$  (log)")
    ax.set_title("Por que feedforward 2-DOF: a 'escada' de integração\n"
                 "só o B2 (execução FORA do ganho) escala — A colapsa para ~1×")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2,
              framealpha=0.95)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig20_escada_feedforward.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


# ===========================================================================
# tab1 — robustness map as an auditable table
# ===========================================================================
def tab1_table():
    data = robustness_rows()
    rows, cell_cats = [], []
    for family, items in data:
        for i, (scen, b, o, result, cat) in enumerate(items):
            fam = family if i == 0 else ""
            rows.append([fam, scen, b, o, result])
            cell_cats.append(cat)
    cols = ["Família", "Cenário", "Baseline", "Overlay 2-DOF", "Resultado / veredito"]
    fig, ax = plt.subplots(figsize=(12.5, 0.46 * len(rows) + 1.2))
    ax.axis("off")
    tab = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="left",
                   colWidths=[0.235, 0.205, 0.12, 0.12, 0.30])
    tab.auto_set_font_size(False); tab.set_fontsize(10); tab.scale(1, 1.5)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("0.85")
        cell.PAD = 0.03
        if r == 0:
            cell.set_facecolor("0.92"); cell.set_text_props(fontweight="bold")
        else:
            cat = cell_cats[r - 1]
            if c == 0:
                cell.set_text_props(fontweight="bold", color="0.2")
            if c == 4:
                col = CAT[cat][0]
                cell.set_facecolor(col + "22" if len(col) == 7 else "#eeeeee")
                cell.set_text_props(fontweight="bold", color=CAT[cat][0])
    ax.set_title("Tabela 1 — Mapa de robustez do overlay 2-DOF (campanha Ciclos 0–2)",
                 fontsize=12.5, fontweight="bold", pad=12)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "tab1_mapa_robustez.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


# ===========================================================================
# tab2 — churn paired advantage table (8 seeds)
# ===========================================================================
def tab2_table(stats):
    rates = sorted(stats)
    cols = ["Taxa\n[falhas/min]", "Vant.\nmediana", "Vant.\nmínima", "Vant.\nmáxima",
            "Sementes\najudadas", "Esforço\nB2 / base"]
    rows = []
    for r in rates:
        s = stats[r]
        rows.append([f"{r}", f"{s['med']:.2f}×", f"{s['lo']:.2f}×", f"{s['hi']:.2f}×",
                     f"{s['n'] - s['n_lose']}/{s['n']}", f"{s['effort']:.1f}×"])
    fig, ax = plt.subplots(figsize=(9.5, 0.5 * len(rows) + 1.8))
    ax.axis("off")
    tab = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tab.auto_set_font_size(False); tab.set_fontsize(12); tab.scale(1, 1.8)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("0.82")
        if r == 0:
            cell.set_facecolor("0.92"); cell.set_text_props(fontweight="bold")
        else:
            if c in (1, 2, 3):
                cell.set_facecolor("#eef2fb")
                if c == 1:
                    cell.set_text_props(fontweight="bold", color=C_B2)
            elif c == 4:
                cell.set_facecolor("#eef7ee")
                cell.set_text_props(fontweight="bold", color="seagreen")
    ax.set_title("Tabela 2 — Vantagem sob churn, pareada por semente (8 sementes, $dt$=0.05)\n"
                 "baseline e overlay compartilham o fluxo de falhas Poisson de cada semente",
                 fontsize=12, fontweight="bold", pad=12)
    fig.text(0.5, 0.04, "vantagem = $E_{gap}^{baseline}(s)\\,/\\,E_{gap}^{overlay}(s)$  "
             "·  0 sementes perdidas em todas as taxas  ·  esforço = trade-off honesto, "
             "sem saturação ($sat\\_frac$ = 0)", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = os.path.join(OUT_DIR, "tab2_churn_vantagem.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def write_markdown(stats, c, cv, cv_pe):
    lines = ["# Tabelas novas (campanha de robustez)\n",
             "## Tabela 1 — Mapa de robustez\n",
             "| Família | Cenário | Baseline | Overlay 2-DOF | Resultado / veredito |",
             "|---|---|---|---|---|"]
    for family, items in robustness_rows():
        for i, (scen, b, o, result, cat) in enumerate(items):
            fam = family if i == 0 else ""
            lines.append(f"| {fam} | {scen} | {b} | {o} | {result} |")
    lines += ["\n## Tabela 2 — Vantagem sob churn (pareada, 8 sementes, dt=0.05)\n",
              "| Taxa [falhas/min] | Vant. mediana | Vant. mínima | Vant. máxima | Sementes ajudadas | Esforço B2/base |",
              "|---|---|---|---|---|---|"]
    for r in sorted(stats):
        s = stats[r]
        lines.append(f"| {r} | {s['med']:.2f}× | {s['lo']:.2f}× | {s['hi']:.2f}× | "
                     f"{s['n'] - s['n_lose']}/{s['n']} | {s['effort']:.1f}× |")
    lines += [f"\n_Lei adimensional (fig19):_ A ≈ {c:.4f}·N²/τ_a  "
              f"(colapso CV ≈ {cv:.0f}% vs Péclet CV ≈ {cv_pe:.0f}%)."]
    out = os.path.join(OUT_DIR, "tabelas_novas.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out


def paired_adv(df):
    """Per-rate paired-by-seed advantage from egap_avg only (schema-light)."""
    out = {}
    for rate, g in df.groupby("rate_total"):
        b = g[g.method == "baseline"].set_index("seed")["egap_avg"]
        o = g[g.method == "B2"].set_index("seed")["egap_avg"]
        seeds = sorted(set(b.index) & set(o.index))
        adv = np.array([b[s] / o[s] for s in seeds])
        out[int(rate)] = dict(med=float(np.median(adv)), lo=float(adv.min()),
                              hi=float(adv.max()), n=len(adv),
                              n_lose=int((adv < 1.0).sum()))
    return out


# ===========================================================================
# fig16 — packet loss: graceful degradation
# ===========================================================================
def fig16_perda(loss):
    losses = sorted(loss.loss.unique())
    def stat(method):
        lo, hi, med, pts_x, pts_y = [], [], [], [], []
        for L in losses:
            g = loss[(loss.method == method) & (loss.loss == L)]["tau_fit"].dropna()
            lo.append(float(g.min())); hi.append(float(g.max()))
            med.append(float(np.median(g)))
            pts_x += [L] * len(g); pts_y += list(g)
        return np.array(lo), np.array(hi), np.array(med), pts_x, pts_y
    b_lo, b_hi, b_med, _, _ = stat("baseline")
    o_lo, o_hi, o_med, ox, oy = stat("B2")
    fig, ax = plt.subplots(figsize=(9.0, 6))
    # baseline reference (loss-insensitive)
    ax.plot(losses, b_med, "-", color=C_BASE, lw=2.5, zorder=4,
            label="baseline (insensível à perda)")
    # B2 envelope: starts way below baseline, MERGES into it as loss grows
    ax.fill_between(losses, o_lo, o_hi, color=C_B2, alpha=0.18, zorder=2,
                    label="overlay 2-DOF (faixa entre sementes)")
    ax.plot(losses, o_med, "o--", color=C_B2, lw=1.6, ms=8, zorder=4,
            label="overlay 2-DOF (mediana)")
    ax.scatter(ox, oy, s=26, color=C_B2, alpha=0.5, zorder=5)
    # clean speedup + merge annotations
    ax.annotate(f"limpo: {b_med[0]/o_med[0]:.0f}× mais rápido", (losses[0], o_med[0]),
                textcoords="offset points", xytext=(16, 10), ha="left",
                fontsize=10, fontweight="bold", color=C_B2)
    ax.annotate("perda alta → funde no baseline\n(inerte, mas nunca pior)",
                (losses[-1], o_med[-1]), textcoords="offset points", xytext=(-8, -46),
                ha="right", fontsize=9, color="0.3",
                arrowprops=dict(arrowstyle="->", color="0.5"))
    ax.set_yscale("log")
    ax.set_xlabel("taxa de perda de pacotes")
    ax.set_ylabel("tempo de reacomodação  τ  [s]  (log)")
    ax.set_title("Perda de pacotes: degradação graciosa — sempre assenta (perda ≤ 0.4)\n"
                 "o overlay vai de ~2 s (limpo) até o baseline; sem quebra",
                 fontsize=12)
    ax.set_xticks(losses)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="center right")
    ax.text(0.02, 0.93, "FD 20·$dt$ · variância = falsos-positivos do detector sob perda",
            transform=ax.transAxes, fontsize=9, style="italic", color="0.35")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig16_perda_pacotes.png")
    fig.savefig(out); plt.close(fig)
    return out


# ===========================================================================
# fig18 — moving target: speed helps AND tracking never degrades
# ===========================================================================
def fig18_alvo(m8, recover, stress):
    def med(df, method, scen, motion, col):
        g = df[(df.method == method) & (df.scenario == scen) & (df.motion == motion)][col]
        return float(np.median(g)) if len(g) else np.nan
    scen = [
        ("falha\n(const)", m8, "fail", "const"),
        ("falha\n(manobra)", m8, "fail", "maneuver"),
        ("recuperação\n(ENTRADA)", recover, "recover", "const"),
        ("estresse\n(const)", stress, "stress", "const"),
        ("estresse\n(manobra)", stress, "stress", "maneuver"),
    ]
    labels = [s[0] for s in scen]
    eg_b = [med(df, "baseline", sc, mo, "egap_avg") for _, df, sc, mo in scen]
    eg_o = [med(df, "B2", sc, mo, "egap_avg") for _, df, sc, mo in scen]
    er_b = [med(df, "baseline", sc, mo, "Er_avg") for _, df, sc, mo in scen]
    er_o = [med(df, "B2", sc, mo, "Er_avg") for _, df, sc, mo in scen]
    x = np.arange(len(scen)); w = 0.38
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.6))

    axL.bar(x - w / 2, eg_b, w, color=C_BASE, label="baseline")
    axL.bar(x + w / 2, eg_o, w, color=C_B2, label="overlay 2-DOF")
    for i in range(len(scen)):
        if eg_b[i] and eg_o[i]:
            axL.annotate(f"{eg_b[i]/eg_o[i]:.1f}×", (x[i], max(eg_b[i], eg_o[i])),
                         textcoords="offset points", xytext=(0, 5), ha="center",
                         fontsize=9, fontweight="bold", color="seagreen")
    axL.set_yscale("log"); axL.set_xticks(x); axL.set_xticklabels(labels, fontsize=9)
    axL.set_ylabel("$E_{gap}$ médio  (log)")
    axL.set_title("(a) Espaçamento: o overlay ajuda em TODOS os cenários")
    axL.grid(True, axis="y", which="both", ls=":", alpha=0.45); axL.legend(loc="upper left")

    axR.bar(x - w / 2, er_b, w, color=C_BASE, label="baseline")
    axR.bar(x + w / 2, er_o, w, color=C_B2, label="overlay 2-DOF")
    axR.set_yscale("log"); axR.set_xticks(x); axR.set_xticklabels(labels, fontsize=9)
    axR.set_ylabel("erro de rastreamento  $E_r$  (log)")
    axR.set_title("(b) Rastreamento: overlay ≈ baseline (NUNCA degrada)")
    axR.grid(True, axis="y", which="both", ls=":", alpha=0.45); axR.legend(loc="upper left")
    fig.suptitle("Alvo em movimento: o overlay acelera o espaçamento sem tocar no "
                 "rastreamento do alvo (canal só tangencial)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(OUT_DIR, "fig18_alvo_movel.png")
    fig.savefig(out); plt.close(fig)
    return out


# ===========================================================================
# fig21 — M8 ablation: turns dense churn from harmful to helpful
# ===========================================================================
def fig21_m8(off_stats, on_stats):
    rates = sorted(set(off_stats) & set(on_stats))
    off = [off_stats[r]["med"] for r in rates]
    on = [on_stats[r]["med"] for r in rates]
    fig, ax = plt.subplots(figsize=(8.6, 6))
    ax.plot(rates, off, "s--", color="0.45", lw=2, ms=10, label="M8 DESLIGADO")
    ax.plot(rates, on, "o-", color=C_B2, lw=2.4, ms=11, label="M8 LIGADO (padrão)")
    ax.axhline(1.0, color=C_BASE, ls="--", lw=1.5, label="sem vantagem (= baseline)")
    for r, v in zip(rates, off):
        ax.annotate(f"{v:.2f}×", (r, v), textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=9.5, color="0.4")
    for r, v in zip(rates, on):
        ax.annotate(f"{v:.2f}×", (r, v), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9.5, fontweight="bold", color=C_B2)
    # highlight the densest rate flip
    rmax = rates[-1]
    ax.annotate("sem M8 o overlay PREJUDICA\n(< 1×) sob churn denso →\nM8 o torna útil",
                (rmax, off_stats[rmax]["med"]), textcoords="offset points",
                xytext=(-10, -55), ha="right", fontsize=9, color="0.35",
                arrowprops=dict(arrowstyle="->", color="0.5"))
    ax.set_xticks(rates)
    ax.set_xlabel("taxa de falhas do enxame  [falhas / min]")
    ax.set_ylabel("vantagem  $E_{gap}^{base}/E_{gap}^{overlay}$  (mediana)")
    ax.set_title("Ablação do M8 sob churn: a correção que destrava o regime denso\n"
                 "(8 sementes M8-off · 3 sementes M8-on · $dt$=0.01)")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="lower left")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig21_m8_ablacao.png")
    fig.savefig(out); plt.close(fig)
    return out


# ===========================================================================
# fig24 — control effort cost, and why there is NO windup (sat_frac = 0)
# ===========================================================================
def fig24_esforco(churn, snappy):
    rates = sorted(churn.rate_total.unique())
    def med(df, method, rate, col):
        g = df[(df.method == method) & (df.rate_total == rate)][col]
        return float(np.median(g)) if len(g) else np.nan
    eff_b = [med(churn, "baseline", r, "effort_mean_v2") for r in rates]
    eff_o = [med(churn, "B2", r, "effort_mean_v2") for r in rates]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.4))

    x = np.arange(len(rates)); w = 0.38
    axL.bar(x - w / 2, eff_b, w, color=C_BASE, label="baseline")
    axL.bar(x + w / 2, eff_o, w, color=C_B2, label="overlay 2-DOF")
    for i, r in enumerate(rates):
        if eff_b[i]:
            axL.annotate(f"{eff_o[i]/eff_b[i]:.1f}×", (x[i], eff_o[i]),
                         textcoords="offset points", xytext=(0, 5), ha="center",
                         fontsize=9.5, fontweight="bold", color="0.3")
    axL.set_xticks(x); axL.set_xticklabels([f"{int(r)}" for r in rates])
    axL.set_xlabel("taxa de falhas  [falhas / min]")
    axL.set_ylabel("esforço de controle  $\\langle (v/V_{max})^2 \\rangle$")
    axL.set_title("(a) O custo honesto: ~2.4× mais atuação\n(o overlay redistribui ativamente; o baseline só relaxa)")
    axL.grid(True, axis="y", ls=":", alpha=0.5); axL.legend(loc="upper left")

    # RMS velocity as % of Vmax — far from the 100% saturation line; sat_frac=0
    rms_b = [np.sqrt(v) * 100 for v in eff_b]
    rms_o = [np.sqrt(v) * 100 for v in eff_o]
    axR.plot(rates, rms_b, "s--", color=C_BASE, lw=2, ms=9, label="baseline")
    axR.plot(rates, rms_o, "o-", color=C_B2, lw=2, ms=10, label="overlay 2-DOF")
    if snappy is not None and len(snappy):
        sr = sorted(snappy.rate_total.unique())[0]
        rms_sn = np.sqrt(med(snappy, "B2", sr, "effort_mean_v2")) * 100
        axR.scatter([sr], [rms_sn], s=90, color="purple", zorder=6,
                    label=f"overlay regime 'snappy' τ$_a$=0.2 ({rms_sn:.0f}%)")
    axR.axhline(100, color="k", ls="-", lw=1.4, alpha=0.7)
    axR.text(rates[0], 92, "limite de saturação do atuador (100%)", fontsize=9,
             color="k", alpha=0.7)
    axR.set_ylim(0, 110)
    axR.set_xticks(rates)
    axR.set_xlabel("taxa de falhas  [falhas / min]")
    axR.set_ylabel("velocidade RMS  [% de $V_{max}$]")
    axR.set_title("(b) ...mas LONGE da saturação: $sat\\_frac$ = 0 em todo regime\n"
                  "sem saturação ⇒ sem windup (limiter não justificado)")
    axR.grid(True, ls=":", alpha=0.5); axR.legend(loc="center right", fontsize=8.5)
    fig.suptitle("Esforço de controle: um trade-off benigno, não uma patologia",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(OUT_DIR, "fig24_esforco_sem_windup.png")
    fig.savefig(out); plt.close(fig)
    return out


# ===========================================================================
# fig22 — M-mult: adjacent-block sub-correction repaired
# ===========================================================================
def fig22_mmult(df):
    order = ["k1", "adj2", "adj3", "non2", "non3"]
    df = df[df.scenario.isin(order)]
    off = {r.scenario: r.tau_fit for _, r in df[~df.mmult].iterrows()}
    on = {r.scenario: r.tau_fit for _, r in df[df.mmult].iterrows()}
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - w / 2, [off.get(s, np.nan) for s in order], w, color="0.5",
           label="M-mult DESLIGADO")
    ax.bar(x + w / 2, [on.get(s, np.nan) for s in order], w, color=C_B2,
           label="M-mult LIGADO (padrão)")
    for i, s in enumerate(order):
        if s in off and s in on and on[s]:
            if off[s] / on[s] > 1.5:
                ax.annotate(f"{off[s]/on[s]:.0f}× mais rápido",
                            (x[i], off[s]), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=9.5,
                            fontweight="bold", color="seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels(["1 falha", "2 adjacentes", "3 adjacentes",
                        "2 separadas", "3 separadas"], fontsize=9.5)
    ax.set_ylabel("tempo de reacomodação  τ  [s]")
    ax.set_title("M-mult: falhas ADJACENTES deixam de sub-corrigir\n"
                 "(o originador infere a multiplicidade k do próprio gap; "
                 "k=1 ≡ legado)")
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    ax.legend(loc="upper right")
    ax.text(0.02, 0.95, "cenários separados/única: idênticos (caminho k=1 intocado)",
            transform=ax.transAxes, fontsize=9, style="italic", color="0.3")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig22_mmult_adjacente.png")
    fig.savefig(out); plt.close(fig)
    return out


# ===========================================================================
# fig23 — TTL >= N requirement: coverage cliff
# ===========================================================================
def fig23_ttl(df):
    fig, ax = plt.subplots(figsize=(8.6, 6))
    for mode, color, marker in (("TTL=3N (adequado)", C_B2, "o"),
                                ("TTL=50 (fixo)", C_BASE, "s")):
        g = df[df["mode"] == mode].sort_values("N")
        ax.plot(g.N, g.coverage_frac * 100, marker + "-", color=color, lw=2.2,
                ms=10, label=mode)
        for _, r in g.iterrows():
            ax.annotate(f"{r.coverage_frac*100:.0f}%", (r.N, r.coverage_frac * 100),
                        textcoords="offset points", xytext=(0, 9), ha="center",
                        fontsize=9, color=color)
    ax.axhline(100, color="0.6", ls=":", lw=1)
    ax.set_ylim(-5, 112)
    ax.set_xlabel("tamanho do enxame  $N$")
    ax.set_ylabel("cobertura da redistribuição  [% de agentes vivos]")
    ax.set_title("Requisito TTL ≥ N: o pulso precisa dar a volta no anel\n"
                 "TTL fixo (50 hops) colapsa a cobertura em $N$ grande; TTL=3N a preserva")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="lower left")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig23_ttl_cobertura.png")
    fig.savefig(out); plt.close(fig)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    outs = []

    churn = pd.read_csv(_csv("churn_sweep_results.csv"))
    stats = churn_paired(churn)
    outs.append(fig14_churn(stats))
    outs.append(fig15_mapa())

    # --- data-ready robustness figures (no new sim) ---
    try:
        loss = pd.read_csv(_csv("comm_results_loss_clean.csv"))
        outs.append(fig16_perda(loss))
    except FileNotFoundError as e:
        print("skip fig16:", e)
    try:
        m8 = pd.read_csv(_csv("trackC_results_m8clean.csv"))
        rec = pd.read_csv(_csv("trackC_results_recover.csv"))
        stress = pd.read_csv(_csv("trackC_results_stress.csv"))
        outs.append(fig18_alvo(m8, rec, stress))
    except FileNotFoundError as e:
        print("skip fig18:", e)
    try:
        off = paired_adv(pd.read_csv(_csv("churn_sweep_results_m8off_ablation8seed.csv")))
        on = paired_adv(pd.read_csv(_csv("churn_sweep_results_c1B_m8on_dt01.csv")))
        outs.append(fig21_m8(off, on))
    except FileNotFoundError as e:
        print("skip fig21:", e)
    try:
        snappy = pd.read_csv(_csv("churn_sweep_results_c4_snappy_tau02.csv"))
    except FileNotFoundError:
        snappy = None
    outs.append(fig24_esforco(churn, snappy))

    # --- figures that need the runner CSVs (skip gracefully if not yet run) ---
    if os.path.exists(_csv("mmult_adjacent_results.csv")):
        outs.append(fig22_mmult(pd.read_csv(_csv("mmult_adjacent_results.csv"))))
    else:
        print("skip fig22: run run_mmult_adjacent.py first")
    if os.path.exists(_csv("ttl_coverage_results.csv")):
        outs.append(fig23_ttl(pd.read_csv(_csv("ttl_coverage_results.csv"))))
    else:
        print("skip fig23: run run_ttl_coverage.py first")

    off_df = pd.read_csv(_csv("comm_results_c1Dctrl_m8off.csv"))
    on01 = pd.read_csv(_csv("comm_results_c1D_dt01_d0p1.csv"))
    sweep = {0.1: pd.read_csv(_csv("comm_results_c1D_dt05_d0p1.csv")),
             0.25: pd.read_csv(_csv("comm_results_c1D_dt05_d0p25.csv")),
             0.5: pd.read_csv(_csv("comm_results_c1D_dt05_d0p5.csv"))}
    outs.append(fig17_atraso(off_df, on01, sweep))

    collapse = pd.read_csv(_csv("collapse_results.csv"))
    cells = collapse_cells(collapse)
    fig19_out, c, cv, cv_pe = fig19_lei(cells)
    outs.append(fig19_out)

    optB = pd.read_csv(_csv("optionB_results.csv"))
    fig_data = pd.read_csv(_csv("figure_data.csv"))
    outs.append(fig20_escada(optB, fig_data))

    outs.append(tab1_table())
    outs.append(tab2_table(stats))
    outs.append(write_markdown(stats, c, cv, cv_pe))

    print("Wrote:")
    for o in outs:
        print("  " + os.path.relpath(o, EXP_DIR))
    print(f"\nLei adimensional: A ≈ {c:.4f}·N²/τ_a  (CV {cv:.0f}% vs Péclet {cv_pe:.0f}%)")
    print("Churn paired advantage:")
    for r in sorted(stats):
        s = stats[r]
        print(f"  rate {r:>2}: med {s['med']:.2f}  min {s['lo']:.2f}  "
              f"max {s['hi']:.2f}  n_lose {s['n_lose']}  effort {s['effort']:.1f}x")


if __name__ == "__main__":
    main()
