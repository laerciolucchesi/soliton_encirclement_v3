#!/usr/bin/env python
"""Reanalise PAREADA da campanha de churn: todas as metricas ja coletadas, nao so egap_avg.

Motivacao (E4): a tese sustenta que o VAO ANGULAR MAXIMO e' a quantidade
mission-critical, mas a robustez sob churn foi reportada sobre o erro MEDIO de
espacamento (egap_avg). Este script pareia baseline x B2 por (rate_total, seed) --
mesmo fluxo de falhas de Poisson, porque compartilham EXPERIMENT_SEED -- e reporta
mediana/min/max/n/n_lose + Wilcoxon pareado para CADA metrica do CSV.

ATENCAO A SEMANTICA (ver docs/experiments/CHURN_PAIRED.md, Tarefa 0):
  egap_* NAO sao o vao maximo. E_gap e' o RMS ESPACIAL do erro relativo de vao
  (protocol_target.py:686); egap_max e' o MAXIMO NO TEMPO desse RMS, sobre a janela
  t >= 20 s (run_churn_sweep.py:51-56). O vao angular maximo de verdade e' G_max
  (protocol_target.py:685), gravado em target_telemetry.csv e NUNCA agregado por
  esta campanha.

NAO roda simulacao: le apenas os *_results.csv ja existentes.

Uso:
    python experiments/scaling_law/analyze_churn_paired.py
    # env: CHURN_PAIRED_CSV="churn_sweep_results_c3_churn8_dt05.csv"
    #      CHURN_PAIRED_OUT="churn_paired_results.csv"
    #      CHURN_PAIRED_NOFIG="1"   (pula a figura)
"""
import os
import sys

import numpy as np
import pandas as pd

try:
    from scipy.stats import norm, wilcoxon
except ImportError:  # pragma: no cover - explicit, actionable failure
    sys.exit("scipy nao instalado. Rode: pip install -r requirements.txt")

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(EXP_DIR, "figures")

SRC_CSV = os.environ.get("CHURN_PAIRED_CSV", "churn_sweep_results_c3_churn8_dt05.csv")
OUT_CSV = os.environ.get("CHURN_PAIRED_OUT", "churn_paired_results.csv")

# Campanhas de churn para o cross-check da Tarefa 3 (robustez do padrao).
CROSS_CHECK = [
    "churn_sweep_results_c3_churn8_dt05.csv",
    "churn_sweep_results_m8off_ablation8seed.csv",
    "churn_sweep_results_c1B_m8on_dt01.csv",
    "churn_sweep_results_c1C_dt05.csv",
]

# (coluna, rotulo, direcao). direcao "lower" = menor e' melhor -> razao = base/B2
# (>1 => B2 melhor). direcao "cost" = razao invertida B2/base (>1 => B2 gasta mais),
# que e' como o esforco de controle deve ser lido.
METRICS = [
    ("egap_avg",       "E_gap medio (RMS espacial)",      "lower"),
    ("egap_p90",       "E_gap P90 no tempo",              "lower"),
    ("egap_max",       "E_gap MAXIMO no tempo",           "lower"),
    ("fairness_p95",   "fairness P95 (|e_tau_real|)",     "lower"),
    ("sat_frac",       "fracao de saturacao",             "lower"),
    ("effort_mean_v2", "esforco medio (v/Vmax)^2",        "cost"),
]

C_BASE = "firebrick"
C_B2 = "royalblue"


# ---------------------------------------------------------------- estatistica

def wilcoxon_paired(base, b2):
    """Wilcoxon pareado sobre os valores BRUTOS (nao sobre as razoes).

    Retorna (W, p, z, r, nota). z vem da aproximacao normal com correcao de
    empates -- scipy nao devolve z -- e r = |z|/sqrt(n_eff) e' o tamanho de efeito
    convencional para o signed-rank. n_eff descarta os pares com diferenca zero
    (zero_method="wilcox", o default do scipy).
    """
    base = np.asarray(base, float)
    b2 = np.asarray(b2, float)
    ok = np.isfinite(base) & np.isfinite(b2)
    base, b2 = base[ok], b2[ok]
    d = base - b2
    nz = d[d != 0.0]
    n_eff = nz.size
    if n_eff == 0:
        return (float("nan"),) * 4 + ("todas as diferencas sao exatamente zero",)
    if n_eff < 5:
        note = f"n_eff={n_eff} < 5: p exato mas sem poder"
    else:
        note = ""
    try:
        res = wilcoxon(base, b2)
    except ValueError as exc:
        return (float("nan"),) * 4 + (f"wilcoxon falhou: {exc}",)
    w = float(res.statistic)
    p = float(res.pvalue)

    # z pela aproximacao normal, com correcao de empates nos |d|.
    mean_w = n_eff * (n_eff + 1) / 4.0
    ranks = pd.Series(np.abs(nz)).rank().to_numpy(float)
    _, counts = np.unique(ranks, return_counts=True)
    tie_corr = float(np.sum(counts ** 3 - counts)) / 48.0
    var_w = n_eff * (n_eff + 1) * (2 * n_eff + 1) / 24.0 - tie_corr
    z = (w - mean_w) / np.sqrt(var_w) if var_w > 0 else float("nan")
    r = abs(z) / np.sqrt(n_eff) if np.isfinite(z) else float("nan")
    return w, p, float(z), float(r), note


def ratios(base, b2, direction):
    """Razao pareada por par. 'lower'/'cost' definem quem vai no numerador."""
    base = np.asarray(base, float)
    b2 = np.asarray(b2, float)
    num, den = (b2, base) if direction == "cost" else (base, b2)
    out = np.full(base.shape, np.nan)
    ok = np.isfinite(num) & np.isfinite(den) & (den > 0)
    out[ok] = num[ok] / den[ok]
    return out


def paired_values(df, metric, rate=None):
    """Pares (baseline, B2) casados explicitamente por (rate_total, seed).

    O merge e' obrigatorio: no escopo agregado o mesmo seed aparece uma vez por
    taxa, entao indexar so por seed casaria celulas de taxas diferentes (e contaria
    n errado). O pareamento e' o que da validade ao teste -- baseline e B2
    compartilham EXPERIMENT_SEED, logo o MESMO fluxo de falhas de Poisson.
    """
    g = df if rate is None else df[df.rate_total == rate]
    keys = ["rate_total", "seed"]
    b = g[g.method == "baseline"][keys + [metric]]
    o = g[g.method == "B2"][keys + [metric]]
    m = b.merge(o, on=keys, suffixes=("_base", "_b2")).sort_values(keys)
    return (m[f"{metric}_base"].to_numpy(float),
            m[f"{metric}_b2"].to_numpy(float),
            len(m))


def summarize(df, metric, direction, rate=None):
    """Uma linha do CSV de saida: par (metrica, escopo)."""
    base, b2, n_pairs = paired_values(df, metric, rate)

    rr = ratios(base, b2, direction)
    fin = rr[np.isfinite(rr)]
    w, p, z, r_eff, note = wilcoxon_paired(base, b2)

    # n_lose = pares em que o B2 e' PIOR. Para 'cost', pior = gasta mais (razao > 1).
    if direction == "cost":
        lose = fin > 1.0
        lose5 = fin > 1.05
    else:
        lose = fin < 1.0
        lose5 = fin < 0.95   # mesma margem de run_churn_sweep.py:151
    return {
        "metric": metric,
        "direction": direction,
        "scope": "agregado" if rate is None else f"rate={rate:g}",
        "rate_total": np.nan if rate is None else float(rate),
        "n": n_pairs,
        "base_median": float(np.nanmedian(base)) if base.size else np.nan,
        "b2_median": float(np.nanmedian(b2)) if b2.size else np.nan,
        "ratio_median": float(np.median(fin)) if fin.size else np.nan,
        "ratio_min": float(np.min(fin)) if fin.size else np.nan,
        "ratio_max": float(np.max(fin)) if fin.size else np.nan,
        "n_lose": int(np.sum(lose)),
        "n_lose_5pct": int(np.sum(lose5)),
        "w_stat": w,
        "p_value": p,
        "z": z,
        "r_effect": r_eff,
        "note": note,
    }


def load(name):
    p = os.path.join(EXP_DIR, name)
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


# -------------------------------------------------------------------- figura

def make_figure(df, present, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11,
                         "axes.labelsize": 10, "legend.fontsize": 8.5,
                         "figure.dpi": 150})
    rates = sorted(df.rate_total.unique())
    ncol = 3
    nrow = int(np.ceil(len(present) / ncol))
    # Cada metrica ocupa DUAS colunas do grid: slopegraph (largo) + razoes (estreito).
    fig, axes = plt.subplots(nrow, 2 * ncol, figsize=(5.4 * ncol, 3.9 * nrow),
                             gridspec_kw={"width_ratios": [2.2, 1.0] * ncol})
    axes = np.atleast_2d(axes)

    for k, (metric, label, direction) in enumerate(present):
        r, c = divmod(k, ncol)
        ax, axr = axes[r, 2 * c], axes[r, 2 * c + 1]

        all_ratios = []
        for i, rate in enumerate(rates):
            base, b2, _ = paired_values(df, metric, rate)
            x0, x1 = i - 0.17, i + 0.17
            for vb, vo in zip(base, b2):
                ax.plot([x0, x1], [vb, vo], color="0.65", lw=0.8, alpha=0.75, zorder=1)
            ax.scatter([x0] * len(base), base, s=22, color=C_BASE, zorder=3,
                       label="baseline" if i == 0 else None)
            ax.scatter([x1] * len(b2), b2, s=22, color=C_B2, zorder=3,
                       label="B2 (overlay)" if i == 0 else None)
            mb, mo = float(np.nanmedian(base)), float(np.nanmedian(b2))
            ax.plot([x0, x1], [mb, mo], color="black", lw=2.2, zorder=4)
            ax.scatter([x0, x1], [mb, mo], s=60, facecolor="white",
                       edgecolor="black", lw=1.6, zorder=5,
                       label="mediana" if i == 0 else None)

            rr = ratios(base, b2, direction)
            rr = rr[np.isfinite(rr)]
            all_ratios.append(rr)
            jitter = np.linspace(-0.14, 0.14, len(rr)) if len(rr) > 1 else np.zeros(len(rr))
            good = (rr > 1.0) if direction != "cost" else (rr < 1.0)
            axr.scatter(i + jitter, rr, s=16,
                        color=np.where(good, C_B2, C_BASE).tolist(), zorder=3)
            if rr.size:
                axr.plot([i - 0.2, i + 0.2], [np.median(rr)] * 2,
                         color="black", lw=2.0, zorder=4)

        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels([f"{r:g}" for r in rates])
        ax.set_xlabel("taxa de churn [falhas/min, total]")
        ax.set_title(label, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        if k == 0:
            ax.legend(loc="upper left", frameon=True)

        axr.axhline(1.0, color="0.35", lw=1.4, ls="--", zorder=2)
        axr.set_xticks(range(len(rates)))
        axr.set_title("razao " + ("B2/base" if direction == "cost" else "base/B2"),
                      fontsize=9)
        axr.grid(True, axis="y", alpha=0.3)
        if not any(a.size for a in all_ratios):
            # Metrica degenerada (ex.: sat_frac == 0 em todas as celulas): dizer
            # isso na figura, em vez de exibir um painel vazio que parece um bug.
            axr.set_xticklabels([f"{r:g}" for r in rates], fontsize=8)
            for a in (ax, axr):
                a.text(0.5, 0.88, "todas as celulas identicas (= 0)",
                       ha="center", va="center", transform=a.transAxes,
                       fontsize=8.5, color="0.35", style="italic")
            continue
        flat = np.concatenate([a for a in all_ratios if a.size])
        lo, hi = float(np.nanmin(flat)), float(np.nanmax(flat))
        pad = 0.12 * max(hi - lo, 0.2)
        axr.set_ylim(min(lo, 1.0) - pad, max(hi, 1.0) + pad)
        axr.set_xticklabels([f"{r:g}" for r in rates], fontsize=8)
        axr.set_xlim(-0.45, len(rates) - 1 + 0.45)   # espaco p/ o rotulo vertical
        # p-valor de Wilcoxon por taxa, escrito na VERTICAL dentro do painel: a
        # figura fica auditavel sem o CSV ao lado, e 4 rotulos nao se sobrepoem
        # num painel estreito (que foi o que aconteceu na horizontal).
        for i, rate in enumerate(rates):
            b_, o_, _ = paired_values(df, metric, rate)
            _, p, _, _, _ = wilcoxon_paired(b_, o_)
            txt = "n/a" if not np.isfinite(p) else (
                f"p={p:.3f}" if p >= 0.001 else "p<0.001")
            axr.annotate(txt, (i + 0.28, 0.97), xycoords=("data", "axes fraction"),
                         rotation=90, ha="center", va="top", fontsize=7,
                         color=("black" if np.isfinite(p) and p < 0.05 else "0.45"))

    for k in range(len(present), nrow * ncol):
        r, c = divmod(k, ncol)
        axes[r, 2 * c].axis("off")
        axes[r, 2 * c + 1].axis("off")

    fig.suptitle("Churn: comparacao PAREADA baseline x B2 por (taxa, seed)\n"
                 "linhas cinza = um par (mesmo fluxo de falhas); preto = mediana; "
                 "tracejado = razao 1 (empate)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    print(f"\nSalvo: {out_path}")


# ---------------------------------------------------------------------- main

def git_provenance_early():
    """Estado do git ANTES de escrever qualquer saida.

    Se capturado depois, os proprios arquivos que este script escreve deixam a
    arvore suja e a linha registraria dirty=True para sempre (auto-referencia).
    Mesmo motivo pelo qual main.py escreve o manifesto antes da simulacao.
    """
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(EXP_DIR)))
        import provenance
        return provenance.git_provenance()
    except Exception:
        return "unknown", True


def main():
    commit, dirty = git_provenance_early()
    df = load(SRC_CSV)
    if df is None:
        sys.exit(f"CSV nao encontrado: {SRC_CSV}")
    present = [(m, lab, d) for (m, lab, d) in METRICS if m in df.columns]
    absent = [m for (m, _, _) in METRICS if m not in df.columns]

    n_pairs = len(df[df.method == "baseline"])
    print(f"Fonte: {SRC_CSV}  ({len(df)} linhas, {n_pairs} celulas baseline, "
          f"N={sorted(df.N.unique())}, tau_xy={sorted(df.tau_xy.unique())}, "
          f"taxas={sorted(df.rate_total.unique())}, seeds={sorted(df.seed.unique())})")
    if absent:
        print(f"AUSENTES neste CSV: {absent}")
    print()

    rows = []
    for metric, label, direction in present:
        for rate in sorted(df.rate_total.unique()):
            rows.append(summarize(df, metric, direction, rate))
        rows.append(summarize(df, metric, direction, None))
    out = pd.DataFrame(rows)
    # Regra 5 da campanha vale tambem para saida de ANALISE: a linha tem de dizer
    # de qual CSV e de qual versao do script ela saiu. Aqui a proveniencia e' a do
    # ESTE processo (o script de analise), nao a das simulacoes-fonte -- as rodadas
    # originais sao anteriores ao P0 e nao tem manifesto (ver check_provenance.py).
    out.insert(0, "source_csv", SRC_CSV)
    out.insert(1, "analysis_git_commit", commit)
    out.insert(2, "analysis_git_dirty", bool(dirty))

    hdr = (f"{'metrica':<16}{'escopo':<12}{'n':>3}{'med':>8}{'min':>8}{'max':>8}"
           f"{'n_lose':>7}{'p':>10}{'r':>7}")
    for metric, label, direction in present:
        print(f"--- {metric}  ({'custo B2/base' if direction == 'cost' else 'vantagem base/B2'}) "
              f"| {label}")
        print(hdr)
        for _, r in out[out.metric == metric].iterrows():
            p = "  n/a" if not np.isfinite(r.p_value) else f"{r.p_value:10.4f}"
            rr = "  n/a" if not np.isfinite(r.r_effect) else f"{r.r_effect:7.2f}"
            print(f"{'':<16}{r.scope:<12}{int(r.n):>3}{r.ratio_median:>8.2f}"
                  f"{r.ratio_min:>8.2f}{r.ratio_max:>8.2f}{int(r.n_lose):>7}{p}{rr}"
                  + (f"   [{r.note}]" if r.note else ""))
        print()

    out_path = os.path.join(EXP_DIR, OUT_CSV)
    out.to_csv(out_path, index=False)
    print(f"Escrito: {out_path}  ({len(out)} linhas)")

    # ---- Tarefa 3: o padrao do egap_max se repete nas outras campanhas? ----
    print("\n=== CROSS-CHECK: egap_max (e egap_avg) nas demais campanhas de churn ===")
    print(f"{'campanha':<44}{'metrica':<10}{'n':>4}{'med':>8}{'min':>8}{'max':>8}{'n_lose':>8}{'p':>9}")
    for name in CROSS_CHECK:
        d = load(name)
        if d is None:
            print(f"  {name:<42} AUSENTE")
            continue
        for metric in ("egap_avg", "egap_max"):
            if metric not in d.columns:
                continue
            s = summarize(d, metric, "lower", None)
            p = "  n/a" if not np.isfinite(s["p_value"]) else f"{s['p_value']:9.4f}"
            print(f"{name:<44}{metric:<10}{s['n']:>4}{s['ratio_median']:>8.2f}"
                  f"{s['ratio_min']:>8.2f}{s['ratio_max']:>8.2f}{s['n_lose']:>8}{p}")

    if os.environ.get("CHURN_PAIRED_NOFIG", "").strip() not in ("1", "true", "True"):
        make_figure(df, present, os.path.join(OUT_DIR, "fig_churn_paired.png"))


if __name__ == "__main__":
    main()
