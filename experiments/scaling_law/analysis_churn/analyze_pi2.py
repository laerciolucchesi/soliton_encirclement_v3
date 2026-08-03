#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Camada FINA sobre analyze_churn_paired.py: vantagem da coordenacao x Pi_2'.

O QUE ESTE SCRIPT NAO FAZ (decisao -1.4): nao reimplementa pareamento, razoes nem
Wilcoxon. Importa paired_values / ratios / wilcoxon_paired de
experiments/scaling_law/analyze_churn_paired.py, cujos numeros ja foram conferidos
celula a celula (25/25, rtol=1e-9) contra churn_paired_results.csv e contra uma
recomputacao independente. Um segundo caminho de calculo poderia divergir sem
ninguem notar -- que e' o modo de falha da regra R4.

O QUE ACRESCENTA, e so isso:
  * Pi_2' = lambda_anel * T_off, com lambda_anel = rate_total/60 (semantica fixada
    na FASE 0), mais a correcao de renovacao E[ausentes] = Pi_2'/(1 + Pi_2'/N).
  * IQR entre sementes (o original reporta min/max).
  * media-das-razoes vs razao-das-medias (assimetria).
  * FASE 2b: teste de tendencia PAREADO POR SEMENTE (n=8 independentes), no lugar
    do Spearman sobre 32 pares (que viola independencia: a mesma semente aparece
    nas 4 taxas). Os dois sao reportados.
  * 10 sentinelas de carga (o original nao valida nada da entrada).

Fonte canonica (DECISAO 2): churn_sweep_results_c3_churn8_dt05.csv.
churn_sweep_results.csv e' byte-identico e NAO deve ser citado.

Uso:
    python experiments/scaling_law/analysis_churn/analyze_pi2.py
"""
import hashlib
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
EXP_DIR = os.path.dirname(HERE)
sys.path.insert(0, EXP_DIR)
import analyze_churn_paired as acp   # noqa: E402  reuso: paired_values/ratios/wilcoxon_paired

SRC_NAME = "churn_sweep_results_c3_churn8_dt05.csv"      # nome canonico (DECISAO 2)
SRC = os.path.join(EXP_DIR, SRC_NAME)
TWIN = "churn_sweep_results.csv"                          # byte-identico; nao citar

FIG_METRICS = ["egap_avg", "egap_p90", "egap_max"]        # entram na figura
DIAG_METRICS = ["effort_mean_v2", "sat_frac", "fairness_p95"]   # diagnosticas
BASE, OVER = "baseline", "B2"

_log = []


def say(msg=""):
    print(msg)
    _log.append(str(msg))


def flush_log():
    with open(os.path.join(HERE, "LOG_execucao.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(_log) + "\n")


def die(msg):
    say("")
    say("=" * 74)
    say("SENTINELA FALHOU -- ANALISE ABORTADA (nenhuma figura gerada)")
    say(msg)
    say("=" * 74)
    flush_log()
    sys.exit(1)


def sha256_16(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


# ---------------------------------------------------------------- Pi_2'
def pi2(rate_total, off):
    """Pi_2' = lambda_anel * T_off. lambda_anel = rate_total/60 [1/s] (FASE 0)."""
    return (rate_total / 60.0) * off


def pi2_renewal(rate_total, off, n):
    """E[ausentes] exato = Pi_2'/(1 + Pi_2'/N).

    Um agente OFF nao sorteia novas falhas (protocol_agent.py:920-922 nao reagenda o
    timer; :966 so reagenda na recuperacao), entao o processo por agente e' up~Exp,
    down=T_off deterministico e a fracao de tempo OFF e' lam_a*T/(1+lam_a*T), nao
    lam_a*T. Pi_2' e' a aproximacao de baixa densidade dessa expressao.
    """
    p = pi2(rate_total, off)
    return p / (1.0 + p / float(n))


# ---------------------------------------------------------------- FASE 1
def load_and_check():
    say("=" * 74)
    say("FASE 1 -- CARGA E SENTINELAS")
    say("=" * 74)
    if not os.path.exists(SRC):
        die(f"S0: fonte canonica ausente: {SRC}")
    digest = sha256_16(SRC)
    say(f"fonte canonica : {SRC_NAME}")
    say(f"sha256[:16]    : {digest}")
    twin = os.path.join(EXP_DIR, TWIN)
    if os.path.exists(twin) and sha256_16(twin) == digest:
        say(f"nota           : {TWIN} e' BYTE-IDENTICO a esta fonte (mesmo sha256). "
            f"Nao e' campanha distinta e nao deve ser citado.")

    df = pd.read_csv(SRC)

    if len(df) != 64:
        die(f"S1: esperado 64 linhas, encontrado {len(df)}.")
    say(f"S1 OK  linhas = {len(df)}")

    methods = sorted(df["method"].unique().tolist())
    if len(methods) != 2:
        die(f"S2: esperado exatamente 2 metodos, encontrado {methods}.")
    say(f"S2 OK  metodos = {methods}")

    if set(methods) != {BASE, OVER}:
        die(f"S8: rotulos {methods} != {{'{BASE}','{OVER}'}} -- a orientacao da razao "
            f"base/B2 nao pode ser garantida.")
    say(f"S8 OK  orientacao travada: numerador '{BASE}', denominador '{OVER}'")

    dup = int(df.duplicated(subset=["method", "rate_total", "seed"]).sum())
    if dup:
        die(f"S6: {dup} linha(s) duplicada(s) em (method, rate_total, seed).")
    say("S6 OK  nenhuma celula (metodo, taxa, semente) duplicada")

    ref = None
    for (m, r), g in df.groupby(["method", "rate_total"]):
        seeds = tuple(sorted(g["seed"].tolist()))
        if ref is None:
            ref = seeds
        elif seeds != ref:
            die(f"S3: ({m}, rate={r}) tem sementes {seeds}, esperado {ref}.")
    say(f"S3 OK  toda celula (metodo, taxa) tem as sementes {list(ref)}")

    consts = {}
    for col in ("N", "tau_xy", "off"):
        vals = df[col].unique()
        if len(vals) != 1:
            die(f"S4: coluna '{col}' NAO e constante: {sorted(vals)}.")
        consts[col] = float(vals[0])
    say(f"S4 OK  N={consts['N']:g} | tau_xy={consts['tau_xy']:g}s | "
        f"off(T_off)={consts['off']:g}s constantes nas 64 linhas")

    for col in FIG_METRICS:
        n_nan = int(df[col].isna().sum())
        if n_nan:
            die(f"S5: coluna '{col}' tem {n_nan} NaN.")
    say(f"S5 OK  sem NaN em {FIG_METRICS}")

    for col in FIG_METRICS:
        bad = int((~(df[col] > 0)).sum())
        if bad:
            die(f"S7: coluna '{col}' tem {bad} valor(es) <= 0 -- razao indefinida.")
    say("S7 OK  metricas da figura todas > 0")

    for r in sorted(df.rate_total.unique()):
        p = pi2(r, consts["off"])
        if not (p < consts["N"]):
            die(f"S9: rate_total={r:g} -> Pi_2'={p:.2f} >= N={consts['N']:g}. "
                f"Semantica de rate_total incoerente com a FASE 0.")
    say(f"S9 OK  Pi_2' < N em todas as taxas (max={max(pi2(r, consts['off']) for r in df.rate_total.unique()):.2f} "
        f"de N={consts['N']:g})")

    # S10: diferencas pareadas identicamente nulas. ABORTA nas metricas da figura;
    # nas diagnosticas registra a nota, no estilo de acp.wilcoxon_paired:83-84.
    notes = {}
    for met in FIG_METRICS + DIAG_METRICS:
        if met not in df.columns:
            continue
        for rate in sorted(df.rate_total.unique()):
            b, o, _ = acp.paired_values(df, met, rate)
            if not np.any(np.asarray(b, float) != np.asarray(o, float)):
                if met in FIG_METRICS:
                    die(f"S10: '{met}' rate={rate:g} tem TODAS as diferencas pareadas "
                        f"exatamente zero -- o Wilcoxon exato e indefinido e a razao "
                        f"seria 1.000 por construcao.")
                notes.setdefault(met, "todas as diferencas sao exatamente zero")
    if notes:
        for met, note in notes.items():
            say(f"S10 nota  '{met}': {note} (metrica diagnostica, fora da figura) "
                f"-- DECLARACAO DE ESCOPO, nao falha")
    say("S10 OK  nenhuma metrica da figura e degenerada")
    return df, consts, digest, notes


def inventory(df, consts, notes):
    say("")
    say("-- 1.2 ESCOPO: o que varia e o que nao varia ---------------------------")
    rates = sorted(df["rate_total"].unique().tolist())
    seeds = sorted(df["seed"].unique().tolist())
    say(f"VARIA    method      = {sorted(df['method'].unique().tolist())}")
    say(f"VARIA    rate_total  = {rates} falhas/min TOTAIS do anel")
    say(f"VARIA    seed        = {seeds} (n={len(seeds)} por celula, pareadas)")
    say(f"CONST    N           = {consts['N']:g}")
    say(f"CONST    tau_xy      = {consts['tau_xy']:g} s")
    say(f"CONST    off (T_off) = {consts['off']:g} s")
    say(f"CONST    dt          = 0.05 s (run_churn_sweep.py:44; campanha '_dt05')")
    say(f"CONST    canal       = loss=0, delay=0 (run_churn_sweep.py:72)")
    say(f"CONST    alvo        = estacionario (run_churn_sweep.py:73)")
    say(f"CONST    init        = equidistante, sem dispersao de raio (run_churn_sweep.py:73)")
    say(f"CONST    ganho       = K_E_TAU = 250/N (run_churn_sweep.py:70)")
    say(f"CONST    janela      = t >= 20 s ate 155 s (run_churn_sweep.py:41-42,51)")

    say("")
    say("-- 1.3 SATURACAO DO ATUADOR --------------------------------------------")
    s = df["sat_frac"].to_numpy(float)
    say(f"sat_frac: min={s.min():.6g} media={s.mean():.6g} max={s.max():.6g}")
    unsat = bool(np.all(s == 0.0))
    if unsat:
        say("=> sat_frac == 0 em 100% das 64 celulas: campanha inteiramente no regime")
        say("   NAO SATURADO do atuador. Isto e' DECLARACAO DE ESCOPO, nao falha da")
        say("   metrica nem do teste (vai tambem na legenda da figura).")
    else:
        say("=> ATENCAO: ha saturacao de atuador; o escopo 'nao saturado' NAO se aplica.")
    return unsat


# ---------------------------------------------------------------- FASE 2
def analyse(df, consts):
    say("")
    say("=" * 74)
    say("FASE 2 -- PAREADA POR SEMENTE (calculo reusado de analyze_churn_paired.py)")
    say("=" * 74)
    off, n_ag = consts["off"], consts["N"]
    rates = sorted(df["rate_total"].unique().tolist())

    r0 = rates[0]
    say(f"-- 2.4 Pi_2', conta explicita para rate_total = {r0:g} falhas/min ------")
    say(f"   lambda_anel = {r0:g}/60          = {r0/60.0:.6g} falhas/s")
    say(f"   Pi_2'       = {r0/60.0:.6g} * {off:g}  = {pi2(r0, off):.4f} agentes ausentes")
    say(f"   % do anel   = {100*pi2(r0, off)/n_ag:.2f}% de N={n_ag:g}")
    say(f"   renovacao   = {pi2(r0, off):.4f}/(1+{pi2(r0, off):.4f}/{n_ag:g}) = "
        f"{pi2_renewal(r0, off, n_ag):.4f} ausentes (exato)")

    rows, pairs = [], []
    for met in FIG_METRICS:
        for rate in rates:
            b, o, n = acp.paired_values(df, met, rate)
            rr = acp.ratios(b, o, "lower")
            rr = rr[np.isfinite(rr)]
            w, p, z, r_eff, note = acp.wilcoxon_paired(b, o)
            rows.append({
                "metric": met, "rate_total": rate, "n": n,
                "pi2": pi2(rate, off), "pi2_renewal": pi2_renewal(rate, off, n_ag),
                "base_median": float(np.median(b)), "b2_median": float(np.median(o)),
                "ratio_median": float(np.median(rr)),
                "ratio_q25": float(np.percentile(rr, 25)),
                "ratio_q75": float(np.percentile(rr, 75)),
                "ratio_min": float(rr.min()), "ratio_max": float(rr.max()),
                "ratio_mean": float(np.mean(rr)),
                "mean_ratio": float(np.mean(b) / np.mean(o)),
                "w_stat": w, "p_value": p, "z": z, "r_effect": r_eff,
                "n_lose": int(np.sum(rr < 1.0)), "note": note,
            })
            # guarda os pares (a ordem de acp.paired_values e' sort por (rate, seed))
            seeds = sorted(df[(df.method == BASE) & (df.rate_total == rate)]["seed"].tolist())
            for s_, vb, vo, vr in zip(seeds, b, o, rr):
                pairs.append({"metric": met, "rate_total": rate, "seed": int(s_),
                              "pi2": pi2(rate, off), "baseline": vb, "B2": vo, "ratio": vr})
    summ = pd.DataFrame(rows)
    prs = pd.DataFrame(pairs)

    for met in FIG_METRICS:
        s = summ[summ.metric == met]
        say("")
        say(f"-- {met} " + "-" * (66 - len(met)))
        say(f"{'taxa':>5}{'Pi2':>7}{'renov':>7}{'base_med':>10}{'b2_med':>9}"
            f"{'raz_med':>9}{'q25':>8}{'q75':>8}{'med_raz':>9}{'raz_med.s':>10}"
            f"{'p':>9}{'n<1':>5}")
        for _, x in s.iterrows():
            say(f"{x['rate_total']:>5g}{x['pi2']:>7.2f}{x['pi2_renewal']:>7.2f}"
                f"{x['base_median']:>10.4f}{x['b2_median']:>9.4f}{x['ratio_median']:>9.4f}"
                f"{x['ratio_q25']:>8.4f}{x['ratio_q75']:>8.4f}{x['ratio_mean']:>9.4f}"
                f"{x['mean_ratio']:>10.4f}{x['p_value']:>9.6f}{int(x['n_lose']):>5}")
        d = (s["ratio_mean"] - s["mean_ratio"]).abs() / s["mean_ratio"]
        say(f"   media-das-razoes vs razao-das-medias: divergencia relativa maxima "
            f"{100*d.max():.2f}% (taxa {s.loc[d.idxmax(), 'rate_total']:g}) "
            f"-> {'assimetria desprezivel' if d.max() < 0.02 else 'ASSIMETRIA relevante'}")
    return summ, prs


def shape(df, consts):
    """FASE 2c -- FORMA da distribuicao: p90/avg e max/avg, por metodo e taxa.

    Calculada POR RODADA (razao dentro da mesma celula) e depois mediana entre
    sementes -- nao razao de medianas. As duas versoes sao impressas porque diferem,
    e a razao-de-medianas e' a que se obtem lendo a tabela 4.2 de fora.
    """
    say("")
    say("=" * 74)
    say("FASE 2c -- FORMA DA DISTRIBUICAO (p90/avg e max/avg)")
    say("=" * 74)
    off = consts["off"]
    rows = []
    for rate in sorted(df.rate_total.unique()):
        for meth in (BASE, OVER):
            g = df[(df.method == meth) & (df.rate_total == rate)]
            p90_avg = (g["egap_p90"] / g["egap_avg"]).to_numpy(float)
            max_avg = (g["egap_max"] / g["egap_avg"]).to_numpy(float)
            rows.append({
                "method": meth, "rate_total": rate, "pi2": pi2(rate, off),
                "p90_avg_med": float(np.median(p90_avg)),
                "p90_avg_q25": float(np.percentile(p90_avg, 25)),
                "p90_avg_q75": float(np.percentile(p90_avg, 75)),
                "max_avg_med": float(np.median(max_avg)),
                "max_avg_q25": float(np.percentile(max_avg, 25)),
                "max_avg_q75": float(np.percentile(max_avg, 75)),
                # versao "de fora": razao das medianas da tabela 4.2
                "p90_avg_from_medians": float(np.median(g["egap_p90"]) / np.median(g["egap_avg"])),
                "max_avg_from_medians": float(np.median(g["egap_max"]) / np.median(g["egap_avg"])),
            })
    sh = pd.DataFrame(rows)
    say(f"{'taxa':>5}{'Pi2':>7} {'metodo':<9}{'p90/avg':>9}{'[q25':>8}{'q75]':>8}"
        f"{'(medianas)':>12}{'max/avg':>9}{'[q25':>8}{'q75]':>8}{'(medianas)':>12}")
    for _, x in sh.iterrows():
        say(f"{x['rate_total']:>5g}{x['pi2']:>7.2f} {x['method']:<9}"
            f"{x['p90_avg_med']:>9.3f}{x['p90_avg_q25']:>8.3f}{x['p90_avg_q75']:>8.3f}"
            f"{x['p90_avg_from_medians']:>12.3f}"
            f"{x['max_avg_med']:>9.3f}{x['max_avg_q25']:>8.3f}{x['max_avg_q75']:>8.3f}"
            f"{x['max_avg_from_medians']:>12.3f}")
    say("")
    lo, hi = sh.rate_total.min(), sh.rate_total.max()
    for meth in (BASE, OVER):
        a = sh[(sh.method == meth) & (sh.rate_total == lo)].iloc[0]
        b = sh[(sh.method == meth) & (sh.rate_total == hi)].iloc[0]
        say(f"   {meth:<9} p90/avg: {a['p90_avg_med']:.2f} -> {b['p90_avg_med']:.2f} "
            f"(por rodada) | {a['p90_avg_from_medians']:.2f} -> "
            f"{b['p90_avg_from_medians']:.2f} (razao das medianas)")
    return sh


def trend(summ, prs):
    """FASE 2b: tendencia com n=8 INDEPENDENTES + o Spearman de 32 pares, para contraste."""
    say("")
    say("=" * 74)
    say("FASE 2b -- A VANTAGEM CAI COM O CHURN? (teste pareado por semente)")
    say("=" * 74)
    out = {}
    for met in FIG_METRICS:
        sub = prs[prs.metric == met]
        rates = sorted(sub.rate_total.unique())
        lo, hi = rates[0], rates[-1]
        a = sub[sub.rate_total == lo].set_index("seed")["ratio"].sort_index()
        b = sub[sub.rate_total == hi].set_index("seed")["ratio"].sort_index()
        if list(a.index) != list(b.index):
            die(f"S3c: sementes desalinhadas entre taxa {lo:g} e {hi:g} em '{met}'.")
        delta = (a - b).to_numpy(float)          # >0 => vantagem caiu de lo para hi
        n_down = int(np.sum(delta > 0))
        n_eff = int(np.sum(delta != 0))
        if n_eff == 0:
            die(f"S10b: '{met}' -- todas as diferencas delta_s sao zero; teste indefinido.")
        w_p = float(stats.wilcoxon(delta, method="exact").pvalue)
        w_s = float(stats.wilcoxon(delta, method="exact").statistic)
        sign_p = float(stats.binomtest(n_down, n_eff, 0.5).pvalue)
        # Spearman sobre os 32 pares (dependente: a mesma semente aparece nas 4 taxas)
        rho32, p32 = stats.spearmanr(sub["pi2"].to_numpy(float), sub["ratio"].to_numpy(float))
        say("")
        say(f"-- {met}: razao(taxa={lo:g}) - razao(taxa={hi:g}), por semente")
        say("   " + "  ".join(f"s{int(i)}={v:+.3f}" for i, v in zip(a.index, delta)))
        n_up = int(np.sum(delta < 0))
        if n_down == len(delta):
            direc = "CAI com o churn (unanime)"
        elif n_up == len(delta):
            direc = "SOBE com o churn (unanime)"
        elif n_down > n_up:
            direc = "cai na maioria"
        elif n_up > n_down:
            direc = "sobe na maioria"
        else:
            direc = "sem direcao (empate)"
        say(f"   sementes com QUEDA da vantagem: {n_down}/{len(delta)}  "
            f"(com ALTA: {n_up}/{len(delta)})  -> {direc}")
        say(f"   Wilcoxon pareado (1 amostra, exato, n={n_eff}): W={w_s:.1f}, p={w_p:.6f}")
        say(f"   Teste de sinal (binomial exato): p={sign_p:.6f}")
        say(f"   [contraste] Spearman razao~Pi_2' sobre {len(sub)} pares NAO independentes: "
            f"rho={rho32:.4f}, p={p32:.6g}")
        out[met] = {"lo": lo, "hi": hi, "delta": delta.tolist(), "n_down": n_down,
                    "n_up": n_up, "direc": direc,
                    "n": len(delta), "w": w_s, "p_wilcoxon": w_p, "p_sign": sign_p,
                    "rho32": float(rho32), "p32": float(p32)}
    say("")
    say("   Qual confiar: o teste de 2b (n=8 independentes -- cada semente contribui UMA")
    say("   diferenca). O Spearman de 32 pares reusa a mesma semente nas 4 taxas, logo seu")
    say("   p e' otimista por construcao; fica so como contraste.")
    say(f"   Piso de p com n=8: Wilcoxon bilateral exato = {2/2**8:.6f}; sinal = {2/2**8:.6f}.")
    return out


# ---------------------------------------------------------------- FASE 3
XLAB = r"$\Pi_2'$ = $\lambda_{anel}\cdot T_{off}$  (agentes ausentes, em media)"
CLAIM = ("Conforme o churn aperta, o ganho MIGRA do corpo da distribuicao para a cauda "
         "superior,\ne o extremo permanece intocado em todos os regimes")
FOOT = ("E_gap = RMS ESPACIAL do erro de vao, normalizado pelo n. de agentes VIVOS "
        "(protocol_target.py:707): mede QUALIDADE DE REDISTRIBUICAO, nao vao maximo nem "
        "cobertura absoluta.\nsat_frac = 0 em 100% das 64 celulas: campanha inteiramente "
        "no regime NAO SATURADO do atuador (declaracao de escopo, nao falha).")


def _rate_labels(ax, x, rates):
    """Rotulos de rastreabilidade em fracao de eixo (nao em coordenada de dado): a
    posicao nao depende da escala da metrica, entao nunca colidem com a curva nem com
    a linha de razao=1. As margens sao abertas ANTES de anotar."""
    y0, y1 = ax.get_ylim()
    pad = 0.13 * (y1 - y0)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_xlim(x.min() / 1.35, x.max() * 1.35)
    for xi, ri in zip(x, rates):
        ax.annotate(f"{ri:g}/min", (xi, 0.015), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=7.5, color="0.35")


def panel_absolute(ax, df, summ, met, letter="A"):
    s = summ[summ.metric == met].sort_values("pi2")
    x = s["pi2"].to_numpy(float)
    for meth, color, mk in ((BASE, "firebrick", "o"), (OVER, "royalblue", "s")):
        med, q25, q75 = [], [], []
        for rate in s["rate_total"]:
            v = df[(df.method == meth) & (df.rate_total == rate)][met].to_numpy(float)
            med.append(np.median(v)); q25.append(np.percentile(v, 25)); q75.append(np.percentile(v, 75))
        ax.plot(x, med, marker=mk, color=color, lw=1.8,
                label=(BASE if meth == BASE else "B2 (overlay)"))
        ax.fill_between(x, q25, q75, color=color, alpha=0.18)
    ax.set_xscale("log"); ax.set_xlabel(XLAB)
    ax.set_ylabel(f"{met}  (mediana; faixa = IQR)")
    ax.set_title(f"{letter}) {met}: os DOIS degradam com o churn", fontweight="bold")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    _rate_labels(ax, x, s["rate_total"])


def panel_ratio(ax, summ, met, letter="B"):
    s = summ[summ.metric == met].sort_values("pi2")
    x = s["pi2"].to_numpy(float)
    med = s["ratio_median"].to_numpy(float)
    err = [med - s["ratio_q25"].to_numpy(float), s["ratio_q75"].to_numpy(float) - med]
    color = MET_STYLE[met][0]
    ax.errorbar(x, med, yerr=err, marker="o", color=color, lw=1.8, capsize=4)
    ax.axhline(1.0, ls="--", color="0.35", lw=1.3)
    ax.set_xscale("log"); ax.set_xlabel(XLAB)
    ax.set_ylabel(f"razao pareada {BASE}/{OVER}")
    ax.set_title(f"{letter}) {met}: vantagem pareada (mediana; barras = IQR)",
                 fontweight="bold")
    ax.grid(alpha=0.3)
    _rate_labels(ax, x, s["rate_total"])
    # p de Wilcoxon + n_pior/n_melhor por taxa: a figura fica auditavel sem o CSV.
    for xi, pv, nl, nn in zip(x, s["p_value"], s["n_lose"], s["n"]):
        ax.annotate(f"p={pv:.4f}  ({int(nn)-int(nl)}/{int(nn)} a favor)",
                    (xi, 0.985), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=7, rotation=90, color="0.3")


def two_panels(axA, axB, df, summ, met):
    panel_absolute(axA, df, summ, met, "A")
    panel_ratio(axB, summ, met, "B")


MET_STYLE = {"egap_avg": ("seagreen", "o", "corpo (media)"),
             "egap_p90": ("darkorange", "s", "cauda superior (P90)"),
             "egap_max": ("slategray", "^", "extremo (max no tempo)")}


def panel_three_ratios(ax, summ):
    """As TRES razoes no mesmo painel: e' aqui que se le a migracao do ganho."""
    for met, (color, mk, lab) in MET_STYLE.items():
        s = summ[summ.metric == met].sort_values("pi2")
        x = s["pi2"].to_numpy(float)
        med = s["ratio_median"].to_numpy(float)
        err = [med - s["ratio_q25"].to_numpy(float), s["ratio_q75"].to_numpy(float) - med]
        ax.errorbar(x, med, yerr=err, marker=mk, color=color, lw=1.9, capsize=3.5,
                    label=f"{met} — {lab}")
    ax.axhline(1.0, ls="--", color="0.35", lw=1.3)
    ax.annotate("sem vantagem", (ax.get_xlim()[0], 1.0), xytext=(4, 3),
                textcoords="offset points", fontsize=7, color="0.4")
    ax.set_xscale("log"); ax.set_xlabel(XLAB)
    ax.set_ylabel(f"razao pareada {BASE}/{OVER}")
    ax.set_title("B) o ganho MIGRA do corpo para a cauda;\no extremo fica em ~1,0",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="center left"); ax.grid(alpha=0.3)
    s0 = summ[summ.metric == "egap_avg"].sort_values("pi2")
    _rate_labels(ax, s0["pi2"].to_numpy(float), s0["rate_total"])


def panel_shape(ax, sh):
    """Forma da distribuicao: por que a razao pico/media SOBE quando o corpo desce."""
    for meth, ls in ((BASE, "-"), (OVER, "--")):
        s = sh[sh.method == meth].sort_values("pi2")
        x = s["pi2"].to_numpy(float)
        ax.plot(x, s["p90_avg_med"], ls=ls, marker="s", color="darkorange",
                label=f"P90/media — {meth}")
        ax.plot(x, s["max_avg_med"], ls=ls, marker="^", color="slategray",
                label=f"max/media — {meth}")
    ax.set_xscale("log"); ax.set_xlabel(XLAB)
    ax.set_ylabel("forma da distribuicao (adimensional)")
    ax.set_title("C) forma: o extremo relativo ao corpo\ncai com o churn (mediana por rodada)",
                 fontweight="bold")
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3)
    s0 = sh[sh.method == BASE].sort_values("pi2")
    _rate_labels(ax, s0["pi2"].to_numpy(float), s0["rate_total"])


def figures(df, summ, sh, consts):
    say("")
    say("=" * 74)
    say("FASE 3 -- FIGURAS")
    say("=" * 74)
    sub = (f"N={consts['N']:g}, tau_xy={consts['tau_xy']:g}s, T_off={consts['off']:g}s, "
           f"dt=0.05s, n={int(summ['n'].iloc[0])} sementes pareadas, "
           f"fonte {SRC_NAME}")

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16.8, 4.9))
    panel_absolute(a1, df, summ, "egap_avg", "A")
    panel_three_ratios(a2, summ)
    panel_shape(a3, sh)
    fig.suptitle(CLAIM + "\n" + sub, fontsize=10.5, fontweight="bold")
    fig.text(0.5, -0.02, FOOT, ha="center", va="top", fontsize=7.5, color="0.3")
    fig.tight_layout(rect=(0, 0.02, 1, 0.88))
    for ext in ("png", "pdf"):
        p = os.path.join(HERE, f"fig_vantagem_vs_pi2linha.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        say(f"  gravado: {os.path.basename(p)}")
    plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 9.2))
    two_panels(ax[0][0], ax[0][1], df, summ, "egap_p90")
    two_panels(ax[1][0], ax[1][1], df, summ, "egap_max")
    fig.suptitle("Suplementar: egap_p90 (topo) e egap_max (base) x $\\Pi_2'$\n" + sub,
                 fontsize=10.5, fontweight="bold")
    fig.text(0.5, -0.01, FOOT + "\negap_max = MAXIMO NO TEMPO de um RMS espacial -- nao e' "
             "o vao angular maximo (esse e' G_max, ausente deste CSV).",
             ha="center", va="top", fontsize=7.5, color="0.3")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    for ext in ("png", "pdf"):
        p = os.path.join(HERE, f"fig_vantagem_vs_pi2linha_suplementar.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        say(f"  gravado: {os.path.basename(p)}")
    plt.close(fig)


# ---------------------------------------------------------------- FASE 4
def report(df, summ, sh, tr, consts, digest, unsat, notes):
    n_ag, off = consts["N"], consts["off"]
    n_seed = int(summ["n"].iloc[0])
    L = []
    A = L.append
    A("# RESULTADO — o ganho da coordenacao x Pi_2' (campanha c3_churn8_dt05)\n")
    A("> **Enunciado da figura.** Conforme o churn aperta, o ganho **migra do corpo da "
      "distribuicao para a cauda superior**, e o **extremo permanece intocado** em todos "
      "os regimes.\n")
    A("> Substitui o enunciado anterior (\"a vantagem cai com Pi_2'\"), que era **falso**: "
      "cai na media (1,31 -> 1,14, 8/8 sementes) e **sobe** no P90 (1,04 -> 1,13, 8/8 "
      "sementes). As duas direcoes tem o mesmo p bilateral, 0,0078.\n")
    A(f"Fonte canonica: `experiments/scaling_law/{SRC_NAME}`, sha256[:16] = `{digest}`.  ")
    A(f"`{TWIN}` e' **byte-identico** a esta fonte (mesmo sha256) — nao e' campanha "
      f"distinta e **nao deve ser citado em lugar nenhum**.  ")
    A("Calculo pareado (mediana, razoes, Wilcoxon) **reusado** de "
      "`experiments/scaling_law/analyze_churn_paired.py` (`paired_values`, `ratios`, "
      "`wilcoxon_paired`), verificado 25/25 celulas a rtol=1e-9 contra "
      "`churn_paired_results.csv`.\n")

    A("## 4.1 Semantica de `rate_total` (FASE 0)\n")
    A("**`rate_total` = taxa TOTAL do anel** (falhas/min somadas sobre os N agentes). "
      "Cadeia: `experiments/scaling_law/run_churn_sweep.py:61` calcula "
      "`per_agent = rate_total / float(N)` e `:76` grava esse valor em "
      "`FAILURE_MEAN_FAILURES_PER_MIN`; `protocol_agent.py:904-908` usa essa taxa num "
      "sorteio Bernoulli `p = 1-exp(-(rate/60)*dt)` com `dt = FAILURE_CHECK_PERIOD = 0.1 s` "
      "(`config_param.py:178`), executado **por agente** (`protocol_agent.py:918-920`, RNG "
      "dedicado semeado em `:99-101`, timer proprio em `:266-267`). As duas operacoes se "
      "cancelam: taxa do anel = N x (rate_total/N) = `rate_total`.\n")
    A("Confirmacao independente por dado de execucao: os `runs_summary.csv` preservados em "
      "`churn_sweep_runs_stamp/` registram `failure_mean_per_min` = 0,25 / 0,5 / 1,0 / 2,0 "
      "para rate 6 / 12 / 24 / 48 com N=24 — exatamente `rate_total/N`.\n")
    A("Teste de coerencia: sob a hipotese 'por agente', rate_total=12 exigiria "
      f"{pi2(12, off)*n_ag:.1f} agentes ausentes num anel de {n_ag:g} — impossivel.\n")

    A("## 4.2 Tabela principal — `egap_avg`\n")
    A("| rate_total (/min) | lambda_anel (1/s) | Pi_2' | % do anel | Pi_2' renovacao* | "
      "% renovacao | mediana baseline | mediana B2 | razao pareada mediana [IQR] | "
      "p Wilcoxon | sementes a favor / contra |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, x in summ[summ.metric == "egap_avg"].iterrows():
        n_, nl = int(x["n"]), int(x["n_lose"])
        A(f"| {x['rate_total']:g} | {x['rate_total']/60:.4f} | {x['pi2']:.2f} | "
          f"{100*x['pi2']/n_ag:.1f}% | {x['pi2_renewal']:.2f} | "
          f"{100*x['pi2_renewal']/n_ag:.1f}% | {x['base_median']:.4f} | "
          f"{x['b2_median']:.4f} | {x['ratio_median']:.3f} [{x['ratio_q25']:.3f}, "
          f"{x['ratio_q75']:.3f}] | {x['p_value']:.6f} | **{n_-nl}/{n_}** a favor, "
          f"{nl}/{n_} contra |")
    A("")
    A("> ### Os quatro p sao o PISO do teste, nao a forca do efeito\n")
    A(f"> Os quatro p de `egap_avg` valem **{2/2**n_seed:.6f} = 2/2^{n_seed}**, que e' o "
      f"**menor p bilateral possivel** no Wilcoxon exato com n={n_seed}. Ele so diz "
      f"\"as {n_seed} sementes concordam no sinal\" — e' o mesmo p para 1,31 e para 1,14. "
      f"**O teste por taxa NAO distingue 1,31 de 1,14 e nao deve ser usado para sustentar "
      f"a tendencia.**\n")
    A("> A tendencia se apoia em duas outras coisas, e so nelas:\n"
      "> 1. o **teste da FASE 2b** (secao 4.3), sobre 8 observacoes **independentes** — "
      "cada semente contribui UMA diferenca;\n"
      f"> 2. os **IQR que nao se sobrepoem** entre os extremos da varredura: "
      f"[{summ[(summ.metric=='egap_avg') & (summ.rate_total==summ.rate_total.min())]['ratio_q25'].iloc[0]:.3f}; "
      f"{summ[(summ.metric=='egap_avg') & (summ.rate_total==summ.rate_total.min())]['ratio_q75'].iloc[0]:.3f}] "
      f"na taxa minima vs "
      f"[{summ[(summ.metric=='egap_avg') & (summ.rate_total==summ.rate_total.max())]['ratio_q25'].iloc[0]:.3f}; "
      f"{summ[(summ.metric=='egap_avg') & (summ.rate_total==summ.rate_total.max())]['ratio_q75'].iloc[0]:.3f}] "
      f"na taxa maxima.\n")
    A("> **Regra adotada neste documento:** todo `p` reportado vem acompanhado da contagem "
      "de sementes a favor/contra. Um `p` sozinho, com n=8, nao e' informacao suficiente.\n")
    A("\\* `Pi_2' renovacao` = `Pi_2'/(1 + Pi_2'/N)` — numero medio exato de ausentes, dado "
      "que um agente OFF nao sorteia novas falhas (`protocol_agent.py:920-922` nao reagenda "
      "o timer; `:966` so reagenda na recuperacao). `Pi_2' = lambda_anel*T_off` e' a "
      "aproximacao de baixa densidade; o eixo x das figuras usa `Pi_2'`.\n")
    A(f"Wilcoxon pareado bilateral sobre os valores brutos, n={n_seed} por taxa.\n")

    A("### As outras duas metricas — onde o ganho vai parar\n")
    for met in ("egap_p90", "egap_max"):
        A(f"**`{met}`** — {MET_STYLE[met][2]}\n")
        A(f"| rate_total | Pi_2' | mediana baseline | mediana B2 | razao mediana [IQR] | "
          f"p Wilcoxon | sementes a favor / contra |")
        A("|---|---|---|---|---|---|---|")
        for _, x in summ[summ.metric == met].iterrows():
            n_, nl = int(x["n"]), int(x["n_lose"])
            A(f"| {x['rate_total']:g} | {x['pi2']:.2f} | {x['base_median']:.4f} | "
              f"{x['b2_median']:.4f} | {x['ratio_median']:.3f} [{x['ratio_q25']:.3f}, "
              f"{x['ratio_q75']:.3f}] | {x['p_value']:.6f} | {n_-nl}/{n_} a favor, "
              f"{nl}/{n_} contra |")
        A("")

    A("### Forma da distribuicao (FASE 2c) — o mecanismo\n")
    A("`p90/avg` e `max/avg` calculados **por rodada** e depois mediana entre sementes. "
      "A coluna \"(medianas)\" e' a razao das medianas da tabela 4.2 — a que se obtem "
      "lendo a tabela de fora; as duas sao dadas porque diferem.\n")
    A("| rate_total | Pi_2' | metodo | P90/media [IQR] | (medianas) | max/media [IQR] | (medianas) |")
    A("|---|---|---|---|---|---|---|")
    for _, x in sh.iterrows():
        A(f"| {x['rate_total']:g} | {x['pi2']:.2f} | {x['method']} | "
          f"{x['p90_avg_med']:.3f} [{x['p90_avg_q25']:.3f}, {x['p90_avg_q75']:.3f}] | "
          f"{x['p90_avg_from_medians']:.3f} | "
          f"{x['max_avg_med']:.3f} [{x['max_avg_q25']:.3f}, {x['max_avg_q75']:.3f}] | "
          f"{x['max_avg_from_medians']:.3f} |")
    A("")

    A("### Media das razoes vs razao das medias\n")
    for met in FIG_METRICS:
        s = summ[summ.metric == met]
        d = (s["ratio_mean"] - s["mean_ratio"]).abs() / s["mean_ratio"]
        parts = "; ".join(f"{x['rate_total']:g}: {x['ratio_mean']:.4f} vs {x['mean_ratio']:.4f}"
                          for _, x in s.iterrows())
        A(f"- `{met}`: {parts} — divergencia relativa maxima {100*d.max():.2f}%")
    A("")

    A("## 4.3 Tendencia: a vantagem cai com o churn?\n")
    A("Teste **pareado por semente** (FASE 2b): para cada semente, "
      "`delta_s = razao(taxa minima) - razao(taxa maxima)`; `delta_s > 0` = a vantagem caiu. "
      "n = 8 diferencas **independentes** (cada semente entra uma vez).\n")
    A("| metrica | taxas comparadas | sementes com queda | sementes com alta | direcao | "
      "Wilcoxon p (exato) | sinal p (exato) | [contraste] Spearman 32 pares rho | p |")
    A("|---|---|---|---|---|---|---|---|---|")
    for met in FIG_METRICS:
        t = tr[met]
        A(f"| {met} | {t['lo']:g} vs {t['hi']:g} | {t['n_down']}/{t['n']} | "
          f"{t['n_up']}/{t['n']} | **{t['direc']}** | "
          f"{t['p_wilcoxon']:.6f} | {t['p_sign']:.6f} | {t['rho32']:.4f} | {t['p32']:.6g} |")
    A("")
    A("O p do Wilcoxon aqui e' **bilateral**: mede se `delta_s` difere de zero, nao a "
      "direcao. Leia a direcao na coluna correspondente — `egap_avg` e `egap_p90` tem o "
      "mesmo p (0,0078, o piso com n=8) e direcoes **opostas**.")
    A("")
    A("O Spearman de 32 pares **viola independencia** (a mesma semente aparece nas 4 taxas), "
      "entao seu p e' otimista por construcao; esta na tabela apenas como contraste. "
      "O teste de referencia e' o pareado por semente.\n")

    A("## 4.4 ESCOPO DECLARADO (constante em toda a campanha)\n")
    A(f"- N = {n_ag:g} agentes")
    A(f"- tau_xy = {consts['tau_xy']:g} s")
    A(f"- T_off = {off:g} s — recuperacao finita: os agentes VOLTAM (churn, nao morte)")
    A("- dt (`CONTROL_PERIOD`) = 0,05 s (`run_churn_sweep.py:44`)")
    A(f"- **regime de saturacao: `sat_frac` == 0 em 100% das 64 celulas — campanha "
      f"inteiramente no regime NAO SATURADO do atuador. Isto e' DECLARACAO DE ESCOPO, "
      f"nao falha.**" if unsat else "- regime de saturacao: HA saturacao (ver LOG_execucao.txt)")
    A("- metrica: `egap_avg`/`egap_p90`/`egap_max` = media / P90 / MAXIMO **no tempo** de "
      "`E_gap`, sobre t >= 20 s ate 155 s (`run_churn_sweep.py:41-42,51-56`). `E_gap` e' o "
      "**RMS espacial** do erro relativo de vao, normalizado pelo numero de agentes **VIVOS** "
      "(`protocol_target.py:707`)")
    A(f"- n = {n_seed} sementes por celula, pareadas entre metodos: baseline e B2 compartilham "
      "`EXPERIMENT_SEED`, e o RNG de falha e' dedicado e independente do metodo "
      "(`protocol_agent.py:99-101`, `:918-919`), logo o fluxo de falhas e' o MESMO nos dois")
    A(f"- taxas varridas: {sorted(summ.rate_total.unique().tolist())} falhas/min TOTAIS "
      f"(Pi_2' de {pi2(min(summ.rate_total), off):.2f} a {pi2(max(summ.rate_total), off):.2f} agentes)")
    A("- canal ideal: `COMMUNICATION_FAILURE_RATE=0`, `COMMUNICATION_DELAY=0` "
      "(`run_churn_sweep.py:72`)")
    A("- alvo estacionario: `TARGET_MOTION_SPEED_XY=0` (`run_churn_sweep.py:73`)")
    A("- inicializacao equidistante, sem dispersao de raio (`run_churn_sweep.py:73`)")
    A("- ganho escalado: `K_E_TAU = 250/N` (`run_churn_sweep.py:70`)")
    A("")

    A("## 4.5 O QUE ESTA FIGURA NAO MOSTRA\n")
    A("- **Nao mostra tempo de assentamento.** `egap_avg` e' erro de REGIME PERMANENTE "
      "(media temporal sobre t >= 20 s). Nenhum `t_settle` foi medido nesta campanha; "
      "`metrics_util.settling_time` existe mas nao e' chamado por `run_churn_sweep.py`.")
    A("- **Nao e' comparavel com a vantagem medida no evento de falha unica.** Aquela usa "
      "falha deterministica e mede o transiente pos-evento; esta usa fluxo de Poisson com "
      "recuperacao e mede media temporal de regime. Estimulo, janela e metrica diferem.")
    A("- **Nao e' o vao angular maximo.** `egap_max` e' o maximo NO TEMPO de um RMS "
      "ESPACIAL — duas agregacoes empilhadas. O vao maximo e' `G_max` "
      "(`protocol_target.py:706`), que **nao existe neste CSV**.")
    A("- **Nao mede cobertura absoluta.** `E_gap` e `G_max` sao normalizados pelo numero de "
      "vivos: um anel com metade dos agentes, perfeitamente redistribuido, pontua igual a um "
      "anel cheio. Mede QUALIDADE DE REDISTRIBUICAO.")
    A("- **Nao separa taxa de duracao.** `T_off` e' constante, entao Pi_2' e `rate_total` sao "
      "proporcionais: a figura nao distingue efeito da TAXA do efeito da DURACAO da ausencia. "
      "Seria preciso variar `T_off` com Pi_2' fixo.")
    A("- **Nao varre N nem tau_xy.** Uma unica coluna do espaco de projeto "
      f"(N={n_ag:g}, tau_xy={consts['tau_xy']:g}). Nada aqui sustenta extrapolacao em N.")
    A("- **Nao mede custo.** `effort_mean_v2` e `fairness_p95` estao no CSV e ficaram fora "
      "desta figura de proposito; `analyze_churn_paired.py` ja os reporta (custo B2/baseline "
      "= 2,41x mediano, 32/32 pares).")
    A("- **Sem correcao para multiplas comparacoes** entre taxas e metricas; os p sao crus.")
    A(f"- **Piso de resolucao do teste:** com n={n_seed}, o menor p bilateral exato possivel e "
      f"{2/2**n_seed:.6f}. Um p igual a esse valor significa 'o maximo que este n permite "
      f"afirmar', nao 'efeito enorme'.")
    for met, note in notes.items():
        A(f"- **`{met}` e' degenerada:** {note} (S10). O Wilcoxon exato e' indefinido nesse "
          f"caso e a razao seria 1,000 por construcao, entao a metrica fica fora da figura. "
          f"Para `sat_frac` isto e' **declaracao de escopo** (regime nao saturado do "
          f"atuador), nao falha de medicao — ver 4.4.")
    A("")
    A("---")
    A("Ver tambem `EGAP_HOMONIMO.md` nesta pasta: existem **duas** definicoes de `egap_avg` "
      "no repositorio, com janelas e estimulos diferentes. Todos os numeros acima sao da "
      "definicao do `run_churn_sweep.py` (t >= 20 s, churn continuo).\n")
    A("Gerado por `analysis_churn/analyze_pi2.py`. Log completo: `LOG_execucao.txt`. "
      "Dados por par: `paired_ratios.csv`; por taxa: `summary_by_rate.csv`.")
    with open(os.path.join(HERE, "RESULTADO_PI2.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    say(f"  gravado: RESULTADO_PI2.md  (o RESULTADO.md consolidado e' curado a mao)")


def main():
    df, consts, digest, notes = load_and_check()
    unsat = inventory(df, consts, notes)
    summ, prs = analyse(df, consts)
    sh = shape(df, consts)
    tr = trend(summ, prs)
    prs.to_csv(os.path.join(HERE, "paired_ratios.csv"), index=False)
    summ.to_csv(os.path.join(HERE, "summary_by_rate.csv"), index=False)
    sh.to_csv(os.path.join(HERE, "shape_by_rate.csv"), index=False)
    figures(df, summ, sh, consts)
    report(df, summ, sh, tr, consts, digest, unsat, notes)
    say("")
    say("OK -- 10 sentinelas passaram.")
    flush_log()


if __name__ == "__main__":
    main()
