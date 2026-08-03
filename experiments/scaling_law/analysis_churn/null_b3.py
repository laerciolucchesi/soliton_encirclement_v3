#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Item 3.3 -- NULO SINTETICO para b3: e' especificacao ou e' do sistema?

b3 (interacao d_gmax_pre x pre_bar) sobreviveu a quatro testes de escala, mas todos
assumem forma aditiva ou log-aditiva. Este script constroi um mundo onde, POR
CONSTRUCAO, nao existe efeito de metodo nenhum alem da geometria: o pico e' funcao
DETERMINISTICA da configuracao sorteada do anel.

  * sorteia M vaos somando 2*pi (Dirichlet; alpha controla a dispersao)
  * o braco "baseline" recebe anel MAIS disperso; o "B2", MAIS uniforme
    (e' a unica diferenca entre bracos -- nao ha efeito sobre o PICO dado o anel)
  * mata um agente UNIFORMEMENTE ao acaso, o MESMO indice nos dois bracos
  * pico = max sobre os vaos pos-fusao, normalizado por 2*pi/(M-1)
  * roda o PIPELINE DE REGRESSAO IDENTICO ao de aggregate_gmax.py

Leitura:
  b3 aparece no nulo  -> e' ARTEFATO DE ESPECIFICACAO; sai do relatorio.
  b3 nao aparece      -> e' do sistema; volta a ser candidato a achado.

Uso:
    python experiments/scaling_law/analysis_churn/null_b3.py
    # env: NULL_REPS="200"  NULL_SEED="0"
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPS = int(os.environ.get("NULL_REPS", "200"))
SEED = int(os.environ.get("NULL_SEED", "0"))
N_RUNS = 24          # "rodadas" sinteticas = clusters, como no dado real
N_EV = 56            # eventos por rodada -> ~1345 pares, como no dado real
_log = []


def say(m=""):
    print(m)
    _log.append(str(m))


def ols_cluster(y, X, groups, names):
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    gs = np.unique(groups)
    meat = np.zeros((k, k))
    for gg in gs:
        m = groups == gg
        s = X[m].T @ u[m]
        meat += np.outer(s, s)
    G = gs.size
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1.0)) * ((n - 1.0) / (n - k))
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    tc = stats.t.ppf(0.975, G - 1)
    return pd.DataFrame({"termo": names, "coef": beta, "se": se,
                         "ic95_lo": beta - tc * se, "ic95_hi": beta + tc * se,
                         "p": 2 * (1 - stats.t.cdf(np.abs(beta / se), G - 1))})


def draw_world(rng, alpha_base, alpha_b2):
    """Um conjunto sintetico completo: eventos pareados, sem efeito de metodo."""
    rows = []
    for run in range(N_RUNS):
        for _ in range(N_EV):
            M = int(rng.integers(18, 25))
            # dois aneis independentes; a UNICA diferenca e' a dispersao
            gA = rng.dirichlet(np.full(M, alpha_base)) * 2.0 * np.pi
            gB = rng.dirichlet(np.full(M, alpha_b2)) * 2.0 * np.pi
            i = int(rng.integers(0, M))          # MESMA vitima nos dois bracos
            out = {}
            for tag, g in (("base", gA), ("b2", gB)):
                ideal = 2.0 * np.pi / M
                pre = g.max() / ideal
                egap_pre = float(np.sqrt(np.mean((g / ideal - 1.0) ** 2)))
                merged = np.concatenate([np.delete(g, [i, (i - 1) % M]),
                                         [g[i] + g[(i - 1) % M]]])
                ideal_new = 2.0 * np.pi / (M - 1)
                out[tag] = (merged.max() / ideal_new, pre, egap_pre, merged.max())
            rows.append({
                "run": run, "M": M,
                "d_pico": out["base"][0] - out["b2"][0],
                "d_gmax_pre": out["base"][1] - out["b2"][1],
                "d_egap_pre": out["base"][2] - out["b2"][2],
                "pre_bar": 0.5 * (out["base"][1] + out["b2"][1]),
                "pico_base": out["base"][0], "pico_b2": out["b2"][0],
                "pre_base": out["base"][1], "pre_b2": out["b2"][1],
                # vao em METROS no pico (R = 20 m), para recalibrar d_arco
                "d_arco": (out["base"][3] - out["b2"][3]) * 20.0,
            })
    return pd.DataFrame(rows)


def run_pipeline(d):
    """O MESMO pipeline de aggregate_gmax.py: aditivo/log, centrado/nao centrado.

    Devolve tambem alpha_total, alpha_direto e d_arco -- a nula recalibra TODO p que
    este pipeline produziu, nao so o de b3.
    """
    out = {}
    grp = d["run"].to_numpy()
    ones = np.ones(len(d))
    # TOTAL: d_pico ~ 1
    out["alpha_total"] = ols_cluster(d.d_pico.to_numpy(float), ones.reshape(-1, 1),
                                     grp, ["alpha_tot"])
    # DIRETO: d_pico ~ 1 + d_gmax_pre + d_egap_pre
    Xd = np.column_stack([ones, d.d_gmax_pre, d.d_egap_pre])
    out["alpha_direto"] = ols_cluster(d.d_pico.to_numpy(float), Xd, grp,
                                      ["alpha", "b1", "b2"])
    # d_arco: diferenca do vao de fusao em METROS
    out["d_arco"] = ols_cluster(d.d_arco.to_numpy(float), ones.reshape(-1, 1),
                                grp, ["d_arco"])
    # aditivo, nao centrado
    X = np.column_stack([ones, d.d_gmax_pre, d.d_egap_pre, d.d_gmax_pre * d.pre_bar])
    t = ols_cluster(d.d_pico.to_numpy(float), X, grp,
                    ["alpha", "b1", "b2", "b3"])
    out["add"] = t
    # aditivo, centrado
    pc = d.pre_bar - d.pre_bar.mean()
    Xc = np.column_stack([ones, d.d_gmax_pre, d.d_egap_pre, d.d_gmax_pre * pc])
    out["add_c"] = ols_cluster(d.d_pico.to_numpy(float), Xc, grp,
                               ["alpha", "b1", "b2", "b3"])
    # log
    dl_pico = np.log(d.pico_base) - np.log(d.pico_b2)
    dl_pre = np.log(d.pre_base) - np.log(d.pre_b2)
    lbar = np.log(d.pre_bar)
    Xl = np.column_stack([ones, dl_pre, d.d_egap_pre, dl_pre * lbar])
    out["log"] = ols_cluster(dl_pico.to_numpy(float), Xl, grp,
                             ["alpha", "b1", "b2", "b3"])
    return out


def main():
    rng = np.random.default_rng(SEED)
    say("=" * 78)
    say("NULO SINTETICO para b3 -- mundo SEM efeito de metodo, por construcao")
    say("=" * 78)
    say("Anel: M vaos ~ Dirichlet(alpha) somando 2*pi. baseline recebe anel MAIS")
    say("disperso (alpha menor); B2, mais uniforme. A vitima e' a MESMA nos dois bracos.")
    say("O pico e' funcao DETERMINISTICA do anel sorteado: nao ha efeito de metodo.")
    say(f"REPS={REPS}  runs/rep={N_RUNS}  eventos/run={N_EV}  seed={SEED}")

    # calibra a dispersao para reproduzir o gmax_pre observado (~1.25 mediano)
    ref = os.path.join(HERE, "gmax_events.csv")
    if os.path.exists(ref):
        ev = pd.read_csv(ref)
        tgt_b = float(ev[ev.method == "baseline"].gmax_pre.median())
        tgt_o = float(ev[ev.method == "B2"].gmax_pre.median())
        say(f"alvo de calibracao (dado real): gmax_pre mediano baseline={tgt_b:.4f} "
            f"B2={tgt_o:.4f}")
    else:
        tgt_b = tgt_o = None
    A_BASE, A_B2 = 12.0, 20.0     # alpha maior = anel mais uniforme
    cal = draw_world(np.random.default_rng(999), A_BASE, A_B2)
    say(f"calibracao sintetica: gmax_pre mediano baseline="
        f"{cal.pre_base.median():.4f} B2={cal.pre_b2.median():.4f} "
        f"(alpha {A_BASE:g} vs {A_B2:g})")

    say("")
    say("-- UMA replica, tabela completa (mesmo formato do dado real) --")
    d0 = draw_world(np.random.default_rng(SEED), A_BASE, A_B2)
    r0 = run_pipeline(d0)
    for lab, t in (("ADITIVO nao centrado", r0["add"]),
                   ("ADITIVO centrado", r0["add_c"]),
                   ("EM LOG nao centrado", r0["log"])):
        say(f"   {lab}:")
        for _, r in t.iterrows():
            say(f"      {r['termo']:<7}{r['coef']:>11.5f}  IC95 [{r['ic95_lo']:+.5f}, "
                f"{r['ic95_hi']:+.5f}]  p={r['p']:.5f}")

    say("")
    say(f"-- {REPS} replicas: com que frequencia b3 sai 'significativo' no NULO? --")
    res = {"add": [], "add_c": [], "log": []}
    for rep in range(REPS):
        d = draw_world(np.random.default_rng(SEED + 1000 + rep), A_BASE, A_B2)
        rr = run_pipeline(d)
        for k in res:
            b3 = rr[k][rr[k].termo == "b3"].iloc[0]
            res[k].append({"coef": b3["coef"], "p": b3["p"],
                           "sig": not (b3["ic95_lo"] <= 0.0 <= b3["ic95_hi"])})
    say(f"{'modelo':<22}{'b3 mediano':>12}{'b3 p05':>10}{'b3 p95':>10}"
        f"{'frac IC exclui 0':>18}")
    for k, lab in (("add", "ADITIVO nao centr."), ("add_c", "ADITIVO centrado"),
                   ("log", "EM LOG")):
        df = pd.DataFrame(res[k])
        say(f"{lab:<22}{df.coef.median():>12.5f}{df.coef.quantile(.05):>10.5f}"
            f"{df.coef.quantile(.95):>10.5f}{df.sig.mean():>18.3f}")

    say("")
    say("-- comparacao com o dado REAL --")
    real = {"add": 1.11016, "add_c": 1.11016, "log": 1.60965}
    say(f"{'modelo':<22}{'b3 REAL':>10}{'percentil no NULO':>20}  veredito")
    for k, lab in (("add", "ADITIVO nao centr."), ("add_c", "ADITIVO centrado"),
                   ("log", "EM LOG")):
        df = pd.DataFrame(res[k])
        pct = float((df.coef < real[k]).mean())
        frac_sig = float(df.sig.mean())
        if frac_sig > 0.20:
            v = "b3 SAI NO NULO -> ARTEFATO DE ESPECIFICACAO"
        elif pct > 0.95:
            v = "b3 real FORA da distribuicao nula -> do SISTEMA"
        else:
            v = "b3 real DENTRO da nula -> nao distinguivel de artefato"
        say(f"{lab:<22}{real[k]:>10.4f}{pct:>20.3f}  {v}")

    say("")
    say("LEITURA: 'frac IC exclui 0' e' a taxa de falso positivo do proprio pipeline num")
    say("mundo sem efeito. Se for muito acima de 0,05, a especificacao fabrica interacao")
    say("sozinha e b3 nao pode ser reportado como achado em nenhuma direcao.")

    # ------------------------------------------------------------------ item 2
    # A nula recalibra TODO p que este pipeline produziu, nao so o de b3.
    say("")
    say("=" * 78)
    say("RECALIBRACAO DOS OUTROS COEFICIENTES CONTRA A MESMA NULA")
    say("=" * 78)
    say("Neste mundo nulo o metodo NAO afeta o pico dado o anel -- mas AFETA o anel")
    say("(dispersao diferente entre bracos, calibrada no dado real). Entao a nula mede")
    say("exatamente quanto de efeito TOTAL surge so da geometria, sem fisica adicional.")
    say("O p honesto e' o PERCENTIL do valor real na distribuicao nula, nao o p nominal.")
    acc = {"alpha_total": [], "alpha_direto": [], "d_arco": [], "b3_add": []}
    for rep in range(REPS):
        d = draw_world(np.random.default_rng(SEED + 5000 + rep), A_BASE, A_B2)
        rr = run_pipeline(d)
        acc["alpha_total"].append(rr["alpha_total"].iloc[0])
        acc["alpha_direto"].append(rr["alpha_direto"].iloc[0])
        acc["d_arco"].append(rr["d_arco"].iloc[0])
        acc["b3_add"].append(rr["add"][rr["add"].termo == "b3"].iloc[0])
    REAL = {"alpha_total": (0.02469, 0.064), "alpha_direto": (-0.00160, 0.922),
            "d_arco": (0.17828, 0.040), "b3_add": (1.11016, 0.00006)}
    say("")
    say(f"{'coeficiente':<16}{'REAL':>10}{'p nominal':>11}{'nula: mediana':>15}"
        f"{'nula: p05':>11}{'nula: p95':>11}{'frac rejeita':>14}{'percentil real':>16}")
    verd = {}
    for k in ("alpha_direto", "alpha_total", "d_arco", "b3_add"):
        df = pd.DataFrame(acc[k])
        real, pnom = REAL[k]
        pct = float((df.coef < real).mean())
        frac = float((~((df.ic95_lo <= 0) & (0 <= df.ic95_hi))).mean())
        verd[k] = (pct, frac)
        say(f"{k:<16}{real:>10.5f}{pnom:>11.3f}{df.coef.median():>15.5f}"
            f"{df.coef.quantile(.05):>11.5f}{df.coef.quantile(.95):>11.5f}"
            f"{frac:>14.3f}{pct:>16.3f}")
    say("")
    say("'percentil real' = fracao das replicas nulas com coeficiente MENOR que o real.")
    say("Perto de 0,50 => o valor real e' tipico do mundo SEM efeito. Acima de 0,95 =>")
    say("fora da nula, e ai sim ha efeito alem da geometria.")
    say("")
    for k, lab in (("alpha_direto", "alpha DIRETO"), ("alpha_total", "alpha TOTAL"),
                   ("d_arco", "d_arco [m]"), ("b3_add", "b3")):
        pct, frac = verd[k]
        if frac > 0.20:
            say(f"  {lab:<14} pipeline rejeita em {frac:.1%} das replicas NULAS -- p nominal "
                f"inflado ~{frac/0.05:.0f}x.")
        if pct > 0.95:
            say(f"  {lab:<14} percentil {pct:.3f}: FORA da nula -> efeito alem da geometria.")
        elif pct < 0.05:
            say(f"  {lab:<14} percentil {pct:.3f}: FORA da nula pelo lado negativo.")
        else:
            say(f"  {lab:<14} percentil {pct:.3f}: DENTRO da nula -> nao distinguivel de um")
            say(f"  {'':14} mundo em que o metodo nao afeta o pico dado o anel.")
    say("")
    say("NOTA sobre alpha_direto: um pipeline que rejeita DEMAIS e que, mesmo assim, NAO")
    say("rejeitou (p=0,922) e' evidencia de nulo MAIS forte, nao menos.")
    with open(os.path.join(HERE, "LOG_null_b3.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(_log) + "\n")
    say("\nEscrito: LOG_null_b3.txt")


if __name__ == "__main__":
    main()
