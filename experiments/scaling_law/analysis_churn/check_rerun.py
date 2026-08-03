#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DECISAO 1.2 -- SENTINELA DE REPRODUTIBILIDADE do re-run contra a campanha c3.

Compara egap_avg / egap_p90 / egap_max celula a celula (metodo, taxa, semente) entre
    experiments/scaling_law/churn_sweep_results_c3_churn8_dt05.csv   (fonte canonica)
e
    analysis_churn/rerun_c3_results.csv                              (re-run desta sessao)
com rtol=1e-9.

Regra de decisao (fixada pelo usuario ANTES de rodar, e nao negociavel depois):
  * reproduz em TODAS as celulas comuns -> exit 0, e a saida vale como prova de
    reprodutibilidade.
  * discorda em QUALQUER celula -> exit 1. NAO gerar figura de G_max. O passo
    seguinte e' bissectar o historico do git entre 2026-07-25 e hoje para achar o
    commit que mudou o resultado -- isso contamina toda comparacao que atravesse
    essa data, e importa mais que a figura.

Uso:
    python experiments/scaling_law/analysis_churn/check_rerun.py
    # (pode ser rodado com o re-run ainda em andamento: compara so as celulas ja
    #  presentes e diz quantas faltam)
"""
import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(HERE)
SRC = os.path.join(EXP_DIR, "churn_sweep_results_c3_churn8_dt05.csv")
RERUN = os.path.join(HERE, "rerun_c3_results.csv")
KEYS = ["method", "rate_total", "seed"]
COLS = ["egap_avg", "egap_p90", "egap_max"]
RTOL = 1e-9


def main():
    if not os.path.exists(SRC):
        sys.exit(f"ABORTA: fonte canonica ausente: {SRC}")
    if not os.path.exists(RERUN):
        sys.exit(f"ABORTA: re-run ainda nao produziu {RERUN}")
    a = pd.read_csv(SRC)
    b = pd.read_csv(RERUN)
    m = a.merge(b, on=KEYS, suffixes=("_orig", "_new"))

    print(f"original : {os.path.basename(SRC)}  ({len(a)} linhas)")
    print(f"re-run   : {os.path.basename(RERUN)} ({len(b)} linhas)")
    print(f"celulas comparaveis: {len(m)} de 64  (faltam {64 - len(m)})")
    print(f"criterio : rtol={RTOL:g} em {COLS}\n")

    print(f"{'metodo':<9}{'taxa':>6}{'seed':>5}  " +
          "".join(f"{c:>26}" for c in COLS))
    bad = []
    for _, r in m.sort_values(KEYS).iterrows():
        cells = []
        for c in COLS:
            o, n = float(r[f"{c}_orig"]), float(r[f"{c}_new"])
            ok = np.isclose(o, n, rtol=RTOL, atol=0.0)
            if not ok:
                bad.append((r["method"], r["rate_total"], r["seed"], c, o, n))
            rel = abs(n - o) / abs(o) if o != 0 else float("inf")
            cells.append(f"{'=' if ok else 'X'} {n:.12g} ({rel:.1e})".rjust(26))
        print(f"{r['method']:<9}{r['rate_total']:>6g}{int(r['seed']):>5}  " + "".join(cells))

    print()
    if bad:
        print("=" * 74)
        print(f"SENTINELA DE REPRODUTIBILIDADE: FALHOU em {len(bad)} valor(es)")
        print("=" * 74)
        for meth, rate, seed, col, o, n in bad:
            print(f"  {meth} rate={rate:g} seed={int(seed)} {col}: "
                  f"original={o!r} re-run={n!r} (dif rel {abs(n-o)/abs(o):.3e})")
        print("\nNAO gerar figura de G_max. Proximo passo obrigatorio: bissectar o git")
        print("entre 2026-07-25 e hoje e identificar o commit que mudou o resultado.")
        sys.exit(1)

    print("=" * 74)
    print(f"SENTINELA DE REPRODUTIBILIDADE: PASSOU em {len(m)} celula(s) x {len(COLS)} "
          f"metricas = {len(m)*len(COLS)} valores, rtol={RTOL:g}")
    print("=" * 74)
    if len(m) < 64:
        print(f"PARCIAL: {64 - len(m)} celulas ainda nao rodaram. Re-executar ao final.")
        sys.exit(2)
    print("Prova de reprodutibilidade: a campanha c3_churn8_dt05 e' bit-reprodutivel")
    print("no commit atual. Nenhuma mudanca de codigo entre a campanha e hoje alterou")
    print("egap_avg/p90/max.")


if __name__ == "__main__":
    main()
