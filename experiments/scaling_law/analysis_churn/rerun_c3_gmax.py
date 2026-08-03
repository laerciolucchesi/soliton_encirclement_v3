#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DECISAO 1.1 -- re-roda as 64 celulas da campanha c3_churn8_dt05 PRESERVANDO a telemetria.

Por que existe: o target_telemetry.csv das 64 rodadas originais nao foi preservado
(nao existe churn_sweep_runs_c3_churn8_dt05/; o unico diretorio de churn sobrevivente
e' churn_sweep_runs_stamp/, que e' outra campanha -- dt=0.01, 3 seeds, metade B2 nao
atribuivel). Sem telemetria nao ha G_max, nem gap_max_rad, nem alive_count.

O QUE ESTE SCRIPT NAO FAZ: nao reimplementa o runner. Importa
experiments/scaling_law/run_churn_sweep.py e apenas REDIRECIONA suas duas globais de
saida (RUNS_DIR, RESULTS_CSV) para dentro de analysis_churn/ (R1: nada novo fora
desta pasta; regra 2 da campanha: nunca sobrescrever um CSV de resultado).

REPRODUCAO EXATA. O runner monta o env do filho com `env = dict(os.environ)` e depois
fixa os parametros da celula (run_churn_sweep.py:67-82). Tudo que ele NAO fixa vaza do
processo pai. Entao, antes de importar, este script REMOVE do os.environ toda variavel
de override reconhecida por config_param.py e main.py -- assim o filho ve exatamente
{o que o runner fixa} + {defaults do config}, que e' a definicao operacional de "mesma
configuracao". As CHURN_* sao re-postas explicitamente com os valores da campanha c3
(CAMPAIGN_LOG.md:406-409: dt=0.05, M8 on, M-mult on, taxas 6/12/24/48, seeds 0-7).

Uso:
    python experiments/scaling_law/analysis_churn/rerun_c3_gmax.py
    # env opcional (para o smoke-test de 2 celulas antes das 64):
    #   RERUN_RATES="6"  RERUN_SEEDS="0"
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(EXP_DIR))

# Parametros da campanha c3 (identicos aos defaults de run_churn_sweep.py; explicitos
# aqui porque "default" nao e' registro -- regra 3: fixar toda variavel).
C3 = {
    "CHURN_RATES": os.environ.get("RERUN_RATES", "6,12,24,48"),
    "CHURN_OFF": "8.0",
    "CHURN_SEEDS": os.environ.get("RERUN_SEEDS", "0,1,2,3,4,5,6,7"),
    "CHURN_N": "24",
    "CHURN_TAU": "1.0",
    "CHURN_BUDGET": "150",
    "CHURN_METHODS": "baseline,dual_pulse",
}
DT = "0.05"   # run_churn_sweep.py:44 le CONTROL_PERIOD; a c3 e' a campanha "_dt05"


def _override_keys():
    """Toda variavel de ambiente que config_param.py ou main.py consultam.

    Descoberta por varredura do proprio codigo em vez de lista fixa: uma lista
    escrita a mao envelhece silenciosamente quando alguem adiciona um override novo,
    e o efeito seria uma rodada com um parametro vazado do shell -- exatamente o erro
    que produz um numero plausivel e errado.
    """
    keys = set()
    pat = re.compile(r"environ(?:\.get)?[\(\[]\s*[\"']([A-Z0-9_]+)[\"']")
    for fn in ("config_param.py", "main.py", "provenance.py"):
        p = os.path.join(REPO, fn)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8", errors="replace") as fh:
                keys.update(pat.findall(fh.read()))
    return keys


def sanitize():
    removed = []
    for k in sorted(_override_keys()):
        if k in os.environ:
            removed.append(f"{k}={os.environ[k]}")
            del os.environ[k]
    for k in [k for k in os.environ if k.startswith(("CHURN_", "PROPAGATION_", "DUAL_PULSE_"))]:
        removed.append(f"{k}={os.environ[k]}")
        del os.environ[k]
    return removed


def main():
    removed = sanitize()
    os.environ.update(C3)
    os.environ["CONTROL_PERIOD"] = DT        # lido pelo runner (DT) e re-fixado no filho
    os.environ["PYTHONIOENCODING"] = "utf-8"

    sys.path.insert(0, EXP_DIR)
    import run_churn_sweep as rcs           # noqa: E402  (env tem de estar pronto antes)

    # Redireciona as saidas para analysis_churn/. Sao globais de modulo, lidas por
    # run_cell()/main() -- por isso o patch funciona sem tocar no arquivo original.
    rcs.RUNS_DIR = os.path.join(HERE, "rerun_runs")
    rcs.RESULTS_CSV = os.path.join(HERE, "rerun_c3_results.csv")

    print("=" * 78)
    print("RE-RUN c3_churn8_dt05 -- telemetria PRESERVADA")
    print("=" * 78)
    if removed:
        print(f"env do pai higienizado ({len(removed)} vars removidas para nao vazarem ao filho):")
        for r in removed:
            print(f"   - {r}")
    else:
        print("env do pai ja estava limpo (nenhuma var de override presente)")
    print(f"RATES={rcs.RATES_TOTAL} OFF={rcs.OFF} SEEDS={rcs.SEEDS} N={rcs.N} "
          f"TAU={rcs.TAU} BUDGET={rcs.BUDGET} DT={rcs.DT} METHODS={rcs.METHODS}")
    print(f"RUNS_DIR    = {rcs.RUNS_DIR}")
    print(f"RESULTS_CSV = {rcs.RESULTS_CSV}")
    print(f"celulas     = {len(rcs.RATES_TOTAL) * len(rcs.SEEDS) * len(rcs.METHODS)}")
    print("=" * 78, flush=True)

    if os.path.exists(rcs.RESULTS_CSV):
        # Regra 2: nunca sobrescrever resultado. O runner faz merge incremental, o que
        # e' desejado para retomar, mas um CSV pre-existente de OUTRA configuracao
        # contaminaria o merge sem aviso.
        print(f"AVISO: {rcs.RESULTS_CSV} ja existe -- o runner fara MERGE incremental "
              f"(celulas ja presentes nao sao re-rodadas).", flush=True)

    rcs.main()


if __name__ == "__main__":
    main()
