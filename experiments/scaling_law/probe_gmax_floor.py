#!/usr/bin/env python
"""Sonda mecanistica: o VAO MAXIMO (G_max) tem um PISO independente de protocolo?

Contexto. A campanha de churn reportou egap_* -- que sao agregados temporais do RMS
ESPACIAL do erro de vao (E_gap). O vao angular maximo, que e' a quantidade
mission-critical da tese, e' G_max = max_k(gap_k / (2*pi/M)) com M = agentes VIVOS
(protocol_target.py:685) -- gravado em TODO target_telemetry.csv e NUNCA agregado.

Hipotese a testar (conjectura do orientando, NAO conclusao): logo apos uma falha o
vao maximo salta para ~2x o ideal e o tempo de fecha-lo e' limitado pela velocidade
maxima e por tau_a, nao pela coordenacao -> existiria um PISO em G_max independente
de protocolo. Predicao quantitativa: com uma morte num anel de M+1 vivos, o vao
fundido vale 2*(2pi/(M+1)) contra um ideal novo de 2pi/M, logo
    G_max_pico = 2*M/(M+1)      (= 1.92 para M=23, N nominal 24)
e esse pico NAO deveria diferir entre baseline e B2.

O que ESTE script pode e nao pode dizer. Ele le os target_telemetry.csv que
sobreviveram em churn_sweep_runs_stamp/ (24 rodadas, dt=0.01, N=24, tau=1.0,
off=8 s, taxas 6/12/24/48, seeds 0-2). ATENCAO: a metade BASELINE desses runs bate
exatamente com o baseline da familia de ablacoes dt=0.01
(churn_sweep_results_m8off_ablation8seed / _add_clean / _gate_clean); a metade B2
NAO bate exatamente com nenhum CSV commitado -- qual configuracao de overlay a
gerou e' irrecuperavel (nao havia proveniencia antes de P0). Portanto: evidencia
MECANISTICA sobre o piso, NAO evidencia de campanha sobre "B2 melhora G_max".

NAO roda simulacao.

Uso:
    python experiments/scaling_law/probe_gmax_floor.py
    # env: GMAX_RUNS="churn_sweep_runs_stamp"  GMAX_T0="20.0"
"""
import os
import sys

import numpy as np
import pandas as pd

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(EXP_DIR, os.environ.get("GMAX_RUNS", "churn_sweep_runs_stamp"))
OUT_CSV = os.path.join(EXP_DIR, os.environ.get("GMAX_OUT", "gmax_probe_results.csv"))
T0 = float(os.environ.get("GMAX_T0", "20.0"))   # mesma janela do run_churn_sweep (T0+WARMUP_AVG)

# Limiares de "brecha": fracao do tempo com o vao acima de X vezes o ideal.
BREACH = (1.25, 1.5, 1.75, 2.0)


def parse_dir(name):
    method = "B2" if name.startswith("dual_pulse") else "baseline"
    rate = float(name.split("_rate")[1].split("_s")[0])
    seed = int(name.split("_s")[-1])
    return method, rate, seed


def run_row(path, name):
    df = pd.read_csv(path, usecols=["timestamp", "E_gap", "G_max"])
    df = df[df["timestamp"] >= T0]
    if df.empty:
        return None
    g = df["G_max"].to_numpy(float)
    e = df["E_gap"].to_numpy(float)
    g = g[np.isfinite(g)]
    method, rate, seed = parse_dir(name)
    row = {
        "method": method, "rate_total": rate, "seed": seed,
        "gmax_avg": float(np.mean(g)), "gmax_p90": float(np.percentile(g, 90)),
        "gmax_p99": float(np.percentile(g, 99)), "gmax_max": float(np.max(g)),
        "egap_avg": float(np.mean(e[np.isfinite(e)])),
        "n_samples": int(g.size),
    }
    for thr in BREACH:
        row[f"frac_gt_{thr:g}"] = float(np.mean(g > thr))
    return row


def paired(df, metric):
    """(base, b2) casados por (rate_total, seed) -- ver analyze_churn_paired.py."""
    keys = ["rate_total", "seed"]
    b = df[df.method == "baseline"][keys + [metric]]
    o = df[df.method == "B2"][keys + [metric]]
    m = b.merge(o, on=keys, suffixes=("_b", "_o")).sort_values(keys)
    return m[f"{metric}_b"].to_numpy(float), m[f"{metric}_o"].to_numpy(float)


def report(df, metric, label, higher_is_worse=True):
    base, b2 = paired(df, metric)
    if base.size == 0:
        return
    ok = np.isfinite(base) & np.isfinite(b2) & (b2 != 0)
    ratio = np.where(ok, base / np.where(b2 == 0, np.nan, b2), np.nan)
    ratio = ratio[np.isfinite(ratio)]
    try:
        from scipy.stats import wilcoxon
        p = float(wilcoxon(base, b2).pvalue) if np.any(base != b2) else float("nan")
    except Exception:
        p = float("nan")
    n_lose = int(np.sum(ratio < 1.0)) if higher_is_worse else int(np.sum(ratio > 1.0))
    print(f"  {label:<26} base med={np.median(base):6.3f}  B2 med={np.median(b2):6.3f}  "
          f"razao med={np.median(ratio):5.2f} [{np.min(ratio):.2f}, {np.max(ratio):.2f}]  "
          f"n={len(base):2d}  n_lose={n_lose:2d}  p={p:.4f}")


def main():
    if not os.path.isdir(RUNS_DIR):
        sys.exit(f"Diretorio de rodadas ausente: {RUNS_DIR}\n"
                 "Este script so funciona onde o target_telemetry.csv sobreviveu.")
    rows = []
    for d in sorted(os.listdir(RUNS_DIR)):
        p = os.path.join(RUNS_DIR, d, "target_telemetry.csv")
        if os.path.exists(p):
            r = run_row(p, d)
            if r:
                rows.append(r)
    if not rows:
        sys.exit(f"Nenhum target_telemetry.csv em {RUNS_DIR}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    n_nom = 24
    predicted_peak = 2.0 * (n_nom - 1) / n_nom
    print(f"Fonte: {os.path.relpath(RUNS_DIR, EXP_DIR)}  ({len(df)} rodadas, "
          f"janela t >= {T0:g}s, {df.n_samples.iloc[0]} amostras/rodada)")
    print(f"Piso previsto p/ UMA morte em anel N={n_nom}: G_max = 2*(N-1)/N = "
          f"{predicted_peak:.3f}\n")

    print("=== G_max PAREADO (baseline x B2), agregado sobre taxas e seeds ===")
    for metric, label in [("gmax_avg", "G_max medio"), ("gmax_p90", "G_max P90"),
                          ("gmax_p99", "G_max P99"), ("gmax_max", "G_max MAXIMO")]:
        report(df, metric, label)
    print()
    report(df, "egap_avg", "E_gap medio (controle)")

    print("\n=== O PICO tem piso? G_max maximo observado por metodo e taxa ===")
    print(f"{'taxa':>6} {'metodo':<10}{'gmax_max med':>13}{'min':>8}{'max':>8}"
          f"{'razao/previsto':>16}")
    for rate in sorted(df.rate_total.unique()):
        for meth in ("baseline", "B2"):
            s = df[(df.rate_total == rate) & (df.method == meth)]["gmax_max"]
            if not len(s):
                continue
            print(f"{rate:>6g} {meth:<10}{s.median():>13.3f}{s.min():>8.3f}"
                  f"{s.max():>8.3f}{s.median() / predicted_peak:>16.2f}")

    print("\n=== JANELA DE BRECHA: fracao do tempo com vao acima de X x o ideal ===")
    print(f"{'taxa':>6} {'metodo':<10}" + "".join(f"{'>' + f'{t:g}':>10}" for t in BREACH))
    for rate in sorted(df.rate_total.unique()):
        for meth in ("baseline", "B2"):
            s = df[(df.rate_total == rate) & (df.method == meth)]
            if not len(s):
                continue
            cells = "".join(f"{s[f'frac_gt_{t:g}'].median():>10.4f}" for t in BREACH)
            print(f"{rate:>6g} {meth:<10}{cells}")
    print("\n  (mediana entre seeds; 'fracao do tempo' e' a leitura mission-critical:")
    print("   quanto tempo existe uma brecha maior que X vezes o vao ideal.)")

    print("\n=== BRECHA pareada: razao base/B2 da fracao de tempo acima do limiar ===")
    for thr in BREACH:
        report(df, f"frac_gt_{thr:g}", f"frac tempo > {thr:g}x ideal")

    print(f"\nEscrito: {OUT_CSV}")


if __name__ == "__main__":
    main()
