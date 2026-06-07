#!/usr/bin/env python
"""Fase 3 / Track B -- diagnostico do OUTLIER de churn (B2 muito pior que baseline).

Caso: rate=12/min total (0.5/agente), seed=0, off=8s, N=24, loss=0. B2 deu egap_avg~0.51 vs
baseline ~0.10 no MESMO churn. Investiga o mecanismo do dano ativo sob churn continuo.

Roda B2 (mantendo agent_telemetry: dual_pulse_shift, e_tau virtual, e_tau_real fisico) e baseline
(target_telemetry). Analisa:
  1) egap(t) por bin: quando o B2 diverge do baseline.
  2) eventos B2: contagem por tipo, N_new dist (corrupcao?), entrada vs recuperacoes.
  3) TESTE DECISIVO: por bin, mean|e_tau| (virtual) vs mean|e_tau_real| (fisico) vs
     mean|dual_pulse_shift|. Se virtual<<real com shift alto => overlay engana o controlador
     com vies errado preso (shift_remaining travado) = dano ativo.

Uso: python experiments/scaling_law/diag_outlier.py
"""
import os
import sys
import subprocess

import numpy as np
import pandas as pd

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))
MAIN_PY = os.path.join(REPO_ROOT, "main.py")
RUNS = os.path.join(EXP_DIR, "outlier_runs")

N = 24
TAU = 1.0
T0 = 5.0
BUDGET = 150.0
RATE_TOTAL = float(os.environ.get("OUT_RATE", "12"))
SEED = int(os.environ.get("OUT_SEED", "0"))
OFF = float(os.environ.get("OUT_OFF", "8.0"))
BIN = 15.0  # s


def run(method, keep_agent):
    run_dir = os.path.join(RUNS, method)
    os.makedirs(run_dir, exist_ok=True)
    for fn in ("events.csv", "agent_telemetry.csv", "target_telemetry.csv"):
        p = os.path.join(run_dir, fn)
        if os.path.exists(p):
            os.remove(p)
    env = dict(os.environ)
    env.update({
        "PYTHONIOENCODING": "utf-8", "PROPAGATION_METHOD": method, "PROPAGATION_K_PROP": "0.0",
        "NUM_AGENTS": str(N), "SIM_DURATION": str(T0 + BUDGET), "K_E_TAU": f"{250.0 / N:.6f}",
        "VM_TAU_XY": str(TAU), "COMMUNICATION_FAILURE_RATE": "0", "COMMUNICATION_DELAY": "0",
        "INIT_ANGLES_EQUIDISTANT": "True", "INIT_RADIUS_RANGE": "0.0", "TARGET_MOTION_SPEED_XY": "0.0",
        "VIS_OPEN_BROWSER": "False", "SKIP_TELEMETRY_PLOTS": "True",
        "DETERMINISTIC_FAILURE_ENABLE": "False", "FAILURE_ENABLE": "True",
        "FAILURE_MEAN_FAILURES_PER_MIN": f"{RATE_TOTAL / N:.6f}", "FAILURE_OFF_TIME": f"{OFF:g}",
        "EXPERIMENT_SEED": str(SEED),
        "RUNS_SUMMARY_CSV_PATH": os.path.join(run_dir, "runs_summary.csv"),
    })
    if method == "dual_pulse":
        env.update({"DUAL_PULSE_INTEGRATION": "B2", "DUAL_PULSE_DELTA_SCALE": "1.0",
                    "DUAL_PULSE_T_FF": f"{TAU:.6f}", "DUAL_PULSE_TTL_HOPS": str(3 * N)})
    print(f"  rodando {method} (rate={RATE_TOTAL}/min, seed={SEED}) ...", flush=True)
    subprocess.run([sys.executable, MAIN_PY], cwd=run_dir, env=env, capture_output=True, text=True)
    if not keep_agent:
        p = os.path.join(run_dir, "agent_telemetry.csv")
        if os.path.exists(p):
            os.remove(p)
    return run_dir


def egap_bins(tgt):
    df = pd.read_csv(tgt)
    df = df[df["timestamp"] >= T0]
    bins = {}
    edges = np.arange(T0, T0 + BUDGET + BIN, BIN)
    for lo in edges[:-1]:
        seg = df[(df["timestamp"] >= lo) & (df["timestamp"] < lo + BIN)]["E_gap"]
        bins[lo] = float(seg.mean()) if len(seg) else float("nan")
    return bins


def main():
    os.makedirs(RUNS, exist_ok=True)
    b2_dir = run("dual_pulse", keep_agent=True)
    base_dir = run("baseline", keep_agent=False)

    print("\n=== 1) egap medio por bin (15s): baseline vs B2 -> divergencia ===")
    eb = egap_bins(os.path.join(base_dir, "target_telemetry.csv"))
    eo = egap_bins(os.path.join(b2_dir, "target_telemetry.csv"))
    print(f"{'t_ini':>6} {'egap_base':>10} {'egap_B2':>9}")
    for lo in eb:
        print(f"{lo:>6.0f} {eb[lo]:>10.4f} {eo.get(lo, float('nan')):>9.4f}")

    print("\n=== 2) eventos B2 ===")
    ev = pd.read_csv(os.path.join(b2_dir, "events.csv"))
    et = ev["event_type"].astype(str)
    print("  por tipo:")
    for k, v in et.value_counts().items():
        print(f"    {k:38s} {v}")
    comp = ev[et.str.contains("completed", na=False)]
    if "N_new" in comp.columns and len(comp):
        print(f"  N_new dist (completed): {dict(comp['N_new'].value_counts().sort_index())}")

    print("\n=== 3) TESTE DECISIVO: virtual (e_tau) vs fisico (e_tau_real) vs shift, por bin ===")
    at = pd.read_csv(os.path.join(b2_dir, "agent_telemetry.csv"))
    col_real = "e_tau_real" if "e_tau_real" in at.columns else "e_tau"
    at = at[at["timestamp"] >= T0]
    print(f"{'t_ini':>6} {'|e_tau|virt':>11} {'|e_real|fis':>11} {'|shift|':>9} {'alive~':>7}")
    edges = np.arange(T0, T0 + BUDGET + BIN, BIN)
    for lo in edges[:-1]:
        seg = at[(at["timestamp"] >= lo) & (at["timestamp"] < lo + BIN)]
        if not len(seg):
            continue
        vt = float(seg["e_tau"].abs().mean())
        rl = float(seg[col_real].abs().mean())
        sh = float(seg["dual_pulse_shift"].abs().mean()) if "dual_pulse_shift" in seg.columns else float("nan")
        # n. de node_ids distintos com linha nesse bin ~ agentes "vivos" reportando
        alive = int(seg["node_id"].nunique()) if "node_id" in seg.columns else -1
        print(f"{lo:>6.0f} {vt:>11.4f} {rl:>11.4f} {sh:>9.4f} {alive:>7d}")

    # limpa agent_telemetry (grande)
    p = os.path.join(b2_dir, "agent_telemetry.csv")
    if os.path.exists(p):
        os.remove(p)
    print("\nLeitura: se em bins de egap alto o |e_tau|virt << |e_real|fis com |shift| alto =>")
    print("overlay engana o controlador com vies preso (shift_remaining travado) = dano ativo.")


if __name__ == "__main__":
    main()
