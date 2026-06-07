#!/usr/bin/env python
"""Diagnostico do bug repeats>=2-sob-perda.

Roda B2 (N=24, tau_a=1, perda 0.2, 1 seed) com repeats=1 e repeats=2, mantendo events.csv.
Cada evento completado registra N_new (=h_CCW+h_CW+1), que DEVERIA ser 23 (N=24 -> 23 apos a
falha) para TODOS os receptores. Se repeats=2 produz N_new != 23, o hop esta' sendo corrompido
(captacao de vizinho de hop errado, travada pelo dedup). Tambem reporta egap_final.

Uso: python experiments/scaling_law/diag_repeats.py
"""
import os
import sys
import subprocess

import numpy as np
import pandas as pd

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))
MAIN_PY = os.path.join(REPO_ROOT, "main.py")
RUNS = os.path.join(EXP_DIR, "diag_runs")

N = 24
TAU = 1.0
LOSS = float(os.environ.get("DIAG_LOSS", "0.2"))
SEED = int(os.environ.get("DIAG_SEED", "0"))
BUDGET = float(os.environ.get("DIAG_BUDGET", "80"))
T0 = 5.0


def run(repeats):
    run_dir = os.path.join(RUNS, f"r{repeats}")
    os.makedirs(run_dir, exist_ok=True)
    # garantir events.csv limpo (AgentProtocol so escreve header se nao existir)
    for fn in ("events.csv", "agent_telemetry.csv", "target_telemetry.csv"):
        p = os.path.join(run_dir, fn)
        if os.path.exists(p):
            os.remove(p)
    env = dict(os.environ)
    env.update({
        "PYTHONIOENCODING": "utf-8", "PROPAGATION_METHOD": "dual_pulse", "PROPAGATION_K_PROP": "0.0",
        "NUM_AGENTS": str(N), "SIM_DURATION": str(T0 + BUDGET), "K_E_TAU": f"{250.0 / N:.6f}",
        "VM_TAU_XY": str(TAU), "EXPERIMENT_SEED": str(SEED),
        "COMMUNICATION_FAILURE_RATE": f"{LOSS:g}", "COMMUNICATION_DELAY": "0",
        "INIT_ANGLES_EQUIDISTANT": "True", "INIT_RADIUS_RANGE": "0.0", "TARGET_MOTION_SPEED_XY": "0.0",
        "VIS_OPEN_BROWSER": "False", "SKIP_TELEMETRY_PLOTS": "True",
        "DETERMINISTIC_FAILURE_ENABLE": "True", "DETERMINISTIC_FAILURE_AGENT_ID": str(2 + N // 2),
        "DETERMINISTIC_FAILURE_TIME_T0": str(T0), "DETERMINISTIC_FAILURE_OFF_TIME": "-1.0",
        "RUNS_SUMMARY_CSV_PATH": os.path.join(run_dir, "runs_summary.csv"),
        "DUAL_PULSE_INTEGRATION": "B2", "DUAL_PULSE_DELTA_SCALE": "1.0",
        "DUAL_PULSE_T_FF": f"{TAU:.6f}", "DUAL_PULSE_TTL_HOPS": str(3 * N),
        "DUAL_PULSE_BROADCAST_REPEATS": str(repeats),
    })
    subprocess.run([sys.executable, MAIN_PY], cwd=run_dir, env=env, capture_output=True, text=True)

    ev_path = os.path.join(run_dir, "events.csv")
    tgt_path = os.path.join(run_dir, "target_telemetry.csv")
    egf = float("nan")
    if os.path.exists(tgt_path):
        t = pd.read_csv(tgt_path)
        if len(t):
            egf = float(t["E_gap"].iloc[-1])
    print(f"\n=== repeats={repeats}  (perda={LOSS}, seed={SEED}) ===")
    print(f"  egap_final = {egf:.4f}")
    if not os.path.exists(ev_path):
        print("  (sem events.csv)"); return
    ev = pd.read_csv(ev_path)
    comp = ev[ev["event_type"].astype(str).str.contains("dual_pulse", na=False)]
    recv = comp[comp["event_type"].astype(str).str.contains("completed", na=False)]
    if "N_new" in recv.columns and len(recv):
        vc = recv["N_new"].value_counts().sort_index()
        n_ok = int((recv["N_new"] == (N - 1)).sum())
        print(f"  eventos completados (receptor): {len(recv)}; N_new correto(=={N-1}): {n_ok}")
        print(f"  distribuicao N_new: {dict(vc)}")
    else:
        print(f"  eventos dual_pulse: {len(comp)} (sem coluna N_new util)")


def main():
    os.makedirs(RUNS, exist_ok=True)
    for r in (1, 2):
        run(r)
    print("\nLeitura: se repeats=2 tem N_new != 23 (e repeats=1 ~todos 23), o hop esta' corrompido.")


if __name__ == "__main__":
    main()
