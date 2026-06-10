#!/usr/bin/env python
"""Fase 3 / Track B -- diagnostico de churn (ISOLADO: loss=0, delay=0). Medir antes de mexer.

Cenarios determinísticos (vitimas falham juntas em t0, falha PERMANENTE -> alvo = uniforme sobre
N-k sobreviventes):
  - k1        : 1 vitima (sanity; esperado N_new=N-1, overlay rapido)
  - adj2/adj3 : k vizinhos ADJACENTES -> previsao: SUB-correcao (so 1 evento dispara; 2o
                originador morto) -> tau MAIOR (baseline faz mais)
  - non2/non3 : drones NAO-vizinhos -> previsao: delta_D ADITIVO (~correto, leve sobre-correcao
                ~1/N) -> tau ~rapido
+ Poisson denso (descritivo): event accounting + N_new + egap.

Discriminante de sub/sobre-correcao = tau_fit (a loss=0 e' valido): overlay eficaz -> tau rapido;
sub-correcao -> baseline assume -> tau lento. N_new checa a corretude do tamanho do anel; egap_final
checa a rede de seguranca (baseline fecha?). Uso: python experiments/scaling_law/diag_churn.py
"""
import os
import sys
import subprocess

import numpy as np
import pandas as pd

from metrics_util import event_metrics   # helper canonico (t_settle + tau_fit + egap)

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))
MAIN_PY = os.path.join(REPO_ROOT, "main.py")
_TAG = os.environ.get("CHURN_TAG", "")
RUNS = os.path.join(EXP_DIR, "churn_runs" + (("_" + _TAG) if _TAG else ""))

N = int(os.environ.get("CHURN_N", "24"))
TAU = 1.0
T0 = 5.0
BUDGET = float(os.environ.get("CHURN_BUDGET", "100"))

# cenarios determinísticos: (nome, lista de ids, k)
DET_SCENARIOS = [
    ("k1",   "14",        1),
    ("adj2", "13,14",     2),
    ("adj3", "13,14,15",  3),
    ("non2", "8,20",      2),
    ("non3", "6,14,22",   3),
]


def metrics_from_tgt(tgt):
    if not os.path.exists(tgt):
        return {}
    return event_metrics(pd.read_csv(tgt), T0)


def parse_events(ev_path):
    if not os.path.exists(ev_path):
        return {}
    ev = pd.read_csv(ev_path)
    et = ev["event_type"].astype(str)
    out = {
        "failures": int(et.str.contains("failure_start").sum()),
        "recoveries": int(et.str.contains("failure_end").sum()),
        "injections": int(et.str.contains("self_shift").sum()),
        "completed": int(et.str.contains("completed").sum()),
        "entrada": int(et.str.contains("entrada").sum()),  # em falha permanente => espurio
    }
    comp = ev[et.str.contains("completed", na=False)]
    if "N_new" in comp.columns and len(comp):
        out["N_new_counts"] = {int(k): int(v) for k, v in comp["N_new"].value_counts().sort_index().items()}
    return out


def run(name, env_extra, expected_N_new=None):
    run_dir = os.path.join(RUNS, name)
    os.makedirs(run_dir, exist_ok=True)
    for fn in ("events.csv", "agent_telemetry.csv", "target_telemetry.csv"):
        p = os.path.join(run_dir, fn)
        if os.path.exists(p):
            os.remove(p)
    env = dict(os.environ)
    env.update({
        "PYTHONIOENCODING": "utf-8", "PROPAGATION_METHOD": "dual_pulse", "PROPAGATION_K_PROP": "0.0",
        "NUM_AGENTS": str(N), "SIM_DURATION": str(T0 + BUDGET), "K_E_TAU": f"{250.0 / N:.6f}",
        "VM_TAU_XY": str(TAU), "COMMUNICATION_FAILURE_RATE": "0", "COMMUNICATION_DELAY": "0",
        "INIT_ANGLES_EQUIDISTANT": "True", "INIT_RADIUS_RANGE": "0.0", "TARGET_MOTION_SPEED_XY": "0.0",
        "VIS_OPEN_BROWSER": "False", "SKIP_TELEMETRY_PLOTS": "True",
        "RUNS_SUMMARY_CSV_PATH": os.path.join(run_dir, "runs_summary.csv"),
        "DUAL_PULSE_INTEGRATION": "B2", "DUAL_PULSE_DELTA_SCALE": "1.0",
        "DUAL_PULSE_T_FF": f"{TAU:.6f}", "DUAL_PULSE_TTL_HOPS": str(3 * N),
    })
    env.update(env_extra)
    subprocess.run([sys.executable, MAIN_PY], cwd=run_dir, env=env, capture_output=True, text=True)
    m = metrics_from_tgt(os.path.join(run_dir, "target_telemetry.csv"))
    ev = parse_events(os.path.join(run_dir, "events.csv"))
    for fn in os.listdir(run_dir):
        if fn == "agent_telemetry.csv" or fn.endswith(".png"):
            try:
                os.remove(os.path.join(run_dir, fn))
            except OSError:
                pass
    nn = ev.get("N_new_counts", {})
    ok = nn.get(expected_N_new, 0) if expected_N_new is not None else "-"
    tot = sum(nn.values()) if nn else 0
    print(f"  [{name:12s}] t_settle={m.get('t_settle', float('nan')):7.2f}(2%={m.get('t_settle_2pct', float('nan')):6.2f}) "
          f"egap_settle={m.get('egap_settle', float('nan')):.4f} egap_final={m.get('egap_final', float('nan')):.4f} "
          f"egap_avg={m.get('egap_avg', float('nan')):.4f} tau={m.get('tau_fit', float('nan')):.2f}(R2={m.get('tau_fit_r2', float('nan')):.2f}) "
          f"fail={ev.get('failures','-')} inj={ev.get('injections','-')} compl={ev.get('completed','-')} entrada={ev.get('entrada','-')}")
    print(f"             N_new (esperado={expected_N_new}): corretos {ok}/{tot}  dist={nn}")
    return {"name": name, **m, **ev, "expected_N_new": expected_N_new}


def main():
    os.makedirs(RUNS, exist_ok=True)
    mode = os.environ.get("CHURN_MODE", "both")  # det | poisson | both
    # taxa POR AGENTE; total = rate*N. default 1.0 -> 24/min total (~1 evento/2.5s, eventos
    # se sobrepoem no tempo -> testa o caso de carimbo-de-N). (antes usei 20 = 480/min = apocalipse.)
    p_rate = os.environ.get("CHURN_POISSON_RATE", "1.0")
    p_off = os.environ.get("CHURN_POISSON_OFF", "8.0")

    if mode in ("det", "both"):
        print(f"\n=== DET. (loss=0, delay=0, falha PERMANENTE em t0={T0}, N={N}) ===")
        for name, ids, k in DET_SCENARIOS:
            run(name, {"DETERMINISTIC_FAILURE_ENABLE": "True", "DETERMINISTIC_FAILURE_AGENT_ID": ids,
                       "DETERMINISTIC_FAILURE_TIME_T0": str(T0), "DETERMINISTIC_FAILURE_OFF_TIME": "-1.0"},
                expected_N_new=N - k)

    if mode in ("poisson", "both"):
        print(f"\n=== POISSON denso (loss=0; {p_rate}/min POR AGENTE = {float(p_rate)*N:.0f}/min total, "
              f"off={p_off}s; BASELINE vs B2; menor egap_avg = melhor) ===")
        for method in ("baseline", "dual_pulse"):
            for seed in (0, 1, 2):
                run(f"{method[:4]}_pois_s{seed}",
                    {"PROPAGATION_METHOD": method,
                     "DETERMINISTIC_FAILURE_ENABLE": "False", "FAILURE_ENABLE": "True",
                     "FAILURE_MEAN_FAILURES_PER_MIN": p_rate, "FAILURE_OFF_TIME": p_off,
                     "EXPERIMENT_SEED": str(seed)},
                    expected_N_new=None)
    print("\nLeitura: adj* tau MAIOR (sub-correcao); non* tau ~rapido (aditivo). Poisson: comparar")
    print("egap_avg baseline vs B2 (B2<baseline => overlay ajuda sob churn; >=  => nao ajuda/atrapalha).")
    print("N_new espalhado / entrada alto no Poisson = topologia muda rapido (eventos sobrepostos).")


if __name__ == "__main__":
    main()
