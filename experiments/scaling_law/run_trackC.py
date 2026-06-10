#!/usr/bin/env python
"""Fase 3 / Track C -- alvo MOVEL (cenario-mae). baseline vs B2 sob movimento + estresses A/B.

Mede DUAS coisas: espacamento (E_gap -> t_settle/egap_avg via metrics_util, trabalho do overlay) e
TRACKING radial (E_r, do target_telemetry, que o overlay NAO deve piorar). Conta injecoes (espurias
sob movimento?). Movimento: 'const' (periodo enorme + fronteira grande = ~velocidade constante) vs
'maneuver' (periodo 1s = direcao aleatoria). Fixes A/B ligados onde fazem sentido (FD-timeout p/
perda; gate p/ churn). Merge incremental.

Uso:
    python experiments/scaling_law/run_trackC.py
    # env: TRACKC_SCENARIO=none|fail|loss|delay|churn_sparse|churn_dense
    #      TRACKC_MOTION=const|maneuver|both  TRACKC_SPEED=3  TRACKC_METHODS=baseline,dual_pulse
    #      TRACKC_SEEDS=0,1,2  TRACKC_N=24  TRACKC_TAU=1.0  TRACKC_BUDGET=100  TRACKC_TAG=
"""
import os
import sys
import subprocess

import numpy as np
import pandas as pd

from metrics_util import event_metrics

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))
MAIN_PY = os.path.join(REPO_ROOT, "main.py")

_TAG = os.environ.get("TRACKC_TAG", "")
_SUF = ("_" + _TAG) if _TAG else ""
RUNS_DIR = os.path.join(EXP_DIR, "trackC_runs" + _SUF)
RESULTS_CSV = os.path.join(EXP_DIR, "trackC_results" + _SUF + ".csv")

SCENARIOS = [s.strip() for s in os.environ.get("TRACKC_SCENARIO", "none").split(",") if s.strip()]
MOTION = os.environ.get("TRACKC_MOTION", "both")  # const | maneuver | both
SPEED = float(os.environ.get("TRACKC_SPEED", "3.0"))
METHODS = [m.strip() for m in os.environ.get("TRACKC_METHODS", "baseline,dual_pulse").split(",") if m.strip()]
SEEDS = [int(x) for x in os.environ.get("TRACKC_SEEDS", "0,1,2").split(",") if x.strip()]
N = int(os.environ.get("TRACKC_N", "24"))
TAU = float(os.environ.get("TRACKC_TAU", "1.0"))
BUDGET = float(os.environ.get("TRACKC_BUDGET", "100"))
T0 = 5.0


def victim_node_id(n, seed=0):
    return 2 + ((n // 2 + seed) % n)


def motion_env(motion):
    if motion == "const":
        return {"TARGET_MOTION_SPEED_XY": f"{SPEED:g}", "TARGET_MOTION_PERIOD": "100000",
                "TARGET_MOTION_BOUNDARY_XY": "200"}
    # maneuver
    return {"TARGET_MOTION_SPEED_XY": f"{SPEED:g}", "TARGET_MOTION_PERIOD": "1.0",
            "TARGET_MOTION_BOUNDARY_XY": "20"}


def scenario_env(scenario, n, seed):
    """Env extras + flag 'has_event' (se ha um evento de reconfiguracao p/ t_settle)."""
    # FAILURE_ENABLE=True por default: o timer de checagem de falha HOSPEDA a logica deterministica
    # (protocol_agent:261 so agenda o timer se FAILURE_ENABLE). 'none' o desliga (sem falhas).
    e = {"DETERMINISTIC_FAILURE_ENABLE": "False", "FAILURE_ENABLE": "True",
         "COMMUNICATION_FAILURE_RATE": "0", "COMMUNICATION_DELAY": "0",
         "DUAL_PULSE_GATE_ENABLE": "False"}
    has_event = False
    if scenario == "none":
        e["FAILURE_ENABLE"] = "False"   # sem nenhuma falha (movimento puro)
    elif scenario == "fail":
        e.update({"DETERMINISTIC_FAILURE_ENABLE": "True",
                  "DETERMINISTIC_FAILURE_AGENT_ID": str(victim_node_id(n, seed)),
                  "DETERMINISTIC_FAILURE_TIME_T0": str(T0), "DETERMINISTIC_FAILURE_OFF_TIME": "-1.0"})
        has_event = True
    elif scenario == "loss":
        e.update({"DETERMINISTIC_FAILURE_ENABLE": "True",
                  "DETERMINISTIC_FAILURE_AGENT_ID": str(victim_node_id(n, seed)),
                  "DETERMINISTIC_FAILURE_TIME_T0": str(T0), "DETERMINISTIC_FAILURE_OFF_TIME": "-1.0",
                  "COMMUNICATION_FAILURE_RATE": "0.1", "AGENT_STATE_TIMEOUT": "0.2"})  # fix A ligado
        has_event = True
    elif scenario == "delay":
        e.update({"DETERMINISTIC_FAILURE_ENABLE": "True",
                  "DETERMINISTIC_FAILURE_AGENT_ID": str(victim_node_id(n, seed)),
                  "DETERMINISTIC_FAILURE_TIME_T0": str(T0), "DETERMINISTIC_FAILURE_OFF_TIME": "-1.0",
                  "COMMUNICATION_DELAY": "0.05"})
        has_event = True
    elif scenario == "churn_sparse":
        # gate DESLIGADO: com o gatilho premissa-limpo o overlay AJUDA sob churn (o gate atrapalha).
        e.update({"FAILURE_ENABLE": "True", "FAILURE_MEAN_FAILURES_PER_MIN": f"{6.0 / n:.6f}",
                  "FAILURE_OFF_TIME": "8.0", "DUAL_PULSE_GATE_ENABLE": "False"})
    elif scenario == "churn_dense":
        e.update({"FAILURE_ENABLE": "True", "FAILURE_MEAN_FAILURES_PER_MIN": f"{24.0 / n:.6f}",
                  "FAILURE_OFF_TIME": "8.0", "DUAL_PULSE_GATE_ENABLE": "False"})
    elif scenario == "recover":
        # ENTRADA controlada: falha determinística em t0 e RECUPERA em t0+OFF (testa a detecção
        # local de ENTRADA com o gatilho premissa-limpo).
        e.update({"DETERMINISTIC_FAILURE_ENABLE": "True", "FAILURE_ENABLE": "True",
                  "DETERMINISTIC_FAILURE_AGENT_ID": str(victim_node_id(n, seed)),
                  "DETERMINISTIC_FAILURE_TIME_T0": str(T0), "DETERMINISTIC_FAILURE_OFF_TIME": "10.0"})
        has_event = True
    elif scenario == "stress":
        # TUDO junto: churn medio-denso + perda (com FD-fix) + atraso ABAIXO do onset (5*dt) + M8.
        e.update({"FAILURE_ENABLE": "True", "FAILURE_MEAN_FAILURES_PER_MIN": f"{18.0 / n:.6f}",
                  "FAILURE_OFF_TIME": "8.0", "DUAL_PULSE_GATE_ENABLE": "False",
                  "COMMUNICATION_FAILURE_RATE": "0.1", "AGENT_STATE_TIMEOUT": "0.2",
                  "COMMUNICATION_DELAY": "0.02"})
    return e, has_event


def tracking_avg(tgt):
    """E_r medio (tracking radial) e E_vr medio na janela estavel."""
    df = pd.read_csv(tgt)
    seg = df[df["timestamp"] >= T0 + 10.0]
    er = float(seg["E_r"].abs().mean()) if "E_r" in seg.columns and len(seg) else float("nan")
    evr = float(seg["E_vr"].abs().mean()) if "E_vr" in seg.columns and len(seg) else float("nan")
    return er, evr


def n_injections(ev_path):
    if not os.path.exists(ev_path):
        return 0
    ev = pd.read_csv(ev_path)
    return int(ev["event_type"].astype(str).str.contains("self_shift").sum())


def run_cell(method, scenario, motion, seed):
    is_b2 = (method == "dual_pulse")
    run_dir = os.path.join(RUNS_DIR, f"{method}_{scenario}_{motion}_s{seed}")
    os.makedirs(run_dir, exist_ok=True)
    for fn in ("events.csv", "agent_telemetry.csv", "target_telemetry.csv"):
        p = os.path.join(run_dir, fn)
        if os.path.exists(p):
            os.remove(p)
    sc_env, has_event = scenario_env(scenario, N, seed)
    env = dict(os.environ)
    env.update({
        "PYTHONIOENCODING": "utf-8", "PROPAGATION_METHOD": method, "PROPAGATION_K_PROP": "0.0",
        "NUM_AGENTS": str(N), "SIM_DURATION": str(T0 + BUDGET), "K_E_TAU": f"{250.0 / N:.6f}",
        "VM_TAU_XY": str(TAU), "INIT_ANGLES_EQUIDISTANT": "True", "INIT_RADIUS_RANGE": "0.0",
        "EXPERIMENT_SEED": str(seed), "VIS_OPEN_BROWSER": "False", "SKIP_TELEMETRY_PLOTS": "True",
        "RUNS_SUMMARY_CSV_PATH": os.path.join(run_dir, "runs_summary.csv"),
    })
    env.update(motion_env(motion))
    env.update(sc_env)
    if is_b2:
        env.update({"DUAL_PULSE_INTEGRATION": "B2", "DUAL_PULSE_DELTA_SCALE": "1.0",
                    "DUAL_PULSE_T_FF": f"{TAU:.6f}", "DUAL_PULSE_TTL_HOPS": str(3 * N)})
    tag = "B2" if is_b2 else "baseline"
    print(f"  -> {tag:8s} {scenario:12s} {motion:8s} s={seed} ...", flush=True)
    proc = subprocess.run([sys.executable, MAIN_PY], cwd=run_dir, env=env, capture_output=True, text=True)
    tgt = os.path.join(run_dir, "target_telemetry.csv")
    if not os.path.exists(tgt):
        print(f"     FAILED (rc={proc.returncode})\n{(proc.stderr or '')[-600:]}")
        return None
    m = event_metrics(pd.read_csv(tgt), T0)
    er, evr = tracking_avg(tgt)
    inj = n_injections(os.path.join(run_dir, "events.csv"))
    for fn in os.listdir(run_dir):
        if fn == "agent_telemetry.csv" or fn.endswith(".png"):
            try:
                os.remove(os.path.join(run_dir, fn))
            except OSError:
                pass
    row = {"method": tag, "scenario": scenario, "motion": motion, "seed": seed, "speed": SPEED,
           "egap_avg": m.get("egap_avg", float("nan")), "t_settle": m.get("t_settle", float("nan")),
           "egap_settle": m.get("egap_settle", float("nan")), "Er_avg": er, "Evr_avg": evr,
           "n_inj": inj, "has_event": has_event}
    print(f"     egap_avg={row['egap_avg']:.4f} t_settle={row['t_settle']:.2f} "
          f"Er_avg={er:.4f} n_inj={inj}")
    return row


def _key(r):
    return (str(r["method"]), str(r["scenario"]), str(r["motion"]), int(r["seed"]))


def main():
    os.makedirs(RUNS_DIR, exist_ok=True)
    store = {}
    if os.path.exists(RESULTS_CSV):
        try:
            for r in pd.read_csv(RESULTS_CSV).to_dict("records"):
                store[_key(r)] = r
        except Exception:
            pass
    motions = ["const", "maneuver"] if MOTION == "both" else [MOTION]
    print(f"Track C: scenarios={SCENARIOS}, motions={motions}, speed={SPEED}, methods={METHODS}, "
          f"seeds={SEEDS}, N={N}; {len(store)} ja no CSV\n")
    for scenario in SCENARIOS:
        for motion in motions:
            for seed in SEEDS:
                for method in METHODS:
                    r = run_cell(method, scenario, motion, seed)
                    if r:
                        store[_key(r)] = r
                        pd.DataFrame(list(store.values())).to_csv(RESULTS_CSV, index=False)
    df = pd.DataFrame(list(store.values()))
    if df.empty:
        print("\nSem resultados."); return
    print("\n=== resumo (mediana entre seeds): espacamento (egap_avg) + tracking (Er_avg) ===")
    print(f"{'scenario':12s} {'motion':9s} {'egap_base':>9} {'egap_B2':>8} {'Er_base':>8} {'Er_B2':>7} {'inj_B2':>6}")
    keys = sorted({(r["scenario"], r["motion"]) for r in store.values()})
    for (sc, mo) in keys:
        b = df[(df.method == "baseline") & (df.scenario == sc) & (df.motion == mo)]
        o = df[(df.method == "B2") & (df.scenario == sc) & (df.motion == mo)]
        def med(x, c): return float(x[c].median()) if len(x) else float("nan")
        print(f"{sc:12s} {mo:9s} {med(b,'egap_avg'):>9.4f} {med(o,'egap_avg'):>8.4f} "
              f"{med(b,'Er_avg'):>8.4f} {med(o,'Er_avg'):>7.4f} {med(o,'n_inj'):>6.0f}")
    print(f"\nWrote {RESULTS_CSV}")
    print("Leitura cenario 'none': B2~baseline (egap+Er) e inj_B2~0 esperado (const). inj alto = espurio.")


if __name__ == "__main__":
    main()
