#!/usr/bin/env python
"""Fase 2 -- Colapso adimensional: varredura (N x agilidade [x dt] x seed) p/ a lei N^2/tau_a.

Cada celula (N, tau_a, dt, seed) roda baseline e B2 no regime estavel, knobs adaptados:
  - DUAL_PULSE_INTEGRATION=B2 ; DELTA_SCALE=1.0 ; T_FF = C_FF*tau_a (c_FF=1.0)
  - K_E_TAU = GAIN_PRODUCT/N (=250/N) ; VM_TAU_XY=tau_a ; VM_MAX_SPEED fixo (opt A)
  - DUAL_PULSE_TTL_HOPS=3N ; CONTROL_PERIOD=dt (default 0.01)
  - MULTI-SEED: EXPERIMENT_SEED=seed + INIT_RADIUS_RANGE=COLLAPSE_INIT_RADIUS (scatter de raio),
    angulos equidistantes (anel uniforme antes da falha), vitima varia por seed.
Saida (merge incremental): collapse_results[/_TAG].csv com tau_fit + Pe=N*dt/tau_a por celula.

Uso:
    python experiments/scaling_law/run_collapse_sweep.py
    # env: COLLAPSE_N_VALUES, COLLAPSE_TAU_VALUES, COLLAPSE_METHODS, COLLAPSE_SEEDS="0,1,2",
    #      COLLAPSE_INIT_RADIUS="0.05", COLLAPSE_C_FF, CONTROL_PERIOD, COLLAPSE_TAG
"""
import os
import sys
import subprocess

import numpy as np
import pandas as pd

from metrics_util import event_metrics   # helper canonico (t_settle + tau_fit + egap)

_THIS = os.path.abspath(__file__)
EXP_DIR = os.path.dirname(_THIS)
REPO_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))
MAIN_PY = os.path.join(REPO_ROOT, "main.py")

_TAG = os.environ.get("COLLAPSE_TAG", "")
_SUF = ("_" + _TAG) if _TAG else ""
RUNS_DIR = os.path.join(EXP_DIR, "collapse_runs" + _SUF)
RESULTS_CSV = os.path.join(EXP_DIR, "collapse_results" + _SUF + ".csv")

N_VALUES = [int(x) for x in os.environ.get("COLLAPSE_N_VALUES", "8,16,24").split(",") if x.strip()]
TAU_VALUES = [float(x) for x in os.environ.get("COLLAPSE_TAU_VALUES", "0.2,0.5,1.0,2.0").split(",") if x.strip()]
METHODS = [m.strip() for m in os.environ.get("COLLAPSE_METHODS", "baseline,dual_pulse").split(",") if m.strip()]
SEEDS = [int(x) for x in os.environ.get("COLLAPSE_SEEDS", "0").split(",") if x.strip()]
INIT_RADIUS = float(os.environ.get("COLLAPSE_INIT_RADIUS", "0.0"))   # scatter de raio p/ multi-seed
C_FF = float(os.environ.get("COLLAPSE_C_FF", "1.0"))                 # T_FF = C_FF * tau_a
GAIN_PRODUCT = float(os.environ.get("GAIN_PRODUCT", "250"))          # K_E_TAU = GAIN_PRODUCT / N
DT = float(os.environ.get("CONTROL_PERIOD", "0.01"))                # control period (for Pe)
BUDGET_BASELINE = float(os.environ.get("COLLAPSE_BUDGET_BASELINE", "20"))   # floor
BUDGET_B2 = float(os.environ.get("COLLAPSE_BUDGET_B2", "15"))               # floor
T0 = float(os.environ.get("SCALING_T0", "5.0"))
TAIL_FLOOR_FRAC = 0.05
LATE_WIN = 20.0


def victim_node_id(n, seed=0):
    return 2 + ((n // 2 + seed) % n)


def metrics_from_csv(tgt):
    if not os.path.exists(tgt):
        return {}
    return event_metrics(pd.read_csv(tgt), T0)   # tau_fit + t_settle + egap (helper canonico)


def run_cell(method, n, tau_xy, seed):
    is_b2 = (method == "dual_pulse")
    k_e_tau = GAIN_PRODUCT / n
    t_ff = C_FF * tau_xy
    # auto-escala a duracao a ~4x o tau esperado; baseline ~3.3*N^2*dt(s)~0.034*N^2(s); B2 ~2.3*T_FF.
    budget = max(BUDGET_B2, 12.0 * t_ff) if is_b2 else max(BUDGET_BASELINE, 14.0 * n * n * DT)
    victim = victim_node_id(n, seed)
    duration = T0 + budget
    run_dir = os.path.join(RUNS_DIR, f"{method}_N{n}_tau{tau_xy:g}_dt{DT:g}_s{seed}")
    os.makedirs(run_dir, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "PYTHONIOENCODING": "utf-8",
        "PROPAGATION_METHOD": method,
        "PROPAGATION_K_PROP": "0.0",
        "NUM_AGENTS": str(n),
        "SIM_DURATION": str(duration),
        "K_E_TAU": f"{k_e_tau:.6f}",
        "VM_TAU_XY": str(tau_xy),
        "EXPERIMENT_SEED": str(seed),
        "INIT_ANGLES_EQUIDISTANT": "True",
        "INIT_RADIUS_RANGE": f"{INIT_RADIUS:g}",
        "TARGET_MOTION_SPEED_XY": "0.0",
        "VIS_OPEN_BROWSER": "False",
        "SKIP_TELEMETRY_PLOTS": "True",
        "DETERMINISTIC_FAILURE_ENABLE": "True",
        "DETERMINISTIC_FAILURE_AGENT_ID": str(victim),
        "DETERMINISTIC_FAILURE_TIME_T0": str(T0),
        "DETERMINISTIC_FAILURE_OFF_TIME": "-1.0",
        "RUNS_SUMMARY_CSV_PATH": os.path.join(run_dir, "runs_summary.csv"),
    })
    if is_b2:
        env.update({"DUAL_PULSE_INTEGRATION": "B2", "DUAL_PULSE_DELTA_SCALE": "1.0",
                    "DUAL_PULSE_T_FF": f"{t_ff:.6f}", "DUAL_PULSE_TTL_HOPS": str(3 * n)})
    tag = "B2" if is_b2 else "baseline"
    print(f"  -> {tag:8s} N={n:3d} tau_a={tau_xy:<4g} dt={DT:<5g} s={seed} "
          f"(K={k_e_tau:.2f}, T_FF={t_ff:g}, budget={budget:g}) ...", flush=True)
    proc = subprocess.run([sys.executable, MAIN_PY], cwd=run_dir, env=env, capture_output=True, text=True)
    tgt = os.path.join(run_dir, "target_telemetry.csv")
    if not os.path.exists(tgt):
        print(f"     FAILED (rc={proc.returncode})\n{(proc.stderr or '')[-800:]}")
        return None
    m = metrics_from_csv(tgt)
    for fn in os.listdir(run_dir):   # footprint: so target_telemetry e' usado
        if fn == "agent_telemetry.csv" or fn.endswith(".png"):
            try:
                os.remove(os.path.join(run_dir, fn))
            except OSError:
                pass
    m.update({"method": tag, "N": n, "tau_xy": tau_xy, "dt": DT, "seed": seed,
              "T_FF": t_ff, "k_e_tau": k_e_tau, "init_radius": INIT_RADIUS, "Pe": n * DT / tau_xy})
    print(f"     tau_fit={m.get('tau_fit'):.3f}  R2={m.get('tau_fit_r2'):.2f}  "
          f"egap_final={m.get('egap_final'):.4f}  late_std={m.get('egap_late_std'):.4f}")
    return m


def _key(r):
    return (str(r["method"]), int(r["N"]), float(r["tau_xy"]),
            round(float(r["dt"]), 6), int(r.get("seed", 0)))


def main():
    os.makedirs(RUNS_DIR, exist_ok=True)
    store = {}
    if os.path.exists(RESULTS_CSV):
        try:
            for r in pd.read_csv(RESULTS_CSV).to_dict("records"):
                store[_key(r)] = r
        except Exception:
            pass
    print(f"Collapse sweep: N={N_VALUES} x tau_a={TAU_VALUES} x seeds={SEEDS}, methods={METHODS}, "
          f"dt={DT}, init_radius={INIT_RADIUS}\n  T_FF=c_FF*tau_a (c_FF={C_FF}), K_E_TAU={GAIN_PRODUCT}/N, "
          f"budgets auto-escalados; {len(store)} celulas ja no CSV\n")
    for n in N_VALUES:
        for tau in TAU_VALUES:
            for seed in SEEDS:
                for method in METHODS:
                    r = run_cell(method, n, tau, seed)
                    if r:
                        store[_key(r)] = r
                        pd.DataFrame(list(store.values())).to_csv(RESULTS_CSV, index=False)
    df = pd.DataFrame(list(store.values()))
    if df.empty:
        print("\nSem resultados."); return
    # resumo: vantagem (mediana entre seeds) por (N, tau, dt)
    print("\n=== ADVANTAGE = mediana_seeds(tau_base) / mediana_seeds(tau_B2)  por (N,tau_a,dt) ===")
    print(f"{'N':>3} {'tau_a':>6} {'dt':>5} {'nseed':>5} {'Pe':>7} {'tau_base':>9} {'tau_B2':>8} {'adv':>7}")
    keys = sorted({(int(r['N']), float(r['tau_xy']), round(float(r['dt']), 6)) for r in store.values()})
    for (n, tau, dt) in keys:
        sub = df[(df.N == n) & (df.tau_xy == tau) & (df.dt.round(6) == dt)]
        b = sub[sub.method == "baseline"]["tau_fit"].dropna()
        o = sub[sub.method == "B2"]["tau_fit"].dropna()
        tb = float(b.median()) if len(b) else float("nan")
        to = float(o.median()) if len(o) else float("nan")
        adv = (tb / to) if (np.isfinite(tb) and np.isfinite(to) and to > 0) else float("nan")
        nseed = max(len(b), len(o))
        print(f"{n:>3} {tau:>6g} {dt:>5g} {nseed:>5d} {n*dt/tau:>7.3f} {tb:>9.2f} {to:>8.2f} {adv:>7.2f}")
    print(f"\nWrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
