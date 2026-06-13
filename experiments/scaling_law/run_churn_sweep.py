#!/usr/bin/env python
"""Fase 3 / Track B -- sweep da TAXA de churn (esparso->denso), baseline vs B2 (ISOLADO: loss=0).

Mede egap_avg (erro de espacamento medio em regime) por (taxa, metodo). Sob churn continuo nao ha
assentamento -> egap_avg e' a metrica certa. Previsao: o overlay (B2) ganha (egap_avg menor) quando
o intervalo-entre-eventos >> tempo de completar do overlay (~2-3s) E o baseline NAO assenta entre
eventos (intervalo < tau_base~20s); converge ao baseline quando os eventos sobrepoem (denso).

Taxas em /min TOTAL (por-agente = total/N). Recovery finito (off). Merge incremental.

Uso:
    python experiments/scaling_law/run_churn_sweep.py
    # env: CHURN_RATES="6,12,24,48" (total/min)  CHURN_OFF="8.0"  CHURN_SEEDS="0,1,2"
    #      CHURN_N="24"  CHURN_TAU="1.0"  CHURN_BUDGET="150"  CHURN_METHODS="baseline,dual_pulse"
"""
import os
import sys
import subprocess

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_util import aggregate_seeds, effort_metrics  # noqa: E402

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))
MAIN_PY = os.path.join(REPO_ROOT, "main.py")
_TAG = os.environ.get("CHURN_TAG", "")
_SUF = ("_" + _TAG) if _TAG else ""
RUNS_DIR = os.path.join(EXP_DIR, "churn_sweep_runs" + _SUF)
RESULTS_CSV = os.path.join(EXP_DIR, "churn_sweep_results" + _SUF + ".csv")

RATES_TOTAL = [float(x) for x in os.environ.get("CHURN_RATES", "6,12,24,48").split(",") if x.strip()]
OFF = float(os.environ.get("CHURN_OFF", "8.0"))
SEEDS = [int(x) for x in os.environ.get("CHURN_SEEDS", "0,1,2").split(",") if x.strip()]
N = int(os.environ.get("CHURN_N", "24"))
TAU = float(os.environ.get("CHURN_TAU", "1.0"))
BUDGET = float(os.environ.get("CHURN_BUDGET", "150"))
METHODS = [m.strip() for m in os.environ.get("CHURN_METHODS", "baseline,dual_pulse").split(",") if m.strip()]
T0 = 5.0
WARMUP_AVG = 15.0  # ignora o transiente inicial no egap_avg
VMAX = float(os.environ.get("VM_MAX_SPEED_XY", "10.0"))  # p/ effort/saturacao (M5/M6)
DT = float(os.environ.get("CONTROL_PERIOD", "0.05"))     # pinned in child (label==reality)


def metrics_from_tgt(tgt):
    if not os.path.exists(tgt):
        return {}
    df = pd.read_csv(tgt)
    steady = df[df["timestamp"] >= T0 + WARMUP_AVG]["E_gap"].to_numpy(float)
    steady = steady[np.isfinite(steady)]
    if not steady.size:
        return {}
    return {"egap_avg": float(np.mean(steady)), "egap_p90": float(np.percentile(steady, 90)),
            "egap_max": float(np.max(steady))}


def run_cell(method, rate_total, seed):
    is_b2 = (method == "dual_pulse")
    per_agent = rate_total / float(N)
    run_dir = os.path.join(RUNS_DIR, f"{method}_rate{rate_total:g}_s{seed}")
    os.makedirs(run_dir, exist_ok=True)
    tgt = os.path.join(run_dir, "target_telemetry.csv")
    if os.path.exists(tgt):
        os.remove(tgt)
    env = dict(os.environ)
    env.update({
        "PYTHONIOENCODING": "utf-8", "PROPAGATION_METHOD": method, "PROPAGATION_K_PROP": "0.0",
        "NUM_AGENTS": str(N), "SIM_DURATION": str(T0 + BUDGET), "K_E_TAU": f"{250.0 / N:.6f}",
        "CONTROL_PERIOD": f"{DT:g}", "AGENT_STATE_TIMEOUT": f"{5.0 * DT:g}",  # churn = real outages, loss=0 -> 5 ticks clean
        "VM_TAU_XY": str(TAU), "COMMUNICATION_FAILURE_RATE": "0", "COMMUNICATION_DELAY": "0",
        "INIT_ANGLES_EQUIDISTANT": "True", "INIT_RADIUS_RANGE": "0.0", "TARGET_MOTION_SPEED_XY": "0.0",
        "VIS_OPEN_BROWSER": "False", "SKIP_TELEMETRY_PLOTS": "True",
        "DETERMINISTIC_FAILURE_ENABLE": "False", "FAILURE_ENABLE": "True",
        "FAILURE_MEAN_FAILURES_PER_MIN": f"{per_agent:.6f}", "FAILURE_OFF_TIME": f"{OFF:g}",
        "EXPERIMENT_SEED": str(seed),
        "RUNS_SUMMARY_CSV_PATH": os.path.join(run_dir, "runs_summary.csv"),
    })
    if is_b2:
        env.update({"DUAL_PULSE_INTEGRATION": "B2", "DUAL_PULSE_DELTA_SCALE": "1.0",
                    "DUAL_PULSE_T_FF": f"{TAU:.6f}", "DUAL_PULSE_TTL_HOPS": str(3 * N)})
    tag = "B2" if is_b2 else "baseline"
    print(f"  -> {tag:8s} rate={rate_total:>4g}/min(={per_agent:.3f}/agent) off={OFF:g} s={seed} ...", flush=True)
    proc = subprocess.run([sys.executable, MAIN_PY], cwd=run_dir, env=env, capture_output=True, text=True)
    m = metrics_from_tgt(tgt)
    if not m:
        print(f"     FAILED (rc={proc.returncode})\n{(proc.stderr or '')[-600:]}")
        return None
    # Effort/saturation/fairness (M5/M6/M2) BEFORE deleting agent_telemetry.csv.
    m.update(effort_metrics(os.path.join(run_dir, "agent_telemetry.csv"),
                            t0=T0 + WARMUP_AVG, vmax=VMAX))
    for fn in os.listdir(run_dir):
        if fn == "agent_telemetry.csv" or fn.endswith(".png"):
            try:
                os.remove(os.path.join(run_dir, fn))
            except OSError:
                pass
    m.update({"method": tag, "N": N, "tau_xy": TAU, "rate_total": rate_total, "off": OFF, "seed": seed})
    print(f"     egap_avg={m['egap_avg']:.4f}  egap_p90={m['egap_p90']:.4f}  "
          f"effort={m.get('effort_mean_v2', float('nan')):.4f}")
    return m


def _key(r):
    return (str(r["method"]), float(r["rate_total"]), int(r["seed"]))


def main():
    os.makedirs(RUNS_DIR, exist_ok=True)
    store = {}
    if os.path.exists(RESULTS_CSV):
        try:
            for r in pd.read_csv(RESULTS_CSV).to_dict("records"):
                store[_key(r)] = r
        except Exception:
            pass
    print(f"Churn-rate sweep: rates(total/min)={RATES_TOTAL}, off={OFF}s, methods={METHODS}, "
          f"seeds={SEEDS}, N={N}, budget={BUDGET}; {len(store)} ja no CSV\n")
    for rate in RATES_TOTAL:
        for seed in SEEDS:
            for method in METHODS:
                r = run_cell(method, rate, seed)
                if r:
                    store[_key(r)] = r
                    pd.DataFrame(list(store.values())).to_csv(RESULTS_CSV, index=False)
    df = pd.DataFrame(list(store.values()))
    if df.empty:
        print("\nSem resultados."); return
    print("\n=== egap_avg por taxa: mediana E pior caso entre seeds; vantagem = baseline/B2 ===")
    print(f"{'rate/min':>9} {'inter(s)':>8} {'egap_base':>10} {'egap_B2':>9} {'vantagem':>9} "
          f"{'B2_worst':>9} {'adv_worst':>9} {'eff_B2/bs':>9}")
    for rate in sorted(df.rate_total.unique()):
        b = df[(df.method == "baseline") & (df.rate_total == rate)]
        o = df[(df.method == "B2") & (df.rate_total == rate)]
        ab = aggregate_seeds(b["egap_avg"].tolist())
        ao = aggregate_seeds(o["egap_avg"].tolist())
        eb, eo = ab["median"], ao["median"]
        adv = (eb / eo) if (np.isfinite(eb) and np.isfinite(eo) and eo > 0) else float("nan")
        # Worst-case advantage: the baseline's BEST seed vs the overlay's WORST —
        # the conservative bound the campaign requires (not just central tendency).
        b_best = float(np.nanmin(b["egap_avg"])) if len(b) else float("nan")
        adv_worst = (b_best / ao["worst"]) if (np.isfinite(b_best) and np.isfinite(ao["worst"])
                                               and ao["worst"] > 0) else float("nan")
        eff_ratio = float("nan")
        if "effort_mean_v2" in df.columns:
            eb_eff = float(b["effort_mean_v2"].median()) if len(b) else float("nan")
            eo_eff = float(o["effort_mean_v2"].median()) if len(o) else float("nan")
            if np.isfinite(eb_eff) and eb_eff > 0 and np.isfinite(eo_eff):
                eff_ratio = eo_eff / eb_eff
        inter = 60.0 / rate if rate > 0 else float("inf")
        print(f"{rate:>9g} {inter:>8.2f} {eb:>10.4f} {eo:>9.4f} {adv:>9.2f} "
              f"{ao['worst']:>9.4f} {adv_worst:>9.2f} {eff_ratio:>9.2f}")
    print(f"\nWrote {RESULTS_CSV}")
    print("Leitura: vantagem>1 = overlay ajuda; ~1 = empata (denso). adv_worst = limite conservador "
          "(melhor seed do baseline vs pior do B2). eff_B2/bs > 1 = overlay gasta mais atuacao.")


if __name__ == "__main__":
    main()
