#!/usr/bin/env python
"""Fase 3 / Track A -- robustez a comunicacao degradada (perda / atraso / redundancia).

Varre COMMUNICATION_FAILURE_RATE (perda), COMMUNICATION_DELAY (atraso) e
DUAL_PULSE_BROADCAST_REPEATS (redundancia dos pulsos), p/ baseline e B2, N e tau_a fixos,
MULTI-SEED (perda e' estocastica). Budget grande (sob perda o B2 nao assenta). Metrica robusta
= egap_final (o tau_fit quebra sob perda). Merge incremental.

O sweep aplica por default o FD-fix validado: AGENT_STATE_TIMEOUT = 20*dt (0.2 s),
porque o default do projeto (5*dt) torna o detector de falhas vulneravel a falsos
positivos sob perda (5 perdas consecutivas marcam um vizinho VIVO como morto ->
tempestade de SAIDA/ENTRADA falsos; diagnostico Track A 2026-06). O timeout usado
e' gravado em cada linha do CSV (coluna agent_state_timeout) para proveniencia.
Use COMM_FD_TIMEOUT para estudar o proprio timeout (ex.: reproduzir a vulnerabilidade
pre-fix com COMM_FD_TIMEOUT=0.05).

Uso:
    python experiments/scaling_law/run_comm_sweep.py
    # env: COMM_LOSS_VALUES="0,0.05,0.1,0.2,0.4"  COMM_DELAY="0.0"  COMM_REPEATS="2"
    #      COMM_N="24" COMM_TAU="1.0" COMM_METHODS="baseline,dual_pulse" COMM_SEEDS="0,1,2"
    #      COMM_BUDGET="150"  COMM_TAG=""  COMM_FD_TIMEOUT="0.2"
"""
import os
import sys
import subprocess

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_util import effort_metrics  # noqa: E402

_THIS = os.path.abspath(__file__)
EXP_DIR = os.path.dirname(_THIS)
REPO_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))
MAIN_PY = os.path.join(REPO_ROOT, "main.py")

_TAG = os.environ.get("COMM_TAG", "")
_SUF = ("_" + _TAG) if _TAG else ""
RUNS_DIR = os.path.join(EXP_DIR, "comm_runs" + _SUF)
RESULTS_CSV = os.path.join(EXP_DIR, "comm_results" + _SUF + ".csv")

LOSS_VALUES = [float(x) for x in os.environ.get("COMM_LOSS_VALUES", "0,0.05,0.1,0.2,0.4").split(",") if x.strip()]
DELAY = float(os.environ.get("COMM_DELAY", "0.0"))
REPEATS = [int(x) for x in os.environ.get("COMM_REPEATS", "2").split(",") if x.strip()]
N = int(os.environ.get("COMM_N", "24"))
TAU = float(os.environ.get("COMM_TAU", "1.0"))
METHODS = [m.strip() for m in os.environ.get("COMM_METHODS", "baseline,dual_pulse").split(",") if m.strip()]
SEEDS = [int(x) for x in os.environ.get("COMM_SEEDS", "0,1,2").split(",") if x.strip()]
BUDGET = float(os.environ.get("COMM_BUDGET", "150"))
GAIN_PRODUCT = float(os.environ.get("GAIN_PRODUCT", "250"))
DT = 0.01
# FD-fix (Track A): failure-detector timeout robust to packet loss. 20*dt covers
# loss up to ~0.4 (0.4^20 ~ 1e-8 false-positive odds per window) vs the project
# default 5*dt which is exactly the pre-fix vulnerable setting.
FD_TIMEOUT = float(os.environ.get("COMM_FD_TIMEOUT", str(20 * DT)))
T0 = 5.0
TAIL_FLOOR_FRAC = 0.05
LATE_WIN = 20.0


def victim_node_id(n, seed=0):
    return 2 + ((n // 2 + seed) % n)


def metrics_from_csv(tgt):
    if not os.path.exists(tgt):
        return {}
    df = pd.read_csv(tgt)
    full = df.copy()
    df = df[df["timestamp"] >= T0].reset_index(drop=True)
    if df.empty:
        return {}
    t = df["timestamp"].to_numpy(float); e = df["E_gap"].to_numpy(float)
    i_pk = int(np.nanargmax(e)); e_pk = e[i_pk]
    mask = (np.arange(e.size) >= i_pk) & (e > TAIL_FLOOR_FRAC * e_pk) & np.isfinite(e)
    tau, r2 = np.nan, np.nan
    if mask.sum() >= 5:
        tt, ee = t[mask], e[mask]
        coef = np.polyfit(tt, np.log(ee), 1)
        if coef[0] < 0:
            tau = -1.0 / coef[0]
        pred = np.polyval(coef, tt)
        ss_res = float(np.sum((np.log(ee) - pred) ** 2)); ss_tot = float(np.sum((np.log(ee) - np.mean(np.log(ee))) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    t_end = float(full["timestamp"].max())
    late = full[full["timestamp"] >= t_end - LATE_WIN]["E_gap"].to_numpy(float)
    late_std = float(np.std(late)) if late.size else np.nan
    egap_final = float(e[-1])
    settled = bool(egap_final < 0.05 and (not np.isfinite(late_std) or late_std < 0.02))
    return {"tau_fit": float(tau), "tau_fit_r2": float(r2), "egap_final": egap_final,
            "egap_late_std": late_std, "settled": settled}


def run_cell(method, loss, seed, repeats):
    is_b2 = (method == "dual_pulse")
    k_e_tau = GAIN_PRODUCT / N
    victim = victim_node_id(N, seed)
    run_dir = os.path.join(RUNS_DIR, f"{method}_loss{loss:g}_delay{DELAY:g}_r{repeats}_s{seed}")
    os.makedirs(run_dir, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "PYTHONIOENCODING": "utf-8", "PROPAGATION_METHOD": method, "PROPAGATION_K_PROP": "0.0",
        "NUM_AGENTS": str(N), "SIM_DURATION": str(T0 + BUDGET), "K_E_TAU": f"{k_e_tau:.6f}",
        "VM_TAU_XY": str(TAU), "EXPERIMENT_SEED": str(seed),
        "COMMUNICATION_FAILURE_RATE": f"{loss:g}", "COMMUNICATION_DELAY": f"{DELAY:g}",
        "AGENT_STATE_TIMEOUT": f"{FD_TIMEOUT:g}",
        "INIT_ANGLES_EQUIDISTANT": "True", "INIT_RADIUS_RANGE": "0.0", "TARGET_MOTION_SPEED_XY": "0.0",
        "VIS_OPEN_BROWSER": "False", "SKIP_TELEMETRY_PLOTS": "True",
        "DETERMINISTIC_FAILURE_ENABLE": "True", "DETERMINISTIC_FAILURE_AGENT_ID": str(victim),
        "DETERMINISTIC_FAILURE_TIME_T0": str(T0), "DETERMINISTIC_FAILURE_OFF_TIME": "-1.0",
        "RUNS_SUMMARY_CSV_PATH": os.path.join(run_dir, "runs_summary.csv"),
    })
    if is_b2:
        env.update({"DUAL_PULSE_INTEGRATION": "B2", "DUAL_PULSE_DELTA_SCALE": "1.0",
                    "DUAL_PULSE_T_FF": f"{TAU:.6f}", "DUAL_PULSE_TTL_HOPS": str(3 * N),
                    "DUAL_PULSE_BROADCAST_REPEATS": str(repeats)})
    tag = "B2" if is_b2 else "baseline"
    print(f"  -> {tag:8s} loss={loss:<5g} delay={DELAY:g} r={repeats} s={seed} ...", flush=True)
    proc = subprocess.run([sys.executable, MAIN_PY], cwd=run_dir, env=env, capture_output=True, text=True)
    tgt = os.path.join(run_dir, "target_telemetry.csv")
    if not os.path.exists(tgt):
        print(f"     FAILED (rc={proc.returncode})\n{(proc.stderr or '')[-800:]}")
        return None
    m = metrics_from_csv(tgt)
    # Effort/saturation/fairness (M5/M6/M2) BEFORE deleting agent_telemetry.csv.
    m.update(effort_metrics(os.path.join(run_dir, "agent_telemetry.csv"), t0=T0, vmax=10.0))
    for fn in os.listdir(run_dir):
        if fn == "agent_telemetry.csv" or fn.endswith(".png"):
            try:
                os.remove(os.path.join(run_dir, fn))
            except OSError:
                pass
    m.update({"method": tag, "N": N, "tau_xy": TAU, "loss": loss, "delay": DELAY,
              "repeats": repeats, "seed": seed, "agent_state_timeout": FD_TIMEOUT})
    print(f"     egap_final={m.get('egap_final'):.4f}  settled={m.get('settled')}")
    return m


def _key(r):
    return (str(r["method"]), float(r["loss"]), round(float(r["delay"]), 6),
            int(r.get("repeats", 2)), int(r["seed"]))


def main():
    os.makedirs(RUNS_DIR, exist_ok=True)
    store = {}
    if os.path.exists(RESULTS_CSV):
        try:
            for r in pd.read_csv(RESULTS_CSV).to_dict("records"):
                store[_key(r)] = r
        except Exception:
            pass
    print(f"Comm sweep: N={N}, tau_a={TAU}, loss={LOSS_VALUES}, delay={DELAY}, repeats={REPEATS}, "
          f"methods={METHODS}, seeds={SEEDS}, budget={BUDGET}; {len(store)} ja no CSV\n")
    for loss in LOSS_VALUES:
        for repeats in REPEATS:
            for seed in SEEDS:
                for method in METHODS:
                    r = run_cell(method, loss, seed, repeats)
                    if r:
                        store[_key(r)] = r
                        pd.DataFrame(list(store.values())).to_csv(RESULTS_CSV, index=False)
    df = pd.DataFrame(list(store.values()))
    if df.empty:
        print("\nSem resultados."); return
    if "repeats" not in df.columns:
        df["repeats"] = 2

    if len(REPEATS) > 1:
        # grade B2: egap_final mediano por (loss, repeats) -> redundancia recupera a cobertura?
        print("\n=== B2: egap_final mediano por (loss x BROADCAST_REPEATS) ===")
        reps = sorted(df.repeats.unique())
        print("loss\\rep | " + " ".join(f"{r:>8d}" for r in reps))
        for loss in sorted(df.loss.unique()):
            cells = []
            for rp in reps:
                o = df[(df.method == "B2") & (df.loss == loss) & (df.repeats == rp)]
                cells.append(f"{float(o['egap_final'].median()):>8.4f}" if len(o) else f"{'-':>8}")
            print(f"{loss:>7g} | " + " ".join(cells))
        print("(abaixo de ~0,05 = reconfigurou)")
    else:
        print("\n=== ROBUSTEZ A PERDA (mediana entre seeds; egap_final) ===")
        print(f"{'loss':>5} {'egf_base':>9} {'set_base':>8} {'egf_B2':>8} {'set_B2':>7}")
        for loss in sorted(df.loss.unique()):
            b = df[(df.method == "baseline") & (df.loss == loss)]
            o = df[(df.method == "B2") & (df.loss == loss)]
            egb = float(b["egap_final"].median()) if len(b) else float("nan")
            ego = float(o["egap_final"].median()) if len(o) else float("nan")
            sb = float(b["settled"].mean()) if len(b) else float("nan")
            so = float(o["settled"].mean()) if len(o) else float("nan")
            print(f"{loss:>5g} {egb:>9.4f} {sb:>8.2f} {ego:>8.4f} {so:>7.2f}")
    print(f"\nWrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
