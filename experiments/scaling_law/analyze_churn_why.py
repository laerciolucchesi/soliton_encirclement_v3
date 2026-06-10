#!/usr/bin/env python
"""POR QUE o overlay falha sob churn? Analise visual + medidas, a partir dos dados.

Parte 1 (dados existentes): le churn_sweep_results{,_stamp,_gated}.csv e quantifica
  - vantagem = egap_base/egap_B2 vs taxa (ajuda>1 / atrapalha<1), p/ cada config
  - inflacao de cauda = egap_p90/egap_avg (baseline vs B2): assinatura de OSCILACAO/picos

Parte 2 (mecanismo): roda baseline+B2 em taxa ESPARSA(6) e DENSA(48) MANTENDO telemetria, e mede a
hipotese central -- SOBREPOSICAO de eventos:
  - 'ocupacao' do overlay = nº de nos com |dual_pulse_shift|>eps ao longo do tempo. Esparso: volta a
    ZERO entre eventos (overlay TERMINA -> coerente). Denso: fica continuamente alto (eventos novos
    chegam antes de consumir o shift anterior -> superposicao INCOERENTE).
  - esforco de controle (velocity_norm) baseline vs B2 (overlay agita?).
  - egap(t) baseline vs B2.

Saida: churn_why.png (4 paineis) + medidas no stdout.

Uso: python experiments/scaling_law/analyze_churn_why.py   # env: WHY_SEED=1 WHY_BUDGET=80
"""
import os
import sys
import subprocess

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))
MAIN_PY = os.path.join(REPO_ROOT, "main.py")
RUNS_DIR = os.path.join(EXP_DIR, "churn_why_runs")

N = 24
TAU = 1.0
T0 = 5.0
WARMUP = 15.0
OFF = 8.0
SEED = int(os.environ.get("WHY_SEED", "1"))
BUDGET = float(os.environ.get("WHY_BUDGET", "80"))
EPS_SHIFT = 0.01            # rad; mesmo DUAL_PULSE_SLEEP_THRESHOLD (overlay "ativo")
RATES = [6.0, 48.0]        # esparso, denso


# ---------- Parte 1: dados existentes ----------
def load_sweep(tag):
    p = os.path.join(EXP_DIR, f"churn_sweep_results{('_'+tag) if tag else ''}.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


def advantage_table(df):
    """mediana por taxa -> vantagem e inflacao de cauda."""
    out = {}
    for rate in sorted(df.rate_total.unique()):
        b = df[(df.method == "baseline") & (df.rate_total == rate)]
        o = df[(df.method == "B2") & (df.rate_total == rate)]
        def med(x, c): return float(x[c].median()) if len(x) and c in x else float("nan")
        eb, eo = med(b, "egap_avg"), med(o, "egap_avg")
        out[rate] = {
            "egap_base": eb, "egap_B2": eo,
            "adv": (eb / eo) if (eo and np.isfinite(eo)) else float("nan"),
            "tail_base": med(b, "egap_p90") / eb if eb else float("nan"),
            "tail_B2": med(o, "egap_p90") / eo if eo else float("nan"),
        }
    return out


# ---------- Parte 2: mecanismo (re-run com telemetria) ----------
def run_cell(method, rate_total):
    is_b2 = (method == "dual_pulse")
    per_agent = rate_total / float(N)
    run_dir = os.path.join(RUNS_DIR, f"{method}_rate{rate_total:g}")
    os.makedirs(run_dir, exist_ok=True)
    for fn in ("target_telemetry.csv", "agent_telemetry.csv", "events.csv"):
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
        "FAILURE_MEAN_FAILURES_PER_MIN": f"{per_agent:.6f}", "FAILURE_OFF_TIME": f"{OFF:g}",
        "EXPERIMENT_SEED": str(SEED),
        "RUNS_SUMMARY_CSV_PATH": os.path.join(run_dir, "runs_summary.csv"),
        "DUAL_PULSE_GATE_ENABLE": "False",
    })
    if is_b2:
        env.update({"DUAL_PULSE_INTEGRATION": "B2", "DUAL_PULSE_DELTA_SCALE": "1.0",
                    "DUAL_PULSE_T_FF": f"{TAU:.6f}", "DUAL_PULSE_TTL_HOPS": str(3 * N)})
    print(f"  rodando {method:10s} rate={rate_total:g}/min ...", flush=True)
    subprocess.run([sys.executable, MAIN_PY], cwd=run_dir, env=env, capture_output=True, text=True)
    return run_dir


def extract(run_dir):
    """Le telemetria -> series agregadas; depois APAGA o agent_telemetry (disco)."""
    tgt = os.path.join(run_dir, "target_telemetry.csv")
    ag = os.path.join(run_dir, "agent_telemetry.csv")
    ev = os.path.join(run_dir, "events.csv")
    res = {}
    if os.path.exists(tgt):
        t = pd.read_csv(tgt)
        res["egap_t"] = t["timestamp"].to_numpy(float)
        res["egap"] = t["E_gap"].to_numpy(float)
        steady = t[t["timestamp"] >= T0 + WARMUP]["E_gap"].to_numpy(float)
        steady = steady[np.isfinite(steady)]
        res["egap_avg"] = float(np.mean(steady)) if steady.size else float("nan")
        res["egap_p90"] = float(np.percentile(steady, 90)) if steady.size else float("nan")
    if os.path.exists(ag):
        a = pd.read_csv(ag)
        a = a[a["timestamp"] >= T0]
        # ocupacao: nº de nos com |shift|>eps por timestamp; esforco: velocity_norm medio
        g = a.groupby("timestamp")
        sh = a.assign(active=(a["dual_pulse_shift"].abs() > EPS_SHIFT).astype(int))
        occ = sh.groupby("timestamp")["active"].sum()
        res["occ_t"] = occ.index.to_numpy(float)
        res["occ"] = occ.to_numpy(float)
        vel = g["velocity_norm"].mean()
        res["vel_t"] = vel.index.to_numpy(float)
        res["vel"] = vel.to_numpy(float)
        steady_mask = a["timestamp"] >= T0 + WARMUP
        res["vel_avg"] = float(a[steady_mask]["velocity_norm"].mean())
        occ_steady = occ[occ.index >= T0 + WARMUP]
        res["busy_frac"] = float((occ_steady > 0).mean()) if len(occ_steady) else float("nan")
        res["occ_mean"] = float(occ_steady.mean()) if len(occ_steady) else float("nan")
        try:
            os.remove(ag)
        except OSError:
            pass
    if os.path.exists(ev):
        e = pd.read_csv(ev)
        et = e["event_type"].astype(str)
        res["n_fail"] = int(et.str.contains("failure_start").sum())
        res["n_inj"] = int(et.str.contains("self_shift").sum())
        res["fail_times"] = e[et.str.contains("failure_start")]["timestamp"].to_numpy(float)
        res["inj_times"] = e[et.str.contains("self_shift")]["timestamp"].to_numpy(float)
    return res


def main():
    os.makedirs(RUNS_DIR, exist_ok=True)
    # ---- Parte 1 ----
    cfgs = {"original": load_sweep(""), "estampa(M2)": load_sweep("stamp"), "gated": load_sweep("gated")}
    adv = {k: advantage_table(v) for k, v in cfgs.items() if v is not None}
    print("\n=== PARTE 1 (dados existentes): vantagem e inflacao de cauda por taxa ===")
    for cfg, tab in adv.items():
        print(f"\n[{cfg}]")
        print(f"  {'taxa':>5} {'egap_base':>10} {'egap_B2':>9} {'vantagem':>9} {'cauda_base':>11} {'cauda_B2':>9}")
        for rate, d in tab.items():
            print(f"  {rate:>5g} {d['egap_base']:>10.4f} {d['egap_B2']:>9.4f} {d['adv']:>9.2f} "
                  f"{d['tail_base']:>11.2f} {d['tail_B2']:>9.2f}")

    # ---- Parte 2 ----
    print("\n=== PARTE 2 (mecanismo): re-run com telemetria (seed={}, budget={}s) ===".format(SEED, BUDGET))
    data = {}
    for rate in RATES:
        for method in ("baseline", "dual_pulse"):
            data[(method, rate)] = extract(run_cell(method, rate))
    print(f"\n  {'cel':22s} {'egap_avg':>9} {'egap_p90':>9} {'busy_frac':>10} {'occ_mean':>9} {'vel_avg':>8} {'n_fail':>7} {'n_inj':>6}")
    for (method, rate), d in data.items():
        tag = ("B2" if method == "dual_pulse" else "baseline") + f" r{rate:g}"
        print(f"  {tag:22s} {d.get('egap_avg', float('nan')):>9.4f} {d.get('egap_p90', float('nan')):>9.4f} "
              f"{d.get('busy_frac', float('nan')):>10.2f} {d.get('occ_mean', float('nan')):>9.2f} "
              f"{d.get('vel_avg', float('nan')):>8.4f} {d.get('n_fail', 0):>7d} {d.get('n_inj', 0):>6d}")

    # ---- Figura (4 paineis) ----
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    # A: egap(t) baseline vs B2 no DENSO (48)
    dense = 48.0
    b, o = data[("baseline", dense)], data[("dual_pulse", dense)]
    if "egap" in b and "egap" in o:
        ax[0, 0].plot(b["egap_t"], b["egap"], lw=0.8, color="#1f77b4", label="baseline")
        ax[0, 0].plot(o["egap_t"], o["egap"], lw=0.8, color="#d62728", label="B2 (overlay)")
        for ft in o.get("inj_times", [])[:200]:
            ax[0, 0].axvline(ft, color="#d62728", alpha=0.06)
        ax[0, 0].set_title(f"A) egap(t) no DENSO ({dense:g}/min): B2 oscila ACIMA do baseline")
        ax[0, 0].set_xlabel("t (s)"); ax[0, 0].set_ylabel("E_gap"); ax[0, 0].legend(fontsize=8)

    # B: ocupacao do overlay (nº nos ativos) ESPARSO vs DENSO -- a sobreposicao
    sp, dn = data[("dual_pulse", 6.0)], data[("dual_pulse", 48.0)]
    if "occ" in sp:
        ax[0, 1].plot(sp["occ_t"], sp["occ"], lw=0.8, color="#2ca02c",
                      label=f"esparso (6/min) busy={sp.get('busy_frac', float('nan')):.0%}")
    if "occ" in dn:
        ax[0, 1].plot(dn["occ_t"], dn["occ"], lw=0.8, color="#d62728",
                      label=f"denso (48/min) busy={dn.get('busy_frac', float('nan')):.0%}")
    ax[0, 1].set_title("B) Ocupacao do overlay = nº de drones redistribuindo (|shift|>eps)")
    ax[0, 1].set_xlabel("t (s)"); ax[0, 1].set_ylabel("nº nos ativos")
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].text(0.5, 0.92, "esparso volta a ZERO (coerente) | denso fica cheio (superposicao)",
                  transform=ax[0, 1].transAxes, ha="center", fontsize=8, color="#555")

    # C: vantagem vs taxa (3 configs)
    for cfg, tab in adv.items():
        rates = sorted(tab.keys())
        ax[1, 0].plot(rates, [tab[r]["adv"] for r in rates], "o-", label=cfg)
    ax[1, 0].axhline(1.0, color="k", ls="--", lw=0.8)
    ax[1, 0].set_title("C) Vantagem = egap_base/egap_B2 vs taxa (>1 ajuda; <1 atrapalha)")
    ax[1, 0].set_xlabel("taxa de churn (/min)"); ax[1, 0].set_ylabel("vantagem")
    ax[1, 0].legend(fontsize=8)

    # D: inflacao de cauda (egap_p90/egap_avg) baseline vs B2, config original
    tab0 = adv.get("original", {})
    if tab0:
        rates = sorted(tab0.keys())
        ax[1, 1].plot(rates, [tab0[r]["tail_base"] for r in rates], "o-", color="#1f77b4", label="baseline")
        ax[1, 1].plot(rates, [tab0[r]["tail_B2"] for r in rates], "o-", color="#d62728", label="B2")
        ax[1, 1].set_title("D) Inflacao de cauda = egap_p90/egap_avg (picos/oscilacao)")
        ax[1, 1].set_xlabel("taxa de churn (/min)"); ax[1, 1].set_ylabel("p90 / avg")
        ax[1, 1].legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(EXP_DIR, "churn_why.png")
    fig.savefig(out, dpi=110)
    print(f"\nFigura: {out}")


if __name__ == "__main__":
    main()
