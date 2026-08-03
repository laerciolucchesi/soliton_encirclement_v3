#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Item 2: `alive_count` esta DEFASADO? E, se estiver, o teorema fecha em 1,000?

Candidato NOMEADO para o residuo de +1,2% na razao E[pico]/(2(M-1)/M):
a referencia usa M = alive_count, que e' o que o ALVO enxerga, e a deteccao tem
~0,30 s de latencia (AGENT_STATE_TIMEOUT = 5*dt = 0,25 s + grade de 0,05 s).
Durante esses 0,30 s o alvo ainda CONTA um agente que ja morreu. Com k contados a
mais, a media verdadeira e' 2*pi/(M-k), nao 2*pi/M.

Este script reconstroi o numero VERDADEIRO de agentes nao-falhados a partir de
events.csv (failure_start / failure_end), compara com alive_count amostra a
amostra, e RECALCULA a razao do teorema com o M verdadeiro.

Tambem responde 3.1: onde estao as violacoes de S16.

Uso:
    python experiments/scaling_law/analysis_churn/probe_alive_lag.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(HERE)
RUNS = os.path.join(HERE, "rerun_runs")
N_NOM = 24
T0 = 20.0
_log = []


def say(m=""):
    print(m)
    _log.append(str(m))


def true_alive(ts, ev, n_nom=N_NOM):
    """Numero VERDADEIRO de agentes nao-falhados em cada timestamp, de events.csv."""
    fs = ev[ev.event_type == "failure_start"]["timestamp"].to_numpy(float)
    fe = ev[ev.event_type == "failure_end"]["timestamp"].to_numpy(float)
    # contagem por diferenca: +1 down em cada failure_start, -1 em cada failure_end
    down = (np.searchsorted(np.sort(fs), ts, side="right")
            - np.searchsorted(np.sort(fe), ts, side="right"))
    return float(n_nom) - down


def main():
    if not os.path.isdir(RUNS):
        sys.exit(f"{RUNS} nao existe.")
    say("=" * 78)
    say("ITEM 2 -- alive_count esta defasado? O teorema fecha com o M verdadeiro?")
    say("=" * 78)

    rows, per_cell = [], []
    for d in sorted(os.listdir(RUNS)):
        tgt = os.path.join(RUNS, d, "target_telemetry.csv")
        evp = os.path.join(RUNS, d, "events.csv")
        if not (os.path.exists(tgt) and os.path.exists(evp)):
            continue
        meth = "B2" if d.startswith("dual_pulse") else "baseline"
        rate = float(d.split("_rate")[1].split("_s")[0])
        seed = int(d.split("_s")[-1])
        tel = pd.read_csv(tgt)
        ev = pd.read_csv(evp)
        t = tel["timestamp"].to_numpy(float)
        m = t >= T0
        ac = tel["alive_count"].to_numpy(float)[m]
        tv = true_alive(t[m], ev)
        dif = ac - tv                      # >0 = alvo conta a MAIS (obsoleto)
        per_cell.append({
            "method": meth, "rate_total": rate, "seed": seed, "n": int(m.sum()),
            "frac_dif": float(np.mean(dif != 0)), "dif_media": float(np.mean(dif)),
            "frac_conta_a_mais": float(np.mean(dif > 0)),
            "dif_max": float(np.max(np.abs(dif))),
        })
        rows.append(pd.DataFrame({"timestamp": t[m], "alive_count": ac,
                                  "true_alive": tv, "dif": dif,
                                  "method": meth, "rate_total": rate, "seed": seed}))
    pc = pd.DataFrame(per_cell)
    say("")
    say(f"{'taxa':>6}{'celulas':>9}{'frac amostras diferentes':>26}{'dif media':>12}"
        f"{'frac conta a MAIS':>19}{'|dif| max':>11}")
    for r in sorted(pc.rate_total.unique()):
        s = pc[pc.rate_total == r]
        say(f"{r:>6g}{len(s):>9}{s.frac_dif.mean():>26.4f}{s.dif_media.mean():>12.4f}"
            f"{s.frac_conta_a_mais.mean():>19.4f}{s.dif_max.max():>11.0f}")
    say("")
    say("dif = alive_count - verdadeiro. dif > 0 = o alvo ainda conta quem ja morreu.")

    # ---- recalculo do teorema com o M VERDADEIRO --------------------------
    say("")
    say("=" * 78)
    say("RECALCULO do teorema com o M VERDADEIRO")
    say("=" * 78)
    evd = pd.read_csv(os.path.join(HERE, "gmax_events.csv"))
    tel_cache = {}
    true_pre, true_peak = [], []
    for (meth, rate, seed), g in evd.groupby(["method", "rate_total", "seed"]):
        d = ("dual_pulse" if meth == "B2" else "baseline") + f"_rate{rate:g}_s{seed}"
        key = (meth, rate, seed)
        if key not in tel_cache:
            tel_cache[key] = (pd.read_csv(os.path.join(RUNS, d, "target_telemetry.csv")),
                              pd.read_csv(os.path.join(RUNS, d, "events.csv")))
        tel, ev = tel_cache[key]
        t = tel["timestamp"].to_numpy(float)
        # instante da amostra 'pre' e do 'pico', reconstruidos do t_fail + lag
        t_pre = g["t_fail"].to_numpy(float) - 1e-9
        t_pk = g["t_fail"].to_numpy(float) + g["t_peak_rel"].to_numpy(float)
        true_pre.append(true_alive(t_pre, ev))
        true_peak.append(true_alive(t_pk, ev))
    evd["true_alive_pre"] = np.concatenate(true_pre)
    evd["true_alive_peak"] = np.concatenate(true_peak)
    evd["lag_pre"] = evd["alive_pre"] - evd["true_alive_pre"]
    evd["lag_peak"] = evd["alive_peak"] - evd["true_alive_peak"]
    say(f"defasagem na amostra PRE : media {evd.lag_pre.mean():+.4f}, "
        f"frac != 0 {np.mean(evd.lag_pre != 0):.4f}")
    say(f"defasagem no instante PICO: media {evd.lag_peak.mean():+.4f}, "
        f"frac != 0 {np.mean(evd.lag_peak != 0):.4f}")

    def cluster_mean(y, groups):
        y = np.asarray(y, float)
        ok = np.isfinite(y)
        y, groups = y[ok], np.asarray(groups)[ok]
        gs = np.unique(groups)
        n, G = y.size, gs.size
        mu = y.mean()
        u = y - mu
        meat = sum(u[groups == gg].sum() ** 2 for gg in gs)
        var = meat / (n ** 2) * (G / (G - 1.0)) * ((n - 1.0) / (n - 1.0))
        se = np.sqrt(var)
        tc = stats.t.ppf(0.975, G - 1)
        return mu, se, mu - tc * se, mu + tc * se, n

    grp = (evd["rate_total"].astype(str) + "_" + evd["seed"].astype(str)).to_numpy()
    say("")
    say(f"{'referencia usada':<44}{'media':>9}{'se_cl':>9}{'IC95 lo':>10}{'IC95 hi':>10}{'n':>7}")
    variants = [
        ("M do alvo (alive_count no pico)  [atual]",
         evd["gmax_peak"] / (2.0 * (evd["alive_peak"] - 1) / evd["alive_peak"])),
        ("vao [rad] / (4pi/M_alvo)",
         evd["gap_rad_peak"] / (4.0 * np.pi / evd["alive_pre"])),
        ("vao [rad] / (4pi/M_VERDADEIRO)",
         evd["gap_rad_peak"] / (4.0 * np.pi / evd["true_alive_pre"])),
    ]
    for lab, y in variants:
        mu, se, lo, hi, n = cluster_mean(y, grp)
        um = "CONTEM 1" if lo <= 1.0 <= hi else "nao contem 1"
        say(f"{lab:<44}{mu:>9.5f}{se:>9.5f}{lo:>10.5f}{hi:>10.5f}{n:>7}  {um}")

    # ---- 3.1: onde estao as violacoes de S16 ------------------------------
    say("")
    say("=" * 78)
    say("ITEM 3.1 -- onde estao as violacoes de S16?")
    say("=" * 78)
    lb = evd["gmax_pre"] * (evd["alive_pre"] - 1.0) / evd["alive_pre"]
    evd["viol"] = evd["gmax_peak"] < lb
    evd["mag"] = (lb - evd["gmax_peak"]) / lb
    v = evd[evd.viol]
    say(f"violacoes: {len(v)} de {len(evd)} ({len(v)/len(evd):.4%})")
    if len(v):
        say(f"magnitude: mediana {v.mag.median():.3%}, max {v.mag.max():.3%}")
        say("")
        say("contingencia (contagem de violacoes / total do balde):")
        for col in ("rate_total", "method", "m_unstable", "w_clip_floor", "peak_late"):
            if col not in evd.columns:
                continue
            say(f"  por {col}:")
            for val in sorted(evd[col].unique()):
                tot = int((evd[col] == val).sum())
                nv = int(((evd[col] == val) & evd.viol).sum())
                say(f"     {str(val):<10} {nv:>3}/{tot:<6} ({nv/max(tot,1):.4%})")
        say("")
        say("  amostra das violacoes:")
        say(v[["method", "rate_total", "seed", "t_fail", "gmax_pre", "gmax_peak",
               "alive_pre", "w_e", "mag"]].to_string(index=False, max_rows=10))
    with open(os.path.join(HERE, "LOG_alive_lag.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(_log) + "\n")
    evd.to_csv(os.path.join(HERE, "gmax_events_truealive.csv"), index=False)
    say("\nEscrito: gmax_events_truealive.csv, LOG_alive_lag.txt")


if __name__ == "__main__":
    main()
