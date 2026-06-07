#!/usr/bin/env python
"""Prepare a clean, consistent dataset for the advisor figures.

For each N in {24,40,50,75,100} and each method in {baseline, B2}, we want ONE
target_telemetry.csv produced by the IDENTICAL controlled-crash protocol
(single permanent crash of node 2+N//2 at t0=5, stable gain K_E_TAU=250/N,
equidistant init, stationary target), so baseline and B2 are apples-to-apples.

  - baseline telemetry is REUSED from the existing runs (re-running the O(N^2)
    baseline at N=100 costs ~20 min; no need -- the data is already on disk and
    clean). Candidate paths are tried in order.
  - B2 telemetry is (re)generated here: B2 settles in ~2 s, so a short budget is
    enough and all 5 N run in a couple of minutes total.

Outputs:
  figure_runs/B2_N{n}/target_telemetry.csv   (fresh B2 runs)
  figure_data.csv  with columns: method,N,tau_fit,tau_fit_r2,egap_final,
                                 egap_late_std,telemetry_path,dt_telem

The tau_fit metric is byte-for-byte the same exp-fit used in
run_baseline_longbudget.py / run_largeN_confirm.py (peak -> 5% floor, t>=t0).
"""
import os
import sys
import subprocess

import numpy as np
import pandas as pd

_THIS = os.path.abspath(__file__)
EXP_DIR = os.path.dirname(_THIS)
REPO_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))
MAIN_PY = os.path.join(REPO_ROOT, "main.py")

FIG_RUNS = os.path.join(EXP_DIR, "figure_runs")
GAIN_RUNS = os.path.join(EXP_DIR, "gain_runs")
BASELINE_LONG = os.path.join(EXP_DIR, "baseline_long_runs")
LARGEN_RUNS = os.path.join(EXP_DIR, "largeN_runs")
OUT_CSV = os.path.join(EXP_DIR, "figure_data.csv")

NS = [int(x) for x in os.environ.get("FIG_NS", "24,40,50,75,100").split(",") if x.strip()]
T0 = 5.0
GAIN_PRODUCT = 250.0
TAIL_FLOOR_FRAC = 0.05
LATE_WIN = 20.0
B2_BUDGET = float(os.environ.get("FIG_B2_BUDGET", "40"))  # plenty: B2 settles ~2 s


def victim(n):
    return 2 + (n // 2)


def metrics_from_csv(tgt):
    if not os.path.exists(tgt):
        return {}
    df = pd.read_csv(tgt)
    full = df.copy()
    df = df[df["timestamp"] >= T0].reset_index(drop=True)
    if df.empty:
        return {}
    t = df["timestamp"].to_numpy(float)
    e = df["E_gap"].to_numpy(float)
    i_pk = int(np.nanargmax(e))
    e_pk = e[i_pk]
    floor = TAIL_FLOOR_FRAC * e_pk
    mask = (np.arange(e.size) >= i_pk) & (e > floor) & np.isfinite(e)
    tau, r2 = np.nan, np.nan
    if mask.sum() >= 5:
        tt, ee = t[mask], e[mask]
        coef = np.polyfit(tt, np.log(ee), 1)
        if coef[0] < 0:
            tau = -1.0 / coef[0]
        pred = np.polyval(coef, tt)
        ss_res = float(np.sum((np.log(ee) - pred) ** 2))
        ss_tot = float(np.sum((np.log(ee) - np.mean(np.log(ee))) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    t_end = float(full["timestamp"].max())
    late = full[full["timestamp"] >= t_end - LATE_WIN]["E_gap"].to_numpy(float)
    late_std = float(np.std(late)) if late.size else np.nan
    ts = full["timestamp"].to_numpy(float)
    dt_telem = float(np.median(np.diff(ts))) if ts.size > 1 else float("nan")
    return {"tau_fit": float(tau), "tau_fit_r2": float(r2),
            "egap_final": float(e[-1]), "egap_late_std": late_std,
            "dt_telem": dt_telem}


def baseline_telemetry_path(n):
    for cand in (
        os.path.join(GAIN_RUNS, f"baseline_N{n}", "target_telemetry.csv"),
        os.path.join(BASELINE_LONG, f"baseline_N{n}", "target_telemetry.csv"),
        os.path.join(LARGEN_RUNS, f"baseline_N{n}_s0", "target_telemetry.csv"),
    ):
        if os.path.exists(cand):
            return cand
    return None


def run_b2(n):
    k = GAIN_PRODUCT / n
    run_dir = os.path.join(FIG_RUNS, f"B2_N{n}")
    os.makedirs(run_dir, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "PYTHONIOENCODING": "utf-8",
        "PROPAGATION_METHOD": "dual_pulse",
        "PROPAGATION_K_PROP": "0.0",
        "NUM_AGENTS": str(n),
        "SIM_DURATION": str(T0 + B2_BUDGET),
        "K_E_TAU": f"{k:.6f}",
        "VIS_OPEN_BROWSER": "False",
        "SKIP_TELEMETRY_PLOTS": "True",
        "INIT_ANGLES_EQUIDISTANT": "True",
        "INIT_RADIUS_RANGE": "0.0",
        "TARGET_MOTION_SPEED_XY": "0.0",
        "DETERMINISTIC_FAILURE_ENABLE": "True",
        "DETERMINISTIC_FAILURE_AGENT_ID": str(victim(n)),
        "DETERMINISTIC_FAILURE_TIME_T0": str(T0),
        "DETERMINISTIC_FAILURE_OFF_TIME": "-1.0",
        "DUAL_PULSE_INTEGRATION": "B2",
        "DUAL_PULSE_DELTA_SCALE": "1.0",
        "DUAL_PULSE_T_FF": "1.0",
        "DUAL_PULSE_TTL_HOPS": str(3 * n),
        "RUNS_SUMMARY_CSV_PATH": os.path.join(run_dir, "runs_summary.csv"),
    })
    print(f"  -> B2 N={n}  (K_E_TAU={k:.2f}, dur={T0 + B2_BUDGET:.0f}s) ...", flush=True)
    proc = subprocess.run([sys.executable, MAIN_PY], cwd=run_dir, env=env,
                          capture_output=True, text=True)
    tgt = os.path.join(run_dir, "target_telemetry.csv")
    if not os.path.exists(tgt):
        print(f"     FAILED (rc={proc.returncode})\n{(proc.stderr or '')[-800:]}")
        return None
    return tgt


def main():
    os.makedirs(FIG_RUNS, exist_ok=True)
    rows = []
    print(f"Preparing figure data for N={NS}\n")
    print("[baseline] reusing existing telemetry:")
    for n in NS:
        p = baseline_telemetry_path(n)
        if not p:
            print(f"  N={n}: NO baseline telemetry found -- SKIP")
            continue
        m = metrics_from_csv(p)
        m.update({"method": "baseline", "N": n, "telemetry_path": p})
        rows.append(m)
        print(f"  N={n}: tau={m['tau_fit']:.2f} R2={m['tau_fit_r2']:.2f}  ({os.path.relpath(p, EXP_DIR)})")

    print("\n[B2] regenerating clean telemetry:")
    for n in NS:
        tgt = run_b2(n)
        if not tgt:
            continue
        m = metrics_from_csv(tgt)
        m.update({"method": "B2", "N": n, "telemetry_path": tgt})
        rows.append(m)
        print(f"     tau={m['tau_fit']:.2f} R2={m['tau_fit_r2']:.2f} "
              f"egap_final={m['egap_final']:.4f} late_std={m['egap_late_std']:.4f}")

    df = pd.DataFrame(rows)[["method", "N", "tau_fit", "tau_fit_r2",
                             "egap_final", "egap_late_std", "dt_telem",
                             "telemetry_path"]]
    df = df.sort_values(["method", "N"]).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}\n")

    # quick scaling summary
    for method in ("baseline", "B2"):
        g = df[df.method == method].sort_values("N")
        Ns = g["N"].to_numpy(float)
        ys = g["tau_fit"].to_numpy(float)
        ok = np.isfinite(Ns) & np.isfinite(ys) & (Ns > 0) & (ys > 0)
        p = (np.polyfit(np.log(Ns[ok]), np.log(ys[ok]), 1)[0]
             if ok.sum() >= 2 else float("nan"))
        print(f"  {method:>8}: tau ~ N^{p:.2f}   tau="
              + " / ".join(f"{v:.2f}" for v in ys))


if __name__ == "__main__":
    main()
