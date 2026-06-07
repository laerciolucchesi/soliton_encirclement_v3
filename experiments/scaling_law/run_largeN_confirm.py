#!/usr/bin/env python
"""Large-N confirmation: does Option B2's FLAT tau extend to N=100, multi-seed?

The flat-tau / complete-escape result was proven only to N=50, 1 seed, clean regime.
This confirms the central claim at larger N with multiple seeds (which node fails),
all at STABLE gain (K_E_TAU=250/N):

  B2 (full cancel, scale=1.0): N in {50,75,100} x seeds {0,1}, short budget (settles ~2s).
  baseline (stable gain):      N in {75,100} x seed 0, long budget (O(N^2) is slow).
                               (baseline N=50 is read from gain_runs/.)

PASS if: tau_B2 stays ~2.1s (FLAT) across N=50..100 and seeds, while baseline keeps O(N^2).
If something breaks at N=100 (e.g. hysteresis vs the shrinking gap), DIAGNOSE then -- do
NOT pre-fix. Discipline: run, then diagnose.

Usage:
    python experiments/scaling_law/run_largeN_confirm.py
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
RUNS = os.path.join(EXP_DIR, "largeN_runs")
GAIN_RUNS = os.path.join(EXP_DIR, "gain_runs")
RESULTS = os.path.join(EXP_DIR, "largeN_results.csv")

T0 = 5.0
GAIN_PRODUCT = 250.0
TAIL_FLOOR_FRAC = 0.05
LATE_WIN = 20.0

B2_NS = [int(x) for x in os.environ.get("LN_B2_NS", "50,75,100").split(",") if x.strip()]
B2_SEEDS = [int(x) for x in os.environ.get("LN_B2_SEEDS", "0,1").split(",") if x.strip()]
BASE_NS = [] if os.environ.get("LN_SKIP_BASE", "").strip() else [int(x) for x in os.environ.get("LN_BASE_NS", "75,100").split(",") if x.strip()]
B2_BUDGET = float(os.environ.get("LN_B2_BUDGET", "90"))
BASE_BUDGET = float(os.environ.get("LN_BASE_BUDGET", "260"))


def victim(n, seed):
    return 2 + ((n // 2 + seed) % n)


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
    return {"tau_fit": float(tau), "tau_fit_r2": float(r2),
            "egap_final": float(e[-1]), "egap_late_std": late_std}


def run(method, n, seed, budget, is_b2):
    k_e_tau = GAIN_PRODUCT / n
    name = f"{'B2' if is_b2 else 'baseline'}_N{n}_s{seed}"
    run_dir = os.path.join(RUNS, name)
    os.makedirs(run_dir, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "PYTHONIOENCODING": "utf-8",
        "PROPAGATION_METHOD": "dual_pulse" if is_b2 else "baseline",
        "PROPAGATION_K_PROP": "0.0",
        "NUM_AGENTS": str(n),
        "SIM_DURATION": str(T0 + budget),
        "K_E_TAU": f"{k_e_tau:.6f}",
        "VIS_OPEN_BROWSER": "False",
        "SKIP_TELEMETRY_PLOTS": "True",
        "INIT_ANGLES_EQUIDISTANT": "True",
        "INIT_RADIUS_RANGE": "0.0",
        "TARGET_MOTION_SPEED_XY": "0.0",
        "DETERMINISTIC_FAILURE_ENABLE": "True",
        "DETERMINISTIC_FAILURE_AGENT_ID": str(victim(n, seed)),
        "DETERMINISTIC_FAILURE_TIME_T0": str(T0),
        "DETERMINISTIC_FAILURE_OFF_TIME": "-1.0",
        "RUNS_SUMMARY_CSV_PATH": os.path.join(run_dir, "runs_summary.csv"),
    })
    if is_b2:
        env.update({
            "DUAL_PULSE_INTEGRATION": "B2",
            "DUAL_PULSE_DELTA_SCALE": "1.0",
            "DUAL_PULSE_T_FF": "1.0",
            "DUAL_PULSE_TTL_HOPS": str(3 * n),   # backstop >= N (was 50 -> truncated N>50)
        })
    print(f"  -> {name}  (K_E_TAU={k_e_tau:.2f}, dur={T0+budget:.0f}s) ...", flush=True)
    proc = subprocess.run([sys.executable, MAIN_PY], cwd=run_dir, env=env, capture_output=True, text=True)
    tgt = os.path.join(run_dir, "target_telemetry.csv")
    if not os.path.exists(tgt):
        print(f"     FAILED (rc={proc.returncode})\n{(proc.stderr or '')[-800:]}")
        return None
    m = metrics_from_csv(tgt)
    m.update({"method": "B2" if is_b2 else "baseline", "N": n, "seed": seed})
    print(f"     tau_fit={m.get('tau_fit'):.2f}  R2={m.get('tau_fit_r2'):.2f}  "
          f"egap_final={m.get('egap_final'):.4f}  late_std={m.get('egap_late_std'):.4f}")
    return m


def main():
    os.makedirs(RUNS, exist_ok=True)
    print(f"Large-N confirmation: B2 N={B2_NS} seeds={B2_SEEDS}; baseline N={BASE_NS} seed=0\n")
    rows = []
    for n in B2_NS:
        for s in B2_SEEDS:
            r = run("dual_pulse", n, s, B2_BUDGET, is_b2=True)
            if r:
                rows.append(r)
                pd.DataFrame(rows).to_csv(RESULTS, index=False)
    for n in BASE_NS:
        r = run("baseline", n, 0, BASE_BUDGET, is_b2=False)
        if r:
            rows.append(r)
            pd.DataFrame(rows).to_csv(RESULTS, index=False)

    # baselines for the comparison table: prefer largeN_runs, fall back to gain_runs.
    have_base = {int(r["N"]) for r in rows if r.get("method") == "baseline"}
    for n in sorted(set(B2_NS)):
        if n in have_base:
            continue
        for cand in (
            os.path.join(RUNS, f"baseline_N{n}_s0", "target_telemetry.csv"),
            os.path.join(GAIN_RUNS, f"baseline_N{n}", "target_telemetry.csv"),
        ):
            m = metrics_from_csv(cand)
            if m:
                m.update({"method": "baseline", "N": n, "seed": 0})
                rows.append(m)
                break
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS, index=False)

    print("\n=== CONFIRMATION ===")
    print(f"{'method':>8} {'N':>4} {'seed':>4} {'tau_fit':>8} {'R2':>5} {'late_std':>9}  stability")
    for _, r in df.sort_values(["method", "N", "seed"]).iterrows():
        stable = "UNSTABLE" if (r["egap_late_std"] > 0.02 or r["egap_final"] > 0.05) else "settled"
        print(f"{r['method']:>8} {int(r['N']):>4} {int(r['seed']):>4} {r['tau_fit']:>8.2f} "
              f"{r['tau_fit_r2']:>5.2f} {r['egap_late_std']:>9.4f}  {stable}")

    b2 = df[df.method == "B2"]
    base = df[df.method == "baseline"]
    if len(b2):
        print(f"\nB2 tau_fit: min={b2['tau_fit'].min():.2f}  max={b2['tau_fit'].max():.2f}  "
              f"-> {'FLAT (escape holds)' if (b2['tau_fit'].max() < 4.0) else 'NOT flat -- investigate'}")
    if len(base) >= 2:
        bb = base.groupby('N')['tau_fit'].median()
        Ns = bb.index.to_numpy(float); ys = bb.to_numpy(float)
        m_ok = np.isfinite(ys) & (ys > 0)
        if m_ok.sum() >= 2:
            p = np.polyfit(np.log(Ns[m_ok]), np.log(ys[m_ok]), 1)[0]
            print(f"baseline tau ~ N^{p:.2f}  (esperado ~2.0)")
    print(f"\nWrote {RESULTS}")


if __name__ == "__main__":
    main()
