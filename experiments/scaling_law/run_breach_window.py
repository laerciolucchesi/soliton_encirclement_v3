#!/usr/bin/env python
"""E4 deciding experiment: is the BREACH WINDOW actuation-limited or coordination-limited?

Question. The churn re-analysis (docs/experiments/CHURN_PAIRED.md) showed the overlay
improves the MEAN spacing error unanimously but shows no reliable effect on the tail --
and that the campaign never measured the mission-critical quantity at all (the maximum
angular gap). The conjecture to decide: after a failure the max gap jumps to ~2x ideal and
closing it is limited by Vmax and tau_a, not by coordination -> a protocol-independent floor.

Design. Deliberately NOT a churn sweep: under continuous churn, concurrent deaths
contaminate the peak (measured: peak G_max climbs 2.11 -> 3.49 from rate 6 to 48). A single
DETERMINISTIC permanent failure isolates one event, so the peak and the closing time are
both attributable to it.

  fixed  : N=24, one permanent victim at t=5 s, uniform initial ring, loss=0, delay=0,
           dt=0.05, K_E_TAU=250/N, B2 knobs = the validated config, TTL=3N
  swept  : VM_MAX_SPEED_XY x VM_TAU_XY x {baseline, dual_pulse} x seeds
           (T_FF follows tau_a, the approved actuation-matched rule)
  measure: peak G_max, peak gap in degrees, t_close (when the gap stops being wide),
           breach AREA (integral of the excess), all anchored on the REAL failure
           timestamp read from events.csv -- not on the nominal t=5.

What decides what (see CHURN_PAIRED.md 5.3):
  peak G_max ~ 2(N-1)/N for both, flat in Vmax/tau_a   -> peak floor is geometric, confirmed
  t_close ~ equal for both and scaling with tau_a       -> CONJECTURE CONFIRMED (actuation-limited)
  t_close flat for B2 while baseline grows              -> CONJECTURE REFUTED (coordination matters)
  t_close worse for B2                                  -> feedforward overshoot, a real defect

Requires the alive_count / gap_max_rad telemetry columns (commit e062eed onward).

Usage:
    python experiments/scaling_law/run_breach_window.py
    # env: BREACH_VMAX="2.5,5,10,20"  BREACH_TAUS="0.5,1.0,2.0"  BREACH_SEEDS="0,1,2,3,4"
    #      BREACH_N="24"  BREACH_BUDGET="90"  BREACH_TAG=""  BREACH_METHODS="baseline,dual_pulse"
"""
import os
import sys
import subprocess

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_util import run_provenance  # noqa: E402

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))
MAIN_PY = os.path.join(REPO_ROOT, "main.py")

_TAG = os.environ.get("BREACH_TAG", "")
_SUF = ("_" + _TAG) if _TAG else ""
RUNS_DIR = os.path.join(EXP_DIR, "breach_runs" + _SUF)
RESULTS_CSV = os.path.join(EXP_DIR, "breach_window_results" + _SUF + ".csv")

VMAXES = [float(x) for x in os.environ.get("BREACH_VMAX", "2.5,5,10,20").split(",") if x.strip()]
TAUS = [float(x) for x in os.environ.get("BREACH_TAUS", "0.5,1.0,2.0").split(",") if x.strip()]
SEEDS = [int(x) for x in os.environ.get("BREACH_SEEDS", "0,1,2,3,4").split(",") if x.strip()]
METHODS = [m.strip() for m in os.environ.get("BREACH_METHODS", "baseline,dual_pulse").split(",") if m.strip()]
N = int(os.environ.get("BREACH_N", "24"))
BUDGET = float(os.environ.get("BREACH_BUDGET", "90"))
DT = float(os.environ.get("CONTROL_PERIOD", "0.05"))
T_FAIL_NOMINAL = float(os.environ.get("BREACH_T_FAIL", "5.0"))

# Breach thresholds, as multiples of the ideal gap for the CURRENT alive count.
# 1.25 is the primary (a gap 25% wider than it should be); 1.10 is the strict check.
THR_PRIMARY = 1.25
THR_STRICT = 1.10


def victim_node_id(n, seed=0):
    """Agents occupy ids 2..N+1; vary the victim with the seed (symmetry check)."""
    return 2 + ((n // 2 + seed) % n)


def failure_time(run_dir, fallback):
    """The REAL failure instant, from events.csv. Falls back to the nominal t0."""
    p = os.path.join(run_dir, "events.csv")
    if not os.path.exists(p):
        return fallback
    try:
        ev = pd.read_csv(p)
    except Exception:
        return fallback
    if "event_type" not in ev.columns:
        return fallback
    f = ev[ev["event_type"] == "failure_start"]
    return float(f["timestamp"].min()) if len(f) else fallback


def close_time(t, g, thr):
    """Time after t[0] at which G_max stops exceeding `thr` for good.

    "Stops for good", not "first crosses": a transient dip below the threshold on the
    way down would make a first-crossing definition report a breach as closed while it
    is still open. inf when it never closes within the budget.
    """
    above = g > thr
    if not above.any():
        return 0.0
    last = int(np.max(np.where(above)[0]))
    if last >= g.size - 1:
        return float("inf")
    return float(t[last + 1] - t[0])


def metrics_from_run(run_dir):
    tgt = os.path.join(run_dir, "target_telemetry.csv")
    if not os.path.exists(tgt):
        return {}
    cols = ["timestamp", "G_max", "E_gap", "alive_count", "gap_max_rad"]
    try:
        df = pd.read_csv(tgt, usecols=cols)
    except ValueError:
        return {"error": "telemetry lacks alive_count/gap_max_rad (pre-e062eed run)"}
    t_fail = failure_time(run_dir, T_FAIL_NOMINAL)
    post = df[df["timestamp"] >= t_fail].reset_index(drop=True)
    if post.empty:
        return {}
    t = post["timestamp"].to_numpy(float)
    g = post["G_max"].to_numpy(float)
    rad = post["gap_max_rad"].to_numpy(float)
    alive = post["alive_count"].to_numpy(float)

    # Breach AREA: integral of the excess above the threshold. Unlike the peak (one
    # instant) and t_close (one crossing), this weights how wide AND how long.
    excess = np.maximum(0.0, g - THR_PRIMARY)
    area = float(np.trapezoid(excess, t)) if t.size > 1 else 0.0

    i_pk = int(np.nanargmax(g))
    alive_end = float(alive[-1]) if alive.size else float("nan")
    return {
        "t_fail": t_fail,
        "gmax_peak": float(g[i_pk]),
        "gap_peak_deg": float(np.degrees(rad[i_pk])),
        "alive_at_peak": float(alive[i_pk]),
        "t_close_125": close_time(t, g, THR_PRIMARY),
        "t_close_110": close_time(t, g, THR_STRICT),
        "breach_area_125": area,
        "gmax_final": float(g[-1]),
        "gap_final_deg": float(np.degrees(rad[-1])),
        "alive_final": alive_end,
        "egap_final": float(post["E_gap"].to_numpy(float)[-1]),
    }


def run_cell(method, vmax, tau, seed):
    is_b2 = (method == "dual_pulse")
    tag = "B2" if is_b2 else "baseline"
    victim = victim_node_id(N, seed)
    run_dir = os.path.join(RUNS_DIR, f"{method}_v{vmax:g}_tau{tau:g}_s{seed}")
    os.makedirs(run_dir, exist_ok=True)
    for fn in ("target_telemetry.csv", "events.csv"):
        p = os.path.join(run_dir, fn)
        if os.path.exists(p):
            os.remove(p)

    env = dict(os.environ)
    env.update({
        "PYTHONIOENCODING": "utf-8",
        # --- run selection
        "PROPAGATION_METHOD": method,
        "PROPAGATION_K_PROP": "0.0",
        # --- scale / loop (campaign rule 3: pin everything relevant)
        "NUM_AGENTS": str(N),
        "SIM_DURATION": f"{T_FAIL_NOMINAL + BUDGET:g}",
        "CONTROL_PERIOD": f"{DT:g}",
        "AGENT_STATE_TIMEOUT": f"{5.0 * DT:g}",     # loss-free -> 5 ticks is clean+fast
        "K_E_TAU": f"{250.0 / N:.6f}",
        "EXPERIMENT_SEED": str(seed),
        "EXPERIMENT_REPRODUCIBLE": "True",
        "METRICS_T0": "0.0",
        # --- the swept kinematic axes
        "VM_MAX_SPEED_XY": f"{vmax:g}",
        "VM_TAU_XY": f"{tau:g}",
        # --- clean single event
        "DETERMINISTIC_FAILURE_ENABLE": "True",
        "DETERMINISTIC_FAILURE_AGENT_ID": str(victim),
        "DETERMINISTIC_FAILURE_TIME_T0": f"{T_FAIL_NOMINAL:g}",
        "DETERMINISTIC_FAILURE_OFF_TIME": "-1.0",   # permanent
        "FAILURE_ENABLE": "True",
        # --- ideal comms, static scenario
        "COMMUNICATION_TRANSMISSION_RANGE": "200",
        "COMMUNICATION_DELAY": "0.0",
        "COMMUNICATION_FAILURE_RATE": "0.0",
        "INIT_ANGLES_EQUIDISTANT": "True",
        "INIT_RADIUS_RANGE": "0.0",
        "TARGET_MOTION_SPEED_XY": "0.0",
        # --- output
        "VIS_OPEN_BROWSER": "False",
        "SKIP_TELEMETRY_PLOTS": "True",
        "RUNS_SUMMARY_CSV_PATH": os.path.join(run_dir, "runs_summary.csv"),
    })
    if is_b2:
        env.update({"DUAL_PULSE_INTEGRATION": "B2", "DUAL_PULSE_DELTA_SCALE": "1.0",
                    "DUAL_PULSE_T_FF": f"{tau:g}", "DUAL_PULSE_TTL_HOPS": str(3 * N)})
    else:
        env.pop("DUAL_PULSE_INTEGRATION", None)

    print(f"  -> {tag:8s} Vmax={vmax:<5g} tau_a={tau:<4g} s={seed} (victim={victim}) ...",
          end="", flush=True)
    proc = subprocess.run([sys.executable, MAIN_PY], cwd=run_dir, env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = metrics_from_run(run_dir)
    if not m or "error" in m:
        print(f" FAILED (rc={proc.returncode}) {m.get('error', '')}\n{(proc.stderr or '')[-500:]}")
        return None
    m.update({"method": tag, "N": N, "vmax": vmax, "tau_xy": tau, "seed": seed,
              "dt": DT, "budget": BUDGET, "victim": victim})
    # Campaign rule 5: every result row carries seed + git state + pinned params,
    # read from the CHILD's run_manifest.json (never from this parent process).
    m.update(run_provenance(run_dir))
    # Footprint: only target_telemetry/events are needed downstream.
    for fn in os.listdir(run_dir):
        if fn == "agent_telemetry.csv" or fn.endswith(".png"):
            try:
                os.remove(os.path.join(run_dir, fn))
            except OSError:
                pass
    tc = m["t_close_125"]
    print(f" peak={m['gmax_peak']:.3f} ({m['gap_peak_deg']:.1f}deg)  "
          f"t_close={'inf' if not np.isfinite(tc) else f'{tc:.2f}s'}  "
          f"area={m['breach_area_125']:.3f}")
    return m


def _key(r):
    return (str(r["method"]), float(r["vmax"]), float(r["tau_xy"]), int(r["seed"]))


def main():
    os.makedirs(RUNS_DIR, exist_ok=True)
    store = {}
    if os.path.exists(RESULTS_CSV):
        try:
            for r in pd.read_csv(RESULTS_CSV).to_dict("records"):
                store[_key(r)] = r
        except Exception:
            pass

    total = len(VMAXES) * len(TAUS) * len(SEEDS) * len(METHODS)
    print(f"Breach-window sweep: N={N}, Vmax={VMAXES} x tau_a={TAUS} x seeds={SEEDS} "
          f"x {METHODS}\n  = {total} runs of {T_FAIL_NOMINAL + BUDGET:g}s sim each; "
          f"dt={DT}, single PERMANENT failure at t={T_FAIL_NOMINAL:g}s.")
    print(f"  geometric prediction for the peak: 2*(N-1)/N = {2.0 * (N - 1) / N:.3f}")
    print(f"  {len(store)} cells already in {os.path.basename(RESULTS_CSV)}\n")

    for vmax in VMAXES:
        for tau in TAUS:
            for seed in SEEDS:
                for method in METHODS:
                    if _key({"method": "B2" if method == "dual_pulse" else "baseline",
                             "vmax": vmax, "tau_xy": tau, "seed": seed}) in store:
                        continue
                    r = run_cell(method, vmax, tau, seed)
                    if r:
                        store[_key(r)] = r
                        pd.DataFrame(list(store.values())).to_csv(RESULTS_CSV, index=False)

    df = pd.DataFrame(list(store.values()))
    if df.empty:
        print("\nSem resultados."); return
    print(f"\nWrote {RESULTS_CSV}  ({len(df)} rows)")
    print("Analise: python experiments/scaling_law/analyze_breach_window.py")


if __name__ == "__main__":
    main()
