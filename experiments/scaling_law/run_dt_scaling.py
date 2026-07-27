#!/usr/bin/env python
"""P3 -- tau_B2 vs N and vs dt, plus the dissemination-vs-actuation crossover.

Why. The thesis headline is "reconfiguration time FLAT in N (~2.1 s up to N=100)",
from largeN_results.csv. P2 left two committed and incompatible measurements for
B2 at N=50/dt=0.01: 4.06 s (ladder, committed+tested tree, everything pinned) and
2.115 s (largeN, May tree never committed). That cell decides whether Lei 1 is
~N^1.9 or ~N^1.1 and whether the headline advantage at N=100 is ~148x or ~43x.

Grids (see docs/experiments/DT_CROSSOVER.md):
  A   B2 core exponent : N {24,32,40,50,75,100} x dt {0.01,0.05} x seed {0,1,2}  = 36
  A2  asymptote        : N {150,200}            x dt {0.01,0.05} x seed 0        =  4
  C   detector control : N 50, AGENT_STATE_TIMEOUT FIXED at 0.25 s, both dt, 3 seeds = 6
  B   baseline         : N 50 (2 seeds) + N 100 (1 seed), both dt               =  6

DISSEMINATION METRIC -- no instrumentation was needed. `events.csv` already logs
`dual_pulse_event_completed_*` rows carrying (timestamp, node_id, h_CCW, h_CW,
N_new), which is exactly the (hop, t_apply) pair. Since there is NO code change,
the prompt's "run one cell before and after and confirm tau is identical" check is
vacuous and is reported as such.

The controlling hop variable is **max(h_CCW, h_CW)**, not h_CCW: a node applies its
shift only once BOTH counter-propagating pulses have reached it. Regressing t_apply
on h_CCW alone gives R2 ~ 0.03; on max(h_CCW,h_CW) it gives R2 ~ 0.9-0.95.

Usage:
    python experiments/scaling_law/run_dt_scaling.py            # env: DTS_GRIDS="A,A2,C,B"
    # DTS_NS / DTS_DTS / DTS_SEEDS override grid A; DTS_TAG names the output
"""
import math
import os
import sys
import subprocess

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_util import event_metrics, run_provenance  # noqa: E402

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))
MAIN_PY = os.path.join(REPO_ROOT, "main.py")

_TAG = os.environ.get("DTS_TAG", "")
_SUF = ("_" + _TAG) if _TAG else ""
RUNS_DIR = os.path.join(EXP_DIR, "dt_scaling_runs" + _SUF)
RESULTS_CSV = os.path.join(EXP_DIR, "dt_scaling_results" + _SUF + ".csv")

T0 = 5.0                    # fault instant AND the analysis window start (pinned here,
                            # recorded as a column; METRICS_T0 is a literal in
                            # config_param.py:762 and never enters tau_fit)
GAIN_PRODUCT = 250.0
T_FF = 1.0
TAU_XY = 1.0
GRIDS = [g.strip().upper() for g in os.environ.get("DTS_GRIDS", "A,A2,C,B").split(",") if g.strip()]


def campaign_git():
    """Captured once, before the sweep writes anything (see PROVENANCE.md)."""
    try:
        sys.path.insert(0, REPO_ROOT)
        import provenance
        return provenance.git_provenance()
    except Exception:
        return "unknown", True


CAMPAIGN_GIT = campaign_git()


def victim_node_id(n, seed):
    """The rule every runner in this folder uses; written into the report.

    Agents occupy ids 2..N+1, so this is the node diametrically opposite id 2,
    rotated by the seed. With an equidistant ring the choice is a symmetry check.
    """
    return 2 + ((n // 2 + seed) % n)


def tau_est_b2(n, dt):
    """Rough prior for budgeting only (never used in analysis).

    From the P2 ladder: B2 ~ 2.2 s at N=24 growing ~N^0.8. Deliberately generous.
    """
    return 2.2 * (n / 24.0) ** 0.9


def tau_est_baseline(n):
    """0.0417 * N^1.94 -- the campaign's own baseline law."""
    return 0.0417 * (n ** 1.94)


def budget_for(method, n, dt):
    if method == "baseline":
        return max(200.0, 3.5 * tau_est_baseline(n))
    return max(90.0, 8.0 * tau_est_b2(n, dt))


def dissemination_metrics(run_dir, n, dt, t_fault):
    """(hop, t_apply) statistics from events.csv -- the crossover diagnostic.

    Returns seconds_per_hop (slope of t_apply vs max(h_CCW,h_CW)), c = that / dt,
    t_dissem (latest apply time), coverage, and the two theoretical lines.
    """
    p = os.path.join(run_dir, "events.csv")
    out = {"sec_per_hop": np.nan, "c_ticks_per_hop": np.nan, "hop_fit_r2": np.nan,
           "t_dissem": np.nan, "t_apply_first": np.nan, "n_with_shift": 0,
           "coverage": np.nan, "max_hop": np.nan,
           "line_half_N_dt": 0.5 * n * dt, "line_N_dt": float(n) * dt}
    if not os.path.exists(p):
        return out
    try:
        d = pd.read_csv(p)
    except Exception:
        return out
    if "event_type" not in d.columns:
        return out
    s = d[d["event_type"].astype(str).str.startswith("dual_pulse")].copy()
    if s.empty:
        return out
    # One row per node: the FIRST shift it applied.
    s = s.sort_values("timestamp").groupby("node_id", as_index=False).first()
    s["hop"] = s[["h_CCW", "h_CW"]].max(axis=1)
    s["t_apply"] = s["timestamp"] - t_fault
    s = s[np.isfinite(s["hop"]) & np.isfinite(s["t_apply"])]
    if s.empty:
        return out
    out["n_with_shift"] = int(s["node_id"].nunique())
    out["coverage"] = out["n_with_shift"] / float(n - 1)   # n-1 alive after the fault
    out["max_hop"] = float(s["hop"].max())
    out["t_dissem"] = float(s["t_apply"].max())
    out["t_apply_first"] = float(s["t_apply"].min())
    x, y = s["hop"].to_numpy(float), s["t_apply"].to_numpy(float)
    if len(np.unique(x)) >= 2:
        slope, _ = np.polyfit(x, y, 1)
        out["sec_per_hop"] = float(slope)
        out["c_ticks_per_hop"] = float(slope / dt) if dt > 0 else np.nan
        r = np.corrcoef(x, y)[0, 1]
        out["hop_fit_r2"] = float(r * r)
    return out


def run_cell(grid, method, n, dt, seed, timeout_override=None):
    is_b2 = (method == "B2")
    k_e_tau = GAIN_PRODUCT / n
    victim = victim_node_id(n, seed)
    budget = budget_for(method, n, dt)
    tmo = timeout_override if timeout_override is not None else 5.0 * dt
    tag = f"{grid}_{method}_N{n}_dt{dt:g}_s{seed}"
    if timeout_override is not None:
        tag += f"_tmo{timeout_override:g}"
    run_dir = os.path.join(RUNS_DIR, tag)
    os.makedirs(run_dir, exist_ok=True)
    for fn in ("target_telemetry.csv", "events.csv"):
        p = os.path.join(run_dir, fn)
        if os.path.exists(p):
            os.remove(p)

    env = dict(os.environ)
    env.update({
        "PYTHONIOENCODING": "utf-8",
        "PROPAGATION_METHOD": "dual_pulse" if is_b2 else "baseline",
        "PROPAGATION_K_PROP": "0.0",
        "NUM_AGENTS": str(n),
        "SIM_DURATION": f"{T0 + budget:.4f}",
        "CONTROL_PERIOD": f"{dt:g}",
        "AGENT_STATE_TIMEOUT": f"{tmo:g}",
        "K_E_TAU": f"{k_e_tau:.6f}",
        "EXPERIMENT_SEED": str(seed),
        "EXPERIMENT_REPRODUCIBLE": "True",
        "VM_TAU_XY": f"{TAU_XY:g}",
        "VM_MAX_SPEED_XY": "10.0",
        "VM_MAX_ACC_XY": "4.0",
        "COMMUNICATION_TRANSMISSION_RANGE": "200",
        "COMMUNICATION_DELAY": "0.0",
        "COMMUNICATION_FAILURE_RATE": "0.0",
        # FAILURE_ENABLE must be TRUE even though we want exactly one deterministic
        # fault. protocol_agent.py:266 only schedules the failure-check timer when
        # FAILURE_ENABLE is set, and the deterministic-fault branch lives INSIDE that
        # timer's handler (protocol_agent.py:866). Setting it False therefore
        # suppresses the deterministic fault too, and the run completes silently with
        # NO event at all -- measured: tau_fit = 22.93 at N=24/dt=0.01 (an exponential
        # fitted to noise) with zero dual_pulse events. The Poisson stream is bypassed
        # for every agent whenever DETERMINISTIC_FAILURE_ENABLE is True
        # (config_param.py section 5b), so True here is both necessary and safe.
        "FAILURE_ENABLE": "True",
        "DETERMINISTIC_FAILURE_ENABLE": "True",
        "DETERMINISTIC_FAILURE_AGENT_ID": str(victim),
        "DETERMINISTIC_FAILURE_TIME_T0": f"{T0:g}",
        "DETERMINISTIC_FAILURE_OFF_TIME": "-1.0",
        "INIT_ANGLES_EQUIDISTANT": "True",
        "INIT_RADIUS_RANGE": "0.0",
        "TARGET_MOTION_SPEED_XY": "0.0",
        "HYSTERESIS_FRAC": "0.0",
        "VIS_OPEN_BROWSER": "False",
        "SKIP_TELEMETRY_PLOTS": "True",
        "RUNS_SUMMARY_CSV_PATH": os.path.join(run_dir, "runs_summary.csv"),
    })
    if is_b2:
        env.update({
            "DUAL_PULSE_INTEGRATION": "B2",
            "DUAL_PULSE_DELTA_SCALE": "1.0",
            "DUAL_PULSE_T_FF": f"{T_FF:g}",
            "DUAL_PULSE_TTL_HOPS": str(3 * n),
            "DUAL_PULSE_BROADCAST_REPEATS": "2",
        })
    else:
        for k in ("DUAL_PULSE_INTEGRATION", "DUAL_PULSE_DELTA_SCALE", "DUAL_PULSE_T_FF",
                  "DUAL_PULSE_TTL_HOPS", "DUAL_PULSE_BROADCAST_REPEATS"):
            env.pop(k, None)

    print(f"  -> {grid} {method:8s} N={n:>3} dt={dt:<5g} s={seed} tmo={tmo:g} "
          f"(budget={budget:.0f}s) ...", end="", flush=True)
    proc = subprocess.run([sys.executable, MAIN_PY], cwd=run_dir, env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    tgt = os.path.join(run_dir, "target_telemetry.csv")
    if not os.path.exists(tgt):
        print(f" FAILED (rc={proc.returncode})\n{(proc.stderr or '')[-600:]}")
        return None
    try:
        tel = pd.read_csv(tgt)
    except Exception as exc:
        print(f" FAILED (telemetria ilegivel: {exc})")
        return None
    m = event_metrics(tel, T0)
    if not m:
        print(" FAILED (sem metricas)")
        return None

    # The real fault instant, from events.csv, so t_apply is anchored on the event
    # rather than on the nominal t0.
    t_fault = T0
    ev = os.path.join(run_dir, "events.csv")
    if os.path.exists(ev):
        try:
            e = pd.read_csv(ev)
            f = e[e["event_type"] == "failure_start"]
            if len(f):
                t_fault = float(f["timestamp"].min())
        except Exception:
            pass
    m.update(dissemination_metrics(run_dir, n, dt, t_fault))
    m.update({
        "grid": grid, "method": method, "N": n, "dt": dt, "seed": seed,
        "victim": victim, "t_fault": t_fault, "analysis_t0": T0,
        "k_e_tau": k_e_tau, "agent_state_timeout": tmo, "budget": budget,
        "sim_duration": T0 + budget, "vm_tau_xy": TAU_XY,
        "t_ff": T_FF if is_b2 else "", "ttl_hops": 3 * n if is_b2 else "",
        "broadcast_repeats": 2 if is_b2 else "",
        "encirclement_radius": 20.0,
        # Literals in config_param (NOT env-overridable): recorded as observed.
        "adversary_roam_speed_xy": 0.0, "target_swarm_spin_enable": False,
    })
    m.update(run_provenance(run_dir))
    m["git_commit"], m["git_dirty"] = CAMPAIGN_GIT

    for fn in os.listdir(run_dir):
        if fn == "agent_telemetry.csv" or fn.endswith(".png"):
            try:
                os.remove(os.path.join(run_dir, fn))
            except OSError:
                pass

    c = m.get("c_ticks_per_hop", float("nan"))
    print(f" tau={m['tau_fit']:7.2f} R2={m['tau_fit_r2']:5.2f} "
          f"t_dis={m['t_dissem']:6.2f} c={c:5.2f} cov={m['coverage']:.2f}")
    return m


def cells():
    """Every (grid, method, N, dt, seed, timeout_override) this runner will execute."""
    out = []
    if "A" in GRIDS:
        ns = [int(x) for x in os.environ.get("DTS_NS", "24,32,40,50,75,100").split(",") if x.strip()]
        dts = [float(x) for x in os.environ.get("DTS_DTS", "0.01,0.05").split(",") if x.strip()]
        seeds = [int(x) for x in os.environ.get("DTS_SEEDS", "0,1,2").split(",") if x.strip()]
        for n in ns:
            for dt in dts:
                for s in seeds:
                    out.append(("A", "B2", n, dt, s, None))
    if "A2" in GRIDS:
        for n in (150, 200):
            for dt in (0.01, 0.05):
                out.append(("A2", "B2", n, dt, 0, None))
    if "C" in GRIDS:
        for dt in (0.01, 0.05):
            for s in (0, 1, 2):
                out.append(("C", "B2", 50, dt, s, 0.25))
    if "B" in GRIDS:
        for dt in (0.01, 0.05):
            for s in (0, 1):
                out.append(("B", "baseline", 50, dt, s, None))
        for dt in (0.01, 0.05):
            out.append(("B", "baseline", 100, dt, 0, None))
    return out


def _key(r):
    return (str(r["grid"]), str(r["method"]), int(r["N"]),
            round(float(r["dt"]), 6), int(r["seed"]),
            round(float(r["agent_state_timeout"]), 6))


def main():
    os.makedirs(RUNS_DIR, exist_ok=True)
    store = {}
    if os.path.exists(RESULTS_CSV):
        try:
            for r in pd.read_csv(RESULTS_CSV).to_dict("records"):
                store[_key(r)] = r
        except Exception:
            pass
    todo = cells()
    print(f"P3 dt/N scaling: grids={GRIDS}, {len(todo)} cells, "
          f"git {CAMPAIGN_GIT[0]} dirty={CAMPAIGN_GIT[1]}")
    print(f"  victim rule: 2 + ((N//2 + seed) % N)   analysis T0 = {T0:g} s")
    print(f"  -> {os.path.basename(RESULTS_CSV)};  {len(store)} cells already present\n")

    for (grid, method, n, dt, seed, tmo) in todo:
        k = (grid, method, n, round(dt, 6), seed,
             round(tmo if tmo is not None else 5.0 * dt, 6))
        if k in store:
            continue
        r = run_cell(grid, method, n, dt, seed, timeout_override=tmo)
        if r:
            store[_key(r)] = r
            pd.DataFrame(list(store.values())).to_csv(RESULTS_CSV, index=False)

    df = pd.DataFrame(list(store.values()))
    if df.empty:
        print("\nSem resultados."); return
    print(f"\nWrote {RESULTS_CSV}  ({len(df)} rows)")
    print("Analise: python experiments/scaling_law/analyze_dt_crossover.py")


if __name__ == "__main__":
    main()
