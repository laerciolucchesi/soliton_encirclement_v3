#!/usr/bin/env python
"""E3' -- regenerate the WHOLE integration ladder on committed code, everything pinned.

Why. The ladder (baseline -> A -> B -> B2) is the thesis's central table, and its
numbers come from optionB_results.csv / gain_results.csv, produced 2026-05-30 in an
UNCOMMITTED tree. DUAL_PULSE_DELTA_SCALE did not exist in the versioned config on that
date (it landed in ff54ade, 2026-06-07), so which scale the "Option B, scale 0.5" row
actually used is unknowable from the artifacts. Neither run_optionB_test.py nor
run_gain_sweep.py pins CONTROL_PERIOD, VM_TAU_XY, AGENT_STATE_TIMEOUT, COMMUNICATION_*,
DUAL_PULSE_TTL_HOPS or DUAL_PULSE_DELTA_SCALE; both run a single seed.

This runner does not modify them -- it is a clean re-run with every relevant parameter
fixed in the child environment and provenance on every row.

Grid: 6 variants x N {24,40,50} x seed {0,1,2} = 54 runs per dt.

  V1 baseline_norm   baseline,   K_E_TAU=250/N
  V2 baseline_fixed  baseline,   K_E_TAU=25          (the historical "high fixed gain")
  V3 A_s05           dual_pulse, INTEGRATION=A,  DELTA_SCALE=0.5
  V4 Bmin_s05        dual_pulse, INTEGRATION=B,  DELTA_SCALE=0.5
  V5 Bmin_s10        dual_pulse, INTEGRATION=B,  DELTA_SCALE=1.0
  V6 B2_s10          dual_pulse, INTEGRATION=B2, DELTA_SCALE=1.0

Victim: 2 + ((N//2 + seed) % N) -- the convention every existing runner uses
(run_collapse_sweep.py:52, run_gain_sweep.py:47, run_optionB_test.py:48,
run_largeN_confirm.py:47). With equidistant init, zero radius scatter and a
deterministic fault, the seed's ONLY effect is which node dies: a symmetry check.

BUDGET -- deliberately UNIFORM across the six variants at a given N:
    budget(N) = max(60, 3.5 * 0.033 * N^2)
Two reasons. (a) The settled criterion uses egap_late_std over the LAST 20 s, so a
per-variant budget would compare different windows and make "settled" mean different
things per row. (b) It satisfies ">= 3.5 * expected tau" for EVERY variant, including
V5: the campaign prompt estimated ~10 s for V4/V5/V6, but the historical table says
B-min@1.0 relaxes in 16.5/43.0/62.6 s -- baseline-like, not overlay-like. Budgeting V5
at 35 s would have fitted tau on a truncated tail at N=50.

Usage:
    python experiments/scaling_law/run_ladder.py
    # env: LADDER_NS="24,40,50" LADDER_SEEDS="0,1,2" LADDER_VARIANTS="V1,...,V6"
    #      CONTROL_PERIOD="0.05"  LADDER_TAG="dt05"
"""
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

DT = float(os.environ.get("CONTROL_PERIOD", "0.05"))
_TAG = os.environ.get("LADDER_TAG", f"dt{str(DT).replace('.', '')}")
RUNS_DIR = os.path.join(EXP_DIR, "ladder_runs_" + _TAG)
RESULTS_CSV = os.path.join(EXP_DIR, f"ladder_results_{_TAG}.csv")

NS = [int(x) for x in os.environ.get("LADDER_NS", "24,40,50").split(",") if x.strip()]
SEEDS = [int(x) for x in os.environ.get("LADDER_SEEDS", "0,1,2").split(",") if x.strip()]
T0 = 5.0                      # fault instant; also the metric window start
GAIN_PRODUCT = 250.0          # K_E_TAU = GAIN_PRODUCT / N for the "stable" variants
FIXED_GAIN = 25.0             # the historical "high fixed gain"
T_FF = 1.0
TAU_XY = 1.0

# STABILITY CRITERION (reported in docs/experiments/LADDER.md).
SETTLED_EGAP_FINAL = 1e-2
SETTLED_LATE_STD = 1e-3
SETTLED_R2 = 0.80

# (key, method, integration, delta_scale, k_e_tau_rule)
VARIANTS = [
    ("V1_baseline_norm",  "baseline",   None, None, "norm"),
    ("V2_baseline_fixed", "baseline",   None, None, "fixed"),
    ("V3_A_s05",          "dual_pulse", "A",  0.5,  "norm"),
    ("V4_Bmin_s05",       "dual_pulse", "B",  0.5,  "norm"),
    ("V5_Bmin_s10",       "dual_pulse", "B",  1.0,  "norm"),
    ("V6_B2_s10",         "dual_pulse", "B2", 1.0,  "norm"),
]
WANTED = [v.strip() for v in os.environ.get("LADDER_VARIANTS", "").split(",") if v.strip()]


def campaign_git():
    """Git state captured ONCE, before the sweep writes anything.

    Per-cell capture is self-referential: the runner's own untracked results CSV
    dirties the tree, so cell 1 would record dirty=False and cells 2..n dirty=True
    for reasons unrelated to the code. Git state is a property of the CAMPAIGN; the
    pinned parameters stay a property of the cell (read from the child's manifest).
    """
    try:
        sys.path.insert(0, REPO_ROOT)
        import provenance
        return provenance.git_provenance()
    except Exception:
        return "unknown", True


CAMPAIGN_GIT = campaign_git()


def victim_node_id(n, seed):
    return 2 + ((n // 2 + seed) % n)


def budget_for(n):
    """Uniform across variants; >= 3.5 * the SLOWEST expected tau at this N."""
    return max(60.0, 3.5 * 0.033 * n * n)


def run_cell(vkey, method, integration, scale, gain_rule, n, seed):
    k_e_tau = (GAIN_PRODUCT / n) if gain_rule == "norm" else FIXED_GAIN
    victim = victim_node_id(n, seed)
    budget = budget_for(n)
    run_dir = os.path.join(RUNS_DIR, f"{vkey}_N{n}_s{seed}")
    os.makedirs(run_dir, exist_ok=True)
    for fn in ("target_telemetry.csv", "events.csv"):
        p = os.path.join(run_dir, fn)
        if os.path.exists(p):
            os.remove(p)

    env = dict(os.environ)
    env.update({
        "PYTHONIOENCODING": "utf-8",
        # --- what defines the variant
        "PROPAGATION_METHOD": method,
        "PROPAGATION_K_PROP": "0.0",
        "K_E_TAU": f"{k_e_tau:.6f}",
        # --- scale / loop
        "NUM_AGENTS": str(n),
        "SIM_DURATION": f"{T0 + budget:.4f}",
        "CONTROL_PERIOD": f"{DT:g}",
        "AGENT_STATE_TIMEOUT": f"{5.0 * DT:g}",
        "EXPERIMENT_SEED": str(seed),
        "EXPERIMENT_REPRODUCIBLE": "True",
        # METRICS_T0 only feeds plot_telemetry's M1..M7 into runs_summary.csv; the
        # ladder's tau comes from metrics_util.event_metrics(df, T0) with T0 passed
        # explicitly. Pinned anyway so the manifest records an unambiguous value.
        "METRICS_T0": f"{T0:g}",
        # --- platform
        "VM_TAU_XY": f"{TAU_XY:g}",
        "VM_MAX_SPEED_XY": "10.0",
        "VM_MAX_ACC_XY": "4.0",
        # --- ideal comms
        "COMMUNICATION_TRANSMISSION_RANGE": "200",
        "COMMUNICATION_DELAY": "0.0",
        "COMMUNICATION_FAILURE_RATE": "0.0",
        # --- clean single permanent fault, uniform static ring
        "DETERMINISTIC_FAILURE_ENABLE": "True",
        "DETERMINISTIC_FAILURE_AGENT_ID": str(victim),
        "DETERMINISTIC_FAILURE_TIME_T0": f"{T0:g}",
        "DETERMINISTIC_FAILURE_OFF_TIME": "-1.0",
        "FAILURE_ENABLE": "True",
        "INIT_ANGLES_EQUIDISTANT": "True",
        "INIT_RADIUS_RANGE": "0.0",
        "TARGET_MOTION_SPEED_XY": "0.0",
        "HYSTERESIS_FRAC": "0.0",
        # --- output
        "VIS_OPEN_BROWSER": "False",
        "SKIP_TELEMETRY_PLOTS": "True",
        "RUNS_SUMMARY_CSV_PATH": os.path.join(run_dir, "runs_summary.csv"),
    })
    # dual_pulse knobs. For the baselines they are removed from the environment so an
    # inherited value cannot silently label a run with knobs it never used.
    if method == "dual_pulse":
        env.update({
            "DUAL_PULSE_INTEGRATION": integration,
            "DUAL_PULSE_DELTA_SCALE": f"{scale:g}",
            "DUAL_PULSE_T_FF": f"{T_FF:g}",
            "DUAL_PULSE_TTL_HOPS": str(3 * n),
        })
    else:
        for k in ("DUAL_PULSE_INTEGRATION", "DUAL_PULSE_DELTA_SCALE",
                  "DUAL_PULSE_T_FF", "DUAL_PULSE_TTL_HOPS"):
            env.pop(k, None)

    print(f"  -> {vkey:<18} N={n:>2} s={seed} (K={k_e_tau:6.2f}, victim={victim:>2}, "
          f"budget={budget:.0f}s) ...", end="", flush=True)
    proc = subprocess.run([sys.executable, MAIN_PY], cwd=run_dir, env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    tgt = os.path.join(run_dir, "target_telemetry.csv")
    if not os.path.exists(tgt):
        print(f" FAILED (rc={proc.returncode})\n{(proc.stderr or '')[-600:]}")
        return None

    m = event_metrics(pd.read_csv(tgt), T0)     # same helper the campaign uses
    if not m:
        print(" FAILED (sem metricas)")
        return None

    settled = bool(
        np.isfinite(m.get("egap_final", np.nan)) and m["egap_final"] < SETTLED_EGAP_FINAL
        and np.isfinite(m.get("egap_late_std", np.nan)) and m["egap_late_std"] < SETTLED_LATE_STD
        and np.isfinite(m.get("tau_fit_r2", np.nan)) and m["tau_fit_r2"] > SETTLED_R2
    )
    m.update({
        "variant": vkey, "method": method,
        "integration": integration if integration else "",
        "delta_scale": scale if scale is not None else "",
        "gain_rule": gain_rule, "k_e_tau": k_e_tau,
        "N": n, "seed": seed, "victim": victim,
        "dt": DT, "budget": budget, "sim_duration": T0 + budget,
        "t_fault": T0, "t_ff": T_FF if method == "dual_pulse" else "",
        "ttl_hops": 3 * n if method == "dual_pulse" else "",
        "vm_tau_xy": TAU_XY, "agent_state_timeout": 5.0 * DT,
        "settled": settled,
        # These two are plain literals in config_param (NOT env-overridable), so they
        # are recorded as observed rather than pinned. See LADDER.md.
        "adversary_roam_speed_xy": 0.0, "target_swarm_spin_enable": False,
    })
    m.update(run_provenance(run_dir))           # pinned params from the CHILD manifest
    m["git_commit"], m["git_dirty"] = CAMPAIGN_GIT   # git state = campaign property

    for fn in os.listdir(run_dir):
        if fn == "agent_telemetry.csv" or fn.endswith(".png"):
            try:
                os.remove(os.path.join(run_dir, fn))
            except OSError:
                pass

    print(f" tau={m['tau_fit']:7.2f} R2={m['tau_fit_r2']:5.2f} "
          f"egap_f={m['egap_final']:.4f} std={m['egap_late_std']:.5f} "
          f"{'SETTLED' if settled else 'NOT-SETTLED'}")
    return m


def _key(r):
    return (str(r["variant"]), int(r["N"]), int(r["seed"]))


def main():
    os.makedirs(RUNS_DIR, exist_ok=True)
    variants = [v for v in VARIANTS if (not WANTED or v[0] in WANTED
                                        or v[0].split("_")[0] in WANTED)]
    store = {}
    if os.path.exists(RESULTS_CSV):
        try:
            for r in pd.read_csv(RESULTS_CSV).to_dict("records"):
                store[_key(r)] = r
        except Exception:
            pass

    total = len(variants) * len(NS) * len(SEEDS)
    print(f"Integration ladder: {len(variants)} variants x N={NS} x seeds={SEEDS} "
          f"= {total} runs, dt={DT:g}")
    print(f"  git {CAMPAIGN_GIT[0]} dirty={CAMPAIGN_GIT[1]}   -> {os.path.basename(RESULTS_CSV)}")
    print(f"  budgets: " + ", ".join(f"N={n}:{budget_for(n):.0f}s" for n in NS))
    print(f"  settled = egap_final < {SETTLED_EGAP_FINAL:g} AND late_std < "
          f"{SETTLED_LATE_STD:g} AND R2 > {SETTLED_R2:g}")
    print(f"  {len(store)} cells already present\n")

    for n in NS:
        for seed in SEEDS:
            for (vkey, method, integ, scale, rule) in variants:
                if (vkey, n, seed) in store:
                    continue
                r = run_cell(vkey, method, integ, scale, rule, n, seed)
                if r:
                    store[_key(r)] = r
                    pd.DataFrame(list(store.values())).to_csv(RESULTS_CSV, index=False)

    df = pd.DataFrame(list(store.values()))
    if df.empty:
        print("\nSem resultados."); return
    print(f"\nWrote {RESULTS_CSV}  ({len(df)} rows)")
    print("Analise: python experiments/scaling_law/analyze_ladder.py")


if __name__ == "__main__":
    main()
