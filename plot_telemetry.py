"""Plot telemetry per node and compute the 7 metrics based on e(t)=|e_tau(t)|.

Expected CSV columns (agent_telemetry.csv):
    node_id, timestamp, e_tau, u, velocity_norm

Optional control-channel columns:
    u_local, u_prop

Plots:
    One figure per node with 3 stacked subplots (top->bottom):
        1) e_tau
        2) u, u_local, u_propag
        3) velocity_norm

Metrics (global / across all nodes):
  M1..M6 computed for t > t0
  M7 computed over the full run (starting at t=0)
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import matplotlib
# Save figures without requiring a GUI/Tk installation.
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config_param import (
    CONTROL_PERIOD,
    ENCIRCLEMENT_RADIUS,
    EXPERIMENT_REPRODUCIBLE,
    FAILURE_ENABLE,
    FAILURE_MEAN_FAILURES_PER_MIN,
    INIT_ANGLES_EQUIDISTANT,
    INIT_RADIUS_RANGE,
    METRICS_E_THR,
    METRICS_MA_W_SEC,
    METRICS_SETTLE_WINDOW_SEC,
    METRICS_T0,
    NUM_AGENTS,
    PROTECTION_ANGLE_DEG,
    SIM_DURATION,
    TANGENTIAL_COMPOSITION_MODE,
    TARGET_SWARM_OMEGA_REF,
    TARGET_SWARM_SPIN_ENABLE,
    VM_MAX_SPEED_XY,
)

CSV_DEFAULT_PATH = "agent_telemetry.csv"
SUMMARY_CSV_DEFAULT_PATH = "runs_summary.csv"

# Column order for the cross-run summary CSV. Keep stable: appending a new row to
# an existing file with a different header would corrupt the table.
SUMMARY_COLUMNS = [
    "run_timestamp_iso",
    "propagation_method",
    "k_prop",
    "composition_mode",
    "num_agents",
    "encirclement_radius",
    "sim_duration",
    "init_radius_range",
    "init_angles_equidistant",
    "failure_enable",
    "failure_mean_per_min",
    "target_swarm_spin_enable",
    "target_swarm_omega_ref",
    "protection_angle_deg",
    "experiment_reproducible",
    "metrics_t0",
    "metrics_e_thr",
    "metrics_ma_w_sec",
    "metrics_settle_window_sec",
    "M1_P95_e_pooled",
    "M2_P95_P95i",
    "M3_P95_abs_dedt",
    "M4_RMS_osc",
    "M5_mean_v2",
    "M6_Pr_sat",
    "M7_settle_median",
    "M7_settle_P95",
    "M7_settled_frac",
]


@dataclass
class MetricParams:
    dt: float
    vmax_xy: float
    t0: float
    e_thr: float
    ma_w: float
    settle_window: float


def _safe_percentile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def _rolling_mean(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.copy()
    s = pd.Series(x)
    return s.rolling(window=win, center=True, min_periods=1).mean().to_numpy(dtype=float)


def _settling_time(t: np.ndarray, e: np.ndarray, e_thr: float, settle_window: float, dt: float) -> Optional[float]:
    """Return first t_s such that e(t)<=e_thr for an entire continuous window of length settle_window."""
    if t.size == 0:
        return None

    # window length in samples
    W = int(round(settle_window / max(dt, 1e-12)))
    W = max(W, 1)

    ok = (e <= e_thr) & np.isfinite(e)

    # Need at least W samples to decide.
    if ok.size < W:
        return None

    # Sliding window: find earliest index i such that ok[i:i+W] are all True.
    # Efficient implementation using convolution over boolean.
    ok_int = ok.astype(int)
    window_sum = np.convolve(ok_int, np.ones(W, dtype=int), mode="valid")
    idx = np.where(window_sum == W)[0]
    if idx.size == 0:
        return None

    i0 = int(idx[0])
    return float(t[i0])


def compute_metrics(df: pd.DataFrame, params: MetricParams) -> Dict[str, float]:
    required = {"node_id", "timestamp", "e_tau", "u", "velocity_norm"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Sort for stable diff.
    df = df.sort_values(["node_id", "timestamp"]).reset_index(drop=True)

    # e(t)=|e_tau(t)|
    df["e"] = df["e_tau"].abs()

    # M1..M6 window: t > t0
    df_w = df[df["timestamp"] > params.t0].copy()

    # ---------- M1 (Accuracy): P95(e) pooled ----------
    m1 = _safe_percentile(df_w["e"].to_numpy(dtype=float), 95)

    # ---------- M2 (Fairness): P95( P95_i(e) ) ----------
    per_node_p95 = df_w.groupby("node_id")["e"].apply(lambda s: _safe_percentile(s.to_numpy(dtype=float), 95)).to_numpy(dtype=float)
    m2 = _safe_percentile(per_node_p95, 95)

    # ---------- M3 (Jitter): P95(|de/dt|) pooled ----------
    # Use per-node diffs, then pool.
    dedt_all = []
    for _, g in df_w.groupby("node_id", sort=False):
        e = g["e"].to_numpy(dtype=float)
        t = g["timestamp"].to_numpy(dtype=float)
        if e.size < 2:
            continue
        dt = np.diff(t)
        de = np.diff(e)
        # Guard against occasional timing jitter: use elementwise dt, fallback to params.dt when dt is non-finite.
        dt = np.where((dt > 0) & np.isfinite(dt), dt, params.dt)
        dedt = np.abs(de / dt)
        dedt_all.append(dedt)
    if len(dedt_all) == 0:
        m3 = float("nan")
    else:
        m3 = _safe_percentile(np.concatenate(dedt_all), 95)

    # ---------- M4 (Oscillation): RMS(e - MA_w(e)) pooled ----------
    win = int(round(params.ma_w / max(params.dt, 1e-12)))
    win = max(win, 1)
    osc_all = []
    for _, g in df_w.groupby("node_id", sort=False):
        e = g["e"].to_numpy(dtype=float)
        if e.size == 0:
            continue
        ma = _rolling_mean(e, win)
        osc = e - ma
        osc_all.append(osc)
    if len(osc_all) == 0:
        m4 = float("nan")
    else:
        osc = np.concatenate(osc_all)
        osc = osc[np.isfinite(osc)]
        m4 = float(np.sqrt(np.mean(osc * osc))) if osc.size else float("nan")

    # ---------- M5 (Effort): mean((velocity_norm / Vmax)^2) pooled ----------
    v = df_w["velocity_norm"].to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0 or not (math.isfinite(params.vmax_xy) and params.vmax_xy > 0):
        m5 = float("nan")
    else:
        vn = v / params.vmax_xy
        m5 = float(np.mean(vn * vn))

    # ---------- M6 (Saturation): Pr(velocity_norm >= Vmax) pooled ----------
    # Use >= with a tiny tolerance because many controllers hit the limit exactly after floating math.
    vraw = df_w["velocity_norm"].to_numpy(dtype=float)
    vraw = vraw[np.isfinite(vraw)]
    if vraw.size == 0 or not (math.isfinite(params.vmax_xy) and params.vmax_xy > 0):
        m6 = float("nan")
    else:
        tol = 1e-9
        m6 = float(np.mean(vraw >= (params.vmax_xy - tol)))

    # ---------- M7 (Transient): settling time per node ----------
    # NOTE: M7 is computed over the whole run (t>=0) because it's about how fast the system enters regime.
    settle_times = []
    for _, g in df.groupby("node_id", sort=False):
        t = g["timestamp"].to_numpy(dtype=float)
        e = g["e"].to_numpy(dtype=float)
        if t.size == 0:
            continue
        ts = _settling_time(t, e, params.e_thr, params.settle_window, params.dt)
        if ts is not None and math.isfinite(ts):
            settle_times.append(ts)

    if len(settle_times) == 0:
        m7_med = float("nan")
        m7_p95 = float("nan")
        settled_frac = 0.0
    else:
        st = np.asarray(settle_times, dtype=float)
        m7_med = float(np.median(st))
        m7_p95 = float(np.percentile(st, 95))
        settled_frac = float(len(settle_times) / df["node_id"].nunique())

    return {
        "M1_P95_e_pooled": m1,
        "M2_P95_P95i": m2,
        "M3_P95_abs_dedt": m3,
        "M4_RMS_osc": m4,
        "M5_mean_v2": m5,
        "M6_Pr_sat": m6,
        "M7_settle_median": m7_med,
        "M7_settle_P95": m7_p95,
        "M7_settled_frac": settled_frac,
        "_ma_win_samples": float(int(round(params.ma_w / max(params.dt, 1e-12)))),
        "_settle_win_samples": float(int(round(params.settle_window / max(params.dt, 1e-12)))),
    }


def print_metrics(metrics: Dict[str, float], params: MetricParams) -> None:
    ma_samp = int(metrics.get("_ma_win_samples", 0))
    st_samp = int(metrics.get("_settle_win_samples", 0))

    print("\n=== METRICS (using e(t)=|e_tau(t)|) ===")
    print(f"Window for M1..M6: t > t0, with t0 = {params.t0:.3f} s")
    print(
        f"Settling for M7: e_thr = {params.e_thr:.6f}, settle_window = {params.settle_window:.3f} s "
        f"(~{st_samp} samples at dt~{params.dt:.3f}s)"
    )
    print(f"Moving average for M4: MA_w = {params.ma_w:.3f} s (~{ma_samp} samples)")
    print(f"Velocity limit for M5/M6: Vmax = {params.vmax_xy:.3f} m/s")

    print(f"\nM1  Accuracy: P95(e) pooled = {metrics['M1_P95_e_pooled']:.6f}")
    print("    Interpretation: how small the error stays most of the time (regime, 95% tail).")

    print(f"\nM2  Fairness: P95(P95_i(e)) = {metrics['M2_P95_P95i']:.6f}")
    print("    Interpretation: checks whether any node is consistently worse (per-node tail, then across nodes).")

    print(f"\nM3  Jitter: P95(|de/dt|) = {metrics['M3_P95_abs_dedt']:.6f}")
    print("    Interpretation: fast error fluctuations (tremor/jitter).")

    print(f"\nM4  Sustained oscillation: RMS(e - MA_w(e)) = {metrics['M4_RMS_osc']:.6f}")
    print("    Interpretation: oscillatory component after removing the slow trend with a moving average.")

    print(f"\nM5  Control effort: mean((v_cmd_xy/Vmax)^2) = {metrics['M5_mean_v2']:.6f}")
    print("    Interpretation: how much of the velocity envelope is used on average (0=low effort, 1=always at the limit).")

    print(f"\nM6  Saturation: Pr(v_cmd_raw_xy >= Vmax) = {metrics['M6_Pr_sat']:.6f}")
    print("    Interpretation: how often the controller requests more than allowed (the usual source of windup).")
    print("    Note: we use >= (with tolerance) because it is common to hit the limit exactly.")

    print(
        f"\nM7  Transient: settling time (per node) = median={metrics['M7_settle_median']:.3f}s, "
        f"P95={metrics['M7_settle_P95']:.3f}s, settled_frac={metrics['M7_settled_frac']:.2f}"
    )
    print(
        "    Interpretation: how quickly it settles (first time t_s such that e(t) <= e_thr for a continuous window).\n"
        "    settled_frac = fraction of nodes that reached this criterion at any time in the simulation."
    )


def plot_per_node(df: pd.DataFrame) -> None:
    for node_id, g in df.groupby("node_id", sort=True):
        g = g.sort_values("timestamp")
        t = g["timestamp"].to_numpy(dtype=float)
        has_u_local = "u_local" in g.columns
        has_u_prop = "u_prop" in g.columns

        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
        fig.suptitle(f"Node {int(node_id)} telemetry")

        axes[0].plot(t, g["e_tau"].to_numpy(dtype=float))
        axes[0].set_ylabel("e_tau")
        axes[0].grid(True)

        axes[1].plot(t, g["u"].to_numpy(dtype=float), label="u", linewidth=2.0)
        if has_u_local:
            axes[1].plot(t, g["u_local"].to_numpy(dtype=float), label="u_local", alpha=0.9)
        if has_u_prop:
            axes[1].plot(t, g["u_prop"].to_numpy(dtype=float), label="u_propag", alpha=0.9)
        axes[1].set_ylabel("control")
        axes[1].grid(True)
        if has_u_local or has_u_prop:
            axes[1].legend(loc="best")

        axes[2].plot(t, g["velocity_norm"].to_numpy(dtype=float))
        axes[2].set_ylabel("velocity_norm")
        axes[2].set_xlabel("time [s]")
        axes[2].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        # plt.show()
        plt.savefig(f"node_{int(node_id)}_telemetry.png")
        plt.close(fig)



def _collect_run_context() -> Dict[str, object]:
    """Snapshot the configuration that defines this run.

    Pulls the propagation method/gain from environment variables (set by main.py
    before the simulation starts) and reads the rest from `config_param`. Used as
    the context portion of each row appended to the cross-run summary CSV.
    """
    method = os.environ.get("PROPAGATION_METHOD", "")
    try:
        k_prop = float(os.environ.get("PROPAGATION_K_PROP", "nan"))
    except ValueError:
        k_prop = float("nan")

    return {
        "propagation_method": method,
        "k_prop": k_prop,
        "composition_mode": str(TANGENTIAL_COMPOSITION_MODE),
        "num_agents": int(NUM_AGENTS),
        "encirclement_radius": float(ENCIRCLEMENT_RADIUS),
        "sim_duration": float(SIM_DURATION),
        "init_radius_range": float(INIT_RADIUS_RANGE),
        "init_angles_equidistant": bool(INIT_ANGLES_EQUIDISTANT),
        "failure_enable": bool(FAILURE_ENABLE),
        "failure_mean_per_min": float(FAILURE_MEAN_FAILURES_PER_MIN),
        "target_swarm_spin_enable": bool(TARGET_SWARM_SPIN_ENABLE),
        "target_swarm_omega_ref": float(TARGET_SWARM_OMEGA_REF),
        "protection_angle_deg": float(PROTECTION_ANGLE_DEG),
        "experiment_reproducible": bool(EXPERIMENT_REPRODUCIBLE),
    }


def append_run_summary(
    metrics: Dict[str, float],
    params: MetricParams,
    summary_csv_path: str = SUMMARY_CSV_DEFAULT_PATH,
    run_context: Optional[Dict[str, object]] = None,
) -> None:
    """Append a single row with this run's metrics + context to a comparative CSV.

    Writes the header line if the file does not yet exist (or is empty), then
    appends one row. Designed to be safe to call repeatedly across runs.
    """
    if run_context is None:
        run_context = _collect_run_context()

    row = {
        "run_timestamp_iso": datetime.now().isoformat(timespec="seconds"),
        "propagation_method": run_context.get("propagation_method", ""),
        "k_prop": run_context.get("k_prop", float("nan")),
        "composition_mode": run_context.get("composition_mode", ""),
        "num_agents": run_context.get("num_agents", ""),
        "encirclement_radius": run_context.get("encirclement_radius", ""),
        "sim_duration": run_context.get("sim_duration", ""),
        "init_radius_range": run_context.get("init_radius_range", ""),
        "init_angles_equidistant": run_context.get("init_angles_equidistant", ""),
        "failure_enable": run_context.get("failure_enable", ""),
        "failure_mean_per_min": run_context.get("failure_mean_per_min", ""),
        "target_swarm_spin_enable": run_context.get("target_swarm_spin_enable", ""),
        "target_swarm_omega_ref": run_context.get("target_swarm_omega_ref", ""),
        "protection_angle_deg": run_context.get("protection_angle_deg", ""),
        "experiment_reproducible": run_context.get("experiment_reproducible", ""),
        "metrics_t0": params.t0,
        "metrics_e_thr": params.e_thr,
        "metrics_ma_w_sec": params.ma_w,
        "metrics_settle_window_sec": params.settle_window,
        "M1_P95_e_pooled": metrics.get("M1_P95_e_pooled", float("nan")),
        "M2_P95_P95i": metrics.get("M2_P95_P95i", float("nan")),
        "M3_P95_abs_dedt": metrics.get("M3_P95_abs_dedt", float("nan")),
        "M4_RMS_osc": metrics.get("M4_RMS_osc", float("nan")),
        "M5_mean_v2": metrics.get("M5_mean_v2", float("nan")),
        "M6_Pr_sat": metrics.get("M6_Pr_sat", float("nan")),
        "M7_settle_median": metrics.get("M7_settle_median", float("nan")),
        "M7_settle_P95": metrics.get("M7_settle_P95", float("nan")),
        "M7_settled_frac": metrics.get("M7_settled_frac", float("nan")),
    }

    existing_header = _read_existing_header(summary_csv_path)
    if existing_header is not None and existing_header != SUMMARY_COLUMNS:
        # Schema drift: appending mismatched rows would silently corrupt the table.
        # Rotate the old file out of the way and start a fresh one with the new schema.
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        backup_path = f"{summary_csv_path}.bak.{ts}"
        os.rename(summary_csv_path, backup_path)
        print(
            f"[runs_summary] Schema changed; rotated previous file to {os.path.abspath(backup_path)}.\n"
            f"               New rows will use the updated columns."
        )
        existing_header = None

    write_header = existing_header is None
    with open(summary_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"\n[runs_summary] Appended 1 row to {os.path.abspath(summary_csv_path)}")


def _read_existing_header(csv_path: str) -> Optional[list]:
    """Return the first row of an existing CSV (treated as the header), or None."""
    if (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0:
        return None
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return None


def main(csv_path: str = CSV_DEFAULT_PATH, summary_csv_path: Optional[str] = None) -> None:
    df = pd.read_csv(csv_path)

    params = MetricParams(
        dt=float(CONTROL_PERIOD),
        vmax_xy=float(VM_MAX_SPEED_XY),
        t0=float(METRICS_T0),
        e_thr=float(METRICS_E_THR),
        ma_w=float(METRICS_MA_W_SEC),
        settle_window=float(METRICS_SETTLE_WINDOW_SEC),
    )

    metrics = compute_metrics(df, params)
    print_metrics(metrics, params)

    plot_per_node(df)

    # Resolve summary path: explicit arg > env var > default. Pass an empty
    # string (either via arg or env var) to skip the append step.
    if summary_csv_path is None:
        summary_csv_path = os.environ.get("RUNS_SUMMARY_CSV_PATH", SUMMARY_CSV_DEFAULT_PATH)
    if summary_csv_path:
        append_run_summary(metrics, params, summary_csv_path=summary_csv_path)


if __name__ == "__main__":
    main()
