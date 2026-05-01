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

import math
from dataclasses import dataclass
from typing import Dict, Optional

import matplotlib
# Save figures without requiring a GUI/Tk installation.
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config_param import (
    CONTROL_PERIOD,
    VM_MAX_SPEED_XY,
    METRICS_T0,
    METRICS_E_THR,
    METRICS_MA_W_SEC,
    METRICS_SETTLE_WINDOW_SEC,
)

CSV_DEFAULT_PATH = "agent_telemetry.csv"


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



def main(csv_path: str = CSV_DEFAULT_PATH) -> None:
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


if __name__ == "__main__":
    main()
