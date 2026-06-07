#!/usr/bin/env python
"""Plot the scaling law: post-failure stabilization time vs N (log-log).

Reads scaling_results.csv (written by run_scaling_sweep.py), aggregates by
(method, N) taking the median over seeds, fits a power law t_stab ~ N^p on the
settled points (linear fit in log-log), and annotates the slope p per method.

Thesis prediction (the whole point of the de-risking experiment):
    baseline   slope p ~ 2   (diffusive, O(N^2))
    dual_pulse slope p ~ 1   (ballistic / diameter-bound, O(N))
If the two slopes are clearly different, this is the central figure of the
thesis. If they coincide, the spine needs rethinking (the honest negative
result: formation control is actuation-limited, fast information is moot).

Usage:
    python experiments/scaling_law/plot_scaling.py
    # choose the metric: $env:SCALING_PLOT_METRIC="t_stab_egap"
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(EXP_DIR, "scaling_results.csv")
METRIC = os.environ.get("SCALING_PLOT_METRIC", "t_stab_gmax")
OUT_PNG = os.path.join(EXP_DIR, f"scaling_law_{METRIC}.png")

_COLORS = {"baseline": "tab:red", "dual_pulse": "tab:blue"}


def fit_loglog_slope(N, y):
    """Return (slope, log_intercept) of log(y) = slope*log(N) + b over valid points."""
    N = np.asarray(N, float)
    y = np.asarray(y, float)
    m = np.isfinite(N) & np.isfinite(y) & (N > 0) & (y > 0)
    if m.sum() < 2:
        return None, None
    slope, b = np.polyfit(np.log(N[m]), np.log(y[m]), 1)
    return float(slope), float(b)


def main():
    if not os.path.exists(RESULTS_CSV):
        raise SystemExit(f"missing {RESULTS_CSV} -- run run_scaling_sweep.py first")
    df = pd.read_csv(RESULTS_CSV)
    if METRIC not in df.columns:
        raise SystemExit(f"metric {METRIC!r} not in results columns {list(df.columns)}")

    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    slopes = {}

    for method, g in df.groupby("method"):
        agg = (
            g.groupby("N")[METRIC]
            .median()
            .reset_index()
            .sort_values("N")
        )
        N = agg["N"].to_numpy(float)
        y = agg[METRIC].to_numpy(float)
        color = _COLORS.get(method, None)

        ax.scatter(N, y, s=55, color=color, zorder=3, label=f"{method} (median)")

        slope, b = fit_loglog_slope(N, y)
        slopes[method] = slope
        if slope is not None:
            xs = np.linspace(np.nanmin(N), np.nanmax(N), 60)
            ax.plot(xs, np.exp(b) * xs ** slope, "--", color=color, alpha=0.9,
                    label=f"{method}: fit p = {slope:.2f}")

    # Reference power-law guides (N^1 and N^2), anchored to the smallest N point.
    allN = df["N"].to_numpy(float)
    if allN.size:
        n0 = np.nanmin(allN)
        y_anchor = np.nanmedian(df.loc[df["N"] == n0, METRIC].to_numpy(float))
        if np.isfinite(y_anchor) and y_anchor > 0:
            xs = np.linspace(np.nanmin(allN), np.nanmax(allN), 60)
            ax.plot(xs, y_anchor * (xs / n0) ** 1.0, ":", color="gray", alpha=0.6, label="ref ~ N^1")
            ax.plot(xs, y_anchor * (xs / n0) ** 2.0, ":", color="black", alpha=0.4, label="ref ~ N^2")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("swarm size  N")
    ax.set_ylabel(f"post-failure stabilization time [s]  ({METRIC})")
    ax.set_title("Scaling law: stabilization time vs swarm size\n"
                 "prediction: baseline p~2 (diffusive)  |  dual_pulse p~1 (ballistic)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    print(f"Saved {OUT_PNG}")

    print("\nFitted log-log slopes (p in t_stab ~ N^p):")
    for method, s in slopes.items():
        print(f"  {method:12s}  p = {s}")


if __name__ == "__main__":
    main()
