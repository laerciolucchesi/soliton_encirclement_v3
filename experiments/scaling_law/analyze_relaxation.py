#!/usr/bin/env python
"""Offline re-analysis: SLOW-MODE relaxation time from existing runs.

The first-pass settling metric (G_max <= 1.05) is satisfied by the fast LOCAL
modes (the dead node's neighbours closing the hole). The O(N) / O(N^2) scaling
the thesis cares about lives in the SLOW GLOBAL mode (everyone reaching their new
uniform slot). That mode is the low-amplitude tail of E_gap(t) after the fast
modes die.

This script reads each run's target_telemetry.csv (already on disk, no re-run)
and estimates, per run, several slow-mode-sensitive observables of E_gap(t) after
the fault at t0:

  tau_fit   : time constant from a log-linear fit of the decay tail
              (E_gap ~ exp(-t/tau)); the most principled slow-mode estimate.
  t_10pct   : time for E_gap to fall to 10% of its post-fault peak (sustained).
  t_05pct   : time for E_gap to fall to  5% of its post-fault peak (sustained).

A relative-to-peak threshold auto-normalizes the perturbation magnitude across N,
so it probes the decay shape rather than the (N-dependent) peak height.

Then it plots each observable vs N (log-log) and fits the exponent p (obs ~ N^p)
for baseline vs dual_pulse. Compare with the loose-settling result.

Usage:
    python experiments/scaling_law/analyze_relaxation.py
"""

import os
import re
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(EXP_DIR, "runs")
T0 = float(os.environ.get("SCALING_T0", "5.0"))
SUSTAIN = 2.0          # s a relative threshold must hold continuously
TAIL_FLOOR_FRAC = 0.05  # fit the decay from peak down to this fraction of peak
OUT_CSV = os.path.join(EXP_DIR, "relaxation_results.csv")

_RUN_RE = re.compile(r"^(?P<method>.+)_N(?P<N>\d+)_s(?P<seed>\d+)$")
_COLORS = {"baseline": "tab:red", "dual_pulse": "tab:blue"}


def sustained_cross(t, y, thr, window, dt):
    """First t_s such that y <= thr continuously for `window` seconds."""
    W = max(int(round(window / max(dt, 1e-12))), 1)
    ok = (y <= thr) & np.isfinite(y)
    if ok.size < W:
        return None
    s = np.convolve(ok.astype(int), np.ones(W, dtype=int), mode="valid")
    idx = np.where(s == W)[0]
    return float(t[int(idx[0])]) if idx.size else None


def analyze_run(run_dir):
    name = os.path.basename(run_dir)
    m = _RUN_RE.match(name)
    if not m:
        return None
    tgt = os.path.join(run_dir, "target_telemetry.csv")
    if not os.path.exists(tgt):
        return None
    df = pd.read_csv(tgt)
    df = df[df["timestamp"] >= T0].reset_index(drop=True)
    if df.empty:
        return None

    t = df["timestamp"].to_numpy(float)
    e = df["E_gap"].to_numpy(float)
    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.01

    # post-fault peak (the failure is at t0, so the peak is right after)
    i_pk = int(np.nanargmax(e))
    t_pk, e_pk = t[i_pk], e[i_pk]

    # relative-threshold decay times (measured from t0)
    t10 = sustained_cross(t, e, 0.10 * e_pk, SUSTAIN, dt)
    t05 = sustained_cross(t, e, 0.05 * e_pk, SUSTAIN, dt)

    # log-linear fit of the decay tail: from the peak down to TAIL_FLOOR_FRAC*peak
    floor = TAIL_FLOOR_FRAC * e_pk
    mask = (np.arange(e.size) >= i_pk) & (e > floor) & np.isfinite(e)
    tau, r2, npts = np.nan, np.nan, int(mask.sum())
    if npts >= 5:
        tt, ee = t[mask], e[mask]
        # take the contiguous decaying segment starting at the peak
        coef = np.polyfit(tt, np.log(ee), 1)
        slope = coef[0]
        if slope < 0:
            tau = -1.0 / slope
        pred = np.polyval(coef, tt)
        ss_res = float(np.sum((np.log(ee) - pred) ** 2))
        ss_tot = float(np.sum((np.log(ee) - np.mean(np.log(ee))) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "method": m.group("method"),
        "N": int(m.group("N")),
        "seed": int(m.group("seed")),
        "egap_peak": float(e_pk),
        "tau_fit": float(tau),
        "tau_fit_r2": float(r2),
        "tau_fit_npts": npts,
        "t_10pct": (t10 - T0) if t10 is not None else np.nan,
        "t_05pct": (t05 - T0) if t05 is not None else np.nan,
    }


def fit_exp(N, y):
    N = np.asarray(N, float); y = np.asarray(y, float)
    ok = np.isfinite(N) & np.isfinite(y) & (N > 0) & (y > 0)
    if ok.sum() < 2:
        return None
    return float(np.polyfit(np.log(N[ok]), np.log(y[ok]), 1)[0])


def main():
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(RUNS_DIR, "*"))):
        if os.path.isdir(run_dir):
            r = analyze_run(run_dir)
            if r:
                rows.append(r)
    if not rows:
        raise SystemExit(f"no runs found under {RUNS_DIR}")
    df = pd.DataFrame(rows).sort_values(["method", "N"]).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)

    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print()

    observables = ["tau_fit", "t_10pct", "t_05pct"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, obs in zip(axes, observables):
        print(f"--- {obs}: exponent p (obs ~ N^p) ---")
        MIN_R2 = 0.90  # exclude runs whose exponential fit is unreliable (density confound at large N)
        for method, g in df.groupby("method"):
            color = _COLORS.get(method)
            gg = g
            if obs == "tau_fit":
                bad = g[~(g["tau_fit_r2"] >= MIN_R2)]
                gg = g[g["tau_fit_r2"] >= MIN_R2]
                if len(bad):
                    print(f"  ({method}) EXCLUDED N={sorted(bad['N'].astype(int).tolist())} (fit R2<{MIN_R2})")
                    aggb = bad.groupby("N")[obs].median().reset_index()
                    ax.scatter(aggb["N"], aggb[obs], s=55, facecolors="none",
                               edgecolors=color, zorder=3)
            agg = gg.groupby("N")[obs].median().reset_index().sort_values("N")
            N = agg["N"].to_numpy(float); y = agg[obs].to_numpy(float)
            ax.scatter(N, y, s=50, color=color, zorder=3, label=f"{method}")
            p = fit_exp(N, y)
            if p is not None:
                ok = np.isfinite(N) & np.isfinite(y) & (y > 0)
                xs = np.linspace(np.nanmin(N[ok]), np.nanmax(N[ok]), 50)
                b = np.polyfit(np.log(N[ok]), np.log(y[ok]), 1)[1]
                ax.plot(xs, np.exp(b) * xs ** p, "--", color=color, alpha=0.8,
                        label=f"{method}: p={p:.2f}")
            print(f"  {method:12s} p = {p}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("N"); ax.set_ylabel(obs)
        ax.set_title(obs); ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend(fontsize=8)
    fig.suptitle("Slow-mode relaxation observables vs N (log-log)")
    fig.tight_layout()
    out_png = os.path.join(EXP_DIR, "relaxation_vs_N.png")
    fig.savefig(out_png, dpi=130)
    print(f"\nSaved {out_png}\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
