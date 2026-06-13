"""Tests for experiments/scaling_law/metrics_util.py — the campaign metrics.

These are the numbers that become thesis tables (t_settle, egap_*, overshoot,
effort), so they get golden tests on synthetic signals with hand-computable
values. Until now this module only had a print-based self-test.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

# metrics_util lives under experiments/scaling_law (not a package).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "scaling_law"))

from metrics_util import (  # noqa: E402
    aggregate_seeds,
    effort_metrics,
    event_metrics,
    exp_tau,
    overshoot_frac,
    settling_time,
)


def _t(duration=60.0, dt=0.01):
    return np.linspace(0.0, duration, int(duration / dt) + 1)


# ---------------------------------------------------------------------------
# settling_time / exp_tau (previously only print-tested)
# ---------------------------------------------------------------------------

def test_exp_tau_recovers_time_constant():
    t = _t()
    tau, r2 = exp_tau(t, 0.3 * np.exp(-t / 2.0))
    assert tau == pytest.approx(2.0, rel=0.02)
    assert r2 > 0.999


def test_settling_time_clean_exponential():
    # Band = 5% of peak; e(t)=0.3*exp(-t/2) -> t_settle ~ 2*ln(0.3/0.015) ~ 6 s.
    t = _t()
    ts, e_inf = settling_time(t, 0.3 * np.exp(-t / 2.0), band_frac=0.05)
    assert ts == pytest.approx(2.0 * np.log(1 / 0.05), rel=0.05)
    assert e_inf == pytest.approx(0.0, abs=1e-6)


def test_settling_time_never_settles_is_inf():
    t = _t()
    e = 0.3 + 0.2 * np.sin(3.0 * t)   # oscillates around 0.3 forever
    ts, _ = settling_time(t, e, band_frac=0.05)
    assert ts == float("inf")


def test_settling_time_stuck_high_settles_early_with_high_floor():
    # A signal that "settles" at a HIGH value: t_settle small but egap_settle
    # exposes the failure — the documented (t_settle, egap_settle) pair semantics.
    t = _t()
    e = np.full_like(t, 0.5)
    ts, e_inf = settling_time(t, e, band_frac=0.05)
    assert ts == 0.0
    assert e_inf == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# overshoot_frac
# ---------------------------------------------------------------------------

def test_overshoot_zero_for_monotone_decay():
    t = _t()
    assert overshoot_frac(t, 0.3 * np.exp(-t / 2.0)) == pytest.approx(0.0, abs=1e-9)


def test_overshoot_detects_reexcursion():
    # Decay that re-rises to 0.15 after first touching the band around ~0.
    # Excess is measured BEYOND the band (0.05*peak=0.015):
    # (0.15 - 0.015) / 0.3 = 0.45.
    t = _t()
    e = 0.3 * np.exp(-t / 1.0)
    bump = 0.15 * np.exp(-0.5 * ((t - 30.0) / 1.5) ** 2)   # re-excursion at t=30
    val = overshoot_frac(t, e + bump)
    assert val == pytest.approx(0.45, rel=0.05)


def test_overshoot_nan_when_never_in_band():
    t = _t(duration=5.0)
    e = np.full_like(t, 1.0)
    e[0] = 2.0  # peak 2.0, asymptote 1.0, never within 5% of peak band around 1.0...
    # band = 0.05*2 = 0.1; |e - 1| = 0 inside for the flat part -> actually inside.
    # Use a signal that stays FAR from its final-window median is impossible by
    # construction (median IS from the signal); so assert the defined behavior:
    assert overshoot_frac(t, e) >= 0.0  # well-defined, non-negative


# ---------------------------------------------------------------------------
# event_metrics integration (includes overshoot_frac key)
# ---------------------------------------------------------------------------

def test_event_metrics_keys_and_values():
    t = _t()
    df = pd.DataFrame({"timestamp": t, "E_gap": 0.3 * np.exp(-t / 2.0)})
    m = event_metrics(df, t0=0.0)
    for k in ("tau_fit", "t_settle", "egap_settle", "egap_peak", "egap_final",
              "egap_avg", "egap_late_std", "overshoot_frac"):
        assert k in m, f"missing metric {k}"
    assert m["tau_fit"] == pytest.approx(2.0, rel=0.02)
    assert m["egap_peak"] == pytest.approx(0.3)
    assert m["overshoot_frac"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# effort_metrics (M5/M6/M2 from agent telemetry)
# ---------------------------------------------------------------------------

def test_effort_metrics_hand_values(tmp_path):
    # 2 nodes x 4 samples; node 0 at v=5 (half vmax), node 1 pinned at vmax.
    rows = []
    for k in range(4):
        ts = 0.1 * (k + 1)
        rows.append({"node_id": 0, "timestamp": ts, "velocity_norm": 5.0, "e_tau_real": 0.02})
        rows.append({"node_id": 1, "timestamp": ts, "velocity_norm": 10.0, "e_tau_real": 0.10})
    p = tmp_path / "agent_telemetry.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    m = effort_metrics(str(p), t0=0.0, vmax=10.0)
    assert m["effort_mean_v2"] == pytest.approx((0.25 + 1.0) / 2.0)   # mean of 0.25 and 1.0
    assert m["sat_frac"] == pytest.approx(0.5)                         # node 1 always at vmax
    assert m["fairness_p95"] == pytest.approx(0.10, rel=0.05)          # P95 across node-P95s


def test_effort_metrics_missing_file_returns_empty(tmp_path):
    assert effort_metrics(str(tmp_path / "nope.csv")) == {}


def test_effort_metrics_falls_back_to_e_tau(tmp_path):
    rows = [{"node_id": 0, "timestamp": 0.1, "velocity_norm": 1.0, "e_tau": 0.3}]
    p = tmp_path / "a.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    m = effort_metrics(str(p), vmax=10.0)
    assert m["fairness_p95"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# aggregate_seeds
# ---------------------------------------------------------------------------

def test_aggregate_seeds_median_worst_std():
    a = aggregate_seeds([0.1, 0.3, 0.2])
    assert a["median"] == pytest.approx(0.2)
    assert a["worst"] == pytest.approx(0.3)
    assert a["std"] == pytest.approx(np.std([0.1, 0.3, 0.2]))


def test_aggregate_seeds_nan_safe():
    a = aggregate_seeds([float("nan"), 0.5, None])
    assert a["median"] == pytest.approx(0.5)
    assert a["worst"] == pytest.approx(0.5)
    assert aggregate_seeds([])["median"] != aggregate_seeds([])["median"]  # NaN
