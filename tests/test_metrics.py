"""Golden tests for the M1..M7 metric computation in plot_telemetry.

These scalar metrics become the thesis figures, so a regression in how they
are computed would silently corrupt published numbers. The tests feed a small
synthetic telemetry frame with hand-computable expected values and also pin
the documented ``e_tau_real``-over-``e_tau`` preference (critical for fair
cross-method comparison when dual_pulse is active).
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

# Ensure repo root is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plot_telemetry import MetricParams, compute_metrics  # noqa: E402


def _params(**overrides):
    base = dict(dt=0.1, vmax_xy=10.0, t0=0.0, e_thr=0.05, ma_w=1.0, settle_window=0.5)
    base.update(overrides)
    return MetricParams(**base)


def _frame(err_value=0.02, v=5.0, n_samples=10, err_col="e_tau", extra_cols=None):
    # One node, timestamps 0.1 .. n*0.1 (strictly inside the t > t0=0 window),
    # constant error and speed so the metrics are hand-computable.
    rows = []
    for k in range(1, n_samples + 1):
        row = {"node_id": 0, "timestamp": round(0.1 * k, 6), err_col: err_value,
               "u": 0.0, "velocity_norm": v}
        if extra_cols:
            row.update(extra_cols)
        rows.append(row)
    return pd.DataFrame(rows)


def test_basic_metrics_match_hand_values():
    df = _frame(err_value=0.02, v=5.0)
    m = compute_metrics(df, _params())
    assert m["M1_P95_e_pooled"] == pytest.approx(0.02)
    assert m["M5_mean_v2"] == pytest.approx(0.25)        # (5/10)^2
    assert m["M6_Pr_sat"] == pytest.approx(0.0)           # 5 < vmax
    # Error stays below e_thr for the whole window -> settles at the first
    # in-window sample (t=0.1), reported relative to t0=0.
    assert m["M7_settle_median"] == pytest.approx(0.1)
    assert m["M7_settled_frac"] == pytest.approx(1.0)


def test_saturation_metric_counts_at_limit_samples():
    df = _frame(err_value=0.02, v=10.0)   # exactly at vmax
    m = compute_metrics(df, _params())
    assert m["M6_Pr_sat"] == pytest.approx(1.0)
    assert m["M5_mean_v2"] == pytest.approx(1.0)


def test_never_settles_when_error_above_threshold():
    df = _frame(err_value=0.2, v=5.0)     # always above e_thr=0.05
    m = compute_metrics(df, _params())
    assert m["M7_settled_frac"] == pytest.approx(0.0)
    assert m["M1_P95_e_pooled"] == pytest.approx(0.2)


def test_prefers_e_tau_real_over_e_tau():
    # e_tau is the VIRTUAL (gap-biased) error; e_tau_real is physical. The
    # metrics must use e_tau_real when present.
    df = _frame(err_value=0.5, v=5.0, extra_cols={"e_tau_real": 0.02})
    m = compute_metrics(df, _params())
    assert m["M1_P95_e_pooled"] == pytest.approx(0.02)   # from e_tau_real, not 0.5


def test_falls_back_to_e_tau_when_no_real_column():
    df = _frame(err_value=0.03, v=5.0)    # only e_tau present
    m = compute_metrics(df, _params())
    assert m["M1_P95_e_pooled"] == pytest.approx(0.03)


def test_missing_required_columns_raises():
    df = pd.DataFrame({"node_id": [0], "timestamp": [0.1]})  # no e_tau / velocity_norm
    with pytest.raises(ValueError):
        compute_metrics(df, _params())
