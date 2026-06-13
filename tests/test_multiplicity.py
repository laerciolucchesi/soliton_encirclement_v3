"""Tests for the M-mult adjacent-block multiplicity (campaign Ciclo 2).

The SAIDA receiver delta uses n_old = n_new + k (k = number of adjacent
removals stamped on the pulse). These tests lock the algebra BEFORE the sim
benchmark:
  - k=1 (and a pulse with NO k field) reproduces the legacy single-removal
    delta exactly (regression — k=1 must never change behavior);
  - k=2/3 match the analytic n_old=n_new+k formula and grow in magnitude.

The hop-distance alpha attenuation is forced to 1.0 so the assertions check the
raw delta algebra (the alpha curve has its own tests).
"""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import dual_pulse_layer as dpl  # noqa: E402
from dual_pulse_layer import DualPulseLayer  # noqa: E402
from config_param import DUAL_PULSE_DELTA_SCALE  # noqa: E402

DPS = float(DUAL_PULSE_DELTA_SCALE)


@pytest.fixture
def no_hop_alpha(monkeypatch):
    monkeypatch.setattr(dpl, "DUAL_PULSE_ALPHA_CLOSE_RATIO", 1.0)
    monkeypatch.setattr(dpl, "DUAL_PULSE_ALPHA_CURVE_POWER", 1.0)
    yield


def _receiver_shift(k, h_ccw=2, h_cw=3, owner=99, with_k_field=True):
    """Drive a receiver layer with both directions of one SAIDA pulse; return shift_target."""
    layer = DualPulseLayer()
    layer.set_owner_id(owner)
    for direction, h in (("CCW", h_ccw), ("CW", h_cw)):
        pulse = {
            "event_id": [7, 1], "event_type": "SAIDA", "hop_count": h,
            "direction": direction, "originator_id": 7, "recovered_id": None,
            "n_stamp": None,
        }
        if with_k_field:
            pulse["k"] = k
        layer._process_received_pulse(pulse)
    return layer.shift_target


def _analytic_delta(k, h_ccw=2, h_cw=3):
    n_total = h_ccw + h_cw + 1
    n_new = n_total - 1
    n_old = n_new + k
    gap_old = 2.0 * math.pi / n_old
    gap_new = 2.0 * math.pi / n_new
    h_anchor = n_old / 2.0
    return (h_ccw - h_anchor) * (gap_new - gap_old) * DPS  # alpha forced to 1.0


def test_k1_matches_analytic_single_removal(no_hop_alpha):
    assert _receiver_shift(1) == pytest.approx(_analytic_delta(1), rel=1e-9)


def test_missing_k_field_is_treated_as_k1(no_hop_alpha):
    # Back-compat: a pulse from an old sender without the "k" field == k=1.
    assert _receiver_shift(1, with_k_field=False) == pytest.approx(_receiver_shift(1), rel=1e-12)


def test_k2_k3_match_analytic_and_grow(no_hop_alpha):
    s1, s2, s3 = _receiver_shift(1), _receiver_shift(2), _receiver_shift(3)
    assert s2 == pytest.approx(_analytic_delta(2), rel=1e-9)
    assert s3 == pytest.approx(_analytic_delta(3), rel=1e-9)
    # More adjacent removals -> larger redistribution shift (same sign).
    assert abs(s2) > abs(s1) > 0.0
    assert abs(s3) > abs(s2)


def test_originator_self_shift_uses_k(no_hop_alpha):
    # The self-originated returning pulse (hop_count = alive ring size) applies
    # delta_orig = (gap_old - gap_new/2) with gap_old = 2pi/(n_new+k).
    def orig_shift(k, n_new=5):
        layer = DualPulseLayer()
        layer.set_owner_id(7)
        eid = (7, 1)
        layer._self_originated[eid] = {"event_type": "SAIDA", "applied": False}
        layer._process_received_pulse({
            "event_id": [7, 1], "event_type": "SAIDA", "hop_count": n_new,
            "direction": "CCW", "originator_id": 7, "recovered_id": None,
            "n_stamp": None, "k": k,
        })
        return layer.shift_target

    def analytic_orig(k, n_new=5):
        n_old = n_new + k
        gap_old = 2.0 * math.pi / n_old
        gap_new = 2.0 * math.pi / n_new
        return (gap_old - gap_new / 2.0) * DPS  # alpha at distance 1 == 1.0 with no_hop_alpha

    assert orig_shift(1) == pytest.approx(analytic_orig(1), rel=1e-9)
    assert orig_shift(2) == pytest.approx(analytic_orig(2), rel=1e-9)
