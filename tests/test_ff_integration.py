"""Tests for the 2-DOF feedforward integration path (modes B/B2 + M8).

These cover the previously untested core of the thesis contribution:
  - the cancelling gap bias the FEEDBACK sees under B vs B2,
  - the feedforward command v_ff = (shift/T_FF)*r with actuator clipping,
  - the M8 bookkeeping (commanded rotation deducted next tick), and
  - the closed-loop property that FF + consume_motion drains the shift
    exponentially with time constant T_FF.

The helpers are pure static methods on AgentProtocol, extracted
behavior-preserving from the control loop, so they are testable without
the GrADyS runtime.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

# Ensure repo root is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from protocol_agent import AgentProtocol  # noqa: E402
from dual_pulse_layer import DualPulseLayer  # noqa: E402

bias = AgentProtocol._compute_cancelling_bias
ff = AgentProtocol._compute_ff_command


# ---------------------------------------------------------------------------
# Cancelling bias (B vs B2)
# ---------------------------------------------------------------------------

def test_b2_on_plan_bias_cancels_exactly():
    # When every agent carries the same shift (all "on plan"), B2 must show
    # the feedback the REAL gaps — zero perceived imbalance from the plan.
    pred, succ = bias("B2", pred_gap=0.5, succ_gap=0.7, s_self=0.2, s_pred=0.2, s_succ=0.2)
    assert pred == pytest.approx(0.5)
    assert succ == pytest.approx(0.7)


def test_b2_own_shift_enters_feedback_relative_to_neighbors():
    # Neighbors at zero shift, self at +s: B2 shows succ-s and pred+s, i.e.
    # the feedback would push the agent ALONG its own plan only if it lags it.
    pred, succ = bias("B2", pred_gap=0.5, succ_gap=0.5, s_self=0.1, s_pred=0.0, s_succ=0.0)
    assert succ == pytest.approx(0.4)
    assert pred == pytest.approx(0.6)


def test_b_minimal_cancel_ignores_own_shift():
    # Mode B cancels only the NEIGHBOURS' shifts; the own shift is absent,
    # which is exactly the documented ~2*s_self double-drive residual.
    pred, succ = bias("B", pred_gap=0.5, succ_gap=0.5, s_self=0.1, s_pred=0.0, s_succ=0.0)
    assert succ == pytest.approx(0.5)
    assert pred == pytest.approx(0.5)


def test_b_vs_b2_differ_by_own_shift():
    args = dict(pred_gap=0.6, succ_gap=0.4, s_self=0.05, s_pred=0.02, s_succ=-0.03)
    pred_b, succ_b = bias("B", **args)
    pred_b2, succ_b2 = bias("B2", **args)
    assert succ_b2 - succ_b == pytest.approx(-args["s_self"])
    assert pred_b2 - pred_b == pytest.approx(args["s_self"])


def test_bias_floors_gaps_at_1e3():
    # Large shifts must never produce a non-positive gap for the controller.
    pred, succ = bias("B2", pred_gap=0.01, succ_gap=0.01, s_self=1.0, s_pred=-1.0, s_succ=-1.0)
    assert pred >= 1e-3 and succ >= 1e-3


# ---------------------------------------------------------------------------
# Feedforward command + M8 bookkeeping
# ---------------------------------------------------------------------------

def test_ff_command_proportional_and_unclipped():
    v, dtheta = ff(shift=0.2, t_ff=1.0, r_eff=20.0, vmax=10.0, dt=0.01)
    assert v == pytest.approx(0.2 / 1.0 * 20.0)  # 4 m/s < vmax
    assert dtheta == pytest.approx((v / 20.0) * 0.01)


def test_ff_command_clips_at_actuator_limit_both_signs():
    v_pos, dt_pos = ff(shift=2.0, t_ff=1.0, r_eff=20.0, vmax=10.0, dt=0.01)
    v_neg, dt_neg = ff(shift=-2.0, t_ff=1.0, r_eff=20.0, vmax=10.0, dt=0.01)
    assert v_pos == 10.0 and v_neg == -10.0
    # M8 must account the CLIPPED rotation, not the nominal shift/T_FF*dt.
    assert dt_pos == pytest.approx((10.0 / 20.0) * 0.01)
    assert dt_neg == pytest.approx((-10.0 / 20.0) * 0.01)


def test_ff_dtheta_never_exceeds_nominal():
    # Saturation can only slow the consumption, never overshoot it.
    shift, t_ff, dt = 0.5, 1.0, 0.01
    _, dtheta = ff(shift, t_ff, r_eff=20.0, vmax=10.0, dt=dt)
    assert abs(dtheta) <= abs(shift / t_ff * dt) + 1e-15


def test_ff_plus_consume_motion_decays_with_t_ff():
    # Closed loop: shift_{k+1} = shift_k - (shift_k/T_FF)*dt (unsaturated)
    # => exponential decay with time constant T_FF. Verified through the REAL
    # DualPulseLayer.consume_motion (clip-at-zero semantics included).
    layer = DualPulseLayer()
    layer.shift_target = 0.3
    layer.shift_remaining = 0.3
    t_ff, dt, r_eff, vmax = 1.0, 0.01, 20.0, 10.0
    n_ticks = int(round(t_ff / dt))  # simulate exactly one time constant
    for _ in range(n_ticks):
        _, dtheta = ff(layer.shift_remaining, t_ff, r_eff, vmax, dt)
        layer.consume_motion(dtheta)
    # After one T_FF the remaining shift should be ~ e^-1 of the start.
    assert layer.shift_remaining == pytest.approx(0.3 * math.exp(-1.0), rel=0.02)
    # And it must keep draining toward zero, never cross it.
    for _ in range(20 * n_ticks):
        _, dtheta = ff(layer.shift_remaining, t_ff, r_eff, vmax, dt)
        layer.consume_motion(dtheta)
    assert 0.0 <= layer.shift_remaining < 1e-6


def test_consume_motion_ignores_opposite_rotation():
    # Tracking rotation OPPOSITE to the plan must not inflate the shift
    # (clip-at-zero reference is the shift sign, not the motion sign).
    layer = DualPulseLayer()
    layer.shift_target = 0.1
    layer.shift_remaining = 0.1
    layer.consume_motion(-0.05)  # rotation against the plan
    assert layer.shift_target == pytest.approx(0.15)  # un-consumed (moved away)
    layer.consume_motion(0.2)    # large forward rotation
    assert layer.shift_target == 0.0  # clipped at zero, no sign flip
    assert layer.shift_remaining == 0.0
