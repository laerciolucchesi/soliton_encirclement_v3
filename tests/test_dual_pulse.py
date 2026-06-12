"""Tests for DualPulseLayer (counter-propagating pulses with hop counts).

Covers:
  1. inject_pulse queues both directions with shared event_id
  2. Forwarded relay path: dedupe by (event_id, direction) refractory cache
  3. Event completion (both CCW and CW received) triggers delta_D
  4. Originator self-shift on returning pulse (delta_orig once, no double-apply)
  5. consume_motion clips at zero crossing
  6. on_reset clears all in-flight state

Plus an end-to-end smoke test on a 5-node ring with a simulated SAIDA event
that verifies every surviving node arrives at its expected delta_D / delta_orig.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

# Ensure repo root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dual_pulse_layer import DualPulseLayer, apply_shift_to_gaps  # noqa: E402
from propagation_layer import create_propagation_layer  # noqa: E402
from config_param import DUAL_PULSE_DELTA_SCALE  # noqa: E402

# Scale applied to delta_D / delta_orig at runtime. Tests that check exact
# magnitudes multiply their expected values by this so the assertions stay
# valid regardless of the project's configured default.
DPS = float(DUAL_PULSE_DELTA_SCALE)


@pytest.fixture
def no_hop_alpha(monkeypatch):
    """Force the hop-distance alpha factor to be 1.0 (i.e. no attenuation).

    Used by tests that verify the raw algebra of delta_D / delta_orig
    accumulation. The hop attenuation is exercised separately by the
    test_hop_alpha_* tests; mixing both concerns into one assertion would
    couple the algebra check to the project's default ALPHA_CLOSE_RATIO.
    """
    import dual_pulse_layer as dpl
    monkeypatch.setattr(dpl, "DUAL_PULSE_ALPHA_CLOSE_RATIO", 1.0)
    monkeypatch.setattr(dpl, "DUAL_PULSE_ALPHA_CURVE_POWER", 1.0)
    yield


DT = 0.01


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _drain_broadcast(layer: DualPulseLayer) -> dict:
    """Drain the layer's outgoing broadcast (the dict goes into AgentState.prop_state)."""
    return layer.get_broadcast_state()


def _ring_step(layers, broadcasts):
    """Advance one tick on a ring of N layers given the previous-tick broadcasts.

    broadcasts[i] is the dict layer i broadcast at the end of the previous
    tick. layer i reads broadcasts[(i-1) % N] from its pred slot and
    broadcasts[(i+1) % N] from its succ slot.
    Returns the new list of broadcasts produced this tick.
    """
    n = len(layers)
    for i, layer in enumerate(layers):
        layer.update(0.0, DT, broadcasts[(i - 1) % n], broadcasts[(i + 1) % n])
    return [layer.get_broadcast_state() for layer in layers]


# ---------------------------------------------------------------------------
# 1. Canonical inject queues both directions with the same event_id
# ---------------------------------------------------------------------------

def test_inject_queues_two_directions_with_shared_event_id():
    layer = DualPulseLayer()
    layer.inject_pulse(originator_id=42)

    state = _drain_broadcast(layer)
    pulses = state.get("pulses", [])
    assert len(pulses) == 2, f"expected 2 pulses (CCW+CW), got {len(pulses)}"

    directions = sorted(p["direction"] for p in pulses)
    assert directions == ["CCW", "CW"], directions

    event_ids = {tuple(p["event_id"]) for p in pulses}
    assert len(event_ids) == 1, "CCW and CW pulses must share the same event_id"
    eid = event_ids.pop()
    assert eid[0] == 42, f"originator id mismatch: {eid}"

    for p in pulses:
        assert p["hop_count"] == 1
        assert p["event_type"] == "SAIDA"
        assert p["originator_id"] == 42

    # Each pulse is broadcast BROADCAST_REPEATS=2 times for robustness against
    # intra-tick agent firing order. Second drain still shows the same pulses.
    state2 = _drain_broadcast(layer)
    assert len(state2.get("pulses", [])) == 2, "second broadcast should still carry the pulses"
    # After the second broadcast, the buffer is empty.
    assert _drain_broadcast(layer) == {}


# ---------------------------------------------------------------------------
# 2. Forwarded relay path with refractory dedupe
# ---------------------------------------------------------------------------

def test_pulse_forward_with_refractory():
    """A non-originator relay should forward an incoming pulse once and drop duplicates."""
    layer = DualPulseLayer()  # this is a relay, not the originator

    pulse_in = {
        "event_id": [99, 1],
        "event_type": "SAIDA",
        "hop_count": 1,
        "direction": "CCW",
        "originator_id": 99,
    }

    # 1st arrival: relay forwards.
    layer.update(0.0, DT, {"pulses": [pulse_in]}, {"pulses": []})
    state = _drain_broadcast(layer)
    fwd = state.get("pulses", [])
    assert len(fwd) == 1
    assert fwd[0]["direction"] == "CCW"
    assert fwd[0]["hop_count"] == 2
    assert tuple(fwd[0]["event_id"]) == (99, 1)

    # 2nd broadcast still carries the same forwarded pulse (BROADCAST_REPEATS=2).
    state_repeat = _drain_broadcast(layer)
    fwd_repeat = state_repeat.get("pulses", [])
    assert len(fwd_repeat) == 1, "second broadcast should still carry the forwarded pulse"
    assert fwd_repeat[0]["hop_count"] == 2, "hop_count must not change between repeats"

    # 2nd arrival of the same (event_id, direction): dropped by refractory cache.
    # No new forward is queued.
    layer.update(0.0, DT, {"pulses": [pulse_in]}, {"pulses": []})
    state2 = _drain_broadcast(layer)
    assert state2 == {}, f"duplicate pulse should not produce broadcast, got {state2}"


# ---------------------------------------------------------------------------
# 3. Event completion (both directions received) computes delta_D
# ---------------------------------------------------------------------------

def test_event_completion_triggers_delta_D(no_hop_alpha):
    """When both CCW and CW arrive at a relay node, delta_D is accumulated."""
    layer = DualPulseLayer()

    # Test layer at h_CCW=1, h_CW=3 -> N_old=5, N_new=4, h_anchor=2.5
    # delta_D = (1 - 2.5) * (2*pi/4 - 2*pi/5) = -1.5 * (pi/10) = -3*pi/20
    pulse_ccw = {
        "event_id": [7, 11], "event_type": "SAIDA", "hop_count": 1,
        "direction": "CCW", "originator_id": 7,
    }
    pulse_cw = {
        "event_id": [7, 11], "event_type": "SAIDA", "hop_count": 3,
        "direction": "CW", "originator_id": 7,
    }

    layer.update(0.0, DT, {"pulses": [pulse_ccw]}, {"pulses": [pulse_cw]})

    expected_delta = -1.5 * (2 * math.pi / 4 - 2 * math.pi / 5) * DPS
    # Check shift_target (the raw accumulator) rather than shift_remaining,
    # because the layer ramps shift_remaining toward shift_target over
    # DUAL_PULSE_RAMP_TICKS update() calls. shift_target is set on the
    # same tick as the formula computation.
    assert math.isclose(layer.get_shift_target(), expected_delta, abs_tol=1e-9), (
        f"shift_target={layer.get_shift_target()}, expected={expected_delta}"
    )

    completed = layer.get_completed_events_and_clear()
    assert len(completed) == 1
    ev = completed[0]
    assert ev["event_id"] == (7, 11)
    assert ev["h_CCW"] == 1
    assert ev["h_CW"] == 3
    assert ev["N_new"] == 4
    assert math.isclose(ev["delta_D"], expected_delta, abs_tol=1e-9)
    assert ev["is_originator"] is False

    # Re-delivery of either direction: should NOT re-apply delta_D.
    target_before = layer.get_shift_target()
    layer.update(0.0, DT, {"pulses": [pulse_ccw]}, {"pulses": []})
    assert layer.get_shift_target() == target_before, (
        "double-application of delta_D after both directions already processed"
    )


# ---------------------------------------------------------------------------
# 4. Originator self-shift on returning pulse
# ---------------------------------------------------------------------------

def test_originator_self_shift_on_returning_pulse(no_hop_alpha):
    """When the originator's own pulse comes back (hop_count = N_new), delta_orig is applied once."""
    layer = DualPulseLayer()
    layer.inject_pulse(originator_id=2)
    # Drain BROADCAST_REPEATS=2 times to fully clear the originator's outgoing buffer
    # before feeding the returning pulse, so any new pulse in get_broadcast_state
    # would be unambiguously a forward of the returning pulse.
    for _ in range(layer.BROADCAST_REPEATS):
        _ = _drain_broadcast(layer)
    assert _drain_broadcast(layer) == {}, "outgoing buffer should be empty after full drain"

    # Simulate the CCW pulse circling a 4-node ring (N_new=4) and arriving back.
    returning_ccw = {
        "event_id": [2, 1],
        "event_type": "SAIDA",
        "hop_count": 4,        # full loop = N_new
        "direction": "CCW",
        "originator_id": 2,
    }
    layer.update(0.0, DT, {"pulses": [returning_ccw]}, {"pulses": []})

    # delta_orig = gap_old - gap_new/2 = 2*pi/5 - (2*pi/4)/2 = 2*pi/5 - pi/4
    expected_orig = ((2 * math.pi / 5) - (2 * math.pi / 4) / 2.0) * DPS
    assert math.isclose(layer.get_shift_target(), expected_orig, abs_tol=1e-9), (
        f"shift_target={layer.get_shift_target()}, expected={expected_orig}"
    )

    # Originator MUST NOT forward its own returning pulse — outgoing buffer
    # stays empty (we already drained the originals before feeding the return).
    state_after = _drain_broadcast(layer)
    assert state_after == {}, f"originator forwarded its own pulse: {state_after}"

    # The CW pulse also returns later -> no second application.
    target_before = layer.get_shift_target()
    returning_cw = {
        "event_id": [2, 1],
        "event_type": "SAIDA",
        "hop_count": 4,
        "direction": "CW",
        "originator_id": 2,
    }
    layer.update(0.0, DT, {"pulses": []}, {"pulses": [returning_cw]})
    assert layer.get_shift_target() == target_before, (
        "delta_orig was applied a second time on the other returning direction"
    )

    # Diagnostic event was recorded exactly once.
    completed = layer.get_completed_events_and_clear()
    assert len(completed) == 1
    assert completed[0]["is_originator"] is True


# ---------------------------------------------------------------------------
# 5. consume_motion clips at zero
# ---------------------------------------------------------------------------

def test_consume_motion_clips_at_zero():
    layer = DualPulseLayer()

    # No shift pending (target == remaining == 0): motion is ignored entirely.
    layer.consume_motion(0.5)
    assert layer.get_shift_remaining() == 0.0
    assert layer.get_shift_target() == 0.0

    # Positive shift: set BOTH target and applied to the same value to mimic
    # a fully-converged ramp. consume_motion drains both symmetrically.
    layer.shift_target = 0.3
    layer.shift_remaining = 0.3
    layer.consume_motion(0.1)
    assert math.isclose(layer.get_shift_remaining(), 0.2, abs_tol=1e-12)
    assert math.isclose(layer.get_shift_target(), 0.2, abs_tol=1e-12)

    # Overshoot: motion bigger than remaining shift clips at zero (does not flip sign).
    layer.consume_motion(1.0)
    assert layer.get_shift_remaining() == 0.0
    assert layer.get_shift_target() == 0.0

    # Negative shift, motion in the opposite (CW/negative) direction reduces magnitude.
    layer.shift_target = -0.3
    layer.shift_remaining = -0.3
    layer.consume_motion(-0.1)
    assert math.isclose(layer.get_shift_remaining(), -0.2, abs_tol=1e-12)
    layer.consume_motion(-1.0)
    assert layer.get_shift_remaining() == 0.0

    # Non-finite delta_theta is ignored (no NaN propagation).
    layer.shift_target = 0.4
    layer.shift_remaining = 0.4
    layer.consume_motion(float("nan"))
    assert math.isclose(layer.get_shift_remaining(), 0.4, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# 6. on_reset clears state
# ---------------------------------------------------------------------------

def test_reset_clears_state():
    layer = DualPulseLayer()
    layer.inject_pulse(originator_id=3)
    _ = _drain_broadcast(layer)

    pulse_in = {
        "event_id": [99, 5], "event_type": "SAIDA", "hop_count": 1,
        "direction": "CCW", "originator_id": 99,
    }
    layer.update(0.0, DT, {"pulses": [pulse_in]}, {"pulses": []})
    layer.shift_target = 0.7
    layer.shift_remaining = 0.7

    seq_before_reset = layer.local_seq

    layer.on_reset()

    assert layer.shift_target == 0.0
    assert layer.shift_remaining == 0.0
    assert len(layer.refractory_cache) == 0
    assert len(layer.pending_pulses_to_forward) == 0
    assert len(layer.received_by_event) == 0
    assert len(layer.processed_events) == 0
    assert len(layer._self_originated) == 0
    assert layer._completed_events == []
    # local_seq is intentionally NOT reset (must stay monotonic across recovery).
    assert layer.local_seq == seq_before_reset


# ---------------------------------------------------------------------------
# 7. End-to-end ring simulation: N_old=5 with a SAIDA, every survivor's shift matches expectation
# ---------------------------------------------------------------------------

def test_end_to_end_ring_n5_saida_redistributes_correctly(no_hop_alpha):
    """Simulate a 4-node ring (after a node-3 death from N_old=5).

    Nodes are indexed by their CCW position relative to A=originator:
      A (h=0), X1 (h=1), X2 (h=2), X3 (h=3).
    Verifies after enough ticks for both pulses to travel the full ring,
    every node's shift_remaining matches the analytical delta_D / delta_orig.
    """
    n_new = 4
    n_old = 5
    gap_old = 2.0 * math.pi / n_old
    gap_new = 2.0 * math.pi / n_new
    h_anchor = n_old / 2.0  # 2.5

    expected = {
        0: (gap_old - gap_new / 2.0) * DPS,                    # delta_orig at A
        1: (1 - h_anchor) * (gap_new - gap_old) * DPS,         # X1
        2: (2 - h_anchor) * (gap_new - gap_old) * DPS,         # X2
        3: (3 - h_anchor) * (gap_new - gap_old) * DPS,         # X3
    }

    layers = [DualPulseLayer() for _ in range(n_new)]
    layers[0].inject_pulse(originator_id=10)  # A's id is arbitrary

    broadcasts = [layer.get_broadcast_state() for layer in layers]
    # Run enough ticks for both pulses to traverse the full ring twice over.
    for _ in range(3 * n_new):
        broadcasts = _ring_step(layers, broadcasts)

    for i in range(n_new):
        # Check shift_target (raw accumulator) — independent of RAMP_TICKS,
        # which slows shift_remaining's convergence to the target.
        actual = layers[i].get_shift_target()
        assert math.isclose(actual, expected[i], abs_tol=1e-9), (
            f"node h={i}: shift={actual}, expected={expected[i]}"
        )


# ---------------------------------------------------------------------------
# Bonus: factory wires dual_pulse to DualPulseLayer
# ---------------------------------------------------------------------------

def test_factory_returns_dual_pulse_layer():
    layer = create_propagation_layer("dual_pulse", {})
    assert isinstance(layer, DualPulseLayer)
    # Default get_signal / get_neighbor_signal: both 0 (does not feed u_prop).
    assert layer.get_signal() == 0.0
    assert layer.get_neighbor_signal() == 0.0


# ---------------------------------------------------------------------------
# Bonus: apply_shift_to_gaps clipping behavior
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ENTRADA tests (drone returns -> ring grows -> symmetric formula)
# ---------------------------------------------------------------------------


def test_inject_entrada_queues_two_directions_with_recovered_id():
    """inject_entrada must queue CCW + CW pulses tagged ENTRADA with recovered_id."""
    layer = DualPulseLayer()
    layer.inject_entrada(originator_id=2, recovered_id=3)
    state = _drain_broadcast(layer)
    pulses = state.get("pulses", [])
    assert len(pulses) == 2
    assert sorted(p["direction"] for p in pulses) == ["CCW", "CW"]
    for p in pulses:
        assert p["event_type"] == "ENTRADA"
        assert p["originator_id"] == 2
        assert p["recovered_id"] == 3
        assert p["hop_count"] == 1


def test_entrada_recovered_node_passthrough():
    """The recovered drone forwards the ENTRADA pulse but does NOT apply delta to itself."""
    layer = DualPulseLayer()
    layer.set_owner_id(3)  # this layer belongs to the recovered drone

    incoming = {
        "event_id": [2, 1],
        "event_type": "ENTRADA",
        "hop_count": 1,
        "direction": "CCW",
        "originator_id": 2,
        "recovered_id": 3,  # this matches owner -> passthrough
    }
    # Recovered drone receives via pred_state (CCW from originator's broadcast).
    layer.update(0.0, DT, {"pulses": [incoming]}, {"pulses": []})

    # Forward must happen.
    state = _drain_broadcast(layer)
    fwd = state.get("pulses", [])
    assert len(fwd) == 1
    assert fwd[0]["direction"] == "CCW"
    assert fwd[0]["hop_count"] == 2
    assert fwd[0]["recovered_id"] == 3

    # No delta applied to self.
    assert layer.get_shift_remaining() == 0.0

    # Even after both directions arrive, no delta because passthrough.
    incoming_cw = {
        **incoming,
        "direction": "CW",
        "hop_count": 5,
    }
    layer.update(0.0, DT, {"pulses": []}, {"pulses": [incoming_cw]})
    assert layer.get_shift_remaining() == 0.0


def test_entrada_completion_delta_has_opposite_sign_of_saida(no_hop_alpha):
    """For the same agent (h_CCW), ENTRADA delta has opposite sign of SAIDA delta."""
    # Setup: receiver sees CCW from a drone at h=2 and CW from same drone at h=7
    # in a 10-node ring. For SAIDA: N_new=9, h_anchor=5; for ENTRADA: N_new=10,
    # h_anchor=6.
    layer_saida = DualPulseLayer()
    layer_entrada = DualPulseLayer()
    layer_saida.set_owner_id(99)
    layer_entrada.set_owner_id(99)

    # SAIDA: h_CCW=2, h_CW=7  -> N_new = 2+7+1 = 10? hmm, let's match formulas.
    # For test_event_completion_triggers_delta_D we used h_CCW=1, h_CW=3 -> N_old=5, N_new=4.
    # Use the same here for symmetry check: h_CCW=1, h_CW=3 in both events.
    pulses_saida = [
        {"event_id": [7, 1], "event_type": "SAIDA",   "hop_count": 1,
         "direction": "CCW", "originator_id": 7, "recovered_id": None},
        {"event_id": [7, 1], "event_type": "SAIDA",   "hop_count": 3,
         "direction": "CW",  "originator_id": 7, "recovered_id": None},
    ]
    pulses_entrada = [
        {"event_id": [7, 2], "event_type": "ENTRADA", "hop_count": 1,
         "direction": "CCW", "originator_id": 7, "recovered_id": 8},
        {"event_id": [7, 2], "event_type": "ENTRADA", "hop_count": 3,
         "direction": "CW",  "originator_id": 7, "recovered_id": 8},
    ]
    layer_saida.update(0.0, DT, {"pulses": [pulses_saida[0]]},
                              {"pulses": [pulses_saida[1]]})
    layer_entrada.update(0.0, DT, {"pulses": [pulses_entrada[0]]},
                                {"pulses": [pulses_entrada[1]]})

    s_shift = layer_saida.get_shift_target()
    e_shift = layer_entrada.get_shift_target()

    # SAIDA: h_total = h_CCW+h_CW+1 = 5 = N_old; N_new = 4. h_anchor = 5/2 = 2.5.
    # gap_old = 2pi/5, gap_new = 2pi/4.
    # delta = (1-2.5) * (gap_new-gap_old) * SCALE = -1.5 * (2pi/4 - 2pi/5) * SCALE.
    # ENTRADA: h_total = 5 = N_new; N_old = 4. h_anchor = 1 + 5/2 = 3.5.
    # gap_old = 2pi/4, gap_new = 2pi/5.
    # delta = (1-3.5) * (gap_new-gap_old) * SCALE = -2.5 * (2pi/5 - 2pi/4) * SCALE.
    # Note: gap_new-gap_old has opposite sign in the two events (since N_new
    # is bigger in ENTRADA than its N_old, smaller in SAIDA).
    expected_saida = -1.5 * (2 * math.pi / 4 - 2 * math.pi / 5) * DPS
    expected_entrada = -2.5 * (2 * math.pi / 5 - 2 * math.pi / 4) * DPS
    assert math.isclose(s_shift, expected_saida, abs_tol=1e-9), (
        f"SAIDA shift={s_shift}, expected={expected_saida}"
    )
    assert math.isclose(e_shift, expected_entrada, abs_tol=1e-9), (
        f"ENTRADA shift={e_shift}, expected={expected_entrada}"
    )

    # Time-reverse symmetry: a node that was a SAIDA receiver in event X and
    # an ENTRADA receiver in event ~X (same drone, same hop topology) should
    # see opposite signs. Here the "topology" differs by 1 hop because D is
    # back in ENTRADA, but the formulas guarantee delta(SAIDA, h_S=k) =
    # -delta(ENTRADA, h_E=k+1) for matching agents (verified in derivation).


def test_originator_entrada_self_shift_on_returning_pulse(no_hop_alpha):
    """ENTRADA originator applies delta_orig = (gap_old/2 - gap_new) * SCALE."""
    layer = DualPulseLayer()
    layer.set_owner_id(2)
    layer.inject_entrada(originator_id=2, recovered_id=3)
    # Drain originals so any new outgoing must come from the returning pulse path.
    for _ in range(layer.BROADCAST_REPEATS):
        _ = _drain_broadcast(layer)

    # Returning pulse: full loop in a 4-node ring (post-recovery). hop_count = N_new.
    returning = {
        "event_id": [2, 1],
        "event_type": "ENTRADA",
        "hop_count": 4,        # N_new = 4
        "direction": "CCW",
        "originator_id": 2,
        "recovered_id": 3,
    }
    layer.update(0.0, DT, {"pulses": [returning]}, {"pulses": []})

    # delta_orig_E = (gap_old/2 - gap_new) * SCALE
    # gap_old = 2*pi/3 (N_old=3), gap_new = 2*pi/4. gap_old/2 - gap_new = pi/3 - pi/2 = -pi/6.
    expected = ((2 * math.pi / 3) / 2.0 - 2 * math.pi / 4) * DPS
    assert math.isclose(layer.get_shift_target(), expected, abs_tol=1e-9), (
        f"shift_target={layer.get_shift_target()}, expected={expected}"
    )

    # Returning pulse must NOT be forwarded.
    assert _drain_broadcast(layer) == {}

    # The other direction's return: no second application.
    target_before = layer.get_shift_target()
    returning_cw = {**returning, "direction": "CW"}
    layer.update(0.0, DT, {"pulses": []}, {"pulses": [returning_cw]})
    assert layer.get_shift_target() == target_before


def test_saida_then_entrada_round_trip_zero_shift(no_hop_alpha):
    """SAIDA followed by ENTRADA on the same node returns shift_remaining to zero
    (modulo SCALE; for SCALE=1.0 exactly zero, for SCALE<1 a residue equal to
    the SCALE-attenuated SAIDA + ENTRADA contributions which still cancel out)."""
    # Use a clean layer that sees both events on its receiver path.
    layer = DualPulseLayer()
    layer.set_owner_id(99)

    # SAIDA: h_CCW=1, h_CW=3 -> N_new=4, N_old=5
    saida_ccw = {"event_id": [7, 1], "event_type": "SAIDA", "hop_count": 1,
                 "direction": "CCW", "originator_id": 7, "recovered_id": None}
    saida_cw = {"event_id": [7, 1], "event_type": "SAIDA", "hop_count": 3,
                "direction": "CW", "originator_id": 7, "recovered_id": None}
    layer.update(0.0, DT, {"pulses": [saida_ccw]}, {"pulses": [saida_cw]})
    shift_after_saida = layer.get_shift_target()

    # The drone D returns: for the same agent, h shifts by +1 (D is now in path).
    # ENTRADA at h_CCW=2, h_CW=3 -> N_new=6, N_old=5. Hmm — the test agent's
    # hops in the new ring depend on D's exact location. Use simulated values
    # that match the time-reverse: ENTRADA at h_CCW=2 would yield delta of
    # (2 - (1+6/2)) * (2pi/6 - 2pi/5) * SCALE = (-2) * (-pi/15) * SCALE
    # = 2*pi/15 * SCALE.
    # SAIDA at h=1 yielded (1 - 2.5) * (2pi/4 - 2pi/5) * SCALE = -1.5 * pi/10 * SCALE
    # = -3*pi/20 * SCALE.
    # These don't cancel exactly because the rings are different sizes.
    #
    # The cleaner round-trip test: identical N before and after, same h_CCW.
    # SAIDA going N=5->4 then ENTRADA going N=4->5 with same receiver topology
    # cancels by construction (formulas are exact inverses).
    entrada_ccw = {"event_id": [7, 2], "event_type": "ENTRADA", "hop_count": 2,
                   "direction": "CCW", "originator_id": 7, "recovered_id": 99 + 1}
    entrada_cw = {"event_id": [7, 2], "event_type": "ENTRADA", "hop_count": 2,
                  "direction": "CW", "originator_id": 7, "recovered_id": 99 + 1}
    # h_CCW + h_CW + 1 = 5 (back to original ring size)
    layer.update(0.0, DT, {"pulses": [entrada_ccw]}, {"pulses": [entrada_cw]})
    shift_after_round_trip = layer.get_shift_target()

    # Receiver was at h_S=1 (shift -1.5 * gap_diff_S * SCALE) in SAIDA;
    # at h_E=2 in ENTRADA. delta_E = (2 - (1+5/2)) * (gap_new_E - gap_old_E) * SCALE
    # = -1.5 * (2pi/5 - 2pi/4) * SCALE = -1.5 * (-pi/10) * SCALE = 1.5 * pi/10 * SCALE.
    # Sum: -1.5 * pi/10 * SCALE + 1.5 * pi/10 * SCALE = 0.
    assert math.isclose(shift_after_round_trip, 0.0, abs_tol=1e-9), (
        f"after SAIDA+ENTRADA round trip: shift={shift_after_round_trip}, "
        f"expected ~0 (SAIDA was {shift_after_saida})"
    )


# ---------------------------------------------------------------------------
# Hop-distance alpha attenuation
# ---------------------------------------------------------------------------


def test_hop_alpha_default_is_no_op():
    """With ALPHA_CLOSE_RATIO=1.0 (default), _hop_alpha returns 1.0 everywhere."""
    import dual_pulse_layer as dpl

    saved = dpl.DUAL_PULSE_ALPHA_CLOSE_RATIO
    try:
        dpl.DUAL_PULSE_ALPHA_CLOSE_RATIO = 1.0
        for n in (3, 5, 10, 20):
            for d in (1, 2, n // 2, n - 1):
                assert math.isclose(
                    DualPulseLayer._hop_alpha(distance_from_d=float(d), n_new=int(n)),
                    1.0, abs_tol=1e-12,
                )
    finally:
        dpl.DUAL_PULSE_ALPHA_CLOSE_RATIO = saved


def test_hop_alpha_linear_interpolation():
    """alpha(d=1) == ALPHA_CLOSE_RATIO; alpha(d=N/2) == 1.0; linear between."""
    import dual_pulse_layer as dpl

    saved = dpl.DUAL_PULSE_ALPHA_CLOSE_RATIO
    try:
        dpl.DUAL_PULSE_ALPHA_CLOSE_RATIO = 0.4
        n_new = 10  # half = 5

        # At distance 1 (immediate neighbor): alpha = 0.4 + 0.6 * (1/5) = 0.52.
        a1 = DualPulseLayer._hop_alpha(distance_from_d=1.0, n_new=n_new)
        assert math.isclose(a1, 0.4 + 0.6 * (1.0 / 5.0), abs_tol=1e-9)

        # At anchor (distance 5): alpha = 0.4 + 0.6 * 1 = 1.0.
        a_anchor = DualPulseLayer._hop_alpha(distance_from_d=5.0, n_new=n_new)
        assert math.isclose(a_anchor, 1.0, abs_tol=1e-9)

        # At midway (distance 3): alpha = 0.4 + 0.6 * (3/5) = 0.76.
        a_mid = DualPulseLayer._hop_alpha(distance_from_d=3.0, n_new=n_new)
        assert math.isclose(a_mid, 0.4 + 0.6 * (3.0 / 5.0), abs_tol=1e-9)

        # Distances beyond half clip at 1.0 (formula caps d_norm at 1).
        a_beyond = DualPulseLayer._hop_alpha(distance_from_d=8.0, n_new=n_new)
        assert math.isclose(a_beyond, 1.0, abs_tol=1e-9)
    finally:
        dpl.DUAL_PULSE_ALPHA_CLOSE_RATIO = saved


def test_hop_alpha_curve_power_p2_concentrates_attenuation_close():
    """With p=2, alpha at small d_norm is closer to ALPHA_CLOSE_RATIO than linear (p=1)."""
    import dual_pulse_layer as dpl
    saved_r = dpl.DUAL_PULSE_ALPHA_CLOSE_RATIO
    saved_p = dpl.DUAL_PULSE_ALPHA_CURVE_POWER
    try:
        dpl.DUAL_PULSE_ALPHA_CLOSE_RATIO = 0.4
        n_new = 10  # half = 5

        # p = 1: alpha(1) = 0.4 + 0.6 * (1/5) = 0.52
        dpl.DUAL_PULSE_ALPHA_CURVE_POWER = 1.0
        a_lin = DualPulseLayer._hop_alpha(distance_from_d=1.0, n_new=n_new)
        assert math.isclose(a_lin, 0.4 + 0.6 * 0.2, abs_tol=1e-9)

        # p = 2: alpha(1) = 0.4 + 0.6 * (1/5)^2 = 0.4 + 0.6 * 0.04 = 0.424
        dpl.DUAL_PULSE_ALPHA_CURVE_POWER = 2.0
        a_quad = DualPulseLayer._hop_alpha(distance_from_d=1.0, n_new=n_new)
        assert math.isclose(a_quad, 0.4 + 0.6 * 0.04, abs_tol=1e-9)
        # quadratic must be MORE attenuating (smaller alpha) than linear at small d.
        assert a_quad < a_lin

        # At anchor (d=5): both p give alpha = 1.0 exactly.
        a_quad_anchor = DualPulseLayer._hop_alpha(distance_from_d=5.0, n_new=n_new)
        assert math.isclose(a_quad_anchor, 1.0, abs_tol=1e-12)

        # Halfway (d=3, d_norm=0.6): p=2 gives 0.4 + 0.6 * 0.36 = 0.616 < linear's 0.76.
        a_quad_mid = DualPulseLayer._hop_alpha(distance_from_d=3.0, n_new=n_new)
        assert math.isclose(a_quad_mid, 0.4 + 0.6 * 0.36, abs_tol=1e-9)
    finally:
        dpl.DUAL_PULSE_ALPHA_CLOSE_RATIO = saved_r
        dpl.DUAL_PULSE_ALPHA_CURVE_POWER = saved_p


def test_hop_alpha_applied_to_receiver_delta_in_saida():
    """With ALPHA_CLOSE_RATIO < 1, an immediate-neighbor receiver gets a
    smaller delta_D than at default ALPHA_CLOSE_RATIO=1.0."""
    import dual_pulse_layer as dpl

    saved = dpl.DUAL_PULSE_ALPHA_CLOSE_RATIO
    try:
        # Reference: full delta_D at h_CCW=1 with no hop attenuation.
        dpl.DUAL_PULSE_ALPHA_CLOSE_RATIO = 1.0
        layer1 = DualPulseLayer()
        layer1.set_owner_id(99)
        layer1.update(0.0, DT,
                      {"pulses": [{"event_id": [7, 1], "event_type": "SAIDA",
                                   "hop_count": 1, "direction": "CCW",
                                   "originator_id": 7, "recovered_id": None}]},
                      {"pulses": [{"event_id": [7, 1], "event_type": "SAIDA",
                                   "hop_count": 3, "direction": "CW",
                                   "originator_id": 7, "recovered_id": None}]})
        full_delta = layer1.get_shift_target()

        # With ALPHA_CLOSE_RATIO=0.4, receiver at h=1 (distance 1 from D's spot)
        # should get approximately 0.4 + 0.6*(1/2) = 0.7 of the full delta.
        # n_new=4, half=2, distance=1, d_norm=0.5, alpha=0.4+0.6*0.5=0.7.
        dpl.DUAL_PULSE_ALPHA_CLOSE_RATIO = 0.4
        layer2 = DualPulseLayer()
        layer2.set_owner_id(99)
        layer2.update(0.0, DT,
                      {"pulses": [{"event_id": [7, 2], "event_type": "SAIDA",
                                   "hop_count": 1, "direction": "CCW",
                                   "originator_id": 7, "recovered_id": None}]},
                      {"pulses": [{"event_id": [7, 2], "event_type": "SAIDA",
                                   "hop_count": 3, "direction": "CW",
                                   "originator_id": 7, "recovered_id": None}]})
        attenuated_delta = layer2.get_shift_target()

        ratio = attenuated_delta / full_delta
        assert math.isclose(ratio, 0.7, abs_tol=1e-9), (
            f"Expected ratio 0.7, got {ratio}"
        )
    finally:
        dpl.DUAL_PULSE_ALPHA_CLOSE_RATIO = saved


def test_apply_shift_to_gaps_clip():
    # No shift => virtual gaps == real gaps.
    pg, sg = apply_shift_to_gaps(1.0, 0.5, 0.0)
    assert pg == 1.0 and sg == 0.5

    # Positive shift expands succ, contracts pred.
    pg, sg = apply_shift_to_gaps(1.0, 0.5, 0.2)
    assert math.isclose(pg, 0.8, abs_tol=1e-12)
    assert math.isclose(sg, 0.7, abs_tol=1e-12)

    # Excessive positive shift is clipped at GAP_CLIP_FRAC * min(real_gaps).
    # min(1.0, 0.5) = 0.5; default DUAL_PULSE_GAP_CLIP_FRAC = 0.8 -> max_shift = 0.4.
    pg, sg = apply_shift_to_gaps(1.0, 0.5, 99.0)
    assert math.isclose(pg, 0.6, abs_tol=1e-12)
    assert math.isclose(sg, 0.9, abs_tol=1e-12)

    # None inputs pass through untouched.
    pg, sg = apply_shift_to_gaps(None, 0.5, 0.2)
    assert pg is None and sg == 0.5


def test_apply_shift_to_gaps_sleep_threshold():
    """When |shift| < SLEEP_THRESHOLD the gap-bias step is skipped (sleeping mode)."""
    import dual_pulse_layer as dpl

    saved = dpl.DUAL_PULSE_SLEEP_THRESHOLD
    try:
        # Use a clearly visible threshold for the test.
        dpl.DUAL_PULSE_SLEEP_THRESHOLD = 0.05

        # Shift below threshold: gaps unchanged (sleeping).
        pg, sg = apply_shift_to_gaps(1.0, 0.5, 0.03)
        assert pg == 1.0 and sg == 0.5

        # Shift exactly at threshold: bias APPLIED (the gate is strict <).
        pg, sg = apply_shift_to_gaps(1.0, 0.5, 0.05)
        assert math.isclose(pg, 0.95, abs_tol=1e-12)
        assert math.isclose(sg, 0.55, abs_tol=1e-12)

        # Shift just above threshold: gaps biased.
        pg, sg = apply_shift_to_gaps(1.0, 0.5, 0.06)
        assert math.isclose(pg, 0.94, abs_tol=1e-12)
        assert math.isclose(sg, 0.56, abs_tol=1e-12)

        # Negative shift below |threshold|: also sleeps.
        pg, sg = apply_shift_to_gaps(1.0, 0.5, -0.02)
        assert pg == 1.0 and sg == 0.5

        # Threshold = 0 disables the gate (recovers v1.6 behaviour).
        dpl.DUAL_PULSE_SLEEP_THRESHOLD = 0.0
        pg, sg = apply_shift_to_gaps(1.0, 0.5, 0.001)
        assert math.isclose(pg, 0.999, abs_tol=1e-12)
        assert math.isclose(sg, 0.501, abs_tol=1e-12)
    finally:
        dpl.DUAL_PULSE_SLEEP_THRESHOLD = saved
