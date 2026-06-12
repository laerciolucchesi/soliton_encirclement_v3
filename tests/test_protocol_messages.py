"""Tests for the JSON (de)serialization of the broadcast message types.

These messages cross the GrADyS communication medium as JSON strings, so a
round-trip (to_json -> from_json) must preserve every field the protocols
read back. A silent field/type regression here surfaces only as a confusing
mid-run failure, which makes these cheap round-trip tests high-value.
"""

from __future__ import annotations

import os
import sys

import pytest

# Ensure repo root is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from protocol_messages import AdversaryState, AgentState, TargetState  # noqa: E402


def test_agent_state_round_trip_preserves_all_fields():
    original = AgentState(
        agent_id=7,
        seq=42,
        position=(1.5, -2.0, 0.0),
        velocity=(0.1, 0.2, -0.3),
        u=0.55,
        u_ss=-0.125,
        prop_state={"u_R": 0.3, "pulses": [{"dir": "CCW", "hop": 2}]},
        fast_state={"u_R": 0.0, "u_L": 1.0},
        dp_shift=0.07,
    )
    restored = AgentState.from_json(original.to_json())

    assert restored.agent_id == 7
    assert restored.seq == 42
    assert restored.position == (1.5, -2.0, 0.0)
    assert restored.velocity == (0.1, 0.2, -0.3)
    assert restored.u == pytest.approx(0.55)
    assert restored.u_ss == pytest.approx(-0.125)
    assert restored.dp_shift == pytest.approx(0.07)
    # prop_state / fast_state survive as plain JSON-native structures.
    assert restored.prop_state == original.prop_state
    assert restored.fast_state == original.fast_state


def test_agent_state_defaults_normalize_optional_dicts():
    # prop_state / fast_state default to {} (never None) so consumers can index.
    restored = AgentState.from_json(
        AgentState(agent_id=1, seq=0, position=(0, 0, 0), velocity=(0, 0, 0), u=0.0).to_json()
    )
    assert restored.prop_state == {}
    assert restored.fast_state == {}
    assert restored.dp_shift == 0.0


def test_json_tuple_in_prop_state_becomes_list():
    # Documents a real JSON limitation the dual_pulse consumers work around:
    # tuple event_ids serialize to lists and must be re-tupled on read.
    state = AgentState(
        agent_id=3, seq=1, position=(0, 0, 0), velocity=(0, 0, 0), u=0.0,
        prop_state={"pulses": [{"event_id": (3, 5)}]},
    )
    restored = AgentState.from_json(state.to_json())
    assert restored.prop_state["pulses"][0]["event_id"] == [3, 5]
    assert tuple(restored.prop_state["pulses"][0]["event_id"]) == (3, 5)


def test_target_state_round_trip():
    original = TargetState(
        target_id=0,
        seq=9,
        position=(0.0, 0.0, 0.0),
        velocity=(1.0, 0.0, 0.0),
        alive_lambdas={2: 1.0, 3: 27.0},
        omega_ref=0.25,
    )
    restored = TargetState.from_json(original.to_json())

    assert restored.target_id == 0
    assert restored.seq == 9
    assert restored.position == (0.0, 0.0, 0.0)
    assert restored.velocity == (1.0, 0.0, 0.0)
    assert restored.omega_ref == pytest.approx(0.25)
    # JSON object keys are always strings: alive_lambda keys come back as str.
    assert {str(k): v for k, v in original.alive_lambdas.items()} == restored.alive_lambdas


def test_adversary_state_round_trip():
    original = AdversaryState(node_id=1, seq=4, position=(40.0, 40.0, 0.0), velocity=(0.0, 0.0, 0.0))
    restored = AdversaryState.from_json(original.to_json())
    assert restored.node_id == 1
    assert restored.seq == 4
    assert restored.position == (40.0, 40.0, 0.0)
    assert restored.velocity == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("cls", [AgentState, TargetState, AdversaryState])
def test_from_json_rejects_wrong_type(cls):
    # A message of the wrong TYPE must raise rather than silently mis-decode.
    with pytest.raises(ValueError):
        cls.from_json('{"type": "NotARealType"}')
