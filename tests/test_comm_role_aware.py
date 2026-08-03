"""Tests for the role-aware (asymmetric) communication ranges.

The motivating requirement: an agent's AgentState is ONE broadcast serving two
audiences with opposite range needs -- the ring neighbours (short, for physical
locality) and the target (must hear everyone, or it prunes live agents as dead).
GrADyS evaluates range at the sender only, so this is expressed per link, in
RoleAwareCommunicationHandler._transmit_message.

The fixtures below drive the real handler (real register_node, real
handle_command, real can_transmit) against stub nodes and a stub event loop, so
these are delivery tests, not mock assertions.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gradysim.protocol.messages.communication import (  # noqa: E402
    CommunicationCommand,
    CommunicationCommandType,
)
from gradysim.simulator.handler.communication import CommunicationMedium  # noqa: E402
from gradysim.simulator.node import Node  # noqa: E402

from comm_role_aware import (  # noqa: E402
    ROLE_ADVERSARY,
    ROLE_AGENT,
    ROLE_TARGET,
    ROLE_UNKNOWN,
    RoleAwareCommunicationHandler,
    agent_target_range_matrix,
)

R = 20.0  # ENCIRCLEMENT_RADIUS used by the ring fixtures
N = 10    # NUM_AGENTS


# --------------------------------------------------------------------------
# stubs
# --------------------------------------------------------------------------

# Named to match DEFAULT_PROTOCOL_ROLES: role resolution is by class NAME, which
# is what keeps comm_role_aware free of imports from the protocol modules.
class AgentProtocol:
    pass


class TargetProtocol:
    pass


class AdversaryProtocol:
    pass


class OrphanProtocol:
    pass


class _StubEncapsulator:
    """Minimal stand-in for PythonEncapsulator: a protocol plus a packet sink."""

    def __init__(self, protocol, inbox):
        self.protocol = protocol
        self._inbox = inbox

    def handle_packet(self, message: str) -> None:
        self._inbox.append(message)


class _ImmediateEventLoop:
    """Runs scheduled callbacks inline so deliveries are observable synchronously."""

    current_time = 0.0

    def schedule_event(self, timestamp, callback, context=""):
        callback()


def _make_node(node_id: int, position, protocol_cls, inboxes):
    node = Node()
    node.id = node_id
    node.position = position
    inboxes[node_id] = []
    node.protocol_encapsulator = _StubEncapsulator(protocol_cls(), inboxes[node_id])
    return node


def _ring_position(k: int, num_agents: int = N, radius: float = R):
    angle = 2.0 * math.pi * k / num_agents
    return (radius * math.cos(angle), radius * math.sin(angle), 0.0)


def chord(k_hops: int, num_agents: int = N, radius: float = R) -> float:
    """Distance between two ring slots ``k_hops`` apart."""
    return 2.0 * radius * math.sin(k_hops * math.pi / num_agents)


@pytest.fixture
def ring():
    """Target at the origin, adversary far out, N agents on the ring.

    Returns ``(handler, nodes, inboxes, build)`` where ``build`` installs a
    handler with a given matrix. Node ids follow main.py: target 0, adversary 1,
    agents 2..N+1.
    """
    inboxes: dict[int, list] = {}
    nodes = {0: _make_node(0, (0.0, 0.0, 0.0), TargetProtocol, inboxes),
             1: _make_node(1, (40.0, 40.0, 0.0), AdversaryProtocol, inboxes)}
    for k in range(N):
        nodes[2 + k] = _make_node(2 + k, _ring_position(k), AgentProtocol, inboxes)

    def build(range_matrix=None, default_range=200.0, **kwargs):
        handler = RoleAwareCommunicationHandler(
            CommunicationMedium(transmission_range=default_range),
            range_matrix=range_matrix,
            **kwargs,
        )
        handler.inject(_ImmediateEventLoop())
        for node in nodes.values():
            handler.register_node(node)
        for inbox in inboxes.values():
            inbox.clear()
        return handler

    return nodes, inboxes, build


def _broadcast(handler, nodes, sender_id: int, message: str = "m") -> None:
    handler.handle_command(
        CommunicationCommand(CommunicationCommandType.BROADCAST, message),
        nodes[sender_id],
    )


def _received_by(inboxes) -> set:
    return {node_id for node_id, inbox in inboxes.items() if inbox}


# --------------------------------------------------------------------------
# geometry the sizing rule rests on
# --------------------------------------------------------------------------

def test_ring_chords_bracket_the_thirty_meter_choice():
    # The geometry the range sweep is read against: 30 m is a 2-hop ring at
    # N=10, R=20. Which chord actually bounds the mechanism is an empirical
    # question, not a geometric one -- see the note in config_param: the
    # measured cliff sits near the 1-hop chord, not the 2-hop one.
    assert chord(1) == pytest.approx(12.36, abs=0.01)
    assert chord(2) == pytest.approx(23.51, abs=0.01)
    assert chord(3) == pytest.approx(32.36, abs=0.01)
    assert chord(2) < 30.0 < chord(3)
    assert 20.0 < chord(2)


# --------------------------------------------------------------------------
# role resolution
# --------------------------------------------------------------------------

def test_roles_resolved_from_protocol_class_name(ring):
    nodes, _inboxes, build = ring
    handler = build()
    assert handler.get_role(0) == ROLE_TARGET
    assert handler.get_role(1) == ROLE_ADVERSARY
    assert handler.get_role(2) == ROLE_AGENT


def test_unregistered_and_unknown_protocols_fall_back_to_unknown(ring):
    nodes, inboxes, build = ring
    handler = build()
    assert handler.get_role(999) == ROLE_UNKNOWN

    orphan = _make_node(50, (0.0, 0.0, 0.0), OrphanProtocol, inboxes)
    handler.register_node(orphan)
    assert handler.get_role(50) == ROLE_UNKNOWN
    # Unknown roles are not in the matrix, so they keep the default range.
    assert handler.range_for(50, 0) == pytest.approx(200.0)


# --------------------------------------------------------------------------
# the asymmetry itself
# --------------------------------------------------------------------------

def test_agent_broadcast_is_short_on_the_ring_and_long_to_the_target(ring):
    """The whole point: one broadcast, two reaches."""
    nodes, inboxes, build = ring
    handler = build(agent_target_range_matrix(agent_agent=30.0, agent_target=200.0))

    _broadcast(handler, nodes, sender_id=2)  # agent at angle 0

    got = _received_by(inboxes)
    # 30 m covers 1-hop (12.36) and 2-hop (23.51) but not 3-hop (32.36).
    assert {3, 4, 10, 11} <= got          # +-1 and +-2 hops
    assert 5 not in got and 9 not in got  # 3 hops away
    # ... and the target still hears it, at 20 m, over the 200 m link.
    assert 0 in got


def test_short_ring_range_does_not_break_the_uplink(ring):
    """Regression for the failure mode that motivated the design.

    With a single global range, a ring short enough to be local (< 20 m) also
    silences every agent at the target, which then prunes them as dead.
    """
    nodes, inboxes, build = ring
    handler = build(agent_target_range_matrix(agent_agent=15.0, agent_target=200.0))

    for agent_id in range(2, 2 + N):
        _broadcast(handler, nodes, agent_id)

    # Every single agent reached the target, though no agent reaches past 1 hop.
    assert len(inboxes[0]) == N
    for agent_id in range(2, 2 + N):
        assert handler.range_for(agent_id, 0) == pytest.approx(200.0)
        assert handler.range_for(agent_id, 3 if agent_id != 3 else 4) == pytest.approx(15.0)


def test_ring_partitions_when_range_is_below_the_two_hop_chord(ring):
    """Static reachability at the moment of a death: 2-hop links do drop.

    At 20 m the two survivors flanking a dead drone are 23.51 m apart and can no
    longer hear each other -- while the target still hears both.

    This link is why the 2-hop chord matters, though not for the reason first
    guessed. The gap still CLOSES below it (the controller pulls the survivors
    together and the pulses circle the ring), but the victim's successor never
    completes the dual_pulse event: one of the two counter-propagating pulses is
    blocked by the corpse, so the successor can only see that direction directly
    from the originator -- across exactly this link. See config_param, section 2.
    """
    nodes, inboxes, build = ring
    handler = build(agent_target_range_matrix(agent_agent=20.0, agent_target=200.0))

    # Agent 3 (ring slot 1) dies; slots 0 and 2 are now each other's neighbours.
    _broadcast(handler, nodes, sender_id=2)
    assert 4 not in _received_by(inboxes)  # 2 hops away: partitioned
    assert 0 in _received_by(inboxes)      # observability intact

    # At 30 m the same pair stays connected.
    handler = build(agent_target_range_matrix(agent_agent=30.0, agent_target=200.0))
    _broadcast(handler, nodes, sender_id=2)
    assert 4 in _received_by(inboxes)


def test_target_downlink_reaches_the_whole_ring_and_the_adversary(ring):
    nodes, inboxes, build = ring
    handler = build(agent_target_range_matrix(agent_agent=30.0, agent_target=200.0))

    _broadcast(handler, nodes, sender_id=0)

    assert _received_by(inboxes) == set(range(1, 2 + N))


def test_adversary_links_keep_the_default_range(ring):
    """The target's spin controller reads AdversaryState from ~56 m away."""
    nodes, inboxes, build = ring
    handler = build(agent_target_range_matrix(agent_agent=30.0, agent_target=200.0))

    _broadcast(handler, nodes, sender_id=1)

    assert 0 in _received_by(inboxes)
    assert handler.range_for(1, 0) == pytest.approx(200.0)


# --------------------------------------------------------------------------
# default-off / degenerate configurations
# --------------------------------------------------------------------------

def test_without_a_matrix_every_link_keeps_the_default_range(ring):
    nodes, inboxes, build = ring
    handler = build(range_matrix=None)

    _broadcast(handler, nodes, sender_id=2)

    assert _received_by(inboxes) == set(range(0, 2 + N)) - {2}
    assert handler.range_for(2, 3) == pytest.approx(200.0)


def test_matrix_equal_to_the_default_reproduces_the_stock_reach(ring):
    nodes, inboxes, build = ring
    handler = build(agent_target_range_matrix(agent_agent=200.0, agent_target=200.0))

    _broadcast(handler, nodes, sender_id=2)

    assert _received_by(inboxes) == set(range(0, 2 + N)) - {2}


@pytest.mark.parametrize("bad", [0.0, -5.0, float("nan"), float("inf")])
def test_unusable_ranges_are_dropped_rather_than_silencing_a_link(ring, bad):
    nodes, _inboxes, build = ring
    handler = build({(ROLE_AGENT, ROLE_AGENT): bad})
    assert handler.range_for(2, 3) == pytest.approx(200.0)


def test_send_command_is_subject_to_the_matrix_too(ring):
    nodes, inboxes, build = ring
    handler = build(agent_target_range_matrix(agent_agent=15.0, agent_target=200.0))

    # Directed send to a 2-hop agent: out of ring range, so not delivered.
    handler.handle_command(
        CommunicationCommand(CommunicationCommandType.SEND, "m", destination=4),
        nodes[2],
    )
    assert inboxes[4] == []

    handler.handle_command(
        CommunicationCommand(CommunicationCommandType.SEND, "m", destination=0),
        nodes[2],
    )
    assert inboxes[0] == ["m"]


def test_explicit_per_command_medium_overrides_the_matrix(ring):
    """Keeps the library's per-command medium (Radio extension) composable."""
    nodes, inboxes, build = ring
    handler = build(agent_target_range_matrix(agent_agent=15.0, agent_target=200.0))

    handler.handle_command(
        CommunicationCommand(CommunicationCommandType.BROADCAST, "m"),
        nodes[2],
        CommunicationMedium(transmission_range=200.0),
    )
    assert 4 in _received_by(inboxes)


def test_delay_and_failure_rate_are_inherited_by_every_pair(ring):
    nodes, _inboxes, build = ring
    handler = RoleAwareCommunicationHandler(
        CommunicationMedium(transmission_range=200.0, delay=0.25, failure_rate=0.5),
        range_matrix=agent_target_range_matrix(agent_agent=30.0, agent_target=200.0),
    )
    handler.inject(_ImmediateEventLoop())
    for node in nodes.values():
        handler.register_node(node)

    ring_medium = handler._media[(ROLE_AGENT, ROLE_AGENT)]
    assert ring_medium.transmission_range == pytest.approx(30.0)
    assert ring_medium.delay == pytest.approx(0.25)
    assert ring_medium.failure_rate == pytest.approx(0.5)
