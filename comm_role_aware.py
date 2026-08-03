"""Per-link communication ranges, selected by the ROLE of sender and receiver.

Why this exists
---------------
GrADyS evaluates communication range at the SENDER only: ``can_transmit()``
takes a single ``CommunicationMedium`` -- the sender's -- and compares the
euclidean distance against its ``transmission_range``. There is no receiver
sensitivity in the model. The stock per-command medium (and the ``Radio``
extension built on top of it) is therefore sender-side too: it can make one
node's transmissions travel further, but it cannot make one node a better
LISTENER.

That is a problem here, because an agent's ``AgentState`` is a single BROADCAST
(``protocol_agent`` ~L1363) serving two audiences with opposite requirements:

  * the ring neighbours, which we may want to restrict to a short, physically
    realistic range so the hop neighbourhood is genuinely local; and
  * the target, which must hear EVERY agent. An agent the target stops hearing
    is pruned from ``agent_states``/``alive_lambdas`` after ``AGENT_STATE_TIMEOUT``
    (``protocol_target._prune_expired_states``) -- i.e. a live drone is declared
    dead. That corrupts ``alive_count``, the lambda map fed back to the agents,
    and every M1..M7 metric, and it does so silently: ``G_max``/``E_gap`` are
    normalized by the number of agents the target heard, so a half-observed ring
    still scores as if it were perfectly distributed.

So the range asymmetry has to be a property of the LINK, not of the sender.
This handler supplies it by overriding the single method that already sees both
endpoints, ``_transmit_message(message, source, destination, medium)``, and
swapping in a medium chosen from a ``(sender_role, receiver_role) -> range``
matrix. The BROADCAST fan-out loop in the base class is untouched, so one
broadcast still costs one message; only its reach varies per destination.

Design notes
------------
* Roles are resolved once, at ``register_node``, from the protocol class name.
  This is safe: ``Simulator.create_node`` encapsulates the protocol BEFORE
  registering the node with the handlers, so ``protocol_encapsulator.protocol``
  already exists. Resolution is by class NAME to keep this module free of any
  import dependency on the protocol modules.
* Only the pairs present in the matrix are overridden; every other pair keeps
  the handler's default medium. Notably ``adversary -> target`` stays long by
  default, which matters because the target's spin controller reads
  ``AdversaryState`` and the adversary spawns ~56 m from the origin.
* ``delay`` and ``failure_rate`` are inherited from the default medium: only
  ``transmission_range`` differs per pair.
* An explicit per-command medium (as used by the ``Radio`` extension) wins over
  the matrix -- the matrix only applies to sends that used the handler default.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, Mapping, Optional, Tuple

from gradysim.simulator.handler.communication import (
    CommunicationDestination,
    CommunicationHandler,
    CommunicationMedium,
    CommunicationSource,
)
from gradysim.simulator.node import Node

ROLE_TARGET = "target"
ROLE_AGENT = "agent"
ROLE_ADVERSARY = "adversary"
ROLE_UNKNOWN = "unknown"

# Protocol class name -> role. Kept as names (not classes) so this module never
# imports the protocols, which keeps it usable from tests and demos.
DEFAULT_PROTOCOL_ROLES: Dict[str, str] = {
    "TargetProtocol": ROLE_TARGET,
    "AgentProtocol": ROLE_AGENT,
    "AdversaryProtocol": ROLE_ADVERSARY,
}

RangeMatrix = Mapping[Tuple[str, str], float]


def agent_target_range_matrix(agent_agent: float, agent_target: float) -> Dict[Tuple[str, str], float]:
    """The two-knob matrix used by the simulation.

    ``agent_target`` is applied in BOTH directions: it models a target with a
    better radio (stronger transmitter and more sensitive receiver), which is a
    link-budget property, not a sender-side one.

    Pairs left out (adversary <-> anything) fall back to the default medium.
    """
    return {
        (ROLE_AGENT, ROLE_AGENT): float(agent_agent),
        (ROLE_AGENT, ROLE_TARGET): float(agent_target),
        (ROLE_TARGET, ROLE_AGENT): float(agent_target),
    }


class RoleAwareCommunicationHandler(CommunicationHandler):
    """``CommunicationHandler`` with a transmission range per (sender, receiver) role pair."""

    def __init__(
        self,
        communication_medium: Optional[CommunicationMedium] = None,
        range_matrix: Optional[RangeMatrix] = None,
        protocol_roles: Optional[Mapping[str, str]] = None,
    ):
        """
        Args:
            communication_medium: Default medium, used for any pair absent from
                the matrix. Also the source of ``delay``/``failure_rate``.
            range_matrix: ``{(sender_role, receiver_role): range_in_meters}``.
                Non-finite or non-positive ranges are dropped (the pair then
                falls back to the default medium) rather than silently creating
                a link that never delivers.
            protocol_roles: Optional ``{protocol_class_name: role}`` override.
        """
        super().__init__(communication_medium if communication_medium is not None else CommunicationMedium())

        self._protocol_roles: Dict[str, str] = dict(
            DEFAULT_PROTOCOL_ROLES if protocol_roles is None else protocol_roles
        )
        self._roles: Dict[int, str] = {}
        self._media: Dict[Tuple[str, str], CommunicationMedium] = {}

        for pair, transmission_range in (range_matrix or {}).items():
            value = float(transmission_range)
            if not math.isfinite(value) or value <= 0.0:
                continue
            self._media[(str(pair[0]), str(pair[1]))] = self._medium_with_range(self.default_medium, value)

    # -- construction helpers -------------------------------------------------

    @staticmethod
    def _medium_with_range(base: CommunicationMedium, transmission_range: float) -> CommunicationMedium:
        """Copy of ``base`` with a new range; ``delay``/``failure_rate`` preserved."""
        medium = copy.copy(base)
        medium.transmission_range = float(transmission_range)
        return medium

    # -- role bookkeeping -----------------------------------------------------

    def register_node(self, node: Node) -> None:
        super().register_node(node)
        self._roles[node.id] = self._resolve_role(node)

    def _resolve_role(self, node: Node) -> str:
        protocol = getattr(getattr(node, "protocol_encapsulator", None), "protocol", None)
        if protocol is None:
            return ROLE_UNKNOWN
        return self._protocol_roles.get(type(protocol).__name__, ROLE_UNKNOWN)

    def get_role(self, node_id: int) -> str:
        """Role recorded for a node id, or ``ROLE_UNKNOWN`` if never registered."""
        return self._roles.get(int(node_id), ROLE_UNKNOWN)

    def range_for(self, source_id: int, destination_id: int) -> float:
        """Effective transmission range for this ordered pair. For diagnostics."""
        pair = (self.get_role(source_id), self.get_role(destination_id))
        return float(self._media.get(pair, self.default_medium).transmission_range)

    # -- introspection, for sweep preflight assertions -------------------------

    def role_census(self) -> Dict[str, int]:
        """``{role: count}`` over registered nodes. A non-zero ``unknown`` means a
        protocol class the matrix cannot address, so its links silently keep the
        default range -- a sweep must abort on it, not average over it."""
        census: Dict[str, int] = {}
        for role in self._roles.values():
            census[role] = census.get(role, 0) + 1
        return census

    def differs_from_default(self) -> bool:
        """True when at least one pair actually departs from the default range.

        Guards the no-op run: gate enabled but every range left at the global
        value, which produces a fully connected run wearing a role-aware label.
        """
        default_range = float(self.default_medium.transmission_range)
        return any(m.transmission_range != default_range for m in self._media.values())

    def describe(self) -> str:
        """One machine-readable line for runners to parse and assert on.

        Stable, sorted, greppable::

            [comm] role_aware=1 roles={agent:24,adversary:1,target:1} \
matrix={agent>agent:6.3,agent>target:200,target>agent:200} default=200 differs=1
        """
        roles = ",".join(f"{role}:{count}" for role, count in sorted(self.role_census().items()))
        matrix = ",".join(
            f"{src}>{dst}:{medium.transmission_range:g}"
            for (src, dst), medium in sorted(self._media.items())
        )
        return (
            f"[comm] role_aware=1 roles={{{roles}}} matrix={{{matrix}}} "
            f"default={self.default_medium.transmission_range:g} "
            f"differs={int(self.differs_from_default())}"
        )

    # -- delivery -------------------------------------------------------------

    def _transmit_message(
        self,
        message: str,
        source: CommunicationSource,
        destination: CommunicationDestination,
        medium: CommunicationMedium,
    ) -> None:
        # Respect an explicit per-command medium (Radio extension); only sends
        # that fell back to the handler default are subject to the matrix.
        if medium is self.default_medium:
            pair = (self.get_role(source.node.id), self.get_role(destination.node.id))
            medium = self._media.get(pair, medium)
        super()._transmit_message(message, source, destination, medium)
