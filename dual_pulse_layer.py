"""Dual-pulse propagation layer (soliton-inspired, hop-count based).

Each ring node discovers N and its own angular shift target locally using
counter-propagating pulses with hop counts. Unlike the e_tau-driven layers,
DualPulseLayer does NOT feed the controller's u_prop channel — it modifies
the pred_gap / succ_gap inputs to compute_e_tau_used (Option A).

Two pulse semantics:
  - Receiver case (h_CCW + h_CW + 1 = N_old): a relay node that has seen the
    pulse arrive from both directions computes:
        delta_D = (h_CCW - N_old/2) * (gap_new - gap_old)
    and accumulates into shift_remaining.
  - Originator case: the canonical originator (the dead drone's predecessor)
    never receives both directions of its own pulse via the standard relay
    path — pulses circle the ring and arrive once at the originator with
    hop_count = N_new. At that moment a special formula is applied:
        delta_orig = gap_old - gap_new / 2
    derived from the constraint that the antipode of the dead drone in the
    new uniform layout stays put. delta_orig is applied at most once per
    self-originated event_id (the second returning direction is ignored).

Integration with the protocol agent:
  - update(...) processes incoming pulses from pred_state["pulses"] and
    succ_state["pulses"], filtering by direction (CCW pulses arrive from the
    pred side; CW pulses arrive from the succ side) and forwarding via
    pending_pulses_to_forward.
  - get_broadcast_state() drains pending_pulses_to_forward into the agent's
    AgentState.prop_state for the next broadcast.
  - get_shift_remaining() is consulted by the protocol agent to bias the gaps:
        succ_gap_used = succ_gap + shift_remaining
        pred_gap_used = pred_gap - shift_remaining
  - consume_motion(delta_theta) is called every tick with the realized
    angular displacement to drive shift_remaining toward zero. Clipping at
    zero prevents spurious drift when shift_remaining is already zero.

Coordination rule (v1 simplification): only the dead drone's predecessor
injects (the side that detected succ_changed). The successor side stays
silent. This keeps the canonical originator's geometry consistent with the
delta_D / delta_orig formulas. Trade-off: if the predecessor fails between
detecting and injecting, the event is missed. Redundancy via successor-side
fallback is left to v2.
"""

from __future__ import annotations

import math
from typing import Optional

from propagation_layer import PropagationLayer
from config_param import (
    DUAL_PULSE_ALPHA_CLOSE_RATIO,
    DUAL_PULSE_ALPHA_CURVE_POWER,
    DUAL_PULSE_BROADCAST_REPEATS,
    DUAL_PULSE_DELTA_SCALE,
    DUAL_PULSE_GAP_CLIP_FRAC,
    DUAL_PULSE_MIN_RING_SIZE,
    DUAL_PULSE_N_CLIP,
    DUAL_PULSE_RAMP_TICKS,
    DUAL_PULSE_SLEEP_THRESHOLD,
    DUAL_PULSE_TTL_HOPS,
    NUM_AGENTS,
)


def _safe(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


class DualPulseLayer(PropagationLayer):
    """Counter-propagating discrete pulses with hop counts on the ring."""

    # Each outgoing pulse is broadcast this many consecutive ticks. Robustness
    # against intra-tick agent firing order: GrADyS-SIM does not guarantee
    # that a sender fires before its receivers within a single simulator
    # time step, and the sender's prop_state is overwritten on every tick.
    # Broadcasting K=2 times ensures every receiver gets at least one chance
    # to read the pulse from the cache. The receiver's refractory_cache
    # eliminates duplicates, so this only costs bandwidth, not correctness.
    BROADCAST_REPEATS: int = DUAL_PULSE_BROADCAST_REPEATS   # env-overridable via config_param

    def __init__(self, params: Optional[dict] = None):
        # No tunable parameters in v1 — gating constants live in config_param.
        del params
        # Two-stage shift state for ramp behaviour:
        #   shift_target    : the cumulative ideal shift (raw sum of all
        #                     delta_D / delta_orig from completed events).
        #   shift_remaining : the applied shift, which lerps toward
        #                     shift_target over DUAL_PULSE_RAMP_TICKS update()
        #                     calls. This is what apply_shift_to_gaps reads,
        #                     and what consume_motion drains.
        # When DUAL_PULSE_RAMP_TICKS == 1 the two collapse (instantaneous step,
        # v1 behaviour). For RAMP_TICKS > 1 the controller sees the gap shift
        # smoothly over a few ticks instead of as a single jolt.
        self.shift_target: float = 0.0
        self.shift_remaining: float = 0.0
        # Monotonic local sequence number for event_id disambiguation.
        self.local_seq: int = 0
        # Per-(event_id, direction) dedup of relayed pulses.
        self.refractory_cache: set = set()
        # Pulses queued for outgoing broadcasts. Each entry is a dict with
        # keys "pulse" (the pulse payload) and "remaining" (broadcasts left).
        self.pending_pulses_to_forward: list = []
        # event_id -> {"CCW": h_CCW, "CW": h_CW}; once both directions seen,
        # delta_D is computed and the event_id is moved to processed_events.
        self.received_by_event: dict = {}
        # event_ids whose delta_D has been applied (relay case).
        self.processed_events: set = set()
        # event_ids self-originated by inject_pulse / inject_entrada.
        # Maps event_id -> {"event_type": "SAIDA"|"ENTRADA", "applied": bool}
        # (False until the first returning pulse triggers delta_orig).
        self._self_originated: dict = {}
        # Diagnostic events drained by the agent after each update().
        self._completed_events: list = []
        self._active: bool = True
        self._churn_suppress: bool = False   # gate (Fase 3): suprime overlay sob churn denso
        # The owning node's id, set by protocol_agent after construction.
        # Required to detect ENTRADA passthrough mode (when the recovered
        # drone forwards the ENTRADA pulse without applying delta to itself).
        self._owner_id: Optional[int] = None

    def set_owner_id(self, owner_id: int) -> None:
        """Set the owning node's id. Called by protocol_agent during init."""
        try:
            self._owner_id = int(owner_id)
        except Exception:
            self._owner_id = None

    # ------------------------------------------------------------------
    # PropagationLayer interface
    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        return bool(self._active)

    def set_churn_suppress(self, suppress: bool) -> None:
        """Gate (Fase 3): quando True, nao acumula novos delta_D e decai o shift -> baseline."""
        self._churn_suppress = bool(suppress)

    def update(self, e_tau, dt, pred_state, succ_state):
        """Process incoming pulses from neighbor broadcasts and ramp the
        applied shift toward its accumulated target.

        e_tau and dt are unused for pulse processing: the layer is
        event-triggered (pulses are introduced via inject_pulse) and
        hop-based (no time integration on the propagation field).
        dt would only matter if RAMP were time-based; we use a tick-based
        ramp instead so the cadence is independent of CONTROL_PERIOD.
        """
        del e_tau, dt
        pred_pulses = []
        succ_pulses = []
        if isinstance(pred_state, dict):
            raw = pred_state.get("pulses", [])
            if isinstance(raw, list):
                pred_pulses = raw
        if isinstance(succ_state, dict):
            raw = succ_state.get("pulses", [])
            if isinstance(raw, list):
                succ_pulses = raw

        # Direction filter: CCW pulses travel from predecessor toward me,
        # CW pulses travel from successor toward me. Pulses with the wrong
        # direction-tag in a given slot are ones that the neighbor was
        # forwarding to its OTHER side — not addressed to me.
        for pulse in pred_pulses:
            if isinstance(pulse, dict) and pulse.get("direction") == "CCW":
                self._process_received_pulse(pulse)
        for pulse in succ_pulses:
            if isinstance(pulse, dict) and pulse.get("direction") == "CW":
                self._process_received_pulse(pulse)

        # Ramp: lerp shift_remaining toward shift_target by 1/RAMP_TICKS of
        # the gap each tick. With RAMP_TICKS=1 this collapses to the v1
        # instantaneous step. With RAMP_TICKS>1 the controller sees the
        # virtual gaps shift smoothly over a few ticks instead of as a jolt.
        if self._churn_suppress:
            self.shift_target = 0.0   # gate: zera o alvo -> a rampa decai o shift p/ baseline
        gap = float(self.shift_target) - float(self.shift_remaining)
        if abs(gap) > 1e-15:
            step = gap / float(int(DUAL_PULSE_RAMP_TICKS))
            self.shift_remaining = _safe(self.shift_remaining + step)

    def get_signal(self) -> float:
        # Layer does not contribute to the controller's u_prop channel.
        return 0.0

    def get_neighbor_signal(self) -> float:
        return 0.0

    def get_broadcast_state(self) -> dict:
        """Build outgoing AgentState.prop_state and decay the broadcast TTL.

        Each pulse is broadcast BROADCAST_REPEATS consecutive ticks before
        being dropped, so a receiver that fires after the sender's NEXT
        tick still has at least one chance to see the pulse in the cache.
        """
        if not self.pending_pulses_to_forward:
            return {}
        pulses_out = []
        new_buffer = []
        for entry in self.pending_pulses_to_forward:
            try:
                pulse = entry["pulse"]
                remaining = int(entry["remaining"])
            except (KeyError, TypeError, ValueError):
                continue
            pulses_out.append(pulse)
            if remaining > 1:
                new_buffer.append({"pulse": pulse, "remaining": remaining - 1})
        self.pending_pulses_to_forward = new_buffer
        if not pulses_out:
            return {}
        # Message-overhead accounting (systems metric): count the pulse payloads
        # actually emitted over the run. Summed across nodes by the scaling-law
        # analysis to report messages-per-fault. Lazily initialized so __init__
        # stays untouched.
        self.broadcast_pulse_count = getattr(self, "broadcast_pulse_count", 0) + len(pulses_out)
        return {"pulses": pulses_out}

    def get_broadcast_pulse_count(self) -> int:
        """Total pulse payloads broadcast by this node over the run (overhead metric)."""
        return int(getattr(self, "broadcast_pulse_count", 0))

    def on_neighbor_change(self) -> None:
        # Pulses currently in transit through this node should not be erased
        # just because the topology shifted. Receivers' refractory_cache
        # naturally absorbs duplicates if a relay path now overlaps.
        pass

    def on_reset(self) -> None:
        """Agent recovered from failure: drop all in-flight state.

        local_seq is intentionally NOT reset so that any post-recovery event
        the agent originates carries a sequence number distinct from any
        pre-failure event still in flight elsewhere on the ring.
        """
        self.shift_target = 0.0
        self.shift_remaining = 0.0
        self.refractory_cache.clear()
        self.pending_pulses_to_forward.clear()
        self.received_by_event.clear()
        self.processed_events.clear()
        self._self_originated.clear()
        self._completed_events.clear()

    # ------------------------------------------------------------------
    # Public API used by protocol_agent
    # ------------------------------------------------------------------

    def get_shift_remaining(self) -> float:
        """Return the currently APPLIED shift (post-ramp).

        This is the value the protocol agent feeds into apply_shift_to_gaps.
        Lerps toward shift_target over DUAL_PULSE_RAMP_TICKS update() calls.
        """
        return float(self.shift_remaining)

    def get_shift_target(self) -> float:
        """Return the IDEAL accumulated shift (raw sum of all delta_D's).

        This is what the algorithm "wants" the shift to be. With RAMP_TICKS=1
        it equals shift_remaining; with RAMP_TICKS>1 shift_remaining lerps
        toward this value. Useful for diagnostic/testing the formula
        accumulation independently of the ramp dynamics.
        """
        return float(self.shift_target)

    def consume_motion(self, delta_theta: float) -> None:
        """Reduce both shift_target and shift_remaining by the realized
        angular displacement, keeping them coherent.

        Clips at zero so spurious motion (controller transient, swarm spin,
        any non-shift-driven angular change) does not push the shift across
        zero and induce a phantom redistribution drive.

        Both shift_target and shift_remaining are updated symmetrically:
        the ramp interpolation in update() is between two values that move
        together as motion is consumed, so the lerp continues to converge
        on the (decreasing) target as the agent rotates.

        Caveat: when TARGET_SWARM_SPIN_ENABLE=True the spin contribution to
        delta_theta WILL be erroneously consumed when shift is non-zero,
        producing a slight under-shift. v1 keeps spin off by default;
        revisit this when enabling spin alongside dual_pulse.
        """
        if abs(self.shift_target) < 1e-12 and abs(self.shift_remaining) < 1e-12:
            return
        try:
            d = float(delta_theta)
        except Exception:
            return
        if not math.isfinite(d):
            return
        # Use the sign of shift_target as the reference (it leads the ramp).
        sign_ref = self.shift_target if abs(self.shift_target) > 1e-12 else self.shift_remaining
        if sign_ref > 0.0:
            self.shift_target = max(0.0, self.shift_target - d)
            self.shift_remaining = max(0.0, self.shift_remaining - d)
        else:
            self.shift_target = min(0.0, self.shift_target - d)
            self.shift_remaining = min(0.0, self.shift_remaining - d)

    def inject_pulse(self, originator_id: int) -> None:
        """Inject CCW + CW pulses for a SAIDA event.

        Called by the protocol agent on the canonical side of the event
        (the dead drone's predecessor). The two pulses circle the ring in
        opposite directions and arrive at every other surviving node, where
        the receiver-side delta_D is computed. The pulses also return to
        this originator after a full loop, where delta_orig is applied via
        the self-originated path in _process_received_pulse.
        """
        try:
            oid = int(originator_id)
        except Exception:
            return
        self.local_seq += 1
        event_id = (oid, int(self.local_seq))
        self._self_originated[event_id] = {"event_type": "SAIDA", "applied": False}
        for direction in ("CCW", "CW"):
            pulse = {
                "event_id": [oid, int(self.local_seq)],
                "event_type": "SAIDA",
                "hop_count": 1,
                "direction": direction,
                "originator_id": oid,
                "recovered_id": None,  # not used for SAIDA
            }
            self.pending_pulses_to_forward.append(
                {"pulse": pulse, "remaining": int(self.BROADCAST_REPEATS)}
            )

    def inject_entrada(self, originator_id: int, recovered_id: int) -> None:
        """Inject CCW + CW pulses for an ENTRADA event.

        Called by the protocol agent on the canonical side of the recovery
        (the predecessor of the drone that just came back). The pulses
        carry recovered_id so the recovered drone can recognize itself and
        operate in passthrough mode (forward but skip self-shift).
        """
        try:
            oid = int(originator_id)
            rid = int(recovered_id)
        except Exception:
            return
        self.local_seq += 1
        event_id = (oid, int(self.local_seq))
        self._self_originated[event_id] = {"event_type": "ENTRADA", "applied": False}
        for direction in ("CCW", "CW"):
            pulse = {
                "event_id": [oid, int(self.local_seq)],
                "event_type": "ENTRADA",
                "hop_count": 1,
                "direction": direction,
                "originator_id": oid,
                "recovered_id": rid,
            }
            self.pending_pulses_to_forward.append(
                {"pulse": pulse, "remaining": int(self.BROADCAST_REPEATS)}
            )

    def get_completed_events_and_clear(self) -> list:
        """Return diagnostic records for events whose delta_D was applied.

        Each record is a dict with keys: event_id (tuple), h_CCW, h_CW,
        N_new, delta_D, is_originator. Drained by the protocol agent for
        events.csv logging.
        """
        out = list(self._completed_events)
        self._completed_events.clear()
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_event_id(raw) -> Optional[tuple]:
        # JSON serialization of tuples becomes a list. Normalize back.
        if isinstance(raw, tuple) and len(raw) == 2:
            try:
                return (int(raw[0]), int(raw[1]))
            except Exception:
                return None
        if isinstance(raw, list) and len(raw) == 2:
            try:
                return (int(raw[0]), int(raw[1]))
            except Exception:
                return None
        return None

    @staticmethod
    def _hop_alpha(distance_from_d: float, n_new: int) -> float:
        """Hop-distance attenuation factor for delta_D / delta_orig.

        Returns ALPHA_CLOSE_RATIO at distance_from_d == 0 (immediate
        neighbor of D's location) and 1.0 at the antipode (distance = N_new/2).
        Interpolation shape is controlled by ALPHA_CURVE_POWER:
            alpha(d_norm) = ALPHA_CLOSE_RATIO + (1 - ALPHA_CLOSE_RATIO) * d_norm^p

        With ALPHA_CLOSE_RATIO == 1.0 (default) this is identically 1.0 —
        no attenuation, preserves backward compatibility with v1.5.
        """
        try:
            r = float(DUAL_PULSE_ALPHA_CLOSE_RATIO)
        except Exception:
            r = 1.0
        if r >= 1.0 - 1e-12:
            return 1.0
        if n_new <= 1:
            return 1.0
        half = float(n_new) / 2.0
        if half <= 0.0:
            return 1.0
        d_norm = max(0.0, min(1.0, float(distance_from_d) / half))
        try:
            p = float(DUAL_PULSE_ALPHA_CURVE_POWER)
        except Exception:
            p = 1.0
        if p <= 0.0:
            p = 1.0
        if abs(p - 1.0) < 1e-12:
            shaped = d_norm
        else:
            shaped = d_norm ** p
        return r + (1.0 - r) * shaped

    def _process_received_pulse(self, pulse: dict) -> None:
        """Dedupe / accumulate / forward a single incoming pulse.

        Handles both SAIDA and ENTRADA event types via the pulse's
        event_type field. The two events use different geometry:
          SAIDA  : drone leaves, ring shrinks. anchor = N_old/2 (CCW from A).
          ENTRADA: drone returns, ring grows. anchor = 1 + N_new/2 (CCW from A).

        For ENTRADA, the recovered drone (matching pulse.recovered_id)
        operates in passthrough mode: forwards the pulse but does NOT apply
        delta to itself, since its physical position is the equilibrium one.
        """
        event_id = self._coerce_event_id(pulse.get("event_id"))
        if event_id is None:
            return
        direction = pulse.get("direction")
        if direction not in ("CCW", "CW"):
            return
        try:
            h = int(pulse.get("hop_count", 0))
        except Exception:
            return
        event_type = pulse.get("event_type", "SAIDA")
        if event_type not in ("SAIDA", "ENTRADA"):
            event_type = "SAIDA"
        try:
            recovered_id = int(pulse.get("recovered_id")) if pulse.get("recovered_id") is not None else None
        except Exception:
            recovered_id = None

        # Self-originated returning pulse: apply delta_orig once, do NOT forward.
        if event_id in self._self_originated:
            entry = self._self_originated[event_id]
            if not entry["applied"]:
                n_new = h  # full-loop hop count = ring size
                if DUAL_PULSE_N_CLIP and (n_new < int(DUAL_PULSE_MIN_RING_SIZE) or n_new > int(NUM_AGENTS)):
                    entry["applied"] = True   # estimativa de anel corrompida -> nao aplica (mitigacao)
                    return
                if n_new >= int(DUAL_PULSE_MIN_RING_SIZE):
                    # Originator is by construction the immediate neighbor of
                    # D's location (predecessor for SAIDA, predecessor for
                    # ENTRADA too — both at distance_from_d == 1). Hop alpha
                    # attenuates the originator's self-shift accordingly.
                    alpha_hop = self._hop_alpha(distance_from_d=1.0, n_new=n_new)
                    if event_type == "SAIDA":
                        # SAIDA: gap_old (with D) > gap_new (without D)
                        n_old = n_new + 1
                        gap_old = 2.0 * math.pi / float(n_old)
                        gap_new = 2.0 * math.pi / float(n_new)
                        delta_orig = (
                            (gap_old - gap_new / 2.0)
                            * float(DUAL_PULSE_DELTA_SCALE)
                            * alpha_hop
                        )
                    else:  # ENTRADA
                        # ENTRADA: ring with D back. delta_orig has opposite sign
                        # by time-reverse symmetry: gap_old/2 - gap_new.
                        n_old = n_new - 1
                        if n_old < int(DUAL_PULSE_MIN_RING_SIZE):
                            entry["applied"] = True
                            return
                        gap_old = 2.0 * math.pi / float(n_old)
                        gap_new = 2.0 * math.pi / float(n_new)
                        delta_orig = (
                            (gap_old / 2.0 - gap_new)
                            * float(DUAL_PULSE_DELTA_SCALE)
                            * alpha_hop
                        )
                    if math.isfinite(delta_orig) and not self._churn_suppress:
                        self.shift_target = _safe(
                            self.shift_target + float(delta_orig)
                        )
                        self._completed_events.append({
                            "event_id": event_id,
                            "event_type": event_type,
                            "h_CCW": n_new if direction == "CCW" else 0,
                            "h_CW": n_new if direction == "CW" else 0,
                            "N_new": int(n_new),
                            "delta_D": float(delta_orig),
                            "is_originator": True,
                        })
                entry["applied"] = True
            return

        # ENTRADA passthrough: the recovered drone forwards but skips self-delta.
        is_recovered_passthrough = (
            event_type == "ENTRADA"
            and recovered_id is not None
            and self._owner_id is not None
            and recovered_id == self._owner_id
        )

        # Standard relay path: dedupe per (event_id, direction).
        cache_key = (event_id, direction)
        if cache_key in self.refractory_cache:
            return
        if h > int(DUAL_PULSE_TTL_HOPS):
            return
        self.refractory_cache.add(cache_key)

        if not is_recovered_passthrough:
            rec = self.received_by_event.setdefault(event_id, {})
            rec[direction] = h

        # Forward (incremented hop_count) in the same direction.
        # Always forward, including in passthrough mode — the pulse must
        # reach the agents downstream of the recovered drone.
        if event_id not in self.processed_events:
            forwarded = {
                "event_id": [int(event_id[0]), int(event_id[1])],
                "event_type": event_type,
                "hop_count": h + 1,
                "direction": direction,
                "originator_id": pulse.get("originator_id", event_id[0]),
                "recovered_id": recovered_id,
            }
            self.pending_pulses_to_forward.append(
                {"pulse": forwarded, "remaining": int(self.BROADCAST_REPEATS)}
            )

        # Passthrough nodes don't compute delta for themselves.
        if is_recovered_passthrough:
            return

        # If both directions have arrived, compute delta_D once.
        rec = self.received_by_event.get(event_id, {})
        if "CCW" in rec and "CW" in rec and event_id not in self.processed_events:
            self.processed_events.add(event_id)
            h_ccw = int(rec["CCW"])
            h_cw = int(rec["CW"])
            n_total = h_ccw + h_cw + 1  # ring size as seen by this receiver
            if DUAL_PULSE_N_CLIP and (n_total < int(DUAL_PULSE_MIN_RING_SIZE) or n_total > int(NUM_AGENTS)):
                return   # estimativa de anel impossivel (corrompida) -> pula delta_D (ja marcado processed)
            if event_type == "SAIDA":
                n_old = n_total
                n_new = n_old - 1
                if n_new < int(DUAL_PULSE_MIN_RING_SIZE):
                    return
                gap_old = 2.0 * math.pi / float(n_old)
                gap_new = 2.0 * math.pi / float(n_new)
                h_anchor = float(n_old) / 2.0
                # Distance from D's old position in the new ring (D was at
                # h_CCW_old=1 from A; it sat between A=h=0 and h=1). For the
                # receiver at h_CCW we measure shortest-path on the new ring.
                distance_from_d = float(min(h_ccw, n_new - h_ccw + 1))
            else:  # ENTRADA
                n_new = n_total
                n_old = n_new - 1
                if n_old < int(DUAL_PULSE_MIN_RING_SIZE):
                    return
                gap_old = 2.0 * math.pi / float(n_old)
                gap_new = 2.0 * math.pi / float(n_new)
                h_anchor = 1.0 + float(n_new) / 2.0
                # D is at h_CCW=1 in the new ring. Distance from D for a
                # receiver at h_CCW is the shorter of the two ring arcs.
                d_offset = abs(int(h_ccw) - 1)
                distance_from_d = float(min(d_offset, n_new - d_offset))
            alpha_hop = self._hop_alpha(distance_from_d=distance_from_d, n_new=n_new)
            delta_d = (
                (float(h_ccw) - h_anchor) * (gap_new - gap_old)
                * float(DUAL_PULSE_DELTA_SCALE)
                * alpha_hop
            )
            if math.isfinite(delta_d) and not self._churn_suppress:
                self.shift_target = _safe(
                    self.shift_target + float(delta_d)
                )
                self._completed_events.append({
                    "event_id": event_id,
                    "event_type": event_type,
                    "h_CCW": h_ccw,
                    "h_CW": h_cw,
                    "N_new": int(n_new),
                    "delta_D": float(delta_d),
                    "is_originator": False,
                })


# ---------------------------------------------------------------------------
# Helper exposed for protocol_agent's gap-modification step
# ---------------------------------------------------------------------------

def apply_shift_to_gaps(
    pred_gap: Optional[float],
    succ_gap: Optional[float],
    shift_remaining: float,
) -> tuple:
    """Return (pred_gap_used, succ_gap_used) with shift applied and clipped.

    Convention: shift_remaining > 0 ==> drone should move CCW ==>
    succ_gap_used > succ_gap (more virtual space ahead) and
    pred_gap_used < pred_gap. The clip keeps both virtual gaps strictly
    positive by limiting |shift| to DUAL_PULSE_GAP_CLIP_FRAC * min(real_gaps).

    For non-finite inputs, returns the inputs unchanged. With shift==0 the
    virtual gaps coincide with the real gaps, so the controller behavior is
    indistinguishable from baseline.
    """
    if pred_gap is None or succ_gap is None:
        return pred_gap, succ_gap
    try:
        pg = float(pred_gap)
        sg = float(succ_gap)
        s = float(shift_remaining)
    except Exception:
        return pred_gap, succ_gap
    if not (math.isfinite(pg) and math.isfinite(sg) and math.isfinite(s)):
        return pred_gap, succ_gap
    # Sleep gate: when the shift magnitude is below threshold, the layer is
    # effectively idle (no in-flight redistribution worth biasing for).
    # Skipping the bias prevents subtle controller-vs-tracking interference
    # that empirically degrades M1 by ~+7-12% when the target is moving.
    if abs(s) < float(DUAL_PULSE_SLEEP_THRESHOLD):
        return pred_gap, succ_gap
    if s == 0.0 or pg <= 0.0 or sg <= 0.0:
        return pred_gap, succ_gap
    max_shift = min(pg, sg) * float(DUAL_PULSE_GAP_CLIP_FRAC)
    if not math.isfinite(max_shift) or max_shift <= 0.0:
        return pred_gap, succ_gap
    s_clipped = max(-max_shift, min(max_shift, s))
    return (pg - s_clipped, sg + s_clipped)
