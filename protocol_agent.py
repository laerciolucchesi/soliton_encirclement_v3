"""
Protocol for the agent node.
"""

import logging
from typing import Dict, Optional, Tuple
import math
import os
import random

import pandas as pd

from gradysim.protocol.interface import IProtocol
from gradysim.protocol.messages.telemetry import Telemetry
from gradysim.protocol.messages.communication import CommunicationCommand, CommunicationCommandType

try:
    from gradysim.simulator.extension.visualization_controller import VisualizationController
except Exception:  # pragma: no cover
    VisualizationController = None

import json

from propagation_layer import create_propagation_layer, DampedAdvectionLayer

from config_param import (
    CONTROL_LOOP_TIMER_STR,
    CONTROL_PERIOD,
    SIM_DEBUG,
    AGENT_STATE_TIMEOUT,
    TARGET_STATE_TIMEOUT,
    HYSTERESIS_RAD,
    PRUNE_EXPIRED_STATES,
    ENCIRCLEMENT_RADIUS,
    R_MIN,
    VM_MAX_SPEED_XY,
    VM_MAX_SPEED_Z,
    K_TAU,
    BETA_U,
    BETA_U_LOCAL,
    BETA_U_PROP,
    K_E_TAU,
    U_CONFLICT_BLEND_WIDTH,
    TANGENTIAL_COMPOSITION_MODE,
    K_R,
    K_DR,
    K_OMEGA_DAMP,
    FAILURE_CHECK_PERIOD,
    FAILURE_CHECK_TIMER_STR,
    FAILURE_ENABLE,
    FAILURE_MEAN_FAILURES_PER_MIN,
    FAILURE_OFF_TIME,
    FAILURE_RECOVER_TIMER_STR,
    EXPERIMENT_REPRODUCIBLE,
    WAVE_DAMPING_GAMMA,
    K_TRIGGER,
    MIN_EVENT_DELTA_FRAC,
    FAST_CHANNEL_WARMUP_SEC,
    )


from protocol_messages import AgentState, TargetState
from controllers import RadialDistanceController, TangentialSpacingController


class AgentProtocol(IProtocol):
    """Implementation of agent protocol."""

    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger()

    def initialize(self):
        self.node_id = self.provider.get_id()  # Get the node ID from the provider

        # Failure RNG: reproducible per-agent if enabled, else non-deterministic
        if EXPERIMENT_REPRODUCIBLE:
            self._failure_rng = random.Random(0xF00DCAFE + int(self.node_id))
        else:
            self._failure_rng = random.Random()

        self.control_period = CONTROL_PERIOD  # Control loop period in seconds
        # Schedule the control loop timer for the first time
        self.schedule_control_loop_timer()

        # Failure injection state
        self._failed: bool = False

        # Access the VelocityMobilityHandler if available
        handlers = getattr(self.provider, "handlers", {}) or {}
        self.velocity_handler = handlers.get("VelocityMobilityHandler")

        # Latest received states (filled by handle_packet)
        # Stored as (state, rxtime) to support future timeout detection.
        self.target_state: Optional[Tuple[TargetState, float]] = None
        self.agent_states: Dict[int, Tuple[AgentState, float]] = {}

        # Sequence tracking to discard out-of-order messages.
        self.last_seq_agent: Dict[int, int] = {}
        self.last_seq_target: int = -1

        # Neighbor selection results (updated every control tick).
        self.neighbor_pred_id: Optional[int] = None
        self.neighbor_succ_id: Optional[int] = None
        self.neighbor_pred_state: Optional[AgentState] = None
        self.neighbor_succ_state: Optional[AgentState] = None
        self._neighbor_pred_gap: Optional[float] = None
        self._neighbor_succ_gap: Optional[float] = None

        # Local lambda (lp) weights for predecessor/successor spacing.
        # Defaults to uniform spacing.
        self.lp_pred: float = 1.0
        self.lp_succ: float = 1.0

        # Broadcast sequence number
        self.agent_state_seq = 1

        self.radial_controller = RadialDistanceController(
            kp=K_R,
            kd=K_DR,
            radius_setpoint=ENCIRCLEMENT_RADIUS,
        )
        self.tangential_controller = TangentialSpacingController(
            beta_u=BETA_U,
            beta_u_local=BETA_U_LOCAL,
            beta_u_prop=BETA_U_PROP,
            k_e_tau=K_E_TAU,
            conflict_blend_width=U_CONFLICT_BLEND_WIDTH,
            composition_mode=TANGENTIAL_COMPOSITION_MODE,
            initial_u=0.0,
        )

        # Control internal state (muscle)
        self.u: float = 0.0        # total u after channel composition
        self.u_local: float = 0.0  # local error channel state
        self.u_prop: float = 0.0   # propagated error channel state
        self.u_ss: float = 0.0

        # Last spacing error values for telemetry and logging.
        self.last_e_tau: float = 0.0
        self.last_e_tau_eff: float = 0.0

        # Per-control-tick increments (dt_u * du) for analysis/telemetry.
        self.delta_u: float = 0.0

        # u-loop term telemetry (derivatives)
        self.du_damp: float = 0.0
        self.du_from_e_tau: float = 0.0


        # Propagation layer — method selected at runtime via main.py menu
        _prop_method = os.environ.get("PROPAGATION_METHOD", "baseline")
        _prop_k_prop = float(os.environ.get("PROPAGATION_K_PROP", "0.0"))
        try:
            _prop_params = json.loads(os.environ.get("PROPAGATION_PARAMS", "{}"))
        except Exception:
            _prop_params = {}
        self.prop_layer = create_propagation_layer(_prop_method, _prop_params)
        self._prop_k_prop: float = _prop_k_prop
        self.last_prop_signal: float = 0.0

        # Fast soliton-like channel — runs in PARALLEL with the main prop_layer.
        # Phase A: observational only (does NOT enter u_total). Pulses are
        # injected on detected events (neighbor identity changes); the field
        # values are logged to telemetry and visualized post-hoc.
        self.fast_layer = DampedAdvectionLayer(params={"gamma": WAVE_DAMPING_GAMMA})
        # Track previous neighbor identities so we can detect transitions tick-to-tick.
        self._last_pred_id_for_event: Optional[int] = None
        self._last_succ_id_for_event: Optional[int] = None
        # Track previous e_tau to compute delta on each tick.
        self._last_e_tau_for_event: float = 0.0
        # Sparse event log (failure_start, failure_end, pulse_injected). Flushed
        # to events.csv in finish().
        self._event_rows: list = []
        self._events_csv_path: Optional[str] = os.environ.get("EVENTS_LOG_CSV_PATH")
        if not self._events_csv_path:
            self._events_csv_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "events.csv"
            )

        # Last commanded velocity (world coordinates).
        self.desired_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)

        # Telemetry logging (in-memory)
        self._csv_path: Optional[str] = os.environ.get("AGENT_LOG_CSV_PATH")
        if not self._csv_path:
            # fallback: write next to this protocol file
            self._csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_telemetry.csv")

        self._telemetry_rows = []

        # Visualization controller
        self._vis = None
        if VisualizationController is not None:
            try:
                self._vis = VisualizationController(self)
                # Default agent color: blue
                self._vis.paint_node(self.node_id, (0.0, 0.0, 255.0))
            except Exception:
                self._vis = None

        # Schedule the first failure-check timer after visualization initialization.
        if FAILURE_ENABLE:
            self.schedule_failure_check_timer()

    def schedule_control_loop_timer(self):
        self.provider.schedule_timer(CONTROL_LOOP_TIMER_STR, self.provider.current_time() + self.control_period)

    def schedule_failure_check_timer(self):
        self.provider.schedule_timer(
            FAILURE_CHECK_TIMER_STR,
            self.provider.current_time() + float(FAILURE_CHECK_PERIOD),
        )

    def schedule_failure_recover_timer(self, off_time: float):
        self.provider.schedule_timer(
            FAILURE_RECOVER_TIMER_STR,
            self.provider.current_time() + float(off_time),
        )

    @staticmethod
    def _dot2(a, b) -> float:
        return a[0] * b[0] + a[1] * b[1]

    @staticmethod
    def _norm2(a) -> float:
        return math.sqrt(a[0] * a[0] + a[1] * a[1])

    @staticmethod
    def _unit2(a, eps: float = 1e-6):
        """Return a unit 2D vector with a safe fallback for near-zero norms."""
        n = AgentProtocol._norm2(a)
        if not math.isfinite(n) or n < eps:
            return (1.0, 0.0), 0.0
        return (a[0] / n, a[1] / n), n

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            val = float(value)
        except Exception:
            return float(default)
        if not math.isfinite(val):
            return float(default)
        return float(val)

    @staticmethod
    def wrap_to_2pi(angle: float) -> float:
        """Wrap angle to [0, 2*pi)."""
        two_pi = 2.0 * math.pi
        wrapped = angle % two_pi
        return wrapped

    @staticmethod
    def _theta_2d(target_pos, agent_pos) -> float:
        """Compute target-centric angle in the horizontal plane."""
        dx = agent_pos[0] - target_pos[0]
        dy = agent_pos[1] - target_pos[1]
        return math.atan2(dy, dx)

    def _target_is_alive(self, now: float) -> bool:
        if self.target_state is None:
            return False
        _, rxtime = self.target_state
        return (now - rxtime) <= TARGET_STATE_TIMEOUT

    def _agent_is_alive(self, agent_id: int, now: float) -> bool:
        entry = self.agent_states.get(agent_id)
        if entry is None:
            return False
        _, rxtime = entry
        return (now - rxtime) <= AGENT_STATE_TIMEOUT

    def _prune_expired_states(self, now: float) -> None:
        """Drop expired cached states to prevent unbounded growth."""
        if not PRUNE_EXPIRED_STATES:
            return

        expired_agent_ids = [
            agent_id
            for agent_id, (_, rxtime) in self.agent_states.items()
            if (now - rxtime) > AGENT_STATE_TIMEOUT
        ]
        for agent_id in expired_agent_ids:
            self.agent_states.pop(agent_id, None)
            self.last_seq_agent.pop(agent_id, None)

    def compute_tangential_unit_vector(self, target_pos, own_pos) -> Tuple[float, float]:
        """Compute unit tangential direction t_hat in the XY plane."""
        r_vec = (own_pos[0] - target_pos[0], own_pos[1] - target_pos[1])
        r_hat, _ = self._unit2(r_vec, eps=1e-6)
        return (-r_hat[1], r_hat[0])

    @staticmethod
    def compute_omega_about_target_xy(
        *,
        target_pos: Tuple[float, float, float],
        target_vel: Tuple[float, float, float],
        pos: Tuple[float, float, float],
        vel: Tuple[float, float, float],
        eps: float = 1e-6,
    ) -> Optional[float]:
        """Estimate angular rate omega (rad/s) around the target in the XY plane."""
        rx = float(pos[0] - target_pos[0])
        ry = float(pos[1] - target_pos[1])
        r = math.hypot(rx, ry)
        if not (math.isfinite(r) and r > eps):
            return None

        r_hat_x = rx / r
        r_hat_y = ry / r
        t_hat_x = -r_hat_y
        t_hat_y = r_hat_x

        vrel_x = float(vel[0] - target_vel[0])
        vrel_y = float(vel[1] - target_vel[1])
        if not (math.isfinite(vrel_x) and math.isfinite(vrel_y)):
            return None

        v_tan = vrel_x * t_hat_x + vrel_y * t_hat_y
        omega = v_tan / r
        if not math.isfinite(omega):
            return None
        return float(omega)

    @staticmethod
    def compute_spacing_error(gap_pred: Optional[float], gap_succ: Optional[float], lp_pred: float, lp_succ: float) -> float:
        """Local spacing imbalance error with weighted lambdas for arbitrary spacing."""
        if gap_pred is None or gap_succ is None:
            return 0.0

        lp_pred_f = float(lp_pred)
        lp_succ_f = float(lp_succ)
        if not (math.isfinite(lp_pred_f) and math.isfinite(lp_succ_f)):
            return 0.0

        num = float(lp_pred_f * float(gap_succ) - lp_succ_f * float(gap_pred))
        denom = float(lp_pred_f * float(gap_succ) + lp_succ_f * float(gap_pred))

        if not math.isfinite(denom) or denom <= 1e-9:
            return 0.0

        val = float(num / denom)
        if not math.isfinite(val):
            return 0.0
        return val

    def compute_e_tau_used(
        self,
        *,
        pred_gap: Optional[float],
        succ_gap: Optional[float],
        t_hat: Optional[Tuple[float, float]] = None,
        r_eff: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """Compute e_tau and optionally apply local omega damping."""
        e_tau = self.compute_spacing_error(pred_gap, succ_gap, self.lp_pred, self.lp_succ)
        e_tau_eff = float(e_tau)

        if (
            math.isfinite(K_OMEGA_DAMP)
            and K_OMEGA_DAMP > 0.0
            and self.target_state is not None
            and t_hat is not None
            and r_eff is not None
        ):
            target_state, _ = self.target_state
            v_cmd_prev = getattr(self, "desired_velocity", (0.0, 0.0, 0.0))
            v_tan_cmd = float(v_cmd_prev[0] * t_hat[0] + v_cmd_prev[1] * t_hat[1])
            omega_self = None
            if math.isfinite(v_tan_cmd) and math.isfinite(r_eff) and r_eff > 0.0:
                omega_self = float(v_tan_cmd / r_eff)

            omega_pred = None
            omega_succ = None
            if self.neighbor_pred_state is not None:
                omega_pred = self.compute_omega_about_target_xy(
                    target_pos=target_state.position,
                    target_vel=target_state.velocity,
                    pos=self.neighbor_pred_state.position,
                    vel=self.neighbor_pred_state.velocity,
                )
            if self.neighbor_succ_state is not None:
                omega_succ = self.compute_omega_about_target_xy(
                    target_pos=target_state.position,
                    target_vel=target_state.velocity,
                    pos=self.neighbor_succ_state.position,
                    vel=self.neighbor_succ_state.velocity,
                )

            omega_ref_local = None
            if omega_pred is not None and omega_succ is not None:
                omega_ref_local = 0.5 * (omega_pred + omega_succ)
            elif omega_pred is not None:
                omega_ref_local = omega_pred
            elif omega_succ is not None:
                omega_ref_local = omega_succ

            if omega_self is not None and omega_ref_local is not None:
                domega = omega_self - omega_ref_local
                if math.isfinite(domega):
                    e_tau_eff = float(e_tau - (K_OMEGA_DAMP * domega))
                if not math.isfinite(e_tau_eff):
                    e_tau_eff = float(e_tau)

        e_tau_used = float(e_tau_eff)
        if not math.isfinite(e_tau_used):
            e_tau_used = 0.0

        return float(e_tau), float(e_tau_eff), float(e_tau_used)

    def _compute_desired_gap_self(self) -> float:
        """Return the agent's own desired arc (radians) at equilibrium.

        Uses the latest TargetState.alive_lambdas: each agent's lambda is its
        weight in the total 2*pi arc budget. Falls back to 2*pi/N when target
        broadcast is unavailable (with N estimated from cached agent_states).
        """
        two_pi = 2.0 * math.pi
        if self.target_state is not None:
            ts, _ = self.target_state
            lambdas = getattr(ts, "alive_lambdas", None) or {}
            if isinstance(lambdas, dict) and lambdas:
                # Robust to int/str keys (JSON parses ints into strings).
                lam_self = lambdas.get(self.node_id)
                if lam_self is None:
                    lam_self = lambdas.get(str(int(self.node_id)))
                total = 0.0
                for v in lambdas.values():
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    if math.isfinite(fv):
                        total += fv
                if (
                    lam_self is not None
                    and total > 1e-9
                    and math.isfinite(float(lam_self))
                ):
                    return two_pi * float(lam_self) / total
        # Fallback: 2*pi/N estimated from local cache (self + alive neighbors).
        n_est = 1 + len(self.agent_states)
        if n_est < 2:
            n_est = 2
        return two_pi / float(n_est)

    def _update_neighbor_lps_from_target(self) -> None:
        """Update lp_pred/lp_succ using the latest TargetState.alive_lambdas."""
        if self.target_state is None:
            return
        ts, _ = self.target_state
        lambdas = getattr(ts, "alive_lambdas", None) or {}
        if not isinstance(lambdas, dict):
            return

        def _lookup(agent_id: Optional[int]) -> Optional[float]:
            if agent_id is None:
                return None
            if agent_id in lambdas:
                return lambdas.get(agent_id)
            key_str = str(int(agent_id))
            if key_str in lambdas:
                return lambdas.get(key_str)
            return None

        lp = _lookup(self.neighbor_pred_id)
        if lp is not None and math.isfinite(float(lp)):
            self.lp_pred = float(lp)

        ls = _lookup(int(self.node_id))
        if ls is not None and math.isfinite(float(ls)):
            self.lp_succ = float(ls)

    def _refresh_neighbors(self, now: float, position) -> Tuple[Optional[float], Optional[float]]:
        """Refresh predecessor/successor selection and cache the latest neighbor states."""
        self._prune_expired_states(now)
        pred_id, succ_id, pred_gap, succ_gap, alive_count, theta_i = self.get_two_neighbors(now, position)

        old_pred_id = self.neighbor_pred_id
        old_succ_id = self.neighbor_succ_id

        self.neighbor_pred_id = pred_id
        self.neighbor_succ_id = succ_id

        if pred_id != old_pred_id or succ_id != old_succ_id:
            self.prop_layer.on_neighbor_change()
        self._neighbor_pred_gap = pred_gap
        self._neighbor_succ_gap = succ_gap

        self._update_neighbor_lps_from_target()

        self.neighbor_pred_state = None
        self.neighbor_succ_state = None
        if pred_id is not None and pred_id in self.agent_states:
            self.neighbor_pred_state = self.agent_states[pred_id][0]
        if succ_id is not None and succ_id in self.agent_states:
            self.neighbor_succ_state = self.agent_states[succ_id][0]

        if SIM_DEBUG:
            theta_str = f"{theta_i:.3f}" if theta_i is not None else "None"
            pred_gap_str = f"{pred_gap:.3f}" if pred_gap is not None else "None"
            succ_gap_str = f"{succ_gap:.3f}" if succ_gap is not None else "None"
            print(
                f"Agent {self.node_id} neighbors: "
                f"theta={theta_str}, pred={pred_id}, succ={succ_id}, "
                f"pred_gap={pred_gap_str}, succ_gap={succ_gap_str}, alive={alive_count}"
            )

        return pred_gap, succ_gap

    def _get_neighbor_values(self) -> Tuple[float, float]:
        """Return neighbor u values with finite fallbacks."""
        u_pred = 0.0
        u_succ = 0.0

        if self.neighbor_pred_state is not None:
            u_pred = self._safe_float(self.neighbor_pred_state.u)
        if self.neighbor_succ_state is not None:
            u_succ = self._safe_float(self.neighbor_succ_state.u)

        return u_pred, u_succ

    def compute_tangential_velocity(
        self,
        u: float,
        t_hat: Tuple[float, float],
        *,
        r_eff: float = 1.0,
    ) -> Tuple[float, float, float]:
        """Convert internal u into tangential velocity (XYZ)."""
        if not math.isfinite(r_eff) or r_eff <= 0.0:
            r_eff = 1.0
        v_tau_corr = float(K_TAU) * u * r_eff
        return (v_tau_corr * t_hat[0], v_tau_corr * t_hat[1], 0.0)

    @staticmethod
    def compose_final_velocity(
        v_rad: Tuple[float, float, float],
        v_tau: Tuple[float, float, float],
        v_target: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """Compose final commanded velocity: v_cmd = v_rad + v_tau + v_target."""
        return (
            v_rad[0] + v_tau[0] + v_target[0],
            v_rad[1] + v_tau[1] + v_target[1],
            v_rad[2] + v_tau[2] + v_target[2],
        )

    @staticmethod
    def _clamp_velocity_to_limits(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Clamp a commanded velocity to configured mobility limits."""
        vx, vy, vz = v
        if not (math.isfinite(vx) and math.isfinite(vy) and math.isfinite(vz)):
            return (0.0, 0.0, 0.0)

        v_xy = math.hypot(vx, vy)
        if v_xy > VM_MAX_SPEED_XY and v_xy > 0.0:
            scale = VM_MAX_SPEED_XY / v_xy
            vx *= scale
            vy *= scale

        if abs(vz) > VM_MAX_SPEED_Z:
            vz = math.copysign(VM_MAX_SPEED_Z, vz)

        return (vx, vy, vz)

    def get_two_neighbors(
        self, now: float, own_position
    ) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[float], int, Optional[float]]:
        """Select predecessor/successor around the target using only locally received states."""
        if not self._target_is_alive(now):
            return None, None, None, None, 0, None

        target_state, _ = self.target_state  # type: ignore[assignment]
        target_pos = target_state.position
        theta_i = self._theta_2d(target_pos, own_position)

        candidates = []
        for agent_id, (state, rxtime) in self.agent_states.items():
            if agent_id == self.node_id:
                continue
            if (now - rxtime) > AGENT_STATE_TIMEOUT:
                continue
            theta_j = self._theta_2d(target_pos, state.position)
            candidates.append((agent_id, theta_j))

        theta_by_id = {int(agent_id): float(theta_j) for agent_id, theta_j in candidates}

        alive_count = len(candidates)
        if alive_count == 0:
            return None, None, None, None, 0, theta_i

        ring: list[tuple[float, int]] = [(theta_i, int(self.node_id))]
        ring.extend((theta_j, int(agent_id)) for agent_id, theta_j in candidates)
        ring.sort(key=lambda x: (x[0], x[1]))

        self_idx = next((k for k, (_, aid) in enumerate(ring) if aid == int(self.node_id)), None)
        if self_idx is None:
            return None, None, None, None, 0, theta_i

        pred_theta, pred_id_int = ring[(self_idx - 1) % len(ring)]
        succ_theta, succ_id_int = ring[(self_idx + 1) % len(ring)]

        pred_id = int(pred_id_int)
        succ_id = int(succ_id_int)
        pred_gap = self.wrap_to_2pi(theta_i - pred_theta)
        succ_gap = self.wrap_to_2pi(succ_theta - theta_i)

        crossed_swap = (
            self.neighbor_pred_id is not None
            and self.neighbor_succ_id is not None
            and pred_id == self.neighbor_succ_id
            and succ_id == self.neighbor_pred_id
        )

        def _expired_or_missing(agent_id: Optional[int]) -> bool:
            return agent_id is None or not self._agent_is_alive(agent_id, now)

        if (
            not crossed_swap
            and not _expired_or_missing(self.neighbor_pred_id)
            and pred_id != self.neighbor_pred_id
        ):
            if pred_gap is not None and self.neighbor_pred_id is not None:
                old_theta = theta_by_id.get(int(self.neighbor_pred_id))
                if old_theta is not None:
                    old_gap = self.wrap_to_2pi(theta_i - old_theta)
                    improvement = old_gap - pred_gap
                    if improvement <= HYSTERESIS_RAD:
                        pred_id = int(self.neighbor_pred_id)
                        pred_gap = old_gap

        if (
            not crossed_swap
            and not _expired_or_missing(self.neighbor_succ_id)
            and succ_id != self.neighbor_succ_id
        ):
            if succ_gap is not None and self.neighbor_succ_id is not None:
                old_theta = theta_by_id.get(int(self.neighbor_succ_id))
                if old_theta is not None:
                    old_gap = self.wrap_to_2pi(old_theta - theta_i)
                    improvement = old_gap - succ_gap
                    if improvement <= HYSTERESIS_RAD:
                        succ_id = int(self.neighbor_succ_id)
                        succ_gap = old_gap

        return pred_id, succ_id, pred_gap, succ_gap, alive_count, theta_i

    def handle_timer(self, timer: str):
        if timer == FAILURE_CHECK_TIMER_STR:
            if self._failed:
                return

            dt = float(FAILURE_CHECK_PERIOD)
            if not (math.isfinite(dt) and dt > 0.0):
                dt = 1.0

            rate_per_min = float(FAILURE_MEAN_FAILURES_PER_MIN)
            if math.isfinite(rate_per_min) and rate_per_min > 0.0:
                lam = rate_per_min / 60.0
                try:
                    p = 1.0 - math.exp(-lam * dt)
                except Exception:
                    p = 0.0
            else:
                p = 0.0

            if not math.isfinite(p):
                p = 0.0
            p = max(0.0, min(1.0, p))

            rng = getattr(self, "_failure_rng", None)
            draw = rng.random() if rng is not None else random.random()
            if p > 0.0 and draw < p:
                self._failed = True

                # Log failure_start event for the heatmap overlay.
                self._event_rows.append({
                    "timestamp": float(self.provider.current_time()),
                    "node_id": int(self.node_id),
                    "event_type": "failure_start",
                    "amplitude": 0.0,
                })

                if self._vis is not None:
                    try:
                        self._vis.paint_node(self.node_id, (255.0, 0.0, 0.0))
                    except Exception:
                        pass

                self.provider.cancel_timer(CONTROL_LOOP_TIMER_STR)

                if self.velocity_handler is not None:
                    try:
                        self.velocity_handler.set_velocity(self.node_id, (0.0, 0.0, 0.0))
                    except Exception:
                        pass

                off_time = float(FAILURE_OFF_TIME)
                if not (math.isfinite(off_time) and off_time > 0.0):
                    off_time = 0.0
                self.schedule_failure_recover_timer(off_time)

                if SIM_DEBUG:
                    print(f"Agent {self.node_id} ENTERED FAILURE for {off_time:.2f}s")
                return

            self.schedule_failure_check_timer()
            return

        if timer == FAILURE_RECOVER_TIMER_STR:
            self._failed = False
            self.prop_layer.on_reset()
            self.fast_layer.on_reset()
            self.last_prop_signal = 0.0
            self.u_local = 0.0
            self.u_prop = 0.0
            # Clear last-seen identity so the next tick doesn't fire a spurious
            # "neighbor changed" pulse from comparing pre-failure state to post-recovery.
            self._last_pred_id_for_event = None
            self._last_succ_id_for_event = None
            self._last_e_tau_for_event = 0.0

            # Log failure_end event for the heatmap overlay.
            self._event_rows.append({
                "timestamp": float(self.provider.current_time()),
                "node_id": int(self.node_id),
                "event_type": "failure_end",
                "amplitude": 0.0,
            })

            if self._vis is not None:
                try:
                    self._vis.paint_node(self.node_id, (0.0, 0.0, 255.0))
                except Exception:
                    pass

            self.schedule_control_loop_timer()
            self.schedule_failure_check_timer()
            if SIM_DEBUG:
                print(f"Agent {self.node_id} RECOVERED from FAILURE")
            return

        if timer == CONTROL_LOOP_TIMER_STR:
            if self._failed:
                return

            # 1) Read local kinematics
            if self.velocity_handler:
                position = self.velocity_handler.get_node_position(self.node_id)
                velocity = self.velocity_handler.get_node_velocity(self.node_id)
            else:
                position = (0.0, 0.0, 0.0)
                velocity = (0.0, 0.0, 0.0)

            # 2) Neighbor calculations
            now = self.provider.current_time()
            pred_gap, succ_gap = self._refresh_neighbors(now, position)
            u_pred, u_succ = self._get_neighbor_values()

            # 3) Radial Control loop
            v_rad: Tuple[float, float, float] = (0.0, 0.0, 0.0)
            v_target: Tuple[float, float, float] = (0.0, 0.0, 0.0)

            if self.target_state is not None:
                target_state, _ = self.target_state
                v_target = target_state.velocity

            if self.velocity_handler is not None and self.target_state is not None:
                target_state, _ = self.target_state
                p_t = target_state.position

                r_vec = (position[0] - p_t[0], position[1] - p_t[1])
                e_r, r = self._unit2(r_vec, eps=1e-6)
                v_r_corr = self.radial_controller.update(
                    measurement=r,
                    dt=float(self.control_period),
                )

                v_rad_xy = (v_r_corr * e_r[0], v_r_corr * e_r[1])
                v_rad_z = 0.0

                if not (math.isfinite(v_rad_xy[0]) and math.isfinite(v_rad_xy[1]) and math.isfinite(v_rad_z)):
                    v_rad_xy = (0.0, 0.0)
                    v_rad_z = 0.0

                v_rad = (v_rad_xy[0], v_rad_xy[1], v_rad_z)

                if SIM_DEBUG:
                    print(
                        f"Agent {self.node_id} radial: r={r:.3f}, "
                        f"v_r_corr={v_r_corr:.3f}, v_rad={v_rad}"
                    )
            else:
                self.radial_controller.reset()

            # 4) Tangential control
            v_tau: Tuple[float, float, float] = (0.0, 0.0, 0.0)
            v_spin: Tuple[float, float, float] = (0.0, 0.0, 0.0)

            if self.target_state is not None:
                target_state, _ = self.target_state

                t_hat = self.compute_tangential_unit_vector(target_state.position, position)

                r_xy = math.hypot(position[0] - target_state.position[0], position[1] - target_state.position[1])
                r_min = float(R_MIN)
                if not (math.isfinite(r_min) and r_min > 0.0):
                    r_min = 1e-6
                if math.isfinite(r_xy):
                    r_eff = max(r_xy, r_min)
                else:
                    r_eff = r_min

                e_tau, e_tau_eff, e_tau_used = self.compute_e_tau_used(
                    pred_gap=pred_gap,
                    succ_gap=succ_gap,
                    t_hat=t_hat,
                    r_eff=r_eff,
                )
                self.last_e_tau = float(e_tau) if math.isfinite(e_tau) else 0.0
                self.last_e_tau_eff = float(e_tau_eff) if math.isfinite(e_tau_eff) else self.last_e_tau

                # Propagation layer update (fast information channel)
                _pred_prop = self.neighbor_pred_state.prop_state if self.neighbor_pred_state is not None else None
                _succ_prop = self.neighbor_succ_state.prop_state if self.neighbor_succ_state is not None else None
                self.prop_layer.update(
                    e_tau=float(e_tau_used),  # aligned with local controller input
                    dt=float(self.control_period),
                    pred_state=_pred_prop,
                    succ_state=_succ_prop,
                )
                # get_neighbor_signal() returns only what arrived from ring neighbors,
                # excluding the node's own self-injection (no double-counting).
                _neighbor_sig = self.prop_layer.get_neighbor_signal()
                self.last_prop_signal = float(_neighbor_sig) if math.isfinite(_neighbor_sig) else 0.0

                # Fast soliton-like channel — runs in parallel, observation-only.
                # Read neighbor fast_state, advect, and (after) detect events to
                # decide whether to inject a pulse this tick.
                _pred_fast = self.neighbor_pred_state.fast_state if self.neighbor_pred_state is not None else None
                _succ_fast = self.neighbor_succ_state.fast_state if self.neighbor_succ_state is not None else None
                self.fast_layer.update(
                    e_tau=0.0,  # ignored by DampedAdvectionLayer; pulses are external
                    dt=float(self.control_period),
                    pred_state=_pred_fast,
                    succ_state=_succ_fast,
                )

                # Event detection: predecessor or successor identity changed
                # since the previous tick. Fire a single pulse with amplitude
                # proportional to delta_e_tau (signed).
                # Suppress firing during the initial warmup window — neighbor
                # identities take a few ticks to settle as broadcasts arrive,
                # producing spurious "changes" that are not real events.
                pred_changed = (
                    self._last_pred_id_for_event is not None
                    and self.neighbor_pred_id is not None
                    and self.neighbor_pred_id != self._last_pred_id_for_event
                )
                succ_changed = (
                    self._last_succ_id_for_event is not None
                    and self.neighbor_succ_id is not None
                    and self.neighbor_succ_id != self._last_succ_id_for_event
                )
                in_warmup = float(now) < float(FAST_CHANNEL_WARMUP_SEC)
                if (pred_changed or succ_changed) and not in_warmup:
                    delta_e = float(e_tau) - float(self._last_e_tau_for_event)
                    # Threshold scales with the agent's own desired arc, computed
                    # from the broadcast lambdas. N-agnostic by construction.
                    desired_gap_self = self._compute_desired_gap_self()
                    threshold = float(MIN_EVENT_DELTA_FRAC) * desired_gap_self
                    if math.isfinite(delta_e) and abs(delta_e) > threshold:
                        amplitude = float(K_TRIGGER) * delta_e
                        self.fast_layer.inject_pulse(amplitude)
                        self._event_rows.append({
                            "timestamp": float(self.provider.current_time()),
                            "node_id": int(self.node_id),
                            "event_type": "pulse_injected",
                            "amplitude": float(amplitude),
                        })

                # Update event-trigger memory for next tick.
                self._last_pred_id_for_event = self.neighbor_pred_id
                self._last_succ_id_for_event = self.neighbor_succ_id
                self._last_e_tau_for_event = float(e_tau)

                tangential_output = self.tangential_controller.update(
                    measurement=float(e_tau_used),
                    dt=float(self.control_period),
                    prop_signal=self.last_prop_signal,
                    k_prop=self._prop_k_prop,
                )

                # Spin term (unchanged)
                omega_ref_target = getattr(target_state, "omega_ref", 0.0)
                try:
                    omega_ref_target = float(omega_ref_target)
                except Exception:
                    omega_ref_target = 0.0

                if math.isfinite(omega_ref_target) and omega_ref_target != 0.0 and math.isfinite(r_xy) and r_xy > 1e-6:
                    v_spin_xy = omega_ref_target * r_xy
                    v_spin = (v_spin_xy * t_hat[0], v_spin_xy * t_hat[1], 0.0)

                self.du_damp = self._safe_float(tangential_output.du_damp)
                self.du_from_e_tau = self._safe_float(tangential_output.du_from_error)
                self.delta_u = self._safe_float(tangential_output.delta_u)
                self.u = self._safe_float(tangential_output.u)
                self.u_local = self._safe_float(tangential_output.u_local)
                self.u_prop = self._safe_float(tangential_output.u_prop)

                # Final v_tau using the (possibly anti-windup adjusted) self.u
                v_tau = self.compute_tangential_velocity(self.u, t_hat, r_eff=r_eff)

                if SIM_DEBUG:
                    print(
                        f"Agent {self.node_id} tangential: e_tau={self.last_e_tau:.3f}, "
                        f"e_tau_eff={self.last_e_tau_eff:.3f}, "
                        f"u_pred={u_pred:.3f}, u_succ={u_succ:.3f}, u={self.u:.3f}, "
                        f"delta_u={self.delta_u:.3f}, v_tau={v_tau}"
                    )

            # Compute local discrete curvature u_ss (1-hop).
            u_ss_local = 0.0
            if self.neighbor_pred_state is not None and self.neighbor_succ_state is not None:
                if math.isfinite(u_pred) and math.isfinite(u_succ) and math.isfinite(self.u):
                    u_ss_local = float(u_succ - 2.0 * self.u + u_pred)
            self.u_ss = float(u_ss_local) if math.isfinite(u_ss_local) else 0.0

            # 5) Broadcast the agent current state
            seq = self.agent_state_seq
            agent_state = AgentState(
                agent_id=self.node_id,
                seq=seq,
                position=position,
                velocity=velocity,
                u=self.u,
                u_ss=self.u_ss,
                prop_state=self.prop_layer.get_broadcast_state(),
                fast_state=self.fast_layer.get_broadcast_state(),
            )
            message_json = agent_state.to_json()
            command = CommunicationCommand(CommunicationCommandType.BROADCAST, message_json)
            self.provider.send_communication_command(command)

            if SIM_DEBUG:
                print(
                    f"Agent {self.node_id} broadcasted AgentState "
                    f"seq={seq}, position={position}, velocity={velocity}, u={self.u}, u_ss={self.u_ss}"
                )

            self.agent_state_seq = seq + 1

            # Final velocity composition and command application.
            if self.velocity_handler is not None:
                v_cmd_base = self.compose_final_velocity(v_rad, v_tau, v_target)
                v_cmd = (
                    v_cmd_base[0] + v_spin[0],
                    v_cmd_base[1] + v_spin[1],
                    v_cmd_base[2] + v_spin[2],
                )

                if not (math.isfinite(v_cmd[0]) and math.isfinite(v_cmd[1]) and math.isfinite(v_cmd[2])):
                    v_cmd = v_target

                v_cmd = self._clamp_velocity_to_limits(v_cmd)
                self.desired_velocity = v_cmd
                self.velocity_handler.set_velocity(self.node_id, self.desired_velocity)

                if SIM_DEBUG:
                    print(f"Agent {self.node_id} v_cmd={self.desired_velocity}")

            self.schedule_control_loop_timer()


    def handle_packet(self, message: str):
        if getattr(self, "_failed", False):
            return
        try:
            data = json.loads(message)
        except Exception as exc:
            self._logger.warning("Agent %s: failed to parse packet as JSON (%s): %r", self.node_id, exc, message)
            return

        msg_type = data.get("type")
        if msg_type == TargetState.TYPE:
            try:
                state = TargetState.from_json(message)
            except Exception as exc:
                self._logger.warning("Agent %s: failed to decode TargetState (%s): %r", self.node_id, exc, message)
            else:
                now = self.provider.current_time()
                target_expired = True
                if self.target_state is not None:
                    _, last_rxtime = self.target_state
                    target_expired = (now - last_rxtime) > TARGET_STATE_TIMEOUT

                if state.seq <= self.last_seq_target and not target_expired:
                    return
                self.last_seq_target = state.seq
                rxtime = now
                self.target_state = (state, rxtime)

                self._update_neighbor_lps_from_target()

                if self._vis is not None:
                    try:
                        self._vis.paint_node(self.node_id, (0.0, 0.0, 255.0))
                    except Exception:
                        pass
                if SIM_DEBUG:
                    ts, rxtime = self.target_state
                    print(
                        f"Agent {self.node_id} received TargetState "
                        f"rxtime={rxtime:.3f}, seq={ts.seq}, target_id={ts.target_id}, position={ts.position}, velocity={ts.velocity}"
                    )
            return

        if msg_type == AgentState.TYPE:
            try:
                state = AgentState.from_json(message)
            except Exception as exc:
                self._logger.warning("Agent %s: failed to decode AgentState (%s): %r", self.node_id, exc, message)
                return

            now = self.provider.current_time()
            last_seq = self.last_seq_agent.get(state.agent_id, -1)

            agent_expired = True
            prev_entry = self.agent_states.get(state.agent_id)
            if prev_entry is not None:
                _, last_rxtime = prev_entry
                agent_expired = (now - last_rxtime) > AGENT_STATE_TIMEOUT

            if state.seq <= last_seq and not agent_expired:
                return

            self.last_seq_agent[state.agent_id] = state.seq
            rxtime = now

            if SIM_DEBUG:
                print(
                    f"Agent {self.node_id} received AgentState "
                    f"rxtime={rxtime:.3f}, seq={state.seq}, agent_id={state.agent_id}, position={state.position}, "
                    f"velocity={state.velocity}, u={state.u}, u_ss={getattr(state, 'u_ss', 0.0)}"
                )

            if state.agent_id != self.node_id:
                self.agent_states[state.agent_id] = (state, rxtime)
            return

        self._logger.debug("Agent %s: ignoring packet type=%r", self.node_id, msg_type)

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        if not self._csv_path:
            return

        now = float(self.provider.current_time())

        if self.velocity_handler:
            vx, vy, vz = self.velocity_handler.get_node_velocity(self.node_id)
        else:
            vx = vy = vz = 0.0

        v_norm = float(math.sqrt(vx * vx + vy * vy + vz * vz))

        # Position relative to target (for spatial ordering in heatmaps).
        # Computed defensively: 0.0 if target state is missing.
        theta_rel = 0.0
        if self.velocity_handler and self.target_state is not None:
            try:
                pos = self.velocity_handler.get_node_position(self.node_id)
                ts, _ = self.target_state
                theta_rel = float(self._theta_2d(ts.position, pos))
            except Exception:
                theta_rel = 0.0

        self._telemetry_rows.append(
            {
                "node_id": int(self.node_id),
                "timestamp": now,

                # dt for correct reconstruction of increments
                "dt_u": float(self.control_period),

                # muscle state
                "u": float(self.u),
                "u_local": float(self.u_local),
                "u_prop": float(self.u_prop),
                "u_ss": float(self.u_ss),
                "prop_signal": float(self.last_prop_signal),

                # muscle per-tick / per-step terms
                "delta_u": float(self.delta_u),
                "du_damp": float(self.du_damp),
                "du_from_e_tau": float(self.du_from_e_tau),

                # spacing error
                "e_tau": float(self.last_e_tau),
                "e_tau_eff": float(self.last_e_tau_eff),

                "velocity_norm": v_norm,

                # Fast-channel observation (Phase A: not used in u_total)
                "u_R": float(self.fast_layer.u_R),
                "u_L": float(self.fast_layer.u_L),
                "fast_signal": float(self.fast_layer.get_signal()),

                # Spatial position around target for heatmap ordering
                "theta_rel": float(theta_rel),
            }
        )

    def finish(self):
        if self._csv_path and self._telemetry_rows:
            try:
                df = pd.DataFrame(self._telemetry_rows)
                file_exists = os.path.exists(self._csv_path)
                df.to_csv(self._csv_path, mode="a", header=not file_exists, index=False)
            except Exception as exc:
                self._logger.warning(
                    "Agent %s: failed to write telemetry CSV (%s): %r",
                    getattr(self, "node_id", "?"),
                    exc,
                    self._csv_path,
                )

        # Append sparse event log (failure_start, failure_end, pulse_injected).
        if self._events_csv_path and self._event_rows:
            try:
                df_events = pd.DataFrame(
                    self._event_rows,
                    columns=["timestamp", "node_id", "event_type", "amplitude"],
                )
                file_exists = os.path.exists(self._events_csv_path)
                df_events.to_csv(
                    self._events_csv_path,
                    mode="a",
                    header=not file_exists,
                    index=False,
                )
            except Exception as exc:
                self._logger.warning(
                    "Agent %s: failed to write events CSV (%s): %r",
                    getattr(self, "node_id", "?"),
                    exc,
                    self._events_csv_path,
                )



