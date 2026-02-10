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

from config_param import (
    CONTROL_LOOP_TIMER_STR,
    CONTROL_PERIOD,
    SOLITON_LOOP_TIMER_STR,
    SOLITON_PERIOD,
    SIM_DEBUG,
    AGENT_STATE_TIMEOUT,
    TARGET_STATE_TIMEOUT,
    HYSTERESIS_RAD,
    PRUNE_EXPIRED_STATES,
    ENCIRCLEMENT_RADIUS,
    R_MIN,
    K_R,
    K_DR,
    VM_MAX_SPEED_XY,
    VM_MAX_SPEED_Z,
    K_TAU,
    BETA_U,
    ALPHA_U,
    C_COUPLING,
    K_E_TAU,
    K_OMEGA_DAMP,
    KAPPA_U_DIFF,
    K_U_STEEPEN,
    KAPPA_U_DISP,
    BETA_Q,
    K_Q_TO_U,
    USE_Q_LAYER,
    USE_SOFT_LIMITER_U,
    U_lim,
    FAILURE_CHECK_PERIOD,
    FAILURE_CHECK_TIMER_STR,
    FAILURE_ENABLE,
    FAILURE_MEAN_FAILURES_PER_MIN,
    FAILURE_OFF_TIME,
    FAILURE_RECOVER_TIMER_STR,
    EXPERIMENT_REPRODUCIBLE,
    ANTI_WINDUP_ENABLE,
    Q_LAYER_ARCH,
    KDV_TYPE,
    GAMMA_Q_CUBIC,
    # Legacy: q -> K_Q_TO_U gain modulator (used when Q_LAYER_ARCH == "MODULATE_K_Q_TO_U")
    Q_MOD_DELTA,
    Q_MOD_MU,
    Q_MOD_NU,
    Q_MOD_EPS,
    Q_MOD_USE_TANH,
    Q_MOD_TANH_GAIN,
    Q_MOD_FREEZE_ON_SAT,
    # New: q -> u-parameter modulation (used when Q_LAYER_ARCH == "MODULATE_PARAMS")
    Q_PARAM_ACC_DELTA,
    Q_PARAM_ACC_BIDIR,
    Q_PARAM_ACC_MIN_FACTOR,
    Q_PARAM_ACC_MAX_FACTOR,
    Q_PARAM_DIFF_ADD,
    Q_PARAM_BETA_ADD,
    Q_PARAM_DIFF_DELTA,
    Q_PARAM_BETA_DELTA,
    Q_ROUGH_MU,
    Q_ROUGH_NU,
    Q_ROUGH_EPS,
    Q_ROUGH_USE_TANH,
    Q_ROUGH_TANH_GAIN,
    )


from protocol_messages import AgentState, TargetState


class AgentProtocol(IProtocol):
    """Implementation of agent protocol."""

    def __init__(self):
        # --- Q-layer modulation state (inicializar ANTES de qualquer uso) ---
        self.q0 = 1.0
        self.q0_rough = 1.0
        self.q_mod = 0.0
        self.q_mod_f = 0.0
        self.m_robust = 0.0
        self.m_robust_f = 0.0
        # Inicializa arquitetura do Q-layer a partir do config
        self.q_layer_arch = Q_LAYER_ARCH

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
        self.soliton_period = SOLITON_PERIOD  # Fast error-soliton loop period in seconds
        self.use_q_layer = bool(USE_Q_LAYER)

        # Schedule the control loop timer for the first time
        self.schedule_control_loop_timer()
        if self.use_q_layer:
            self.schedule_soliton_loop_timer()

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

        # Control internal state (muscle)
        self.u: float = 0.0
        self.u_ss: float = 0.0

        # Optional legacy fields kept only for backward compatibility in telemetry/plots.
        # In the two-layer architecture (q -> u), KdV lives ONLY in q.
        self.u_kdv: float = 0.0
        self.u_nom: float = 0.0
        self.u_err: float = 0.0

        # Error-soliton state (KdV-like field)
        self.q: float = 0.0
        self.q_ss: float = 0.0
        self.delta_q: float = 0.0
        self.dq_steepen: float = 0.0
        self.dq_disp: float = 0.0
        self.dq_damp: float = 0.0
        self.dq_force: float = 0.0

        # Last spacing error values used for q forcing.
        self.last_e_tau: float = 0.0
        self.last_e_tau_eff: float = 0.0

        # Per-control-tick increments (dt_u * du) for analysis/telemetry.
        self.delta_u: float = 0.0

        # NEW (v2) u-loop term telemetry (derivatives)
        self.du_coupling: float = 0.0
        self.du_diff: float = 0.0
        self.du_damp: float = 0.0
        self.du_nonlinear: float = 0.0
        self.du_from_q: float = 0.0
        self.du_from_e_tau: float = 0.0

        # Last commanded velocity (world coordinates).
        self.desired_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)

        # Anti-windup per-tick flag (set in CONTROL loop, read in SOLITON loop)
        self._aw_active: bool = False

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

            # --- Q-layer modulation state ---
            self.q0 = 1.0
            self.q0_rough = 1.0
            self.q_mod = 0.0
            self.q_mod_f = 0.0
            self.m_robust = 0.0
            self.m_robust_f = 0.0

    def schedule_control_loop_timer(self):
        self.provider.schedule_timer(CONTROL_LOOP_TIMER_STR, self.provider.current_time() + self.control_period)

    def schedule_soliton_loop_timer(self):
        self.provider.schedule_timer(SOLITON_LOOP_TIMER_STR, self.provider.current_time() + self.soliton_period)

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

        self.neighbor_pred_id = pred_id
        self.neighbor_succ_id = succ_id
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

    def _get_neighbor_values(self) -> Tuple[float, float, float, float, float, float, float, float]:
        """Return neighbor u/q values with finite fallbacks."""
        u_pred = 0.0
        u_succ = 0.0
        u_ss_pred = 0.0
        u_ss_succ = 0.0
        q_pred = 0.0
        q_succ = 0.0
        q_ss_pred = 0.0
        q_ss_succ = 0.0

        if self.neighbor_pred_state is not None:
            u_pred = self._safe_float(self.neighbor_pred_state.u)
            u_ss_pred = self._safe_float(getattr(self.neighbor_pred_state, "u_ss", 0.0))
            q_pred = self._safe_float(getattr(self.neighbor_pred_state, "q", 0.0))
            q_ss_pred = self._safe_float(getattr(self.neighbor_pred_state, "q_ss", 0.0))
        if self.neighbor_succ_state is not None:
            u_succ = self._safe_float(self.neighbor_succ_state.u)
            u_ss_succ = self._safe_float(getattr(self.neighbor_succ_state, "u_ss", 0.0))
            q_succ = self._safe_float(getattr(self.neighbor_succ_state, "q", 0.0))
            q_ss_succ = self._safe_float(getattr(self.neighbor_succ_state, "q_ss", 0.0))

        return u_pred, u_succ, u_ss_pred, u_ss_succ, q_pred, q_succ, q_ss_pred, q_ss_succ

    def update_error_soliton_q(
        self,
        *,
        q: float,
        q_pred: float,
        q_succ: float,
        q_ss_pred: float = 0.0,
        q_ss_succ: float = 0.0,
        e_tau: float,
        dt_q: float,
    ) -> Tuple[float, float, float, dict]:
        """Discrete-time update for q (error-soliton layer).

        Two options (choose via KDV_TYPE in config_param.py):
        - "ORIGINAL": central-form steepening term  (KdV-like, but can blow up under shocks)
        - "BURGERS":  conservative nonlinear transport using a Rusanov flux (KdV-Burgers-like, shock-safe)

        Model (dx=1):
            q_t + a * d/dx (0.5*q^2) = nu*q_xx + kappa*q_xxx - beta*q - gamma*q^3 + K_E_TAU*e_tau

        Notes:
        - a      = K_U_STEEPEN (nonlinear transport strength)
        - kappa  = KAPPA_U_DISP (dispersive, uses neighbor-provided q_ss to approximate q_xxx)
        - beta   = BETA_Q
        - gamma  = GAMMA_Q_CUBIC
        - forcing uses K_E_TAU (same forcing you already had)
        - nu (extra diffusion) is set to 0.0 by default (set in-code if needed)
        """

        # --------- safety: dt and inputs ----------
        dt = self._safe_float(dt_q, 0.0)
        if not (math.isfinite(dt) and dt > 0.0):
            dt = 0.0

        q = self._safe_float(q)
        q_pred = self._safe_float(q_pred)
        q_succ = self._safe_float(q_succ)
        q_ss_pred = self._safe_float(q_ss_pred)
        q_ss_succ = self._safe_float(q_ss_succ)
        e_tau_f = self._safe_float(e_tau)

        # Parameters
        a = self._safe_float(K_U_STEEPEN)
        kappa = self._safe_float(KAPPA_U_DISP)
        beta = self._safe_float(BETA_Q)
        gamma = self._safe_float(GAMMA_Q_CUBIC)
        k_force = self._safe_float(K_E_TAU)

        # Optional extra diffusion for stability (keep 0.0 to preserve your design)
        nu_q = 0.0

        kdv_type = str(KDV_TYPE).upper().strip()
        # Allow some aliases that better describe what's implemented
        if kdv_type in {"CONSERVATIVE", "CONSERVATIVE_FLUX", "RUSANOV", "GODUNOV"}:
            kdv_type = "BURGERS"

        def _rhs(q_local: float) -> Tuple[float, dict]:
            ql = self._safe_float(q_local)

            # 1-hop second derivative (curvature) at current q (for RHS evaluation)
            q_ss_local = float(q_succ - 2.0 * ql + q_pred)

            # neighbor-provided curvature values -> approximate 3rd derivative (1-hop)
            q_sss = float(q_ss_succ - q_ss_pred)

            # --- transport / steepening ---
            dq_adv = 0.0
            terms_flux = {}

            if kdv_type == "BURGERS":
                # Conservative nonlinear flux for Burgers-like transport:
                # f(q)=0.5*q^2, dq_adv = -a*(F_{i+1/2} - F_{i-1/2})  with Rusanov flux
                def f(x: float) -> float:
                    return 0.5 * x * x

                alpha_p = max(abs(ql), abs(q_succ))
                if not math.isfinite(alpha_p):
                    alpha_p = 0.0
                F_p = 0.5 * (f(ql) + f(q_succ)) - 0.5 * alpha_p * (q_succ - ql)

                alpha_m = max(abs(q_pred), abs(ql))
                if not math.isfinite(alpha_m):
                    alpha_m = 0.0
                F_m = 0.5 * (f(q_pred) + f(ql)) - 0.5 * alpha_m * (ql - q_pred)

                dq_adv = float(-a * (F_p - F_m))
                terms_flux = {"F_p": F_p, "F_m": F_m, "alpha_p": alpha_p, "alpha_m": alpha_m}
            else:
                # ORIGINAL: central steepening term (classic discretization you had)
                q_s = float(q_succ - q_pred)
                dq_adv = float(-a * ql * q_s)
                terms_flux = {"q_s": q_s}

            # Linear diffusion (optional)
            dq_diff = float(nu_q * q_ss_local)

            # Dispersive term (KdV-like)
            dq_disp = float(kappa * q_sss)

            # Damping (linear)
            dq_damp = float(-beta * ql)

            # Damping (cubic) - compute safely only if gamma != 0
            if gamma != 0.0:
                if abs(ql) > 5.0e102:
                    dq_damp_cubic = float(-gamma * math.copysign(1.0, ql) * 1.0e308)
                else:
                    dq_damp_cubic = float(-gamma * (ql * ql * ql))
            else:
                dq_damp_cubic = 0.0

            # Forcing by spacing error
            dq_force = float(k_force * e_tau_f)

            dq_total = float(dq_adv + dq_diff + dq_disp + dq_damp + dq_damp_cubic + dq_force)

            terms = {
                "kdv_type": kdv_type,
                "q_ss": q_ss_local,
                "q_sss": q_sss,
                "dq_adv": dq_adv,
                "dq_diff": dq_diff,
                "dq_disp": dq_disp,
                "dq_damp": dq_damp,
                "dq_damp_cubic": dq_damp_cubic,
                "dq_force": dq_force,
            }
            terms.update(terms_flux)
            return dq_total, terms

        if dt <= 0.0:
            dq0, terms0 = _rhs(q)
            q_ss_out = float(q_succ - 2.0 * q + q_pred)
            return float(q), float(q_ss_out), float(dq0), terms0

        # --------- CFL-style substepping (only critical for BURGERS) ----------
        n_sub = 1
        if kdv_type == "BURGERS" and a != 0.0:
            q_max = max(abs(q), abs(q_pred), abs(q_succ))
            if not math.isfinite(q_max):
                q_max = 0.0
            if q_max > 0.0:
                cfl = dt * abs(a) * q_max  # dx=1
                if cfl > 0.35:
                    n_sub = int(math.ceil(cfl / 0.35))
                    n_sub = max(1, min(n_sub, 10))

        dt_sub = dt / float(n_sub)

        q_curr = float(q)
        dq_last = 0.0
        terms_last: dict = {}
        for _ in range(n_sub):
            dq_last, terms_last = _rhs(q_curr)
            if not math.isfinite(dq_last):
                dq_last = 0.0
            q_next = float(q_curr + dt_sub * dq_last)
            if not math.isfinite(q_next):
                q_next = 0.0
            q_curr = q_next

        q_next = float(q_curr)

        # Standardize: curvature at updated q
        q_ss_out = float(q_succ - 2.0 * q_next + q_pred)
        if not math.isfinite(q_ss_out):
            q_ss_out = 0.0

        dq_out = float(dq_last)
        if not math.isfinite(dq_out):
            dq_out = 0.0

        # Compatibility/debug keys (old plots)
        terms_debug = dict(terms_last)
        terms_debug["dq_steepen"] = float(terms_last.get("dq_adv", 0.0))
        if "q_s" not in terms_debug:
            terms_debug["q_s"] = float(q_succ - q_pred)

        if not (math.isfinite(q_next) and math.isfinite(q_ss_out) and math.isfinite(dq_out)):
            return 0.0, 0.0, 0.0, {k: 0.0 for k in terms_debug}

        return q_next, q_ss_out, dq_out, terms_debug

    def update_control_u(
            self,
            *,
            u: float,
            u_pred: float,
            u_succ: float,
            q: float,
            e_tau: Optional[float] = None,
            dt_u: float,
            k_e_tau_override: Optional[float] = None,
            k_q_to_u_override: Optional[float] = None,
            kappa_u_diff_override: Optional[float] = None,
            beta_u_override: Optional[float] = None,
        ) -> Tuple[float, float, float, dict]:
            """Discrete-time control update for u (muscle), forced by q or directly by e_tau."""
            dt = float(dt_u)
            if not (math.isfinite(dt) and dt > 0.0):
                dt = 0.0

            u = self._safe_float(u)
            u_pred = self._safe_float(u_pred)
            u_succ = self._safe_float(u_succ)
            q_f = self._safe_float(q)

            if USE_SOFT_LIMITER_U and math.isfinite(U_lim) and U_lim > 0.0:
                u_abs = abs(u)
                u3 = u * u * u
                nonlinear = u3 / (1.0 + (u_abs / U_lim) * (u_abs / U_lim))
            else:
                nonlinear = u * u * u

            u_s = float(u_succ - u_pred)
            u_ss = float(u_succ - 2.0 * u + u_pred)

            # Use overrides if provided; otherwise use effective gains (which may be modulated by q-layer)
            beta_u = float(beta_u_override) if (beta_u_override is not None) else float(self.beta_u_eff)
            kappa_u = float(kappa_u_diff_override) if (kappa_u_diff_override is not None) else float(self.kappa_u_diff_eff)
            k_q_to_u = float(k_q_to_u_override) if (k_q_to_u_override is not None) else float(self.k_q_to_u_eff)
            k_e_tau = float(k_e_tau_override) if (k_e_tau_override is not None) else float(self.k_e_tau_eff)

            du_coupling = float(C_COUPLING * u_s)
            du_damp = float(-beta_u * u)
            du_nonlinear = float(-ALPHA_U * nonlinear)
            du_diff = float(kappa_u * u_ss)

            du_from_q = float(k_q_to_u * q_f)
            du_from_e_tau = 0.0
            if e_tau is not None:
                du_from_e_tau = float(k_e_tau * self._safe_float(e_tau))


            du = float(du_coupling + du_damp + du_nonlinear + du_diff + du_from_q + du_from_e_tau)
            u_next = float(u + dt * du)

            terms_debug = {
                "du_coupling": du_coupling,
                "du_damp": du_damp,
                "du_nonlinear": du_nonlinear,
                "du_diff": du_diff,
                "du_from_q": du_from_q,
                "du_from_e_tau": du_from_e_tau,
            }

            if not (math.isfinite(u_next) and math.isfinite(u_ss) and math.isfinite(du)):
                return 0.0, 0.0, 0.0, {key: 0.0 for key in terms_debug}

            return u_next, u_ss, du, terms_debug

    def compute_tangential_velocity(
        self,
        u: float,
        t_hat: Tuple[float, float],
        *,
        r_eff: float = 1.0,
    ) -> Tuple[float, float, float]:
        """Convert internal u into tangential velocity (XYZ), usando k_tau_eff modulado se disponível."""
        if not math.isfinite(r_eff) or r_eff <= 0.0:
            r_eff = 1.0
        k_tau = getattr(self, "k_tau_eff", float(K_TAU))
        v_tau_corr = k_tau * u * r_eff
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

    @staticmethod
    def _would_saturate(v: Tuple[float, float, float]) -> bool:
        """Check whether velocity exceeds configured limits (pre-clamp)."""
        vx, vy, vz = v
        if not (math.isfinite(vx) and math.isfinite(vy) and math.isfinite(vz)):
            return False
        v_xy = math.hypot(vx, vy)
        if math.isfinite(VM_MAX_SPEED_XY) and VM_MAX_SPEED_XY > 0.0 and v_xy > VM_MAX_SPEED_XY:
            return True
        if math.isfinite(VM_MAX_SPEED_Z) and VM_MAX_SPEED_Z > 0.0 and abs(vz) > VM_MAX_SPEED_Z:
            return True
        return False

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

                if self._vis is not None:
                    try:
                        self._vis.paint_node(self.node_id, (255.0, 0.0, 0.0))
                    except Exception:
                        pass

                self.provider.cancel_timer(CONTROL_LOOP_TIMER_STR)
                self.provider.cancel_timer(SOLITON_LOOP_TIMER_STR)

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

            if self._vis is not None:
                try:
                    self._vis.paint_node(self.node_id, (0.0, 0.0, 255.0))
                except Exception:
                    pass

            self.schedule_control_loop_timer()
            if self.use_q_layer:
                self.schedule_soliton_loop_timer()
            self.schedule_failure_check_timer()
            if SIM_DEBUG:
                print(f"Agent {self.node_id} RECOVERED from FAILURE")
            return

        if timer == SOLITON_LOOP_TIMER_STR:
            if not self.use_q_layer:
                return
            if self._failed:
                return

            if self.velocity_handler:
                position = self.velocity_handler.get_node_position(self.node_id)
            else:
                position = (0.0, 0.0, 0.0)

            now = self.provider.current_time()
            pred_gap, succ_gap = self._refresh_neighbors(now, position)
            (
                _u_pred,
                _u_succ,
                _u_ss_pred,
                _u_ss_succ,
                q_pred,
                q_succ,
                q_ss_pred,
                q_ss_succ,
            ) = self._get_neighbor_values()

            t_hat = None
            r_eff = None
            if math.isfinite(K_OMEGA_DAMP) and K_OMEGA_DAMP > 0.0 and self.target_state is not None:
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

            # Store "raw" errors for telemetry (even if we later zero e_tau_used for anti-windup)
            self.last_e_tau = float(e_tau) if math.isfinite(e_tau) else 0.0
            self.last_e_tau_eff = float(e_tau_eff) if math.isfinite(e_tau_eff) else self.last_e_tau

            # ----------------------------------------------------------------------------------
            # Anti-windup for q: if CONTROL loop saturated recently, drop forcing dq_force (e_tau_used -> 0)
            # ----------------------------------------------------------------------------------
            if bool(ANTI_WINDUP_ENABLE) and bool(getattr(self, "_aw_active", False)):
                e_tau_used = 0.0

            dt_q = float(self.soliton_period)
            q_next, q_ss_local, dq, terms = self.update_error_soliton_q(
                q=self.q,
                q_pred=q_pred,
                q_succ=q_succ,
                q_ss_pred=q_ss_pred,
                q_ss_succ=q_ss_succ,
                e_tau=e_tau_used,
                dt_q=dt_q,
            )

            if self.neighbor_pred_state is None or self.neighbor_succ_state is None:
                q_ss_local = 0.0

            self.q = float(q_next)
            self.q_ss = float(q_ss_local) if math.isfinite(q_ss_local) else 0.0
            self.delta_q = float(dt_q * dq) if math.isfinite(dq) else 0.0
            self.dq_steepen = self._safe_float(terms.get("dq_steepen", 0.0))
            self.dq_disp = self._safe_float(terms.get("dq_disp", 0.0))
            self.dq_damp = self._safe_float(terms.get("dq_damp", 0.0))
            self.dq_force = self._safe_float(terms.get("dq_force", 0.0))

            if SIM_DEBUG:
                q_s_dbg = self._safe_float(terms.get("q_s", 0.0))
                q_sss_dbg = self._safe_float(terms.get("q_sss", 0.0))
                print(
                    f"Agent {self.node_id} q-loop: e_tau={e_tau:.3f}, e_tau_eff={e_tau_eff:.3f}, "
                    f"q={self.q:.3f}, q_s={q_s_dbg:.3f}, q_sss={q_sss_dbg:.3f}, delta_q={self.delta_q:.3f}"
                )

            self.schedule_soliton_loop_timer()
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
            (
                u_pred,
                u_succ,
                _u_ss_pred,
                _u_ss_succ,
                _q_pred,
                _q_succ,
                _q_ss_pred,
                _q_ss_succ,
            ) = self._get_neighbor_values()

            # 3) Radial Control loop
            v_rad: Tuple[float, float, float] = (0.0, 0.0, 0.0)
            v_target: Tuple[float, float, float] = (0.0, 0.0, 0.0)

            if self.target_state is not None:
                target_state, _ = self.target_state
                v_target = target_state.velocity

            if self.velocity_handler is not None and self.target_state is not None:
                target_state, _ = self.target_state
                p_t = target_state.position
                v_t = target_state.velocity

                r_vec = (position[0] - p_t[0], position[1] - p_t[1])
                e_r, r = self._unit2(r_vec, eps=1e-6)

                e = r - ENCIRCLEMENT_RADIUS

                v_rel_xy = (velocity[0] - v_t[0], velocity[1] - v_t[1])
                v_r = self._dot2(v_rel_xy, e_r)

                v_r_corr = -K_R * e - K_DR * v_r

                v_rad_xy = (v_r_corr * e_r[0], v_r_corr * e_r[1])
                v_rad_z = 0.0

                if not (math.isfinite(v_rad_xy[0]) and math.isfinite(v_rad_xy[1]) and math.isfinite(v_rad_z)):
                    v_rad_xy = (0.0, 0.0)
                    v_rad_z = 0.0

                v_rad = (v_rad_xy[0], v_rad_xy[1], v_rad_z)

                if SIM_DEBUG:
                    print(
                        f"Agent {self.node_id} radial: r={r:.3f}, e={e:.3f}, "
                        f"v_r={v_r:.3f}, v_r_corr={v_r_corr:.3f}, v_rad={v_rad}"
                    )

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

                e_tau_used = None
                if not self.use_q_layer:
                    e_tau, e_tau_eff, e_tau_used = self.compute_e_tau_used(
                        pred_gap=pred_gap,
                        succ_gap=succ_gap,
                        t_hat=t_hat,
                        r_eff=r_eff,
                    )
                    self.last_e_tau = float(e_tau) if math.isfinite(e_tau) else 0.0
                    self.last_e_tau_eff = float(e_tau_eff) if math.isfinite(e_tau_eff) else self.last_e_tau

                # Q-layer: update modulation signals and choose how u is driven
                if self.use_q_layer:
                    # Use previous saturation flag to optionally freeze/zero modulation
                    self._update_q_modulators(abs_q=abs(self.q), abs_q_ss=abs(self.q_ss), freeze=bool(self._aw_active))
                    self._compute_effective_params()

                    if self.q_layer_arch == "MODULATE_K_Q_TO_U":
                        # Legacy architecture: u uses e_tau with a q-modulated K_E_TAU
                        e_tau_used = float(self.last_e_tau_eff)
                    else:
                        # FORCE_U / MODULATE_PARAMS: u is driven by q (no direct e_tau)
                        e_tau_used = None
                else:
                    # No q layer: reset effective params (no modulation)
                    self._compute_effective_params()

                dt_u = float(self.control_period)

                # --- First pass: compute u candidate with full injection (as usual) ---
                u_prev = float(self.u)
                # Select q input and per-parameter overrides depending on architecture
                q_for_u = 0.0
                u_overrides: dict = {}
                # if self.use_q_layer:
                #     if self.q_layer_arch == "MODULATE_K_Q_TO_U":
                #         q_for_u = float(self.q)
                #         u_overrides["k_q_to_u_override"] = self.k_q_to_u_eff
                #     else:
                #         q_for_u = 0.0
                #         if self.q_layer_arch == "MODULATE_PARAMS":
                #             u_overrides["k_e_tau_override"] = self.k_e_tau_eff
                if self.use_q_layer:

                    arch = str(self.q_layer_arch).upper().strip()

                    if arch == "FORCE_E":
                        q_for_u = 0.0
                        e_tau_used = float(self.last_e_tau_eff)
                        # Não modula nenhum ganho, usa apenas os valores default
                    elif arch in ("FORCE_U", "MODULATE_PARAMS", "MODULATE_K_Q_TO_U"):
                        q_for_u = float(self.q)

                    if arch == "MODULATE_K_Q_TO_U":
                        u_overrides["k_q_to_u_override"] = float(self.k_q_to_u_eff)

                u_next_full, _u_ss_local, du_full, u_terms = self.update_control_u(
                    u=u_prev,
                    u_pred=u_pred,
                    u_succ=u_succ,
                    q=q_for_u,
                    e_tau=e_tau_used,
                    dt_u=dt_u,
                    **u_overrides,
                )

                # Candidate tangential velocity using full u
                v_tau_full = self.compute_tangential_velocity(u_next_full, t_hat, r_eff=r_eff)

                # Spin term (unchanged)
                omega_ref_target = getattr(target_state, "omega_ref", 0.0)
                try:
                    omega_ref_target = float(omega_ref_target)
                except Exception:
                    omega_ref_target = 0.0

                if math.isfinite(omega_ref_target) and omega_ref_target != 0.0 and math.isfinite(r_xy) and r_xy > 1e-6:
                    v_spin_xy = omega_ref_target * r_xy
                    v_spin = (v_spin_xy * t_hat[0], v_spin_xy * t_hat[1], 0.0)

                # Compose PRE-CLAMP command candidate
                v_cmd_base_full = self.compose_final_velocity(v_rad, v_tau_full, v_target)
                v_cmd_candidate = (
                    v_cmd_base_full[0] + v_spin[0],
                    v_cmd_base_full[1] + v_spin[1],
                    v_cmd_base_full[2] + v_spin[2],
                )

                # Detect saturation (pre-clamp)
                did_sat = self._would_saturate(v_cmd_candidate)

                # Store aw flag for use by q-loop
                self._aw_active = bool(ANTI_WINDUP_ENABLE) and bool(did_sat)

                # --- Anti-windup action for u: if saturated, drop injection terms and re-integrate ---
                if self._aw_active:
                    du_coupling = self._safe_float(u_terms.get("du_coupling", 0.0))
                    du_diff = self._safe_float(u_terms.get("du_diff", 0.0))
                    du_damp = self._safe_float(u_terms.get("du_damp", 0.0))
                    du_nonlinear = self._safe_float(u_terms.get("du_nonlinear", 0.0))

                    du_aw = float(du_coupling + du_diff + du_damp + du_nonlinear)
                    u_next = float(u_prev + dt_u * du_aw)

                    # Telemetry terms: injection does not contribute in AW tick
                    self.du_coupling = du_coupling
                    self.du_diff = du_diff
                    self.du_damp = du_damp
                    self.du_nonlinear = du_nonlinear
                    self.du_from_q = 0.0
                    self.du_from_e_tau = 0.0

                    self.delta_u = float(dt_u * du_aw) if math.isfinite(du_aw) else 0.0
                    self.u = float(u_next) if math.isfinite(u_next) else 0.0

                else:
                    # Normal update: accept full u_next
                    self.du_coupling = self._safe_float(u_terms.get("du_coupling", 0.0))
                    self.du_diff = self._safe_float(u_terms.get("du_diff", 0.0))
                    self.du_damp = self._safe_float(u_terms.get("du_damp", 0.0))
                    self.du_nonlinear = self._safe_float(u_terms.get("du_nonlinear", 0.0))
                    self.du_from_q = self._safe_float(u_terms.get("du_from_q", 0.0))
                    self.du_from_e_tau = self._safe_float(u_terms.get("du_from_e_tau", 0.0))

                    self.delta_u = float(dt_u * du_full) if math.isfinite(du_full) else 0.0
                    self.u = float(u_next_full)

                if not math.isfinite(self.u) or not math.isfinite(self.delta_u):
                    self.u = 0.0
                    self.delta_u = 0.0
                    self.du_coupling = self.du_diff = self.du_damp = self.du_nonlinear = 0.0
                    self.du_from_q = self.du_from_e_tau = 0.0

                # Legacy fields (compatibility only): KdV is NOT in u in v2.
                self.u_kdv = 0.0
                self.u_nom = float(self.u)
                self.u_err = 0.0

                # Final v_tau using the (possibly anti-windup adjusted) self.u
                v_tau = self.compute_tangential_velocity(self.u, t_hat, r_eff=r_eff)

                if SIM_DEBUG:
                    u_s_dbg = float(u_succ - u_pred)
                    aw_str = "AW=1" if self._aw_active else "AW=0"
                    print(
                        f"Agent {self.node_id} tangential: {aw_str}, e_tau={self.last_e_tau:.3f}, "
                        f"e_tau_eff={self.last_e_tau_eff:.3f}, "
                        f"u_pred={u_pred:.3f}, u_succ={u_succ:.3f}, u={self.u:.3f}, "
                        f"q={self.q:.3f}, u_s={u_s_dbg:.3f}, delta_u={self.delta_u:.3f}, v_tau={v_tau}"
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
                q=self.q,
                q_ss=self.q_ss,
            )
            message_json = agent_state.to_json()
            command = CommunicationCommand(CommunicationCommandType.BROADCAST, message_json)
            self.provider.send_communication_command(command)

            if SIM_DEBUG:
                print(
                    f"Agent {self.node_id} broadcasted AgentState "
                    f"seq={seq}, position={position}, velocity={velocity}, u={self.u}, u_ss={self.u_ss}, "
                    f"q={self.q}, q_ss={self.q_ss}"
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
                    f"velocity={state.velocity}, u={state.u}, u_ss={getattr(state, 'u_ss', 0.0)}, "
                    f"q={getattr(state, 'q', 0.0)}, q_ss={getattr(state, 'q_ss', 0.0)}"
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

        self._telemetry_rows.append(
            {
                "node_id": int(self.node_id),
                "timestamp": now,

                # dt for correct reconstruction of increments
                "dt_u": float(self.control_period),
                "dt_q": float(self.soliton_period),

                # muscle state
                "u": float(self.u),
                "u_ss": float(self.u_ss),

                # legacy placeholders (compatibility)
                "u_kdv": float(self.u_kdv),
                "u_nom": float(self.u_nom),
                "u_err": float(self.u_err),

                # muscle per-tick / per-step terms
                "delta_u": float(self.delta_u),
                "du_coupling": float(self.du_coupling),
                "du_diff": float(self.du_diff),
                "du_damp": float(self.du_damp),
                "du_nonlinear": float(self.du_nonlinear),
                "du_from_q": float(self.du_from_q),
                "du_from_e_tau": float(self.du_from_e_tau),

                # KdV field
                "q": float(self.q),
                "q_ss": float(self.q_ss),
                "delta_q": float(self.delta_q),
                "dq_steepen": float(self.dq_steepen),
                "dq_disp": float(self.dq_disp),
                "dq_damp": float(self.dq_damp),
                "dq_force": float(self.dq_force),

                # spacing error
                "e_tau": float(self.last_e_tau),
                "e_tau_eff": float(self.last_e_tau_eff),

                "velocity_norm": v_norm,
            }
        )

    def finish(self):
        if not self._csv_path or not self._telemetry_rows:
            return

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

    # -------------------------------------------------------------------------
    # Helpers: numerical safety + q-driven modulation signals
    # -------------------------------------------------------------------------
    @staticmethod
    def _safe_float(x: float, default: float = 0.0) -> float:
        """Return float(x) if finite, else default."""
        try:
            xf = float(x)
        except Exception:
            return float(default)
        return xf if math.isfinite(xf) else float(default)

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    @staticmethod
    def _clamp01(x: float) -> float:
        return AgentProtocol._clamp(float(x), 0.0, 1.0)

    def _map_to_unit(self, z: float, *, use_tanh: bool, tanh_gain: float) -> float:
        """Map z>=0 to m in [0,1]."""
        zf = self._safe_float(z, 0.0)
        if zf < 0.0:
            zf = 0.0
        if use_tanh:
            g = self._safe_float(tanh_gain, 1.0)
            return self._clamp01(math.tanh(g * zf))
        # linear clip
        return self._clamp01(zf)

    def _update_q_modulators(self, *, abs_q: float, abs_q_ss: float, freeze: bool) -> None:
        """Update (q_mod_f, m_robust_f) from |q| and |q_ss| with scale tracking + low-pass filtering."""
        if freeze and Q_MOD_FREEZE_ON_SAT:
            # Freeze/zero to avoid gain run-away while saturated.
            self.q_mod = 0.0
            self.q_mod_f = 0.0
            self.m_robust = 0.0
            self.m_robust_f = 0.0
            return

        # --- scale tracking for |q| ---
        a_q = self._safe_float(abs_q, 0.0)
        mu = self._clamp01(self._safe_float(Q_MOD_MU, 0.0))
        # Update q0 as EMA of |q|
        self.q0 = (1.0 - mu) * self.q0 + mu * a_q
        if not math.isfinite(self.q0) or self.q0 < Q_MOD_EPS:
            self.q0 = 1.0

        z = a_q / (self.q0 + Q_MOD_EPS)
        self.q_mod = self._map_to_unit(z, use_tanh=bool(Q_MOD_USE_TANH), tanh_gain=float(Q_MOD_TANH_GAIN))
        nu = self._clamp01(self._safe_float(Q_MOD_NU, 0.0))
        self.q_mod_f = (1.0 - nu) * self.q_mod_f + nu * self.q_mod

        # --- scale tracking for |q_ss| (roughness) ---
        a_qss = self._safe_float(abs_q_ss, 0.0)
        mu_r = self._clamp01(self._safe_float(Q_ROUGH_MU, 0.0))
        self.q0_rough = (1.0 - mu_r) * self.q0_rough + mu_r * a_qss
        if not math.isfinite(self.q0_rough) or self.q0_rough < Q_ROUGH_EPS:
            self.q0_rough = 1.0

        z_r = a_qss / (self.q0_rough + Q_ROUGH_EPS)
        self.m_robust = self._map_to_unit(z_r, use_tanh=bool(Q_ROUGH_USE_TANH), tanh_gain=float(Q_ROUGH_TANH_GAIN))
        nu_r = self._clamp01(self._safe_float(Q_ROUGH_NU, 0.0))
        self.m_robust_f = (1.0 - nu_r) * self.m_robust_f + nu_r * self.m_robust

        # Safety: keep in bounds
        self.q_mod_f = self._clamp01(self.q_mod_f)
        self.m_robust_f = self._clamp01(self.m_robust_f)

    def _compute_effective_params(self) -> None:
        """Compute effective gains according to Q layer architecture."""
        # Defaults (no modulation)
        self.k_e_tau_eff = float(K_E_TAU)
        self.k_q_to_u_eff = float(K_Q_TO_U)
        self.kappa_u_diff_eff = float(KAPPA_U_DIFF)
        self.beta_u_eff = float(BETA_U)
        self.k_tau_eff = float(K_TAU)

        if not self.use_q_layer:
            return

        arch = str(self.q_layer_arch).upper().strip()

        if arch == "FORCE_E":
            # Não modula nenhum ganho, usa apenas os valores default
            self.k_tau_eff = float(K_TAU)
            self.kappa_u_diff_eff = float(KAPPA_U_DIFF)
            self.beta_u_eff = float(BETA_U)
            self.k_q_to_u_eff = float(K_Q_TO_U)
            return

        if arch == "MODULATE_K_Q_TO_U":
            # Agora: boost K_Q_TO_U based on q_mod_f
            base_kq = float(K_Q_TO_U)
            delta = self._safe_float(Q_PARAM_ACC_DELTA, 0.0)
            if bool(Q_PARAM_ACC_BIDIR):
                factor = 1.0 + delta * (2.0 * float(self.q_mod_f) - 1.0)
            else:
                factor = 1.0 + delta * float(self.q_mod_f)

            factor = self._clamp(factor, float(Q_PARAM_ACC_MIN_FACTOR), float(Q_PARAM_ACC_MAX_FACTOR))
            self.k_q_to_u_eff = base_kq * factor
            return

        if arch == "MODULATE_PARAMS":
            # Modulação de K_TAU (aceleração)
            base_ktau = float(K_TAU)
            delta = self._safe_float(Q_PARAM_ACC_DELTA, 0.0)
            if bool(Q_PARAM_ACC_BIDIR):
                factor = 1.0 + delta * (2.0 * float(self.q_mod_f) - 1.0)
            else:
                factor = 1.0 + delta * float(self.q_mod_f)
            factor = self._clamp(factor, float(Q_PARAM_ACC_MIN_FACTOR), float(Q_PARAM_ACC_MAX_FACTOR))
            self.k_tau_eff = base_ktau * factor

            # Modulação de KAPPA_U_DIFF (robustez/difusão)
            base_kappa = float(KAPPA_U_DIFF)
            add = self._safe_float(Q_PARAM_DIFF_ADD, 0.0) * float(self.m_robust_f)
            mult = 1.0 + self._safe_float(Q_PARAM_DIFF_DELTA, 0.0) * float(self.m_robust_f)
            self.kappa_u_diff_eff = (base_kappa + add) * mult

            # Modulação de BETA_U (robustez/damping)
            base_beta = float(BETA_U)
            add_b = self._safe_float(Q_PARAM_BETA_ADD, 0.0) * float(self.m_robust_f)
            mult_b = 1.0 + self._safe_float(Q_PARAM_BETA_DELTA, 0.0) * float(self.m_robust_f)
            self.beta_u_eff = (base_beta + add_b) * mult_b
            return




