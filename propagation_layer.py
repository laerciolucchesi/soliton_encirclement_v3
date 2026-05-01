"""Propagation layer: fast information channel between ring neighbors.

Each agent in the ring maintains a local instance of a PropagationLayer subclass.
Every control tick the layer:
  1. Receives e_tau (raw gap error) and the broadcast state dicts of pred/succ neighbors.
  2. Updates its internal variables.
  3. Returns a scalar ``get_signal()`` (full local state, includes self-injection).
  4. Returns a scalar ``get_neighbor_signal()`` (only what arrived from ring neighbors,
     excluding the node's own e_tau injection — used by the propagated control channel).
  5. Returns ``get_broadcast_state()`` dict included in the AgentState message.

Method selection: done at runtime via main.py menu; factory ``create_propagation_layer``
instantiates the correct class.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class PropagationLayer(ABC):
    """Abstract base for all propagation mechanisms."""

    @abstractmethod
    def update(
        self,
        e_tau: float,
        dt: float,
        pred_state: dict | None,
        succ_state: dict | None,
    ) -> None:
        """Update internal state. Called every control tick."""

    @abstractmethod
    def get_signal(self) -> float:
        """Return the scalar signal based on the full local state (includes self-injection)."""

    def get_neighbor_signal(self) -> float:
        """Return the scalar signal representing only what arrived from ring neighbors.

        Excludes the node's own e_tau injection so there is no double-counting when
        this value is used alongside the explicit local-error term in the controller.
        Default implementation returns 0.0 (safe for BaselineLayer and any subclass
        that does not distinguish neighbor vs. self contributions).
        """
        return 0.0

    def get_broadcast_state(self) -> dict:
        """Return fields to include in AgentState.prop_state."""
        return {}

    def on_neighbor_change(self) -> None:
        """Called when predecessor or successor changes (reordering or failure)."""

    def on_reset(self) -> None:
        """Reset all internal state (agent returning from failure)."""


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _safe(x: float, default: float = 0.0) -> float:
    """Return x if finite, else default."""
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _get(state: dict | None, key: str, default: float = 0.0) -> float:
    if state is None:
        return default
    return _safe(state.get(key, default), default)


# ---------------------------------------------------------------------------
# Mecanismo 0: BaselineLayer — current controller, no propagation
# ---------------------------------------------------------------------------

class BaselineLayer(PropagationLayer):
    """Represents the existing controller with zero modification.

    get_signal() always returns 0.0, so prop_du = K_PROP * 0.0 = 0.0
    regardless of K_PROP.  This is the reference baseline for comparison.
    """

    def __init__(self, params: dict):
        pass

    def update(self, e_tau, dt, pred_state, succ_state):
        pass

    def get_signal(self) -> float:
        return 0.0

    def get_broadcast_state(self) -> dict:
        return {}


# ---------------------------------------------------------------------------
# Mecanismo 1: AdvectionLayer — Advecção-Difusão Amortecida Bidirecional
# ---------------------------------------------------------------------------

class AdvectionLayer(PropagationLayer):
    """Bidirectional advection-diffusion field with decay.

    Two scalar fields travel in opposite directions around the ring:
      q_fwd: pred → self → succ  (forward)
      q_bwd: succ → self → pred  (backward)

    Update (Euler):
      lap_fwd = q_fwd_succ - 2*q_fwd + q_fwd_pred
      dq_fwd/dt = -alpha*q_fwd + c*(q_fwd_succ - q_fwd) + D*lap_fwd + injection
      (symmetric for q_bwd with reversed advection direction)

    CFL stability requires c*dt ≤ 1.  Default c=5, dt=0.01 → c*dt=0.05 ✓.
    """

    DEFAULT_PARAMS = {"alpha": 1.0, "c": 5.0, "D": 0.1, "S": 2.0, "kappa": 3.0}

    def __init__(self, params: dict):
        p = {**self.DEFAULT_PARAMS, **(params or {})}
        self.alpha = float(p["alpha"])
        self.c = float(p["c"])
        self.D = float(p["D"])
        self.S = float(p["S"])
        self.kappa = float(p["kappa"])
        self.q_fwd: float = 0.0
        self.q_bwd: float = 0.0
        # Cached neighbor values for get_neighbor_signal()
        self._q_fwd_from_pred: float = 0.0  # q_fwd received from predecessor
        self._q_bwd_from_succ: float = 0.0  # q_bwd received from successor

    def update(self, e_tau, dt, pred_state, succ_state):
        dt = _safe(dt, 0.01)
        e = _safe(e_tau)

        q_fwd_pred = _get(pred_state, "q_fwd")
        q_fwd_succ = _get(succ_state, "q_fwd")
        q_bwd_pred = _get(pred_state, "q_bwd")
        q_bwd_succ = _get(succ_state, "q_bwd")

        # Cache the directional neighbor values before updating local state
        self._q_fwd_from_pred = q_fwd_pred
        self._q_bwd_from_succ = q_bwd_succ

        injection = self.S * math.tanh(self.kappa * e)

        lap_fwd = q_fwd_succ - 2.0 * self.q_fwd + q_fwd_pred
        dq_fwd = (
            -self.alpha * self.q_fwd
            # Forward wave travels pred -> self -> succ, so the upwind
            # linear advection term must use the predecessor as upstream state.
            + self.c * (q_fwd_pred - self.q_fwd)
            + self.D * lap_fwd
            + injection
        )

        lap_bwd = q_bwd_succ - 2.0 * self.q_bwd + q_bwd_pred
        dq_bwd = (
            -self.alpha * self.q_bwd
            # Backward wave travels succ -> self -> pred, so the upwind
            # linear advection term must use the successor as upstream state.
            + self.c * (q_bwd_succ - self.q_bwd)
            + self.D * lap_bwd
            + injection
        )

        self.q_fwd = _safe(self.q_fwd + dt * dq_fwd)
        self.q_bwd = _safe(self.q_bwd + dt * dq_bwd)

    def get_signal(self) -> float:
        return self.q_fwd + self.q_bwd

    def get_neighbor_signal(self) -> float:
        # Forward wave (from pred) + backward wave (from succ): purely from neighbors
        return self._q_fwd_from_pred + self._q_bwd_from_succ

    def get_broadcast_state(self) -> dict:
        return {"q_fwd": self.q_fwd, "q_bwd": self.q_bwd}

    def on_neighbor_change(self) -> None:
        self.q_fwd = 0.0
        self.q_bwd = 0.0
        self._q_fwd_from_pred = 0.0
        self._q_bwd_from_succ = 0.0

    def on_reset(self) -> None:
        self.q_fwd = 0.0
        self.q_bwd = 0.0
        self._q_fwd_from_pred = 0.0
        self._q_bwd_from_succ = 0.0


# ---------------------------------------------------------------------------
# Mecanismo 2: WaveLayer — Onda de Segunda Ordem
# ---------------------------------------------------------------------------

class WaveLayer(PropagationLayer):
    """Second-order wave equation on the ring.

    State: q (amplitude), p (momentum / time derivative).

    dp/dt = c_sq * lap(q) - gamma * p - alpha_q * q + injection
    dq/dt = p

    alpha_q > 0 acts as a restoring term (spring constant), ensuring that
    the DC mode (uniform q, zero Laplacian) also decays to zero.  Without it
    a spatially uniform initial condition would persist forever.
    Critical damping of the DC mode requires gamma^2 >= 4*alpha_q.
    Default alpha_q=1.0, gamma=2.0 → critically damped at ω=0.

    signal = Kq * q + Kp * p  (optional anticipatory component via Kp)
    """

    DEFAULT_PARAMS = {
        "c_sq": 25.0, "gamma": 2.0, "alpha_q": 1.0, "S": 2.0, "kappa": 3.0,
        "Kq": 1.0, "Kp": 0.3,
    }

    def __init__(self, params: dict):
        p = {**self.DEFAULT_PARAMS, **(params or {})}
        self.c_sq = float(p["c_sq"])
        self.gamma = float(p["gamma"])
        self.alpha_q = float(p["alpha_q"])
        self.S = float(p["S"])
        self.kappa = float(p["kappa"])
        self.Kq = float(p["Kq"])
        self.Kp = float(p["Kp"])
        self.q: float = 0.0
        self.p: float = 0.0
        # Cached neighbor q values for get_neighbor_signal()
        self._q_pred: float = 0.0
        self._q_succ: float = 0.0

    def update(self, e_tau, dt, pred_state, succ_state):
        dt = _safe(dt, 0.01)
        e = _safe(e_tau)

        q_pred = _get(pred_state, "q")
        q_succ = _get(succ_state, "q")

        # Cache neighbor values before updating local state
        self._q_pred = q_pred
        self._q_succ = q_succ

        injection = self.S * math.tanh(self.kappa * e)
        lap_q = q_succ - 2.0 * self.q + q_pred

        dp = self.c_sq * lap_q - self.gamma * self.p - self.alpha_q * self.q + injection
        self.p = _safe(self.p + dt * dp)
        self.q = _safe(self.q + dt * self.p)

    def get_signal(self) -> float:
        return self.Kq * self.q + self.Kp * self.p

    def get_neighbor_signal(self) -> float:
        # Mean of neighbor q fields weighted by Kq (excludes local injection)
        return self.Kq * (self._q_pred + self._q_succ) / 2.0

    def get_broadcast_state(self) -> dict:
        return {"q": self.q}

    def on_neighbor_change(self) -> None:
        self.q = 0.0
        self.p = 0.0
        self._q_pred = 0.0
        self._q_succ = 0.0

    def on_reset(self) -> None:
        self.q = 0.0
        self.p = 0.0
        self._q_pred = 0.0
        self._q_succ = 0.0


# ---------------------------------------------------------------------------
# Mecanismo 3: ExcitableLayer — Meio Excitável (FitzHugh-Nagumo)
# ---------------------------------------------------------------------------

def _fhn_equilibrium(a: float, b: float):
    """Newton iteration to find the FHN rest point (v_eq, w_eq).

    At equilibrium: v - v^3/3 - w = 0  and  v + a - b*w = 0.
    Substituting w = (v+a)/b gives f(v) = v - v^3/3 - (v+a)/b = 0.
    """
    v = -1.2  # initial guess (typical rest value for a=0.7, b=0.8)
    for _ in range(60):
        w = (v + a) / b
        fv = v - v ** 3 / 3.0 - w
        dfv = 1.0 - v ** 2 - 1.0 / b
        if abs(dfv) < 1e-14:
            break
        step = fv / dfv
        v -= step
        if abs(step) < 1e-12:
            break
    w = (v + a) / b
    return float(v), float(w)


class ExcitableLayer(PropagationLayer):
    """FitzHugh-Nagumo excitable medium on the ring.

    Fast variable v (excitation), slow variable w (recovery/refractory).
    Internal state is stored as DEVIATIONS from the rest point (v_eq, w_eq):
      self.v = V_abs - v_eq,  self.w = W_abs - w_eq

    This ensures (self.v, self.w) = (0, 0) is the true equilibrium, so
    on_reset() and the stability test both work with get_signal() = 0.

    dV/dt = (1/epsilon)*(V - V^3/3 - W) + Dv*lap(V) + I_ext
    dW/dt = epsilon*(V + a - b*W)
    where V = self.v + v_eq, W = self.w + w_eq.

    Neighbors broadcast their deviation self.v under key "v", so the
    Laplacian lap(V) = lap(self.v) (v_eq is uniform, cancels out).

    4 internal substeps per control tick for numerical stability with
    epsilon=0.08 (1/epsilon = 12.5 is stiff at dt=0.01).
    """

    DEFAULT_PARAMS = {
        "epsilon": 0.08, "a": 0.7, "b": 0.8, "Dv": 30.0,
        "theta": 0.3, "S": 3.0, "kappa_out": 2.0,
    }
    _N_SUBSTEPS = 4
    _V_CLAMP = 3.0
    _W_CLAMP = 3.0

    def __init__(self, params: dict):
        p = {**self.DEFAULT_PARAMS, **(params or {})}
        self.epsilon = float(p["epsilon"])
        self.a = float(p["a"])
        self.b = float(p["b"])
        self.Dv = float(p["Dv"])
        self.theta = float(p["theta"])
        self.S = float(p["S"])
        self.kappa_out = float(p["kappa_out"])
        self.v_eq, self.w_eq = _fhn_equilibrium(self.a, self.b)
        self.v: float = 0.0  # deviation from v_eq
        self.w: float = 0.0  # deviation from w_eq
        # Cached neighbor v deviations for get_neighbor_signal()
        self._v_pred: float = 0.0
        self._v_succ: float = 0.0

    def update(self, e_tau, dt, pred_state, succ_state):
        dt = _safe(dt, 0.01)
        e = _safe(e_tau)

        # Neighbors also store deviations — lap(V_abs) = lap(v_dev) since v_eq uniform
        v_pred = _get(pred_state, "v")
        v_succ = _get(succ_state, "v")

        # Cache neighbor values before updating local state
        self._v_pred = v_pred
        self._v_succ = v_succ

        lap_v = v_succ - 2.0 * self.v + v_pred

        abs_e = abs(e)
        if abs_e > self.theta:
            I_ext = self.S * math.copysign(1.0, e) * (abs_e - self.theta)
        else:
            I_ext = 0.0

        # Work with absolute values for FHN dynamics
        V_abs = self.v + self.v_eq
        W_abs = self.w + self.w_eq

        sub_dt = dt / self._N_SUBSTEPS
        for _ in range(self._N_SUBSTEPS):
            dv = (1.0 / self.epsilon) * (V_abs - V_abs ** 3 / 3.0 - W_abs) + self.Dv * lap_v + I_ext
            dw = self.epsilon * (V_abs + self.a - self.b * W_abs)
            V_abs = max(-self._V_CLAMP, min(self._V_CLAMP, _safe(V_abs + sub_dt * dv)))
            W_abs = max(-self._W_CLAMP, min(self._W_CLAMP, _safe(W_abs + sub_dt * dw)))

        self.v = V_abs - self.v_eq
        self.w = W_abs - self.w_eq

    def get_signal(self) -> float:
        return math.tanh(self.kappa_out * self.v)

    def get_neighbor_signal(self) -> float:
        # Mean excitation received from both neighbors (excludes local injection)
        return math.tanh(self.kappa_out * (self._v_pred + self._v_succ) / 2.0)

    def get_broadcast_state(self) -> dict:
        return {"v": self.v}

    def on_neighbor_change(self) -> None:
        self.v = 0.0
        self.w = 0.0
        self._v_pred = 0.0
        self._v_succ = 0.0

    def on_reset(self) -> None:
        self.v = 0.0
        self.w = 0.0
        self._v_pred = 0.0
        self._v_succ = 0.0


# ---------------------------------------------------------------------------
# Mecanismo 4: KdVLayer — KdV Discreto (Soliton-Inspired)
# ---------------------------------------------------------------------------

class KdVLayer(PropagationLayer):
    """Discrete KdV-inspired field with nonlinear steepening and dispersion.

    State: q (field amplitude), Lq (local Laplacian for broadcast).

    dq/dt = -alpha_nl * q * grad(q)    [nonlinear steepening]
            - beta_disp * d3q/dx3      [dispersion via relayed Laplacians]
            - gamma * q                [damping]
            + S * tanh(e_tau)          [level injection from gap error]

    3rd derivative approximated as (Lq_succ - Lq_pred) / 2, using the
    Laplacians broadcast by neighbors (1-tick delay — negligible at dt=0.01).
    """

    DEFAULT_PARAMS = {
        "alpha_nl": 2.0, "beta_disp": 0.5, "gamma": 1.5,
        "S": 5.0, "q_max": 5.0,
    }

    def __init__(self, params: dict):
        p = {**self.DEFAULT_PARAMS, **(params or {})}
        self.alpha_nl = float(p["alpha_nl"])
        self.beta_disp = float(p["beta_disp"])
        self.gamma = float(p["gamma"])
        self.S = float(p["S"])
        self.q_max = float(p["q_max"])
        self.q: float = 0.0
        self.Lq: float = 0.0
        # Cached neighbor q values for get_neighbor_signal()
        self._q_pred: float = 0.0
        self._q_succ: float = 0.0

    def update(self, e_tau, dt, pred_state, succ_state):
        dt = _safe(dt, 0.01)
        e = _safe(e_tau)

        q_pred = _get(pred_state, "q")
        q_succ = _get(succ_state, "q")
        Lq_pred = _get(pred_state, "Lq")
        Lq_succ = _get(succ_state, "Lq")

        # Cache neighbor values before updating local state
        self._q_pred = q_pred
        self._q_succ = q_succ

        # Compute Lq BEFORE updating q (so broadcast value reflects current state)
        self.Lq = q_succ - 2.0 * self.q + q_pred

        grad_q = (q_succ - q_pred) / 2.0
        disp = (Lq_succ - Lq_pred) / 2.0

        dq = (
            -self.alpha_nl * self.q * grad_q
            - self.beta_disp * disp
            - self.gamma * self.q
            + self.S * math.tanh(e)
        )

        q_new = _safe(self.q + dt * dq)
        self.q = max(-self.q_max, min(self.q_max, q_new))

    def get_signal(self) -> float:
        return self.q

    def get_neighbor_signal(self) -> float:
        # Mean of neighbor q fields: the propagating wave component from both sides
        return (self._q_pred + self._q_succ) / 2.0

    def get_broadcast_state(self) -> dict:
        return {"q": self.q, "Lq": self.Lq}

    def on_neighbor_change(self) -> None:
        self.q = 0.0
        self.Lq = 0.0
        self._q_pred = 0.0
        self._q_succ = 0.0

    def on_reset(self) -> None:
        self.q = 0.0
        self.Lq = 0.0
        self._q_pred = 0.0
        self._q_succ = 0.0


# ---------------------------------------------------------------------------
# Mecanismo 5: AlarmLayer — Alarmes Discretos com TTL
# ---------------------------------------------------------------------------

_INACTIVE_ALARM = {"active": False, "polarity": 0.0, "intensity": 0.0, "ttl": 0}


class AlarmLayer(PropagationLayer):
    """Discrete alarm pulses that hop along the ring with TTL and attenuation.

    When |e_tau| exceeds theta and the node is not refractory, it generates
    two alarms (forward and backward) with TTL = TTL_MAX.  Each subsequent
    node relays the alarm, decrementing TTL and multiplying intensity by lam.
    A node that generates an alarm enters refractory state for T_refract ticks.

    signal = sum of active alarm polarities weighted by intensity.
    """

    DEFAULT_PARAMS = {
        "theta": 0.4, "I_max": 1.0, "TTL_MAX": 20,
        "lam": 0.92, "T_refract": 15,
    }

    def __init__(self, params: dict):
        p = {**self.DEFAULT_PARAMS, **(params or {})}
        self.theta = float(p["theta"])
        self.I_max = float(p["I_max"])
        self.TTL_MAX = int(p["TTL_MAX"])
        self.lam = float(p["lam"])
        self.T_refract = int(p["T_refract"])
        self.alarm_fwd: dict = dict(_INACTIVE_ALARM)
        self.alarm_bwd: dict = dict(_INACTIVE_ALARM)
        self.refractory: int = 0
        self._relay_signal: float = 0.0  # signal from relayed alarms only

    def _age(self, alarm: dict) -> dict:
        """Decrement TTL by one tick; deactivate if expired."""
        if not alarm.get("active", False):
            return alarm
        ttl = int(alarm.get("ttl", 0)) - 1
        if ttl <= 0:
            return dict(_INACTIVE_ALARM)
        return {**alarm, "ttl": ttl}

    def update(self, e_tau, dt, pred_state, succ_state):
        e = _safe(e_tau)

        if self.refractory > 0:
            self.refractory -= 1

        abs_e = abs(e)

        # Compute relay signal from neighbor states regardless of self-generation
        relay_sig = 0.0
        pred_fwd = pred_state.get("alarm_fwd", _INACTIVE_ALARM) if pred_state else _INACTIVE_ALARM
        if pred_fwd.get("active", False) and pred_fwd.get("ttl", 0) > 0:
            relay_sig += float(pred_fwd.get("polarity", 0.0)) * self.lam * float(pred_fwd.get("intensity", 0.0))
        succ_bwd = succ_state.get("alarm_bwd", _INACTIVE_ALARM) if succ_state else _INACTIVE_ALARM
        if succ_bwd.get("active", False) and succ_bwd.get("ttl", 0) > 0:
            relay_sig += float(succ_bwd.get("polarity", 0.0)) * self.lam * float(succ_bwd.get("intensity", 0.0))
        self._relay_signal = relay_sig

        # 1. Generate alarm if above threshold and not refractory
        if abs_e > self.theta and self.refractory == 0:
            pol = math.copysign(1.0, e)
            inten = min(abs_e, self.I_max)
            self.alarm_fwd = {"active": True, "polarity": pol, "intensity": inten, "ttl": self.TTL_MAX}
            self.alarm_bwd = {"active": True, "polarity": pol, "intensity": inten, "ttl": self.TTL_MAX}
            self.refractory = self.T_refract
            return  # no relay in the same tick as generation

        # 2. Age own alarms (natural TTL decay for source node and relay nodes)
        self.alarm_fwd = self._age(self.alarm_fwd)
        self.alarm_bwd = self._age(self.alarm_bwd)

        # 3. Relay forward alarm from predecessor — overwrites only if pred has active alarm
        if pred_fwd.get("active", False) and pred_fwd.get("ttl", 0) > 0:
            self.alarm_fwd = {
                "active": True,
                "polarity": float(pred_fwd.get("polarity", 0.0)),
                "intensity": self.lam * float(pred_fwd.get("intensity", 0.0)),
                "ttl": int(pred_fwd.get("ttl", 0)) - 1,
            }

        # 4. Relay backward alarm from successor
        if succ_bwd.get("active", False) and succ_bwd.get("ttl", 0) > 0:
            self.alarm_bwd = {
                "active": True,
                "polarity": float(succ_bwd.get("polarity", 0.0)),
                "intensity": self.lam * float(succ_bwd.get("intensity", 0.0)),
                "ttl": int(succ_bwd.get("ttl", 0)) - 1,
            }

    def get_signal(self) -> float:
        sig = 0.0
        if self.alarm_fwd.get("active", False):
            sig += float(self.alarm_fwd.get("polarity", 0.0)) * float(self.alarm_fwd.get("intensity", 0.0))
        if self.alarm_bwd.get("active", False):
            sig += float(self.alarm_bwd.get("polarity", 0.0)) * float(self.alarm_bwd.get("intensity", 0.0))
        return sig

    def get_neighbor_signal(self) -> float:
        # Only the relay signal from neighbors (computed from neighbor states in update())
        return self._relay_signal

    def get_broadcast_state(self) -> dict:
        return {"alarm_fwd": self.alarm_fwd, "alarm_bwd": self.alarm_bwd}

    def on_neighbor_change(self) -> None:
        self.alarm_fwd = dict(_INACTIVE_ALARM)
        self.alarm_bwd = dict(_INACTIVE_ALARM)
        self._relay_signal = 0.0
        # keep refractory

    def on_reset(self) -> None:
        self.alarm_fwd = dict(_INACTIVE_ALARM)
        self.alarm_bwd = dict(_INACTIVE_ALARM)
        self.refractory = 0
        self._relay_signal = 0.0


# ---------------------------------------------------------------------------
# Mecanismo 6: BurgersLayer — Burgers Amortecido com Saturação
# ---------------------------------------------------------------------------

class BurgersLayer(PropagationLayer):
    """Damped Burgers equation with saturation, bidirectional.

    Two scalar fields (q_fwd, q_bwd) travel in opposite directions.
    Nonlinear self-advection: velocity proportional to sat(q).
    Viscous diffusion (nu) prevents shock singularities.

    sat(q) = q_max * tanh(q / q_max)  — smooth saturation of wave speed.
    Hard clamp |q| ≤ q_clamp as additional safety.
    """

    DEFAULT_PARAMS = {
        "alpha_nl": 3.0, "nu": 5.0, "gamma": 0.8,
        "q_max": 2.0, "q_clamp": 5.0, "S": 2.0, "kappa": 3.0,
    }

    def __init__(self, params: dict):
        p = {**self.DEFAULT_PARAMS, **(params or {})}
        self.alpha_nl = float(p["alpha_nl"])
        self.nu = float(p["nu"])
        self.gamma = float(p["gamma"])
        self.q_max = float(p["q_max"])
        self.q_clamp = float(p["q_clamp"])
        self.S = float(p["S"])
        self.kappa = float(p["kappa"])
        self.q_fwd: float = 0.0
        self.q_bwd: float = 0.0
        # Cached directional neighbor values for get_neighbor_signal()
        self._q_fwd_from_pred: float = 0.0
        self._q_bwd_from_succ: float = 0.0

    def _sat(self, q: float) -> float:
        if self.q_max <= 0.0:
            return q
        return self.q_max * math.tanh(q / self.q_max)

    def update(self, e_tau, dt, pred_state, succ_state):
        dt = _safe(dt, 0.01)
        e = _safe(e_tau)

        q_fwd_pred = _get(pred_state, "q_fwd")
        q_fwd_succ = _get(succ_state, "q_fwd")
        q_bwd_pred = _get(pred_state, "q_bwd")
        q_bwd_succ = _get(succ_state, "q_bwd")

        # Cache directional neighbor values before updating local state
        self._q_fwd_from_pred = q_fwd_pred
        self._q_bwd_from_succ = q_bwd_succ

        injection = self.S * math.tanh(self.kappa * e)

        # Forward field: travels pred → succ  (upwind: q_fwd - q_fwd_pred)
        lap_fwd = q_fwd_succ - 2.0 * self.q_fwd + q_fwd_pred
        dq_fwd = (
            -self.gamma * self.q_fwd
            - self.alpha_nl * self._sat(self.q_fwd) * (self.q_fwd - q_fwd_pred)
            + self.nu * lap_fwd
            + injection
        )

        # Backward field: travels succ → pred  (upwind: q_bwd_succ - q_bwd)
        lap_bwd = q_bwd_succ - 2.0 * self.q_bwd + q_bwd_pred
        dq_bwd = (
            -self.gamma * self.q_bwd
            + self.alpha_nl * self._sat(self.q_bwd) * (q_bwd_succ - self.q_bwd)
            + self.nu * lap_bwd
            + injection
        )

        q_fwd_new = _safe(self.q_fwd + dt * dq_fwd)
        q_bwd_new = _safe(self.q_bwd + dt * dq_bwd)
        self.q_fwd = max(-self.q_clamp, min(self.q_clamp, q_fwd_new))
        self.q_bwd = max(-self.q_clamp, min(self.q_clamp, q_bwd_new))

    def get_signal(self) -> float:
        return self.q_fwd + self.q_bwd

    def get_neighbor_signal(self) -> float:
        # Forward wave (from pred) + backward wave (from succ): purely from neighbors
        return self._q_fwd_from_pred + self._q_bwd_from_succ

    def get_broadcast_state(self) -> dict:
        return {"q_fwd": self.q_fwd, "q_bwd": self.q_bwd}

    def on_neighbor_change(self) -> None:
        self.q_fwd = 0.0
        self.q_bwd = 0.0
        self._q_fwd_from_pred = 0.0
        self._q_bwd_from_succ = 0.0

    def on_reset(self) -> None:
        self.q_fwd = 0.0
        self.q_bwd = 0.0
        self._q_fwd_from_pred = 0.0
        self._q_bwd_from_succ = 0.0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type] = {
    "baseline":  BaselineLayer,
    "advection": AdvectionLayer,
    "wave":      WaveLayer,
    "excitable": ExcitableLayer,
    "kdv":       KdVLayer,
    "alarm":     AlarmLayer,
    "burgers":   BurgersLayer,
}


def create_propagation_layer(method: str, params: dict | None = None) -> PropagationLayer:
    """Instantiate the propagation layer for the given method name.

    Falls back to BaselineLayer for unknown method names.
    """
    cls = _REGISTRY.get(method, BaselineLayer)
    return cls(params or {})
