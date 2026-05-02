"""Shared controller adapters for agent and target protocols.

This module provides a minimal unified API:

- ``reset()`` clears controller memory
- ``update(measurement=..., dt=...)`` returns the controller output

Where it makes technical sense, the implementation delegates to ``simple-pid``.
The tangential spacing controller keeps the project's current first-order
internal-state dynamics unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional

try:
    from simple_pid import PID as _PIDBackend
except Exception:  # pragma: no cover
    class _PIDBackend:
        """Small compatibility backend matching the subset of simple-pid used here."""

        def __init__(
            self,
            Kp: float,
            Ki: float,
            Kd: float,
            *,
            setpoint: float = 0.0,
            sample_time=None,
            output_limits=(None, None),
            error_map: Optional[Callable[[float], float]] = None,
        ):
            self.Kp = float(Kp)
            self.Ki = float(Ki)
            self.Kd = float(Kd)
            self.setpoint = float(setpoint)
            self.sample_time = sample_time
            self.output_limits = output_limits
            self.error_map = error_map
            self._integral = 0.0
            self._last_input: Optional[float] = None
            self._last_output = 0.0

        @staticmethod
        def _clamp(value: float, limits) -> float:
            low, high = limits
            if low is not None and value < low:
                value = float(low)
            if high is not None and value > high:
                value = float(high)
            return float(value)

        def reset(self) -> None:
            self._integral = 0.0
            self._last_input = None
            self._last_output = 0.0

        def __call__(self, input_, dt=None):
            inp = float(input_)
            try:
                dt_f = float(dt)
            except Exception:
                dt_f = 0.0
            if not (math.isfinite(dt_f) and dt_f > 0.0):
                dt_f = 0.0

            error = float(self.setpoint - inp)
            if self.error_map is not None:
                error = float(self.error_map(error))

            if self.Ki != 0.0 and dt_f > 0.0:
                self._integral = self._clamp(self._integral + (self.Ki * error * dt_f), self.output_limits)

            d_input = 0.0
            if self._last_input is not None and dt_f > 0.0:
                d_input = (inp - self._last_input) / dt_f

            output = (self.Kp * error) + self._integral - (self.Kd * d_input)
            output = self._clamp(output, self.output_limits)

            self._last_input = inp
            self._last_output = float(output)
            return float(output)


def wrap_to_pi(angle: float) -> float:
    """Wrap angle to (-pi, pi]."""
    two_pi = 2.0 * math.pi
    wrapped = (float(angle) + math.pi) % two_pi
    wrapped -= math.pi
    return float(wrapped)


class BaseController:
    """Small shared interface for all controllers used in the protocols."""

    def reset(self) -> None:
        raise NotImplementedError

    def update(self, *, measurement: float, dt: float):
        raise NotImplementedError


class RadialDistanceController(BaseController):
    """PD controller for orbit-radius regulation."""

    def __init__(self, *, kp: float, kd: float, radius_setpoint: float):
        self._pid = _PIDBackend(
            kp,
            0.0,
            kd,
            setpoint=float(radius_setpoint),
            sample_time=None,
        )

    def reset(self) -> None:
        self._pid.reset()

    def update(self, *, measurement: float, dt: float) -> float:
        value = self._pid(float(measurement), dt=float(dt))
        return float(value) if math.isfinite(value) else 0.0


class WrappedAnglePDController(BaseController):
    """PD controller over wrapped angular error, using simple-pid for P and D terms."""

    def __init__(self, *, kp: float, kd: float, max_abs_output: float):
        limits = (None, None)
        if math.isfinite(float(max_abs_output)) and float(max_abs_output) > 0.0:
            bound = abs(float(max_abs_output))
            limits = (-bound, bound)

        self._pid = _PIDBackend(
            kp,
            0.0,
            kd,
            setpoint=0.0,
            sample_time=None,
            output_limits=limits,
            error_map=wrap_to_pi,
        )
        self._last_wrapped_error: Optional[float] = None
        self._unwrapped_error: float = 0.0

    def reset(self) -> None:
        self._pid.reset()
        self._last_wrapped_error = None
        self._unwrapped_error = 0.0

    def update(self, *, measurement: float, dt: float) -> float:
        err_wrapped = wrap_to_pi(float(measurement))
        if self._last_wrapped_error is None:
            self._unwrapped_error = err_wrapped
        else:
            delta = wrap_to_pi(err_wrapped - self._last_wrapped_error)
            self._unwrapped_error += delta

        self._last_wrapped_error = err_wrapped

        # Feed the negated unwrapped error as the "measurement":
        # error = setpoint - input = 0 - (-err) = err.
        value = self._pid(-self._unwrapped_error, dt=float(dt))
        return float(value) if math.isfinite(value) else 0.0


@dataclass(frozen=True)
class TangentialControlOutput:
    u: float           # total tangential drive (u_local + u_prop after composition)
    u_local: float     # local error channel state
    u_prop: float      # propagated error channel state
    delta_u: float     # total increment applied this tick
    du_local: float    # du for the local channel
    du_prop: float     # du for the propagated channel
    du_damp_local: float
    du_damp_prop: float
    du_from_error: float
    du_from_prop_signal: float
    # Legacy aliases kept for backward-compatible telemetry reads
    @property
    def du(self) -> float:
        return self.du_local + self.du_prop

    @property
    def du_damp(self) -> float:
        return self.du_damp_local + self.du_damp_prop

    @property
    def du_from_prop(self) -> float:
        return self.du_from_prop_signal


VALID_COMPOSITION_MODES = ("blend", "sum")


class TangentialSpacingController(BaseController):
    """Two-channel tangential spacing controller.

    Maintains separate state variables for the local error response (u_local)
    and the propagated neighbor response (u_prop). The composition into the
    final u is selected by ``composition_mode``:

    - "blend": cooperative sum when channels agree in sign; smooth tanh
      dominance blend (width = ``conflict_blend_width``) when they conflict.
      The dominant channel is preserved and the smaller one is attenuated —
      defensive against unreliable propagation.
    - "sum": pure addition u = u_local + u_prop in all regimes. The propagation
      channel always contributes; aligned with the soliton-inspired premise
      that neighbor signals carry useful information.

    No saturation is applied to u at the controller level in either mode.
    Velocity-level saturation in the mobility handler is the single source of
    truth for actuator limits.

    The two-channel design prevents the propagated channel from double-counting
    the local error (which is injected independently via k_e_tau * e_tau).
    """

    def __init__(
        self,
        *,
        beta_u: float,
        k_e_tau: float,
        initial_u: float = 0.0,
        beta_u_local: float | None = None,
        beta_u_prop: float | None = None,
        conflict_blend_width: float = 0.0,
        composition_mode: str = "blend",
    ):
        # beta_u is kept as the legacy fallback; explicit per-channel values take priority
        self.beta_u_local = float(beta_u_local if beta_u_local is not None else beta_u)
        self.beta_u_prop = float(beta_u_prop if beta_u_prop is not None else beta_u)
        self.k_e_tau = float(k_e_tau)
        width = float(conflict_blend_width) if math.isfinite(float(conflict_blend_width)) else 0.0
        self.conflict_blend_width = abs(width)
        if composition_mode not in VALID_COMPOSITION_MODES:
            raise ValueError(
                f"composition_mode must be one of {VALID_COMPOSITION_MODES}, "
                f"got {composition_mode!r}"
            )
        self.composition_mode = composition_mode
        init = float(initial_u) if math.isfinite(float(initial_u)) else 0.0
        self.u_local: float = init
        self.u_prop: float = 0.0

    @property
    def u(self) -> float:
        """Total tangential drive after channel composition."""
        return self._compose(self.u_local, self.u_prop)

    def _compose(self, u_local: float, u_prop: float) -> float:
        """Combine the two channel states into u according to composition_mode."""
        if self.composition_mode == "sum":
            return u_local + u_prop

        # "blend": cooperative when channels agree; smooth dominance on conflict.
        if u_local * u_prop >= 0.0:
            return u_local + u_prop
        if self.conflict_blend_width <= 0.0:
            return u_local if abs(u_local) >= abs(u_prop) else u_prop

        diff = abs(u_local) - abs(u_prop)
        w_local = 0.5 * (1.0 + math.tanh(diff / self.conflict_blend_width))
        return (w_local * u_local) + ((1.0 - w_local) * u_prop)

    def reset(self) -> None:
        self.u_local = 0.0
        self.u_prop = 0.0

    def update(
        self,
        *,
        measurement: float,
        dt: float,
        prop_signal: float = 0.0,
        k_prop: float = 0.0,
        # Legacy parameter name accepted for backward compatibility
        prop_du: float | None = None,
    ) -> TangentialControlOutput:
        try:
            dt_f = float(dt)
        except Exception:
            dt_f = 0.0
        if not (math.isfinite(dt_f) and dt_f > 0.0):
            dt_f = 0.0

        e_tau = float(measurement) if math.isfinite(float(measurement)) else 0.0
        u_local_prev = float(self.u_local) if math.isfinite(float(self.u_local)) else 0.0
        u_prop_prev = float(self.u_prop) if math.isfinite(float(self.u_prop)) else 0.0

        # --- Local channel ---
        du_damp_local = -self.beta_u_local * u_local_prev
        du_from_error = self.k_e_tau * e_tau
        du_local = du_damp_local + du_from_error
        u_local_next = u_local_prev + dt_f * du_local

        # --- Propagated channel ---
        # prop_signal is the pure neighbor signal (no self-injection).
        # k_prop is applied here so the controller owns the scaling.
        # Legacy prop_du path: if caller passes pre-scaled prop_du, use it as the
        # neighbor signal term directly (k_prop is ignored in that case).
        if prop_du is not None:
            effective_prop_drive = float(prop_du) if math.isfinite(float(prop_du)) else 0.0
        else:
            sig = float(prop_signal) if math.isfinite(float(prop_signal)) else 0.0
            kp = float(k_prop) if math.isfinite(float(k_prop)) else 0.0
            effective_prop_drive = kp * sig

        du_damp_prop = -self.beta_u_prop * u_prop_prev
        du_from_prop_signal = effective_prop_drive
        du_prop = du_damp_prop + du_from_prop_signal
        u_prop_next = u_prop_prev + dt_f * du_prop

        # Guard against non-finite states
        if not math.isfinite(u_local_next):
            u_local_next = 0.0
        if not math.isfinite(u_prop_next):
            u_prop_next = 0.0

        self.u_local = u_local_next
        self.u_prop = u_prop_next

        u_total = self._compose(u_local_next, u_prop_next)
        delta_u = u_total - self._compose(u_local_prev, u_prop_prev)

        return TangentialControlOutput(
            u=float(u_total),
            u_local=float(u_local_next),
            u_prop=float(u_prop_next),
            delta_u=float(delta_u),
            du_local=float(du_local),
            du_prop=float(du_prop),
            du_damp_local=float(du_damp_local),
            du_damp_prop=float(du_damp_prop),
            du_from_error=float(du_from_error),
            du_from_prop_signal=float(du_from_prop_signal),
        )
