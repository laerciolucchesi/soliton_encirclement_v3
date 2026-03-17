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
    u: float
    delta_u: float
    du: float
    du_damp: float
    du_from_error: float


class TangentialSpacingController(BaseController):
    """Current tangential first-order state dynamics kept unchanged."""

    def __init__(self, *, beta_u: float, k_e_tau: float, initial_u: float = 0.0):
        self.beta_u = float(beta_u)
        self.k_e_tau = float(k_e_tau)
        self.u = float(initial_u) if math.isfinite(float(initial_u)) else 0.0

    def reset(self) -> None:
        self.u = 0.0

    def update(self, *, measurement: float, dt: float) -> TangentialControlOutput:
        try:
            dt_f = float(dt)
        except Exception:
            dt_f = 0.0
        if not (math.isfinite(dt_f) and dt_f > 0.0):
            dt_f = 0.0

        e_tau = float(measurement) if math.isfinite(float(measurement)) else 0.0
        u_prev = float(self.u) if math.isfinite(float(self.u)) else 0.0

        du_damp = float(-self.beta_u * u_prev)
        du_from_error = float(self.k_e_tau * e_tau)
        du = float(du_damp + du_from_error)
        u_next = float(u_prev + (dt_f * du))

        if not (math.isfinite(u_next) and math.isfinite(du)):
            self.u = 0.0
            return TangentialControlOutput(
                u=0.0,
                delta_u=0.0,
                du=0.0,
                du_damp=0.0,
                du_from_error=0.0,
            )

        self.u = u_next
        return TangentialControlOutput(
            u=float(u_next),
            delta_u=float(dt_f * du),
            du=float(du),
            du_damp=float(du_damp),
            du_from_error=float(du_from_error),
        )
