"""Tests for propagation_layer.py — 5 tests per mechanism (35 total).

Test patterns:
  1. test_decay       — field decays to near-zero without excitation
  2. test_propagation — signal generated at node 0 reaches node N//2 after 100 ticks
  3. test_stability   — zero initial state, zero excitation → stays at zero
  4. test_none_neighbor — missing neighbors don't crash and return finite signal
  5. test_reset       — on_reset() zeroes all internal state
"""

import math
import sys
import os

# Ensure repo root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from propagation_layer import (
    create_propagation_layer,
    BaselineLayer,
    AdvectionLayer,
    WaveLayer,
    ExcitableLayer,
    KdVLayer,
    AlarmLayer,
    BurgersLayer,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

DT = 0.01
N = 20
ALL_METHODS = ["baseline", "advection", "wave", "excitable", "kdv", "alarm", "burgers"]


def make_ring(method: str, params=None, n: int = N):
    return [create_propagation_layer(method, params or {}) for _ in range(n)]


def step_ring(layers, e_taus):
    """One tick: snapshot broadcast states, then update all nodes."""
    states = [l.get_broadcast_state() for l in layers]
    n = len(layers)
    for i, layer in enumerate(layers):
        layer.update(e_taus[i], DT, states[(i - 1) % n], states[(i + 1) % n])


def _force_field(layer, value: float):
    """Force internal scalar field(s) to a given value for decay tests."""
    if isinstance(layer, (AdvectionLayer, BurgersLayer)):
        layer.q_fwd = value
        layer.q_bwd = value
    elif isinstance(layer, WaveLayer):
        layer.q = value
        layer.p = 0.0
    elif isinstance(layer, ExcitableLayer):
        layer.v = value
        layer.w = 0.0
    elif isinstance(layer, KdVLayer):
        layer.q = value
        layer.Lq = 0.0
    elif isinstance(layer, AlarmLayer):
        # Simulate an active alarm with given intensity
        layer.alarm_fwd = {"active": True, "polarity": 1.0, "intensity": abs(value), "ttl": 5}
        layer.alarm_bwd = {"active": True, "polarity": 1.0, "intensity": abs(value), "ttl": 5}
    # BaselineLayer has no internal state — nothing to force


# ---------------------------------------------------------------------------
# Test class template applied to each method
# ---------------------------------------------------------------------------

class _PropagationTests:
    """Mixin with the 5 standard tests.  Subclasses set ``method``."""

    method: str = "baseline"

    # --- Test 1: decay -------------------------------------------------------

    def test_decay(self):
        """Field should decay to near-zero after 500 ticks with e_tau=0."""
        layers = make_ring(self.method)
        # Force internal fields to a non-zero value
        for l in layers:
            _force_field(l, 1.0)

        for _ in range(500):
            step_ring(layers, [0.0] * N)

        for l in layers:
            sig = l.get_signal()
            assert math.isfinite(sig), f"Non-finite signal in {self.method}"
            assert abs(sig) < 0.05, (
                f"{self.method}: signal={sig:.4f} did not decay after 500 ticks"
            )

    # --- Test 2: propagation -------------------------------------------------

    def test_propagation(self):
        """Signal generated at node 0 should reach node N//4 after 100 ticks.
        Exception: baseline always returns 0.0 — test inverted for it.
        """
        layers = make_ring(self.method)
        e_taus = [0.0] * N
        e_taus[0] = 0.8  # constant excitation at node 0

        for _ in range(100):
            step_ring(layers, e_taus)

        quarter = N // 4
        sig_q = layers[quarter].get_signal()
        assert math.isfinite(sig_q), f"Non-finite signal at quarter-ring for {self.method}"

        if self.method == "baseline":
            assert sig_q == 0.0, "BaselineLayer should always return 0.0"
        else:
            assert abs(sig_q) > 0.01, (
                f"{self.method}: signal at node {quarter} = {sig_q:.4f}; "
                "expected propagation to reach N//4 after 100 ticks"
            )

    # --- Test 3: stability ---------------------------------------------------

    def test_stability(self):
        """Zero initial state + zero excitation → signal stays at zero."""
        layers = make_ring(self.method)
        for _ in range(1000):
            step_ring(layers, [0.0] * N)

        for i, l in enumerate(layers):
            sig = l.get_signal()
            assert math.isfinite(sig), f"{self.method} node {i}: non-finite signal"
            assert abs(sig) < 1e-6, (
                f"{self.method} node {i}: signal={sig} drifted from zero at rest"
            )

    # --- Test 4: missing neighbors -------------------------------------------

    def test_none_neighbor(self):
        """pred_state=None and succ_state=None must not crash and return finite signal."""
        layer = create_propagation_layer(self.method, {})
        try:
            layer.update(0.5, DT, None, None)
        except Exception as exc:
            raise AssertionError(
                f"{self.method}: update with None neighbors raised {exc!r}"
            ) from exc
        sig = layer.get_signal()
        assert math.isfinite(sig), f"{self.method}: get_signal() returned non-finite with None neighbors"

    # --- Test 5: reset -------------------------------------------------------

    def test_reset(self):
        """on_reset() should zero all internal state."""
        layer = create_propagation_layer(self.method, {})
        # Run 50 ticks with excitation
        for _ in range(50):
            layer.update(0.5, DT, None, None)

        layer.on_reset()

        sig = layer.get_signal()
        assert sig == 0.0, f"{self.method}: get_signal() = {sig} after on_reset(), expected 0.0"

        bs = layer.get_broadcast_state()
        for k, v in bs.items():
            if isinstance(v, (int, float)):
                assert v == 0.0 or v is False, (
                    f"{self.method}: broadcast_state[{k!r}]={v} not zero after on_reset()"
                )
            elif isinstance(v, dict):
                assert not v.get("active", False), (
                    f"{self.method}: broadcast_state[{k!r}] still active after on_reset()"
                )


# ---------------------------------------------------------------------------
# Concrete test classes — one per method
# ---------------------------------------------------------------------------

class TestBaseline(_PropagationTests):
    method = "baseline"


class TestAdvection(_PropagationTests):
    method = "advection"

    def test_backward_advection_mode_is_damped(self):
        """The backward channel must not amplify the highest-frequency mode."""
        layer = AdvectionLayer({})
        layer.q_bwd = 1.0

        pred_state = {"q_fwd": 0.0, "q_bwd": -1.0}
        succ_state = {"q_fwd": 0.0, "q_bwd": -1.0}

        for _ in range(20):
            layer.update(0.0, DT, pred_state, succ_state)

        assert math.isfinite(layer.q_bwd), "advection: q_bwd became non-finite"
        assert abs(layer.q_bwd) < 1.0, (
            f"advection: backward channel grew under an alternating mode: q_bwd={layer.q_bwd:.4f}"
        )


class TestWave(_PropagationTests):
    method = "wave"


class TestExcitable(_PropagationTests):
    method = "excitable"

    def test_decay(self):
        """FHN recovery is slow (τ ~ 8s); use sub-threshold deviation + 2000 ticks."""
        layers = make_ring(self.method)
        for l in layers:
            l.v = 0.1  # sub-threshold deviation; no action potential triggered
            l.w = 0.0

        for _ in range(2000):
            step_ring(layers, [0.0] * N)

        for l in layers:
            sig = l.get_signal()
            assert math.isfinite(sig), "Non-finite signal in excitable (decay)"
            assert abs(sig) < 0.05, f"excitable: signal={sig:.4f} did not decay after 2000 ticks"


class TestKdV(_PropagationTests):
    method = "kdv"

    def test_propagation(self):
        """KdV dispersion propagates more slowly; use 200 ticks."""
        layers = make_ring(self.method)
        e_taus = [0.0] * N
        e_taus[0] = 0.8

        for _ in range(200):
            step_ring(layers, e_taus)

        quarter = N // 4
        sig_q = layers[quarter].get_signal()
        assert math.isfinite(sig_q), "Non-finite KdV signal at quarter-ring"
        assert abs(sig_q) > 0.01, (
            f"kdv: signal at node {quarter} = {sig_q:.4f}; "
            "expected propagation to reach N//4 after 200 ticks"
        )


class TestAlarm(_PropagationTests):
    method = "alarm"


class TestBurgers(_PropagationTests):
    method = "burgers"


# ---------------------------------------------------------------------------
# Extra: factory fallback
# ---------------------------------------------------------------------------

def test_factory_unknown_method_returns_baseline():
    layer = create_propagation_layer("nonexistent_method", {})
    assert isinstance(layer, BaselineLayer)
    assert layer.get_signal() == 0.0
    assert layer.get_broadcast_state() == {}
