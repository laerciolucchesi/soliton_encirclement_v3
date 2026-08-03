"""Tests for the target telemetry geometry: G_max, E_gap, alive_count, gap_max_rad.

`G_max` is the quantity the thesis calls mission-critical (the breach window), and
it is normalised by the CURRENT alive count — so a half-dead ring spread perfectly
scores exactly 1, like a full one. `gap_max_rad` and `alive_count` (added
2026-07-26, campaign E4) are what make that normalisation invertible. These tests
lock the geometry, including the property that makes the caveat visible: after a
death and a perfect redistribution, `G_max` returns to 1 while the physical gap
stays permanently wider.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from protocol_messages import AgentState  # noqa: E402
from protocol_target import TARGET_TELEMETRY_COLUMNS, TargetProtocol, skip_telemetry_plots  # noqa: E402


class _Provider:
    def __init__(self, now=0.0):
        self._now = now

    def current_time(self):
        return self._now


class _VelocityHandler:
    """Target parked at the origin; agents are placed by the test itself."""

    def get_node_position(self, node_id):
        return (0.0, 0.0, 0.0)

    def get_node_velocity(self, node_id):
        return (0.0, 0.0, 0.0)


def _target_with_agents(angles_rad, radius=20.0, now=1.0):
    """A TargetProtocol whose cached agent states sit at the given ring angles.

    `initialize()` needs a full simulator provider, so the handful of attributes
    `handle_telemetry` (and the `_prune_expired_states` it calls) touch are set
    directly instead.
    """
    t = TargetProtocol()
    t.provider = _Provider(now)
    t.velocity_handler = _VelocityHandler()
    t.node_id = 0
    t._csv_path = "unused.csv"          # only needs to be truthy
    t._telemetry_rows = []
    t._special_agent_id = None
    t.last_seq_agent = {}
    t.alive_lambdas = {}
    t.agent_states = {}
    for i, th in enumerate(angles_rad):
        pos = (radius * math.cos(th), radius * math.sin(th), 0.0)
        state = AgentState(agent_id=100 + i, seq=1, position=pos,
                           velocity=(0.0, 0.0, 0.0), u=0.0)
        t.agent_states[100 + i] = (state, now)   # rxtime = now -> never expired
    return t


def _row(angles_rad, **kw):
    t = _target_with_agents(angles_rad, **kw)
    t.handle_telemetry(telemetry=None)
    assert len(t._telemetry_rows) == 1
    return t._telemetry_rows[0]


def _equidistant(n, offset=0.0):
    return [offset + i * (2.0 * math.pi / n) for i in range(n)]


# ------------------------------------------------------------------- schema

def test_row_keys_match_the_declared_schema():
    row = _row(_equidistant(8))
    assert set(row) == set(TARGET_TELEMETRY_COLUMNS)


def test_main_writes_the_header_from_the_same_constant():
    """main.py and TargetProtocol.finish() must not carry two copies of the schema."""
    with open(os.path.join(REPO_ROOT, "main.py"), encoding="utf-8") as f:
        src = f.read()
    assert 'TARGET_TELEMETRY_COLUMNS' in src
    assert '",".join(TARGET_TELEMETRY_COLUMNS)' in src
    # No stale literal header left behind.
    assert "timestamp,E_r,E_vr,rho,G_max,E_gap\\n" not in src


# ----------------------------------------------------------------- geometry

@pytest.mark.parametrize("n", [3, 5, 8, 24])
def test_uniform_ring_is_the_reference_point(n):
    """Perfectly spread ring: G_max = 1, E_gap = 0, gap = the ideal gap exactly."""
    row = _row(_equidistant(n))
    assert row["alive_count"] == n
    assert row["G_max"] == pytest.approx(1.0, abs=1e-12)
    assert row["E_gap"] == pytest.approx(0.0, abs=1e-12)
    assert row["gap_max_rad"] == pytest.approx(2.0 * math.pi / n, rel=1e-12)


@pytest.mark.parametrize("n", [8, 24])
def test_gap_max_rad_inverts_the_normalisation(n):
    """The identity that makes the dimensionless metrics convertible to angles."""
    angles = _equidistant(n)
    del angles[3]                                  # one node missing -> non-uniform
    row = _row(angles)
    assert row["gap_max_rad"] == pytest.approx(
        row["G_max"] * 2.0 * math.pi / row["alive_count"], rel=1e-12)


@pytest.mark.parametrize("n", [8, 24])
def test_single_death_peak_matches_the_geometric_prediction(n):
    """Instant after one death, before anyone moves: G_max = 2*(n-1)/n.

    Two adjacent gaps of 2*pi/n merge while the ideal becomes 2*pi/(n-1). This is
    the protocol-independent floor on the breach peak — no coordination scheme can
    beat it, because it holds at the instant of the failure.
    """
    angles = _equidistant(n)
    del angles[3]
    row = _row(angles)
    assert row["alive_count"] == n - 1
    assert row["G_max"] == pytest.approx(2.0 * (n - 1) / n, rel=1e-12)
    assert row["gap_max_rad"] == pytest.approx(2.0 * (2.0 * math.pi / n), rel=1e-12)


@pytest.mark.parametrize("n", [8, 24])
def test_perfect_redistribution_hides_a_permanently_wider_gap(n):
    """The caveat, as an executable statement.

    After the survivors redistribute perfectly, G_max returns to 1.0 — identical to
    the intact ring — but the physical gap is now 2*pi/(n-1), strictly wider than
    the original 2*pi/n. Reporting G_max alone therefore cannot express that the
    breach got bigger; gap_max_rad can.
    """
    before = _row(_equidistant(n))
    after = _row(_equidistant(n - 1))

    assert after["G_max"] == pytest.approx(before["G_max"], abs=1e-12)   # both 1.0
    assert after["E_gap"] == pytest.approx(before["E_gap"], abs=1e-12)   # both 0.0
    assert after["gap_max_rad"] > before["gap_max_rad"]
    assert after["gap_max_rad"] / before["gap_max_rad"] == pytest.approx(
        n / (n - 1), rel=1e-12)


def test_adjacent_double_death_widens_the_peak_further():
    """k adjacent deaths merge k+1 gaps: G_max = 3*(n-2)/n for k = 2."""
    n = 24
    angles = _equidistant(n)
    del angles[4]
    del angles[3]
    row = _row(angles)
    assert row["alive_count"] == n - 2
    assert row["G_max"] == pytest.approx(3.0 * (n - 2) / n, rel=1e-12)
    assert row["gap_max_rad"] == pytest.approx(3.0 * (2.0 * math.pi / n), rel=1e-12)


def test_rotation_invariance():
    """Metrics depend on the gaps, not on where the ring starts."""
    a = _row(_equidistant(12))
    b = _row(_equidistant(12, offset=0.937))
    for key in ("G_max", "E_gap", "alive_count", "gap_max_rad"):
        assert b[key] == pytest.approx(a[key], abs=1e-12)


def test_empty_ring_does_not_crash():
    row = _row([])
    assert row["alive_count"] == 0
    assert row["G_max"] == 0.0
    assert row["gap_max_rad"] == 0.0


# ---------------------------------------------------------------------------
# SKIP_TELEMETRY_PLOTS gate
# ---------------------------------------------------------------------------
# TargetProtocol.finish() is the SECOND PNG producer in this repo (the five
# metric_*.png); plot_telemetry is the first. Only plot_telemetry honoured the
# flag, so every sweep cell rendered five figures nobody consumed. These lock
# the gate to the same spelling and the same accepted values as plot_telemetry,
# since the two are coupled only by convention.

@pytest.mark.parametrize("value", ["True", "true", "1", "yes", "y", "  TRUE  "])
def test_skip_plots_accepts_the_same_truthy_values_as_plot_telemetry(monkeypatch, value):
    monkeypatch.setenv("SKIP_TELEMETRY_PLOTS", value)
    assert skip_telemetry_plots() is True


@pytest.mark.parametrize("value", ["False", "false", "0", "no", "", "off"])
def test_skip_plots_defaults_to_rendering(monkeypatch, value):
    monkeypatch.setenv("SKIP_TELEMETRY_PLOTS", value)
    assert skip_telemetry_plots() is False


def test_skip_plots_absent_env_renders(monkeypatch):
    monkeypatch.delenv("SKIP_TELEMETRY_PLOTS", raising=False)
    assert skip_telemetry_plots() is False


def test_flag_is_read_at_call_time_not_import_time(monkeypatch):
    # Runners set the env for the CHILD process, and main.py imports this module
    # before anything reads the flag; caching it at import would silently ignore
    # the setting for exactly the runs that need it.
    monkeypatch.setenv("SKIP_TELEMETRY_PLOTS", "True")
    assert skip_telemetry_plots() is True
    monkeypatch.setenv("SKIP_TELEMETRY_PLOTS", "False")
    assert skip_telemetry_plots() is False
