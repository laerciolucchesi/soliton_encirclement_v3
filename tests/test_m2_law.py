"""Unit tests for the m=2 densified-coupling law (item 9; SCOPING_M2 + addendum).

What is locked here, and why:
  * the pinned gain-scale numbers from addendum A.2 — the 3.16x prediction is
    DERIVED from them, so a drift here silently changes the pre-registration;
  * the fairness identity (equal nominal sampled margin g*dt*lambda_max);
  * the anti-aliasing guard — the scoping's key finding was that ring[self+2]
    on a 3-member visible ring returns pred1 as succ2;
  * pin (b): a degraded or w2=0 tick must reproduce the baseline error
    float-for-float, because the campaign's P7 rests on that identity.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config_param  # noqa: E402
import protocol_agent as pa  # noqa: E402
from protocol_agent import AgentProtocol  # noqa: E402

spacing_error = AgentProtocol.compute_spacing_error
second = AgentProtocol.second_neighbors_from_ring
gain_scale = config_param._m2_gain_scale_auto


# ---------------------------------------------------------------------------
# gain scale — addendum A.2 pinned values
# ---------------------------------------------------------------------------

def test_gain_scale_pinned_values():
    assert gain_scale(24, 2.0) == pytest.approx(1.9200956, abs=1e-6)
    assert gain_scale(50, 2.0) == pytest.approx(1.9242895, abs=1e-6)


def test_gain_scale_w2_zero_is_exactly_one():
    # S2 relies on this: with w2=0 the auto scale must be 1.0 EXACTLY, so the
    # m2 arm becomes the baseline law with no float perturbation.
    assert gain_scale(24, 0.0) == 1.0
    assert gain_scale(50, 0.0) == 1.0


@pytest.mark.parametrize("n", [24, 50])
def test_fairness_identity_margin_restored(n):
    """scale * lambda_max(A) == lambda_max(L1): the addendum's criterion."""
    w2 = 2.0
    scale = gain_scale(n, w2)
    lmax_l1 = 0.0
    lmax_a = 0.0
    for k in range(1, n):
        phi = 2.0 * math.pi * k / n
        l1 = 2.0 * (1.0 - math.cos(phi))
        a = (l1 + w2 * (1.0 - math.cos(2.0 * phi))) / (1.0 + w2)
        lmax_l1 = max(lmax_l1, l1)
        lmax_a = max(lmax_a, a)
    assert scale * lmax_a == pytest.approx(lmax_l1, rel=1e-12)


def test_predicted_speedups_from_the_fair_formula():
    """3.1565 (N=24) and 3.1970 (N=50) — the numbers the pre-registration pins."""
    for n, expected in ((24, 3.1565), (50, 3.1970)):
        phi1 = 2.0 * math.pi / n
        l2_l1 = 2.0 * (1.0 - math.cos(phi1))
        l2_s = l2_l1 + 2.0 * (1.0 - math.cos(2.0 * phi1))
        lmax_l1 = max(2.0 * (1.0 - math.cos(2.0 * math.pi * k / n)) for k in range(1, n))
        lmax_s = max(
            2.0 * (1.0 - math.cos(t)) + 2.0 * (1.0 - math.cos(2.0 * t))
            for t in (2.0 * math.pi * k / n for k in range(1, n))
        )
        speedup = (l2_s / lmax_s) / (l2_l1 / lmax_l1)
        assert speedup == pytest.approx(expected, abs=2e-4)


# ---------------------------------------------------------------------------
# the k=2 law is the SAME law — algebra
# ---------------------------------------------------------------------------

def test_e2_zero_at_uniform_equilibrium():
    g = 2.0 * math.pi / 24.0
    assert spacing_error(2 * g, 2 * g, 2.0, 2.0) == 0.0


def test_e2_zero_at_nonuniform_lambda_equilibrium():
    # Arcs proportional to lambdas: lam = [pred2, pred1, self, succ1] = [3,1,2,4].
    # Span pred2->self crosses arcs (3,1); span self->succ2 crosses (2,4).
    c = 0.1
    g_pred_2 = c * (3.0 + 1.0)
    g_succ_2 = c * (2.0 + 4.0)
    assert spacing_error(g_pred_2, g_succ_2, 3.0 + 1.0, 2.0 + 4.0) == pytest.approx(0.0, abs=1e-15)


def test_e2_sign_symmetry():
    v = spacing_error(0.5, 0.7, 2.0, 2.0)
    assert spacing_error(0.7, 0.5, 2.0, 2.0) == pytest.approx(-v, rel=1e-12)
    assert v > 0.0  # succ gap larger -> positive error, same convention as k=1


# ---------------------------------------------------------------------------
# anti-aliasing guard
# ---------------------------------------------------------------------------

def _ring(thetas_ids):
    return sorted([(t, i) for t, i in thetas_ids], key=lambda x: (x[0], x[1]))


def test_guard_drops_on_three_member_ring():
    # The scoping's finding: naive ring[self+2] on a 3-ring returns pred1.
    ring = _ring([(0.0, 10), (1.0, 11), (2.0, 12)])
    self_idx = next(k for k, (_, i) in enumerate(ring) if i == 11)
    assert second(ring, self_idx) is None


def test_guard_drops_on_four_member_ring():
    ring = _ring([(0.0, 10), (1.0, 11), (2.0, 12), (3.0, 13)])
    self_idx = next(k for k, (_, i) in enumerate(ring) if i == 11)
    assert second(ring, self_idx) is None


def test_five_member_ring_resolves_correct_ids():
    ring = _ring([(0.0, 20), (1.0, 21), (2.0, 22), (3.0, 23), (4.0, 24)])
    self_idx = next(k for k, (_, i) in enumerate(ring) if i == 22)
    p2, s2, p2t, s2t, p1, s1 = second(ring, self_idx)
    assert (p2, p1, s1, s2) == (20, 21, 23, 24)
    assert (p2t, s2t) == (0.0, 4.0)
    # No aliasing: the four ids and self are five DISTINCT nodes.
    assert len({p2, p1, 22, s1, s2}) == 5


def test_guard_none_inputs():
    assert second(None, 0) is None
    assert second([], None) is None


# ---------------------------------------------------------------------------
# pin (b): degraded / w2=0 ticks are the baseline error, float for float
# ---------------------------------------------------------------------------

class _FakeTS:
    def __init__(self, lambdas=None):
        self.position = (0.0, 0.0, 0.0)
        self.alive_lambdas = lambdas or {}


class _FakeState:
    def __init__(self, x, y):
        self.position = (x, y, 0.0)


def _make_agent(n_ring, now=30.0, radius=20.0, self_slot=0):
    """Bare AgentProtocol via __new__: only the attributes the m2 path touches."""
    a = AgentProtocol.__new__(AgentProtocol)
    a.node_id = 100 + self_slot
    a.lp_pred = 1.0
    a.lp_succ = 1.0
    a.target_state = (_FakeTS(), now)
    a.agent_states = {}
    for j in range(1, n_ring):
        slot = (self_slot + j) % n_ring
        ang = 2.0 * math.pi * slot / n_ring
        a.agent_states[100 + slot] = (
            _FakeState(radius * math.cos(ang), radius * math.sin(ang)), now
        )
    a._m2_ticks_total = 0
    a._m2_k2_dropped = 0
    a._m2_k2_toggles = 0
    a._m2_ticks_steady = 0
    a._m2_k2_dropped_steady = 0
    a._m2_k2_toggles_steady = 0
    a._m2_prev_dropped = None
    return a


def _own_pos(n_ring, radius=20.0, self_slot=0):
    ang = 2.0 * math.pi * self_slot / n_ring
    return (radius * math.cos(ang), radius * math.sin(ang), 0.0)


def test_m2_error_zero_at_uniform_equilibrium():
    a = _make_agent(8)
    g = 2.0 * math.pi / 8.0
    e = a._m2_error_or_none(30.0, _own_pos(8), g, g)
    assert e is not None and e == pytest.approx(0.0, abs=1e-12)
    assert a._m2_k2_dropped == 0


def test_m2_guard_fires_and_counts_on_small_ring():
    a = _make_agent(3)
    g = 2.0 * math.pi / 3.0
    e = a._m2_error_or_none(30.0, _own_pos(3), g, g)
    assert e is None
    assert a._m2_ticks_total == 1 and a._m2_k2_dropped == 1
    assert a._m2_ticks_steady == 1 and a._m2_k2_dropped_steady == 1  # now >= 20


def test_m2_toggle_counts_edges_not_states(monkeypatch):
    a = _make_agent(8)
    g = 2.0 * math.pi / 8.0
    pos = _own_pos(8)
    a._m2_error_or_none(30.0, pos, g, g)          # alive
    a.agent_states = {k: v for k, v in list(a.agent_states.items())[:2]}  # shrink ring
    a._m2_error_or_none(30.1, pos, g, g)          # dropped  -> edge 1
    a._m2_error_or_none(30.2, pos, g, g)          # dropped  -> no edge
    assert a._m2_k2_dropped == 2
    assert a._m2_k2_toggles == 1                  # edges, not dropped ticks


def test_w2_zero_reproduces_baseline_error_bit_for_bit(monkeypatch):
    """Pin (b) unit form: w2=0 & scale=1 -> e_m2 IS the baseline float."""
    monkeypatch.setattr(pa, "M2_W2", 0.0)
    monkeypatch.setattr(pa, "M2_GAIN_SCALE", 1.0)
    a = _make_agent(8)
    g_pred, g_succ = 0.71, 0.83
    e = a._m2_error_or_none(30.0, _own_pos(8), g_pred, g_succ)
    base = spacing_error(g_pred, g_succ, 1.0, 1.0)
    assert e == base  # equality of floats, not approx — the smoke S2 depends on it


def test_m2_combination_matches_formula(monkeypatch):
    monkeypatch.setattr(pa, "M2_W2", 2.0)
    monkeypatch.setattr(pa, "M2_GAIN_SCALE", 1.92)
    a = _make_agent(8)
    # Perturb: shrink own succ gap (pass k=1 gaps directly), ring stays uniform.
    g = 2.0 * math.pi / 8.0
    e = a._m2_error_or_none(30.0, _own_pos(8), 1.2 * g, 0.8 * g)
    e1 = spacing_error(1.2 * g, 0.8 * g, 1.0, 1.0)
    e2 = 0.0  # ring positions are the uniform ones -> k=2 spans equal
    assert e == pytest.approx(1.92 * (e1 + 2.0 * e2) / 3.0, rel=1e-12)
