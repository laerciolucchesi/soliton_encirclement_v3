"""Negative tests for the three sweep assertions in run_comm_range_sweep.

An assertion that never fires is worth nothing, and the positive path is already
covered by the live calibration run (which parses a real main.py marker). What
needs locking is that each guard actually rejects the failure it was written
for, since each corresponds to a run that looks fine and reports fiction:

  A1  gate on, ranges left at the default   -> fully connected run, role-aware label
  A2  target stops hearing live agents      -> metrics normalized by the wrong M
  A3  a protocol class outside the role map -> its links silently keep the default
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "scaling_law"))

import run_comm_range_sweep as sweep  # noqa: E402

N = sweep.N  # 24 by default


def marker(roles=None, matrix=None, default=200.0, differs=1):
    roles = roles if roles is not None else {"adversary": 1, "agent": N, "target": 1}
    matrix = matrix if matrix is not None else {
        "agent>agent": 6.3, "agent>target": 200.0, "target>agent": 200.0,
    }
    roles_txt = ",".join(f"{k}:{v}" for k, v in sorted(roles.items()))
    matrix_txt = ",".join(f"{k}:{v:g}" for k, v in sorted(matrix.items()))
    return (f"[comm] role_aware=1 roles={{{roles_txt}}} matrix={{{matrix_txt}}} "
            f"default={default:g} differs={differs}")


def write_telemetry(tmp_path, alive_series, t0=0.0, dt=0.05):
    """Minimal target_telemetry.csv with just the columns the assertion reads."""
    path = tmp_path / "target_telemetry.csv"
    lines = ["timestamp,alive_count"]
    for i, alive in enumerate(alive_series):
        lines.append(f"{t0 + i * dt:.6f},{alive}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(tmp_path)


@pytest.fixture
def healthy_run(tmp_path):
    """A run that satisfies every assertion: N alive, then N-1 after the death."""
    n_pre = int(sweep.WARMUP / 0.05) + 40
    return write_telemetry(tmp_path, [N] * n_pre + [N - 1] * 200)


# --------------------------------------------------------------------------
# the happy path, so the negatives below mean something
# --------------------------------------------------------------------------

def test_healthy_cell_passes_all_three(healthy_run):
    out = sweep.assert_cell("ok", healthy_run, marker())
    assert out["assert_alive_min"] == N - 1
    assert out["assert_roles_agent"] == N
    assert out["assert_matrix_differs"] is True


# --------------------------------------------------------------------------
# A1 -- no-op matrix
# --------------------------------------------------------------------------

def test_a1_rejects_matrix_equal_to_default(healthy_run):
    with pytest.raises(sweep.AssertionFailed, match="A1"):
        sweep.assert_cell("noop", healthy_run, marker(differs=0))


def test_missing_marker_is_rejected(healthy_run):
    # Gate never reached the child, or it died before build(): no line at all.
    with pytest.raises(sweep.AssertionFailed, match="nao emitiu"):
        sweep.assert_cell("nogate", healthy_run, "simulation finished\n")


# --------------------------------------------------------------------------
# A2 -- the target stops hearing live agents
# --------------------------------------------------------------------------

def test_a2_rejects_alive_count_below_n_minus_one(tmp_path):
    n_pre = int(sweep.WARMUP / 0.05) + 40
    run_dir = write_telemetry(tmp_path, [N] * n_pre + [N - 4] * 50)
    with pytest.raises(sweep.AssertionFailed, match="A2"):
        sweep.assert_cell("uplink", run_dir, marker())


def test_a2_ignores_the_warmup_ramp(tmp_path):
    """alive_count climbing from 0 before the warmup ends is normal, not a fault."""
    n_warm = int(sweep.WARMUP / 0.05)
    ramp = list(range(0, N)) + [N] * max(0, n_warm - N)
    run_dir = write_telemetry(tmp_path, ramp[:n_warm] + [N] * 200)
    out = sweep.assert_cell("ramp", run_dir, marker())
    assert out["assert_alive_min"] == N


def test_a2_reports_when_telemetry_is_missing(tmp_path):
    with pytest.raises(sweep.AssertionFailed, match="nao existe"):
        sweep.assert_cell("empty", str(tmp_path), marker())


# --------------------------------------------------------------------------
# A3 -- role census
# --------------------------------------------------------------------------

def test_a3_rejects_any_unknown_role(healthy_run):
    roles = {"adversary": 1, "agent": N - 1, "target": 1, "unknown": 1}
    with pytest.raises(sweep.AssertionFailed, match="unknown"):
        sweep.assert_cell("unknown", healthy_run, marker(roles=roles))


def test_a3_rejects_wrong_agent_count(healthy_run):
    roles = {"adversary": 1, "agent": N - 1, "target": 1}
    with pytest.raises(sweep.AssertionFailed, match="A3"):
        sweep.assert_cell("short", healthy_run, marker(roles=roles))


def test_a3_rejects_missing_target(healthy_run):
    roles = {"adversary": 1, "agent": N}
    with pytest.raises(sweep.AssertionFailed, match="A3"):
        sweep.assert_cell("notarget", healthy_run, marker(roles=roles))


# --------------------------------------------------------------------------
# the c-normalization the report is read in
# --------------------------------------------------------------------------

def test_c_units_are_the_one_hop_chord():
    assert sweep.chord(1, n=24, radius=20.0) == pytest.approx(5.2214, abs=1e-3)
    assert sweep.c_units(sweep.chord(1)) == pytest.approx(1.0)
    # The swept grid, in hop-chord units.
    assert sweep.c_units(6.3) == pytest.approx(1.207, abs=1e-3)
    assert sweep.c_units(26.1) == pytest.approx(4.999, abs=1e-3)


def test_runner_parses_what_the_handler_actually_emits():
    """Contract test across the process boundary.

    The handler formats the line, the runner regex-parses it, and nothing else
    couples them -- so a change to describe() would silently disable A1 and A3
    (parse returns None, every cell aborts) unless this fails first.
    """
    from gradysim.simulator.handler.communication import CommunicationMedium
    from gradysim.simulator.node import Node

    from comm_role_aware import RoleAwareCommunicationHandler, agent_target_range_matrix

    class TargetProtocol:
        pass

    class AgentProtocol:
        pass

    class _Enc:
        def __init__(self, protocol):
            self.protocol = protocol

    handler = RoleAwareCommunicationHandler(
        CommunicationMedium(transmission_range=200.0),
        range_matrix=agent_target_range_matrix(agent_agent=6.3, agent_target=200.0),
    )
    handler.inject(object())
    for node_id, protocol_cls in [(0, TargetProtocol)] + [(2 + i, AgentProtocol) for i in range(N)]:
        node = Node()
        node.id = node_id
        node.position = (0.0, 0.0, 0.0)
        node.protocol_encapsulator = _Enc(protocol_cls())
        handler.register_node(node)

    parsed = sweep.parse_comm_marker(handler.describe())
    assert parsed is not None, "the runner can no longer parse describe()"
    assert parsed["roles"] == {"target": 1, "agent": N}
    assert parsed["matrix"]["agent>agent"] == pytest.approx(6.3)
    assert parsed["default"] == pytest.approx(200.0)
    assert parsed["differs"] is True


def test_paired_seeds_pick_the_same_victim_for_both_methods():
    # The pairing the sweep relies on: the seed, not the method, chooses the victim.
    assert sweep.victim_node_id(24, 3) == sweep.victim_node_id(24, 3)
    assert len({sweep.victim_node_id(24, s) for s in range(8)}) == 8
