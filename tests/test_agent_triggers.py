"""Tests for the SAIDA/ENTRADA topology-event classification.

The dual_pulse method distinguishes a node LEAVING (SAIDA) from a node
ENTERING (ENTRADA) using only neighbor-local information: whether the
agent's previous successor is still alive when the successor identity
changes. This predicate is the heart of the flagship method's trigger
gate, so its truth table is worth pinning explicitly.
"""

from __future__ import annotations

import os
import sys

import pytest

# Ensure repo root is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from protocol_agent import AgentProtocol  # noqa: E402

classify = AgentProtocol._classify_succ_event


def test_no_change_yields_no_event():
    assert classify(False, False) == (False, False)
    assert classify(False, True) == (False, False)


def test_succ_changed_and_old_succ_dead_is_saida():
    saida, entrada = classify(True, False)
    assert saida is True
    assert entrada is False


def test_succ_changed_and_old_succ_alive_is_entrada():
    saida, entrada = classify(True, True)
    assert saida is False
    assert entrada is True


@pytest.mark.parametrize("succ_changed", [False, True])
@pytest.mark.parametrize("old_alive", [False, True])
def test_saida_and_entrada_are_mutually_exclusive(succ_changed, old_alive):
    saida, entrada = classify(succ_changed, old_alive)
    assert not (saida and entrada)
    # An event fires only when the successor actually changed.
    assert (saida or entrada) == bool(succ_changed)
