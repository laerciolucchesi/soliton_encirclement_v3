"""Tests for robustness to stale / out-of-order AgentState delivery.

Distinct from packet LOSS (never arrives) and uniform DELAY (all arrive late):
a STALE message arrives carrying state older than a message already processed
from the same sender (reordering / out-of-order delivery). The protocol guards
the neighbor cache with a per-sender sequence number; this locks that guard so
the Cap. 7 "tolerates out-of-order delivery" claim is backed by a test.

The decision is the pure helper AgentProtocol._accept_neighbor_state, extracted
behavior-preserving from handle_packet.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from protocol_agent import AgentProtocol  # noqa: E402

accept = AgentProtocol._accept_neighbor_state


def test_fresher_message_accepted():
    # seq strictly greater than the last seen -> accept.
    assert accept(seq=11, last_seq=10, neighbor_expired=False) is True


def test_stale_reordered_message_rejected_when_live():
    # An OLD message arriving after a newer one must NOT overwrite fresh state.
    assert accept(seq=7, last_seq=10, neighbor_expired=False) is False


def test_duplicate_seq_rejected_when_live():
    # Exact duplicate (same seq) is dropped.
    assert accept(seq=10, last_seq=10, neighbor_expired=False) is False


def test_stale_message_accepted_after_expiry():
    # After the sender expired (liveness timeout), accept even an old/reset seq
    # so a recovered neighbor can be re-acquired.
    assert accept(seq=7, last_seq=10, neighbor_expired=True) is True


def test_fresh_message_accepted_after_expiry():
    assert accept(seq=11, last_seq=10, neighbor_expired=True) is True


def test_first_message_from_unseen_sender_accepted():
    # last_seq defaults to -1 for an unseen sender -> any non-negative seq accepted.
    assert accept(seq=0, last_seq=-1, neighbor_expired=True) is True
    assert accept(seq=1, last_seq=-1, neighbor_expired=False) is True


def test_guard_is_monotone_freshness_or_expiry():
    # Property: for a LIVE neighbor, acceptance is exactly seq-monotonicity.
    for last in range(0, 5):
        for s in range(0, 10):
            assert accept(s, last, neighbor_expired=False) == (s > last)
            assert accept(s, last, neighbor_expired=True) is True
