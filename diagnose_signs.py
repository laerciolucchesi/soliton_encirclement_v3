"""Diagnostic script: verify the angular geometry around each failure event.

Reads the existing run artifacts (agent_telemetry.csv + events.csv) and, for
each `failure_start` event, reconstructs the angular ring order at the moment
of failure to predict what the pulse amplitude *should* be at each immediate
neighbor of the failed node. Compares the prediction against the actual
`pulse_injected` rows in events.csv and flags sign mismatches.

Use this to investigate sign anomalies: for example, a case where both
neighbors of a failed node fired pulses with the same sign, even though pure
geometry would predict opposite signs.

Usage:
    python diagnose_signs.py                      # use CSVs in cwd
    python diagnose_signs.py path/to/run/dir      # use CSVs in given directory

Requires the telemetry CSV to contain a `theta_rel` column (added by the
Phase A patch). If the column is missing, re-run main.py to regenerate.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


# Match the constants used by protocol_agent so the prediction is faithful.
AGENT_STATE_TIMEOUT_DEFAULT = 0.25  # 5 * CONTROL_PERIOD = 5 * 0.05
WINDOW_BEFORE_FAILURE = 0.05
WINDOW_AFTER_TRIGGER = 0.30


def wrap_to_2pi(angle: float) -> float:
    return float(angle) % (2.0 * math.pi)


def compute_spacing_error(gap_pred: float, gap_succ: float) -> float:
    """Mirror of protocol_agent.compute_spacing_error with uniform lambdas (=1)."""
    num = float(gap_succ) - float(gap_pred)
    denom = float(gap_succ) + float(gap_pred)
    if denom <= 1e-9:
        return 0.0
    return num / denom


def snapshot_theta_at(df: pd.DataFrame, t_target: float, tol: float = 0.05) -> pd.DataFrame:
    """For each node, return the row whose timestamp is closest to ``t_target``.

    Drops nodes that have no row within ``tol`` seconds of t_target.
    """
    rows = []
    for nid, g in df.groupby("node_id"):
        idx = (g["timestamp"] - t_target).abs().idxmin()
        if abs(float(g.loc[idx, "timestamp"]) - t_target) <= tol:
            rows.append(g.loc[idx])
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_ring_order(snap: pd.DataFrame) -> List[Tuple[int, float]]:
    """Sort agents by theta_rel (CCW order). Returns (node_id, theta) tuples."""
    pairs = [(int(r["node_id"]), float(r["theta_rel"])) for _, r in snap.iterrows()]
    pairs.sort(key=lambda x: (x[1], x[0]))
    return pairs


def find_index(ring: List[Tuple[int, float]], node_id: int) -> Optional[int]:
    for i, (nid, _) in enumerate(ring):
        if nid == node_id:
            return i
    return None


def predict_neighbor_deltas(
    ring: List[Tuple[int, float]],
    failed_idx: int,
) -> Tuple[Tuple[int, float, float, float], Tuple[int, float, float, float]]:
    """Predict (e_tau_before, e_tau_after, delta) for the pred-side and succ-side
    neighbors of the failed node, using the ring snapshot taken before the failure.
    """
    n = len(ring)
    failed_id, failed_theta = ring[failed_idx]
    pred_id, pred_theta = ring[(failed_idx - 1) % n]
    succ_id, succ_theta = ring[(failed_idx + 1) % n]
    pred_pred_id, pred_pred_theta = ring[(failed_idx - 2) % n]
    succ_succ_id, succ_succ_theta = ring[(failed_idx + 2) % n]

    # Pred-side neighbor (its succ WAS the failed node; will skip to failed.succ)
    pred_old_gap_pred = wrap_to_2pi(pred_theta - pred_pred_theta)
    pred_old_gap_succ = wrap_to_2pi(failed_theta - pred_theta)
    pred_new_gap_succ = wrap_to_2pi(succ_theta - pred_theta)
    e_pred_before = compute_spacing_error(pred_old_gap_pred, pred_old_gap_succ)
    e_pred_after = compute_spacing_error(pred_old_gap_pred, pred_new_gap_succ)
    delta_pred = e_pred_after - e_pred_before

    # Succ-side neighbor (its pred WAS the failed node; will skip back to failed.pred)
    succ_old_gap_pred = wrap_to_2pi(succ_theta - failed_theta)
    succ_old_gap_succ = wrap_to_2pi(succ_succ_theta - succ_theta)
    succ_new_gap_pred = wrap_to_2pi(succ_theta - pred_theta)
    e_succ_before = compute_spacing_error(succ_old_gap_pred, succ_old_gap_succ)
    e_succ_after = compute_spacing_error(succ_new_gap_pred, succ_old_gap_succ)
    delta_succ = e_succ_after - e_succ_before

    return (
        (pred_id, e_pred_before, e_pred_after, delta_pred),
        (succ_id, e_succ_before, e_succ_after, delta_succ),
    )


def lookup_e_tau_at(df: pd.DataFrame, node_id: int, t_target: float, tol: float = 0.06) -> Optional[float]:
    g = df[df["node_id"] == node_id]
    if g.empty:
        return None
    idx = (g["timestamp"] - t_target).abs().idxmin()
    if abs(float(g.loc[idx, "timestamp"]) - t_target) > tol:
        return None
    return float(g.loc[idx, "e_tau"])


def diagnose_one_failure(
    df_telem: pd.DataFrame,
    df_events: pd.DataFrame,
    t_fail: float,
    failed_node: int,
    timeout: float,
) -> None:
    print("=" * 78)
    print(f"Failure event:  t = {t_fail:.3f} s,  failed node = {failed_node}")
    print("=" * 78)

    # Snapshot just before failure
    t_before = max(0.0, t_fail - WINDOW_BEFORE_FAILURE)
    snap = snapshot_theta_at(df_telem, t_before)
    if snap.empty:
        print(f"  [skip] no telemetry near t_before = {t_before:.3f}")
        return

    ring = build_ring_order(snap)
    failed_idx = find_index(ring, failed_node)
    if failed_idx is None:
        print(f"  [skip] failed node {failed_node} not in ring at t_before")
        return

    n = len(ring)
    print(f"\n  Ring order at t = {t_before:.3f} s ({n} agents, sorted by theta CCW):")
    print("    pos  node   theta")
    for i, (nid, theta) in enumerate(ring):
        marker = "  <-- FAILED" if nid == failed_node else ""
        print(f"    {i:3d}  {nid:>4d}   {theta:+.4f}{marker}")

    # Geometry prediction
    (pred_id, e_pred_b, e_pred_a, delta_pred), (succ_id, e_succ_b, e_succ_a, delta_succ) = (
        predict_neighbor_deltas(ring, failed_idx)
    )
    print(f"\n  Geometric prediction (failed node's immediate neighbors):")
    print(f"    pred-side neighbor: node {pred_id}")
    print(f"      e_tau before  = {e_pred_b:+.4f}")
    print(f"      e_tau after   = {e_pred_a:+.4f}")
    print(f"      delta_e_tau   = {delta_pred:+.4f}   -> expected pulse sign = {'+' if delta_pred >= 0 else '-'}")
    print(f"    succ-side neighbor: node {succ_id}")
    print(f"      e_tau before  = {e_succ_b:+.4f}")
    print(f"      e_tau after   = {e_succ_a:+.4f}")
    print(f"      delta_e_tau   = {delta_succ:+.4f}   -> expected pulse sign = {'+' if delta_succ >= 0 else '-'}")

    # Observed e_tau in telemetry, just before and just after the trigger
    t_before_trigger = t_fail + timeout - 0.05
    t_after_trigger = t_fail + timeout + 0.02
    e_pred_obs_before = lookup_e_tau_at(df_telem, pred_id, t_before_trigger)
    e_pred_obs_after = lookup_e_tau_at(df_telem, pred_id, t_after_trigger)
    e_succ_obs_before = lookup_e_tau_at(df_telem, succ_id, t_before_trigger)
    e_succ_obs_after = lookup_e_tau_at(df_telem, succ_id, t_after_trigger)

    print(f"\n  Observed e_tau in telemetry (around trigger time t = {t_fail + timeout:.3f}):")
    if e_pred_obs_before is not None and e_pred_obs_after is not None:
        observed_delta_pred = e_pred_obs_after - e_pred_obs_before
        print(
            f"    node {pred_id}: e_tau {e_pred_obs_before:+.4f} -> {e_pred_obs_after:+.4f}  "
            f"(observed delta = {observed_delta_pred:+.4f})"
        )
    if e_succ_obs_before is not None and e_succ_obs_after is not None:
        observed_delta_succ = e_succ_obs_after - e_succ_obs_before
        print(
            f"    node {succ_id}: e_tau {e_succ_obs_before:+.4f} -> {e_succ_obs_after:+.4f}  "
            f"(observed delta = {observed_delta_succ:+.4f})"
        )

    # Actual pulse_injected events fired in the window after this failure
    actual = df_events[
        (df_events["event_type"] == "pulse_injected")
        & (df_events["timestamp"] >= t_fail)
        & (df_events["timestamp"] <= t_fail + WINDOW_AFTER_TRIGGER + timeout)
    ]
    print(f"\n  Logged pulse_injected events in [{t_fail:.3f}, {t_fail + timeout + WINDOW_AFTER_TRIGGER:.3f}]:")
    if actual.empty:
        print("    (none)")
    else:
        for _, ev in actual.iterrows():
            nid = int(ev["node_id"])
            amp = float(ev["amplitude"])
            t = float(ev["timestamp"])
            tag = ""
            if nid == pred_id:
                expected = delta_pred
                ok = math.copysign(1.0, expected) == math.copysign(1.0, amp)
                tag = f"  <- pred-side; predicted {expected:+.4f}; sign {'OK' if ok else 'MISMATCH **'}"
            elif nid == succ_id:
                expected = delta_succ
                ok = math.copysign(1.0, expected) == math.copysign(1.0, amp)
                tag = f"  <- succ-side; predicted {expected:+.4f}; sign {'OK' if ok else 'MISMATCH **'}"
            else:
                tag = "  <- not an immediate neighbor of failed in pre-failure ring"
            print(f"    t={t:.3f}  node {nid:>2d}  amp={amp:+.4f}{tag}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "run_dir", nargs="?", default=".",
        help="directory containing agent_telemetry.csv and events.csv (default: cwd)",
    )
    ap.add_argument(
        "--timeout", type=float, default=AGENT_STATE_TIMEOUT_DEFAULT,
        help=f"AGENT_STATE_TIMEOUT in seconds (default {AGENT_STATE_TIMEOUT_DEFAULT})",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    telem_path = run_dir / "agent_telemetry.csv"
    events_path = run_dir / "events.csv"

    if not telem_path.exists():
        print(f"[error] {telem_path} not found", file=sys.stderr)
        return 1
    if not events_path.exists():
        print(f"[error] {events_path} not found", file=sys.stderr)
        return 1

    print(f"Loading {telem_path.name} and {events_path.name} from {run_dir}\n")
    df_telem = pd.read_csv(telem_path)
    df_events = pd.read_csv(events_path)

    if "theta_rel" not in df_telem.columns:
        print("[error] agent_telemetry.csv has no 'theta_rel' column.", file=sys.stderr)
        print("        Re-run main.py to regenerate with the Phase A telemetry.", file=sys.stderr)
        return 2

    failures = df_events[df_events["event_type"] == "failure_start"].copy()
    if failures.empty:
        print("No failure_start events in events.csv — nothing to diagnose.")
        return 0

    failures = failures.sort_values("timestamp").reset_index(drop=True)
    print(f"Found {len(failures)} failure_start event(s). Diagnosing each...\n")

    for _, ev in failures.iterrows():
        diagnose_one_failure(
            df_telem, df_events,
            t_fail=float(ev["timestamp"]),
            failed_node=int(ev["node_id"]),
            timeout=float(args.timeout),
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
