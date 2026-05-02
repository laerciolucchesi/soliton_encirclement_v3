"""Batch sweep across propagation methods, composition modes, and K_PROP values.

Generates the full grid:

    1 baseline run (no propagation, no mode/K_PROP variation)
  + 6 propagation methods x 2 composition modes x 5 K_PROP values
  = 61 runs

Each run invokes ``python main.py`` as a subprocess with the relevant choice
exposed via environment variables (PROPAGATION_METHOD, PROPAGATION_K_PROP,
TANGENTIAL_COMPOSITION_MODE). main.py's interactive menu is bypassed when
PROPAGATION_METHOD is set, so the run is fully unattended. The
post-simulation analysis already wired into main.py appends one row per run
to runs_summary.csv.

Usage:
    python run_sweep.py             # interactive confirm; resume by skipping done combos
    python run_sweep.py --yes       # auto-confirm
    python run_sweep.py --fresh     # ignore existing runs_summary.csv rows
    python run_sweep.py --verbose   # stream subprocess output instead of capturing it

Resume support: a row in runs_summary.csv with matching
(propagation_method, composition_mode, k_prop) is treated as already done and
skipped. Use --fresh to re-run everything regardless.

Failed runs: stdout/stderr is captured to sweep_failure_<method>_<mode>_kX.X.log
in the repo root for diagnosis. The sweep continues past failures.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parent
RUNS_SUMMARY_PATH = REPO_ROOT / "runs_summary.csv"

PROPAGATION_METHODS: Tuple[str, ...] = (
    "advection",
    "wave",
    "excitable",
    "kdv",
    "alarm",
    "burgers",
)
COMPOSITION_MODES: Tuple[str, ...] = ("blend", "sum")
K_PROP_VALUES: Tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0)


RunSpec = Tuple[str, str, float]


def build_run_list() -> List[RunSpec]:
    """Build the full list of (method, mode, k_prop) tuples for the sweep.

    Baseline appears once (mode and k_prop are placeholders, ignored by main.py
    because it forces k_prop=0 and never instantiates the propagation layer).
    """
    runs: List[RunSpec] = [("baseline", "blend", 0.0)]
    for method in PROPAGATION_METHODS:
        for mode in COMPOSITION_MODES:
            for k in K_PROP_VALUES:
                runs.append((method, mode, k))
    return runs


def _round_k(k: float) -> float:
    return round(float(k), 6)


def load_completed_combos() -> Set[Tuple[str, str, float]]:
    """Return (method, mode, k_prop_rounded) tuples already present in the CSV."""
    completed: Set[Tuple[str, str, float]] = set()
    if not RUNS_SUMMARY_PATH.exists() or RUNS_SUMMARY_PATH.stat().st_size == 0:
        return completed
    with open(RUNS_SUMMARY_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "composition_mode" not in reader.fieldnames:
            # Old schema (pre-composition_mode) — treat as nothing matches.
            return completed
        for row in reader:
            method = (row.get("propagation_method") or "").strip()
            mode = (row.get("composition_mode") or "").strip()
            try:
                k = _round_k(float(row.get("k_prop", "nan")))
            except (TypeError, ValueError):
                continue
            if method:
                completed.add((method, mode, k))
    return completed


def fmt_dur(seconds: float) -> str:
    return str(timedelta(seconds=int(max(0.0, seconds))))


def run_single(method: str, mode: str, k_prop: float, verbose: bool) -> subprocess.CompletedProcess:
    """Spawn one main.py run with the env vars wired up. Block until it exits."""
    env = dict(os.environ)
    env["PROPAGATION_METHOD"] = method
    env["PROPAGATION_K_PROP"] = f"{k_prop:.6f}"
    env["PROPAGATION_PARAMS"] = "{}"
    env["TANGENTIAL_COMPOSITION_MODE"] = mode
    # Force UTF-8 so unicode chars in main.py's prints (e.g. "→") don't crash on
    # Windows when stdout is captured by a pipe (cp1252 default has no '→').
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [sys.executable, "-u", str(REPO_ROOT / "main.py")]

    if verbose:
        return subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdin=subprocess.DEVNULL,
        )
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _save_failure_log(proc: subprocess.CompletedProcess, method: str, mode: str, k_prop: float) -> Path:
    log_path = REPO_ROOT / f"sweep_failure_{method}_{mode}_k{k_prop:.1f}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"=== Command exit code: {proc.returncode} ===\n\n")
        f.write("=== STDOUT ===\n")
        f.write((proc.stdout or "") + "\n")
        f.write("=== STDERR ===\n")
        f.write((proc.stderr or "") + "\n")
    return log_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch sweep over propagation/mode/K_PROP.")
    ap.add_argument("--yes", "-y", action="store_true", help="skip the confirmation prompt")
    ap.add_argument("--fresh", action="store_true", help="ignore existing rows in runs_summary.csv")
    ap.add_argument("--verbose", "-v", action="store_true", help="stream subprocess output instead of capturing")
    args = ap.parse_args()

    all_runs = build_run_list()
    total = len(all_runs)
    completed = set() if args.fresh else load_completed_combos()
    pending: List[RunSpec] = [
        r for r in all_runs if (r[0], r[1], _round_k(r[2])) not in completed
    ]

    print("=== Sweep plan ===")
    print(f"  Methods:        {', '.join(PROPAGATION_METHODS)}")
    print(f"  Modes:          {', '.join(COMPOSITION_MODES)}")
    print(f"  K_PROP values:  {', '.join(f'{k:.1f}' for k in K_PROP_VALUES)}")
    print(f"  Total runs:     {total}")
    print(f"  Already done:   {total - len(pending)}")
    print(f"  Pending runs:   {len(pending)}")
    print(f"  Output CSV:     {RUNS_SUMMARY_PATH}")

    if not pending:
        print("\nNothing to do. Use --fresh to force re-running all combinations.")
        return 0

    if not args.yes:
        try:
            ans = input(f"\nProceed with {len(pending)} runs? [y/N]: ").strip().lower()
        except EOFError:
            ans = ""
        if ans not in ("y", "yes"):
            print("Aborted.")
            return 1

    print()
    sweep_start = time.monotonic()
    successes = 0
    failures = 0
    failed_runs: List[Tuple[str, str, float, int, Path]] = []

    for i, (method, mode, k_prop) in enumerate(pending):
        elapsed = time.monotonic() - sweep_start
        avg = elapsed / i if i > 0 else 0.0
        remaining = (len(pending) - i) * avg
        eta = (datetime.now() + timedelta(seconds=remaining)).strftime("%H:%M:%S") if i > 0 else "--:--:--"

        label = f"{method:<10s} mode={mode:<5s} K_PROP={k_prop:>4.1f}"
        progress = f"[{i+1:>3d}/{len(pending)}]"
        if i == 0:
            print(f"{progress} {label} | elapsed={fmt_dur(elapsed)}", flush=True)
        else:
            print(
                f"{progress} {label} | elapsed={fmt_dur(elapsed)}"
                f" | avg/run={fmt_dur(avg)} | ETA={eta}",
                flush=True,
            )

        run_start = time.monotonic()
        try:
            proc = run_single(method, mode, k_prop, args.verbose)
        except KeyboardInterrupt:
            print("\n[sweep] Interrupted by user. Partial results are in runs_summary.csv.", flush=True)
            return 130
        run_dur = time.monotonic() - run_start

        if proc.returncode == 0:
            print(f"           OK   ({run_dur:.1f}s)", flush=True)
            successes += 1
        else:
            log_path: Path | None = None
            if not args.verbose:
                log_path = _save_failure_log(proc, method, mode, k_prop)
            print(
                f"           FAIL ({run_dur:.1f}s, rc={proc.returncode})"
                + (f" -- log: {log_path.name}" if log_path else ""),
                flush=True,
            )
            failures += 1
            failed_runs.append((method, mode, k_prop, proc.returncode, log_path or Path("(stdout)")))

    total_dur = time.monotonic() - sweep_start
    print("\n=== Sweep finished ===")
    print(f"  Successes:   {successes}")
    print(f"  Failures:    {failures}")
    print(f"  Total time:  {fmt_dur(total_dur)}")
    if failures:
        print("\n  Failed combinations:")
        for method, mode, k_prop, rc, log_path in failed_runs:
            print(f"    - {method:<10s} mode={mode:<5s} K_PROP={k_prop:.1f} (rc={rc}, log: {log_path.name})")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
