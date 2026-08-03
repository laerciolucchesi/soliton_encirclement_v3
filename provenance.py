"""Run provenance: git state + the fully-resolved parameter snapshot of a run.

Why this module exists
----------------------
The main campaign's result CSVs (``experiments/scaling_law/*.csv``, produced
between 2026-05-30 and 2026-06-06) were generated from an uncommitted working
tree and record neither the seed, nor the code version, nor the parameters that
were pinned for the run. Those numbers cannot be re-derived: the tree that
produced them no longer exists anywhere. This module makes every run
self-describing so that never happens again.

Contract
--------
Nothing here influences the simulation. It only *reads* ``config_param``,
``os.environ`` and ``git``. Every entry point is exception-safe: a missing
``git`` binary, a non-repository working directory or an unreadable config
attribute degrades to a recorded "unknown", never to a crashed run.

Two levels of detail are produced:

* :func:`summary_provenance` — a flat ``{column: value}`` dict of the fields that
  go into every row of ``runs_summary.csv`` (see ``plot_telemetry.SUMMARY_COLUMNS``).
* :func:`write_run_manifest` — ``run_manifest.json`` in the run directory, with
  the COMPLETE resolved parameter set, the argv, and the raw ``git status``
  output. This file is the source of truth whenever the CSV row and reality
  seem to disagree.

Values always come from the *resolved* state of ``config_param`` at run time
(read through the module object), never from a hand-maintained parallel copy
that could silently drift from the defaults it claims to mirror.

Capture timing
--------------
The git state is captured ONCE per process and cached (:func:`git_provenance`).
``main.py`` primes the cache by writing the manifest BEFORE the simulation
starts, so the recorded state is the state of the tree that produced the run —
not the post-run tree, which the run's own outputs may have dirtied.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import config_param

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

MANIFEST_FILENAME = "run_manifest.json"
MANIFEST_SCHEMA = "run_manifest/1"

# Cap the raw `git status --porcelain` dump stored in the manifest. A tree with
# thousands of untracked run artefacts should not produce a multi-megabyte JSON.
_STATUS_MAX_CHARS = 20000

_GIT_TIMEOUT_SEC = 15.0

# Process-wide cache: (commit_short, dirty, extra_details). Populated on first
# call so that every consumer in the process reports the same, pre-run state.
_GIT_CACHE: Optional[Tuple[str, bool, Dict[str, Any]]] = None


# ----------------------------------------------------------------------------
# git
# ----------------------------------------------------------------------------

def _run_git(*args: str) -> Optional[str]:
    """Run a git command in the repo root. Return stdout, or None on any failure.

    Deliberately silent: provenance capture must never be able to abort a
    simulation. `git` missing from PATH, a detached/empty repository, a
    non-repository checkout and a hung invocation all map to None.
    """
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_GIT_TIMEOUT_SEC,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _git_provenance() -> Tuple[str, bool]:
    """Return ``(short_commit_sha, dirty)`` for the repository holding this file.

    ``dirty`` is True when ``git status --porcelain`` reports ANY entry — staged,
    unstaged or untracked-and-not-ignored. That is deliberately the same test
    campaign rule 1 applies by hand before starting a run, so the recorded flag
    and the operational rule cannot disagree.

    Falls back to ``("unknown", True)`` when git is unavailable or this is not a
    repository: unknown provenance is assumed non-reproducible, never clean.
    """
    sha = _run_git("rev-parse", "--short", "HEAD")
    status = _run_git("status", "--porcelain")
    if sha is None or status is None:
        return "unknown", True
    sha = sha.strip()
    if not sha:
        return "unknown", True
    return sha, bool(status.strip())


def _git_details() -> Dict[str, Any]:
    """Richer git context for the manifest (never consumed by the CSV row)."""
    commit_full = _run_git("rev-parse", "HEAD")
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    status = _run_git("status", "--porcelain")
    describe = _run_git("describe", "--always", "--dirty", "--tags")

    status_txt = "" if status is None else status.strip()
    truncated = len(status_txt) > _STATUS_MAX_CHARS
    if truncated:
        status_txt = status_txt[:_STATUS_MAX_CHARS]

    return {
        "commit_full": (commit_full or "").strip() or "unknown",
        "branch": (branch or "").strip() or "unknown",
        "describe": (describe or "").strip() or "unknown",
        "status_porcelain": status_txt,
        "status_porcelain_truncated": truncated,
        "repo_root": REPO_ROOT,
    }


def git_provenance(refresh: bool = False) -> Tuple[str, bool]:
    """Cached :func:`_git_provenance`. See the module docstring on capture timing."""
    global _GIT_CACHE
    if _GIT_CACHE is None or refresh:
        sha, dirty = _git_provenance()
        _GIT_CACHE = (sha, dirty, _git_details())
    return _GIT_CACHE[0], _GIT_CACHE[1]


def git_details(refresh: bool = False) -> Dict[str, Any]:
    git_provenance(refresh=refresh)
    assert _GIT_CACHE is not None
    return dict(_GIT_CACHE[2])


# ----------------------------------------------------------------------------
# resolved configuration
# ----------------------------------------------------------------------------

def _jsonable(value: Any) -> Any:
    """Coerce a config value into something json.dump can write."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (frozenset, set)):
        return sorted(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return repr(value)


def resolved_config() -> Dict[str, Any]:
    """Every resolved public constant of ``config_param``, as a plain dict.

    Read from the imported module object, so env-var overrides applied at import
    time are already folded in — this is what the run actually used.
    """
    out: Dict[str, Any] = {}
    for name in dir(config_param):
        if name.startswith("_") or not name.isupper():
            continue
        value = getattr(config_param, name, None)
        if callable(value) or isinstance(value, type(os)):
            continue
        out[name] = _jsonable(value)
    return dict(sorted(out.items()))


def experiment_seed() -> int:
    """The seed ``main.py`` feeds to ``random.seed()``, resolved the same way.

    Recorded unconditionally. When ``experiment_reproducible`` is False (the
    adjacent column) the seed was parsed but never applied — the pair is what
    makes the row unambiguous.
    """
    try:
        return int(os.environ.get("EXPERIMENT_SEED", "0"))
    except ValueError:
        return 0


# Summary column -> config_param attribute. Single mapping table: adding a
# column means adding one entry here and one entry to plot_telemetry's column
# list, and the value is then pulled from the live module. There is deliberately
# no second dict holding literal values that could drift from config_param.
#
# Name notes (the campaign prompt used shorthand for three of these):
#   communication_range           <- COMMUNICATION_TRANSMISSION_RANGE
#   broadcast_repeats             <- DUAL_PULSE_BROADCAST_REPEATS
#   deterministic_failure_time    <- DETERMINISTIC_FAILURE_TIME_T0
SUMMARY_FROM_CONFIG: Dict[str, str] = {
    # Core loop / scale
    "control_period": "CONTROL_PERIOD",
    "k_e_tau": "K_E_TAU",
    "composition_mode": "TANGENTIAL_COMPOSITION_MODE",
    "num_agents": "NUM_AGENTS",
    "encirclement_radius": "ENCIRCLEMENT_RADIUS",
    "sim_duration": "SIM_DURATION",
    "protection_angle_deg": "PROTECTION_ANGLE_DEG",
    # dual_pulse overlay
    "dual_pulse_integration": "DUAL_PULSE_INTEGRATION",
    "dual_pulse_delta_scale": "DUAL_PULSE_DELTA_SCALE",
    "dual_pulse_ttl_hops": "DUAL_PULSE_TTL_HOPS",
    "dual_pulse_t_ff": "DUAL_PULSE_T_FF",
    # Both default ON and both change the delta that is applied; without them a
    # post-Ciclo-2 row is indistinguishable from a pre-M8/pre-M-mult one.
    "dual_pulse_multiplicity": "DUAL_PULSE_MULTIPLICITY",
    "dual_pulse_consume_ff_only": "DUAL_PULSE_CONSUME_FF_ONLY",
    # Communication medium + failure detector
    "communication_delay": "COMMUNICATION_DELAY",
    "communication_failure_rate": "COMMUNICATION_FAILURE_RATE",
    "communication_range": "COMMUNICATION_TRANSMISSION_RANGE",
    # Asymmetric ranges. communication_range alone cannot distinguish a run with
    # a 30 m ring from a fully connected one, so the gate AND both ranges have
    # to be on the row.
    "comm_role_aware_ranges": "COMM_ROLE_AWARE_RANGES",
    "comm_range_agent_agent": "COMM_RANGE_AGENT_AGENT",
    "comm_range_agent_target": "COMM_RANGE_AGENT_TARGET",
    "agent_state_timeout": "AGENT_STATE_TIMEOUT",
    "broadcast_repeats": "DUAL_PULSE_BROADCAST_REPEATS",
    # Failure injection
    "deterministic_failure_enable": "DETERMINISTIC_FAILURE_ENABLE",
    "deterministic_failure_agent_id": "DETERMINISTIC_FAILURE_AGENT_ID",
    # The full victim set: churn scenarios fail k nodes at once and the
    # back-compat _AGENT_ID column only keeps the first one.
    "deterministic_failure_agent_ids": "DETERMINISTIC_FAILURE_AGENT_IDS",
    "deterministic_failure_time": "DETERMINISTIC_FAILURE_TIME_T0",
    # Permanent crash (<0) vs SAIDA+ENTRADA pair: different event semantics.
    "deterministic_failure_off_time": "DETERMINISTIC_FAILURE_OFF_TIME",
    "failure_enable": "FAILURE_ENABLE",
    "failure_mean_per_min": "FAILURE_MEAN_FAILURES_PER_MIN",
    "experiment_reproducible": "EXPERIMENT_REPRODUCIBLE",
    # Platform / scenario
    "vm_tau_xy": "VM_TAU_XY",
    "vm_max_speed_xy": "VM_MAX_SPEED_XY",
    "target_motion_speed_xy": "TARGET_MOTION_SPEED_XY",
    "target_swarm_spin_enable": "TARGET_SWARM_SPIN_ENABLE",
    "target_swarm_omega_ref": "TARGET_SWARM_OMEGA_REF",
    "init_radius_range": "INIT_RADIUS_RANGE",
    "init_angles_equidistant": "INIT_ANGLES_EQUIDISTANT",
}

# Columns produced by this module that are not a direct config_param read.
COMPUTED_COLUMNS = ("git_commit", "git_dirty", "experiment_seed")

# Full ordered set of columns summary_provenance() returns.
PROVENANCE_COLUMNS = list(COMPUTED_COLUMNS) + list(SUMMARY_FROM_CONFIG.keys())


def _flatten(value: Any) -> Any:
    """CSV-safe rendering. Collections become ';'-joined so no quoting is needed."""
    if isinstance(value, (frozenset, set)):
        return ";".join(str(v) for v in sorted(value))
    if isinstance(value, (list, tuple)):
        return ";".join(str(v) for v in value)
    return value


def summary_provenance() -> Dict[str, Any]:
    """Flat ``{column: value}`` of provenance + pinned parameters for one row.

    Safe to call from anywhere in the repo (``plot_telemetry``, the sweep runners
    via ``metrics_util.provenance_fields``). Missing config attributes surface as
    the string ``"missing"`` rather than raising, so a future rename degrades a
    single cell instead of killing the run.
    """
    commit, dirty = git_provenance()
    row: Dict[str, Any] = {
        "git_commit": commit,
        "git_dirty": bool(dirty),
        "experiment_seed": experiment_seed(),
    }
    for column, attr in SUMMARY_FROM_CONFIG.items():
        row[column] = _flatten(getattr(config_param, attr, "missing"))
    return row


# ----------------------------------------------------------------------------
# manifest
# ----------------------------------------------------------------------------

# Env vars that steer a run but are not config_param constants.
_EXTRA_ENV_KEYS = (
    "EXPERIMENT_SEED",
    "PROPAGATION_METHOD",
    "PROPAGATION_K_PROP",
    "PROPAGATION_PARAMS",
    "TANGENTIAL_COMPOSITION_MODE",
    "SKIP_TELEMETRY_PLOTS",
    "RUNS_SUMMARY_CSV_PATH",
    "AGENT_LOG_CSV_PATH",
    "TARGET_LOG_CSV_PATH",
    "EVENTS_LOG_CSV_PATH",
    "PYTHONIOENCODING",
    "PYTHONHASHSEED",
)


def env_overrides() -> Dict[str, str]:
    """Env vars actually SET for this process that can steer the simulation.

    Every config_param override uses the constant's own name, so the constant
    list doubles as the override key list; ``_EXTRA_ENV_KEYS`` covers the few
    that live outside config_param (seed, propagation selection, output paths).
    Recording only what is set distinguishes "pinned by the runner" from
    "inherited from the default" — which is exactly what campaign rule 3 is about.
    """
    keys = {n for n in dir(config_param) if n.isupper() and not n.startswith("_")}
    keys.update(_EXTRA_ENV_KEYS)
    return {k: os.environ[k] for k in sorted(keys) if k in os.environ}


def build_manifest(argv: Optional[list] = None) -> Dict[str, Any]:
    commit, dirty = git_provenance()
    return {
        "schema": MANIFEST_SCHEMA,
        "run_timestamp_iso": datetime.now().isoformat(timespec="seconds"),
        "argv": list(sys.argv if argv is None else argv),
        "cwd": os.getcwd(),
        "python_version": platform.python_version(),
        "platform": sys.platform,
        "git": {"commit": commit, "dirty": bool(dirty), **git_details()},
        "env_overrides": env_overrides(),
        "resolved_config": resolved_config(),
    }


def write_run_manifest(
    path: Optional[str] = None,
    argv: Optional[list] = None,
) -> Optional[str]:
    """Write ``run_manifest.json`` into the run directory (default: cwd).

    Returns the absolute path written, or None if writing failed. Called by
    ``main.py`` BEFORE the simulation starts, which also primes the git cache so
    the run's own output files cannot make it look like the code was dirty.
    """
    if path is None:
        path = os.path.join(os.getcwd(), MANIFEST_FILENAME)
    try:
        manifest = build_manifest(argv=argv)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=False)
            f.write("\n")
    except Exception as exc:  # provenance must never abort a run
        print(f"[provenance] could not write {path}: {exc}")
        return None
    return os.path.abspath(path)


def read_run_manifest(run_dir: str) -> Optional[Dict[str, Any]]:
    """Load ``run_manifest.json`` from a run directory. None if absent/unreadable."""
    path = os.path.join(run_dir, MANIFEST_FILENAME)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def summary_from_manifest(manifest: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Same flat row as :func:`summary_provenance`, but read from a CHILD's manifest.

    A sweep runner MUST use this rather than calling :func:`summary_provenance`
    directly: the runner is the PARENT process, so its own ``config_param`` holds
    the parent's defaults, not the values it pinned in the child's ``env`` dict.
    Stamping parent values onto a child's result row would record parameters the
    run never used — the precise failure this module exists to prevent.

    Returns ``{}`` when the manifest is missing, so a caller can tell "not
    recorded" apart from "recorded as default".
    """
    if not manifest:
        return {}
    cfg = manifest.get("resolved_config") or {}
    git = manifest.get("git") or {}
    env = manifest.get("env_overrides") or {}
    try:
        seed = int(env.get("EXPERIMENT_SEED", "0"))
    except (TypeError, ValueError):
        seed = 0

    row: Dict[str, Any] = {
        "git_commit": git.get("commit", "unknown"),
        "git_dirty": bool(git.get("dirty", True)),
        "experiment_seed": seed,
    }
    for column, attr in SUMMARY_FROM_CONFIG.items():
        row[column] = _flatten(cfg.get(attr, "missing"))
    return row


if __name__ == "__main__":
    commit, dirty = git_provenance()
    print(f"git_commit={commit}  git_dirty={dirty}  seed={experiment_seed()}")
    print(f"resolved config constants: {len(resolved_config())}")
    print(f"env overrides set: {sorted(env_overrides())}")
