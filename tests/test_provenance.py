"""Tests for run provenance: git capture, the summary schema, and the manifest.

The failure this guards against is not a crash — it is a run that completes
happily while recording a blank or wrong provenance cell, which is exactly how
the 2026-05/06 campaign lost its reproducibility. So the assertions are about
*coverage and truthfulness of the row*, not just about the functions returning.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys

import pytest

# Ensure repo root is importable.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import config_param  # noqa: E402
import provenance  # noqa: E402
from plot_telemetry import (  # noqa: E402
    SUMMARY_COLUMNS,
    MetricParams,
    append_run_summary,
)


# ---------------------------------------------------------------- git capture

def test_git_provenance_shape():
    sha, dirty = provenance._git_provenance()
    assert isinstance(sha, str) and sha
    assert isinstance(dirty, bool)


def test_git_provenance_falls_back_when_git_unavailable(monkeypatch):
    """No git binary / not a repository must degrade, never raise.

    The fallback is ("unknown", True): unknown provenance is assumed
    NON-reproducible, so a lost git never silently reads as a clean tree.
    """
    monkeypatch.setattr(provenance, "_run_git", lambda *a: None)
    assert provenance._git_provenance() == ("unknown", True)


def test_run_git_survives_missing_binary(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(provenance.subprocess, "run", boom)
    assert provenance._run_git("rev-parse", "HEAD") is None


def test_git_provenance_is_cached(monkeypatch):
    provenance.git_provenance(refresh=True)
    calls = []
    monkeypatch.setattr(provenance, "_run_git", lambda *a: calls.append(a) or "x")
    provenance.git_provenance()
    assert calls == [], "cached capture must not re-shell out to git"


# ------------------------------------------------------------- resolved config

def test_resolved_config_is_json_serializable():
    cfg = provenance.resolved_config()
    json.dumps(cfg)  # must not raise (frozensets etc. are coerced)
    assert cfg["NUM_AGENTS"] == config_param.NUM_AGENTS
    # frozenset -> sorted list
    assert isinstance(cfg["DETERMINISTIC_FAILURE_AGENT_IDS"], list)


def test_resolved_config_covers_every_public_constant():
    expected = {n for n in dir(config_param) if n.isupper() and not n.startswith("_")}
    missing = expected - set(provenance.resolved_config())
    # Only callables/modules may be dropped; config_param has none of those upper-cased.
    assert not missing, f"resolved_config dropped constants: {sorted(missing)}"


def test_summary_from_config_names_all_exist():
    """Guards against a config rename silently blanking a provenance column."""
    bad = [(col, attr) for col, attr in provenance.SUMMARY_FROM_CONFIG.items()
           if not hasattr(config_param, attr)]
    assert not bad, f"SUMMARY_FROM_CONFIG points at missing config_param names: {bad}"


# ------------------------------------------------------------- summary schema

def test_summary_provenance_matches_declared_columns():
    row = provenance.summary_provenance()
    assert list(row) == list(provenance.PROVENANCE_COLUMNS)


def test_summary_columns_have_no_duplicates():
    assert len(SUMMARY_COLUMNS) == len(set(SUMMARY_COLUMNS))


@pytest.mark.parametrize("field", [
    "git_commit", "git_dirty", "experiment_seed",
    "control_period", "k_e_tau", "num_agents", "encirclement_radius", "sim_duration",
    "dual_pulse_integration", "dual_pulse_delta_scale", "dual_pulse_ttl_hops",
    "dual_pulse_t_ff", "communication_delay", "communication_failure_rate",
    "communication_range", "agent_state_timeout", "broadcast_repeats",
    "deterministic_failure_enable", "deterministic_failure_agent_id",
    "deterministic_failure_time", "failure_enable", "failure_mean_per_min",
    "metrics_t0", "vm_tau_xy", "vm_max_speed_xy", "target_motion_speed_xy",
    "init_radius_range", "init_angles_equidistant",
])
def test_required_campaign_field_present(field):
    """Campaign rule 5: every result row carries seed, git state and the pinned params."""
    assert field in SUMMARY_COLUMNS


def test_experiment_seed_follows_env(monkeypatch):
    monkeypatch.setenv("EXPERIMENT_SEED", "17")
    assert provenance.experiment_seed() == 17
    monkeypatch.setenv("EXPERIMENT_SEED", "not-a-number")
    assert provenance.experiment_seed() == 0


def test_appended_row_fills_every_column(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERIMENT_SEED", "5")
    monkeypatch.setenv("PROPAGATION_METHOD", "dual_pulse")
    monkeypatch.setenv("PROPAGATION_K_PROP", "0.0")
    out = tmp_path / "runs_summary.csv"

    append_run_summary(
        metrics={"M1_P95_e_pooled": 0.1},
        params=MetricParams(dt=0.05, vmax_xy=10.0, t0=5.0, e_thr=0.05,
                            ma_w=1.0, settle_window=5.0),
        summary_csv_path=str(out),
    )

    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert list(row) == SUMMARY_COLUMNS

    # No provenance cell may be blank: a silent blank is the failure mode the
    # whole schema exists to prevent.
    blank = [c for c in provenance.PROVENANCE_COLUMNS if not str(row[c]).strip()]
    assert not blank, f"blank provenance cells: {blank}"

    assert row["experiment_seed"] == "5"
    assert row["propagation_method"] == "dual_pulse"
    assert row["metrics_t0"] == "5.0"            # from MetricParams, not the config default
    assert row["num_agents"] == str(config_param.NUM_AGENTS)
    assert row["dual_pulse_integration"] == config_param.DUAL_PULSE_INTEGRATION


def test_schema_change_rotates_instead_of_corrupting(tmp_path):
    out = tmp_path / "runs_summary.csv"
    out.write_text("old_col_a,old_col_b\n1,2\n", encoding="utf-8")

    append_run_summary(
        metrics={},
        params=MetricParams(dt=0.05, vmax_xy=10.0, t0=0.0, e_thr=0.05,
                            ma_w=1.0, settle_window=5.0),
        summary_csv_path=str(out),
    )

    backups = list(tmp_path.glob("runs_summary.csv.bak.*"))
    assert len(backups) == 1, "old-schema file must be rotated, never overwritten"
    assert backups[0].read_text(encoding="utf-8").startswith("old_col_a")
    with open(out, newline="", encoding="utf-8") as f:
        assert next(csv.reader(f)) == SUMMARY_COLUMNS


# ------------------------------------------------------------------- manifest

def test_manifest_round_trip(tmp_path):
    path = provenance.write_run_manifest(path=str(tmp_path / "run_manifest.json"),
                                         argv=["main.py", "--x"])
    assert path is not None

    manifest = provenance.read_run_manifest(str(tmp_path))
    assert manifest is not None
    assert manifest["schema"] == provenance.MANIFEST_SCHEMA
    assert manifest["argv"] == ["main.py", "--x"]
    assert manifest["resolved_config"]["NUM_AGENTS"] == config_param.NUM_AGENTS
    assert set(manifest["git"]) >= {"commit", "dirty", "branch", "status_porcelain"}


def test_write_run_manifest_never_raises(tmp_path):
    """An unwritable path must degrade to None, not abort the simulation."""
    bad = str(tmp_path / "no-such-dir" / "run_manifest.json")
    assert provenance.write_run_manifest(path=bad) is None


def test_summary_from_manifest_matches_live_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERIMENT_SEED", "11")
    provenance.write_run_manifest(path=str(tmp_path / "run_manifest.json"))
    from_manifest = provenance.summary_from_manifest(
        provenance.read_run_manifest(str(tmp_path))
    )
    live = provenance.summary_provenance()
    assert list(from_manifest) == list(live)
    assert from_manifest == live


def test_summary_from_manifest_missing_is_empty():
    """{} means 'not recorded' — distinguishable from 'recorded as default'."""
    assert provenance.summary_from_manifest(None) == {}
    assert provenance.read_run_manifest(os.path.join(REPO_ROOT, "does-not-exist")) is None


def test_env_overrides_reports_only_what_is_set(monkeypatch):
    monkeypatch.delenv("DUAL_PULSE_TTL_HOPS", raising=False)
    assert "DUAL_PULSE_TTL_HOPS" not in provenance.env_overrides()
    monkeypatch.setenv("DUAL_PULSE_TTL_HOPS", "42")
    assert provenance.env_overrides()["DUAL_PULSE_TTL_HOPS"] == "42"


# --------------------------------------------------------- backward compatibility

def test_run_sweep_resume_survives_the_new_schema(tmp_path, monkeypatch):
    """run_sweep.py is the only reader of runs_summary.csv — the one schema P0 changed.

    Its resume logic must still recognise completed (method, mode, k_prop) combos
    after the provenance columns were inserted ahead of them.
    """
    monkeypatch.setenv("PROPAGATION_METHOD", "kdv")
    monkeypatch.setenv("PROPAGATION_K_PROP", "1.5")
    # composition_mode comes from the ALREADY-IMPORTED config_param: env overrides
    # are applied at import time, so only a child process sees a new value. That is
    # unchanged pre-existing behaviour — assert against the resolved value.
    mode = config_param.TANGENTIAL_COMPOSITION_MODE
    out = tmp_path / "runs_summary.csv"
    append_run_summary(
        metrics={},
        params=MetricParams(dt=0.05, vmax_xy=10.0, t0=0.0, e_thr=0.05,
                            ma_w=1.0, settle_window=5.0),
        summary_csv_path=str(out),
    )

    import run_sweep
    monkeypatch.setattr(run_sweep, "RUNS_SUMMARY_PATH", out)
    assert ("kdv", mode, 1.5) in run_sweep.load_completed_combos()


def test_mixed_provenance_rows_merge_without_error():
    """A results CSV that gains provenance mid-campaign still concatenates.

    The sweep runners merge old rows (read back from the CSV) with new ones via
    pd.DataFrame(list(store.values())); rows missing the new keys must fill with
    NaN and leave the existing metric columns readable, not raise.
    """
    import numpy as np
    import pandas as pd

    old = {"method": "B2", "N": 24, "tau_fit": 2.1, "seed": 0}
    new = dict(old, seed=1, **provenance.summary_provenance())
    df = pd.DataFrame([old, new])

    assert float(df[df.method == "B2"]["tau_fit"].iloc[0]) == 2.1
    assert np.isnan(df.loc[0, "git_commit"]) if isinstance(df.loc[0, "git_commit"], float) \
        else df.loc[0, "git_commit"] != df.loc[1, "git_commit"]
    assert df.loc[1, "git_commit"] == provenance.summary_provenance()["git_commit"]


# ------------------------------------------- config_param env-override additions

@pytest.mark.parametrize("env,expected", [
    ({"METRICS_T0": "7.5"}, ("METRICS_T0", 7.5)),
    ({"METRICS_T0": "garbage"}, ("METRICS_T0", 0.0)),
    ({}, ("METRICS_T0", 0.0)),
    ({"EXPERIMENT_REPRODUCIBLE": "False"}, ("EXPERIMENT_REPRODUCIBLE", False)),
    ({"EXPERIMENT_REPRODUCIBLE": "0"}, ("EXPERIMENT_REPRODUCIBLE", False)),
    ({}, ("EXPERIMENT_REPRODUCIBLE", True)),
])
def test_config_env_override(env, expected):
    """METRICS_T0 / EXPERIMENT_REPRODUCIBLE became env-overridable (campaign rule 3).

    Runs in a subprocess because config_param resolves overrides at import time.
    """
    name, value = expected
    child_env = dict(os.environ)
    for key in ("METRICS_T0", "EXPERIMENT_REPRODUCIBLE"):
        child_env.pop(key, None)
    child_env.update(env)
    child_env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, "-c", f"import config_param; print(repr(config_param.{name}))"],
        cwd=REPO_ROOT, env=child_env, capture_output=True, text=True, check=True,
    )
    assert proc.stdout.strip() == repr(value)
