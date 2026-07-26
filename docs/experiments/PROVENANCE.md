# Run provenance

Every simulation run describes itself: which code produced it, with which seed,
under which parameters. This page defines the schema, the operational rule, and
how to audit what already exists.

## Why this exists

The main campaign's result CSVs (`experiments/scaling_law/*.csv`, produced
between 2026-05-30 and 2026-06-06) were generated from an **uncommitted working
tree** and record neither the seed, nor the code version, nor the parameters that
were pinned for the run. Those numbers cannot be re-derived: the tree that
produced them no longer exists anywhere, and the defaults in `config_param.py`
have moved since (`CONTROL_PERIOD` 0.01 → 0.05, `AGENT_STATE_TIMEOUT` 5·dt →
max(20·dt, 0.2), `DUAL_PULSE_CONSUME_FF_ONLY` and `DUAL_PULSE_MULTIPLICITY`
default ON). A row without provenance is a number without an experiment.

Run `python experiments/scaling_law/check_provenance.py` for the current
inventory of what is and is not reproducible.

## The operational rule

**Commit before you run.** A result generated in a dirty tree is not
reproducible from its recorded commit, and the row will say so: `git_dirty=True`
is a permanent mark on that data point, and `main.py` prints a warning at
startup. Treat a dirty-tree result as a scratch measurement, never as campaign
evidence.

The `git_dirty` test is exactly `git status --porcelain` reporting any entry —
staged, unstaged, or untracked-and-not-ignored. That is deliberately the same
check you run by hand before starting a campaign, so the flag and the rule cannot
disagree. Per-run artefacts (`agent_telemetry.csv`, `*.png`, `*_runs/`,
`run_manifest.json`, `runs_summary*.csv`) are gitignored and therefore never make
the tree look dirty on their own.

## What gets written

### 1. `run_manifest.json` — one per run directory

Written by `main.py` **before** the simulation starts. This is the source of
truth whenever a CSV row and reality seem to disagree.

```json
{
  "schema": "run_manifest/1",
  "run_timestamp_iso": "2026-07-26T14:03:11",
  "argv": ["C:/.../main.py"],
  "cwd": "C:/.../collapse_runs/dual_pulse_N24_tau1_dt0.05_s0",
  "python_version": "3.11.9",
  "platform": "win32",
  "git": {
    "commit": "fc08491", "dirty": false,
    "commit_full": "fc084919...", "branch": "main",
    "describe": "fc08491", "status_porcelain": "", "repo_root": "C:/..."
  },
  "env_overrides": { "NUM_AGENTS": "24", "CONTROL_PERIOD": "0.05", "...": "..." },
  "resolved_config": { "AGENT_STATE_TIMEOUT": 1.0, "...": "..." }
}
```

* `resolved_config` — **every** public constant of `config_param` after env
  overrides were applied at import time. Not a curated subset: if a parameter
  exists, it is here.
* `env_overrides` — only the variables actually **set** for this process. This is
  what distinguishes "pinned by the runner" (campaign rule 3) from "inherited
  from the default".
* `status_porcelain` — the raw dirty-file list, so `git_dirty=true` is
  explainable after the fact. Truncated at 20 000 chars.

Capture timing matters: the manifest is written **before** the simulation so the
git state is that of the tree that *produced* the run, not the post-run tree that
the run's own outputs may have dirtied. Writing it also primes a process-wide
cache, so the summary row appended after the run reports the same pre-run state.

### 2. `runs_summary.csv` — one row per run

`plot_telemetry.SUMMARY_COLUMNS` is assembled as:

| block | columns | source |
|---|---|---|
| run identity | `run_timestamp_iso`, `propagation_method`, `k_prop` | env set by `main.py` |
| provenance | `git_commit`, `git_dirty`, `experiment_seed` | `provenance.git_provenance()` / `EXPERIMENT_SEED` |
| pinned parameters | `control_period`, `k_e_tau`, `composition_mode`, `num_agents`, `encirclement_radius`, `sim_duration`, `protection_angle_deg`, `dual_pulse_integration`, `dual_pulse_delta_scale`, `dual_pulse_ttl_hops`, `dual_pulse_t_ff`, `dual_pulse_multiplicity`, `dual_pulse_consume_ff_only`, `communication_delay`, `communication_failure_rate`, `communication_range`, `agent_state_timeout`, `broadcast_repeats`, `deterministic_failure_enable`, `deterministic_failure_agent_id`, `deterministic_failure_agent_ids`, `deterministic_failure_time`, `deterministic_failure_off_time`, `failure_enable`, `failure_mean_per_min`, `experiment_reproducible`, `vm_tau_xy`, `vm_max_speed_xy`, `target_motion_speed_xy`, `target_swarm_spin_enable`, `target_swarm_omega_ref`, `init_radius_range`, `init_angles_equidistant` | resolved `config_param` |
| metric window | `metrics_t0`, `metrics_e_thr`, `metrics_ma_w_sec`, `metrics_settle_window_sec` | the `MetricParams` actually used |
| metrics | `M1_P95_e_pooled` … `M7_settled_frac` | `compute_metrics()` |

Values are read from the **resolved** `config_param` module object at run time,
through the single mapping table `provenance.SUMMARY_FROM_CONFIG`. There is no
second dictionary of literal values that could drift from the defaults it claims
to mirror. Adding a parameter to the row means adding one entry to that table.

`append_run_summary` cross-checks the assembled row against `SUMMARY_COLUMNS` and
prints a loud warning on any mismatch — a silently blank provenance cell is the
exact failure this schema exists to prevent.

Three of the column names differ from the underlying constant (the short form is
the CSV column, the long form is the config name):

| column | `config_param` constant |
|---|---|
| `communication_range` | `COMMUNICATION_TRANSMISSION_RANGE` |
| `broadcast_repeats` | `DUAL_PULSE_BROADCAST_REPEATS` |
| `deterministic_failure_time` | `DETERMINISTIC_FAILURE_TIME_T0` |

`experiment_seed` has **no** `config_param` constant: `main.py` reads
`EXPERIMENT_SEED` from the environment directly. It is recorded unconditionally;
when the adjacent `experiment_reproducible` column is `False`, the seed was
parsed but never applied.

### Schema rotation

`SUMMARY_COLUMNS` changed with this schema, so the first run in a directory that
still holds an old `runs_summary.csv` **rotates** it to
`runs_summary.csv.bak.<timestamp>` and starts a fresh file. Nothing is deleted
(campaign rule 2); the rotated file is gitignored.

## How a sweep runner stamps its rows

The sweep runners in `experiments/scaling_law/` write their own
`*_results.csv` — one row per cell — and those rows need provenance too
(campaign rule 5).

```python
from metrics_util import event_metrics, run_provenance

m = event_metrics(pd.read_csv(tgt), T0)
m.update({"method": tag, "N": n, "seed": seed, ...})
m.update(run_provenance(run_dir))     # git_commit, git_dirty, experiment_seed, params
```

`run_provenance(run_dir)` reads the `run_manifest.json` that the **child**
process wrote into `run_dir`.

> **Do not** call `provenance.summary_provenance()` from a runner. The runner is
> the *parent* process: its own `config_param` holds the parent's defaults, not
> the values it pinned in the child's `env` dict. Stamping parent values onto a
> child's result row records parameters the run never used — precisely the
> failure mode this whole mechanism exists to prevent.

`run_provenance` returns `{}` when the manifest is absent (an older run, or a
`main.py` predating this schema), so "not recorded" stays distinguishable from
"recorded as default".

## Auditing what already exists

```powershell
python experiments/scaling_law/check_provenance.py              # top-level CSVs
python experiments/scaling_law/check_provenance.py --all        # includes _archive/
python experiments/scaling_law/check_provenance.py --verbose    # field by field
```

It reads only each file's header plus a line count (campaign rule 7: never load a
telemetry-sized CSV into memory) and reports, per file, how many of the three
core fields (`experiment_seed`, `git_commit`, `git_dirty`) and how many of the
eleven key parameters are present. Historical short column names are accepted as
aliases (`seed`, `N`, `dt`, `tau_xy`, `T_FF`, …), so the report credits what the
runners already recorded.

A missing field in an old CSV is **not** a bug to fix by rewriting the file.
Campaign rule 2: re-run and write a **new** file, move the old one to
`experiments/scaling_law/_archive/`, and add a line to
[CAMPAIGN_LOG.md](CAMPAIGN_LOG.md) saying which result was superseded and why.

## Related

* [README.md](README.md) — metric definitions and the reporting rule
  (median + min + max + n, never central tendency alone).
* [CAMPAIGN_LOG.md](CAMPAIGN_LOG.md) — dated log of what each campaign run
  changed.
* `provenance.py` — the implementation. `python provenance.py` prints the current
  git state, the number of resolved constants, and which env overrides are set.
