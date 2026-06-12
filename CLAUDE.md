# CLAUDE.md

Orientation for AI assistants working in this repository. The user-facing
documentation is in [README.md](README.md); this file captures the
information you need to navigate, modify, and run the code without
re-discovering the layout each session.

## What this project is

`soliton_encirclement_v3` is a research codebase for **distributed swarm
encirclement** experiments built on top of
[GrADyS-SIM NG](https://github.com/Project-GrADyS/gradys-sim-nextgen).
A target node moves in the XY plane and `NUM_AGENTS` agents must surround
it at radius `ENCIRCLEMENT_RADIUS` while keeping a desired angular spacing.

The defining feature of the **v3** branch is a pluggable
**propagation layer** that adds a second control channel (`u_prop`) to the
tangential spacing controller. **Eight mechanisms** are available
(`baseline`, `advection`, `wave`, `excitable`, `kdv`, `alarm`, `burgers`,
`dual_pulse`), selected interactively at the start of each run. The
thesis context is soliton-inspired information propagation around the
swarm ring, hence the repo name.

`dual_pulse` is the **flagship method** — a discrete soliton-inspired
mechanism with counter-propagating pulses, hop-count topology discovery,
and a separate "Option A" integration that biases the spacing-error gaps
seen by the controller (it does NOT feed `u_prop` like the other layers).
See its dedicated section below.

## Top-level layout

```
soliton_encirclement_v3/
├── main.py                      # Simulation builder + interactive menu
├── config_param.py              # Single source of truth for ALL parameters
├── protocol_agent.py            # AgentProtocol — distributed controller
├── protocol_target.py           # TargetProtocol — broadcast + metrics + spin PD
├── protocol_adversary.py        # AdversaryProtocol — random roaming intruder
├── protocol_messages.py         # AgentState, TargetState, AdversaryState (JSON)
├── controllers.py               # Radial PD, Wrapped-angle PD, Tangential 2-channel
├── propagation_layer.py         # ABC + 7 e_tau-driven layers + factory
├── dual_pulse_layer.py          # 8th layer: hop-count discrete pulses (Option A)
├── plot_telemetry.py            # Per-node plots and 7 scalar metrics (M1..M7)
├── pyproject.toml               # Editable install; src/ is the package root
├── README.md, CONTROLE.md       # User documentation; control-law derivations
├── src/
│   └── velocity_mobility/       # Reusable velocity-driven mobility handler
├── demos/velocity_mobility/     # Standalone mobility demo (single node)
├── examples/                    # Core-only (no GrADyS runtime) examples
└── tests/                       # pytest: test_controllers, test_core_*,
                                 #         test_propagation, test_dual_pulse,
                                 #         test_damped_advection
```

CSV telemetry (`agent_telemetry.csv`, `target_telemetry.csv`,
`events.csv`) and metric PNGs are written next to the run's working
directory.

## Running the simulation

The primary entry point is interactive:

```powershell
python main.py
```

It prints a numbered menu of propagation methods and prompts for `K_PROP`
(the propagation-channel gain). For non-interactive runs (CI, batch
sweeps), bypass the menu by exporting the same env vars `main.py` sets
internally:

```powershell
$env:PROPAGATION_METHOD = "kdv"
$env:PROPAGATION_K_PROP = "1.0"
$env:PROPAGATION_PARAMS = "{}"   # JSON dict, optional per-method overrides
python main.py                    # menu prompt is fully bypassed when env is set
```

When `PROPAGATION_METHOD` is set in the environment, `_select_propagation_method()`
returns immediately without reading stdin — true non-interactive batch
runs work without piping anything.

For `dual_pulse` and `baseline`, `K_PROP` is irrelevant (neither feeds
`u_prop`); the menu and the env path skip the K_PROP prompt for those
two methods (see `_METHODS_WITHOUT_K_PROP` in `main.py`).

### Env-var overrides for sweeps

Most tunables in `config_param.py` accept env-var overrides applied at
import time. The most useful for batch experiments:

```powershell
# Scenario controls (newly added — for randomized init / target motion sweeps)
$env:INIT_RADIUS_RANGE        = "0.1"     # ±10% radius scatter
$env:INIT_ANGLES_EQUIDISTANT  = "False"   # random initial angles
$env:TARGET_MOTION_SPEED_XY   = "4.0"     # m/s; 0.0 = stationary

# dual_pulse tuning
$env:DUAL_PULSE_DELTA_SCALE       = "0.5"   # global delta scale
$env:DUAL_PULSE_RAMP_TICKS        = "4"     # smoothing ticks
$env:DUAL_PULSE_ALPHA_CLOSE_RATIO = "0.7"   # immediate-neighbor attenuation
$env:DUAL_PULSE_ALPHA_CURVE_POWER = "1.0"   # alpha shape exponent (1=linear)
$env:DUAL_PULSE_SLEEP_THRESHOLD   = "0.01"  # rad — skip bias below this
```

Tests: `python -m pytest` (configured in `pyproject.toml`,
`addopts = "-v --tb=short"`).

## Architecture notes

### Package install layout

`pyproject.toml` declares `package-dir = {"" = "src"}` and only ships
`velocity_mobility`. The protocol files
at the repo **root** (`protocol_agent.py`, `controllers.py`,
`propagation_layer.py`, `dual_pulse_layer.py`, `config_param.py`, …)
are **not** part of the installed package — they are imported by `main.py`
because it sits next to them, and `main.py` injects `src/` into `sys.path`
at startup.

Implication: `from controllers import ...` only works when the cwd is the
repo root or when `sys.path` is set up the way `main.py` does it.
Tests handle this with `sys.path.insert(0, repo_root)`.

### Configuration discipline

`config_param.py` is the **single source of truth**. Adding a new tunable
means: define the constant there, import it where needed, and document it
in the section header comments. Do not hardcode magic numbers in
protocols.

`EXPERIMENT_REPRODUCIBLE` (default `True`) seeds `random` and the per-agent
failure RNGs deterministically. Tests and runs that compare metrics
across propagation methods rely on this — be cautious about removing it.

### Two-channel tangential controller

`TangentialSpacingController.update()` maintains two scalar states:

- `u_local`: driven by the local spacing error `e_tau` (gain `K_E_TAU`,
  damping `BETA_U_LOCAL`).
- `u_prop`: driven by `k_prop * prop_signal` from the propagation layer
  (damping `BETA_U_PROP`).

Composition: cooperative sum when channels agree in sign; smooth
dominance blend (`tanh` over width `U_CONFLICT_BLEND_WIDTH`) when they
conflict. Setting `U_CONFLICT_BLEND_WIDTH = 0.0` reproduces the legacy
hard winner-takes-all behaviour. Tests in
`tests/test_controllers.py` lock in the numerical dynamics — touch them
only with the user's agreement.

### Propagation layer contract

Subclasses of `PropagationLayer` (in `propagation_layer.py`) must
implement:

- `update(e_tau, dt, pred_state, succ_state)` — Euler step using broadcast
  state dicts from ring neighbours.
- `get_signal()` — full local state including self-injection (telemetry).
- `get_neighbor_signal()` — **only** what arrived from neighbours, no
  self-injection. This is what feeds `u_prop` to avoid double-counting
  the local error term.
- `get_broadcast_state()` — fields included in `AgentState.prop_state`.
- `on_neighbor_change()` / `on_reset()` — invoked when ring topology
  changes or an agent recovers from failure.

Adding a new mechanism: implement the class, register it in
`_REGISTRY` at the bottom of `propagation_layer.py`, and add an entry to
`_METHODS` in `main.py` so the menu can offer it.
`tests/test_propagation.py` runs five standard tests (decay, propagation,
stability, missing-neighbour robustness, reset) against every registered
method — new layers should pass them.

`dual_pulse` is the exception: it is registered via a lazy import in
`create_propagation_layer` (avoids a circular dependency) and is
**deliberately excluded** from `test_propagation.py`'s `ALL_METHODS`
because its semantics (event-triggered hop-count pulses, gap-bias
integration) don't match the standard tests. It has its own
`tests/test_dual_pulse.py` suite (~17 tests).

### Dual Pulse propagation layer (v1.7)

Implemented in `dual_pulse_layer.py` (separate from `propagation_layer.py`
to keep the abstract base class clean). Selected via menu option 7 or
`PROPAGATION_METHOD=dual_pulse`.

**Algorithm.** When a topology event happens (failure or recovery), the
canonical originator (the dead/recovered drone's predecessor) injects two
counter-propagating pulses (CCW and CW) tagged with an `event_id`,
`event_type` (SAIDA / ENTRADA), `hop_count`, and (for ENTRADA only) the
`recovered_id`. Every receiver records the pulse's hop count and forwards
it. Once both directions have been seen by a receiver, that node knows
its own hop position relative to the dead/recovered drone and can compute
the angular shift `delta_D` it should apply to redistribute uniformly in
the new ring size.

The shift is **not** fed into `u_prop`; instead it modifies the
`pred_gap` / `succ_gap` measurements that `compute_e_tau_used` consumes
("Option A integration"). This lets the controller's existing dynamics
drive the redistribution motion rather than overlaying a second control
channel.

**SAIDA δ formula (receiver):**
```
delta_D = (h_CCW - N_old/2) * (gap_new - gap_old) * scale
```
where `N_old = h_CCW + h_CW + 1`, `gap_old = 2π/N_old`, `gap_new = 2π/N_new`,
and `N_new = N_old - 1`.

**ENTRADA δ formula (receiver):** same shape but `h_anchor = 1 + N_new/2`,
`N_new = h_CCW + h_CW + 1`, `N_old = N_new - 1`. The recovered drone D
itself runs in **passthrough** mode (forwards the pulse but skips the
self-shift) because its physical position is the equilibrium one already.

**Originator self-shift.** A's own pulse never reaches A through the
relay path (refractory cache blocks). Instead the originator detects its
own returning pulse via `_self_originated[event_id]`, reads `N_new` from
the returning pulse's `hop_count` (full ring traversal), and applies a
special formula:
- SAIDA originator: `delta_orig = gap_old - gap_new/2`
- ENTRADA originator: `delta_orig = gap_old/2 - gap_new` (sign-inverted)

**Coordination v1 simplification.** Only the canonical (`min(self.id,
partner.id)`) injects. The non-canonical side stays silent. If the canonical
fails between detection and injection, the event is lost — accepted in v1.

**v1.7 tunable knobs** (all in `config_param.py`):
- `DUAL_PULSE_TTL_HOPS=50` — circulation safety.
- `DUAL_PULSE_GAP_CLIP_FRAC=0.8` — keeps virtual gaps positive.
- `DUAL_PULSE_MIN_RING_SIZE=3` — below this, skip the math.
- `DUAL_PULSE_DELTA_SCALE=0.5` — global δ scale (1.0 = analytical formula).
- `DUAL_PULSE_RAMP_TICKS=4` — lerp ramp on shift accumulation (~40ms).
- `DUAL_PULSE_ALPHA_CLOSE_RATIO=0.7` — attenuation at the immediate
  neighbours of D (1.0 = no attenuation).
- `DUAL_PULSE_ALPHA_CURVE_POWER=1.0` — interpolation exponent for the
  alpha curve (1.0 = linear; >1.0 concentrates attenuation near D).
- `DUAL_PULSE_SLEEP_THRESHOLD=0.01` — when `|shift_remaining|` is below
  this, the gap-bias step is skipped entirely (sleeping mode). Prevents
  subtle controller-vs-tracking interference between events.

**Trigger gates** (in `protocol_agent.py` control loop):
- Inject only when `succ_changed AND not in_warmup`.
- SAIDA gate: also require `alive_count` decreased since last tick.
- ENTRADA gate: require `alive_count` increased AND `recovered_id =
  neighbor_succ_id`.
- Warmup window (`FAST_CHANNEL_WARMUP_SEC=1.0s`) silences both injection
  paths during initial neighbor discovery.

**Pulse transport robustness.** Each outgoing pulse is broadcast
`BROADCAST_REPEATS=2` consecutive ticks (constant in `DualPulseLayer`)
to defend against the GrADyS-SIM intra-tick agent firing order, which
can cause a sender's prop_state to be overwritten before the receiver
reads it. Receivers' refractory cache filters duplicates.

### Failure injection

Each agent independently draws Bernoulli trials every
`FAILURE_CHECK_PERIOD` seconds with rate
`FAILURE_MEAN_FAILURES_PER_MIN / 60`. On failure: timer cancelled,
velocity zeroed, node painted red, recovery scheduled in
`FAILURE_OFF_TIME` seconds. The target never fails. The propagation
layer's `on_reset()` is invoked on recovery so dynamic fields don't
restart with stale state.

**Warmup suppression** (added to fix observable "non-equidistant start"
artefact): failures are NOT injected during the first
`FAST_CHANNEL_WARMUP_SEC` seconds of the simulation. With seeds where
Poisson would have fired in the first second, this avoids an event that
no propagation layer (dual_pulse / fast_layer) can handle anyway because
both are gated by the same warmup, leaving a transient that looks like
"agents not equidistant at t=0".

### Edge / non-uniform spacing (`PROTECTION_ANGLE_DEG`)

The target broadcasts a `lambda` weight per agent in `TargetState.alive_lambdas`.
At equilibrium each arc size is proportional to its lambda. One agent
holds an "edge lambda" derived from `PROTECTION_ANGLE_DEG`; this
implements arbitrary protected/covered arcs without changing $N$. The
holder is reassigned geometrically (predecessor of the largest gap) with
hysteresis and a 1 s cooldown to prevent chattering. See
`protocol_target._update_special_lambda_by_geometry`.

### Swarm spin controller

When `TARGET_SWARM_SPIN_ENABLE=True`, the target runs a `WrappedAnglePDController`
on the angle between the swarm's resultant unit vector and the
target → adversary direction, and broadcasts the resulting `omega_ref`
inside `TargetState`. Agents add `omega_ref * r` along $\hat t$ to the
commanded velocity. When the swarm is nearly uniformly distributed
(Kuramoto $\rho < $ `TARGET_SWARM_SPIN_RHO_MIN`) the angular error is
disabled to avoid arbitrary direction bias.

## Telemetry contract

`agent_telemetry.csv` columns (written by `AgentProtocol.finish()`):

```
node_id, timestamp, dt_u, u, u_local, u_prop, u_ss, prop_signal,
delta_u, du_damp, du_from_e_tau,
e_tau, e_tau_eff, e_tau_real,
velocity_norm,
u_R, u_L, fast_signal,
dual_pulse_shift, dual_pulse_target,
theta_rel
```

Notes:
- `e_tau`, `e_tau_eff`: as seen by the controller. With `dual_pulse` active
  these reflect the **virtual** gaps (real gaps biased by `shift_remaining`).
- `e_tau_real`: the **physical** spacing error computed from unmodified
  gaps. Identical to `e_tau` for non-dual_pulse methods. **Use this for
  cross-method M1..M7 comparisons.** `plot_telemetry.py` automatically
  prefers `e_tau_real` when present.
- `dual_pulse_shift`: applied (post-ramp) shift; what the controller saw.
- `dual_pulse_target`: ideal accumulated shift (pre-ramp); useful to
  diagnose residual at end-of-run (`shift == target` ⇒ ramp finished).
- `u_R`, `u_L`, `fast_signal`: observational fast-channel
  (`DampedAdvectionLayer`) — runs in parallel with the main `prop_layer`
  and is logged for analysis. Does NOT feed `u_total` (Phase A semantics).

**Telemetry skip.** A row is NOT written if `target_state is None`
(typically only the very first 10 ms of the run for late-initialized
agents). Without target_state, `theta_rel` would default to 0.0 and
mislead downstream plots — better to drop than to lie.

`target_telemetry.csv` columns (written by `TargetProtocol.finish()`):

```
timestamp, E_r, E_vr, rho, G_max, E_gap
```

`events.csv` schema (extended for v1.7):

```
timestamp, node_id, event_type, amplitude,
event_id, h_CCW, h_CW, N_new
```

Event types written by `AgentProtocol`:
- `failure_start`, `failure_end` — outage start/end (existing).
- `pulse_injected` — fast_layer (DampedAdvectionLayer) pulse injection.
- `dual_pulse_self_shift_saida`, `dual_pulse_self_shift_entrada` —
  originator's own δ_orig applied on returning pulse.
- `dual_pulse_event_completed_saida`, `dual_pulse_event_completed_entrada` —
  receiver's δ_D applied after both directions arrived.

The `event_id` is encoded as `"originator_id_seq"` (e.g. `"7_3"`).
Dual-pulse-specific fields are blank for older event types.

`main.py` deletes `agent_telemetry.csv` and `events.csv` before each run
and creates `target_telemetry.csv` with a header. **Do not** pre-create
`agent_telemetry.csv` or `events.csv` — `AgentProtocol.finish()` only
writes the header when the file does not exist. Pre-creating produces
header-less files.

## Conventions and gotchas

- **Path with diacritics.** The repo lives at
  `…\PUC\Laércio - Doutorado\12 Códigos\soliton_encirclement_v3`. Always
  quote paths in shell commands (PowerShell or bash). Globs work fine.
- **Primary shell is PowerShell** (Windows). The README's PowerShell
  snippets are canonical. Bash is available via the Bash tool but use
  `python -m pytest` not `pytest` to ensure the right interpreter.
- **PowerShell locale and decimals.** Brazilian PowerShell formats floats
  with comma decimals when interpolating into strings. When using
  `"$alpha"` in a `Copy-Item -Path ... ("..._{0}.csv" -f $alpha)` pattern,
  expect file names like `alpha0,5.csv`. Either use English-locale strings
  or pre-replace `,` → `_` before file naming.
- **PowerShell here-strings.** `python -c "@..."` does not work in
  PowerShell because `@` is parsed differently. Write the script to a
  temp file with `Write-Output > tmp.py` or use the `Bash` tool instead.
- **stdout encoding crashes.** When piping `python main.py` through
  `Tee-Object` or PowerShell redirection, the default cp1252 encoding
  fails on Unicode characters (`→`, `≈`, …). Set
  `$env:PYTHONIOENCODING = "utf-8"` first.
- **Activation script path** in README has a stray leading backslash
  (`\.venv\Scripts\Activate.ps1`); the correct invocation is
  `.\.venv\Scripts\Activate.ps1`.
- **Don't mass-edit telemetry PNGs / CSVs.** Many tracked PNGs are
  regenerated on every run (`git status` after a sim run will show them
  as modified). Avoid committing them unless the change is intentional.
- **Numerical stability of propagation layers.** Each layer documents its
  CFL / stiffness assumptions in its docstring. The `excitable` (FHN)
  layer uses 4 internal RK1 substeps per control tick because
  $1/\epsilon = 12.5$ is stiff at `dt = 0.01`. Don't reduce substeps
  without reproducing the propagation tests.
- **Hysteresis in neighbour selection.** `HYSTERESIS_RAD` (radians)
  prevents predecessor/successor flapping when two agents are nearly
  equidistant in angle. Removing or lowering it can break the spacing
  controller in dense formations.
- **`dual_pulse` testing.** Tests of the raw δ algebra in
  `tests/test_dual_pulse.py` use the `no_hop_alpha` pytest fixture to
  force `ALPHA_CLOSE_RATIO=1.0`, isolating the formula from the
  hop-attenuation feature (which has its own `test_hop_alpha_*` tests).
  Add this fixture as an argument when writing new algebra tests.
- **Residual `dual_pulse_shift` at end of run** is a measurement artefact,
  not a bug. Events fire up to ~0.6 s before sim end and need ~5-10 s of
  physical motion to fully consume. Algorithm has been verified correct
  via `cum_delta` accounting (see investigation log).

## What's intentionally out of scope here

- The thesis material and reference binaries have been **moved out of the
  repository** to a sibling local archive (`../_soliton_v3_local_archive/`)
  to keep the public repo lean. That archive holds: the Portuguese thesis
  drafts (former `docs/` tree, `equacoes_controle_tangencial*.md`,
  `Ideias relacionadas ao projeto soliton.md`), the MATLAB `.m`/`.mat`/`.mp4`
  KdV reference material, `metrics.pptx`, and the orphaned figure/diagnostic
  scripts (`plot_limiter_soft.py`, `diagnose_signs.py`). None of these are
  used by the Python simulation. The control-law reference that stays public
  is `CONTROLE.md`.
- The other 6 propagation methods (`advection`, `wave`, `excitable`,
  `kdv`, `alarm`, `burgers`) are kept in the codebase for completeness
  but are documented as **earlier failed attempts** by the user. The
  `dual_pulse` method is the one used in current research; do not
  benchmark dual_pulse against the other 6 unless explicitly asked.
- ENTRADA-with-failed-canonical recovery: when a canonical originator is
  itself failed at the moment of a recovery event, the ENTRADA is silently
  missed (~3/24 in dense Phase 3 runs). v2 could add a successor-side
  fallback timer; not implemented.
