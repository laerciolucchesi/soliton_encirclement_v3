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
tangential spacing controller. Seven mechanisms are available
(`baseline`, `advection`, `wave`, `excitable`, `kdv`, `alarm`, `burgers`),
selected interactively at the start of each run. The thesis context is
soliton-inspired information propagation around the swarm ring, hence the
repo name.

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
├── propagation_layer.py         # ABC + 7 layer implementations + factory
├── plot_telemetry.py            # Per-node plots and 7 scalar metrics (M1..M7)
├── pyproject.toml               # Editable install; src/ is the package root
├── README.md, CONTROLE.md       # User documentation; control-law derivations
├── src/
│   ├── velocity_mobility/       # Reusable velocity-driven mobility handler
│   └── gradysim_velocity_mobility/  # Back-compat shim — re-exports velocity_mobility
├── demos/velocity_mobility/     # Standalone mobility demo (single node)
├── examples/                    # Core-only (no GrADyS runtime) examples
└── tests/                       # pytest: test_controllers, test_core_*, test_propagation
```

CSV telemetry (`agent_telemetry.csv`, `target_telemetry.csv`) and metric
PNGs are written next to the run's working directory.

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
python main.py                    # still prompts unless you stub stdin
```

`main.py` always re-reads stdin via `_select_propagation_method()`, so true
non-interactive batch runs require either piping a choice into stdin or
patching the function. Keep this in mind before suggesting automation.

Tests: `python -m pytest` (configured in `pyproject.toml`,
`addopts = "-v --tb=short"`).

## Architecture notes

### Two-package install layout

`pyproject.toml` declares `package-dir = {"" = "src"}` and only ships
`velocity_mobility` and `gradysim_velocity_mobility`. The protocol files
at the repo **root** (`protocol_agent.py`, `controllers.py`,
`propagation_layer.py`, `config_param.py`, …) are **not** part of the
installed package — they are imported by `main.py` because it sits next to
them, and `main.py` injects `src/` into `sys.path` at startup.

Implication: `from controllers import ...` only works when the cwd is the
repo root or when `sys.path` is set up the way `main.py` does it.
Tests handle this with `sys.path.insert(0, repo_root)`.

### Configuration discipline

`config_param.py` is the **single source of truth**. Adding a new tunable
means: define the constant there, import it where needed, and document it
in the section header comments. Do not hardcode magic numbers in
protocols. The file is organized into 11 numbered sections; keep that
structure when extending.

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

### Failure injection

Each agent independently draws Bernoulli trials every
`FAILURE_CHECK_PERIOD` seconds with rate
`FAILURE_MEAN_FAILURES_PER_MIN / 60`. On failure: timer cancelled,
velocity zeroed, node painted red, recovery scheduled in
`FAILURE_OFF_TIME` seconds. The target never fails. The propagation
layer's `on_reset()` is invoked on recovery so dynamic fields don't
restart with stale state.

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
delta_u, du_damp, du_from_e_tau, e_tau, e_tau_eff, velocity_norm
```

`target_telemetry.csv` columns (written by `TargetProtocol.finish()`):

```
timestamp, E_r, E_vr, rho, G_max, E_gap
```

`main.py` deletes `agent_telemetry.csv` before each run and creates
`target_telemetry.csv` with a header. **Do not** pre-create
`agent_telemetry.csv` — `AgentProtocol.finish()` only writes the header
when the file does not exist. Pre-creating it produces a header-less file.

## Conventions and gotchas

- **Path with diacritics.** The repo lives at
  `…\PUC\Laércio - Doutorado\12 Códigos\soliton_encirclement_v3`. Always
  quote paths in shell commands (PowerShell or bash). Globs work fine.
- **Primary shell is PowerShell** (Windows). The README's PowerShell
  snippets are canonical. Bash is available via the Bash tool but use
  `python -m pytest` not `pytest` to ensure the right interpreter.
- **Activation script path** in README has a stray leading backslash
  (`\.venv\Scripts\Activate.ps1`); the correct invocation is
  `.\.venv\Scripts\Activate.ps1`.
- **Don't mass-edit telemetry PNGs / CSVs.** Many tracked PNGs are
  regenerated on every run (`git status` after a sim run will show them
  as modified). Avoid committing them unless the change is intentional.
- **`gradysim_velocity_mobility` is a back-compat shim only.** New code
  should import from `velocity_mobility`.
- **Numerical stability of propagation layers.** Each layer documents its
  CFL / stiffness assumptions in its docstring. The `excitable` (FHN)
  layer uses 4 internal RK1 substeps per control tick because
  $1/\epsilon = 12.5$ is stiff at `dt = 0.01`. Don't reduce substeps
  without reproducing the propagation tests.
- **Hysteresis in neighbour selection.** `HYSTERESIS_RAD` (radians)
  prevents predecessor/successor flapping when two agents are nearly
  equidistant in angle. Removing or lowering it can break the spacing
  controller in dense formations.

## What's intentionally out of scope here

- The KdV `.m`, `.mat`, `.mp4` files at the repo root are MATLAB
  reference material from earlier exploration of soliton dynamics.
  They are not used by the Python simulation.
- `metrics.pptx`, `equacoes_controle_tangencial*.md`, and
  `Ideias relacionadas ao projeto soliton.md` are working notes for the
  thesis; treat them as read-only context unless asked to update them.
