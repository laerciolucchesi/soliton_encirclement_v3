# Radial and Tangential Control Laws (Current Implementation)

This document summarizes the radial and tangential control laws executed by each agent to achieve encirclement around a moving target. The equations map directly to the logic in [protocol_agent.py](protocol_agent.py) and [controllers.py](controllers.py), with parameters defined in [config_param.py](config_param.py).

The v3 controller carries **two tangential state channels** (`u_local`, `u_prop`) and adds an optional swarm-spin term broadcast by the target. Both extensions are described in sections 4–7 below.

## 1) Notation

All quantities are in the horizontal plane (XY), unless stated otherwise.

- Agent position: $p_i = (x_i, y_i)$
- Agent velocity: $v_i = (v_{x,i}, v_{y,i})$
- Target position: $p_T = (x_T, y_T)$
- Target velocity: $v_T = (v_{x,T}, v_{y,T})$

Define the target-centric radial vector:

$$
\mathbf r_i = p_i - p_T, \qquad r_i = \|\mathbf r_i\|.
$$

Unit vectors:

$$
\hat e_{r,i} = \frac{\mathbf r_i}{r_i}, \qquad \hat e_{\tau,i} = (-\hat e_{r,i,y}, \hat e_{r,i,x}).
$$

The effective radius used by the tangential mapping is:

$$
 r_{eff,i} = \max(r_i, R_{min}).
$$

Internal scalar states (per agent):

- $u_{local,i}$ — tangential drive accumulated from the local spacing error.
- $u_{prop,i}$  — tangential drive accumulated from the propagation channel (neighbour signals).
- $u_i$         — total tangential drive after channel composition (see §5).

The propagation layer also exposes a scalar **neighbour signal** $s_i$ (independent of $u_{prop,i}$), which is the layer's contribution that arrived from ring neighbours, excluding the agent's own self-injection. The available propagation mechanisms are listed in [propagation_layer.py](propagation_layer.py); $s_i$ corresponds to `get_neighbor_signal()`.

## 2) Radial control

Radial objective: $r_i \to R$.

Radial error (m):

$$
 e_r = r_i - R.
$$

Relative radial speed (w.r.t. target velocity):

$$
 v_r = (v_i - v_T) \cdot \hat e_{r,i}.
$$

Radial correction (m/s):

$$
 v_{r,corr} = -K_R\, e_r - K_{DR}\, v_r.
$$

Radial velocity contribution:

$$
 v_{rad} = v_{r,corr}\,\hat e_{r,i}.
$$

Implementation note: the PD law is realised by a `RadialDistanceController` (PID with $K_i=0$) over the measurement $r_i$ with setpoint $R$, see [controllers.py](controllers.py).

## 3) Tangential spacing error

Each agent selects a predecessor and successor in the target-centric angle order. Let the corresponding gaps be $\Delta\theta_{pred}$ and $\Delta\theta_{succ}$, both wrapped to $[0, 2\pi)$.

The local spacing error uses arc weights $\lambda_{pred}, \lambda_{self}$ broadcast by the target (`TargetState.alive_lambdas`):

$$
 e_\tau = \frac{\lambda_{pred}\,\Delta\theta_{succ} - \lambda_{self}\,\Delta\theta_{pred}}
 {\lambda_{pred}\,\Delta\theta_{succ} + \lambda_{self}\,\Delta\theta_{pred}}.
$$

If gaps are missing or degenerate, $e_\tau = 0$.

In the uniform-spacing case ($\lambda_{pred}=\lambda_{self}=1$) this reduces to the unweighted contrast. Setting `PROTECTION_ANGLE_DEG < 360` gives one agent an "edge lambda" so the formation reserves a protected/covered arc; the holder of that lambda is reassigned geometrically by the target (see `TargetProtocol._update_special_lambda_by_geometry`).

### Optional local omega damping

A local angular-rate damping term may be applied using neighbour estimates of $\omega$:

$$
 e_{\tau,eff} = e_\tau - K_{\omega}\,(\omega_i - \omega_{ref,i}^{local}),
$$

where $\omega_{ref,i}^{local}$ is averaged from the predecessor and successor angular rates around the target. When `K_OMEGA_DAMP = 0`, $e_{\tau,eff} = e_\tau$. Note that $\omega_{ref,i}^{local}$ here is **distinct** from the swarm-spin reference $\omega_{ref}$ broadcast by the target (§6); the former is a purely local low-pass on the spacing error, the latter is a global rotation setpoint.

## 4) Two-channel tangential state update

Each agent integrates two scalar states in parallel.

**Local channel** — driven by the agent's own spacing error:

$$
 u_{local,k+1} = u_{local,k} + \Delta t\left(-\beta_{local}\,u_{local,k} + K_{E\tau}\,e_{\tau,eff}\right).
$$

**Propagated channel** — driven by the neighbour signal $s_i$ from the active propagation layer:

$$
 u_{prop,k+1} = u_{prop,k} + \Delta t\left(-\beta_{prop}\,u_{prop,k} + K_{prop}\,s_i\right).
$$

Parameters: $\beta_{local}=$ `BETA_U_LOCAL`, $\beta_{prop}=$ `BETA_U_PROP`, $K_{E\tau}=$ `K_E_TAU`, $K_{prop}=$ `K_PROP` (chosen at runtime in `main.py`). Setting `BETA_U_LOCAL = BETA_U_PROP = BETA_U` and using the `baseline` propagation layer (which forces $s_i \equiv 0$) recovers the legacy single-state dynamics.

## 5) Channel composition

The total tangential drive $u_i$ is composed from the two channels:

- **Cooperative regime** (channels agree in sign, $u_{local} \cdot u_{prop} \ge 0$):

  $$ u_i = u_{local,i} + u_{prop,i}. $$

- **Conflict regime** (channels oppose in sign): a smooth dominance blend over width $W=$ `U_CONFLICT_BLEND_WIDTH`:

  $$ w = \tfrac{1}{2}\bigl(1 + \tanh((|u_{local}| - |u_{prop}|)/W)\bigr), $$

  $$ u_i = w\,u_{local,i} + (1-w)\,u_{prop,i}. $$

  When $W \to 0$ the blend reduces to the hard winner-takes-all rule
  $u_i = u_{local}$ if $|u_{local}|\ge|u_{prop}|$ else $u_i = u_{prop}$
  (legacy v2 behaviour).

Rationale: the local error term is already injected explicitly via $K_{E\tau}\,e_{\tau,eff}$ in the local channel, so the propagated channel must be fed by the **neighbour-only** signal $s_i$ (`get_neighbor_signal()`) to avoid double-counting. The smooth blend prevents discontinuous jumps when $|u_{local}|$ and $|u_{prop}|$ become nearly equal with opposing signs.

## 6) Mapping to tangential velocity and swarm spin

Tangential velocity from the composed drive:

$$
 v_{\tau,i} = (K_{\tau}\,u_i\,r_{eff,i})\,\hat e_{\tau,i}.
$$

This yields an induced angular rate $\omega \approx K_{\tau}\,u_i$ for $r_i > R_{min}$, independent of $R$.

**Swarm spin term.** When `TARGET_SWARM_SPIN_ENABLE` is true, the target broadcasts a desired swarm angular velocity $\omega_{ref}$ (rad/s) inside `TargetState`. It is generated by a wrapped-angle PD controller on the angle between the swarm's resultant unit vector (target → agents) and the target → adversary direction, with output saturated at $\pm$ `TARGET_SWARM_OMEGA_PD_MAX_ABS`. The error is gated by the Kuramoto order parameter: when $\rho<$ `TARGET_SWARM_SPIN_RHO_MIN` the swarm direction is ill-defined and the angular error is set to zero.

Each agent converts $\omega_{ref}$ into a tangential velocity contribution:

$$
 v_{spin,i} = (\omega_{ref}\,r_i)\,\hat e_{\tau,i}, \qquad r_i > 0.
$$

When the spin controller is disabled (or $\omega_{ref} = 0$), $v_{spin,i} = 0$.

## 7) Final command

The final commanded velocity is:

$$
 v_{cmd,i} = v_T + v_{rad,i} + v_{\tau,i} + v_{spin,i}.
$$

It is then clamped component-wise to the mobility envelope `VM_MAX_SPEED_XY` (norm of $(v_x, v_y)$) and `VM_MAX_SPEED_Z` (absolute value of $v_z$) before being passed to `VelocityMobilityHandler.set_velocity(...)`. The mobility handler additionally enforces acceleration limits and (optionally) a 1st-order velocity-tracking time constant — see [README.md](README.md) for the mobility model.

## 8) Parameter reference

| Symbol | Constant in `config_param.py` | Default | Role |
|---|---|---|---|
| $K_R$ | `K_R` | 1.0 | Radial proportional gain (1/s) |
| $K_{DR}$ | `K_DR` | 0.5 | Radial derivative gain (dimensionless) |
| $K_{E\tau}$ | `K_E_TAU` | 25.0 | Spacing-error injection gain into $u_{local}$ |
| $K_{prop}$ | `K_PROP` (runtime) | menu | Propagation-channel gain (set in `main.py`) |
| $\beta_{local}$ | `BETA_U_LOCAL` | 7.0 | Damping of $u_{local}$ |
| $\beta_{prop}$ | `BETA_U_PROP`  | 7.0 | Damping of $u_{prop}$  |
| $K_{\tau}$ | `K_TAU` | 0.2 | Tangential velocity scaling |
| $K_{\omega}$ | `K_OMEGA_DAMP` | 0.1 | Local angular-rate damping (0 disables) |
| $W$ | `U_CONFLICT_BLEND_WIDTH` | 0.2 | Channel-conflict blend width (0 = legacy hard switch) |
| $R$ | `ENCIRCLEMENT_RADIUS` | 20.0 m | Target encirclement radius |
| $R_{min}$ | `R_MIN` | 1.0 m | Minimum effective radius for $r_{eff}$ |
| $\Delta t$ | `CONTROL_PERIOD` | 0.01 s | Discrete integration step |
| $\omega_{ref}$ saturation | `TARGET_SWARM_OMEGA_PD_MAX_ABS` | 1.0 rad/s | Spin output bound |
| $\rho_{min}$ | `TARGET_SWARM_SPIN_RHO_MIN` | 0.05 | Kuramoto gate for spin error |
