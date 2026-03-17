# Radial and Tangential Control Laws (Current Implementation)

This document summarizes the radial and tangential control laws executed by each agent to achieve encirclement around a moving target. The equations map directly to the logic in `protocol_agent.py`, with parameters defined in `config_param.py`.

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
 v_{r,corr} = -K_R e_r - K_{DR} v_r.
$$

Radial velocity contribution:

$$
 v_{rad} = v_{r,corr} \hat e_{r,i}.
$$

## 3) Tangential spacing error

Each agent selects a predecessor and successor in the target-centric angle order. Let the corresponding gaps be $\Delta\theta_{pred}$ and $\Delta\theta_{succ}$, both wrapped to $[0, 2\pi)$.

The local spacing error is:

$$
 e_\tau = \frac{\lambda_{pred}\,\Delta\theta_{succ} - \lambda_{self}\,\Delta\theta_{pred}}
 {\lambda_{pred}\,\Delta\theta_{succ} + \lambda_{self}\,\Delta\theta_{pred}}.
$$

If gaps are missing or degenerate, $e_\tau = 0$.

### Optional local omega damping

A local angular-rate damping term can be applied using neighbor estimates:

$$
 e_{\tau,eff} = e_\tau - K_{\omega}(\omega_i - \omega_{ref}).
$$

If `K_OMEGA_DAMP = 0`, then $e_{\tau,eff} = e_\tau$.

## 4) Tangential state update

Each agent maintains a scalar state $u$ that evolves as:

$$
 u_{k+1} = u_k + \Delta t\left(-BETA_U\,u_k + K_{E\tau}\,e_{\tau,eff}\right).
$$

## 5) Mapping to tangential velocity

Tangential velocity is computed as:

$$
 v_{\tau} = (K_{\tau}\,u\,r_{eff})\,\hat e_{\tau,i}.
$$

This yields an induced angular rate $\omega \approx K_{\tau} u$ for $r_i > R_{min}$.

## 6) Final command

The final commanded velocity is:

$$
 v_{cmd} = v_T + v_{rad} + v_{\tau}.
$$

The mobility handler clamps $v_{cmd}$ to respect `VM_MAX_SPEED_XY` and `VM_MAX_SPEED_Z`.
