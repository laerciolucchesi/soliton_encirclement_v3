# Tangential Control Equations (Current Simplified Form)

This document mirrors the current implementation in `protocol_agent.py`. It is intentionally minimal and focuses on the two-term tangential update used today.

## 1) Local spacing error

With predecessor and successor gaps $\Delta\theta_{pred}$ and $\Delta\theta_{succ}$ and arc weights $\lambda_{pred}$ and $\lambda_{self}$:

$$
 e_\tau = \frac{\lambda_{pred}\,\Delta\theta_{succ} - \lambda_{self}\,\Delta\theta_{pred}}
 {\lambda_{pred}\,\Delta\theta_{succ} + \lambda_{self}\,\Delta\theta_{pred}}.
$$

## 2) Optional angular-rate damping

If enabled:

$$
 e_{\tau,eff} = e_\tau - K_{\Omega}(\omega_i - \omega_{ref}).
$$

Otherwise $e_{\tau,eff} = e_\tau$.

## 3) Tangential state update

Continuous time:

$$
 \dot u = -BETA_U\,u + K_{E\tau}\,e_{\tau,eff}.
$$

Discrete time (Euler):

$$
 u_{k+1} = u_k + \Delta t\left(-BETA_U\,u_k + K_{E\tau}\,e_{\tau,eff}\right).
$$

## 4) Mapping to tangential velocity

$$
 v_{\tau} = (K_{\tau}\,u\,r_{eff})\,\hat t.
$$
