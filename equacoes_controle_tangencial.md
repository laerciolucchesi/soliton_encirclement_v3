# Tangential Control Equations (Time Domain)

This document summarizes the tangential control law as implemented in `protocol_agent.py`. It mirrors the time-domain equations and keeps the notation close to the code.

## 1) Geometry around the target (XY plane)

Let the target position be $p_T(t) \in \mathbb{R}^2$ and agent $i$ be $p_i(t) \in \mathbb{R}^2$.

$$
\mathbf r_i(t) = p_i(t) - p_T(t),\qquad r_i(t) = \|\mathbf r_i(t)\|.
$$

$$
\hat{\mathbf r}_i(t) = \frac{\mathbf r_i(t)}{r_i(t)}.
$$

The unit tangential direction (counter-clockwise) is:

$$
\hat{\mathbf t}_i(t) = \begin{bmatrix}-\hat r_{iy}(t)\\ \hat r_{ix}(t)\end{bmatrix}.
$$

To avoid singularity when $r_i \to 0$:

$$
 r_{\mathrm{eff},i}(t) = \max\{r_i(t), R_{\min}\}.
$$

## 2) Tangential spacing error (dimensionless)

Let $\Delta\theta_{i^-}(t)$ be the angular gap to the predecessor and $\Delta\theta_{i^+}(t)$ be the gap to the successor. With arc weights $\lambda_{i^-} > 0$ and $\lambda_{i^+} > 0$:

$$
 e_{\tau,i}(t) =
 \frac{\lambda_{i^-}\,\Delta\theta_{i^+}(t) - \lambda_{i^+}\,\Delta\theta_{i^-}(t)}
 {\lambda_{i^-}\,\Delta\theta_{i^+}(t) + \lambda_{i^+}\,\Delta\theta_{i^-}(t)}.
$$

Uniform spacing is recovered with $\lambda_j = 1$ for all arcs.

## 3) Optional local angular-rate damping

The local angular rate around the target is estimated as:

$$
 \omega_i(t) = \frac{(\dot p_i(t) - \dot p_T(t))\cdot \hat{\mathbf t}_i(t)}{r_i(t)}.
$$

A local reference from neighbors (when available) yields:

$$
 e^{\mathrm{eff}}_{\tau,i}(t) = e_{\tau,i}(t) - K_{\Omega}(\omega_i(t) - \omega^{\mathrm{local}}_{\mathrm{ref},i}(t)).
$$

## 4) Tangential state dynamics

Each agent maintains a scalar state $u_i(t)$:

$$
 \dot u_i(t) = -BETA_U\,u_i(t) + K_{E_\tau}\,e^{\mathrm{eff}}_{\tau,i}(t).
$$

Discrete-time update with $\Delta t = \text{CONTROL_PERIOD}$:

$$
 u_i(t+\Delta t) = u_i(t) + \Delta t\left(-BETA_U\,u_i(t) + K_{E_\tau}\,e^{\mathrm{eff}}_{\tau,i}(t)\right).
$$

## 5) Mapping to commanded tangential velocity

$$
\mathbf v_{\tau,i}(t) = \big(K_{\tau}\,u_i(t)\,r_{\mathrm{eff},i}(t)\big)\,\hat{\mathbf t}_i(t).
$$

## 6) Optional spin term

If the target broadcasts a reference spin $\omega^{\mathrm{target}}_{\mathrm{ref}}(t)$:

$$
\mathbf v_{\mathrm{spin},i}(t) = \big(\omega^{\mathrm{target}}_{\mathrm{ref}}(t)\,r_i(t)\big)\,\hat{\mathbf t}_i(t).
$$
