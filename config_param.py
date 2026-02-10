"""Centralized parameter/config constants for the project.

This module is intended to be the single source of truth for shared
configuration parameters used across protocols and other components.
"""

# --------------------------------------------------------------------------------------
# 1) Simulation framework (timing + simulator timers)
# --------------------------------------------------------------------------------------

# Base control loop period (seconds) for the "muscular" u controller.
CONTROL_PERIOD: float = 0.01

# Fast soliton-error update period (seconds) for the "nervous" q controller.
SOLITON_PERIOD: float = 0.01

# TargetState broadcast period (seconds). Keep equal to control loop by default.
TARGET_STATE_BROADCAST_PERIOD: float = CONTROL_PERIOD

# AdversaryState broadcast period (seconds). Keep equal to control loop by default.
ADVERSARY_STATE_BROADCAST_PERIOD: float = CONTROL_PERIOD

# Timer IDs (string keys used by the simulator)
CONTROL_LOOP_TIMER_STR: str = "control_loop_timer"
SOLITON_LOOP_TIMER_STR: str = "soliton_loop_timer"
TARGET_STATE_BROADCAST_TIMER_STR: str = "broadcast_timer"

# Adversary timer IDs
ADVERSARY_STATE_BROADCAST_TIMER_STR: str = "adversary_state_broadcast_timer"

# Simulation defaults (used by main simulation entrypoints)
SIM_DURATION: float = 60          # Simulation duration (seconds)
SIM_REAL_TIME: bool = False          # Run in real time
SIM_DEBUG: bool = False             # Enable simulator debug mode

# --------------------------------------------------------------------------------------
# Global reproducibility control
# --------------------------------------------------------------------------------------
# If True, all randomness (initialization, target/adversary motion, agent failures, etc.)
# will be seeded deterministically for fully reproducible experiments.
# If False, all random draws will be non-deterministic (true randomness).
EXPERIMENT_REPRODUCIBLE: bool = True

# --------------------------------------------------------------------------------------
# 2) Communication + visualization (medium + UI)
# --------------------------------------------------------------------------------------

# Communication medium defaults
COMMUNICATION_TRANSMISSION_RANGE: float = 200  # Communication range (meters)
COMMUNICATION_DELAY: float = 0.0               # Communication delay (seconds)
COMMUNICATION_FAILURE_RATE: float = 0.0        # Packet loss probability [0.0, 1.0]

# Visualization defaults (used by main simulation entrypoints)
VIS_OPEN_BROWSER: bool = True       # Open the visualization in a browser
VIS_UPDATE_RATE: float = 0.1        # Visualization update period (seconds)

# --------------------------------------------------------------------------------------
# 3) Engine node / mobility model (VelocityMobility)
# --------------------------------------------------------------------------------------

# Velocity mobility default parameters (used by main simulation entrypoints)
VM_UPDATE_RATE: float = 0.01        # Update every 0.01 seconds
VM_MAX_SPEED_XY: float = 10.0       # Max horizontal speed: 10 m/s
VM_MAX_SPEED_Z: float = 5.0         # Max vertical speed: 5 m/s
VM_MAX_ACC_XY: float = 4.0          # Max horizontal acceleration: 4.0 m/s²
VM_MAX_ACC_Z: float = 5.0           # Max vertical acceleration: 5.0 m/s²
VM_TAU_XY: float = 1.0              # Optional: 1st-order horizontal tracking time constant (s)
VM_TAU_Z: float = 1.2               # Optional: 1st-order vertical tracking time constant (s)
VM_SEND_TELEMETRY: bool = True      # Enable telemetry
VM_TELEMETRY_DECIMATION: int = 1    # Send telemetry every update

# --------------------------------------------------------------------------------------
# 4a) Target motion (optional)
# --------------------------------------------------------------------------------------

# Move the target with a constant speed setpoint in the XY plane,
# changing direction randomly at a fixed period.
TARGET_MOTION_TIMER_STR: str = "target_motion_timer"
TARGET_MOTION_PERIOD: float = 1.0        # change velocity direction every this many seconds
TARGET_MOTION_SPEED_XY: float = 0.0      # target speed (m/s)
TARGET_MOTION_BOUNDARY_XY: float = 20.0  # meters; if |x| or |y| exceeds this, steer back to (0,0)

# --------------------------------------------------------------------------------------
# 4b) Adversary motion (random roaming)
# --------------------------------------------------------------------------------------

# Adversary random roaming region in XY: [-ADVERSARY_ROAM_BOUND_XY, +ADVERSARY_ROAM_BOUND_XY]
ADVERSARY_ROAM_BOUND_XY: float = 40.0
# Minimum allowed distance between adversary and target in XY (meters)
ADVERSARY_MIN_TARGET_DISTANCE: float = 30.0
# Nominal adversary roaming speed in XY (m/s)
ADVERSARY_ROAM_SPEED_XY: float = 4.0 #4.0

# --------------------------------------------------------------------------------------
# 5) Failure injection (agent outages)
# --------------------------------------------------------------------------------------

# Failure injection: simulate agent node outages by cancelling timers for a while.
#
# Interpretation:
# - Every FAILURE_CHECK_PERIOD seconds, each agent draws a random trial whose probability is
#   computed from FAILURE_MEAN_FAILURES_PER_MIN (Poisson-like approximation).
# - If it "fails", it goes OFF for FAILURE_OFF_TIME seconds.
# - While OFF, the agent ignores packets and does not reschedule its main timers.
#
# Note: in this project the target never fails; only agents can.
FAILURE_CHECK_TIMER_STR: str = "failure_check_timer"
FAILURE_RECOVER_TIMER_STR: str = "failure_recover_timer"
FAILURE_ENABLE: bool = False           # Whether to enable failure injection
FAILURE_CHECK_PERIOD: float = 0.1     # seconds
FAILURE_MEAN_FAILURES_PER_MIN: float = 1.0  # mean failures per minute
FAILURE_OFF_TIME: float = 8.0         # seconds

# --------------------------------------------------------------------------------------
# 6) Failure detection and liveness (timeouts, neighbor selection)
# --------------------------------------------------------------------------------------

# Protocol liveness / neighbor selection tuning
# Note: these values are local (not transmitted) and are expressed in seconds/radians.
# Timeout guards are scaled with CONTROL_PERIOD.
AGENT_STATE_TIMEOUT: float = 5.0 * CONTROL_PERIOD    # AgentState liveness timeout (s)
TARGET_STATE_TIMEOUT: float = 10.0 * CONTROL_PERIOD  # TargetState liveness timeout (s)
HYSTERESIS_RAD: float = 0.05                         # Neighbor switching hysteresis (rad)

# Optional housekeeping: prune expired cached states to avoid unbounded growth.
PRUNE_EXPIRED_STATES: bool = True

# --------------------------------------------------------------------------------------
# 7) Swarm formation structure
# --------------------------------------------------------------------------------------

# Swarm/encirclement defaults
NUM_AGENTS: int = 10                # Number of agent nodes
ENCIRCLEMENT_RADIUS: float = 20.0   # Desired encirclement radius in meters

# Desired angular velocity for the whole swarm to spin around the target (rad/s).
# This value is broadcast by the target inside TargetState.
TARGET_SWARM_OMEGA_REF: float = 0.0

# Protection angle (degrees): desired protected/covered arc between the two boundary nodes.
#
# The controller uses `lambda` as an arc *weight* (dimensionless): at equilibrium, each
# arc size is proportional to its lambda, and all arcs sum to 360 degrees.
#
# We assign one special arc (edge node -> successor) with weight `edge_lambda`, and the
# other (alive_count-1) arcs with weight 1.0. The complement of the protected arc is the
# "edge gap":
#   edge_gap_deg = 360 - PROTECTION_ANGLE_DEG
#
# For any desired edge_gap_deg in [0, 360), the corresponding edge_lambda is:
#   edge_lambda = edge_gap_deg * (alive_count - 1) / (360 - edge_gap_deg)
#
# Notes:
# - PROTECTION_ANGLE_DEG = 360 means edge_gap_deg = 0 => no boundary arc (uniform lambdas=1).
# - If edge_gap_deg is smaller than the uniform gap (360/alive_count), then edge_lambda < 1 and
#   the boundary arc is the *smallest* gap (the token should track the minimum gap, not maximum).
PROTECTION_ANGLE_DEG: float = 360.0

# Minimum effective radius used by the tangential mapping to avoid division by
# near-zero radii and to keep the angular-rate interpretation well-conditioned.
# This does NOT change the desired encirclement radius; it only bounds r_eff.
R_MIN: float = 1.0

# --------------------------------------------------------------------------------------
# 8) Radial Controller (Proportional & Derivative terms)
# --------------------------------------------------------------------------------------

# Encirclement control gains (radial controller)
# The controller outputs a *radial* velocity correction (m/s) in the horizontal plane.
# - K_R scales radial distance error (m) into a velocity correction (m/s): units ~ 1/s.
# - K_DR damps radial motion using relative radial speed (m/s): units ~ dimensionless.
K_R: float = 1.0
K_DR: float = 0.5

# --------------------------------------------------------------------------------------
# 9) Tangential Controller (Soliton-like dynamics)
# --------------------------------------------------------------------------------------

# Soliton-like tangential controller parameters
#
#   1) First we update the internal scalar state u ("soliton" state):
#
#      u_next = u + dt * (
#          C_COUPLING * (u_succ - u_pred)
#          - BETA_U * u
#          - ALPHA_U * u^3
#          + K_E_TAU * e_tau_eff
#      )
#
#      where:
#        - gap_pred, gap_succ are the target-centric angular gaps (radians) to the
#          predecessor and successor, wrapped to (0, 2*pi)
#        - e_tau is the local spacing imbalance error (normalized):
#            e_tau = (gap_succ - gap_pred) / (gap_succ + gap_pred)
#          (returns 0.0 when gaps are missing/degenerate)
#        - e_tau_eff is either e_tau or (optionally) a damped version:
#            e_tau_eff = e_tau - K_OMEGA_DAMP * (omega_self - omega_ref)
#
#   2) Then we convert u into a tangential velocity vector in the XY plane:
#
#      v_tau_vec = (K_TAU * u * r_eff) * t_hat
#
#      where t_hat is the unit tangential direction around the target and
#      r_eff = max(r_xy, R_MIN). This yields an induced angular rate
#      omega ≈ v_tau / r ≈ K_TAU * u (for r > R_MIN), independent of ENCIRCLEMENT_RADIUS.

# Optional alternative to the cubic containment term (-ALPHA_U*u^3):
#
# Soft limiter (recommended p=2):
#   g(u) = u^3 / (1 + (|u|/U_lim)^2)
#
# This matches u^3 near u=0, but becomes ~U_lim^2*u for |u| >> |U_lim|.
USE_SOFT_LIMITER_U: bool = False  # Whether to use soft limiter on u update
U_lim: float = 2.0                # Soft limiter scale (only used if USE_SOFT_LIMITER_U is True)

# Tangential controller gains (u dynamics)
K_TAU: float = 0.2        # tangential control gain (velocity scaling) (=0.2)
BETA_U: float = 7.0       # linear damping coefficient (=7.0)
ALPHA_U: float = 0.45      # nonlinear amplitude containment (=0.45)
C_COUPLING: float = 1.8   # antisymmetric coupling strength (=1.8)
K_E_TAU: float = 25.0     # spacing error injection gain (e_tau multiplier) (=25)
K_Q_TO_U: float = 12.0     # coupling gain from q into u (=12.0)

# Enable/disable the intermediate q layer; when False, e_tau injects directly into u.
USE_Q_LAYER: bool = True

# Optional local angular-rate damping (no global information required).
# Use e_tau_eff instead of e_tau in the u update above.
# When enabled, the agent forms an omega reference from its two neighbors and
# subtracts a proportional term from the spacing error injection:
#   e_tau_eff = e_tau - K_OMEGA_DAMP * (omega_self - omega_ref)
K_OMEGA_DAMP: float = 0.1  # angular-rate damping gain (0.0 to disable); default value = 0.1

# Optional diffusion (discrete Laplacian) on the soliton state u:
#   + KAPPA_U_DIFF * (u_succ - 2u + u_pred)
# Helps damp high-frequency spatial oscillations of u along the ring.
KAPPA_U_DIFF: float = 0.1  # diffusion gain; 0.0 to disable. (=0.1)

# Soliton-error (q dynamics) damping.

BETA_Q: float = 5.5 # (=5.5) 

# Dissipação cúbica da camada q
GAMMA_Q_CUBIC: float = 0.1  # ajuste para ativar, ex: 0.1

# Optional additional KdV-like terms (used in q dynamics).
# Steepening / nonlinear transport:
#   - K_U_STEEPEN * q * q_s
# where q_s = (q_succ - q_pred) (central 1-hop; scaling absorbed into the gain).
#K_U_STEEPEN: float = 0.4
K_U_STEEPEN: float = 0.9 #(=0.9)

# Dispersion (KdV-like) using the 1-hop gradient of curvature:
#   + KAPPA_U_DISP * q_sss
# where q_sss = (q_ss_succ - q_ss_pred) and q_ss is received from 1-hop neighbors.
KAPPA_U_DISP: float = 0.04 #(=0.04)

# Optional saturation of the soliton state u and q if |v_cmd| is greater than its max magnitude.
ANTI_WINDUP_ENABLE: bool = True

# -----------------------------------------------------------------------------
# Q layer architecture selection
# -----------------------------------------------------------------------------
# Options:
# Mode FORCE_E
# q function: not used for modulation; input e_tau (spacing error), output q.
# u function: input e_tau; output u (velocity), using only fixed parameters (no modulation).

# Mode MODULATE_K_Q_TO_U
# q function: input e_tau; output q (for modulation signal purposes).
# u function: input q; output u, where q modulates only the K_Q_TO_U parameter (gain of q in the u equation).

# Mode MODULATE_PARAMS
# q function: input e_tau; output q (as modulation source using |q| and qss).
# u function: input q; output u, where q modulates multiple parameters (K_TAU, KAPPA_U_DIFF, BETA_U) that affect the dynamics of u.

# Mode FORCE_U
# q function: input e_tau; output q (modulation signal).
# u function: input q; output u, where q directly determines the value of u (no parameter modulation).
# -----------------------------------------------------------------------------
Q_LAYER_ARCH = "FORCE_E"

# -----------------------------------------------------------------------------
# KdV type selection for the q layer (only used if USE_Q_LAYER == True)
# "ORIGINAL": central form: dq_adv = -a * q * (q_succ - q_pred)
# "BURGERS" : stable conservative-flux form (Rusanov) + CFL substepping
# -----------------------------------------------------------------------------
KDV_TYPE = "BURGERS"  # Options: "ORIGINAL", "BURGERS"

# -----------------------------------------------------------------------------
# Q -> K_Q_TO_U gain modulation (only used when Q_LAYER_ARCH == "MODULATE_K_Q_TO_U")
# -----------------------------------------------------------------------------
# Q -> K_Q_TO_U gain modulation (only used when Q_LAYER_ARCH == "MODULATE_K_Q_TO_U")
# Maximum factor: K_Q_TO_U_eff = K_Q_TO_U * factor(m_f)
Q_MOD_DELTA = 0.4

# EMA update for q0 scale (0..1). Higher -> adapts faster.
Q_MOD_MU = 0.07

# Low-pass on modulation signal m (0..1). Higher -> reacts faster (more jitter risk).
Q_MOD_NU = 0.30

# Numerical floor to avoid division by zero in normalization.
Q_MOD_EPS = 1e-3

# Map normalized z=|q|/q0 to m in [0,1] using tanh(gain*z)
Q_MOD_USE_TANH = True
Q_MOD_TANH_GAIN = 7.0

# If True, freeze/zero the modulator on saturation (helps stability).
Q_MOD_FREEZE_ON_SAT = True



# -----------------------------------------------------------------------------
# Q -> K_TAU gain modulation (usado quando Q_LAYER_ARCH == "MODULATE_PARAMS")
# -----------------------------------------------------------------------------
# Idea:
#  - Acceleration channel (m_acc): when |q| is large, strengthen the velocity scaling gain K_TAU so the ring reacts faster.
#  - Robustness channel (m_robust): when q is "rough" (large curvature |q_ss|), add smoothing/damping to reduce jitter.
#
# Ambos os canais produzem sinais de modulação em [0,1] e aplicam filtragem low-pass.

# Acceleration: modulate K_TAU (velocity scaling gain)
Q_PARAM_ACC_DELTA = 0.3  # fraction around baseline (see Q_PARAM_ACC_BIDIR)
Q_PARAM_ACC_BIDIR = True  # True: factor in [1-delta, 1+delta]; False: [1, 1+delta]
Q_PARAM_ACC_MIN_FACTOR = 0.10
Q_PARAM_ACC_MAX_FACTOR = 3.00

# Robustness: increase smoothing/damping when oscillatory (convenient even if baseline is 0)
Q_PARAM_DIFF_ADD = 0.03   # kappa_eff = KAPPA_U_DIFF + Q_PARAM_DIFF_ADD * m_robust_f
Q_PARAM_BETA_ADD = 0.03   # beta_eff  = BETA_U        + Q_PARAM_BETA_ADD * m_robust_f

# Optional multiplicative boosts (set to 0.0 to disable)
Q_PARAM_DIFF_DELTA = 0.01  # kappa_eff *= (1 + Q_PARAM_DIFF_DELTA * m_robust_f)
Q_PARAM_BETA_DELTA = 0.01  # beta_eff  *= (1 + Q_PARAM_BETA_DELTA * m_robust_f)

# Roughness estimator (based on |q_ss|)
Q_ROUGH_MU = 0.05   # EMA update for qss0 scale
Q_ROUGH_NU = 0.05   # Low-pass on m_robust
Q_ROUGH_EPS = 1e-3
Q_ROUGH_USE_TANH = True
Q_ROUGH_TANH_GAIN = 1.0

# --------------------------------------------------------------------------------------
# 10) Spin Controller (Proportional & Derivative terms)
# --------------------------------------------------------------------------------------

# Enable/disable the swarm spin controller that tracks the adversary direction.
# - If False: the target will NOT try to align the swarm spin with the adversary; it will
#   broadcast omega_ref = TARGET_SWARM_OMEGA_REF (typically 0.0 for no spin).
# - If True: omega_ref is generated by the PD controller below (with optional open-loop
#   bias when KP=KD=0).
TARGET_SWARM_SPIN_ENABLE: bool = False

# PD controller for omega_ref generation (based on angular error in radians).
# - If KP=KD=0: omega_ref = TARGET_SWARM_OMEGA_REF (pure open-loop spin)
# - Otherwise:  omega_ref = KP * err + KD * derr (no constant bias)
TARGET_SWARM_OMEGA_PD_KP: float = 1.0
TARGET_SWARM_OMEGA_PD_KD: float = 0.2
TARGET_SWARM_OMEGA_PD_MAX_ABS: float = 1.0

# If the swarm is close to uniformly distributed, the sum of unit vectors
# target->agents has near-zero magnitude and its direction becomes ill-defined.
# This threshold (Kuramoto-like rho in [0,1]) disables the angular error when
# rho is too small, avoiding a fixed-direction bias.
TARGET_SWARM_SPIN_RHO_MIN: float = 0.05

# -------------------------
# 11) Metrics / experiment constants (used by plot_telemetry.py)
# -------------------------

# METRICS_T0: start time (s) of the "regime" window for M1..M6.
# Example: METRICS_T0=10 means "ignore the first 10 seconds" when computing the regime metrics.
METRICS_T0: float = 1.0

# METRICS_E_THR: absolute error threshold for settling time M7, using e(t)=|e_tau(t)|.
# Choose a value that is meaningful in your normalized e_tau scale (often 0.01~0.05).
METRICS_E_THR: float = 0.05

# METRICS_MA_W_SEC: moving-average window (seconds) used by M4 (oscillation metric).
# Example: 1.0 means "remove the slow trend with a 1-second moving average".
METRICS_MA_W_SEC: float = 1.0

# METRICS_SETTLE_WINDOW_SEC: continuous time window (seconds) required for M7 settling.
# Example: 2.0 means "consider settled when e(t)<=E_THR continuously for 2 seconds".
METRICS_SETTLE_WINDOW_SEC: float = 5.0

