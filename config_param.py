"""Centralized parameter/config constants for the project.

This module is intended to be the single source of truth for shared
configuration parameters used across protocols and other components.

A small number of constants accept runtime overrides via environment
variables (used by batch sweep scripts). Such overrides are applied at
import time and take precedence over the literal defaults below.
"""

import os as _os


# --------------------------------------------------------------------------------------
# 1) Simulation framework (timing + simulator timers)
# --------------------------------------------------------------------------------------

# Base control loop period (seconds) for the "muscular" u controller. A single
# `dt` drives the control loop, the state-broadcast periods (below) AND the mobility
# update (VM_UPDATE_RATE, tied to it in section 3). Env-overridable.
#
# Default changed 0.01 -> 0.05 (campaign Ciclo 1, 2026-06): the relaxation times
# are dt-invariant in SECONDS (measured CV < 5% over dt 0.01..0.1) and the B2
# regime jitter is likewise dt-invariant, so 0.05 gives ~5x faster simulations
# (5x fewer ticks/s) with no qualitative change to any conclusion. Set
# CONTROL_PERIOD=0.01 for higher-fidelity / legacy-comparison runs. NOTE: at the
# coarser dt the failure-detector timeout must stay >= ~20 ticks for loss
# robustness -- AGENT_STATE_TIMEOUT's default handles this (see section 6).
try:
    CONTROL_PERIOD: float = float(_os.environ.get("CONTROL_PERIOD", "0.05"))
except ValueError:
    CONTROL_PERIOD = 0.05
if CONTROL_PERIOD <= 0.0:
    CONTROL_PERIOD = 0.05

# TargetState broadcast period (seconds). Keep equal to control loop by default.
TARGET_STATE_BROADCAST_PERIOD: float = CONTROL_PERIOD

# AdversaryState broadcast period (seconds). Keep equal to control loop by default.
ADVERSARY_STATE_BROADCAST_PERIOD: float = CONTROL_PERIOD

# Timer IDs (string keys used by the simulator)
CONTROL_LOOP_TIMER_STR: str = "control_loop_timer"
TARGET_STATE_BROADCAST_TIMER_STR: str = "broadcast_timer"

# Adversary timer IDs
ADVERSARY_STATE_BROADCAST_TIMER_STR: str = "adversary_state_broadcast_timer"

# Simulation defaults (used by main simulation entrypoints)
try:
    SIM_DURATION: float = float(_os.environ.get("SIM_DURATION", "60"))
except ValueError:
    SIM_DURATION = 60.0          # Simulation duration (seconds); env-overridable for sweeps
SIM_REAL_TIME: bool = False          # Run in real time
SIM_DEBUG: bool = False             # Enable simulator debug mode

# --------------------------------------------------------------------------------------
# Global reproducibility control
# --------------------------------------------------------------------------------------
# If True, all randomness (initialization, target/adversary motion, agent failures, etc.)
# will be seeded deterministically for fully reproducible experiments.
# If False, all random draws will be non-deterministic (true randomness).
# Env-overridable (campaign rule 3: every relevant parameter must be pinnable in
# the child process env instead of relying on this default). Default unchanged.
EXPERIMENT_REPRODUCIBLE: bool = (
    _os.environ.get("EXPERIMENT_REPRODUCIBLE", "True").strip().lower()
    in ("true", "1", "yes", "y")
)

# --------------------------------------------------------------------------------------
# 2) Communication + visualization (medium + UI)
# --------------------------------------------------------------------------------------

# Communication medium defaults
# env-overridable: for the PROOF of the neighbor-only premise (short range ~ few neighbors).
try:
    COMMUNICATION_TRANSMISSION_RANGE: float = float(_os.environ.get("COMMUNICATION_TRANSMISSION_RANGE", "200"))
except ValueError:
    COMMUNICATION_TRANSMISSION_RANGE = 200.0  # Communication range (meters)
# env-overridable (Phase 3 - robustness: degraded communication). Defaults 0.0 preserve old runs.
try:
    COMMUNICATION_DELAY: float = float(_os.environ.get("COMMUNICATION_DELAY", "0.0"))   # Communication delay (seconds)
except ValueError:
    COMMUNICATION_DELAY = 0.0
try:
    COMMUNICATION_FAILURE_RATE: float = float(_os.environ.get("COMMUNICATION_FAILURE_RATE", "0.0"))  # Packet loss probability [0.0, 1.0]
except ValueError:
    COMMUNICATION_FAILURE_RATE = 0.0
if not (0.0 <= COMMUNICATION_FAILURE_RATE <= 1.0):
    COMMUNICATION_FAILURE_RATE = 0.0

# Role-aware (asymmetric) communication ranges -- OFF by default.
#
# COMMUNICATION_TRANSMISSION_RANGE above is a single range for every link. That
# conflates two links with opposite requirements, because an agent's AgentState
# is ONE broadcast serving two audiences: the ring neighbours (which we may want
# short, for physical locality) and the target, which must hear every agent or
# it prunes the silent ones as dead (protocol_target._prune_expired_states) and
# drops them from alive_lambdas -- corrupting alive_count, the lambda map fed
# back to the agents, and M1..M7, silently.
#
# When COMM_ROLE_AWARE_RANGES is False, main.py builds the stock
# CommunicationHandler and nothing changes. When True it builds
# comm_role_aware.RoleAwareCommunicationHandler, which picks the range per
# (sender_role, receiver_role) pair:
#   agent  <-> agent   : COMM_RANGE_AGENT_AGENT
#   agent  <-> target  : COMM_RANGE_AGENT_TARGET   (both directions: the target
#                        has the better radio -- stronger Tx AND better Rx)
#   anything involving the adversary: COMMUNICATION_TRANSMISSION_RANGE.
#
# Sizing (R = ENCIRCLEMENT_RADIUS): the k-hop chord of the ring is
# 2*R*sin(k*pi/N). There are TWO thresholds, for two different quantities, and
# earlier drafts of this note collapsed them into one and got it wrong twice.
# Both are measured at N=24, R=20, single deterministic death, 8 paired seeds,
# baseline and B2 (phase 8a (i), comm_range_results.csv).
#
#   (1) CLOSING AT ALL needs ~1 hop. Below it the gap never closes within the
#       budget for EITHER method, so it is a property of the ring and the
#       controller, not of the overlay. Cliff measured in c = range/(1-hop
#       chord): c in (1.21, 1.61], identical for baseline and B2.
#
#   (2) THE B2 ADVANTAGE needs 2 hops, i.e. range >= 2*R*sin(2*pi/N).
#       Mechanism, read from the event_ids: the originator is the victim's
#       PREDECESSOR, and one of its two counter-propagating pulses is blocked
#       immediately by the corpse. The victim's SUCCESSOR could only ever see
#       that direction directly from the originator, across the merged gap --
#       exactly the 2-hop chord. A receiver needs BOTH directions, so below
#       that chord NOBODY completes the SAIDA and it vanishes without a trace
#       (the protocol logs completions, not injections). The ring then
#       contracts, the successor drifts into range, and locally that reads as
#       a node APPEARING: the swarm executes a sign-inverted ENTRADA instead,
#       with 22/23 coverage, for a node that never joined. Hence B2 is not
#       weaker below the chord but WRONG: 6.45 s against the baseline's 3.27 s
#       at 8.4 m (c=1.61), a 2x PENALTY. At 10.4 m -- the first point above the
#       10.353 m chord -- the SAIDA lands first try, coverage is 23/23 and B2
#       wins 2.30 s against 3.20 s. On the strict 1.10 threshold the inversion
#       is sharper still: 16.58 s against 8.00 s below, 3.42 s against 7.65 s
#       above. General form: with a finite radio range, "came into range" and
#       "joined the ring" are locally indistinguishable, and the neighbour-only
#       premise is what makes the ambiguity unresolvable.
#
# The advantage then SATURATES: B2 sits at 2.30-2.32 s from c=1.99 to c=5.00,
# so range beyond the 2-hop chord buys nothing. Design rule: size the ring
# radio at 2*R*sin(2*pi/N) and stop.
#
# Caveat on the short end: AGENT_STATE_TIMEOUT was pinned at 5*dt for that
# campaign, and to a failure detector an out-of-range neighbour is
# indistinguishable from a dead one. At c=1.21 the links flap around the range
# boundary and each flap injects a fresh SAIDA/ENTRADA -- coverage came back
# ABOVE 1.0 (several events, not one) with G_max peaks up to 8.2. That regime
# is contaminated by the detector, not a clean reading of the mechanism; the
# campaign's FD-fix value under degraded comms is 20*dt.
#
# Note that a FIXED metric range means the hop neighbourhood GROWS with N at
# fixed R (30 m is 2 hops at N=10 but 13 hops at N=50 and 27 at N=100) -- that
# is physical realism, not constant locality. To hold locality constant across
# a scaling sweep, set the range per run as a multiple of 2*R*sin(pi/N).
COMM_ROLE_AWARE_RANGES: bool = (
    _os.environ.get("COMM_ROLE_AWARE_RANGES", "False").strip().lower()
    in ("true", "1", "yes", "y")
)


def _comm_range_env(name: str) -> float:
    """Read a role-aware range, falling back to the global range when unusable."""
    try:
        value = float(_os.environ.get(name, COMMUNICATION_TRANSMISSION_RANGE))
    except ValueError:
        return float(COMMUNICATION_TRANSMISSION_RANGE)
    if not (value > 0.0) or value != value or value == float("inf"):
        return float(COMMUNICATION_TRANSMISSION_RANGE)
    return value


# Ring link: agent <-> agent.
COMM_RANGE_AGENT_AGENT: float = _comm_range_env("COMM_RANGE_AGENT_AGENT")
# Uplink/downlink to the target, both directions.
COMM_RANGE_AGENT_TARGET: float = _comm_range_env("COMM_RANGE_AGENT_TARGET")

# Visualization defaults (used by main simulation entrypoints)
# VIS_OPEN_BROWSER controls whether the GrADyS visualization auto-opens a
# browser window. Disabling speeds up batch sweeps and avoids spawning
# windows during Monte Carlo runs.
VIS_OPEN_BROWSER: bool = (
    _os.environ.get("VIS_OPEN_BROWSER", "True").strip().lower()
    in ("true", "1", "yes", "y")
)
VIS_UPDATE_RATE: float = 0.1        # Visualization update period (seconds)

# --------------------------------------------------------------------------------------
# 3) Engine node / mobility model (VelocityMobility)
# --------------------------------------------------------------------------------------

# Velocity mobility default parameters (used by main simulation entrypoints)
VM_UPDATE_RATE: float = CONTROL_PERIOD   # mobility step tied to the control period (dt); was fixed 0.01
try:
    VM_MAX_SPEED_XY: float = float(_os.environ.get("VM_MAX_SPEED_XY", "10.0"))  # env-overridable (agility axis)
except ValueError:
    VM_MAX_SPEED_XY = 10.0           # Max horizontal speed: 10 m/s
VM_MAX_SPEED_Z: float = 5.0         # Max vertical speed: 5 m/s
try:
    VM_MAX_ACC_XY: float = float(_os.environ.get("VM_MAX_ACC_XY", "4.0"))       # env-overridable (agility axis)
except ValueError:
    VM_MAX_ACC_XY = 4.0              # Max horizontal acceleration: 4.0 m/s^2
VM_MAX_ACC_Z: float = 5.0           # Max vertical acceleration: 5.0 m/s^2
try:
    VM_TAU_XY: float = float(_os.environ.get("VM_TAU_XY", "1.0"))               # env-overridable: UAV sluggishness knob (agility axis)
except ValueError:
    VM_TAU_XY = 1.0                  # 1st-order horizontal tracking time constant (s)
VM_TAU_Z: float = 1.2               # Optional: 1st-order vertical tracking time constant (s)
VM_SEND_TELEMETRY: bool = True      # Enable telemetry
VM_TELEMETRY_DECIMATION: int = 1    # Send telemetry every update

# --------------------------------------------------------------------------------------
# 4a) Target motion (optional)
# --------------------------------------------------------------------------------------

# Move the target with a constant speed setpoint in the XY plane,
# changing direction randomly at a fixed period.
TARGET_MOTION_TIMER_STR: str = "target_motion_timer"
# env-overridable (Phase 3 Track C - moving target). Huge PERIOD + large BOUNDARY => ~constant
# velocity; small PERIOD (1s) => maneuver (new random direction every period).
try:
    TARGET_MOTION_PERIOD: float = float(_os.environ.get("TARGET_MOTION_PERIOD", "1.0"))
except ValueError:
    TARGET_MOTION_PERIOD = 1.0
try:
    TARGET_MOTION_SPEED_XY: float = float(_os.environ.get("TARGET_MOTION_SPEED_XY", "0.0"))
except ValueError:
    TARGET_MOTION_SPEED_XY = 0.0
try:
    TARGET_MOTION_BOUNDARY_XY: float = float(_os.environ.get("TARGET_MOTION_BOUNDARY_XY", "20.0"))
except ValueError:
    TARGET_MOTION_BOUNDARY_XY = 20.0  # meters; if |x| or |y| exceeds this, steer back to (0,0)

# --------------------------------------------------------------------------------------
# 4b) Adversary motion (random roaming)
# --------------------------------------------------------------------------------------

# Adversary random roaming region in XY: [-ADVERSARY_ROAM_BOUND_XY, +ADVERSARY_ROAM_BOUND_XY]
ADVERSARY_ROAM_BOUND_XY: float = 40.0
# Minimum allowed distance between adversary and target in XY (meters)
ADVERSARY_MIN_TARGET_DISTANCE: float = 30.0
# Nominal adversary roaming speed in XY (m/s)
ADVERSARY_ROAM_SPEED_XY: float = 0.0 #4.0

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
# env-overridable (Phase 3 - churn): allows sweeping failure density and recovery time.
FAILURE_ENABLE: bool = (
    _os.environ.get("FAILURE_ENABLE", "True").strip().lower() in ("true", "1", "yes", "y")
)
FAILURE_CHECK_PERIOD: float = 0.1     # seconds
try:
    FAILURE_MEAN_FAILURES_PER_MIN: float = float(
        _os.environ.get("FAILURE_MEAN_FAILURES_PER_MIN", "2.0")
    )  # mean failures per minute (per agent stream)
except ValueError:
    FAILURE_MEAN_FAILURES_PER_MIN = 2.0
try:
    FAILURE_OFF_TIME: float = float(_os.environ.get("FAILURE_OFF_TIME", "8.0"))  # recovery (s)
except ValueError:
    FAILURE_OFF_TIME = 8.0

# --------------------------------------------------------------------------------------
# 5b) Deterministic single-failure mode (controlled-event experiments)
# --------------------------------------------------------------------------------------
#
# For clean post-event measurements (e.g. the stabilization-time-vs-N scaling
# law), the random Poisson failure stream is too noisy: the number, timing and
# identity of failures vary run-to-run, so a single recovery cannot be isolated.
# When DETERMINISTIC_FAILURE_ENABLE is True:
#   - NO agent ever draws a Poisson failure (the random stream is bypassed for all).
#   - The single agent whose node_id == DETERMINISTIC_FAILURE_AGENT_ID crashes
#     exactly once, at the first failure-check tick with t >= DETERMINISTIC_FAILURE_TIME_T0.
#   - It stays failed for the rest of the run when DETERMINISTIC_FAILURE_OFF_TIME < 0
#     (permanent crash -> cleanest single SAIDA event); otherwise it recovers after
#     that many seconds (a single SAIDA+ENTRADA pair).
#
# Node-id note: main.py adds the target (id 0) and adversary (id 1) first, so the
# agents occupy ids 2 .. NUM_AGENTS+1. The victim id must be in that agent range.
DETERMINISTIC_FAILURE_ENABLE: bool = (
    _os.environ.get("DETERMINISTIC_FAILURE_ENABLE", "True").strip().lower()
    in ("true", "1", "yes", "y")
)
# Accepts a comma-separated LIST (Phase 3 churn: fail k victims at once, scenarios A/B).
# DETERMINISTIC_FAILURE_AGENT_ID keeps the first id (back-compat); _IDS is the full set.
_raw_det_ids = _os.environ.get("DETERMINISTIC_FAILURE_AGENT_ID", "6")
_det_ids = []
for _tok in _raw_det_ids.split(","):
    _tok = _tok.strip()
    if _tok:
        try:
            _det_ids.append(int(_tok))
        except ValueError:
            pass
DETERMINISTIC_FAILURE_AGENT_IDS = frozenset(_det_ids)
DETERMINISTIC_FAILURE_AGENT_ID: int = _det_ids[0] if _det_ids else -1
try:
    DETERMINISTIC_FAILURE_TIME_T0: float = float(
        _os.environ.get("DETERMINISTIC_FAILURE_TIME_T0", "15.0")
    )
except ValueError:
    DETERMINISTIC_FAILURE_TIME_T0 = -1.0
try:
    DETERMINISTIC_FAILURE_OFF_TIME: float = float(
        _os.environ.get("DETERMINISTIC_FAILURE_OFF_TIME", "-1.0")
    )
except ValueError:
    DETERMINISTIC_FAILURE_OFF_TIME = -1.0

# --------------------------------------------------------------------------------------
# 6) Failure detection and liveness (timeouts, neighbor selection)
# --------------------------------------------------------------------------------------

# Protocol liveness / neighbor selection tuning
# Note: these values are local (not transmitted) and are expressed in seconds/radians.
# Timeout guards are scaled with CONTROL_PERIOD.
# AgentState liveness timeout (s). Env-overridable (Phase 3). A failure detector
# under packet loss needs a timeout that tolerates several CONSECUTIVE losses,
# otherwise live neighbors "flicker" dead (storm of false SAIDA/ENTRADA). The
# tolerance is TICK-denominated: P(false positive window) ~ loss_rate^(timeout/dt),
# so what matters is the number of broadcast periods covered, not absolute seconds.
#
# Default changed 5*dt -> max(20*dt, 0.2) (campaign Ciclo 1, 2026-06): ~20 ticks
# covers loss up to ~0.4 (0.4^20 ~ 1e-8). This equals 0.2 s at dt=0.01 (the
# campaign's validated FD-fix) and 1.0 s at the new default dt=0.05, where the
# old 5*dt=0.25 s (= only 5 ticks) was measured to break the detector under loss
# 0.2 for BOTH baseline and overlay (Ciclo 1 Block E). Trade-off: a real dead
# node is declared dead after this timeout, so larger dt buys loss-robustness at
# the cost of detection latency (a ~1 s additive offset to recovery at dt=0.05,
# separate from the dt-invariant relaxation time). Loss-free experiments may
# override to a shorter value for tighter settling measurements.
try:
    AGENT_STATE_TIMEOUT: float = float(_os.environ["AGENT_STATE_TIMEOUT"])
except (KeyError, ValueError):
    AGENT_STATE_TIMEOUT = max(20.0 * CONTROL_PERIOD, 0.2)   # AgentState liveness timeout (s)
TARGET_STATE_TIMEOUT: float = 10.0 * CONTROL_PERIOD  # TargetState liveness timeout (s)
HYSTERESIS_RAD: float = 0.05                         # Neighbor switching hysteresis (rad)

# HYSTERESIS_FRAC: when > 0, the neighbour-switching hysteresis is DIMENSIONLESS -- a fraction
# of the ideal gap (2*pi/N_alive) instead of the fixed absolute HYSTERESIS_RAD. The fixed value
# has HYSTERESIS_RAD/(2*pi/N) growing with N; it exceeds the gap at N ~ 2*pi/HYSTERESIS_RAD
# (~126 for 0.05), which would break neighbour switching at very large N (only in scenarios with
# neighbour REORDERING -- churn / passing / moving target; the order-preserving single-failure
# case is unaffected). The fraction keeps the hysteresis-to-gap ratio constant for any N
# (PATH-B dimensionless gain). 0.0 = legacy absolute HYSTERESIS_RAD. To match the legacy ~8%
# of the gap at the default N=10, use ~0.08. Env-overridable for sweeps.
try:
    HYSTERESIS_FRAC: float = float(_os.environ.get("HYSTERESIS_FRAC", "0.0"))
except ValueError:
    HYSTERESIS_FRAC = 0.0
if HYSTERESIS_FRAC < 0.0:
    HYSTERESIS_FRAC = 0.0

# Optional housekeeping: prune expired cached states to avoid unbounded growth.
PRUNE_EXPIRED_STATES: bool = True

# --------------------------------------------------------------------------------------
# 7) Swarm formation structure
# --------------------------------------------------------------------------------------

# Swarm/encirclement defaults
try:
    NUM_AGENTS: int = int(_os.environ.get("NUM_AGENTS", "10"))   # env-overridable for N-sweeps
except ValueError:
    NUM_AGENTS = 10                 # Number of agent nodes
ENCIRCLEMENT_RADIUS: float = 20.0   # Desired encirclement radius in meters

# Initial agent placement controls (consumed by main.py when spawning agent nodes).
#
# INIT_RADIUS_RANGE: half-width of the uniform radius scatter, expressed as a
# fraction of ENCIRCLEMENT_RADIUS. Each agent's initial radius is drawn from
#   r_i ~ Uniform((1 - INIT_RADIUS_RANGE), (1 + INIT_RADIUS_RANGE)) * ENCIRCLEMENT_RADIUS.
# Set to 0.0 to place every agent exactly at ENCIRCLEMENT_RADIUS. Default 0.1
# reproduces the legacy [0.9, 1.1] * R scatter.
#
# INIT_ANGLES_EQUIDISTANT: if False, initial angles are drawn uniformly in
# [0, 2*pi). If True, agents are placed at equidistant angles 2*pi/NUM_AGENTS apart
# starting from 0 rad.
try:
    INIT_RADIUS_RANGE: float = float(_os.environ.get("INIT_RADIUS_RANGE", "0.0"))
except ValueError:
    INIT_RADIUS_RANGE = 0.0
INIT_ANGLES_EQUIDISTANT: bool = (
    _os.environ.get("INIT_ANGLES_EQUIDISTANT", "True").strip().lower()
    in ("true", "1", "yes", "y")
)

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
# 9) Tangential Controller
# --------------------------------------------------------------------------------------

# Tangential controller parameters
#
# The controller maintains two independent state channels:
#
#   LOCAL channel — responds to the node's own spacing error:
#      u_local_next = u_local + dt * (- BETA_U_LOCAL * u_local + K_E_TAU * e_tau_eff)
#
#   PROPAGATED channel — responds to neighbor signals (pure neighbor info, no self-injection):
#      u_prop_next  = u_prop  + dt * (- BETA_U_PROP  * u_prop  + K_PROP * neighbor_signal)
#
#   Composition rule:
#      if u_local * u_prop >= 0:    u_total = u_local + u_prop   (same direction → sum)
#      else:                        u_total = smooth_dominance_blend(u_local, u_prop)
#
#   Tangential velocity:
#      v_tau_vec = (K_TAU * u_total * r_eff) * t_hat
#
#   where:
#     - e_tau_eff = e_tau or e_tau - K_OMEGA_DAMP*(omega_self - omega_ref)
#     - neighbor_signal = propagation_layer.get_neighbor_signal() (excludes self-injection)
#     - t_hat is the unit tangential direction; r_eff = max(r_xy, R_MIN)
#
# Setting BETA_U_LOCAL = BETA_U_PROP = BETA_U reproduces the previous single-state behaviour
# when channels always agree (e.g. baseline where u_prop = 0 always).

# Tangential controller gains (u dynamics)
K_TAU: float = 0.2          # tangential control gain (velocity scaling)
BETA_U: float = 7.0         # legacy single-channel damping (kept for reference)
BETA_U_LOCAL: float = 7.0   # damping for the local error channel
BETA_U_PROP: float = 7.0    # damping for the propagated error channel
# Spacing-error injection gain (e_tau multiplier). Default is the STABLE
# normalized gain K_E_TAU = 250/N: the normalized e_tau injects an effective
# gain factor ~N near equilibrium, so a FIXED gain destabilizes large rings
# (measured: smooth ~2 s limit cycle at N>=50). Normalizing keeps the gain
# product constant across N. At the default NUM_AGENTS=10 this equals the
# legacy literal 25.0, so default-N runs are numerically unchanged.
# Env-overridable for gain-normalization studies.
try:
    K_E_TAU: float = float(_os.environ.get("K_E_TAU", str(250.0 / NUM_AGENTS)))
except ValueError:
    K_E_TAU = 250.0 / NUM_AGENTS
U_CONFLICT_BLEND_WIDTH: float = 0.2  # 0.0 restores the previous hard winner-takes-all conflict composition

# Composition policy for u_local + u_prop in TangentialSpacingController._compose:
# - "blend": cooperative sum when channels agree in sign; smooth tanh dominance
#            blend (width = U_CONFLICT_BLEND_WIDTH) when they conflict. Defensive
#            against noisy propagation: discards the smaller channel in conflict.
# - "sum":   pure addition u = u_local + u_prop, regardless of sign. The propagation
#            channel always contributes — closer to the soliton-inspired premise
#            that neighbor signals carry useful information.
# No saturation is applied at the controller level in either mode; velocity-level
# saturation in the mobility handler (VM_MAX_SPEED_XY) remains the only limit.
#
# Runtime override: env var TANGENTIAL_COMPOSITION_MODE takes precedence over the
# literal default below. Used by run_sweep.py to vary the mode across runs without
# editing this file. Invalid values are caught later, at controller construction.
TANGENTIAL_COMPOSITION_MODE: str = _os.environ.get("TANGENTIAL_COMPOSITION_MODE", "sum")  # "blend" or "sum"

# Optional local angular-rate damping (no global information required).
# Use e_tau_eff instead of e_tau in the u update above.
# When enabled, the agent forms an omega reference from its two neighbors and
# subtracts a proportional term from the spacing error injection:
#   e_tau_eff = e_tau - K_OMEGA_DAMP * (omega_self - omega_ref)
K_OMEGA_DAMP: float = 0.1  # angular-rate damping gain (0.0 to disable); default value = 0.1


# --------------------------------------------------------------------------------------
# 9b) Fast soliton-like channel (event-triggered, observation-only in Phase A)
# --------------------------------------------------------------------------------------

# DampedAdvectionLayer parameters. The layer maintains two scalar fields per
# agent (u_R and u_L) propagating in opposite directions around the ring with
# exponential decay. Pulses are injected externally on detected events
# (predecessor/successor identity change) and propagate without affecting the
# control loop in Phase A — they are recorded in telemetry for visualization.
#
# WAVE_DAMPING_GAMMA: decay rate of each field (1/s). Half-life ~ 0.69/gamma.
#   gamma = 6.0 -> half-life 0.115 s. With one-hop-per-tick advection at dt=0.05,
#   ring traversal is ~N*dt seconds; for N=10 that's 0.5 s, so a pulse decays
#   to ~e^-3.0 = 5% after one full lap.
WAVE_DAMPING_GAMMA: float = 6.0

# K_TRIGGER: gain converting the detected delta_e_tau into pulse amplitude.
# pulse_amplitude = K_TRIGGER * delta_e_tau
K_TRIGGER: float = 1.0

# MIN_EVENT_DELTA_FRAC: minimum |delta_e_tau| to trigger a pulse, expressed as
# a fraction of the agent's own desired gap (which the agent computes from the
# alive_lambdas broadcast by target). N-agnostic by construction.
#   0.4 means: only fire when delta_e_tau exceeds 40% of the desired arc.
#   Lowers in larger rings (more agents -> smaller gaps -> smaller threshold).
MIN_EVENT_DELTA_FRAC: float = 0.4

# FAST_CHANNEL_WARMUP_SEC: simulation-time delay before pulse triggers are armed.
# During this initial window every agent is still discovering its neighbors and
# the pred/succ identity can flip a few times spuriously. Suppressing pulses
# during the warmup avoids flooding the channel with phantom events at startup.
FAST_CHANNEL_WARMUP_SEC: float = 1.0


# --------------------------------------------------------------------------------------
# 9c) Dual Pulse propagation method (soliton-inspired, gap-modification integration)
# --------------------------------------------------------------------------------------
#
# Discrete bidirectional pulses with hop counts. Each ring node discovers N and its
# own angular shift target locally — no global topology knowledge required.
#
# Integration path: DualPulseLayer never feeds u_prop via get_neighbor_signal().
# The computed shift enters the control loop according to DUAL_PULSE_INTEGRATION
# below. The DEFAULT is "B2" (2-DOF feedforward with the full cancelling bias) —
# the thesis-validated configuration: stable, flat reconfiguration time
# (~2*T_FF, N-independent up to N=100 with TTL>=N). The legacy gap-bias
# integration remains available via DUAL_PULSE_INTEGRATION=A.

# DUAL_PULSE_INTEGRATION: how the computed redistribution shift enters the control loop.
#   "B2"  : (DEFAULT) feedforward velocity v_ff = (shift/T_FF)*r (gain-INDEPENDENT) plus
#           the FULL cancelling bias (succ+(s_succ-s_self), pred-(s_pred-s_self)) ->
#           feedback ~0 on-plan (NO double-drive); the feedforward is the SOLE driver.
#   "B"   : like B2 but with the MINIMAL cancelling bias (succ+s_succ, pred-s_pred) ->
#           feedback ~2*s_self (mild double-drive). Kept for the ablation study.
#   "A"   : legacy Option A — bias the gaps by the agent's OWN shift; the spacing
#           controller's reflex drives the motion (execution GATED by the controller gain).
#   "OFF" : dual_pulse layer runs (pulses circulate) but does NOT touch control (pure baseline).
DUAL_PULSE_INTEGRATION: str = _os.environ.get("DUAL_PULSE_INTEGRATION", "B2").strip().upper()
if DUAL_PULSE_INTEGRATION not in ("A", "B", "B2", "OFF"):
    DUAL_PULSE_INTEGRATION = "B2"

# DUAL_PULSE_T_FF: feedforward time constant (s) for modes B/B2. The agent consumes its
# remaining shift at rate shift/T_FF, so the redistribution executes with a ~T_FF time
# constant INDEPENDENT of the controller gain. Default follows the approved adaptation
# rule T_FF = tau_a (= VM_TAU_XY, the actuation time constant; c_FF = 1.0): the
# feedforward should not command faster than the platform can actuate. At the default
# VM_TAU_XY=1.0 this equals the legacy literal 1.0.
try:
    DUAL_PULSE_T_FF: float = float(_os.environ.get("DUAL_PULSE_T_FF", str(VM_TAU_XY)))
except ValueError:
    DUAL_PULSE_T_FF = VM_TAU_XY
if DUAL_PULSE_T_FF < 0.0:
    DUAL_PULSE_T_FF = VM_TAU_XY

# DUAL_PULSE_TTL_HOPS: pulse discarded after this many forwards. Safety BACKSTOP only
# (the refractory_cache already self-terminates a pulse after ~one ring). MUST be >= ~N:
# a receiver needs the long-way pulse up to ~N-1 hops and the originator needs ~N hops for
# its return, so TTL < N TRUNCATES the redistribution (measured: TTL=50 collapsed shift
# coverage 96% -> 36% -> 1% at N=50/75/100). A large value does NOT hurt small N (the
# refractory cache stops the pulse after ~one ring regardless). Default scales with the
# ring size: 3*NUM_AGENTS with a floor of 50 (= the legacy literal at the default N=10).
# Env-overridable (absolute value) for TTL studies.
try:
    DUAL_PULSE_TTL_HOPS: int = int(_os.environ.get("DUAL_PULSE_TTL_HOPS", str(max(50, 3 * NUM_AGENTS))))
except ValueError:
    DUAL_PULSE_TTL_HOPS = max(50, 3 * NUM_AGENTS)

# DUAL_PULSE_MULTIPLICITY (M-mult, campaign Ciclo 2): repair the MAGNITUDE
# under-correction when ADJACENT drones fail simultaneously. Only ONE pulse fires
# in that case (the predecessor of the FIRST dead drone survives; the others'
# predecessors are themselves dead), and it reads the correct alive ring size
# N_new via hop-count, but the delta formula assumes a SINGLE removal
# (n_old = n_new + 1) -> it delivers ~1/k of the needed shift -> the slow baseline
# cleans the rest (measured tau 13.6/15.5 s for adj2/adj3 vs ~2 s).
# With this flag the surviving originator infers the death multiplicity k from its
# OWN post-event succ_gap (~(k+1)*ideal_gap, since k frozen dead drones lie between
# it and its new successor -- NEIGHBOR-ONLY) and stamps k on the pulse; receivers
# and the originator self-shift then use n_old = n_new + k. Reduces EXACTLY to the
# legacy behavior at k=1 (single / non-adjacent failures).
# DEFAULT ON (campaign Ciclo 2, validated): fixes adjacent-block sub-correction
# (adj2/adj3 tau 13.6/15.5 -> 2.2 s, ~6-7x) with ZERO regression elsewhere
# (single/non-adjacent byte-identical at k=1; churn identical/slightly better --
# the k>1 path only engages on a genuine adjacent block). Disable via env for the
# ablation. See docs/experiments/CAMPAIGN_LOG.md (Ciclo 2).
DUAL_PULSE_MULTIPLICITY: bool = (
    _os.environ.get("DUAL_PULSE_MULTIPLICITY", "True").strip().lower() in ("true", "1", "yes", "y")
)
# Safety clamp on the inferred k: an implausibly large block is measurement noise
# (non-uniform pre-event spacing), so cap it and fall back toward k=1.
try:
    DUAL_PULSE_MAX_MULTIPLICITY: int = int(_os.environ.get("DUAL_PULSE_MAX_MULTIPLICITY", "6"))
except ValueError:
    DUAL_PULSE_MAX_MULTIPLICITY = 6
if DUAL_PULSE_MAX_MULTIPLICITY < 1:
    DUAL_PULSE_MAX_MULTIPLICITY = 1

# DUAL_PULSE_GAP_CLIP_FRAC: shift_remaining is clipped so that virtual gaps stay
# positive with margin. Value is fraction of the smaller real gap.
DUAL_PULSE_GAP_CLIP_FRAC: float = 0.8

# DUAL_PULSE_MIN_RING_SIZE: minimum N_new for the redistribution math to be
# meaningful. Below this, the layer skips the delta_D computation.
DUAL_PULSE_MIN_RING_SIZE: int = 3

# DUAL_PULSE_DELTA_SCALE: multiplicative factor applied to delta_D (receivers)
# and delta_orig (originator). Range: (0, 1]. The default is MODE-DEPENDENT:
#   - B/B2 (feedforward): 1.0 — deliver the full analytical shift. Validated:
#     with the B2 full cancelling bias there is no feedback double-drive, so
#     the full shift executes cleanly; scale < 1 leaves a slow residual tail
#     for the O(N^2) feedback to clean (measured on the scaling-law campaign).
#   - A (legacy gap-bias): 0.5 — scale = 1.0 overshoots under Option A because
#     execution runs through the controller gain (M3 jitter / M5 effort
#     degrade, measured); 0.5 was the tuned operating point.
# Env-overridable for sweeps.
_DPS_DEFAULT = 1.0 if DUAL_PULSE_INTEGRATION in ("B", "B2") else 0.5
try:
    DUAL_PULSE_DELTA_SCALE: float = float(_os.environ.get("DUAL_PULSE_DELTA_SCALE", str(_DPS_DEFAULT)))
except ValueError:
    DUAL_PULSE_DELTA_SCALE = _DPS_DEFAULT
if not (0.0 < DUAL_PULSE_DELTA_SCALE <= 1.0):
    DUAL_PULSE_DELTA_SCALE = _DPS_DEFAULT

# DUAL_PULSE_RAMP_TICKS: number of control ticks over which a freshly
# computed delta_D is ramped into the applied shift, instead of being
# applied instantaneously. Range: positive integer (1 = no ramp, recovers
# the v1 step behaviour). Default 4 ticks (~40ms at CONTROL_PERIOD=0.01)
# attenuates the controller transient that the instantaneous step caused
# (M3 jitter, M5 effort) without unduly slowing the redistribution. The
# RAMP_TICKS sweep on dense Phase 3 confirmed K=4 matches K=6/K=8 byte-
# for-byte on M1/M3/M5/M7 while leaving the smallest residual shift at
# sim end and giving the fastest response to events. K>=12 starts losing
# M7_settle_median because the ramp becomes slow enough for e_tau_real
# to cross the settling threshold mid-ramp.
# Tunable via env var DUAL_PULSE_RAMP_TICKS for sweeps.
try:
    DUAL_PULSE_RAMP_TICKS: int = int(_os.environ.get("DUAL_PULSE_RAMP_TICKS", "4"))
except ValueError:
    DUAL_PULSE_RAMP_TICKS = 4
if DUAL_PULSE_RAMP_TICKS < 1:
    DUAL_PULSE_RAMP_TICKS = 1

# DUAL_PULSE_BROADCAST_REPEATS: how many consecutive ticks each outgoing pulse is
# re-broadcast (redundancy against intra-tick firing order / packet loss). Default 2.
# env-overridable (Phase 3 - robustness: bandwidth vs robustness trade-off under packet loss).
try:
    DUAL_PULSE_BROADCAST_REPEATS: int = int(_os.environ.get("DUAL_PULSE_BROADCAST_REPEATS", "2"))
except ValueError:
    DUAL_PULSE_BROADCAST_REPEATS = 2
if DUAL_PULSE_BROADCAST_REPEATS < 1:
    DUAL_PULSE_BROADCAST_REPEATS = 1

# DUAL_PULSE_N_CLIP: rejects events whose inferred ring size (hop-count) is IMPOSSIBLE
# (< MIN_RING or > NUM_AGENTS) -> does not apply a corrupted delta_D. Mitigates N corruption under
# churn (topology changes during the pulse's flight). Default False (env A/B). Phase 3 Track B.
DUAL_PULSE_N_CLIP: bool = (
    _os.environ.get("DUAL_PULSE_N_CLIP", "False").strip().lower() in ("true", "1", "yes", "y")
)

# DUAL_PULSE_GATE_*: "gating" of the overlay under dense churn (Phase 3 Track B). When topology
# events become frequent (> GATE_MAX_EVENTS in GATE_WINDOW s) the agent SUPPRESSES injection and
# DECAYS the shift -> degrades to the baseline (safety net) instead of interfering. Default off.
DUAL_PULSE_GATE_ENABLE: bool = (
    _os.environ.get("DUAL_PULSE_GATE_ENABLE", "False").strip().lower() in ("true", "1", "yes", "y")
)
try:
    DUAL_PULSE_GATE_WINDOW: float = float(_os.environ.get("DUAL_PULSE_GATE_WINDOW", "5.0"))
except ValueError:
    DUAL_PULSE_GATE_WINDOW = 5.0
try:
    DUAL_PULSE_GATE_MAX_EVENTS: int = int(_os.environ.get("DUAL_PULSE_GATE_MAX_EVENTS", "1"))
except ValueError:
    DUAL_PULSE_GATE_MAX_EVENTS = 1

# DUAL_PULSE_CONSUME_FF_ONLY (M8, Phase 3 redesign): in consume_motion, deduct only the
# redistribution rotation COMMANDED by the feedforward ((v_ff/r)*dt, already clipped by saturation),
# NOT the total measured Δθ (which under MANEUVER includes the tracking rotation -> under-
# redistribution). Only applies to B/B2 (where v_ff exists). Default off (A/B). Constant: no effect
# (Δθ_ff ≈ Δθ_total).
# DEFAULT TRUE (validated): safe in every regime (stationary/constant = no-op; maneuver = helps,
# including churn+maneuver 1.16-1.20; tracking intact). Disable with DUAL_PULSE_CONSUME_FF_ONLY=False.
DUAL_PULSE_CONSUME_FF_ONLY: bool = (
    _os.environ.get("DUAL_PULSE_CONSUME_FF_ONLY", "True").strip().lower() in ("true", "1", "yes", "y")
)

# DUAL_PULSE_USE_STAMPED_N (M2-core, Phase 3 redesign): δ_D uses the ring SIZE STAMPED by the
# originator at injection (its alive count = M_eff), instead of the hop-sum (h_CCW+h_CW+1) inferred
# DURING the pulse's flight (which becomes stale/inconsistent under churn). h_CCW (position) still
# comes from the hop. A value consistent per event -> coherent corrections. Default off (A/B).
DUAL_PULSE_USE_STAMPED_N: bool = (
    _os.environ.get("DUAL_PULSE_USE_STAMPED_N", "False").strip().lower() in ("true", "1", "yes", "y")
)

# DUAL_PULSE_IDEMPOTENT (M5, Phase 3 redesign): on arrival of a δ_D, OVERWRITES shift_target
# (= the last event DEFINES the redistribution target) instead of ACCUMULATING (+=). Under dense
# churn, accumulation stacks corrections computed for rings that no longer exist (~20/24 nodes in
# permanent transit -> agitation). Overwriting tests whether discarding the history reduces the
# agitation. For a single event it is equivalent to summing (shift_target starts from 0). Default off (A/B).
DUAL_PULSE_IDEMPOTENT: bool = (
    _os.environ.get("DUAL_PULSE_IDEMPOTENT", "False").strip().lower() in ("true", "1", "yes", "y")
)

# DUAL_PULSE_ADD_IF_SETTLED (CONDITIONAL accumulation, Phase 3 redesign): only ACCUMULATES the δ_D
# if the agent was SETTLED (|shift_remaining| < DUAL_PULSE_SETTLED_EPS) when it arrived -> VALID δ_D
# (uniform local ring = the calculation's hypothesis). If the agent was moving (redistributing), the
# δ_D was computed for a config that does not exist -> INVALID -> DISCARD. Aims for the best of both:
# sums the valid ones (simultaneous failures/discrete events) and discards the stale ones (churn). Default off.
DUAL_PULSE_ADD_IF_SETTLED: bool = (
    _os.environ.get("DUAL_PULSE_ADD_IF_SETTLED", "False").strip().lower() in ("true", "1", "yes", "y")
)
try:
    DUAL_PULSE_SETTLED_EPS: float = float(_os.environ.get("DUAL_PULSE_SETTLED_EPS", "0.02"))
except ValueError:
    DUAL_PULSE_SETTLED_EPS = 0.02

# DUAL_PULSE_ALPHA_CLOSE_RATIO: hop-distance attenuation of delta_D.
# Receivers close to the dead/recovered drone D already feel a strong
# physical perturbation from the gap discontinuity at the event moment.
# Adding the full algorithmic delta_D on top of that physical jolt amplifies
# M3 jitter without proportional M7 settling benefit (verified empirically).
# This ratio attenuates delta_D for those nearby receivers while keeping it
# near 1.0 (= no attenuation) for receivers near the anchor / antipode of D.
#
# Linear interpolation:  alpha(h) = ALPHA_CLOSE_RATIO + (1 - ALPHA_CLOSE_RATIO) * d_norm
# where d_norm = (hop distance from D in the affected ring) / (N_new / 2),
# clipped to [0, 1].
#
# Setting ALPHA_CLOSE_RATIO = 1.0 disables the hop attenuation (recovers the
# uniform DUAL_PULSE_DELTA_SCALE behavior). Lower values attenuate immediate
# neighbors more aggressively; the per-hop factor is multiplied on top of
# DUAL_PULSE_DELTA_SCALE.
try:
    DUAL_PULSE_ALPHA_CLOSE_RATIO: float = float(
        _os.environ.get("DUAL_PULSE_ALPHA_CLOSE_RATIO", "0.7")
    )
except ValueError:
    DUAL_PULSE_ALPHA_CLOSE_RATIO = 0.7
if not (0.0 <= DUAL_PULSE_ALPHA_CLOSE_RATIO <= 1.0):
    DUAL_PULSE_ALPHA_CLOSE_RATIO = 0.7

# DUAL_PULSE_ALPHA_CURVE_POWER: exponent shaping how alpha(d_norm) interpolates
# between ALPHA_CLOSE_RATIO (at d=0, immediate neighbor of D) and 1.0 (at the
# anchor / antipode of D in the affected ring).
#
# Formula:  alpha(d_norm) = ALPHA_CLOSE_RATIO + (1 - ALPHA_CLOSE_RATIO) * d_norm^p
#
# p = 1.0  → linear (default v1 behaviour)
# p > 1.0  → convex from below: attenuation stays near ALPHA_CLOSE_RATIO for
#            small d, snaps toward 1.0 only near the anchor. Concentrates the
#            attenuation budget on receivers close to D (where M3 jitter is
#            empirically worst), preserving full delta for mid/distant nodes.
# p < 1.0  → concave: attenuation drops off fast away from D. Almost no
#            attenuation past the immediate neighbour. Rarely useful.
#
# Tunable via env var DUAL_PULSE_ALPHA_CURVE_POWER for sweeps.
try:
    DUAL_PULSE_ALPHA_CURVE_POWER: float = float(
        _os.environ.get("DUAL_PULSE_ALPHA_CURVE_POWER", "1.0")
    )
except ValueError:
    DUAL_PULSE_ALPHA_CURVE_POWER = 1.0
if DUAL_PULSE_ALPHA_CURVE_POWER <= 0.0:
    DUAL_PULSE_ALPHA_CURVE_POWER = 1.0

# DUAL_PULSE_SLEEP_THRESHOLD: when the magnitude of the shift to be applied to
# the gaps is below this threshold (radians), the gap-bias step is skipped
# entirely — the controller sees the real gaps. This prevents subtle
# interference between dual_pulse and tracking dynamics during the "no
# in-flight events" regime, which empirically caused a small M1 degradation
# (~+7-12%) when the target is moving.
#
# Why a threshold (vs always-on bias):
#   - Above threshold: shift carries meaningful redistribution information,
#     bias yields the M7 settling-time gain we want.
#   - Below threshold: shift is a tiny residual; the local controller
#     suffices, and skipping bias avoids the small but persistent jitter
#     that the controller picks up tracking a near-zero virtual offset.
#
# 0.01 rad ≈ 0.57° ≈ 0.2 m arc at r=20m. Default chosen well below typical
# delta_D magnitudes (~0.1-0.3 rad) so all real events still trigger bias.
# Set to 0.0 to disable the sleep gate (recovers v1.6 behaviour).
try:
    DUAL_PULSE_SLEEP_THRESHOLD: float = float(
        _os.environ.get("DUAL_PULSE_SLEEP_THRESHOLD", "0.01")
    )
except ValueError:
    DUAL_PULSE_SLEEP_THRESHOLD = 0.01
if DUAL_PULSE_SLEEP_THRESHOLD < 0.0:
    DUAL_PULSE_SLEEP_THRESHOLD = 0.0

# NOTE: DUAL_PULSE_INTEGRATION and DUAL_PULSE_T_FF are defined at the TOP of this
# section (9c) because DUAL_PULSE_DELTA_SCALE's default depends on the mode.


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
# Env-overridable (campaign rule 3: sweeps pin the metric window explicitly in the
# child env rather than relying on this default). Default unchanged.
try:
    METRICS_T0: float = float(_os.environ.get("METRICS_T0", "0.0"))
except ValueError:
    METRICS_T0 = 0.0

# METRICS_E_THR: absolute error threshold for settling time M7, using e(t)=|e_tau(t)|.
# Choose a value that is meaningful in your normalized e_tau scale (often 0.01~0.05).
METRICS_E_THR: float = 0.05

# METRICS_MA_W_SEC: moving-average window (seconds) used by M4 (oscillation metric).
# Example: 1.0 means "remove the slow trend with a 1-second moving average".
METRICS_MA_W_SEC: float = 1.0

# METRICS_SETTLE_WINDOW_SEC: continuous time window (seconds) required for M7 settling.
# Example: 2.0 means "consider settled when e(t)<=E_THR continuously for 2 seconds".
METRICS_SETTLE_WINDOW_SEC: float = 5.0
