"""This script builds a GrADyS-SIM simulation with a 1 target node and n agent
nodes controlled by a tangential spacing controller to enforce encirclement behavior.
The target node moves according to a random trajectory, while the agent nodes
attempt to encircle the target at a specified radius equally spaced between them.
Run:
    python main.py
"""

# Suppress websockets handshake warnings
import json as _json
import logging
import math
import os
import random
import sys

logging.getLogger("websockets").setLevel(logging.CRITICAL)


# Ensure local src/ packages are importable without installation.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC_PATH = os.path.join(_REPO_ROOT, "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium  # noqa: E402
from gradysim.simulator.handler.timer import TimerHandler  # noqa: E402
from gradysim.simulator.handler.visualization import VisualizationHandler, VisualizationConfiguration  # noqa: E402
from gradysim.simulator.simulation import SimulationBuilder, SimulationConfiguration  # noqa: E402
from protocol_target import TARGET_TELEMETRY_COLUMNS, TargetProtocol  # noqa: E402
from protocol_agent import AgentProtocol  # noqa: E402
from protocol_adversary import AdversaryProtocol  # noqa: E402
import provenance  # noqa: E402
from comm_role_aware import RoleAwareCommunicationHandler, agent_target_range_matrix  # noqa: E402
from velocity_mobility import VelocityMobilityHandler, VelocityMobilityConfiguration  # noqa: E402
from config_param import (  # noqa: E402
    COMMUNICATION_DELAY,
    COMMUNICATION_FAILURE_RATE,
    COMMUNICATION_TRANSMISSION_RANGE,
    COMM_RANGE_AGENT_AGENT,
    COMM_RANGE_AGENT_TARGET,
    COMM_ROLE_AWARE_RANGES,
    ENCIRCLEMENT_RADIUS,
    INIT_ANGLES_EQUIDISTANT,
    INIT_RADIUS_RANGE,
    NUM_AGENTS,
    SIM_DEBUG,
    SIM_DURATION,
    SIM_REAL_TIME,
    VIS_OPEN_BROWSER,
    VIS_UPDATE_RATE,
    VM_MAX_ACC_XY,
    VM_MAX_ACC_Z,
    VM_MAX_SPEED_XY,
    VM_MAX_SPEED_Z,
    VM_SEND_TELEMETRY,
    VM_TAU_XY,
    VM_TAU_Z,
    VM_TELEMETRY_DECIMATION,
    VM_UPDATE_RATE,
    EXPERIMENT_REPRODUCIBLE,
)


mobility_config = VelocityMobilityConfiguration(
    update_rate=VM_UPDATE_RATE,        # Update every in seconds
    max_speed_xy=VM_MAX_SPEED_XY,      # Max horizontal speed in m/s
    max_speed_z=VM_MAX_SPEED_Z,        # Max vertical speed in m/s
    max_acc_xy=VM_MAX_ACC_XY,          # Max horizontal acceleration in m/s^2
    max_acc_z=VM_MAX_ACC_Z,            # Max vertical acceleration in m/s^2
    tau_xy=VM_TAU_XY,                  # Optional: 1st-order horizontal tracking time constant (s)
    tau_z=VM_TAU_Z,                    # Optional: 1st-order vertical tracking time constant (s)
    send_telemetry=VM_SEND_TELEMETRY,  # Enable telemetry
    telemetry_decimation=VM_TELEMETRY_DECIMATION,  # Send telemetry every update
)


_METHODS = [
    ("baseline",   "Current controller - no propagation (comparison baseline)"),
    ("advection",  "Bidirectional damped advection-diffusion"),
    ("wave",       "Second-order wave"),
    ("excitable",  "Excitable medium - FitzHugh-Nagumo"),
    ("kdv",        "Discrete KdV - soliton-inspired"),
    ("alarm",      "Discrete alarms with TTL"),
    ("burgers",    "Damped Burgers with saturation"),
    ("dual_pulse", "Dual counter-propagating pulses with hop-count topology discovery"),
]

# Methods that do NOT use the u_prop channel (k_prop irrelevant; menu skips
# the prompt). Keep this list in sync with method semantics in propagation_layer.
_METHODS_WITHOUT_K_PROP = {"baseline", "dual_pulse"}


def _select_propagation_method() -> tuple:
    # Non-interactive override for batch sweeps: when PROPAGATION_METHOD is set
    # in the environment, skip the prompt and read the choice directly. This is
    # what run_sweep.py relies on.
    env_method = os.environ.get("PROPAGATION_METHOD")
    if env_method:
        valid = {key for key, _ in _METHODS}
        if env_method not in valid:
            raise ValueError(
                f"PROPAGATION_METHOD={env_method!r} is invalid; expected one of {sorted(valid)}"
            )
        if env_method in _METHODS_WITHOUT_K_PROP:
            k_prop = 0.0
        else:
            try:
                k_prop = float(os.environ.get("PROPAGATION_K_PROP", "1.0"))
            except ValueError:
                k_prop = 1.0
            if not (k_prop == k_prop) or k_prop < 0.0:  # NaN-safe + non-negative guard
                k_prop = 1.0
        print(f"\n  -> Method: {env_method}  |  K_PROP: {k_prop}  (from environment)\n")
        return env_method, k_prop

    print("\n=== Propagation Method Selection ===")
    for i, (key, desc) in enumerate(_METHODS):
        print(f"  [{i}] {key:12s} - {desc}")
    while True:
        try:
            choice = int(input(f"\nChoose a method [0-{len(_METHODS) - 1}]: ").strip())
            if 0 <= choice < len(_METHODS):
                break
        except (ValueError, EOFError):
            pass
        print(f"  Invalid input. Enter a number between 0 and {len(_METHODS) - 1}.")
    method = _METHODS[choice][0]

    k_prop = 0.0
    if method not in _METHODS_WITHOUT_K_PROP:
        while True:
            try:
                raw = input("  K_PROP gain (suggested: 1.0, Enter for 1.0): ").strip()
                k_prop = float(raw) if raw else 1.0
                if k_prop >= 0.0:
                    break
            except (ValueError, EOFError):
                pass
            print("  Invalid value. Enter a number >= 0.")

    print(f"\n  -> Method: {method}  |  K_PROP: {k_prop}\n")
    return method, k_prop


def main():
    """Execute the simulation."""

    method, k_prop = _select_propagation_method()

    # Pass propagation config to AgentProtocol via environment variables
    # (same pattern used for AGENT_LOG_CSV_PATH)
    os.environ["PROPAGATION_METHOD"] = method
    os.environ["PROPAGATION_K_PROP"] = str(k_prop)
    os.environ["PROPAGATION_PARAMS"] = "{}"

    # Global deterministic randomness for reproducibility across the whole project.
    # EXPERIMENT_SEED env var allows Monte Carlo sweeps over multiple seeds
    # without editing source. Default 0 reproduces the legacy single-seed runs.
    if EXPERIMENT_REPRODUCIBLE:
        try:
            _seed = int(os.environ.get("EXPERIMENT_SEED", "0"))
        except ValueError:
            _seed = 0
        random.seed(_seed)

    # Write the run manifest BEFORE the simulation starts. Two reasons:
    #   1) it snapshots argv + the complete resolved config_param + git state, so
    #      the run stays reproducible even if the summary CSV is lost or edited;
    #   2) it primes provenance's git cache with the PRE-run tree state, so the
    #      run's own output files cannot later make the code look uncommitted.
    manifest_path = provenance.write_run_manifest()
    _commit, _dirty = provenance.git_provenance()
    if manifest_path:
        print(f"[provenance] commit={_commit} dirty={_dirty} -> {manifest_path}")
    if _dirty:
        print(
            "[provenance] WARNING: working tree is DIRTY — this run is not "
            "reproducible from the recorded commit alone."
        )

    # Create a shared CSV file for agent telemetry logs.
    csv_path = os.path.join(os.getcwd(), "agent_telemetry.csv")
    os.environ["AGENT_LOG_CSV_PATH"] = csv_path
    # Start with a fresh agent telemetry CSV.
    # IMPORTANT: don't pre-create the file here, because AgentProtocol.finish() writes the header
    # only when the file does not exist.
    if os.path.exists(csv_path):
        os.remove(csv_path)


    # Create a shared CSV file for target telemetry logs.
    # The target will append its rows on protocol finish().
    target_csv_path = os.path.join(os.getcwd(), "target_telemetry.csv")
    os.environ["TARGET_LOG_CSV_PATH"] = target_csv_path
    with open(target_csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(TARGET_TELEMETRY_COLUMNS) + "\n")

    # Sparse event log for the fast soliton-like channel observations.
    # Each agent appends rows in finish(); we wipe the file so each run is fresh.
    events_csv_path = os.path.join(os.getcwd(), "events.csv")
    os.environ["EVENTS_LOG_CSV_PATH"] = events_csv_path
    if os.path.exists(events_csv_path):
        os.remove(events_csv_path)

    # Dual-pulse message-count CSV (appended by AgentProtocol.finish()). Wipe it so a
    # reused working directory does not accumulate rows across runs (which silently
    # inflates the summed message count).
    dpmsg_csv_path = os.path.join(os.getcwd(), "dual_pulse_messages.csv")
    if os.path.exists(dpmsg_csv_path):
        os.remove(dpmsg_csv_path)

    duration = SIM_DURATION        # Simulation duration (seconds)
    real_time = SIM_REAL_TIME      # Run in real time (True) or as-fast-as-possible (False)
    debug = SIM_DEBUG              # Enable simulator debug mode

    builder = SimulationBuilder(
        SimulationConfiguration(
            duration=duration,
            debug=debug,
            real_time=real_time,
        )
    )

    transmission_range = COMMUNICATION_TRANSMISSION_RANGE  # Communication range (meters)
    delay = COMMUNICATION_DELAY                       # Communication delay (seconds)
    failure_rate = COMMUNICATION_FAILURE_RATE         # Packet loss probability [0.0, 1.0]
    medium = CommunicationMedium(
        transmission_range=transmission_range,
        delay=delay,
        failure_rate=failure_rate,
    )
    comm_handler = None
    if COMM_ROLE_AWARE_RANGES:
        # Asymmetric ranges per (sender, receiver) role. See comm_role_aware for
        # why this cannot be done with a sender-side radio.
        comm_handler = RoleAwareCommunicationHandler(
            medium,
            range_matrix=agent_target_range_matrix(
                agent_agent=COMM_RANGE_AGENT_AGENT,
                agent_target=COMM_RANGE_AGENT_TARGET,
            ),
        )
        builder.add_handler(comm_handler)
    else:
        builder.add_handler(CommunicationHandler(medium))

    builder.add_handler(TimerHandler())

    velocity_handler = VelocityMobilityHandler(mobility_config)
    builder.add_handler(velocity_handler)

    vis_config = VisualizationConfiguration(
        open_browser=VIS_OPEN_BROWSER,
        update_rate=VIS_UPDATE_RATE,
    )
    builder.add_handler(VisualizationHandler(vis_config))

    # Add target node at origin
    builder.add_node(TargetProtocol, (0, 0, 0))

    # Add adversary node at a fixed initial position
    builder.add_node(AdversaryProtocol, (40, 40, 0))

    # Add agent nodes around the target.
    # Initial radius and angle distributions are configured via:
    #   - INIT_RADIUS_RANGE: half-width of the uniform radius scatter (fraction of R).
    #     0.0 places every agent exactly at ENCIRCLEMENT_RADIUS.
    #   - INIT_ANGLES_EQUIDISTANT: if True, angles are spaced uniformly by 2*pi/num_agents;
    #     if False, angles are drawn uniformly in [0, 2*pi).
    num_agents = NUM_AGENTS # Number of agent nodes
    encirclement_radius = ENCIRCLEMENT_RADIUS # Desired encirclement radius in meters
    radius_range = max(0.0, INIT_RADIUS_RANGE)
    r_lo = 1.0 - radius_range
    r_hi = 1.0 + radius_range
    for i in range(num_agents):
        if INIT_ANGLES_EQUIDISTANT:
            angle = i * (2 * math.pi / num_agents)
        else:
            angle = random.uniform(0, 2 * math.pi)
        radius_scale = 1.0 if radius_range == 0.0 else random.uniform(r_lo, r_hi)
        x = encirclement_radius * radius_scale * math.cos(angle)
        y = encirclement_radius * radius_scale * math.sin(angle)
        z = 0.0 # Keep agents at ground level
        builder.add_node(AgentProtocol, (x, y, z))

    simulation = builder.build()

    # Emitted only after build(), because that is when the builder registers the
    # nodes and the handler can resolve their roles. Sweep runners parse this
    # line to assert the matrix is not a no-op and that no role came back
    # unknown -- both are silent data-corrupting misconfigurations otherwise.
    if comm_handler is not None:
        print(comm_handler.describe(), flush=True)

    simulation.start_simulation()

    # Post-simulation analysis: print metrics, render per-node telemetry plots,
    # and append one summary row (context + M1..M7) to runs_summary.csv so
    # multiple runs can be compared later.
    import plot_telemetry  # local import: defers matplotlib backend setup until needed
    if os.path.exists(csv_path):
        summary_csv_path = os.environ.get(
            "RUNS_SUMMARY_CSV_PATH",
            os.path.join(os.getcwd(), "runs_summary.csv"),
        )
        plot_telemetry.main(csv_path=csv_path, summary_csv_path=summary_csv_path)
    else:
        print(f"[main] Skipping post-simulation analysis: {csv_path} not found.")

if __name__ == "__main__":
    main()
