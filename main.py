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
from protocol_target import TargetProtocol  # noqa: E402
from protocol_agent import AgentProtocol  # noqa: E402
from protocol_adversary import AdversaryProtocol  # noqa: E402
from velocity_mobility import VelocityMobilityHandler, VelocityMobilityConfiguration  # noqa: E402
from config_param import (  # noqa: E402
    COMMUNICATION_DELAY,
    COMMUNICATION_FAILURE_RATE,
    COMMUNICATION_TRANSMISSION_RANGE,
    ENCIRCLEMENT_RADIUS,
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
    ("baseline",  "Controlador atual — sem propagação (referência de comparação)"),
    ("advection", "Advecção-Difusão Amortecida Bidirecional"),
    ("wave",      "Onda de Segunda Ordem"),
    ("excitable", "Meio Excitável — FitzHugh-Nagumo"),
    ("kdv",       "KdV Discreto — Soliton-Inspired"),
    ("alarm",     "Alarmes Discretos com TTL"),
    ("burgers",   "Burgers Amortecido com Saturação"),
]


def _select_propagation_method() -> tuple:
    print("\n=== Seleção do Método de Propagação ===")
    for i, (key, desc) in enumerate(_METHODS):
        print(f"  [{i}] {key:12s} — {desc}")
    while True:
        try:
            choice = int(input(f"\nEscolha o método [0-{len(_METHODS) - 1}]: ").strip())
            if 0 <= choice < len(_METHODS):
                break
        except (ValueError, EOFError):
            pass
        print(f"  Entrada inválida. Digite um número entre 0 e {len(_METHODS) - 1}.")
    method = _METHODS[choice][0]

    k_prop = 0.0
    if method != "baseline":
        while True:
            try:
                raw = input("  Ganho K_PROP (sugestão: 1.0, Enter para 1.0): ").strip()
                k_prop = float(raw) if raw else 1.0
                if k_prop >= 0.0:
                    break
            except (ValueError, EOFError):
                pass
            print("  Valor inválido. Digite um número ≥ 0.")

    print(f"\n  → Método: {method}  |  K_PROP: {k_prop}\n")
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
    if EXPERIMENT_REPRODUCIBLE:
        random.seed(0)

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
        f.write("timestamp,E_r,E_vr,rho,G_max,E_gap\n")

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

    # Add agent nodes at random positions around the target
    # and randomly vary the desired encirclement radius

    num_agents = NUM_AGENTS # Number of agent nodes
    encirclement_radius = ENCIRCLEMENT_RADIUS # Desired encirclement radius in meters
    for i in range(num_agents):
        angle = random.uniform(0, 2 * math.pi)
        x = encirclement_radius * random.uniform(0.8, 1.2) * math.cos(angle)
        y = encirclement_radius * random.uniform(0.8, 1.2) * math.sin(angle)
        z = 0.0 # Keep agents at ground level
        builder.add_node(AgentProtocol, (x, y, z))

    simulation = builder.build()
    simulation.start_simulation()

if __name__ == "__main__":
    main()
