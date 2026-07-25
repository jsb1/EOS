"""Solver-agnostic energy simulation package.

Provides a clean separation between simulation physics and optimization
algorithms, allowing multiple solvers (GA, DP, MPC, LP) to share the
same simulation engine.

Submodules:
    physics: Pure stateless functions for SOC transitions and energy balance.
    step: EnergySimulationStep dataclass (per-hour result).
    result: SimulationResult (lightweight dataclass).
    devices: DeviceFactory and SimulationDevices container.
    engine: EnergySimulationEngine (main simulation orchestrator).
"""

from akkudoktoreos.optimization.simulation.context import SimulationContext
from akkudoktoreos.optimization.simulation.physics import (
    compute_battery_next_soc,
    compute_ev_next_soc,
)
from akkudoktoreos.optimization.simulation.step import EnergySimulationStep
from akkudoktoreos.optimization.simulation.result import SimulationResult
from akkudoktoreos.optimization.simulation.devices import (
    DeviceFactory,
    SimulationDevices,
)
from akkudoktoreos.optimization.simulation.engine import EnergySimulationEngine
from akkudoktoreos.optimization.simulation.session import SimulationSession
from akkudoktoreos.optimization.simulation.solution import (
    SimulationSolution,
)
from akkudoktoreos.optimization.simulation.parameters import (
    OptimizationParameters,
)

__all__ = [
    # Context
    "SimulationContext",
    # Physics
    "compute_battery_next_soc",
    "compute_ev_next_soc",
    # Step
    "EnergySimulationStep",
    # Result
    "SimulationResult",
    # Devices
    "DeviceFactory",
    "SimulationDevices",
    # Engine
    "EnergySimulationEngine",
    # Session
    "SimulationSession",
    # Solution
    "SimulationSolution",
    # Parameters
    "OptimizationParameters",
]
