#!/usr/bin/env python3
"""Analyze DP strategy vs GA for optimize_input_2.json."""

import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from akkudoktoreos.config.config import ConfigEOS
from akkudoktoreos.optimization.simulation.session import SimulationSession
from akkudoktoreos.optimization.simulation.parameters import OptimizationParameters
from akkudoktoreos.optimization.dp.dpoptimizer import DPOptimizer
from akkudoktoreos.optimization.dp.dpparams import DPOptimizationParameters

def load_input(filepath: Path) -> dict:
    with open(filepath) as f:
        return json.load(f)

def create_session_from_input(input_data: dict) -> SimulationSession:
    """Create a SimulationSession from input JSON."""
    ConfigEOS.set_config(input_data)
    params = OptimizationParameters.prepare_sync()
    session = SimulationSession.prepare(params)
    return session

def analyze_dp_solution(session: SimulationSession, params: OptimizationParameters):
    """Run DP and analyze the solution."""
    optimizer = DPOptimizer()
    solution = optimizer.optimize(params)
    
    print("DP Solution:")
    print(f"  Total Cost: {solution.simulation_result.total_costs:.4f}")
    print(f"  Total Balance: {solution.simulation_result.total_balance:.4f}")
    print(f"  Battery End SoC: {solution.simulation_result.battery_soc_per_hour[-1]:.1f}%")
    print(f"  EV End SoC: {solution.simulation_result.ev_soc_per_hour[-1]:.1f}%")
    print(f"  Total Losses: {solution.simulation_result.total_losses:.0f} Wh")
    print(f"  AC Charge Count: {sum(1 for x in solution.ac_charge if x > 0)}")
    
    # Show battery SoC trajectory
    print("\n  Battery SoC trajectory:")
    for i in range(0, 48, 4):
        print(f"    Hour {i:2d}: {solution.simulation_result.battery_soc_per_hour[i]:.1f}%")
    
    # Show EV SoC trajectory
    print("\n  EV SoC trajectory:")
    for i in range(0, 48, 4):
        print(f"    Hour {i:2d}: {solution.simulation_result.ev_soc_per_hour[i]:.1f}%")
    
    # Show grid consumption
    print("\n  Grid consumption by hour:")
    for i in range(0, 48, 8):
        grid = solution.simulation_result.grid_consumption_wh_per_hour[i:i+8]
        print(f"    Hours {i:2d}-{i+7:2d}: {sum(grid):.0f} Wh")
    
    return solution

def main():
    input_file = Path("tests/testdata/optimize_input_2.json")
    input_data = load_input(input_file)
    
    # Create session
    session = create_session_from_input(input_data)
    
    # Analyze DP solution
    dp_solution = analyze_dp_solution(session, session.optimization_params)
    
    # Show key parameters
    print("\nKey Parameters:")
    ev = session.optimization_params.ev
    bat = session.optimization_params.pv_battery
    print(f"  EV capacity: {ev.capacity_wh} Wh")
    print(f"  EV initial_soc: {ev.initial_soc_percentage}%")
    print(f"  EV min_soc: {ev.min_soc_percentage}%")
    print(f"  EV needs: {(ev.min_soc_percentage - ev.initial_soc_percentage) * ev.capacity_wh / 100:.0f} Wh")
    print(f"  Battery capacity: {bat.capacity_wh} Wh")
    print(f"  Battery initial_soc: {bat.initial_soc_percentage}%")

if __name__ == "__main__":
    main()
