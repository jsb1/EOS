#!/usr/bin/env python3
"""Test script to debug DP EV terminal penalty behavior.

Uses optimize_input_2.json with penalty_factor=50 to verify:
1. V[horizon] initialization with correct penalty_factor
2. Bellman equation considers terminal penalty at each step
3. DP selects EV=80% (0€ penalty) instead of EV=5% (3750€ penalty)
"""

import json
from pathlib import Path
from unittest.mock import patch

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from akkudoktoreos.config.config import ConfigEOS
from akkudoktoreos.core.coreabc import get_ems, singletons_init, get_config


def main():
    # Initialize EMS
    ems = get_ems(init=True)

    # Load test data
    test_data_path = Path(__file__).parent.parent.parent / "tests/testdata/optimize_input_2.json"
    with test_data_path.open("r") as f:
        input_data_dict = json.load(f)

    from akkudoktoreos.optimization.dp.dpparams import DPOptimizationParameters

    config = ConfigEOS()
    config.merge_settings_from_dict(
        {
            "prediction": {"hours": 48},
            "optimization": {
                "horizon_hours": 48,
                "algorithm": "DP",
                "genetic": {
                    "individuals": 100,
                    "generations": 10,
                    "penalties": {
                        "ev_soc_miss": 50,  # High penalty factor
                        "ac_charge_break_even": 0,
                    },
                },
            },
            "devices": {
                "max_electric_vehicles": 1,
                "electric_vehicles": [{"charge_rates": [0.0, 0.5, 1.0]}],
            },
        }
    )

    # Verify config
    penalty_factor = config.optimization.genetic.penalties.get("ev_soc_miss", 10)
    print(f"Config penalty_factor: {penalty_factor}")

    # Create DP optimizer with config
    from akkudoktoreos.optimization.dp.dpoptimizer import DPOptimizer

    # Convert to DPOptimizationParameters
    input_data = DPOptimizationParameters(**input_data_dict)

    # Print EV parameters
    ev = input_data.ev
    print(f"\nEV Parameters:")
    print(f"  capacity_wh: {ev.capacity_wh}")
    print(f"  initial_soc_percentage: {ev.initial_soc_percentage}")
    print(f"  min_soc_percentage: {ev.min_soc_percentage}")
    print(f"  max_soc_percentage: {ev.max_soc_percentage}")

    # Run DP optimization
    print("\n" + "=" * 60)
    print("Running DP optimization with penalty_factor=50...")
    print("=" * 60)

    dp_optimizer = DPOptimizer()
    solution = dp_optimizer.optimize(
        params=input_data,
        ha_params=input_data.dishwasher,
        start_hour=10,
        worst_case=False,
        optimize_ev=True,
        optimize_dc_charge=True,
    )

    # Print results
    print("\n" + "=" * 60)
    print("DP Solution Results:")
    print("=" * 60)
    print(f"Total states explored: {solution.total_states_explored}")
    print(f"Computation time: {solution.computation_time_ms:.2f}ms")

    # Check EV charging
    if solution.ev_charge_hours_float is not None:
        ev_charging_hours = sum(1 for h in solution.ev_charge_hours_float if h > 0)
        ev_total_charge = sum(solution.ev_charge_hours_float)
        print(f"\nEV Charging:")
        print(f"  Hours charging: {ev_charging_hours}/48")
        print(f"  Total charge factors: {ev_total_charge:.2f}")

        # Show charging schedule (first 12 hours)
        print(f"  Schedule (first 12h): {[round(h, 2) for h in solution.ev_charge_hours_float[:12]]}")

    # Check battery discharging
    discharge_hours = sum(solution.discharge_allowed)
    print(f"\nBattery:")
    print(f"  Discharge hours: {discharge_hours}/48")
    print(f"  Start SOC index: {solution.dp_start_soc_index}")
    print(f"  End SOC index: {solution.dp_end_soc_index}")

    # Check simulation result
    if solution.result is not None:
        print(f"\nSimulation Result:")
        print(f"  Total Balance: {solution.result.total_balance:.2f}")

        # Check final EV SOC from simulation
        if solution.result.ev_soc_per_hour is not None:
            final_ev_soc = solution.result.ev_soc_per_hour[-1]
            print(f"  Final EV SOC: {final_ev_soc:.1f}%")

            # Show EV SOC progression
            print(f"  EV SOC progression (first 12h): {[round(s, 1) for s in solution.result.ev_soc_per_hour[:12]]}")


if __name__ == "__main__":
    main()
