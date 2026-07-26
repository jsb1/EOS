#!/usr/bin/env python3
"""Analyze DP decisions step by step for optimize_input_2.json."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from akkudoktoreos.config.config import ConfigEOS
from akkudoktoreos.core.coreabc import get_ems
from akkudoktoreos.optimization.simulation.parameters import OptimizationParameters
from akkudoktoreos.optimization.dp.dpoptimizer import DPOptimizer
from akkudoktoreos.optimization.dp.dpparams import DPOptimizationParameters

def load_input(filepath: Path) -> dict:
    with open(filepath) as f:
        return json.load(f)

def main():
    input_file = Path("tests/testdata/optimize_input_2.json")
    input_data = load_input(input_file)
    
    # Initialize EMS
    ems_eos = get_ems(init=True)
    
    # Configure
    config = ConfigEOS()
    config.merge_settings_from_dict(
        {
            "optimization": {
                "genetic": {
                    "individuals": 100,
                    "generations": 50,
                    "penalties": {"ev_soc_miss": 10, "ac_charge_break_even": 0},
                },
            },
            "devices": {
                "max_electric_vehicles": 1,
                "electric_vehicles": [{"charge_rates": [0.0, 0.5, 1.0]}],
            },
        }
    )
    
    # Prepare params
    params = DPOptimizationParameters(**input_data)
    
    optimizer = DPOptimizer()
    solution = optimizer.optimize(params)
    
    print("DP Solution Decisions:")
    print("=" * 100)
    
    ems = input_data["ems"]
    prices = ems["strompreis_euro_pro_wh"]
    pv = ems["pv_prognose_wh"]
    load = ems["gesamtlast"]
    
    for i in range(0, 48, 4):
        print(f"\nHour {i:2d}:")
        print(f"  PV: {pv[i]:.0f} Wh, Load: {load[i]:.0f} Wh, Price: {prices[i]:.6f} €/Wh")
        print(f"  AC Charge: {solution.ac_charge[i]:.2f}, DC Charge: {solution.dc_charge[i]:.0f}, Discharge: {solution.discharge_allowed[i]}")
        if solution.ev_charge_hours_float:
            print(f"  EV Charge: {solution.ev_charge_hours_float[i]:.2f}")
        print(f"  Battery SoC: {solution.result.battery_soc_per_hour[i]:.1f}%")
        if solution.result.ev_soc_per_hour:
            print(f"  EV SoC: {solution.result.ev_soc_per_hour[i]:.1f}%")
        print(f"  Grid Import: {solution.result.grid_consumption_wh_per_hour[i]:.0f} Wh")
        print(f"  Grid Export: {solution.result.grid_feed_in_wh_per_hour[i]:.0f} Wh")
        print(f"  Cost: {solution.result.costs_per_hour[i]:.4f} €")
    
    print("\n" + "=" * 100)
    print(f"Total Cost: {solution.result.total_costs:.4f} €")
    print(f"Total Revenue: {solution.result.total_revenue:.4f} €")
    print(f"Total Balance: {solution.result.total_balance:.4f} €")
    print(f"Battery End SoC: {solution.result.battery_soc_per_hour[-1]:.1f}%")
    print(f"EV End SoC: {solution.result.ev_soc_per_hour[-1]:.1f}%")

if __name__ == "__main__":
    main()
