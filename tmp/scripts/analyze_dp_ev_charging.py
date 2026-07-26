#!/usr/bin/env python3
"""Analyze DP EV charging decisions in detail."""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main():
    input_path = Path("tests/testdata/optimize_input_2.json")
    with open(input_path) as f:
        raw_data = json.load(f)

    # Set penalty_factor directly before creating DPOptimizer
    # This works because DPOptimizer reads from the global config singleton
    from akkudoktoreos.core.coreabc import get_config
    config = get_config()
    config.optimization.genetic.penalties["ev_soc_miss"] = 50
    config.optimization.genetic.penalties["ac_charge_break_even"] = 0
    print(f"Set penalty_factor to: {config.optimization.genetic.penalties.get('ev_soc_miss')}")

    from akkudoktoreos.optimization.dp.dpoptimizer import DPOptimizer
    from akkudoktoreos.optimization.dp.dpparams import DPOptimizationParameters

    # Create DP params
    dp_params = DPOptimizationParameters(**raw_data)

    # Run DP
    print("Running DP...")
    dp_opt = DPOptimizer()
    dp_solution = dp_opt.optimize(params=dp_params, ha_params=None, start_hour=10, worst_case=False)
    print("DP done.")

    result = dp_solution.result
    print(f"\nBalance: {result.total_balance:.4f}€")
    print(f"Total Cost: {result.total_costs:.4f}€")
    print(f"Battery Final SoC: {result.battery_soc_per_hour[-1]:.2f}%")
    print(f"EV Final SoC: {result.ev_soc_per_hour[-1]:.2f}%")

    # Check EV charging
    ev_charge = dp_solution.ev_charge_hours_float or []
    ac_charge = dp_solution.ac_charge
    discharge = dp_solution.discharge_allowed

    print(f"\nDP Solution ev_charge_hours_float: {ev_charge}")
    print(f"AC Charge: {ac_charge[:10]}...")
    print(f"Discharge: {discharge[:10]}...")

    # Count actions
    ev_hours = sum(1 for x in ev_charge if x > 0.01)
    ac_hours = sum(1 for x in ac_charge if x > 0.01)
    dis_hours = sum(1 for x in discharge if x == 1)

    print(f"\nEV Charge Hours (rate>0.01): {ev_hours}")
    print(f"AC Charge Hours (rate>0.01): {ac_hours}")
    print(f"Discharge Hours: {dis_hours}")

    # Show hourly breakdown
    ems = raw_data["ems"]
    horizon = len(ems["strompreis_euro_pro_wh"])

    print(f"\n{'Hour':>4} | {'AC':>4} | {'Dis':>4} | {'EV':>5} | {'Bat%':>6} | {'EV%':>6} | {'Grid':>8}")
    print("-" * 55)

    for t in range(horizon):
        ac = ac_charge[t] if t < len(ac_charge) else 0
        dis = discharge[t] if t < len(discharge) else 0
        ev_c = ev_charge[t] if t < len(ev_charge) else 0
        bs = result.battery_soc_per_hour[t] if t < len(result.battery_soc_per_hour) else 0
        es = result.ev_soc_per_hour[t] if t < len(result.ev_soc_per_hour) else 0
        g = result.grid_consumption_wh_per_hour[t] if t < len(result.grid_consumption_wh_per_hour) else 0

        print(f"{t:>4} | {ac:>4.1f} | {dis:>4} | {ev_c:>5.2f} | {bs:>6.1f} | {es:>6.1f} | {g:>8.0f}")


if __name__ == "__main__":
    main()
