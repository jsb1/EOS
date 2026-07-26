#!/usr/bin/env python3
"""Analyze how GA vs DP handle EV charging."""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def set_ems_start_hour(hour: int) -> None:
    """Set the EMS start hour to use for optimization."""
    from akkudoktoreos.core.ems import EnergyManagement
    EnergyManagement.set_start_hour(hour)


def load_input_data(filepath: Path) -> dict:
    """Load JSON input data for optimization."""
    with open(filepath) as f:
        data = json.load(f)
    # Normalize field names
    if "strompreis_euro_pro_wh" in data["ems"]:
        data["ems"]["electricity_price_per_wh"] = data["ems"]["strompreis_euro_pro_wh"]
    if "pv_prognose_wh" in data["ems"]:
        data["ems"]["pv_forecast_wh"] = data["ems"]["pv_prognose_wh"]
    if "gesamtlast" in data["ems"]:
        data["ems"]["total_load"] = data["ems"]["gesamtlast"]
    if "pv_akku" in data:
        data["pv_battery"] = data.pop("pv_akku")
    if "eauto" in data:
        data["ev"] = data.pop("eauto")
    # Add feed_in_tariff if missing
    if "feed_in_tariff_per_wh" not in data["ems"]:
        data["ems"]["feed_in_tariff_per_wh"] = 0.07
    if "price_per_wh_battery" not in data["ems"]:
        data["ems"]["price_per_wh_battery"] = 0.0
    return data


async def run_optimization(input_data: dict, start_hour: int):
    """Run both GA and DP optimizations."""
    from akkudoktoreos.optimization.genetic.genetic import GeneticOptimization
    from akkudoktoreos.optimization.genetic.geneticparams import GeneticOptimizationParameters
    from akkudoktoreos.optimization.dp.dpoptimizer import DPOptimizer
    from akkudoktoreos.optimization.dp.dpparams import DPOptimizationParameters

    set_ems_start_hour(start_hour)

    # Prepare GA params
    ga_params = await GeneticOptimizationParameters.prepare()
    if ga_params is None:
        print("Failed to prepare GA params")
        return

    # Prepare DP params
    dp_params = await DPOptimizationParameters.prepare()
    if dp_params is None:
        print("Failed to prepare DP params")
        return

    # Run GA
    print("Running GA...")
    ga_opt = GeneticOptimization(verbose=False, fixed_seed=42)
    ga_solution = ga_opt.optimize_ems(parameters=ga_params, start_hour=start_hour, ngen=50)
    print("GA done.")

    # Run DP
    print("Running DP...")
    dp_opt = DPOptimizer()
    dp_solution = dp_opt.optimize(params=dp_params, ha_params=None, start_hour=start_hour, worst_case=False)
    print("DP done.")

    return ga_solution, dp_solution


def analyze_solution(input_data: dict, solution, label: str):
    """Analyze a solution's EV charging strategy."""
    ems = input_data["ems"]
    ev = input_data["eauto"] if "eauto" in input_data else input_data.get("ev")

    horizon = len(ems["strompreis_euro_pro_wh"])
    pv = [ems["pv_prognose_wh"][t] if t < len(ems["pv_prognose_wh"]) else 0 for t in range(horizon)]
    load = [ems["gesamtlast"][t] if t < len(ems["gesamtlast"]) else 0 for t in range(horizon)]
    price = [ems["strompreis_euro_pro_wh"][t] if t < len(ems["strompreis_euro_pro_wh"]) else 0 for t in range(horizon)]

    ac_charge = solution.ac_charge
    discharge = solution.discharge_allowed
    ev_charge = solution.ev_charge_hours_float or []
    grid = solution.result.grid_consumption_wh_per_hour
    bat_soc = solution.result.battery_soc_per_hour
    ev_soc = solution.result.ev_soc_per_hour

    total_grid = sum(grid)
    total_cost = solution.result.total_costs
    balance = solution.result.total_balance

    print(f"\n{'='*100}")
    print(f"{label}")
    print(f"{'='*100}")
    print(f"Grid Import Total: {total_grid:.0f}Wh")
    print(f"Total Cost: {total_cost:.2f}€")
    print(f"Balance: {balance:.2f}€")
    print(f"Battery Final SoC: {bat_soc[-1]:.1f}%")
    print(f"EV Final SoC: {ev_soc[-1]:.1f}%")

    # Count actions
    ac_hours = sum(1 for x in ac_charge if x > 0.5)
    discharge_hours = sum(1 for x in discharge if x == 1)
    ev_charge_hours = sum(1 for x in ev_charge if x > 0.5)

    print(f"\nAC-Charge Hours: {ac_hours}")
    print(f"Discharge Hours: {discharge_hours}")
    print(f"EV-Charge Hours: {ev_charge_hours}")

    # Show all hours
    print(f"\n{'Hour':>4} | {'PV':>6} | {'Load':>6} | {'Price':>7} | {'AC':>4} | {'Dis':>4} | {'EV':>4} | {'Grid':>8} | {'Bat%':>6} | {'EV%':>6}")
    print("-" * 100)

    for t in range(horizon):
        ac = ac_charge[t] if t < len(ac_charge) else 0
        dis = discharge[t] if t < len(discharge) else 0
        ev_c = ev_charge[t] if t < len(ev_charge) else 0
        g = grid[t] if t < len(grid) else 0
        bs = bat_soc[t] if t < len(bat_soc) else 0
        es = ev_soc[t] if t < len(ev_soc) else 0
        p = price[t] if t < len(price) else 0

        print(f"{t:>4} | {pv[t]:>6.0f} | {load[t]:>6.0f} | {p:>7.5f} | {ac:>4.1f} | {dis:>4} | {ev_c:>4.1f} | {g:>8.0f} | {bs:>6.1f} | {es:>6.1f}")


async def main():
    input_path = Path("tests/testdata/optimize_input_2.json")
    with open(input_path) as f:
        input_data = json.load(f)

    ga_solution, dp_solution = await run_optimization(input_data, start_hour=10)

    analyze_solution(input_data, ga_solution, "GA RESULT")
    analyze_solution(input_data, dp_solution, "DP RESULT")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
