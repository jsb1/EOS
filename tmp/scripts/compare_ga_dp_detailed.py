#!/usr/bin/env python3
"""Detailed step-by-step comparison of GA vs DP solutions for optimize_input_2.json."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from datetime import datetime, timezone

import numpy as np

from akkudoktoreos.config.config import ConfigEOS
from akkudoktoreos.core.coreabc import get_ems
from akkudoktoreos.core.ems import EnergyManagement
from akkudoktoreos.optimization.dp.dpoptimizer import DPOptimizer
from akkudoktoreos.optimization.dp.dpparams import DPOptimizationParameters
from akkudoktoreos.optimization.genetic.genetic import GeneticOptimization
from akkudoktoreos.optimization.genetic.geneticparams import GeneticOptimizationParameters
from akkudoktoreos.optimization.simulation.session import SimulationSession


def load_input(filepath: Path) -> dict:
    with open(filepath) as f:
        return json.load(f)


def set_ems_start_hour(hour: int) -> None:
    now = datetime.now(timezone.utc)
    EnergyManagement._start_datetime = now.replace(hour=hour, minute=0, second=0, microsecond=0)


def run_ga(input_data: dict, start_hour: int, generations: int, seed: int):
    params = GeneticOptimizationParameters(**input_data)
    optimizer = GeneticOptimization(verbose=False, fixed_seed=seed)
    solution = optimizer.optimize_ems(parameters=params, start_hour=start_hour, ngen=generations)
    return solution


def run_dp(input_data: dict, start_hour: int):
    params = DPOptimizationParameters(**input_data)
    optimizer = DPOptimizer()
    solution = optimizer.optimize(params, ha_params=params.dishwasher, start_hour=start_hour)
    return solution


def main():
    input_file = Path("tests/testdata/optimize_input_2.json")
    input_data = load_input(input_file)

    # Set start hour BEFORE initializing EMS
    start_hour = 15
    set_ems_start_hour(start_hour)

    # Initialize
    ems_eos = get_ems(init=True)
    config = ConfigEOS()
    config.merge_settings_from_dict({
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
    })

    # Run GA
    print("=" * 80)
    print("Running GA optimization...")
    print("=" * 80)
    ga_solution = run_ga(input_data, start_hour, 50, 42)
    ga_result = ga_solution.result

    # Run DP
    print("\n" + "=" * 80)
    print("Running DP optimization...")
    print("=" * 80)
    dp_solution = run_dp(input_data, start_hour)
    dp_result = dp_solution.result

    # Print summary comparison
    print("\n" + "=" * 120)
    print("SUMMARY COMPARISON")
    print("=" * 120)
    print(f"{'Metric':<30} {'GA':>15} {'DP':>15} {'DIFF (GA-DP)':>15}")
    print("-" * 120)
    print(f"{'Balance (€)':<30} {ga_result.total_balance:>15.4f} {dp_result.total_balance:>15.4f} {ga_result.total_balance - dp_result.total_balance:>15.4f}")
    print(f"{'Total Costs (€)':<30} {ga_result.total_costs:>15.4f} {dp_result.total_costs:>15.4f} {ga_result.total_costs - dp_result.total_costs:>15.4f}")
    print(f"{'Total Revenue (€)':<30} {ga_result.total_revenue:>15.4f} {dp_result.total_revenue:>15.4f} {ga_result.total_revenue - dp_result.total_revenue:>15.4f}")
    print(f"{'Total Losses (Wh)':<30} {ga_result.total_losses:>15.2f} {dp_result.total_losses:>15.2f} {ga_result.total_losses - dp_result.total_losses:>15.2f}")
    print(f"{'Battery End SoC (%)':<30} {ga_result.battery_soc_per_hour[-1]:>15.2f} {dp_result.battery_soc_per_hour[-1]:>15.2f} {ga_result.battery_soc_per_hour[-1] - dp_result.battery_soc_per_hour[-1]:>15.2f}")
    print(f"{'EV End SoC (%)':<30} {ga_result.ev_soc_per_hour[-1]:>15.2f} {dp_result.ev_soc_per_hour[-1]:>15.2f} {ga_result.ev_soc_per_hour[-1] - dp_result.ev_soc_per_hour[-1]:>15.2f}")

    # Step-by-step comparison
    print("\n" + "=" * 160)
    print("STEP-BY-STEP COMPARISON")
    print("=" * 160)
    print(f"{'Hour':>4} {'PV(Wh)':>8} {'Load(Wh)':>9} {'GridImp(Wh)':>10} {'GridExp(Wh)':>10} {'BatSoC%':>8} {'EVSoC%':>7} {'Cost(€)':>8} {'Rev(€)':>8} {'Loss(Wh)':>8}")
    print("-" * 160)

    # Compare GA vs DP step by step
    ga_grid_import = np.array(ga_result.grid_consumption_wh_per_hour)
    ga_grid_export = np.array(ga_result.grid_feed_in_wh_per_hour)
    ga_bat_soc = np.array(ga_result.battery_soc_per_hour)
    ga_ev_soc = np.array(ga_result.ev_soc_per_hour)
    ga_costs = np.array(ga_result.costs_per_hour)
    ga_revenue = np.array(ga_result.revenue_per_hour)
    ga_losses = np.array(ga_result.losses_per_hour)

    dp_grid_import = np.array(dp_result.grid_consumption_wh_per_hour)
    dp_grid_export = np.array(dp_result.grid_feed_in_wh_per_hour)
    dp_bat_soc = np.array(dp_result.battery_soc_per_hour)
    dp_ev_soc = np.array(dp_result.ev_soc_per_hour)
    dp_costs = np.array(dp_result.costs_per_hour)
    dp_revenue = np.array(dp_result.revenue_per_hour)
    dp_losses = np.array(dp_result.losses_per_hour)

    pv = np.array(input_data["ems"]["pv_prognose_wh"])[:len(ga_grid_import)]
    load = np.array(input_data["ems"]["gesamtlast"])[:len(ga_grid_import)]

    for h in range(len(ga_grid_import)):
        print(
            f"{h:>4} {pv[h]:>8.1f} {load[h]:>9.1f} "
            f"{ga_grid_import[h]:>10.1f} {ga_grid_export[h]:>10.1f} "
            f"{ga_bat_soc[h]:>8.2f} {ga_ev_soc[h]:>7.2f} "
            f"{ga_costs[h]:>8.4f} {ga_revenue[h]:>8.4f} {ga_losses[h]:>8.2f}"
        )

    # DP step by step
    print("\nDP:")
    print(f"{'Hour':>4} {'PV(Wh)':>8} {'Load(Wh)':>9} {'GridImp(Wh)':>10} {'GridExp(Wh)':>10} {'BatSoC%':>8} {'EVSoC%':>7} {'Cost(€)':>8} {'Rev(€)':>8} {'Loss(Wh)':>8}")
    print("-" * 160)
    for h in range(len(dp_grid_import)):
        print(
            f"{h:>4} {pv[h]:>8.1f} {load[h]:>9.1f} "
            f"{dp_grid_import[h]:>10.1f} {dp_grid_export[h]:>10.1f} "
            f"{dp_bat_soc[h]:>8.2f} {dp_ev_soc[h]:>7.2f} "
            f"{dp_costs[h]:>8.4f} {dp_revenue[h]:>8.4f} {dp_losses[h]:>8.2f}"
        )

    # Difference table
    print("\n" + "=" * 120)
    print("DIFFERENCES (GA - DP)")
    print("=" * 120)
    print(f"{'Hour':>4} {'GridImp Diff':>12} {'GridExp Diff':>12} {'BatSoC Diff':>12} {'EVSoC Diff':>12} {'Cost Diff':>12} {'Loss Diff':>12}")
    print("-" * 120)

    total_cost_diff = 0.0
    for h in range(len(ga_grid_import)):
        diff_cost = ga_costs[h] - dp_costs[h]
        total_cost_diff += diff_cost
        if abs(diff_cost) > 0.001:
            print(
                f"{h:>4} {ga_grid_import[h] - dp_grid_import[h]:>12.1f} {ga_grid_export[h] - dp_grid_export[h]:>12.1f} "
                f"{ga_bat_soc[h] - dp_bat_soc[h]:>12.2f} {ga_ev_soc[h] - dp_ev_soc[h]:>12.2f} "
                f"{diff_cost:>12.4f} {ga_losses[h] - dp_losses[h]:>12.2f}"
            )

    print(f"\nTotal Cost Difference (sum of hourly diffs): {total_cost_diff:.4f}€")
    print(f"Total Balance Difference: {ga_result.total_balance - dp_result.total_balance:.4f}€")

    # Check if costs match balance
    ga_balance_check = ga_result.total_revenue - ga_result.total_costs
    dp_balance_check = dp_result.total_revenue - dp_result.total_costs
    print(f"\nGA Balance Check (Revenue - Costs): {ga_balance_check:.4f}€ (actual: {ga_result.total_balance:.4f}€)")
    print(f"DP Balance Check (Revenue - Costs): {dp_balance_check:.4f}€ (actual: {dp_result.total_balance:.4f}€)")

    # Decision variable comparison
    print("\n" + "=" * 120)
    print("DECISION VARIABLE COMPARISON")
    print("=" * 120)
    print(f"{'Hour':>4} {'GA discharge':>14} {'DP discharge':>14} {'GA ac_charge':>14} {'DP ac_charge':>14} {'GA dc_charge':>14} {'DP dc_charge':>14} {'GA ev_charge':>14} {'DP ev_charge':>14}")
    print("-" * 120)

    ga_discharge = np.array(ga_solution.discharge_allowed)
    dp_discharge = np.array(dp_solution.discharge_allowed)
    ga_ac = np.array(ga_solution.ac_charge)
    dp_ac = np.array(dp_solution.ac_charge)
    ga_dc = np.array(ga_solution.dc_charge)
    dp_dc = np.array(dp_solution.dc_charge)
    ga_ev = np.array(ga_solution.ev_charge_hours_float or [0.0] * len(ga_discharge))
    dp_ev = np.array(dp_solution.ev_charge_hours_float or [0.0] * len(dp_discharge))

    for h in range(len(ga_discharge)):
        print(
            f"{h:>4} {ga_discharge[h]:>14} {dp_discharge[h]:>14} "
            f"{ga_ac[h]:>14.4f} {dp_ac[h]:>14.4f} "
            f"{ga_dc[h]:>14.4f} {dp_dc[h]:>14.4f} "
            f"{ga_ev[h]:>14.4f} {dp_ev[h]:>14.4f}"
        )


if __name__ == "__main__":
    main()
