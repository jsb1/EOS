#!/usr/bin/env python3
"""Fine-tune price_per_wh_battery for DP optimizer."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from akkudoktoreos.config.config import ConfigEOS
from akkudoktoreos.core.coreabc import get_ems
from akkudoktoreos.core.ems import EnergyManagement
from akkudoktoreos.optimization.dp.dpoptimizer import DPOptimizer
from akkudoktoreos.optimization.dp.dpparams import DPOptimizationParameters
from akkudoktoreos.optimization.genetic.genetic import GeneticOptimization
from akkudoktoreos.optimization.genetic.geneticparams import GeneticOptimizationParameters


def load_input(filepath: Path) -> dict:
    with open(filepath) as f:
        return json.load(f)


def set_ems_start_hour(hour: int) -> None:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    EnergyManagement._start_datetime = now.replace(hour=hour, minute=0, second=0, microsecond=0)


def run_ga(input_data: dict, start_hour: int, generations: int, seed: int) -> dict:
    params = GeneticOptimizationParameters(**input_data)
    optimizer = GeneticOptimization(verbose=False, fixed_seed=seed)
    solution = optimizer.optimize_ems(parameters=params, start_hour=start_hour, ngen=generations)
    result = solution.result
    return {
        "solver": "GA",
        "total_balance": round(result.total_balance, 4),
        "battery_end_soc": round(result.battery_soc_per_hour[-1], 2),
    }


def run_dp(input_data: dict, start_hour: int, price_per_wh_battery: float) -> dict:
    input_data["ems"]["preis_euro_pro_wh_akku"] = price_per_wh_battery
    params = DPOptimizationParameters(**input_data)
    optimizer = DPOptimizer()
    solution = optimizer.optimize(params, ha_params=params.dishwasher, start_hour=start_hour)
    result = solution.result
    return {
        "solver": "DP",
        "price_per_wh_battery": price_per_wh_battery,
        "total_balance": round(result.total_balance, 4),
        "total_costs": round(result.total_costs, 4),
        "total_revenue": round(result.total_revenue, 4),
        "total_losses": round(result.total_losses, 2),
        "battery_end_soc": round(result.battery_soc_per_hour[-1], 2),
        "ev_end_soc": round(result.ev_soc_per_hour[-1], 2) if result.ev_soc_per_hour else None,
    }


def main():
    input_file = Path("tests/testdata/optimize_input_2.json")
    input_data = load_input(input_file)

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

    start_hour = 10
    set_ems_start_hour(start_hour)

    # Reference: GA baseline
    print("Running GA baseline...")
    ga_result = run_ga(input_data, start_hour, 50, 42)
    print(f"GA: Balance={ga_result['total_balance']:.4f}€, Battery={ga_result['battery_end_soc']}%")

    # Fine-tune price_per_wh_battery around the transition point
    prices = [
        0.00000,
        0.00001,
        0.00002,
        0.00003,
        0.00005,
        0.00007,
        0.00008,
        0.00009,
        0.00010,
        0.00011,
        0.00012,
        0.00013,
        0.00014,
        0.00015,
        0.00016,
        0.00017,
        0.00018,
        0.00019,
        0.00020,
    ]

    print("\n" + "=" * 110)
    print(f"{'Price€/Wh':>10} {'Pricect/kWh':>10} {'Balance€':>10} {'Costs€':>10} {'Revenue€':>10} {'LossesWh':>10} {'BatEnd%':>8} {'EVEnd%':>8}")
    print("=" * 110)

    best_balance = float("inf")
    best_price: float = 0.0
    best_result = None

    for price in prices:
        result = run_dp(input_data, start_hour, price)
        ct_kwh = price * 1000

        # Mark if EV is not charged
        ev_warning = " ⚠️" if (result["ev_end_soc"] or 0) < 10 else ""

        print(
            f" {price:.6f}{ct_kwh:8.2f} {result['total_balance']:10.4f} {result['total_costs']:10.4f} "
            f"{result['total_revenue']:10.4f} {result['total_losses']:10.2f} {result['battery_end_soc']:7.2f} "
            f"{result['ev_end_soc']:7.2f}{ev_warning}"
        )

        # Best balance with EV charged
        if (result["ev_end_soc"] or 0) > 50 and result["total_balance"] < best_balance:
            best_balance = result["total_balance"]
            best_price = price
            best_result = result

    print("=" * 110)
    print(f"\nGA Reference: Balance={ga_result['total_balance']:.4f}€, Battery={ga_result['battery_end_soc']}%")

    if best_result is not None:
        print(
            f"\nBest DP (EV charged): price_per_wh_battery={best_price:.6f} ({best_price*1000:.2f}ct/kWh)"
        )
        print(f"  Balance={best_result['total_balance']:.4f}€, BatEnd={best_result['battery_end_soc']}%, EVEnd={best_result['ev_end_soc']}%")
        print(f"  Costs={best_result['total_costs']:.4f}€, Revenue={best_result['total_revenue']:.4f}€, Losses={best_result['total_losses']:.2f}Wh")


if __name__ == "__main__":
    main()
