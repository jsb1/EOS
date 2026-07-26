#!/usr/bin/env python3
"""Benchmark: HYBRID mode with mutated DP variants vs. pure DP."""

import asyncio
import json
import random
import time
from pathlib import Path

import pendulum

from akkudoktoreos.core.coreabc import (
    get_config,
)
from akkudoktoreos.core.ems import (
    EnergyManagement,
)
from akkudoktoreos.optimization.dp.dpparams import (
    DPOptimizationParameters,
)
from akkudoktoreos.optimization.dp.dpoptimizer import (
    DPOptimizer,
)
from akkudoktoreos.optimization.genetic.genetic import (
    GeneticOptimization,
)
from akkudoktoreos.optimization.genetic.geneticparams import (
    GeneticOptimizationParameters,
)
from akkudoktoreos.optimization.simulation.parameters import (
    OptimizationParameters,
)

# Test data
input_path = Path("tests/testdata/optimize_input_2.json")


def set_ems_start_hour(hour: int) -> None:
    """Set EMS start_datetime to midnight of today with given hour."""
    now = pendulum.now("Europe/Berlin")
    start_dt = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    EnergyManagement.set_start_datetime(start_dt)


def load_input_data(filepath: Path) -> dict:
    """Load test input JSON."""
    with open(filepath, "r") as f:
        return json.load(f)


async def run_ga(input_data: dict, start_hour: int, generations: int = 50, seed: int = 42) -> dict:
    """Run GA optimization."""
    set_ems_start_hour(start_hour)

    # Prepare parameters
    params = await GeneticOptimizationParameters.prepare()
    if not params:
        raise RuntimeError("Failed to prepare GA parameters")

    # Override with test data
    params.ems.total_load = input_data["ems"]["gesamtlast"]
    params.ems.pv_forecast_wh = input_data["ems"]["pv_prognose_wh"]
    params.ems.electricity_price_per_wh = input_data["ems"]["strompreis_euro_pro_wh"]
    params.ems.feed_in_tariff_per_wh = input_data["ems"]["einspeisevergütung_euro_pro_wh"]
    params.ems.hours = len(input_data["ems"]["gesamtlast"])

    if "pv_akku" in input_data:
        params.pv_battery.initial_soc_percentage = input_data["pv_akku"].get("init_soc", 50)
    if "eauto" in input_data:
        params.ev.initial_soc_percentage = input_data["eauto"].get("init_soc", 20)

    # Run GA
    ga = GeneticOptimization(params)
    start_time = time.time()

    try:
        result = await ga.optimize_ems()
    except Exception as e:
        print(f"GA failed: {e}")
        return {"balance": float("inf"), "time": 0.0, "error": str(e)}

    elapsed = time.time() - start_time
    balance = result["simulation_result"]["total_balance"]
    return {"balance": balance, "time": elapsed}


async def run_dp(input_data: dict, start_hour: int) -> dict:
    """Run DP optimization."""
    set_ems_start_hour(start_hour)

    # Prepare parameters
    params = await DPOptimizationParameters.prepare()
    if not params:
        raise RuntimeError("Failed to prepare DP parameters")

    # Override with test data
    params.ems.total_load = input_data["ems"]["gesamtlast"]
    params.ems.pv_forecast_wh = input_data["ems"]["pv_prognose_wh"]
    params.ems.electricity_price_per_wh = input_data["ems"]["strompreis_euro_pro_wh"]
    params.ems.feed_in_tariff_per_wh = input_data["ems"]["einspeisevergütung_euro_pro_wh"]
    params.ems.hours = len(input_data["ems"]["gesamtlast"])

    if "pv_akku" in input_data:
        params.pv_battery.initial_soc_percentage = input_data["pv_akku"].get("init_soc", 50)
    if "eauto" in input_data:
        params.ev.initial_soc_percentage = input_data["eauto"].get("init_soc", 20)

    # Run DP
    dp = DPOptimizer(params)
    start_time = time.time()

    try:
        solution = await dp.optimize()
    except Exception as e:
        print(f"DP failed: {e}")
        return {"balance": float("inf"), "time": 0.0, "error": str(e)}

    elapsed = time.time() - start_time
    balance = solution.simulation_result["total_balance"]
    return {"balance": balance, "time": elapsed, "solution": solution}


async def run_hybrid(input_data: dict, start_hour: int, generations: int = 50, seed: int = 42) -> dict:
    """Run HYBRID mode: DP solution as GA warmup with mutated variants."""
    set_ems_start_hour(start_hour)

    # First run DP
    dp_result = await run_dp(input_data, start_hour)
    if "error" in dp_result:
        return dp_result

    dp_solution = dp_result["solution"]
    dp_time = dp_result["time"]

    # Convert DP solution to GA individual
    ga_individual = dp_solution.to_ga_individual(dp_solution)

    # Prepare GA parameters with start_solution
    params = await GeneticOptimizationParameters.prepare()
    if not params:
        raise RuntimeError("Failed to prepare GA parameters")

    # Override with test data
    params.ems.total_load = input_data["ems"]["gesamtlast"]
    params.ems.pv_forecast_wh = input_data["ems"]["pv_prognose_wh"]
    params.ems.electricity_price_per_wh = input_data["ems"]["strompreis_euro_pro_wh"]
    params.ems.feed_in_tariff_per_wh = input_data["ems"]["einspeisevergütung_euro_pro_wh"]
    params.ems.hours = len(input_data["ems"]["gesamtlast"])

    if "pv_akku" in input_data:
        params.pv_battery.initial_soc_percentage = input_data["pv_akku"].get("init_soc", 50)
    if "eauto" in input_data:
        params.ev.initial_soc_percentage = input_data["eauto"].get("init_soc", 20)

    # Set start_solution for HYBRID mode
    params.start_solution = ga_individual

    # Run GA with DP warmup
    ga = GeneticOptimization(params)
    start_time = time.time()

    try:
        result = await ga.optimize_ems()
    except Exception as e:
        print(f"HYBRID failed: {e}")
        return {"balance": float("inf"), "time": 0.0, "error": str(e)}

    elapsed = time.time() - start_time
    balance = result["simulation_result"]["total_balance"]
    return {"balance": balance, "time": elapsed, "dp_time": dp_time}


async def main():
    """Main benchmark function."""
    input_data = load_input_data(input_path)
    start_hour = 10

    print("=" * 70)
    print("Benchmark: HYBRID Mode with Mutated DP Variants")
    print("=" * 70)
    print(f"Input: {input_path.name}")
    print(f"Horizon: {len(input_data['ems']['gesamtlast'])} hours")
    print()

    # Run GA
    print("Running GA...")
    ga_result = await run_ga(input_data, start_hour, generations=50, seed=42)
    print(f"GA Balance: {ga_result['balance']:.4f}€, Time: {ga_result['time']:.2f}s")

    # Run DP
    print("Running DP...")
    dp_result = await run_dp(input_data, start_hour)
    print(f"DP Balance: {dp_result['balance']:.4f}€, Time: {dp_result['time']:.2f}s")

    # Run HYBRID
    print("Running HYBRID (DP + GA with mutated variants)...")
    hybrid_result = await run_hybrid(input_data, start_hour, generations=50, seed=42)
    print(
        f"HYBRID Balance: {hybrid_result['balance']:.4f}€, "
        f"Time: {hybrid_result['time']:.2f}s (DP: {hybrid_result.get('dp_time', 0):.2f}s)"
    )

    # Summary
    print()
    print("=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"{'Solver':<10} {'Balance':>12} {'Time':>10}")
    print("-" * 35)
    print(f"{'GA':<10} {ga_result['balance']:>12.4f}€ {ga_result['time']:>10.2f}s")
    print(f"{'DP':<10} {dp_result['balance']:>12.4f}€ {dp_result['time']:>10.2f}s")
    print(
        f"{'HYBRID':<10} {hybrid_result['balance']:>12.4f}€ "
        f"{hybrid_result['time'] + hybrid_result.get('dp_time', 0):>10.2f}s"
    )

    # Improvement
    dp_improvement = hybrid_result["balance"] - dp_result["balance"]
    ga_improvement = hybrid_result["balance"] - ga_result["balance"]
    print()
    print(f"HYBRID vs DP: {dp_improvement:+.4f}€ ({dp_improvement / abs(dp_result['balance']) * 100:+.1f}%)")
    print(f"HYBRID vs GA: {ga_improvement:+.4f}€ ({ga_improvement / abs(ga_result['balance']) * 100:+.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
