#!/usr/bin/env python3
"""Benchmark script for comparing GA vs DP vs HYBRID solvers on all data files.

Compares:
- Performance (time, states explored)
- Optimization quality (total costs, total revenue, total balance)

Usage:
    uv run tmp/scripts/benchmark_all_data.py [--ga-generations N] [--horizon HOURS]
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from akkudoktoreos.config.config import ConfigEOS
from akkudoktoreos.core.coreabc import get_ems
from akkudoktoreos.core.ems import EnergyManagement
from akkudoktoreos.optimization.dp.dpoptimizer import DPOptimizer
from akkudoktoreos.optimization.dp.dpparams import DPOptimizationParameters
from akkudoktoreos.optimization.genetic.genetic import GeneticOptimization
from akkudoktoreos.optimization.genetic.geneticparams import GeneticOptimizationParameters

DIR_TESTDATA = Path(__file__).parent.parent.parent / "tests" / "testdata"


def set_ems_start_hour(hour: int) -> None:
    """Set EMS start_datetime to given hour (needed for GA sync check)."""
    now = datetime.now(timezone.utc)
    EnergyManagement._start_datetime = now.replace(hour=hour, minute=0, second=0, microsecond=0)


def load_input_data(filepath: Path) -> dict:
    """Load optimization input data from JSON file."""
    with filepath.open("r") as f:
        return json.load(f)


def run_ga(
    params: GeneticOptimizationParameters,
    start_hour: int,
    generations: int,
    seed: int = 42,
) -> dict:
    """Run GA solver and return results with timing."""
    optimizer = GeneticOptimization(verbose=False, fixed_seed=seed)
    start = time.perf_counter()
    solution = optimizer.optimize_ems(parameters=params, start_hour=start_hour, ngen=generations)
    elapsed = time.perf_counter() - start

    result = solution.result
    return {
        "solver": "GA",
        "time_seconds": round(elapsed, 3),
        "generations": generations,
        "total_costs": round(result.total_costs, 4),
        "total_revenue": round(result.total_revenue, 4),
        "total_balance": round(result.total_balance, 4),
        "total_losses": round(result.total_losses, 2),
        "ac_charge_count": len(solution.ac_charge),
        "has_ev_charge": solution.ev_charge_hours_float is not None,
        "has_washingstart": solution.washingstart is not None,
        "battery_end_soc": round(result.battery_soc_per_hour[-1], 2) if result.battery_soc_per_hour else None,
    }


def run_dp(
    params: DPOptimizationParameters,
    start_hour: int,
    ha_params=None,
    worst_case: bool = False,
) -> dict:
    """Run DP solver and return results with timing."""
    optimizer = DPOptimizer()
    start = time.perf_counter()
    solution = optimizer.optimize(
        params=params,
        ha_params=ha_params,
        start_hour=start_hour,
        worst_case=worst_case,
    )
    elapsed = time.perf_counter() - start

    result = solution.result
    return {
        "solver": "DP",
        "time_seconds": round(elapsed, 3),
        "total_states_explored": solution.total_states_explored,
        "total_costs": round(result.total_costs, 4),
        "total_revenue": round(result.total_revenue, 4),
        "total_balance": round(result.total_balance, 4),
        "total_losses": round(result.total_losses, 2),
        "ac_charge_count": len(solution.ac_charge),
        "start_soc_index": solution.dp_start_soc_index,
        "end_soc_index": solution.dp_end_soc_index,
        "has_ev_charge": solution.ev_charge_hours_float is not None,
        "has_washingstart": solution.washingstart is not None,
        "battery_end_soc": round(result.battery_soc_per_hour[-1], 2) if result.battery_soc_per_hour else None,
    }


def run_hybrid(
    dp_params: DPOptimizationParameters,
    ga_params: GeneticOptimizationParameters,
    start_hour: int,
    ga_generations: int,
    seed: int = 42,
) -> dict:
    """Run HYBRID solver (DP as GA warmup) and return results with timing."""
    # Phase 1: DP optimization
    dp_optimizer = DPOptimizer()
    dp_start = time.perf_counter()
    dp_solution = dp_optimizer.optimize(
        params=dp_params,
        ha_params=dp_params.dishwasher,
        start_hour=start_hour,
    )
    dp_time = time.perf_counter() - dp_start

    # Phase 2: Convert to GA individual
    ga_individual = dp_optimizer.to_ga_individual(dp_solution)

    # Phase 3: GA refinement with DP solution as starting point
    ga_params.start_solution = ga_individual
    ga_optimizer = GeneticOptimization(verbose=False, fixed_seed=seed)
    ga_start = time.perf_counter()
    ga_solution = ga_optimizer.optimize_ems(
        parameters=ga_params,
        start_hour=start_hour,
        ngen=ga_generations,
    )
    ga_time = time.perf_counter() - ga_start

    result = ga_solution.result
    return {
        "solver": "HYBRID",
        "dp_time_seconds": round(dp_time, 3),
        "ga_time_seconds": round(ga_time, 3),
        "total_time_seconds": round(dp_time + ga_time, 3),
        "dp_states_explored": dp_solution.total_states_explored,
        "ga_generations": ga_generations,
        "total_costs": round(result.total_costs, 4),
        "total_revenue": round(result.total_revenue, 4),
        "total_balance": round(result.total_balance, 4),
        "total_losses": round(result.total_losses, 2),
        "ac_charge_count": len(ga_solution.ac_charge),
        "has_ev_charge": ga_solution.ev_charge_hours_float is not None,
        "has_washingstart": ga_solution.washingstart is not None,
        "battery_end_soc": round(result.battery_soc_per_hour[-1], 2) if result.battery_soc_per_hour else None,
    }


def detect_horizon(raw_data: dict) -> int:
    """Detect horizon from input data."""
    ems = raw_data.get("ems", {})
    return len(ems.get("gesamtlast", []))


def main():
    parser = argparse.ArgumentParser(description="Benchmark all solvers on all data files")
    parser.add_argument(
        "--ga-generations",
        type=int,
        default=50,
        help="GA generations (default: 50)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Optimization horizon in hours (default: auto-detect from data)",
    )
    parser.add_argument("--start-hour", type=int, default=10, help="Start hour (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    # Initialize EMS
    ems_eos = get_ems(init=True)

    # Find optimize input files
    input_files = sorted(DIR_TESTDATA.glob("optimize_input_*.json"))
    if not input_files:
        print("Error: No optimize_input_*.json files found in testdata")
        sys.exit(1)

    # Configure
    config = ConfigEOS()
    config.merge_settings_from_dict(
        {
            "optimization": {
                "genetic": {
                    "individuals": 100,
                    "generations": args.ga_generations,
                    "penalties": {"ev_soc_miss": 10, "ac_charge_break_even": 0},
                },
            },
            "devices": {
                "max_electric_vehicles": 1,
                "electric_vehicles": [{"charge_rates": [0.0, 0.5, 1.0]}],
            },
        }
    )

    all_results = []

    print("=" * 80)
    print("Comprehensive Solver Benchmark on All Data Files")
    print("=" * 80)
    print(f"Data files: {len(input_files)}")
    print(f"GA Generations: {args.ga_generations}")
    print(f"Start Hour: {args.start_hour}")
    print(f"Seed: {args.seed}")
    print("=" * 80)

    for input_path in input_files:
        filename = input_path.name
        print(f"\n{'=' * 80}")
        print(f"Processing: {filename}")
        print(f"{'=' * 80}")

        raw_data = load_input_data(input_path)
        horizon = args.horizon if args.horizon is not None else detect_horizon(raw_data)

        print(f"Horizon: {horizon} hours")

        # Update config horizon
        config.merge_settings_from_dict({"prediction": {"hours": horizon}})

        # Sync EMS start hour for GA
        set_ems_start_hour(args.start_hour)

        file_results = {"file": filename, "horizon": horizon, "solvers": []}

        try:
            # GA
            print("\n  Running GA...")
            ga_params = GeneticOptimizationParameters(**raw_data)
            ga_result = run_ga(ga_params, args.start_hour, args.ga_generations, args.seed)
            file_results["solvers"].append(ga_result)
            print(f"    Time: {ga_result['time_seconds']}s | Balance: {ga_result['total_balance']:.4f}")

        except Exception as e:
            print(f"    GA failed: {e}")

        try:
            # DP
            print("\n  Running DP...")
            dp_params = DPOptimizationParameters(**raw_data)
            dp_result = run_dp(dp_params, args.start_hour, dp_params.dishwasher)
            file_results["solvers"].append(dp_result)
            print(f"    Time: {dp_result['time_seconds']}s | States: {dp_result['total_states_explored']} | Balance: {dp_result['total_balance']:.4f}")

        except Exception as e:
            print(f"    DP failed: {e}")

        try:
            # HYBRID
            print("\n  Running HYBRID...")
            dp_params = DPOptimizationParameters(**raw_data)
            ga_params = GeneticOptimizationParameters(**raw_data)
            hybrid_result = run_hybrid(
                dp_params, ga_params, args.start_hour, args.ga_generations, args.seed
            )
            file_results["solvers"].append(hybrid_result)
            print(f"    Time: {hybrid_result['total_time_seconds']}s (DP: {hybrid_result['dp_time_seconds']}s + GA: {hybrid_result['ga_time_seconds']}s) | Balance: {hybrid_result['total_balance']:.4f}")

        except Exception as e:
            print(f"    HYBRID failed: {e}")

        all_results.append(file_results)

    # Summary
    print("\n\n" + "=" * 80)
    print("SUMMARY - Performance Comparison")
    print("=" * 80)

    for fr in all_results:
        print(f"\n{fr['file']} (horizon={fr['horizon']}h):")
        for sr in fr["solvers"]:
            solver_name = sr["solver"]
            if solver_name == "GA":
                time_str = f"{sr['time_seconds']}s"
            elif solver_name == "DP":
                time_str = f"{sr['time_seconds']}s ({sr['total_states_explored']} states)"
            elif solver_name == "HYBRID":
                time_str = f"{sr['total_time_seconds']}s (DP:{sr['dp_time_seconds']}s+GA:{sr['ga_time_seconds']}s)"
            print(f"  {solver_name:8s} | Time: {time_str:>25s} | Balance: {sr['total_balance']:>10.4f}")

    # Cost comparison
    print("\n\n" + "=" * 80)
    print("SUMMARY - Optimization Quality (Total Balance: higher is better)")
    print("=" * 80)

    for fr in all_results:
        solvers = fr["solvers"]
        if len(solvers) < 2:
            continue

        print(f"\n{fr['file']}:")
        balances = {s["solver"]: s["total_balance"] for s in solvers}

        # Find best
        best_solver = max(balances, key=balances.get)
        best_balance = balances[best_solver]

        for solver, balance in balances.items():
            diff = balance - best_balance
            marker = " <-- BEST" if solver == best_solver else ""
            print(f"  {solver:8s} | Balance: {balance:10.4f} (diff: {diff:+.4f}){marker}")

    # JSON output
    print("\n\n" + "=" * 80)
    print("JSON Output:")
    print("=" * 80)
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
