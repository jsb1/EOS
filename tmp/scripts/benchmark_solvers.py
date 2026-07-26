#!/usr/bin/env python3
"""Benchmark script for comparing GA vs DP vs HYBRID solvers.

Usage:
    uv run tmp/scripts/benchmark_solvers.py [--input FILE] [--horizon HOURS] [--ga-generations N]
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


def benchmark_ga(
    params: GeneticOptimizationParameters,
    start_hour: int,
    generations: int,
    seed: int = 42,
) -> dict:
    """Benchmark GA solver."""
    optimizer = GeneticOptimization(verbose=False, fixed_seed=seed)
    start = time.perf_counter()
    solution = optimizer.optimize_ems(parameters=params, start_hour=start_hour, ngen=generations)
    elapsed = time.perf_counter() - start

    return {
        "solver": "GA",
        "time_seconds": round(elapsed, 3),
        "generations": generations,
        "ac_charge_count": len(solution.ac_charge),
        "has_ev_charge": solution.ev_charge_hours_float is not None,
        "has_washingstart": solution.washingstart is not None,
    }


def benchmark_dp(
    params: DPOptimizationParameters,
    start_hour: int,
    ha_params=None,
    worst_case: bool = False,
) -> dict:
    """Benchmark DP solver."""
    optimizer = DPOptimizer()
    start = time.perf_counter()
    solution = optimizer.optimize(
        params=params,
        ha_params=ha_params,
        start_hour=start_hour,
        worst_case=worst_case,
    )
    elapsed = time.perf_counter() - start

    return {
        "solver": "DP",
        "time_seconds": round(elapsed, 3),
        "computation_time_ms": round(solution.computation_time_ms, 3),
        "total_states_explored": solution.total_states_explored,
        "ac_charge_count": len(solution.ac_charge),
        "start_soc_index": solution.dp_start_soc_index,
        "end_soc_index": solution.dp_end_soc_index,
        "has_ev_charge": solution.ev_charge_hours_float is not None,
        "has_washingstart": solution.washingstart is not None,
    }


def benchmark_hybrid(
    dp_params: DPOptimizationParameters,
    ga_params: GeneticOptimizationParameters,
    start_hour: int,
    ga_generations: int,
    seed: int = 42,
) -> dict:
    """Benchmark HYBRID solver (DP as GA warmup)."""
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

    # Phase 3: GA refinement
    ga_optimizer = GeneticOptimization(verbose=False, fixed_seed=seed)
    ga_start = time.perf_counter()
    ga_solution = ga_optimizer.optimize_ems(
        parameters=ga_params,
        start_hour=start_hour,
        ngen=ga_generations,
        warmup_individual=ga_individual,
    )
    ga_time = time.perf_counter() - ga_start

    return {
        "solver": "HYBRID",
        "dp_time_seconds": round(dp_time, 3),
        "ga_time_seconds": round(ga_time, 3),
        "total_time_seconds": round(dp_time + ga_time, 3),
        "dp_states_explored": dp_solution.total_states_explored,
        "ga_generations": ga_generations,
        "ac_charge_count": len(ga_solution.ac_charge),
        "has_ev_charge": ga_solution.ev_charge_hours_float is not None,
        "has_washingstart": ga_solution.washingstart is not None,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark GA vs DP vs HYBRID solvers")
    parser.add_argument(
        "--input",
        type=str,
        default="optimize_input_1.json",
        help="Input JSON file (default: optimize_input_1.json)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=48,
        help="Optimization horizon in hours (default: 48)",
    )
    parser.add_argument(
        "--ga-generations",
        type=int,
        default=50,
        help="GA generations (default: 50)",
    )
    parser.add_argument("--start-hour", type=int, default=10, help="Start hour (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--ga-only", action="store_true", help="Run only GA benchmark"
    )
    parser.add_argument(
        "--dp-only", action="store_true", help="Run only DP benchmark"
    )
    parser.add_argument(
        "--hybrid-only", action="store_true", help="Run only HYBRID benchmark"
    )
    args = parser.parse_args()

    # Initialize EMS
    ems_eos = get_ems(init=True)

    # Load input data
    input_path = DIR_TESTDATA / args.input
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    raw_data = load_input_data(input_path)

    # Configure
    config = ConfigEOS()
    config.merge_settings_from_dict(
        {
            "prediction": {"hours": args.horizon},
            "optimization": {
                "horizon_hours": args.horizon,
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

    print("=" * 70)
    print("Solver Benchmark")
    print("=" * 70)
    print(f"Input: {args.input}")
    print(f"Horizon: {args.horizon} hours")
    print(f"GA Generations: {args.ga_generations}")
    print(f"Start Hour: {args.start_hour}")
    print(f"Seed: {args.seed}")
    print("=" * 70)

    # Sync EMS start hour for GA (GA checks EMS time)
    set_ems_start_hour(args.start_hour)

    run_all = not (args.ga_only or args.dp_only or args.hybrid_only)

    results = []

    # GA Benchmark
    if run_all or args.ga_only:
        print("\n[1/3] Benchmarking GA solver...")
        ga_params = GeneticOptimizationParameters(**raw_data)
        ga_result = benchmark_ga(ga_params, args.start_hour, args.ga_generations, args.seed)
        results.append(ga_result)
        print(f"  Time: {ga_result['time_seconds']}s")
        print(f"  AC charge entries: {ga_result['ac_charge_count']}")

    # DP Benchmark
    if run_all or args.dp_only:
        print("\n[2/3] Benchmarking DP solver...")
        dp_params = DPOptimizationParameters(**raw_data)
        dp_result = benchmark_dp(dp_params, args.start_hour, dp_params.dishwasher)
        results.append(dp_result)
        print(f"  Time: {dp_result['time_seconds']}s")
        print(f"  States explored: {dp_result['total_states_explored']}")
        print(f"  AC charge entries: {dp_result['ac_charge_count']}")

    # HYBRID Benchmark
    if run_all or args.hybrid_only:
        print("\n[3/3] Benchmarking HYBRID solver (DP+GA)...")
        dp_params = DPOptimizationParameters(**raw_data)
        ga_params = GeneticOptimizationParameters(**raw_data)
        hybrid_result = benchmark_hybrid(
            dp_params, ga_params, args.start_hour, args.ga_generations, args.seed
        )
        results.append(hybrid_result)
        print(f"  DP Time: {hybrid_result['dp_time_seconds']}s")
        print(f"  GA Time: {hybrid_result['ga_time_seconds']}s")
        print(f"  Total Time: {hybrid_result['total_time_seconds']}s")
        print(f"  DP States explored: {hybrid_result['dp_states_explored']}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for r in results:
        print(f"\n{r['solver']}:")
        if r["solver"] == "GA":
            print(f"  Time: {r['time_seconds']}s")
        elif r["solver"] == "DP":
            print(f"  Time: {r['time_seconds']}s")
            print(f"  States explored: {r['total_states_explored']}")
        elif r["solver"] == "HYBRID":
            print(f"  DP Time: {r['dp_time_seconds']}s")
            print(f"  GA Time: {r['ga_time_seconds']}s")
            print(f"  Total: {r['total_time_seconds']}s")

    # JSON output
    print("\n" + "=" * 70)
    print("JSON Output:")
    print("=" * 70)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
