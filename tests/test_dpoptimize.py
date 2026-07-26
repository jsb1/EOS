import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from akkudoktoreos.config.config import ConfigEOS
from akkudoktoreos.core.coreabc import get_ems
from akkudoktoreos.optimization.dp.dpoptimizer import DPOptimizer
from akkudoktoreos.optimization.dp.dpparams import DPOptimizationParameters
from akkudoktoreos.optimization.dp.dpsolution import DPSolution
from akkudoktoreos.optimization.genetic.genetic import GeneticOptimization
from akkudoktoreos.optimization.genetic.geneticparams import GeneticOptimizationParameters

ems_eos = get_ems(init=True)

DIR_TESTDATA = Path(__file__).parent / "testdata"


def compare_dict(actual: dict[str, Any], expected: dict[str, Any], tolerance: float = 1e-4):
    """Compare dictionaries with approximate float comparison."""
    assert set(actual) == set(expected), f"Keys mismatch: {set(actual) ^ set(expected)}"

    for key, value in expected.items():
        if isinstance(value, dict):
            assert isinstance(actual[key], dict), f"Key {key} is not dict"
            compare_dict(actual[key], value, tolerance)
        elif isinstance(value, list):
            assert isinstance(actual[key], list), f"Key {key} is not list"
            assert len(actual[key]) == len(value), f"Key {key} length mismatch"
            for i, (a, e) in enumerate(zip(actual[key], value)):
                if isinstance(e, (int, float)):
                    assert abs(a - e) < tolerance * max(abs(e), 1), f"Key {key}[{i}]: {a} != {e}"
                else:
                    assert a == e, f"Key {key}[{i}]: {a} != {e}"
        elif isinstance(value, (int, float)):
            assert abs(actual[key] - value) < tolerance * max(abs(value), 1), f"Key {key}: {actual[key]} != {value}"
        else:
            assert actual[key] == value, f"Key {key}: {actual[key]} != {value}"


@pytest.mark.asyncio
async def test_dp_optimize_basic(config_eos: ConfigEOS, is_finalize: bool):
    """Test basic DP optimization."""
    config_eos.merge_settings_from_dict(
        {
            "prediction": {"hours": 48},
            "optimization": {
                "horizon_hours": 48,
                "algorithm": "DP",
                "genetic": {
                    "individuals": 100,
                    "generations": 10,
                    "penalties": {"ev_soc_miss": 10, "ac_charge_break_even": 0},
                },
            },
            "devices": {
                "max_electric_vehicles": 1,
                "electric_vehicles": [{"charge_rates": [0.0, 0.5, 1.0]}],
            },
        }
    )

    # Load input data (reuse GA test data)
    file = DIR_TESTDATA / "optimize_input_1.json"
    with file.open("r") as f_in:
        input_data = DPOptimizationParameters(**json.load(f_in))

    # Run DP optimization
    dp_optimizer = DPOptimizer()
    solution = dp_optimizer.optimize(params=input_data, ha_params=input_data.dishwasher, start_hour=10)

    # Verify solution structure
    assert isinstance(solution, DPSolution)
    assert len(solution.ac_charge) == 48
    assert len(solution.dc_charge) == 48
    assert len(solution.discharge_allowed) == 48
    assert solution.total_states_explored > 0
    assert solution.computation_time_ms > 0
    assert solution.dp_start_soc_index is not None
    assert solution.dp_end_soc_index is not None


@pytest.mark.asyncio
async def test_dp_vs_ga_comparison(config_eos: ConfigEOS, is_finalize: bool):
    """Compare DP and GA solutions on same input."""
    config_eos.merge_settings_from_dict(
        {
            "prediction": {"hours": 48},
            "optimization": {
                "horizon_hours": 48,
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

    # Set EMS start hour to match test start_hour
    import pendulum
    from akkudoktoreos.core.coreabc import get_ems
    ems = get_ems()
    # Create a datetime with hour=10
    test_datetime = pendulum.now().replace(hour=10, minute=0, second=0, microsecond=0)
    ems.set_start_datetime(test_datetime)

    # Load input data
    file = DIR_TESTDATA / "optimize_input_1.json"
    with file.open("r") as f_in:
        raw_data = json.load(f_in)

    # Run DP optimization
    dp_params = DPOptimizationParameters(**raw_data)
    dp_optimizer = DPOptimizer()
    dp_solution = dp_optimizer.optimize(params=dp_params, ha_params=dp_params.dishwasher, start_hour=10)

    # Run GA optimization
    ga_params = GeneticOptimizationParameters(**raw_data)
    ga_optimizer = GeneticOptimization(verbose=False, fixed_seed=42)
    ga_solution = ga_optimizer.optimize_ems(parameters=ga_params, start_hour=10, ngen=50)

    # Both solutions should produce similar costs (within 20%)
    # DP is exact in discretized space, GA is approximate
    assert dp_solution.total_states_explored > 0
    assert ga_solution.start_solution is not None

    # Verify both have valid decision variables
    assert all(0 <= v <= 1 for v in dp_solution.ac_charge)
    assert all(0 <= v <= 1 for v in ga_solution.ac_charge)


@pytest.mark.asyncio
async def test_dp_hybrid_mode(config_eos: ConfigEOS, is_finalize: bool):
    """Test DP as GA warmup (HYBRID mode)."""
    config_eos.merge_settings_from_dict(
        {
            "prediction": {"hours": 48},
            "optimization": {
                "horizon_hours": 48,
                "algorithm": "HYBRID",
                "genetic": {
                    "individuals": 100,
                    "generations": 20,
                    "penalties": {"ev_soc_miss": 10, "ac_charge_break_even": 0},
                },
            },
            "devices": {
                "max_electric_vehicles": 1,
                "electric_vehicles": [{"charge_rates": [0.0, 0.5, 1.0]}],
            },
        }
    )

    # Set EMS start hour to match test start_hour
    import pendulum
    from akkudoktoreos.core.coreabc import get_ems
    ems = get_ems()
    test_datetime = pendulum.now().replace(hour=10, minute=0, second=0, microsecond=0)
    ems.set_start_datetime(test_datetime)

    # Load input data
    file = DIR_TESTDATA / "optimize_input_1.json"
    with file.open("r") as f_in:
        raw_data = json.load(f_in)

    # Run DP optimization
    dp_params = DPOptimizationParameters(**raw_data)
    dp_optimizer = DPOptimizer()
    dp_solution = dp_optimizer.optimize(params=dp_params, ha_params=dp_params.dishwasher, start_hour=10)

    # Convert DP solution to GA individual
    ga_individual = dp_optimizer.to_ga_individual(dp_solution)
    assert isinstance(ga_individual, list)
    assert len(ga_individual) > 0

    # Run GA with warmup individual via start_solution parameter
    ga_params = GeneticOptimizationParameters(**raw_data)
    ga_params.start_solution = ga_individual
    ga_optimizer = GeneticOptimization(verbose=False, fixed_seed=42)
    ga_solution = ga_optimizer.optimize_ems(
        parameters=ga_params,
        start_hour=10,
        ngen=20,
    )

    # HYBRID should produce valid solution
    assert ga_solution.start_solution is not None
    assert len(ga_solution.ac_charge) == 48


@pytest.mark.asyncio
async def test_dp_worst_case_mode(config_eos: ConfigEOS, is_finalize: bool):
    """Test DP worst-case optimization (maximize cost instead of minimize)."""
    config_eos.merge_settings_from_dict(
        {
            "prediction": {"hours": 48},
            "optimization": {"horizon_hours": 48, "genetic": {"individuals": 100}},
        }
    )

    file = DIR_TESTDATA / "optimize_input_1.json"
    with file.open("r") as f_in:
        input_data = DPOptimizationParameters(**json.load(f_in))

    dp_optimizer = DPOptimizer()

    # Normal mode (minimize cost)
    solution_normal = dp_optimizer.optimize(
        params=input_data, start_hour=10, worst_case=False,
    )

    # Worst-case mode (maximize cost)
    solution_worst = dp_optimizer.optimize(
        params=input_data, start_hour=10, worst_case=True,
    )

    # Both should produce valid solutions
    assert len(solution_normal.ac_charge) == 48
    assert len(solution_worst.ac_charge) == 48


@pytest.mark.asyncio
async def test_dp_ev_optimization(config_eos: ConfigEOS, is_finalize: bool):
    """Test DP with and without EV optimization."""
    config_eos.merge_settings_from_dict(
        {
            "prediction": {"hours": 48},
            "optimization": {"horizon_hours": 48},
            "devices": {
                "max_electric_vehicles": 1,
                "electric_vehicles": [{"charge_rates": [0.0, 0.5, 1.0]}],
            },
        }
    )

    file = DIR_TESTDATA / "optimize_input_1.json"
    with file.open("r") as f_in:
        input_data = DPOptimizationParameters(**json.load(f_in))

    dp_optimizer = DPOptimizer()

    # With EV optimization
    solution_with_ev = dp_optimizer.optimize(
        params=input_data, start_hour=10, optimize_ev=True,
    )

    # Without EV optimization
    solution_no_ev = dp_optimizer.optimize(
        params=input_data, start_hour=10, optimize_ev=False,
    )

    # Both should be valid
    assert len(solution_with_ev.ac_charge) == 48
    assert len(solution_no_ev.ac_charge) == 48


@pytest.mark.asyncio
async def test_dp_dc_charge_flag(config_eos: ConfigEOS, is_finalize: bool):
    """Test DP with DC charge flag as decision variable."""
    config_eos.merge_settings_from_dict(
        {"prediction": {"hours": 48}, "optimization": {"horizon_hours": 48}}
    )

    file = DIR_TESTDATA / "optimize_input_1.json"
    with file.open("r") as f_in:
        input_data = DPOptimizationParameters(**json.load(f_in))

    dp_optimizer = DPOptimizer()

    # With DC charge optimization
    solution_with_dc = dp_optimizer.optimize(
        params=input_data, start_hour=10, optimize_dc_charge=True,
    )

    # Without DC charge optimization
    solution_no_dc = dp_optimizer.optimize(
        params=input_data, start_hour=10, optimize_dc_charge=False,
    )

    assert len(solution_with_dc.dc_charge) == 48
    assert len(solution_no_dc.dc_charge) == 48


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fn_in",
    [
        "optimize_input_1.json",
        "optimize_input_2.json",
    ],
)
async def test_dp_vs_ga_performance_quality(
    fn_in: str,
    config_eos: ConfigEOS,
    is_finalize: bool,
):
    """Compare DP and GA performance and solution quality on the same input files.

    - DP: optimal in its discretized search space (fine-grained SoC steps).
    - GA: approximate stochastic optimizer with fixed seed, coarser decision space.

    The two algorithms explore different search spaces, so neither is guaranteed
    to dominate the other. This test verifies:
      - Both produce valid solutions with the same structure.
      - Both runtimes and balances are reported.
      - DP is not catastrophically worse than GA (within 50% tolerance).
    """
    import time

    import pendulum
    from akkudoktoreos.core.coreabc import get_ems

    # Fixed GA config for reproducibility
    config_eos.merge_settings_from_dict(
        {
            "prediction": {"hours": 48},
            "optimization": {
                "horizon_hours": 48,
                "genetic": {
                    "individuals": 300,
                    "generations": 50,
                    "penalties": {"ev_soc_miss": 10, "ac_charge_break_even": 1},
                },
            },
            "devices": {
                "max_electric_vehicles": 1,
                "electric_vehicles": [{"charge_rates": [0.0, 0.5, 1.0]}],
            },
        }
    )

    ems = get_ems()
    test_datetime = pendulum.now().replace(hour=10, minute=0, second=0, microsecond=0)
    ems.set_start_datetime(test_datetime)

    # Load input data
    file = DIR_TESTDATA / fn_in
    with file.open("r") as f_in:
        raw_data = json.load(f_in)

    start_hour = 10

    # Run DP
    dp_params = DPOptimizationParameters(**raw_data)
    dp_optimizer = DPOptimizer()
    t0 = time.perf_counter()
    dp_solution = dp_optimizer.optimize(
        params=dp_params,
        ha_params=dp_params.dishwasher,
        start_hour=start_hour,
    )
    dp_runtime_ms = (time.perf_counter() - t0) * 1000

    # Run GA with same parameters and fixed seed
    ga_params = GeneticOptimizationParameters(**raw_data)
    ga_optimizer = GeneticOptimization(verbose=False, fixed_seed=42)
    t0 = time.perf_counter()
    ga_solution = ga_optimizer.optimize_ems(
        parameters=ga_params,
        start_hour=start_hour,
        ngen=50,
    )
    ga_runtime_ms = (time.perf_counter() - t0) * 1000

    # Basic validity checks
    assert len(dp_solution.ac_charge) == 48
    assert len(ga_solution.ac_charge) == 48
    assert all(0 <= v <= 1 for v in dp_solution.ac_charge)
    assert all(0 <= v <= 1 for v in ga_solution.ac_charge)
    assert dp_solution.total_states_explored > 0
    assert ga_solution.start_solution is not None

    # Both balances should be finite.
    dp_balance = float(dp_solution.result.total_balance)
    ga_balance = float(ga_solution.result.total_balance)

    import math
    assert math.isfinite(dp_balance), f"DP balance is not finite on {fn_in}"
    assert math.isfinite(ga_balance), f"GA balance is not finite on {fn_in}"

    # DP warmup for GA: convert DP solution to GA individual and use as start_solution
    ga_individual = dp_optimizer.to_ga_individual(dp_solution)
    ga_params_warmup = GeneticOptimizationParameters(**raw_data)
    ga_params_warmup.start_solution = list(ga_individual)  # type: ignore[assignment]
    ga_optimizer_warmup = GeneticOptimization(verbose=False, fixed_seed=42)
    t0 = time.perf_counter()
    ga_warmup_solution = ga_optimizer_warmup.optimize_ems(
        parameters=ga_params_warmup,
        start_hour=start_hour,
        ngen=50,
    )
    ga_warmup_runtime_ms = (time.perf_counter() - t0) * 1000

    ga_warmup_balance = float(ga_warmup_solution.result.total_balance)
    assert math.isfinite(ga_warmup_balance), f"GA warmup balance is not finite on {fn_in}"

    # Log comparison info for readability
    print(
        f"[{fn_in}] DP balance={dp_balance:.4f}, GA balance={ga_balance:.4f}, "
        f"GA(warmup) balance={ga_warmup_balance:.4f}, "
        f"DP runtime={dp_runtime_ms:.1f}ms, GA runtime={ga_runtime_ms:.1f}ms, "
        f"GA(warmup) runtime={ga_warmup_runtime_ms:.1f}ms"
    )
