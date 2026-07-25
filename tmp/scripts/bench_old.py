#!/usr/bin/env python3
"""Benchmark: Old GeneticSimulation.simulate() (pre-refactoring).

Same test as bench_refactor.py but against the original genetic.py
without EnergySimulationEngine wrapper.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from akkudoktoreos.config.config import ConfigEOS
from akkudoktoreos.core.coreabc import get_ems
from akkudoktoreos.optimization.genetic.genetic import GeneticOptimization, GeneticSimulation
from akkudoktoreos.optimization.genetic.geneticparams import (
    GeneticEnergyManagementParameters,
    GeneticOptimizationParameters,
)


def load_input(path: Path) -> tuple[GeneticOptimizationParameters, int]:
    """Load test input JSON and return parameters + start_hour."""
    data = json.loads(path.read_text())
    ems_data = data.get("ems", {})
    pv_akku_data = data.get("pv_akku", {})
    ev_data = data.get("eauto", None)
    inv_data = data.get("inverter", {})
    ha_data = data.get("home_appliance", None)
    temp_data = data.get("temperature_forecast", None)
    start_solution = data.get("start_solution", None)

    ems_params = GeneticEnergyManagementParameters(
        electricity_price_per_wh=np.array(ems_data.get("strompreis_euro_pro_wh", [])).tolist(),
        feed_in_tariff_per_wh=np.array(ems_data.get("einspeiseverguetung_euro_pro_wh", [])).tolist(),
        total_load=np.array(ems_data.get("gesamtlast", [])).tolist(),
        pv_forecast_wh=np.array(ems_data.get("pv_prognose_wh", [])).tolist(),
        price_per_wh_battery=ems_data.get("preis_euro_pro_wh_akku", 0.0),
    )

    pv_battery_params = None
    if pv_akku_data:
        from akkudoktoreos.optimization.genetic.geneticdevices import (
            SolarPanelBatteryParameters,
        )

        pv_battery_params = SolarPanelBatteryParameters(**pv_akku_data)

    ev_params = None
    if ev_data:
        from akkudoktoreos.optimization.genetic.geneticdevices import (
            ElectricVehicleParameters,
        )

        ev_params = ElectricVehicleParameters(**ev_data)

    inv_params = None
    if inv_data:
        from akkudoktoreos.optimization.genetic.geneticdevices import InverterParameters

        inv_params = InverterParameters(**inv_data)

    ha_params = None
    if ha_data:
        from akkudoktoreos.optimization.genetic.geneticdevices import HomeApplianceParameters

        ha_params = HomeApplianceParameters(**ha_data)

    params = GeneticOptimizationParameters(
        ems=ems_params,
        pv_battery=pv_battery_params,
        ev=ev_params,
        inverter=inv_params,
        dishwasher=ha_params,
        start_solution=start_solution,
        temperature_forecast=temp_data,
    )
    return params, 0


def bench_simulate_old(params: GeneticOptimizationParameters, config: ConfigEOS, n_runs: int = 100):
    """Benchmark GeneticSimulation.simulate() (pre-refactoring)."""
    print(f"\n{'='*70}")
    print(f"  Benchmark: Direct simulate() calls (n={n_runs})")
    print(f"  (pre-refactoring, no Engine wrapper)")
    print(f"{'='*70}")

    start_hour = 0
    pred_hours = config.prediction.hours or 24
    opt_hours = config.optimization.horizon_hours or 24

    sim = GeneticSimulation()
    sim.prepare(
        parameters=params.ems,
        optimization_hours=opt_hours,
        prediction_hours=pred_hours,
        ev=None,
        home_appliance=None,
        inverter=None,
    )

    n_hours = len(params.ems.total_load) if params.ems.total_load else pred_hours
    sim.ac_charge_hours = np.array([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.5] * (n_hours // 8 + 1))[:n_hours]
    sim.dc_charge_hours = np.ones(n_hours)
    sim.bat_discharge_hours = np.array([0, 1, 0, 0, 1, 0, 0, 0] * (n_hours // 8 + 1))[:n_hours]
    if params.ev:
        sim.ev_charge_hours = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0] * (n_hours // 8 + 1))[:n_hours]
    else:
        sim.ev_charge_hours = np.zeros(n_hours)

    # Warmup
    for _ in range(5):
        sim.simulate(start_hour)
        sim.reset()

    # Timed runs
    t0 = time.perf_counter()
    for i in range(n_runs):
        result = sim.simulate(start_hour)
        if i < n_runs - 1:
            sim.reset()
    t_total = time.perf_counter() - t0

    print(f"\n  GeneticSimulation:  {t_total*1000:8.2f} ms total ({t_total*1000/n_runs:8.2f} ms/call)")

    # Show some result values
    print(f"\n  Sample results:")
    print(f"    Gesamtbilanz_Euro: {result.get('Gesamtbilanz_Euro', 'N/A')}")
    print(f"    Gesamt_Verluste:   {result.get('Gesamt_Verluste', 'N/A')}")
    print(f"    EAuto_SoC[0:5]:    {result.get('EAuto_SoC_pro_Stunde', [])[:5]}")

    return t_total / n_runs, result


def bench_full_optimize_old(input_path: Path, config: ConfigEOS, ngen: int = 20, seed: int = 42):
    """Run full optimize_ems() and report wall-clock time + results."""
    params, start_hour = load_input(input_path)

    print(f"\n{'='*70}")
    print(f"  Full optimize_ems() on {input_path.name}")
    print(f"  (ngen={ngen}, seed={seed})")
    print(f"{'='*70}")

    opt = GeneticOptimization(fixed_seed=seed)
    t0 = time.perf_counter()
    solution = opt.optimize_ems(parameters=params, ngen=ngen)
    t_wall = time.perf_counter() - t0

    result = solution.result
    print(f"\n  Wall-clock time:    {t_wall*1000:8.0f} ms ({t_wall:.2f} s)")
    print(f"  Gesamtbilanz_Euro:  {result.total_balance:.6f}")
    print(f"  Gesamtkosten_Euro:  {result.total_costs:.6f}")
    print(f"  Gesamteinnahmen_Euro: {result.total_revenue:.6f}")
    print(f"  Gesamt_Verluste:    {result.total_losses:.2f}")

    total_pv = sum(params.ems.pv_forecast_wh) if params.ems.pv_forecast_wh else 0
    total_feedin = sum(result.grid_feed_in_wh_per_hour) if result.grid_feed_in_wh_per_hour else 0
    if total_pv > 0:
        self_consumption_ratio = 1.0 - (total_feedin / total_pv)
    else:
        self_consumption_ratio = 0.0
    print(f"  self_consumption:   {self_consumption_ratio:.2%}")
    print(f"  ac_charge[:8]:      {list(solution.ac_charge[:8])}")
    print(f"  discharge_allowed[:8]: {list(solution.discharge_allowed[:8])}")

    return t_wall, result


def main() -> None:
    TESTDATA = ROOT / "tests" / "testdata"

    ems_eos = get_ems(init=True)
    config = ems_eos.config

    for name in ["optimize_input_1.json", "optimize_input_2.json"]:
        input_path = TESTDATA / name
        if not input_path.exists():
            print(f"\n⚠️  Skipping {name} (file not found)")
            continue

        print(f"\n{'#'*70}")
        print(f"# {name} (OLD / pre-refactoring)")
        print(f"{'#'*70}")

        params, start_hour = load_input(input_path)

        # Benchmark 1: Direct simulate
        t_per_call, _ = bench_simulate_old(params, config, n_runs=200)

        # Benchmark 2: Full optimization
        t_wall, _ = bench_full_optimize_old(input_path, config, ngen=20, seed=42)

    print(f"\n{'='*70}")
    print("  Benchmark complete (old version)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
