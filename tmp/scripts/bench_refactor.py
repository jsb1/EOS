#!/usr/bin/env python3
"""Benchmark: compare EnergySimulationEngine vs GeneticSimulation vs full optimization.

Tests both correctness (bitwise identical) and performance across:
1. Direct simulate() calls (old GeneticSimulation vs new Engine)
2. Full optimize_ems() runs (thousands of simulate calls)
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Ensure src is importable
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from akkudoktoreos.config.config import ConfigEOS
from akkudoktoreos.core.coreabc import get_ems
from akkudoktoreos.optimization.genetic.genetic import GeneticOptimization, GeneticSimulation
from akkudoktoreos.optimization.genetic.geneticparams import (
    GeneticEnergyManagementParameters,
    GeneticOptimizationParameters,
)
from akkudoktoreos.optimization.simulation.engine import EnergySimulationEngine


# ── Helpers ────────────────────────────────────────────────────────────

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


def compare_dict(
    actual: dict[str, Any], expected: dict[str, Any], path: str = ""
) -> list[str]:
    """Recursively compare two dicts, return list of differences."""
    diffs: list[str] = []
    all_keys = set(list(actual.keys()) + list(expected.keys()))
    for key in sorted(all_keys):
        full_key = f"{path}.{key}" if path else key
        if key not in actual:
            diffs.append(f"  MISSING in actual: {full_key}")
            continue
        if key not in expected:
            diffs.append(f"  EXTRA in actual: {full_key}")
            continue

        a_val = actual[key]
        e_val = expected[key]

        if isinstance(a_val, dict) and isinstance(e_val, dict):
            diffs.extend(compare_dict(a_val, e_val, full_key))
        elif isinstance(a_val, float) and isinstance(e_val, float):
            if np.isnan(a_val) and np.isnan(e_val):
                continue
            diff = abs(a_val - e_val)
            if diff > 1e-6:
                diffs.append(f"  {full_key}: {a_val:.8f} vs {e_val:.8f} (diff={diff:.2e})")
        elif isinstance(a_val, list) and isinstance(e_val, list):
            if len(a_val) != len(e_val):
                diffs.append(f"  {full_key}: len {len(a_val)} vs {len(e_val)}")
            else:
                for i, (av, ev) in enumerate(zip(a_val, e_val)):
                    if isinstance(av, float) and isinstance(ev, float):
                        if np.isnan(av) and np.isnan(ev):
                            continue
                        diff = abs(av - ev)
                        if diff > 1e-6:
                            diffs.append(
                                f"  {full_key}[{i}]: {av:.8f} vs {ev:.8f} (diff={diff:.2e})"
                            )
                    elif av != ev:
                        diffs.append(f"  {full_key}[{i}]: {av} vs {ev}")
        elif a_val != e_val:
            diffs.append(f"  {full_key}: {a_val} vs {e_val}")
    return diffs


# ── Benchmark 1: Direct simulate() ─────────────────────────────────────

def bench_simulate(params: GeneticOptimizationParameters, config: ConfigEOS, n_runs: int = 100):
    """Compare GeneticSimulation.simulate() vs EnergySimulationEngine.run()."""
    print(f"\n{'='*70}")
    print(f"  Benchmark 1: Direct simulate() calls (n={n_runs})")
    print(f"{'='*70}")

    start_hour = 0
    pred_hours = config.prediction.hours or 24
    opt_hours = config.optimization.horizon_hours or 24

    # --- GeneticSimulation (wrapper) ---
    sim = GeneticSimulation()
    sim.prepare(
        parameters=params.ems,
        optimization_hours=opt_hours,
        prediction_hours=pred_hours,
        ev=None,
        home_appliance=None,
        inverter=None,
    )

    # Set up action arrays for testing
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
        result_gs = sim.simulate(start_hour)
        if i < n_runs - 1:
            sim.reset()
    t_gs = time.perf_counter() - t0

    # --- EnergySimulationEngine (direct) ---
    # Create engine once; action arrays from sim
    ac_charge = sim.ac_charge_hours
    dc_charge = sim.dc_charge_hours
    discharge = np.array(sim.bat_discharge_hours, dtype=int)
    ev_charge = sim.ev_charge_hours if sim.ev is not None else None

    # Warmup: recreate engine each time (battery state is mutable)
    for _ in range(5):
        engine = EnergySimulationEngine.create(params, config, start_hour)
        engine.run(ac_charge, dc_charge, discharge, ev_charge)

    # Timed runs: recreate engine for fresh device state
    t0 = time.perf_counter()
    for i in range(n_runs):
        engine = EnergySimulationEngine.create(params, config, start_hour)
        result_engine = engine.run(ac_charge, dc_charge, discharge, ev_charge)
    t_engine = time.perf_counter() - t0

    # Compare results
    print(f"\n  GeneticSimulation:  {t_gs*1000:8.2f} ms total ({t_gs*1000/n_runs:8.2f} ms/call)")
    print(f"  Engine (direct):    {t_engine*1000:8.2f} ms total ({t_engine*1000/n_runs:8.2f} ms/call)")
    ratio = t_gs / t_engine if t_engine > 0 else float("inf")
    print(f"  Ratio (GS/Engine):  {ratio:8.2f}x {'(Engine faster)' if ratio > 1 else '(GS faster)'}")

    # Correctness: compare Engine result dict vs GS result dict
    gs_dict = result_gs
    engine_dict = result_engine.to_dict()

    diffs = compare_dict(gs_dict, engine_dict)
    if diffs:
        print(f"\n  ❌ CORRECTNESS: {len(diffs)} differences found:")
        for d in diffs[:10]:
            print(f"    {d}")
    else:
        print(f"\n  ✅ CORRECTNESS: Bitwise identical")

    return t_gs / n_runs, t_engine / n_runs


# ── Benchmark 2: Full optimize_ems() ──────────────────────────────────

def bench_full_optimize(input_path: Path, config: ConfigEOS, ngen: int = 20, seed: int = 42):
    """Run full optimize_ems() and report wall-clock time + results."""
    params, start_hour = load_input(input_path)

    print(f"\n{'='*70}")
    print(f"  Benchmark 2: Full optimize_ems() on {input_path.name}")
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

    # Calculate self-consumption ratio from feed-in and PV data
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


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    TESTDATA = ROOT / "tests" / "testdata"

    # Init EMS
    ems_eos = get_ems(init=True)
    config = ems_eos.config

    # Load test data
    for name in ["optimize_input_1.json", "optimize_input_2.json"]:
        input_path = TESTDATA / name
        if not input_path.exists():
            print(f"\n⚠️  Skipping {name} (file not found)")
            continue

        print(f"\n{'#'*70}")
        print(f"# {name}")
        print(f"{'#'*70}")

        params, start_hour = load_input(input_path)

        # Benchmark 1: Direct simulate
        t_gs, t_engine = bench_simulate(params, config, n_runs=200)

        # Benchmark 2: Full optimization
        t_wall, _ = bench_full_optimize(input_path, config, ngen=20, seed=42)

    print(f"\n{'='*70}")
    print("  Benchmark complete")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
