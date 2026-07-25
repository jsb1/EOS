#!/usr/bin/env python3
"""Performance benchmark: old vs new GeneticSimulation.simulate().

Runs both versions N times per input file and reports timing statistics.
"""

import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
TESTDATA = ROOT / "tests" / "testdata"
SRC = ROOT / "src"

# ── Load old module ───────────────────────────────────────────────────
old_path = Path("/tmp/old_genetic.py")
spec_old = importlib.util.spec_from_file_location("old_genetic", old_path)
old_genetic = importlib.util.module_from_spec(spec_old)  # type: ignore
spec_old.loader.exec_module(old_genetic)  # type: ignore
OldGeneticSimulation = old_genetic.GeneticSimulation

# ── Load new module ───────────────────────────────────────────────────
sys.path.insert(0, str(SRC))
from akkudoktoreos.optimization.genetic.genetic import GeneticSimulation  # noqa: E402

from akkudoktoreos.devices.genetic.battery import Battery  # noqa: E402
from akkudoktoreos.devices.genetic.homeappliance import HomeAppliance  # noqa: E402
from akkudoktoreos.devices.genetic.inverter import Inverter  # noqa: E402
from akkudoktoreos.optimization.genetic.geneticdevices import (  # noqa: E402
    ElectricVehicleParameters,
    HomeApplianceParameters,
    InverterParameters,
    SolarPanelBatteryParameters,
)
from akkudoktoreos.optimization.genetic.geneticparams import (  # noqa: E402
    GeneticEnergyManagementParameters,
    GeneticOptimizationParameters,
)


N_WARMUP = 5
N_RUNS = 500


def build_fresh_devices(input_data: GeneticOptimizationParameters):
    """Return a tuple of (battery, inverter, ev, home_appliance) built from input."""
    pred_hours = 48
    opt_hours = 48
    bat_param = input_data.pv_battery
    assert bat_param is not None

    battery = Battery(
        SolarPanelBatteryParameters(
            device_id=bat_param.device_id,
            capacity_wh=bat_param.capacity_wh,
            initial_soc_percentage=bat_param.initial_soc_percentage,
            min_soc_percentage=bat_param.min_soc_percentage,
            max_charge_power_w=getattr(bat_param, "max_charge_power_w", None),
            charging_efficiency=getattr(bat_param, "charging_efficiency", None),
            discharging_efficiency=getattr(bat_param, "discharging_efficiency", None),
        ),
        prediction_hours=pred_hours,
    )
    battery.set_charge_per_hour(np.full(pred_hours, 0))

    inv_param = input_data.inverter
    assert inv_param is not None
    inverter = Inverter(
        InverterParameters(
            device_id=inv_param.device_id,
            max_power_wh=inv_param.max_power_wh,
            battery_id=inv_param.battery_id,
        ),
        battery=battery,
    )

    ev_param = input_data.ev
    ev = None
    if ev_param:
        ev = Battery(
            ElectricVehicleParameters(
                device_id=ev_param.device_id,
                capacity_wh=ev_param.capacity_wh,
                initial_soc_percentage=ev_param.initial_soc_percentage,
                min_soc_percentage=ev_param.min_soc_percentage,
                charging_efficiency=getattr(ev_param, "charging_efficiency", None),
            ),
            prediction_hours=pred_hours,
        )
        ev.set_charge_per_hour(np.full(pred_hours, 1))

    ha_param = input_data.dishwasher
    home_appliance = None
    if ha_param:
        home_appliance = HomeAppliance(
            HomeApplianceParameters(
                device_id=ha_param.device_id,
                consumption_wh=ha_param.consumption_wh,
                duration_h=ha_param.duration_h,
                time_windows=ha_param.time_windows,
            ),
            optimization_hours=opt_hours,
            prediction_hours=pred_hours,
        )

    return battery, inverter, ev, home_appliance


def build_sim(
    SimClass: type,
    input_data: GeneticOptimizationParameters,
    battery,
    inverter,
    ev,
    home_appliance,
):
    """Build and prepare a simulation instance."""
    pred_hours = 48
    opt_hours = 48
    ems_param = input_data.ems
    ems = GeneticEnergyManagementParameters(
        pv_prognose_wh=ems_param.pv_forecast_wh,
        strompreis_euro_pro_wh=ems_param.electricity_price_per_wh,
        einspeiseverguetung_euro_pro_wh=ems_param.feed_in_tariff_per_wh,
        preis_euro_pro_wh_akku=ems_param.price_per_wh_battery,
        gesamtlast=ems_param.total_load,
    )

    sim = SimClass()
    sim.prepare(
        parameters=ems,
        optimization_hours=opt_hours,
        prediction_hours=pred_hours,
        inverter=inverter,
        ev=ev,
        home_appliance=home_appliance,
    )
    return sim


def setup_action_arrays(sim, start_hour: int) -> None:
    """Set non-trivial action arrays for a realistic benchmark."""
    assert sim.ac_charge_hours is not None
    sim.ac_charge_hours[start_hour] = 1.0
    sim.dc_charge_hours[start_hour] = 1.0
    sim.bat_discharge_hours[start_hour] = 1.0
    if sim.ev_charge_hours is not None:
        sim.ev_charge_hours[start_hour] = 1.0
    sim.home_appliance_start_hour = start_hour + 1


def benchmark_simulate(sim_class: type, input_data: GeneticOptimizationParameters, n: int) -> list[float]:
    """Run simulate() n times, return list of elapsed seconds."""
    start_hour = 10
    times: list[float] = []

    for _ in range(n):
        # Fresh devices each iteration (simulate what the GA does)
        battery, inverter, ev, home_appliance = build_fresh_devices(input_data)
        sim = build_sim(sim_class, input_data, battery, inverter, ev, home_appliance)
        setup_action_arrays(sim, start_hour)

        t0 = time.perf_counter()
        sim.simulate(start_hour=start_hour)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return times


def main() -> None:
    input_files = [
        TESTDATA / "optimize_input_1.json",
        TESTDATA / "optimize_input_2.json",
    ]

    for inp in input_files:
        print(f"\n{'=' * 70}")
        print(f"File: {inp.name}")
        print(f"{'=' * 70}")

        with inp.open() as f:
            data = GeneticOptimizationParameters(**json.load(f))

        # Warmup (JIT / cache effects)
        for _ in range(N_WARMUP):
            build_fresh_devices(data)

        print(f"\nRuns: {N_RUNS}  (warmup: {N_WARMUP})")

        old_times = benchmark_simulate(OldGeneticSimulation, data, N_RUNS)
        new_times = benchmark_simulate(GeneticSimulation, data, N_RUNS)

        def stats_label(label: str, times: list[float]) -> None:
            mean_ms = statistics.mean(times) * 1000
            median_ms = statistics.median(times) * 1000
            stdev_ms = statistics.stdev(times) * 1000 if len(times) > 1 else 0.0
            min_ms = min(times) * 1000
            max_ms = max(times) * 1000
            print(f"  {label:10s}: mean={mean_ms:8.2f}ms  median={median_ms:8.2f}ms  "
                  f"stdev={stdev_ms:6.2f}ms  min={min_ms:7.2f}ms  max={max_ms:7.2f}ms")

        stats_label("OLD", old_times)
        stats_label("NEW", new_times)

        speedup = statistics.mean(old_times) / statistics.mean(new_times)
        pct = (speedup - 1) * 100
        if speedup > 1:
            print(f"  Speedup:  {speedup:.3f}x  (NEW is {pct:+.1f}%)")
        elif speedup < 1:
            print(f"  Slowdown: {1/speedup:.3f}x  (NEW is {pct:+.1f}%)")
        else:
            print(f"  Identical ({pct:+.1f}%)")

    print(f"\n{'=' * 70}")
    print("Benchmark complete")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
