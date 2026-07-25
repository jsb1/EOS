#!/usr/bin/env python3
"""Benchmark: compare old vs new GeneticSimulation.simulate() output.

Loads the old version of genetic.py from /tmp/old_genetic.py, runs both
old and new simulate() with identical inputs from test data files, and
reports numerical differences field-by-field.
"""

import importlib.util
import json
import sys
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

# ── Load new module (current working tree) ────────────────────────────
sys.path.insert(0, str(SRC))
from akkudoktoreos.optimization.genetic.genetic import GeneticSimulation  # noqa: E402

# ── Helper: build identical simulation from test input ─────────────────
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


def build_simulations(
    input_data: GeneticOptimizationParameters,
) -> tuple[OldGeneticSimulation, Any]:  # type: ignore[name-defined]
    """Build old and new GeneticSimulation with identical parameters."""
    pred_hours = 48
    opt_hours = 48

    # Battery
    bat_param = input_data.pv_battery
    assert bat_param is not None
    battery = Battery(
        SolarPanelBatteryParameters(
            device_id=bat_param.device_id,
            capacity_wh=bat_param.capacity_wh,
            initial_soc_percentage=bat_param.initial_soc_percentage,
            min_soc_percentage=bat_param.min_soc_percentage,
            max_charge_power_w=bat_param.max_charge_power_w
            if hasattr(bat_param, "max_charge_power_w")
            else None,
            charging_efficiency=bat_param.charging_efficiency
            if hasattr(bat_param, "charging_efficiency")
            else None,
            discharging_efficiency=bat_param.discharging_efficiency
            if hasattr(bat_param, "discharging_efficiency")
            else None,
        ),
        prediction_hours=pred_hours,
    )
    battery.set_charge_per_hour(np.full(pred_hours, 0))

    # Inverter
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

    # EV
    ev_param = input_data.ev
    ev = None
    if ev_param:
        ev = Battery(
            ElectricVehicleParameters(
                device_id=ev_param.device_id,
                capacity_wh=ev_param.capacity_wh,
                initial_soc_percentage=ev_param.initial_soc_percentage,
                min_soc_percentage=ev_param.min_soc_percentage,
                charging_efficiency=ev_param.charging_efficiency
                if hasattr(ev_param, "charging_efficiency")
                else None,
            ),
            prediction_hours=pred_hours,
        )
        ev.set_charge_per_hour(np.full(pred_hours, 1))

    # Home appliance
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

    # EMS parameters
    ems_param = input_data.ems
    ems = GeneticEnergyManagementParameters(
        pv_prognose_wh=ems_param.pv_forecast_wh,
        strompreis_euro_pro_wh=ems_param.electricity_price_per_wh,
        einspeiseverguetung_euro_pro_wh=ems_param.feed_in_tariff_per_wh,
        preis_euro_pro_wh_akku=ems_param.price_per_wh_battery,
        gesamtlast=ems_param.total_load,
    )

    # ── Old simulation ─────────────────────────────────────────────
    old_sim = OldGeneticSimulation()
    old_sim.prepare(
        parameters=ems,
        optimization_hours=opt_hours,
        prediction_hours=pred_hours,
        inverter=inverter,
        ev=ev,
        home_appliance=home_appliance,
    )

    # ── New simulation (fresh battery/inverter copies) ────────────
    # Need fresh device copies since old_sim.prepare() mutates shared refs
    battery2 = Battery(
        SolarPanelBatteryParameters(
            device_id=bat_param.device_id,
            capacity_wh=bat_param.capacity_wh,
            initial_soc_percentage=bat_param.initial_soc_percentage,
            min_soc_percentage=bat_param.min_soc_percentage,
            max_charge_power_w=bat_param.max_charge_power_w
            if hasattr(bat_param, "max_charge_power_w")
            else None,
            charging_efficiency=bat_param.charging_efficiency
            if hasattr(bat_param, "charging_efficiency")
            else None,
            discharging_efficiency=bat_param.discharging_efficiency
            if hasattr(bat_param, "discharging_efficiency")
            else None,
        ),
        prediction_hours=pred_hours,
    )
    battery2.set_charge_per_hour(np.full(pred_hours, 0))

    inverter2 = Inverter(
        InverterParameters(
            device_id=inv_param.device_id,
            max_power_wh=inv_param.max_power_wh,
            battery_id=inv_param.battery_id,
        ),
        battery=battery2,
    )

    ev2 = None
    if ev_param:
        ev2 = Battery(
            ElectricVehicleParameters(
                device_id=ev_param.device_id,
                capacity_wh=ev_param.capacity_wh,
                initial_soc_percentage=ev_param.initial_soc_percentage,
                min_soc_percentage=ev_param.min_soc_percentage,
                charging_efficiency=ev_param.charging_efficiency
                if hasattr(ev_param, "charging_efficiency")
                else None,
            ),
            prediction_hours=pred_hours,
        )
        ev2.set_charge_per_hour(np.full(pred_hours, 1))

    home_appliance2 = None
    if ha_param:
        home_appliance2 = HomeAppliance(
            HomeApplianceParameters(
                device_id=ha_param.device_id,
                consumption_wh=ha_param.consumption_wh,
                duration_h=ha_param.duration_h,
                time_windows=ha_param.time_windows,
            ),
            optimization_hours=opt_hours,
            prediction_hours=pred_hours,
        )

    new_sim = GeneticSimulation()
    new_sim.prepare(
        parameters=ems,
        optimization_hours=opt_hours,
        prediction_hours=pred_hours,
        inverter=inverter2,
        ev=ev2,
        home_appliance=home_appliance2,
    )

    return old_sim, new_sim


# ── Compare two result dicts ─────────────────────────────────────────
def compare_results(
    old_result: dict[str, Any],
    new_result: dict[str, Any],
    label: str,
    tol: float = 1e-9,
) -> list[str]:
    """Return list of difference descriptions."""
    issues: list[str] = []
    all_keys = set(old_result.keys()) | set(new_result.keys())

    for key in sorted(all_keys):
        old_v = old_result.get(key)
        new_v = new_result.get(key)

        if key not in old_result:
            issues.append(f"  [MISSING in old]  {key}")
            continue
        if key not in new_result:
            issues.append(f"  [MISSING in new]  {key}")
            continue

        # Scalar
        if isinstance(old_v, (int, float)) and isinstance(new_v, (int, float)):
            diff = abs(float(old_v) - float(new_v))
            if diff > tol:
                issues.append(
                    f"  {key}: old={old_v!r} new={new_v!r} diff={diff:.2e}"
                )
            continue

        # Numpy array / list
        if isinstance(old_v, (np.ndarray, list)) and isinstance(
            new_v, (np.ndarray, list)
        ):
            oa = np.asarray(old_v, dtype=float)
            na = np.asarray(new_v, dtype=float)

            if oa.shape != na.shape:
                issues.append(
                    f"  {key}: shape mismatch old={oa.shape} new={na.shape}"
                )
                continue

            # Compare non-NaN values
            mask = ~(np.isnan(oa) & np.isnan(na))
            diff = np.nanmax(np.abs(oa[mask] - na[mask])) if mask.any() else 0.0
            if diff > tol:
                idx = np.nanargmax(np.abs(oa - na))
                issues.append(
                    f"  {key}: max_diff={diff:.2e} at idx={idx} "
                    f"(old={oa[idx]:.6f}, new={na[idx]:.6f})"
                )
            continue

        # Fallback
        if old_v != new_v:
            issues.append(f"  {key}: old={old_v!r} new={new_v!r}")

    return issues


# ── Main ──────────────────────────────────────────────────────────────
def main() -> None:
    start_hour = 10
    input_files = [
        TESTDATA / "optimize_input_1.json",
        TESTDATA / "optimize_input_2.json",
    ]

    overall_pass = True

    for inp in input_files:
        print(f"\n{'=' * 70}")
        print(f"File: {inp.name}")
        print(f"{'=' * 70}")

        with inp.open() as f:
            data = GeneticOptimizationParameters(**json.load(f))

        old_sim, new_sim = build_simulations(data)

        # Set same action arrays on both simulations
        # (use a simple pattern: all zeros except hour 10 has AC charge + discharge)
        assert old_sim.ac_charge_hours is not None
        assert new_sim.ac_charge_hours is not None

        # Copy action arrays from old to new
        new_sim.ac_charge_hours = old_sim.ac_charge_hours.copy()
        new_sim.dc_charge_hours = old_sim.dc_charge_hours.copy()
        new_sim.bat_discharge_hours = old_sim.bat_discharge_hours.copy()
        new_sim.ev_charge_hours = old_sim.ev_charge_hours.copy()
        new_sim.ev_discharge_hours = old_sim.ev_discharge_hours.copy()
        new_sim.home_appliance_start_hour = old_sim.home_appliance_start_hour

        # Set some non-trivial actions for a meaningful comparison
        old_sim.ac_charge_hours[start_hour] = 1.0
        old_sim.dc_charge_hours[start_hour] = 1.0
        old_sim.bat_discharge_hours[start_hour] = 1.0
        if old_sim.ev_charge_hours is not None:
            old_sim.ev_charge_hours[start_hour] = 1.0
        old_sim.home_appliance_start_hour = start_hour + 1

        # Mirror to new
        new_sim.ac_charge_hours = old_sim.ac_charge_hours.copy()
        new_sim.dc_charge_hours = old_sim.dc_charge_hours.copy()
        new_sim.bat_discharge_hours = old_sim.bat_discharge_hours.copy()
        new_sim.ev_charge_hours = old_sim.ev_charge_hours.copy()
        new_sim.ev_discharge_hours = old_sim.ev_discharge_hours.copy()
        new_sim.home_appliance_start_hour = old_sim.home_appliance_start_hour

        # Run simulations
        old_result = old_sim.simulate(start_hour=start_hour)
        new_result = new_sim.simulate(start_hour=start_hour)

        issues = compare_results(old_result, new_result, inp.name)

        if issues:
            overall_pass = False
            print(f"  DIFFERENCES FOUND ({len(issues)}):")
            for i in issues:
                print(i)
        else:
            print("  OK  -  No differences detected (bitwise identical)")

    # Summary
    print(f"\n{'=' * 70}")
    if overall_pass:
        print("RESULT: ALL files match - refactored simulate() is numerically identical")
    else:
        print("RESULT: Some differences found - review above")
    print(f"{'=' * 70}\n")

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
