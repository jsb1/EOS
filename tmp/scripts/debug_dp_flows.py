#!/usr/bin/env python3
"""Debug script: Compare DP cost calculation vs Simulation step-by-step."""

import json
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from akkudoktoreos.config.config import ConfigEOS
from akkudoktoreos.optimization.simulation.physics import (
    compute_battery_next_soc,
    compute_battery_next_soc_with_flows,
)
from akkudoktoreos.optimization.simulation.devices import Battery, Inverter
from akkudoktoreos.optimization.simulation.session import SimulationSession
from akkudoktoreos.optimization.simulation.parameters import OptimizationParameters


def load_input(filepath: Path) -> dict:
    with open(filepath) as f:
        return json.load(f)


def compare_step(input_data: dict, hour: int):
    """Compare DP flow calculation vs Simulation for one hour."""
    print(f"\n{'='*60}")
    print(f"Hour {hour}")
    print(f"{'='*60}")

    ems = input_data["ems"]
    pv_wh = ems["pv_prognose_wh"][hour]
    load_wh = ems["gesamtlast"][hour]
    price = ems["strompreis_euro_pro_wh"][hour]
    revenue_val = ems.get("einspeiseverguetung_euro_pro_wh", 0.0)
    revenue = revenue_val if isinstance(revenue_val, float) else revenue_val[hour]

    print(f"  PV: {pv_wh:.0f} Wh")
    print(f"  Load: {load_wh:.0f} Wh")
    print(f"  Price: {price:.6f} €/Wh")
    print(f"  Revenue: {revenue:.6f} €/Wh")

    # Battery params
    bat = input_data["pv_akku"]
    bat_params = {
        "capacity_wh": bat.get("capacity_wh", 26400),
        "charging_efficiency": bat.get("charging_efficiency", 0.95),
        "discharging_efficiency": bat.get("discharging_efficiency", 0.95),
        "min_soc_percentage": bat.get("min_soc_percentage", 0),
        "max_soc_percentage": bat.get("max_soc_percentage", 100),
        "initial_soc_percentage": bat.get("initial_soc_percentage", 80),
        "device_id": bat.get("device_id", "battery_0"),
    }

    # Inverter params - use defaults since optimize_input_2.json lacks efficiency fields
    inv = input_data["inverter"]
    ac_to_dc_eff = inv.get("ac_to_dc_efficiency", 0.95)
    dc_to_ac_eff = inv.get("dc_to_ac_efficiency", 0.95)
    max_ac_power = inv.get("max_ac_charge_power_w", bat_params["capacity_wh"])

    # Test scenario: discharge allowed, no AC charge, EV charging at 50%
    initial_soc_wh = (bat_params["initial_soc_percentage"] / 100.0) * bat_params["capacity_wh"]
    min_soc_wh = (bat_params["min_soc_percentage"] / 100.0) * bat_params["capacity_wh"]
    max_soc_wh = (bat_params["max_soc_percentage"] / 100.0) * bat_params["capacity_wh"]
    ev_draw_wh = 1500.0  # EV charging

    print(f"\n  Initial SoC: {initial_soc_wh:.0f} Wh")
    print(f"  EV draw: {ev_draw_wh:.0f} Wh")
    print(f"  Action: discharge_allowed=True, ac_charge=0")

    # --- DP flow calculation ---
    flows = compute_battery_next_soc_with_flows(
        current_soc_wh=initial_soc_wh,
        min_soc_wh=min_soc_wh,
        max_soc_wh=max_soc_wh,
        charging_efficiency=bat_params["charging_efficiency"],
        discharging_efficiency=bat_params["discharging_efficiency"],
        ac_charge_factor=0.0,
        dc_charge_allowed=True,
        discharge_allowed=True,
        pv_wh=pv_wh,
        load_wh=load_wh,
        ev_draw_wh=ev_draw_wh,
        max_ac_charge_power_w=max_ac_power,
        ac_to_dc_efficiency=ac_to_dc_eff,
        dc_to_ac_efficiency=dc_to_ac_eff,
    )

    dp_cost = flows.grid_import * price - flows.grid_export * revenue
    print(f"\n  DP EnergyFlows:")
    print(f"    Next SoC: {flows.next_soc_wh:.0f} Wh")
    print(f"    Grid Import: {flows.grid_import:.0f} Wh")
    print(f"    Grid Export: {flows.grid_export:.0f} Wh")
    print(f"    Battery Discharge AC: {flows.battery_discharge_ac:.0f} Wh")
    print(f"    Losses: {flows.losses:.0f} Wh")
    print(f"    Cost: {dp_cost:.4f} €")

    # --- Simulation calculation ---
    # Create battery and inverter
    bat_params_obj = type('SolarPanelBatteryParameters', (), {
        'capacity_wh': bat_params['capacity_wh'],
        'charging_efficiency': bat_params['charging_efficiency'],
        'discharging_efficiency': bat_params['discharging_efficiency'],
        'min_soc_percentage': bat_params['min_soc_percentage'],
        'max_soc_percentage': bat_params['max_soc_percentage'],
        'initial_soc_percentage': bat_params['initial_soc_percentage'],
        'device_id': bat_params['device_id'],
        'max_charge_power_w': bat_params['capacity_wh'],
        'charge_rates': None,
    })()

    inv_params_obj = type('InverterParameters', (), {
        'battery_id': bat_params['device_id'],
        'ac_to_dc_efficiency': ac_to_dc_eff,
        'dc_to_ac_efficiency': dc_to_ac_eff,
        'max_ac_charge_power_w': max_ac_power,
        'max_power_wh': bat_params['capacity_wh'] * 2,
        'device_id': inv.get('device_id', 'inverter_0'),
    })()

    battery = Battery(parameters=bat_params_obj, prediction_hours=48)
    battery.soc_wh = initial_soc_wh
    battery.charge_array = np.zeros(48)
    battery.discharge_array = np.ones(48)  # discharge allowed

    inverter = Inverter(parameters=inv_params_obj, battery=battery)

    # Simulate one hour
    total_consumption = load_wh + ev_draw_wh
    grid_export, grid_import, losses, self_consumption = inverter.process_energy(
        pv_wh, total_consumption, hour
    )

    sim_soc_wh = battery.soc_wh
    sim_cost = grid_import * price - grid_export * revenue

    print(f"\n  Simulation (Inverter.process_energy):")
    print(f"    Next SoC: {sim_soc_wh:.0f} Wh")
    print(f"    Grid Import: {grid_import:.0f} Wh")
    print(f"    Grid Export: {grid_export:.0f} Wh")
    print(f"    Losses: {losses:.0f} Wh")
    print(f"    Cost: {sim_cost:.4f} €")

    print(f"\n  DIFF:")
    print(f"    SoC diff: {flows.next_soc_wh - sim_soc_wh:.0f} Wh")
    print(f"    Grid Import diff: {flows.grid_import - grid_import:.0f} Wh")
    print(f"    Grid Export diff: {flows.grid_export - grid_export:.0f} Wh")
    print(f"    Cost diff: {dp_cost - sim_cost:.4f} €")


def main():
    input_file = Path("tests/testdata/optimize_input_2.json")
    input_data = load_input(input_file)

    # Compare first few hours
    for hour in [0, 1, 2, 10, 20, 30]:
        compare_step(input_data, hour)


if __name__ == "__main__":
    main()
