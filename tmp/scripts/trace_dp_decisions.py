#!/usr/bin/env python3
"""Trace DP decisions step-by-step to understand why it chooses wrong strategy."""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from akkudoktoreos.optimization.simulation.physics import compute_battery_next_soc_with_flows, compute_ev_next_soc


def max_charge_power_w(params) -> float:
    """Get max charge power from battery params."""
    return params.capacity_wh * 0.5


def analyze_dp_horizon(input_data: dict):
    """Simulate DP decision-making for the full horizon."""
    ems = input_data["ems"]
    bat = input_data["pv_akku"]
    ev = input_data["eauto"]
    inv = input_data["inverter"]

    horizon = len(ems["strompreis_euro_pro_wh"])
    pv_array = np.array(ems["pv_prognose_wh"])
    load_array = np.array(ems["gesamtlast"])
    price_array = np.array(ems["strompreis_euro_pro_wh"])

    bat_capacity = bat["capacity_wh"]
    bat_initial_soc = bat["initial_soc_percentage"]
    bat_min_soc = bat.get("min_soc_percentage", 0)
    bat_max_soc = bat.get("max_soc_percentage", 100)
    bat_charging_eff = bat.get("charging_efficiency", 0.95)
    bat_discharging_eff = bat.get("discharging_efficiency", 0.95)

    ev_capacity = ev["capacity_wh"]
    ev_initial_soc = ev["initial_soc_percentage"]
    ev_min_soc = ev["min_soc_percentage"]
    ev_max_soc = ev.get("max_soc_percentage", 100)
    ev_charging_eff = ev.get("charging_efficiency", 0.95)
    ev_max_charge_w = ev.get("max_charge_power_w", ev_capacity * 0.5)

    dc_to_ac_eff = inv.get("dc_to_ac_efficiency", 0.96)
    ac_to_dc_eff = inv.get("ac_to_dc_efficiency", 0.96)
    max_ac_charge = inv.get("max_ac_charge_power_w", bat_capacity * 0.5)

    bat_min_wh = bat_min_soc / 100.0 * bat_capacity
    bat_max_wh = bat_max_soc / 100.0 * bat_capacity
    bat_initial_wh = bat_initial_soc / 100.0 * bat_capacity

    ev_min_wh = ev_min_soc / 100.0 * ev_capacity
    ev_max_wh = ev_max_soc / 100.0 * ev_capacity
    ev_initial_wh = ev_initial_soc / 100.0 * ev_capacity

    print(f"Battery: capacity={bat_capacity}Wh, initial={bat_initial_wh:.0f}Wh ({bat_initial_soc}%), range=[{bat_min_wh:.0f}, {bat_max_wh:.0f}]")
    print(f"EV: capacity={ev_capacity}Wh, initial={ev_initial_wh:.0f}Wh ({ev_initial_soc}%), target={ev_min_soc}-{ev_max_soc}%")
    print(f"EV needs: {ev_min_wh - ev_initial_wh:.0f}Wh to reach {ev_min_soc}%")
    print(f"Horizon: {horizon} hours")
    print()

    # Simple greedy comparison: for each hour, what's the best action?
    # Strategy A: Always discharge battery first, then grid
    # Strategy B: Always AC-charge battery, then grid for EV
    # Strategy C: Hybrid - discharge when price high, charge when price low

    bat_soc_a = bat_initial_wh  # Discharge first
    bat_soc_b = bat_initial_wh  # AC-charge
    ev_soc_a = ev_initial_wh
    ev_soc_b = ev_initial_wh

    cost_a = 0.0
    cost_b = 0.0

    print("=" * 100)
    print(f"{'Hour':>4} | {'PV':>7} | {'Load':>7} | {'Price':>8} | {'Strategy A (Discharge)':>25} | {'Strategy B (AC-Charge)':>25}")
    print("=" * 100)

    for t in range(horizon):
        pv = pv_array[t] if t < len(pv_array) else 0.0
        load = load_array[t] if t < len(load_array) else 0.0
        price = price_array[t] if t < len(price_array) else 0.0

        # EV needs to charge: priority for both strategies
        ev_target_wh = ev_min_wh
        ev_deficit = max(0, ev_target_wh - ev_soc_a)
        max_ev_charge = ev_max_charge_w
        ev_charge_a = min(ev_deficit, max_ev_charge)

        # Strategy A: Discharge battery first for EV, then grid
        # EV charging from battery
        ev_from_bat_a = min(ev_charge_a, (bat_soc_a - bat_min_wh) * bat_discharging_eff * dc_to_ac_eff)
        ev_from_grid_a = max(0, ev_charge_a - ev_from_bat_a)
        bat_soc_a -= ev_from_bat_a / (bat_discharging_eff * dc_to_ac_eff) if ev_from_bat_a > 0 else 0

        # Load from battery
        load_from_bat_a = min(load, (bat_soc_a - bat_min_wh) * bat_discharging_eff * dc_to_ac_eff)
        load_from_grid_a = max(0, load - load_from_bat_a)
        bat_soc_a -= load_from_bat_a / (bat_discharging_eff * dc_to_ac_eff) if load_from_bat_a > 0 else 0

        ev_soc_a += ev_charge_a * ev_charging_eff

        grid_import_a = ev_from_grid_a + load_from_grid_a
        cost_a += grid_import_a * price

        # Strategy B: AC-charge battery from grid, use grid for EV
        # AC-charge battery (like DP does)
        ac_charge_b = max_ac_charge
        dc_stored_b = ac_charge_b * ac_to_dc_eff * bat_charging_eff
        headroom_b = bat_max_wh - bat_soc_b
        dc_stored_b = min(dc_stored_b, headroom_b)
        bat_soc_b += dc_stored_b

        # EV from grid
        ev_charge_b = min(max(0, ev_target_wh - ev_soc_b), max_ev_charge)
        ev_soc_b += ev_charge_b * ev_charging_eff

        # Load from grid
        load_from_grid_b = load

        grid_import_b = ac_charge_b + ev_charge_b + load_from_grid_b
        cost_b += grid_import_b * price

        if t < 10 or t >= horizon - 5:
            print(f"{t:>4} | {pv:>7.0f} | {load:>7.0f} | {price:>8.5f} | "
                  f"B:{bat_soc_a/bat_capacity*100:5.1f}%, E:{ev_soc_a/ev_capacity*100:5.1f}%, G:{grid_import_a:7.0f}Wh | "
                  f"B:{bat_soc_b/bat_capacity*100:5.1f}%, E:{ev_soc_b/ev_capacity*100:5.1f}%, G:{grid_import_b:7.0f}Wh")

    print("=" * 100)
    print(f"\nFinal Results:")
    print(f"Strategy A (Discharge first): Cost={cost_a:.2f}€, Bat={bat_soc_a/bat_capacity*100:.1f}%, EV={ev_soc_a/ev_capacity*100:.1f}%")
    print(f"Strategy B (AC-Charge): Cost={cost_b:.2f}€, Bat={bat_soc_b/bat_capacity*100:.1f}%, EV={ev_soc_b/ev_capacity*100:.1f}%")
    print(f"Difference: {cost_b - cost_a:.2f}€")


def main():
    input_path = Path("tests/testdata/optimize_input_2.json")
    with open(input_path) as f:
        input_data = json.load(f)

    analyze_dp_horizon(input_data)


if __name__ == "__main__":
    main()
