#!/usr/bin/env python3
"""Debug DP cost calculation for specific hours."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from akkudoktoreos.optimization.simulation.physics import compute_battery_next_soc_with_flows


def analyze_hour(hour: int, pv_wh: float, load_wh: float, ev_draw_wh: float, price: float, revenue: float):
    """Compare different battery actions for one hour."""
    # Parameters from optimize_input_2.json
    bat_soc_wh = 21120.0  # 80% of 26400Wh
    bat_min_wh = 0.0
    bat_max_wh = 26400.0
    charging_efficiency = 0.95
    discharging_efficiency = 0.95
    ac_to_dc_eff = 0.96
    dc_to_ac_eff = 0.96
    max_charge_power_w = 2500.0  # From inverter max_ac_charge_power_w

    print(f"\n{'='*80}")
    print(f"HOUR {hour}: PV={pv_wh:.0f}Wh, Load={load_wh:.0f}Wh, EV={ev_draw_wh:.0f}Wh, Price={price:.5f}€/Wh")
    print(f"{'='*80}")

    actions = [
        {"name": "AC-Laden (Grid)", "ac_rate": 1.0, "dc_allowed": True, "discharge": False},
        {"name": "Kein Laden/Entladen", "ac_rate": 0.0, "dc_allowed": True, "discharge": False},
        {"name": "Nur Entladen", "ac_rate": 0.0, "dc_allowed": False, "discharge": True},
    ]

    for a in actions:
        name = a["name"]
        ac_rate = a["ac_rate"]
        dc_allowed = a["dc_allowed"]
        discharge = a["discharge"]
        flows = compute_battery_next_soc_with_flows(
            current_soc_wh=bat_soc_wh,
            min_soc_wh=bat_min_wh,
            max_soc_wh=bat_max_wh,
            charging_efficiency=charging_efficiency,
            discharging_efficiency=discharging_efficiency,
            ac_charge_factor=ac_rate,
            dc_charge_allowed=dc_allowed,
            discharge_allowed=discharge,
            pv_wh=pv_wh,
            load_wh=load_wh,
            ev_draw_wh=ev_draw_wh,
            ac_to_dc_efficiency=ac_to_dc_eff,
            dc_to_ac_efficiency=dc_to_ac_eff,
            max_ac_charge_power_w=max_charge_power_w,
        )

        cost = flows.grid_import * price - flows.grid_export * revenue
        print(f"\n{name}:")
        print(f"  Grid Import: {flows.grid_import:.1f}Wh")
        print(f"  Grid Export: {flows.grid_export:.1f}Wh")
        print(f"  Battery Next SoC: {flows.next_soc_wh:.1f}Wh ({flows.next_soc_wh/bat_max_wh*100:.1f}%)")
        print(f"  Losses: {flows.losses:.1f}Wh")
        print(f"  Cost: {cost:.4f}€")
        print(f"  (Grid Import Cost: {flows.grid_import * price:.4f}€, Grid Export Revenue: {flows.grid_export * revenue:.4f}€)")


def main():
    # Data from optimize_input_2.json - hours 16-20 where DP vs GA differ
    data = [
        # (hour, pv, load, ev_draw, price, revenue)
        (16, 1800.0, 1053.07, 2500.0, 0.0002004, 0.00007),
        (17, 0.0, 1063.91, 2500.0, 0.0003054, 0.00007),
        (18, 0.0, 1320.56, 2500.0, 0.0003049, 0.00007),
        (19, 0.0, 1132.03, 2500.0, 0.0002998, 0.00007),
        (20, 0.0, 1163.67, 2500.0, 0.0002948, 0.00007),
    ]

    for hour, pv, load, ev, price, rev in data:
        analyze_hour(hour, pv, load, ev, price, rev)


if __name__ == "__main__":
    main()
