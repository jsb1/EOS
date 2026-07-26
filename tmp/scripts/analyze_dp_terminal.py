#!/usr/bin/env python3
"""Analyze DP terminal penalty and strategy for optimize_input_2.json."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from akkudoktoreos.optimization.simulation.penalties import (
    battery_residual_value_penalty,
    ev_soc_miss_penalty,
    ac_charge_break_even_penalty,
)
import numpy as np

def load_input(filepath: Path) -> dict:
    with open(filepath) as f:
        return json.load(f)

def analyze_terminal_penalty(input_data: dict, bat_end_soc_pct: float, ev_end_soc_pct: float):
    """Calculate what terminal penalty would be applied."""
    ems = input_data["ems"]
    bat = input_data["pv_akku"]
    ev = input_data["eauto"]
    inv = input_data["inverter"]
    
    bat_capacity = bat["capacity_wh"]
    bat_end_soc_wh = bat_capacity * bat_end_soc_pct / 100.0
    price_per_wh_battery = ems["preis_euro_pro_wh_akku"]
    dc_to_ac_eff = inv.get("dc_to_ac_efficiency", 0.95)
    
    # Battery residual penalty
    bat_penalty = battery_residual_value_penalty(
        battery_energy_content_wh=bat_end_soc_wh,
        dc_to_ac_efficiency=dc_to_ac_eff,
        price_per_wh_battery=price_per_wh_battery,
    )
    
    # EV miss penalty
    ev_penalty = ev_soc_miss_penalty(
        ev_soc_percentage=ev_end_soc_pct,
        min_soc_percentage=ev["min_soc_percentage"],
        max_soc_percentage=ev.get("max_soc_percentage", 100),
        penalty_factor=10.0,
    )
    
    print(f"\nTerminal Penalty Analysis:")
    print(f"  price_per_wh_battery: {price_per_wh_battery}")
    print(f"  Battery End SoC: {bat_end_soc_pct:.1f}% = {bat_end_soc_wh:.0f} Wh")
    print(f"  Battery residual penalty: {bat_penalty:.4f} €")
    print(f"  EV End SoC: {ev_end_soc_pct:.1f}%")
    print(f"  EV min_soc: {ev['min_soc_percentage']}%")
    print(f"  EV miss penalty: {ev_penalty:.4f} €")
    
    # What would penalty be with non-zero price_per_wh_battery?
    avg_price = np.mean(ems["strompreis_euro_pro_wh"])
    bat_penalty_with_price = battery_residual_value_penalty(
        battery_energy_content_wh=bat_end_soc_wh,
        dc_to_ac_efficiency=dc_to_ac_eff,
        price_per_wh_battery=avg_price,
    )
    print(f"\n  With price_per_wh_battery = avg_price ({avg_price:.6f}):")
    print(f"  Battery residual penalty: {bat_penalty_with_price:.4f} €")

def analyze_strategy(input_data: dict):
    """Analyze the optimization problem."""
    ems = input_data["ems"]
    bat = input_data["pv_akku"]
    ev = input_data["eauto"]
    
    # PV total
    pv_total = sum(ems["pv_prognose_wh"])
    load_total = sum(ems["gesamtlast"])
    net_energy = pv_total - load_total
    
    # EV needs
    ev_capacity = ev["capacity_wh"]
    ev_initial = ev["initial_soc_percentage"]
    ev_target = ev["min_soc_percentage"]
    ev_needed_wh = ev_capacity * (ev_target - ev_initial) / 100.0
    
    # Battery available
    bat_capacity = bat["capacity_wh"]
    bat_initial = bat["initial_soc_percentage"]
    bat_available_wh = bat_capacity * (bat_initial - bat.get("min_soc_percentage", 0)) / 100.0
    
    # Price analysis
    prices = ems["strompreis_euro_pro_wh"]
    avg_price = np.mean(prices)
    min_price = np.min(prices)
    max_price = np.max(prices)
    
    print("Strategy Analysis:")
    print(f"  PV Total: {pv_total:.0f} Wh")
    print(f"  Load Total: {load_total:.0f} Wh")
    print(f"  Net (PV - Load): {net_energy:.0f} Wh")
    print(f"  EV needs: {ev_needed_wh:.0f} Wh")
    print(f"  Battery available (from {bat_initial}% to {bat.get('min_soc_percentage', 0)}%): {bat_available_wh:.0f} Wh")
    print(f"  Grid energy needed (approx): {ev_needed_wh + load_total - pv_total:.0f} Wh")
    print(f"  Avg price: {avg_price:.6f} €/Wh")
    print(f"  Min price: {min_price:.6f} €/Wh")
    print(f"  Max price: {max_price:.6f} €/Wh")
    
    # Cost analysis
    grid_needed = ev_needed_wh + load_total - pv_total
    if grid_needed > 0:
        cost_if_all_grid = grid_needed * avg_price
        print(f"\n  Approx cost if all from grid: {cost_if_all_grid:.2f} €")
        
        # Cost if we use battery instead of grid
        cost_if_use_battery = (grid_needed - bat_available_wh) * avg_price if grid_needed > bat_available_wh else 0
        savings = cost_if_all_grid - cost_if_use_battery
        print(f"  Cost if we use battery first: {cost_if_use_battery:.2f} €")
        print(f"  Savings from using battery: {savings:.2f} €")

def main():
    input_file = Path("tests/testdata/optimize_input_2.json")
    input_data = load_input(input_file)
    
    # Analyze DP result (Battery 100%, EV ~80%)
    print("=== DP Strategy (Battery 100%, EV ~80%) ===")
    analyze_terminal_penalty(input_data, 100.0, 80.0)
    
    # Analyze GA result (Battery 32.61%, EV ~80%)
    print("\n=== GA Strategy (Battery 32.61%, EV ~80%) ===")
    analyze_terminal_penalty(input_data, 32.61, 80.0)
    
    # Strategy analysis
    print("\n=== Overall Strategy ===")
    analyze_strategy(input_data)

if __name__ == "__main__":
    main()
