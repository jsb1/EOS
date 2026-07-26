#!/usr/bin/env python3
"""Analyze SOC changes and their value contribution for GA vs DP vs HYBRID."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from akkudoktoreos.config.config import ConfigEOS
from akkudoktoreos.optimization.simulation.penalties import (
    battery_residual_value_penalty,
    ev_soc_miss_penalty,
)

# Load benchmark results
benchmark_file = Path("/tmp/benchmark_results.json")
with open(benchmark_file, "r") as f:
    all_results = json.load(f)

# Use optimize_input_2.json
for data in all_results:
    if data["file"] == "optimize_input_2.json":
        results = data
        break
else:
    print("optimize_input_2.json not found in benchmark results")
    sys.exit(1)

print("=" * 70)
print("SOC Change Analysis - optimize_input_2.json")
print("=" * 70)

# Device parameters from config
cfg = ConfigEOS()
bat = cfg.devices.batteries[0] if cfg.devices.batteries else None
ev = cfg.devices.electric_vehicles[0] if cfg.devices.electric_vehicles else None
inv = cfg.devices.inverters[0] if cfg.devices.inverters else None

print(f"\nDevice Parameters:")
print(f"  Battery: capacity={bat.capacity_wh}Wh, min_soc={bat.min_soc_percentage}%, max_soc={bat.max_soc_percentage}%")
print(f"  EV: capacity={ev.capacity_wh}Wh, min_soc={ev.min_soc_percentage}%, max_soc={ev.max_soc_percentage}%")
print(f"  Inverter: dc_to_ac_eff={inv.dc_to_ac_efficiency}, ac_to_dc_eff={inv.ac_to_dc_efficiency}")

# EMS parameters
ems = cfg.ems
print(f"\nEMS price_per_wh_battery: {ems.price_per_wh_battery}")

print("\n" + "=" * 70)
print("Solver Comparison:")
print("=" * 70)

for solver in results["solvers"]:
    name = solver["solver"]
    print(f"\n{name}:")
    print(f"  Balance: {solver['total_balance']:.4f}€")
    print(f"  Costs: {solver['total_costs']:.4f}€")
    print(f"  Revenue: {solver['total_revenue']:.4f}€")
    print(f"  Losses: {solver['total_losses']:.2f}Wh")

    # Battery SOC analysis
    if "battery_end_soc" in solver:
        end_soc = solver["battery_end_soc"]
        # Assuming initial SOC from input data
        init_soc = 80  # from optimize_input_2.json pv_akku.init_soc
        soc_change = end_soc - init_soc

        # Calculate battery residual value penalty
        if bat and inv:
            end_soc_wh = (end_soc / 100.0) * bat.capacity_wh
            penalty = battery_residual_value_penalty(
                battery_energy_content_wh=end_soc_wh,
                dc_to_ac_efficiency=inv.dc_to_ac_efficiency,
                price_per_wh_battery=ems.price_per_wh_battery,
            )
            print(f"  Battery SOC: {init_soc}% -> {end_soc}% (change: {soc_change:+.1f}%)")
            print(f"  Battery end energy: {end_soc_wh:.0f}Wh")
            print(f"  Battery residual penalty: {penalty:.4f}€")

        # Battery energy value calculation
        if bat:
            soc_change_wh = (soc_change / 100.0) * bat.capacity_wh
            # Value of stored energy (round-trip efficiency)
            rt_eff = inv.dc_to_ac_efficiency * inv.ac_to_dc_efficiency if inv else 1.0
            value_stored = soc_change_wh * rt_eff * ems.price_per_wh_battery
            print(f"  Battery SOC change: {soc_change_wh:.0f}Wh")
            print(f"  Round-trip value (stored energy): {value_stored:.4f}€")

    # EV analysis
    if "has_ev_charge" in solver:
        print(f"  EV charged: {solver['has_ev_charge']}")

print("\n" + "=" * 70)
print("Value Calculation Explanation:")
print("=" * 70)
print("""
Battery SOC Change Value:
- battery_residual_value_penalty penalizes ALL remaining energy at horizon end
  Formula: -(end_soc_wh * dc_to_ac_eff * price_per_wh_battery)
  This is a NEGATIVE penalty (reduces balance)
- Higher end SOC = larger penalty = lower balance
- DP should prefer lower end SOC to avoid penalty

EV SOC Change Value:
- ev_soc_miss_penalty only penalizes if EV < min_soc_percentage at end
- Formula: abs(min_soc - ev_soc) * penalty_factor if ev_soc < min_soc else 0
- No penalty for EV > min_soc_percentage (current implementation)
- EV charging beyond min_soc_percentage is "free" in the optimization

Missing Value Components:
1. Battery SOC delta value: Not directly considered
   - If end_soc > start_soc, we paid for stored energy
   - If end_soc < start_soc, we used stored (free) energy
2. EV SOC delta value: Not considered
   - EV charging consumes grid energy (cost)
   - But EV SOC at end is "free" if >= min_soc_percentage
3. Opportunity cost of EV charging:
   - Energy used for EV could have been used for battery
   - EV charging during expensive hours reduces balance
""")

# Load input data to check initial SOC
input_file = Path("tests/testdata/optimize_input_2.json")
with open(input_file, "r") as f:
    input_data = json.load(f)

print("\nInput Data:")
if "pv_akku" in input_data:
    print(f"  pv_akku.init_soc: {input_data['pv_akku'].get('init_soc', 'N/A')}")
if "eauto" in input_data:
    print(f"  eauto.init_soc: {input_data['eauto'].get('init_soc', 'N/A')}")
