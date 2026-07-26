#!/usr/bin/env python3
"""Debug: Check if terminal penalty is applied in last step of DP."""

import sys
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main():
    input_path = Path("tests/testdata/optimize_input_2.json")
    with open(input_path) as f:
        raw_data = json.load(f)

    # Set config BEFORE any imports that trigger get_config()
    from akkudoktoreos.config.config import ConfigEOS
    ConfigEOS._init_config_eos = {
        "optimization": {
            "genetic": {
                "penalties": {"ev_soc_miss": 50, "ac_charge_break_even": 0}
            }
        }
    }

    # Import after config setup
    from akkudoktoreos.optimization.dp.dpoptimizer import DPOptimizer, NUM_SOC_LEVELS, INF_COST
    from akkudoktoreos.optimization.dp.dpparams import DPOptimizationParameters
    from akkudoktoreos.optimization.simulation.penalties import ev_soc_miss_penalty

    # Create DP params
    dp_params = DPOptimizationParameters(**raw_data)
    ev = dp_params.ev
    bat = dp_params.pv_battery
    horizon = len(dp_params.ems.electricity_price_per_wh)

    dp_opt = DPOptimizer()

    # Check penalty_factor
    try:
        pf = float(dp_opt.config.optimization.genetic.penalties.get("ev_soc_miss", 10.0))
        print(f"penalty_factor from config: {pf}")
    except Exception as e:
        print(f"Error reading penalty_factor: {e}")
        pf = 10.0

    # Test penalty calculation
    penalty_99 = ev_soc_miss_penalty(99.4, ev.min_soc_percentage, ev.max_soc_percentage, pf)
    penalty_80 = ev_soc_miss_penalty(80.0, ev.min_soc_percentage, ev.max_soc_percentage, pf)
    print(f"EV at 99.4%: penalty = {penalty_99:.2f}€")
    print(f"EV at 80.0%: penalty = {penalty_80:.2f}€")

    # Manually check V[horizon] for different EV indices after running DP
    # We'll monkey-patch the optimize method to print V[horizon] values
    original_optimize = dp_opt.optimize

    def debug_optimize(params, ha_params=None, start_hour=0, worst_case=False, optimize_ev=True, optimize_dc_charge=True):
        # Run original
        return original_optimize(params, ha_params, start_hour, worst_case, optimize_ev, optimize_dc_charge)

    # Instead, let's trace the terminal penalty computation
    original_terminal = dp_opt._compute_terminal_penalty
    call_count = 0

    def traced_terminal(bat_idx, ev_idx, ac_charge_hours, params):
        nonlocal call_count
        call_count += 1
        penalty = original_terminal(bat_idx, ev_idx, ac_charge_hours, params)
        if call_count <= 10 or penalty > 100:
            ev_soc_wh = dp_opt._get_soc_from_index(ev_idx, 0.0, ev.capacity_wh)
            ev_soc_pct = (ev_soc_wh / ev.capacity_wh) * 100.0
            print(f"Terminal penalty call #{call_count}: bat_idx={bat_idx}, ev_idx={ev_idx}, EV={ev_soc_pct:.1f}%, penalty={penalty:.2f}€")
        return penalty

    dp_opt._compute_terminal_penalty = traced_terminal

    print(f"\nRunning DP with traced terminal penalty (horizon={horizon})...")
    print("="*70)

    dp_solution = debug_optimize(dp_params, ha_params=None, start_hour=10, worst_case=False)

    print(f"\nTotal terminal penalty calls: {call_count}")

    result = dp_solution.result
    print(f"\nResult:")
    print(f"  Balance: {result.total_balance:.4f}€")
    print(f"  EV Final SoC: {result.ev_soc_per_hour[-1]:.2f}%")


if __name__ == "__main__":
    main()
