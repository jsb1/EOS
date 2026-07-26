#!/usr/bin/env python3
"""Debug DP terminal penalty integration."""

import sys
import json
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

    from akkudoktoreos.optimization.dp.dpoptimizer import DPOptimizer, NUM_SOC_LEVELS, INF_COST
    from akkudoktoreos.optimization.dp.dpparams import DPOptimizationParameters

    # Create DP params
    dp_params = DPOptimizationParameters(**raw_data)
    ev = dp_params.ev
    horizon = len(dp_params.ems.electricity_price_per_wh)

    # Create optimizer
    dp_opt = DPOptimizer()

    # Test terminal penalty calculation for different EV indices
    print("Terminal penalties for different EV indices (horizon=48):")
    print("="*70)

    for ev_idx in [0, 20, 30, 40, 45, 49, 50]:
        ev_soc_wh = dp_opt._get_soc_from_index(ev_idx, 0.0, ev.capacity_wh)
        ev_soc_pct = (ev_soc_wh / ev.capacity_wh) * 100.0

        ac_charge_hours = np.zeros(horizon)
        penalty = dp_opt._compute_terminal_penalty(ev_idx, ev_idx, ac_charge_hours, dp_params)

        print(f"ev_idx={ev_idx:3d}, EV={ev_soc_pct:6.2f}%, penalty={penalty:10.2f}€")

    print("\nNow running full DP optimization...")
    print("="*70)

    dp_solution = dp_opt.optimize(params=dp_params, ha_params=None, start_hour=10, worst_case=False)

    result = dp_solution.result
    print(f"\nBalance: {result.total_balance:.4f}€")
    print(f"EV Final SoC: {result.ev_soc_per_hour[-1]:.2f}%")
    print(f"Optimal cost from DP: {dp_solution.optimal_cost:.4f}€")


if __name__ == "__main__":
    main()
