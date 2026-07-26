#!/usr/bin/env python3
"""Debug DP terminal penalty calculation in context."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main():
    input_path = Path("tests/testdata/optimize_input_2.json")
    with open(input_path) as f:
        raw_data = json.load(f)

    from akkudoktoreos.optimization.dp.dpoptimizer import DPOptimizer, NUM_SOC_LEVELS
    from akkudoktoreos.optimization.dp.dpparams import DPOptimizationParameters
    import numpy as np

    # Create DP params
    dp_params = DPOptimizationParameters(**raw_data)

    # Create DP optimizer
    dp_opt = DPOptimizer()

    # Test terminal penalty for different EV SoCs
    ev = dp_params.ev
    bat = dp_params.pv_battery

    if ev is None:
        print("No EV in params")
        return

    print(f"EV: capacity={ev.capacity_wh}, min_soc={ev.min_soc_percentage}%, max_soc={ev.max_soc_percentage}%")
    print(f"Battery: capacity={bat.capacity_wh}, initial_soc={bat.initial_soc_percentage}%")

    # Test terminal penalty for different EV indices
    print(f"\nTerminal Penalty for different EV indices:")
    for ev_idx in [0, 20, 40, 49, 50]:
        ac_charge_hours = np.zeros(48)
        penalty = dp_opt._compute_terminal_penalty(
            bat_idx=50,  # Battery at 100%
            ev_idx=ev_idx,
            ac_charge_hours=ac_charge_hours,
            params=dp_params,
        )
        print(f"  ev_idx={ev_idx}, terminal_penalty={penalty:.2f}")

    # Test with battery at different SoCs
    print(f"\nTerminal Penalty for different Battery indices (EV fixed at 40):")
    for bat_idx in [0, 25, 40, 50]:
        ac_charge_hours = np.zeros(48)
        penalty = dp_opt._compute_terminal_penalty(
            bat_idx=bat_idx,
            ev_idx=40,  # EV at 80%
            ac_charge_hours=ac_charge_hours,
            params=dp_params,
        )
        print(f"  bat_idx={bat_idx}, terminal_penalty={penalty:.2f}")


if __name__ == "__main__":
    main()
