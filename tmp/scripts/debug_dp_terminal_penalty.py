#!/usr/bin/env python3
"""Debug DP terminal penalty calculation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main():
    # Set config BEFORE any imports that trigger get_config()
    from akkudoktoreos.config.config import ConfigEOS
    ConfigEOS._init_config_eos = {
        "optimization": {
            "genetic": {
                "penalties": {"ev_soc_miss": 50, "ac_charge_break_even": 0}
            }
        }
    }

    from akkudoktoreos.optimization.dp.dpoptimizer import DPOptimizer

    dp_opt = DPOptimizer()

    # Check what penalty_factor is read
    try:
        pf = float(dp_opt.config.optimization.genetic.penalties.get("ev_soc_miss", 10.0))
        print(f"penalty_factor from config: {pf}")
    except Exception as e:
        print(f"Error reading penalty_factor: {e}")
        pf = 10.0

    # Test penalty calculation
    from akkudoktoreos.optimization.simulation.penalties import ev_soc_miss_penalty

    # EV at 99.4%, min_soc=80%, penalty_factor=50
    penalty = ev_soc_miss_penalty(99.4, 80.0, 100.0, pf)
    print(f"EV at 99.4%: penalty = {penalty:.2f}")

    # EV at 80%, min_soc=80%
    penalty = ev_soc_miss_penalty(80.0, 80.0, 100.0, pf)
    print(f"EV at 80.0%: penalty = {penalty:.2f}")

    # EV at 85%, min_soc=80%
    penalty = ev_soc_miss_penalty(85.0, 80.0, 100.0, pf)
    print(f"EV at 85.0%: penalty = {penalty:.2f}")

    # EV at 5%, min_soc=80%
    penalty = ev_soc_miss_penalty(5.0, 80.0, 100.0, pf)
    print(f"EV at 5.0%: penalty = {penalty:.2f}")

    # Compare: Grid cost for charging EV vs penalty
    # EV capacity = 60000 Wh
    # From 5% to 80%: 0.75 * 60000 = 45000 Wh
    # From 5% to 99.4%: 0.944 * 60000 = 56640 Wh
    # Extra energy: 11640 Wh
    # Avg grid price ~0.0003 €/Wh
    # Extra grid cost: 11640 * 0.0003 = 3.49€
    # Extra penalty: (99.4-80)*50 = 970€
    print("\nCost comparison:")
    print(f"  Extra grid cost (5%->99.4% vs 5%->80%): ~3.49€")
    print(f"  Extra penalty (99.4% vs 80%): {(99.4-80)*pf:.2f}€")
    print(f"  DP SHOULD choose EV=80% with this penalty!")


if __name__ == "__main__":
    main()
