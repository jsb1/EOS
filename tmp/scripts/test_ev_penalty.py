#!/usr/bin/env python3
"""Test EV penalty calculation."""

from akkudoktoreos.optimization.simulation.penalties import ev_soc_miss_penalty

# Test cases
test_cases = [
    (50, 80, 100, 10.0),   # Below min
    (80, 80, 100, 10.0),   # Exactly at min
    (90, 80, 100, 10.0),   # Above min
    (99.4, 80, 100, 10.0), # EV result from DP
    (100, 80, 100, 10.0),  # At max
]

print("EV Penalty Test:")
for ev_soc, min_soc, max_soc, penalty_factor in test_cases:
    penalty = ev_soc_miss_penalty(ev_soc, min_soc, max_soc, penalty_factor)
    print(f"  EV={ev_soc:.1f}%, min={min_soc}%, max={max_soc}%, factor={penalty_factor} → Penalty={penalty:.2f}")
