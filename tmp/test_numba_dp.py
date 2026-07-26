#!/usr/bin/env python3
"""Test script to benchmark Numba-JIT on DP backward pass."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from numba import njit, prange

# Constants
NUM_SOC_LEVELS = 52
INF_COST = 1e18

# Test with simple synthetic data
horizon = 48
num_actions = 24

# Create synthetic action arrays
action_ac_rate_idx = np.random.randint(0, 3, size=num_actions, dtype=np.int32)
action_dc_charge = np.random.randint(0, 2, size=num_actions, dtype=np.int32)
action_discharge = np.random.randint(0, 2, size=num_actions, dtype=np.int32)
action_ev_rate_idx = np.random.randint(0, 3, size=num_actions, dtype=np.int32)

bat_charge_rates_arr = np.array([0.0, 0.5, 1.0], dtype=np.float64)

# Synthetic EV transition tables
ev_trans_next = np.random.randint(0, NUM_SOC_LEVELS, size=(num_actions, NUM_SOC_LEVELS), dtype=np.int32)
ev_trans_draw = np.random.rand(num_actions, NUM_SOC_LEVELS) * 5000.0

bat_soc_lookup = np.linspace(0, 10000, NUM_SOC_LEVELS, dtype=np.float64)

# Value function
V = np.full((horizon + 1, NUM_SOC_LEVELS, NUM_SOC_LEVELS, 2), INF_COST, dtype=np.float64)
V[horizon, :, :, :] = np.random.rand(NUM_SOC_LEVELS, NUM_SOC_LEVELS, 2) * 100.0

# Policy storage
policy_action = np.full((horizon, NUM_SOC_LEVELS, NUM_SOC_LEVELS, 2), -1, dtype=np.int32)
policy_bat = np.full((horizon, NUM_SOC_LEVELS, NUM_SOC_LEVELS, 2), -1, dtype=np.int32)
policy_ev = np.full((horizon, NUM_SOC_LEVELS, NUM_SOC_LEVELS, 2), -1, dtype=np.int32)
policy_appliance = np.full((horizon, NUM_SOC_LEVELS, NUM_SOC_LEVELS, 2), -1, dtype=np.int8)

# Synthetic time series
pv_array = np.random.rand(horizon) * 3000
load_array = np.random.rand(horizon) * 2000
price_array = np.random.rand(horizon) * 0.0003
revenue_array = np.random.rand(horizon) * 0.0001

has_ev = True
has_ha = True
worst_case = False

@njit(cache=True, parallel=True)
def _bellman_backward_pass(
    V,
    policy_action,
    policy_bat,
    policy_ev,
    policy_appliance,
    bat_soc_lookup,
    ev_trans_next,
    ev_trans_draw,
    action_ac_rate_idx,
    action_dc_charge,
    action_discharge,
    action_ev_rate_idx,
    bat_charge_rates_arr,
    pv_array,
    load_array,
    price_array,
    revenue_array,
    ha_start_windows_set,
    horizon,
    num_actions,
    has_ev,
    has_ha,
    worst_case,
):
    """Numba-accelerated backward pass (simplified without battery physics)."""
    for t in range(horizon - 1, -1, -1):
        pv_wh = pv_array[t] if t < len(pv_array) else 0.0
        load_wh = load_array[t] if t < len(load_array) else 0.0
        price = price_array[t] if t < len(price_array) else 0.0
        revenue = revenue_array[t] if t < len(revenue_array) else 0.0

        ha_can_start = False
        if has_ha:
            for w in ha_start_windows_set:
                if w == t:
                    ha_can_start = True
                    break

        for bat_idx in range(NUM_SOC_LEVELS):
            for ev_idx in range(NUM_SOC_LEVELS):
                for ap_started in range(2):
                    best_cost = INF_COST
                    best_a_idx = -1
                    best_next_bat = -1
                    best_next_ev = -1
                    best_next_ap = -1

                    for a_idx in range(num_actions):
                        new_ap_started = ap_started

                        # EV transition
                        if has_ev:
                            next_ev_idx = ev_trans_next[a_idx, ev_idx]
                            ev_draw_wh = ev_trans_draw[a_idx, ev_idx]
                            if ev_idx == NUM_SOC_LEVELS - 1 and next_ev_idx > ev_idx:
                                continue
                        else:
                            next_ev_idx = ev_idx
                            ev_draw_wh = 0.0

                        # Simplified battery transition (no physics function)
                        ac_rate = bat_charge_rates_arr[action_ac_rate_idx[a_idx]]
                        dc_charge_allowed = bool(action_dc_charge[a_idx])
                        discharge_allowed = bool(action_discharge[a_idx])

                        bat_soc_wh = bat_soc_lookup[bat_idx]
                        net_energy = pv_wh - load_wh - ev_draw_wh

                        if ac_rate > 0:
                            net_energy += 5000 * ac_rate  # Simplified AC charge

                        # Clamp
                        next_bat_soc = bat_soc_wh + net_energy
                        next_bat_soc = max(0.0, min(10000.0, next_bat_soc))
                        ratio = (next_bat_soc / 10000.0) * (NUM_SOC_LEVELS - 1)
                        next_bat_idx = int(max(0, min(NUM_SOC_LEVELS - 1, ratio)))

                        # Immediate cost
                        immediate_cost = max(0.0, load_wh + ev_draw_wh - pv_wh) * price

                        # Future cost
                        future_cost = V[t + 1, next_bat_idx, next_ev_idx, new_ap_started]
                        if future_cost >= INF_COST:
                            continue

                        new_cost = immediate_cost + future_cost
                        if worst_case:
                            new_cost = -new_cost

                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_a_idx = a_idx
                            best_next_bat = next_bat_idx
                            best_next_ev = next_ev_idx
                            best_next_ap = new_ap_started

                    # HA start
                    if not ap_started and ha_can_start:
                        future_cost = V[t + 1, bat_idx, ev_idx, 1]
                        if future_cost < best_cost:
                            best_cost = future_cost
                            best_a_idx = -2
                            best_next_bat = bat_idx
                            best_next_ev = ev_idx
                            best_next_ap = 1

                    if best_a_idx >= -2:
                        V[t, bat_idx, ev_idx, ap_started] = best_cost
                        policy_action[t, bat_idx, ev_idx, ap_started] = best_a_idx
                        policy_bat[t, bat_idx, ev_idx, ap_started] = best_next_bat
                        policy_ev[t, bat_idx, ev_idx, ap_started] = best_next_ev
                        policy_appliance[t, bat_idx, ev_idx, ap_started] = best_next_ap


# Convert ha_start_windows to tuple for numba
ha_start_windows_set = tuple(range(horizon))

# Warmup run
print("Warming up Numba JIT...")
V_test = V.copy()
policy_action_test = policy_action.copy()
policy_bat_test = policy_bat.copy()
policy_ev_test = policy_ev.copy()
policy_appliance_test = policy_appliance.copy()

t0 = time.perf_counter()
_bellman_backward_pass(
    V_test,
    policy_action_test,
    policy_bat_test,
    policy_ev_test,
    policy_appliance_test,
    bat_soc_lookup,
    ev_trans_next,
    ev_trans_draw,
    action_ac_rate_idx,
    action_dc_charge,
    action_discharge,
    action_ev_rate_idx,
    bat_charge_rates_arr,
    pv_array,
    load_array,
    price_array,
    revenue_array,
    ha_start_windows_set,
    horizon,
    num_actions,
    has_ev,
    has_ha,
    worst_case,
)
t1 = time.perf_counter()
print(f"First run (with JIT compilation): {t1 - t0:.3f}s")

# Benchmark run
V_test = V.copy()
policy_action_test = policy_action.copy()
policy_bat_test = policy_bat.copy()
policy_ev_test = policy_ev.copy()
policy_appliance_test = policy_appliance.copy()

t0 = time.perf_counter()
_bellman_backward_pass(
    V_test,
    policy_action_test,
    policy_bat_test,
    policy_ev_test,
    policy_appliance_test,
    bat_soc_lookup,
    ev_trans_next,
    ev_trans_draw,
    action_ac_rate_idx,
    action_dc_charge,
    action_discharge,
    action_ev_rate_idx,
    bat_charge_rates_arr,
    pv_array,
    load_array,
    price_array,
    revenue_array,
    ha_start_windows_set,
    horizon,
    num_actions,
    has_ev,
    has_ha,
    worst_case,
)
t1 = time.perf_counter()
print(f"JIT-compiled run: {t1 - t0:.3f}s")
print("Numba JIT test successful!")
