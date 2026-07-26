#!/usr/bin/env python3
"""Test Numba-JIT with real battery physics inline."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from numba import njit

NUM_SOC_LEVELS = 52
INF_COST = 1e18

# Inline _self_consumption_ratio (from physics.py:151-164)
@njit(cache=True)
def _self_consumption_ratio(consumption, generation):
    if generation <= 0.0:
        return 1.0
    if consumption <= 0.0:
        return 0.0
    return consumption / generation

@njit(cache=True)
def compute_battery_flows(
    current_soc_wh,
    min_soc_wh,
    max_soc_wh,
    charging_efficiency,
    discharging_efficiency,
    ac_charge_factor,
    dc_charge_allowed,
    discharge_allowed,
    pv_wh,
    load_wh,
    ev_draw_wh,
    max_ac_charge_power_w,
    ac_to_dc_efficiency,
    dc_to_ac_efficiency,
):
    """Numba-compatible battery flow computation."""
    bsv = current_soc_wh
    total_consumption = load_wh + ev_draw_wh
    grid_import = 0.0
    grid_export = 0.0
    bat_charge_wh = 0.0
    ac_draw_grid = 0.0

    # AC charging from grid
    if ac_charge_factor > 0.0:
        ac_draw = max_ac_charge_power_w * ac_charge_factor
        dc_stored = ac_draw * ac_to_dc_efficiency * charging_efficiency
        headroom = max_soc_wh - bsv
        if headroom > 0:
            dc_stored = min(dc_stored, headroom)
            ac_draw_grid = ac_draw
            bat_charge_wh = dc_stored
        else:
            dc_stored = 0.0
            ac_draw_grid = 0.0
            bat_charge_wh = 0.0

    total_consumption += ac_draw_grid

    if pv_wh >= total_consumption:
        # PV surplus
        scr = _self_consumption_ratio(total_consumption, pv_wh)
        remaining_power = (pv_wh - total_consumption) * scr
        remaining_load_evq = (pv_wh - total_consumption) * (1.0 - scr)

        battery_discharge_ac = 0.0
        if remaining_load_evq > 0 and discharge_allowed:
            ac_needed = remaining_load_evq
            dc_request = ac_needed / dc_to_ac_efficiency / discharging_efficiency
            max_discharge_dc = max(0.0, (bsv - min_soc_wh))
            raw_used_dc = min(max_discharge_dc, dc_request)
            delivered_dc = raw_used_dc * discharging_efficiency
            battery_discharge_ac = delivered_dc * dc_to_ac_efficiency
            bsv -= raw_used_dc
            remaining_load_evq -= battery_discharge_ac

        if remaining_load_evq > 0:
            grid_import += remaining_load_evq

        # DC charging
        if remaining_power > 0 and dc_charge_allowed and bsv < max_soc_wh:
            headroom = max_soc_wh - bsv - bat_charge_wh
            if headroom > 0:
                pv_for_charge = min(remaining_power, headroom / charging_efficiency)
                stored = pv_for_charge * charging_efficiency
                bat_charge_wh += stored
                remaining_power -= pv_for_charge
                bsv += stored

        grid_export = max(0.0, remaining_power)
    else:
        # PV deficit
        shortfall = total_consumption - pv_wh
        battery_discharge_ac = 0.0

        if discharge_allowed and shortfall > 0:
            ac_needed = shortfall
            dc_request = ac_needed / dc_to_ac_efficiency / discharging_efficiency
            max_discharge_dc = max(0.0, (bsv - min_soc_wh))
            raw_used_dc = min(max_discharge_dc, dc_request)
            delivered_dc = raw_used_dc * discharging_efficiency
            battery_discharge_ac = delivered_dc * dc_to_ac_efficiency
            bsv -= raw_used_dc
            shortfall -= battery_discharge_ac

        grid_import += max(0.0, shortfall)

    bsv += bat_charge_wh
    next_soc = max(min_soc_wh, min(max_soc_wh, bsv))

    return next_soc, grid_import, grid_export


@njit(cache=True)
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
    bat_charge_rates_arr,
    pv_array,
    load_array,
    price_array,
    revenue_array,
    ha_start_windows_arr,
    horizon,
    num_actions,
    num_ha_windows,
    has_ev,
    has_ha,
    worst_case,
    bat_min_wh,
    bat_max_wh,
    bat_charging_eff,
    bat_discharging_eff,
    ac_to_dc_eff,
    dc_to_ac_eff,
    max_ac_power,
):
    for t in range(horizon - 1, -1, -1):
        pv_wh = pv_array[t] if t < len(pv_array) else 0.0
        load_wh = load_array[t] if t < len(load_array) else 0.0
        price = price_array[t] if t < len(price_array) else 0.0
        revenue = revenue_array[t] if t < len(revenue_array) else 0.0

        ha_can_start = False
        if has_ha:
            for wi in range(num_ha_windows):
                if ha_start_windows_arr[wi] == t:
                    ha_can_start = True
                    break

        for bat_idx in range(NUM_SOC_LEVELS):
            bat_soc_wh = bat_soc_lookup[bat_idx]

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

                        # Battery
                        ac_rate = bat_charge_rates_arr[action_ac_rate_idx[a_idx]]
                        dc_charge_allowed = bool(action_dc_charge[a_idx])
                        discharge_allowed = bool(action_discharge[a_idx])

                        next_soc, grid_import, grid_export = compute_battery_flows(
                            bat_soc_wh,
                            bat_min_wh,
                            bat_max_wh,
                            bat_charging_eff,
                            bat_discharging_eff,
                            ac_rate,
                            dc_charge_allowed,
                            discharge_allowed,
                            pv_wh,
                            load_wh,
                            ev_draw_wh,
                            max_ac_power,
                            ac_to_dc_eff,
                            dc_to_ac_eff,
                        )

                        next_bat_idx = int(max(0, min(NUM_SOC_LEVELS - 1,
                            (next_soc - bat_min_wh) / (bat_max_wh - bat_min_wh) * (NUM_SOC_LEVELS - 1))))

                        immediate_cost = grid_import * price - grid_export * revenue
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


# Test with synthetic data
horizon = 48
num_actions = 24

action_ac_rate_idx = np.random.randint(0, 3, size=num_actions, dtype=np.int32)
action_dc_charge = np.random.randint(0, 2, size=num_actions, dtype=np.int32)
action_discharge = np.random.randint(0, 2, size=num_actions, dtype=np.int32)
bat_charge_rates_arr = np.array([0.0, 0.5, 1.0], dtype=np.float64)

ev_trans_next = np.random.randint(0, NUM_SOC_LEVELS, size=(num_actions, NUM_SOC_LEVELS), dtype=np.int32)
ev_trans_draw = np.random.rand(num_actions, NUM_SOC_LEVELS) * 5000.0

bat_soc_lookup = np.linspace(1000, 9000, NUM_SOC_LEVELS, dtype=np.float64)

V = np.full((horizon + 1, NUM_SOC_LEVELS, NUM_SOC_LEVELS, 2), INF_COST, dtype=np.float64)
V[horizon, :, :, :] = np.random.rand(NUM_SOC_LEVELS, NUM_SOC_LEVELS, 2) * 100.0

policy_action = np.full((horizon, NUM_SOC_LEVELS, NUM_SOC_LEVELS, 2), -1, dtype=np.int32)
policy_bat = np.full((horizon, NUM_SOC_LEVELS, NUM_SOC_LEVELS, 2), -1, dtype=np.int32)
policy_ev = np.full((horizon, NUM_SOC_LEVELS, NUM_SOC_LEVELS, 2), -1, dtype=np.int32)
policy_appliance = np.full((horizon, NUM_SOC_LEVELS, NUM_SOC_LEVELS, 2), -1, dtype=np.int8)

pv_array = np.random.rand(horizon) * 3000
load_array = np.random.rand(horizon) * 2000
price_array = np.random.rand(horizon) * 0.0003
revenue_array = np.random.rand(horizon) * 0.0001

ha_start_windows_arr = np.arange(horizon, dtype=np.int32)

has_ev = True
has_ha = True
worst_case = False
bat_min_wh = 1000.0
bat_max_wh = 9000.0
bat_charging_eff = 0.95
bat_discharging_eff = 0.95
ac_to_dc_eff = 0.96
dc_to_ac_eff = 0.96
max_ac_power = 3600.0

# Warmup
print("Warming up Numba JIT...")
V_test = V.copy()
policy_action_test = policy_action.copy()
policy_bat_test = policy_bat.copy()
policy_ev_test = policy_ev.copy()
policy_appliance_test = policy_appliance.copy()

t0 = time.perf_counter()
_bellman_backward_pass(
    V_test, policy_action_test, policy_bat_test, policy_ev_test, policy_appliance_test,
    bat_soc_lookup, ev_trans_next, ev_trans_draw,
    action_ac_rate_idx, action_dc_charge, action_discharge, bat_charge_rates_arr,
    pv_array, load_array, price_array, revenue_array, ha_start_windows_arr,
    horizon, num_actions, horizon, has_ev, has_ha, worst_case,
    bat_min_wh, bat_max_wh, bat_charging_eff, bat_discharging_eff,
    ac_to_dc_eff, dc_to_ac_eff, max_ac_power,
)
t1 = time.perf_counter()
print(f"First run (with JIT): {t1 - t0:.3f}s")

# Benchmark
V_test = V.copy()
policy_action_test = policy_action.copy()
policy_bat_test = policy_bat.copy()
policy_ev_test = policy_ev.copy()
policy_appliance_test = policy_appliance.copy()

t0 = time.perf_counter()
_bellman_backward_pass(
    V_test, policy_action_test, policy_bat_test, policy_ev_test, policy_appliance_test,
    bat_soc_lookup, ev_trans_next, ev_trans_draw,
    action_ac_rate_idx, action_dc_charge, action_discharge, bat_charge_rates_arr,
    pv_array, load_array, price_array, revenue_array, ha_start_windows_arr,
    horizon, num_actions, horizon, has_ev, has_ha, worst_case,
    bat_min_wh, bat_max_wh, bat_charging_eff, bat_discharging_eff,
    ac_to_dc_eff, dc_to_ac_eff, max_ac_power,
)
t1 = time.perf_counter()
print(f"JIT-compiled run: {t1 - t0:.3f}s")
print("Numba JIT with real physics test successful!")
