"""Pure physics functions for energy simulation.

Stateless, vectorizable functions for SOC transitions and energy balance.
Used by all solver types (GA, DP, MPC, LP) for consistent physics.

Extracted from genetic_setup.py on branch feature/alternative-mpc-solver.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# State-transition helpers (pure physics, no mutable state)
# ---------------------------------------------------------------------------

def compute_battery_next_soc(
    current_soc_wh: float,
    *,
    # Battery parameters
    min_soc_wh: float,
    max_soc_wh: float,
    charging_efficiency: float,
    discharging_efficiency: float,
    # Actions
    ac_charge_factor: float,
    dc_charge_allowed: bool,
    discharge_allowed: bool,
    # Environment
    pv_wh: float,
    load_wh: float,
    # AC charging via inverter
    max_ac_charge_power_w: float,
    ac_to_dc_efficiency: float = 1.0,
    dc_to_ac_efficiency: float = 1.0,
) -> float:
    """Compute next battery SoC [Wh] after one time step.

    Pure function: does not mutate any object state.

    Args:
        current_soc_wh: Current battery SoC in Wh.
        min_soc_wh: Minimum allowed SoC in Wh.
        max_soc_wh: Maximum allowed SoC in Wh.
        charging_efficiency: DC charging efficiency (0-1).
        discharging_efficiency: DC discharging efficiency (0-1).
        ac_charge_factor: AC charge factor (0=off, 0-1=fraction of max power).
        dc_charge_allowed: Whether DC (PV) charging is allowed this step.
        discharge_allowed: Whether discharging is allowed this step.
        pv_wh: PV production this step in Wh.
        load_wh: Total load this step in Wh (including appliances).
        max_ac_charge_power_w: Maximum AC charge power from grid in W.
        ac_to_dc_efficiency: Inverter AC-to-DC efficiency for charging (0-1).
        dc_to_ac_efficiency: Inverter DC-to-AC efficiency for discharging (0-1).

    Returns:
        Next SoC in Wh (clipped to [min_soc_wh, max_soc_wh]).
    """
    bsv = current_soc_wh

    # --- 1) AC charging from grid ---
    bat_charge_wh = 0.0
    if ac_charge_factor > 0.0:
        ac_draw = max_ac_charge_power_w * ac_charge_factor
        dc_stored = ac_draw * ac_to_dc_efficiency * charging_efficiency
        headroom = max_soc_wh - bsv
        if headroom > 0:
            dc_stored = min(dc_stored, headroom)
        else:
            dc_stored = 0.0
        bat_charge_wh = dc_stored

    # --- 2) DC (PV) charging ---
    if pv_wh > 0.0 and dc_charge_allowed:
        headroom = max_soc_wh - bsv - bat_charge_wh
        if headroom > 0:
            pv_for_charge = min(pv_wh, headroom / charging_efficiency)
            bat_charge_wh += pv_for_charge * charging_efficiency

    # SOC after charging
    bat_soc_after = bsv + bat_charge_wh

    # --- 3) Discharge ---
    if discharge_allowed:
        pv_used_load = min(pv_wh, load_wh)
        remaining_load = load_wh - pv_used_load
        max_discharge = max(0.0, (bat_soc_after - min_soc_wh) / discharging_efficiency)
        if remaining_load > 0:
            dis_dc = min(max_discharge, remaining_load / dc_to_ac_efficiency)
            bat_soc_after -= dis_dc * discharging_efficiency

    # Clamp to valid range
    new_soc = max(min_soc_wh, min(max_soc_wh, bat_soc_after))
    return new_soc


def compute_ev_next_soc(
    current_soc_wh: float,
    *,
    min_soc_wh: float,
    max_soc_wh: float,
    charging_efficiency: float,
    charge_factor: float,
    max_charge_power_w: float,
) -> float:
    """Compute next EV SoC [Wh] after one time step.

    Pure function: does not mutate any object state.

    Args:
        current_soc_wh: Current EV SoC in Wh.
        min_soc_wh: Minimum allowed SoC in Wh.
        max_soc_wh: Maximum allowed SoC in Wh.
        charging_efficiency: EV charging efficiency (0-1).
        charge_factor: Charge factor (0=off, 0-1=fraction of max power).
        max_charge_power_w: Maximum EV charge power in W.

    Returns:
        Next SoC in Wh (clipped to [min_soc_wh, max_soc_wh]).
    """
    if charge_factor <= 0.0:
        return current_soc_wh

    evsv = current_soc_wh
    requested = max_charge_power_w * charge_factor
    headroom = max_soc_wh - evsv
    actual = min(requested, max(0.0, headroom / charging_efficiency))
    new_soc = evsv + actual * charging_efficiency
    return max(min_soc_wh, min(max_soc_wh, new_soc))
