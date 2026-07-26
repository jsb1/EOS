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


from dataclasses import dataclass


@dataclass(frozen=True)
class EnergyFlows:
    """Energy flows for one time step.

    Returned by compute_battery_next_soc_with_flows() to enable
    consistent cost calculation in DP optimizer.
    """

    next_soc_wh: float
    pv_used_for_load: float
    pv_used_for_dc_charge: float
    battery_discharge_ac: float
    grid_import: float
    grid_export: float
    losses: float


def _self_consumption_ratio(consumption: float, generation: float) -> float:
    """Calculate self-consumption ratio (SCR) for PV energy.

    Approximates the SCR from self_consumption_predictor.calculate_self_consumption().
    Higher surplus → lower SCR (more PV wasted).
    """
    if generation <= 0 or consumption <= 0:
        return 1.0 if generation > 0 else 0.0
    ratio = consumption / generation
    # SCR drops as surplus increases
    # Approx: SCR = 1 / (1 + 0.5 * (1-ratio)) when generation > consumption
    if ratio >= 1.0:
        return 1.0
    return 1.0 / (1.0 + 0.5 * (1.0 - ratio))


def compute_battery_next_soc_with_flows(
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
    ev_draw_wh: float,
    # AC charging via inverter
    max_ac_charge_power_w: float,
    ac_to_dc_efficiency: float = 1.0,
    dc_to_ac_efficiency: float = 1.0,
) -> EnergyFlows:
    """Compute next battery SoC with detailed energy flows.

    Mirrors Inverter.process_energy() logic for consistent cost calculation.

    Energy flow priority (matches simulation engine):
    1. PV → Load (self-consumption, with SCR factor)
    2. If PV > Load: Battery discharge → remaining load (due to SCR imperfection)
    3. If PV > Load: PV → Battery DC charge (with SCR factor)
    4. If PV < Load: Battery discharge → shortfall
    5. AC charge from grid (always grid-import)
    6. EV charge from grid

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
        load_wh: Total household load this step in Wh.
        ev_draw_wh: EV charging draw from grid this step in Wh.
        max_ac_charge_power_w: Maximum AC charge power from grid in W.
        ac_to_dc_efficiency: Inverter AC-to-DC efficiency for charging (0-1).
        dc_to_ac_efficiency: Inverter DC-to-AC efficiency for discharging (0-1).

    Returns:
        EnergyFlows with next_soc_wh and all energy flow components.
    """
    bsv = current_soc_wh

    # Track energy flows
    pv_used_for_load = 0.0
    pv_used_for_dc_charge = 0.0
    battery_discharge_ac = 0.0
    total_losses = 0.0
    grid_import = 0.0
    grid_export = 0.0

    # Total AC consumption: load + EV charging
    total_consumption = load_wh + ev_draw_wh

    # --- AC charging from grid (always draws from grid) ---
    ac_draw_grid = 0.0
    bat_charge_wh = 0.0
    if ac_charge_factor > 0.0:
        ac_draw = max_ac_charge_power_w * ac_charge_factor
        dc_stored = ac_draw * ac_to_dc_efficiency * charging_efficiency
        headroom = max_soc_wh - bsv
        if headroom > 0:
            dc_stored = min(dc_stored, headroom)
            ac_draw_grid = ac_draw
        else:
            dc_stored = 0.0
            ac_draw_grid = 0.0
        bat_charge_wh = dc_stored
        total_losses += ac_draw - dc_stored if ac_draw > dc_stored else 0.0

    # Add AC charge to total consumption
    total_consumption += ac_draw_grid

    # --- PV processing (mirrors Inverter.process_energy) ---
    if pv_wh >= total_consumption:
        # Case 1: PV surplus
        pv_used_for_load = min(total_consumption, pv_wh)

        # Self-consumption ratio for surplus handling
        scr = _self_consumption_ratio(total_consumption, pv_wh)
        remaining_power = (pv_wh - total_consumption) * scr
        remaining_load_evq = (pv_wh - total_consumption) * (1.0 - scr)

        if remaining_load_evq > 0 and discharge_allowed:
            # Battery must cover remaining consumption due to SCR imperfection
            # Request DC from battery accounting for both battery discharge and inverter DC→AC loss
            ac_needed = remaining_load_evq
            dc_request = ac_needed / dc_to_ac_efficiency / discharging_efficiency
            max_discharge_dc = max(0.0, (bsv - min_soc_wh))
            raw_used_dc = min(max_discharge_dc, dc_request)
            delivered_dc = raw_used_dc * discharging_efficiency
            battery_discharge_ac = delivered_dc * dc_to_ac_efficiency
            # Losses: battery discharge loss + inverter DC→AC loss
            total_losses += (raw_used_dc - delivered_dc) + (delivered_dc - battery_discharge_ac)
            bsv -= raw_used_dc
            remaining_load_evq -= battery_discharge_ac

        # If battery couldn't cover, draw from grid
        if remaining_load_evq > 0:
            grid_import += remaining_load_evq

        # DC (PV) charging with surplus
        if remaining_power > 0 and dc_charge_allowed and bsv < max_soc_wh:
            headroom = max_soc_wh - bsv - bat_charge_wh
            if headroom > 0:
                pv_for_charge = min(remaining_power, headroom / charging_efficiency)
                stored = pv_for_charge * charging_efficiency
                bat_charge_wh += stored
                pv_used_for_dc_charge = pv_for_charge
                total_losses += pv_for_charge - stored
                remaining_power -= pv_for_charge
                bsv += stored

        # Grid export: remaining PV after charge
        grid_export = max(0.0, remaining_power)
    else:
        # Case 2: PV deficit
        pv_used_for_load = pv_wh
        shortfall = total_consumption - pv_wh

        # Battery discharge to cover shortfall
        if discharge_allowed and shortfall > 0:
            # Request DC from battery accounting for both battery discharge and inverter DC→AC loss
            ac_needed = shortfall
            dc_request = ac_needed / dc_to_ac_efficiency / discharging_efficiency
            max_discharge_dc = max(0.0, (bsv - min_soc_wh))
            raw_used_dc = min(max_discharge_dc, dc_request)
            delivered_dc = raw_used_dc * discharging_efficiency
            battery_discharge_ac = delivered_dc * dc_to_ac_efficiency
            # Losses: battery discharge loss + inverter DC→AC loss
            total_losses += (raw_used_dc - delivered_dc) + (delivered_dc - battery_discharge_ac)
            bsv -= raw_used_dc
            shortfall -= battery_discharge_ac

        # Grid import: remaining shortfall
        grid_import += max(0.0, shortfall)

    # Apply DC charge
    bsv += bat_charge_wh

    # Clamp to valid range
    next_soc = max(min_soc_wh, min(max_soc_wh, bsv))

    return EnergyFlows(
        next_soc_wh=next_soc,
        pv_used_for_load=pv_used_for_load,
        pv_used_for_dc_charge=pv_used_for_dc_charge,
        battery_discharge_ac=battery_discharge_ac,
        grid_import=grid_import,
        grid_export=grid_export,
        losses=total_losses,
    )
