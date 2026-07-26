"""Solver-agnostic penalty functions for energy optimization.

These pure functions compute penalty values for constraint violations
or economically suboptimal decisions. They are designed to be reused
by different optimizers (GA, DP, LP, MPC).
"""

from typing import Optional

import numpy as np


def battery_residual_value_penalty(
    battery_energy_content_wh: float,
    dc_to_ac_efficiency: float,
    price_per_wh_battery: float,
    initial_soc_wh: Optional[float] = None,
    electricity_prices: Optional[np.ndarray] = None,
    feed_in_tariffs: Optional[np.ndarray] = None,
) -> float:
    """Compute the penalty for net battery energy change at end of simulation.

    Values battery energy based on the average of grid price and PV feed-in tariff.
    Uses the LAST hour prices (end of horizon) since the penalty is applied at
    the terminal state. This captures the true opportunity cost at that moment:
    - Grid price: cost to buy replacement energy from grid at horizon end
    - Feed-in tariff: revenue lost from not selling stored energy at horizon end

    When battery energy changes relative to initial SOC:
    - If end_soc > initial_soc: penalty for energy invested (negative)
    - If end_soc < initial_soc: credit for stored energy used (positive)

    Args:
        battery_energy_content_wh: Current energy content of the battery in Wh.
        dc_to_ac_efficiency: DC-to-AC inverter efficiency (0.0 to 1.0).
        price_per_wh_battery: Price per Wh of battery energy (LCOS). Used as fallback.
        initial_soc_wh: Initial battery SOC in Wh.
        electricity_prices: Array of electricity prices per hour.
        feed_in_tariffs: Array of feed-in tariffs per hour.

    Returns:
        The penalty value to subtract from total balance (negative = cost).
        Returns positive value (credit) if battery energy decreased.
    """
    # Use prices at the LAST hour (end of horizon) to value battery energy
    if (
        electricity_prices is not None
        and len(electricity_prices) > 0
        and feed_in_tariffs is not None
        and len(feed_in_tariffs) > 0
    ):
        # Last hour prices represent the value at horizon end
        last_grid_price = float(electricity_prices[-1])
        last_feed_in = float(feed_in_tariffs[-1])
        avg_price = (last_grid_price + last_feed_in) / 2.0
    elif electricity_prices is not None and len(electricity_prices) > 0:
        last_grid_price = float(electricity_prices[-1])
        avg_price = last_grid_price
    else:
        avg_price = price_per_wh_battery

    if initial_soc_wh is not None:
        # Value net change relative to initial SOC
        net_change_wh = battery_energy_content_wh - initial_soc_wh
        adjusted_energy = net_change_wh * dc_to_ac_efficiency
    else:
        # Backward compatible: value full end SOC if initial not provided
        adjusted_energy = battery_energy_content_wh * dc_to_ac_efficiency

    # Negative penalty = cost (energy invested), positive = credit (energy used)
    return -(adjusted_energy * avg_price)


def ev_residual_value_penalty(
    ev_energy_content_wh: float,
    initial_soc_wh: Optional[float],
    electricity_prices: Optional[np.ndarray] = None,
    feed_in_tariffs: Optional[np.ndarray] = None,
    price_per_wh_fallback: float = 0.00025,
) -> float:
    """Compute the penalty for net EV energy change at end of simulation.

    Values EV energy based on the average of grid price and PV feed-in tariff
    at the last hour (end of horizon). This represents the opportunity cost:
    energy stored in EV could have been bought from grid or generated from PV.

    When EV energy changes relative to initial SOC:
    - If end_soc > initial_soc: penalty for energy invested (negative)
    - If end_soc < initial_soc: credit for stored energy used (positive)

    Args:
        ev_energy_content_wh: Current energy content of the EV battery in Wh.
        initial_soc_wh: Initial EV SOC in Wh.
        electricity_prices: Array of electricity prices per hour.
        feed_in_tariffs: Array of feed-in tariffs per hour.
        price_per_wh_fallback: Fallback price per Wh if no prices available.

    Returns:
        The penalty value to subtract from total balance (negative = cost).
        Returns positive value (credit) if EV energy decreased.
    """
    # Use prices at the LAST hour (end of horizon) to value EV energy
    if (
        electricity_prices is not None
        and len(electricity_prices) > 0
        and feed_in_tariffs is not None
        and len(feed_in_tariffs) > 0
    ):
        # Last hour prices represent the value at horizon end
        last_grid_price = float(electricity_prices[-1])
        last_feed_in = float(feed_in_tariffs[-1])
        avg_price = (last_grid_price + last_feed_in) / 2.0
    elif electricity_prices is not None and len(electricity_prices) > 0:
        last_grid_price = float(electricity_prices[-1])
        avg_price = last_grid_price
    else:
        avg_price = price_per_wh_fallback

    # Value net change relative to initial SOC
    if initial_soc_wh is not None:
        net_change_wh = ev_energy_content_wh - initial_soc_wh
    else:
        # Backward compatible: value full end SOC if initial not provided
        net_change_wh = ev_energy_content_wh

    # Negative penalty = cost (energy invested), positive = credit (energy used)
    return -(net_change_wh * avg_price)


def ev_soc_miss_penalty(
    ev_soc_percentage: float,
    min_soc_percentage: float,
    max_soc_percentage: float,
    penalty_factor: float,
) -> float:
    """Compute the penalty for EV SOC missing target range.

    If the EV state of charge is below min_soc_percentage at the end of the
    simulation, a penalty is applied proportional to the deviation.

    Additionally, SOC above min_soc_percentage is penalized as an opportunity
    cost (wasted charging energy) to avoid overcharging beyond the required minimum.

    Args:
        ev_soc_percentage: Current EV SOC as a percentage (0-100).
        min_soc_percentage: Minimum required EV SOC percentage.
        max_soc_percentage: Maximum allowed EV SOC percentage.
        penalty_factor: Penalty multiplier per percentage point of deviation.

    Returns:
        The penalty value (positive = additional cost). Returns 0 only if SOC
        equals min_soc_percentage exactly.
    """
    if ev_soc_percentage < min_soc_percentage:
        # Penalty for not reaching minimum SOC
        return abs(min_soc_percentage - ev_soc_percentage) * penalty_factor
    return 0.0


def ac_charge_break_even_penalty(
    ac_charge_hours: np.ndarray,
    electricity_prices: np.ndarray,
    load_wh_per_hour: np.ndarray,
    start_hour: int,
    initial_soc_wh: float,
    min_soc_wh: float,
    battery_charging_efficiency: float,
    battery_discharging_efficiency: float,
    inverter_dc_to_ac_efficiency: float,
    inverter_ac_to_dc_efficiency: float,
    battery_max_charge_power_w: float,
    ac_penalty_factor: float = 1.0,
) -> float:
    """Compute the penalty for economically unjustified AC charging.

    Penalizes AC charging decisions that cannot be economically justified
    given the round-trip losses and the best available future electricity
    prices. Energy already stored in the battery (from PV, zero grid cost)
    covers the most expensive future hours first.

    Args:
        ac_charge_hours: Array of AC charge factors per hour (0.0 to 1.0).
        electricity_prices: Array of electricity prices per hour (currency/Wh).
        load_wh_per_hour: Array of expected load per hour in Wh.
        start_hour: Simulation start hour index.
        initial_soc_wh: Initial battery SOC in Wh.
        min_soc_wh: Minimum battery SOC in Wh.
        battery_charging_efficiency: Battery charging efficiency (0.0 to 1.0).
        battery_discharging_efficiency: Battery discharging efficiency (0.0 to 1.0).
        inverter_dc_to_ac_efficiency: Inverter DC-to-AC efficiency (0.0 to 1.0).
        inverter_ac_to_dc_efficiency: Inverter AC-to-DC efficiency (0.0 to 1.0).
        battery_max_charge_power_w: Maximum battery charge power in W.
        ac_penalty_factor: Configurable penalty multiplier (default 1.0).

    Returns:
        The penalty value (positive = additional cost). Returns 0 if all
        AC charging is economically justified.
    """
    round_trip_eff = (
        inverter_ac_to_dc_efficiency
        * battery_charging_efficiency
        * battery_discharging_efficiency
        * inverter_dc_to_ac_efficiency
    )

    if round_trip_eff <= 0:
        return 0.0

    n = len(electricity_prices)

    # Usable AC energy already in battery from prior PV charging (zero grid cost).
    free_ac_wh = (
        max(0.0, initial_soc_wh - min_soc_wh)
        * battery_discharging_efficiency
        * inverter_dc_to_ac_efficiency
    )

    total_penalty = 0.0

    for hour in range(start_hour, min(len(ac_charge_hours), n)):
        ac_factor = ac_charge_hours[hour]
        if ac_factor <= 0.0:
            continue

        charge_price = electricity_prices[hour]
        if charge_price <= 0:
            continue

        # Price that a future discharge hour must reach to break even
        break_even_price = charge_price / round_trip_eff

        # Build list of (price, load_wh) for all future hours in the horizon
        future = [
            (float(electricity_prices[h]), float(load_wh_per_hour[h]))
            for h in range(hour + 1, n)
        ]
        # Sort descending by price so we "use" the most expensive hours first
        future.sort(key=lambda x: -x[0])

        # Consume free PV energy against the highest-price future hours.
        remaining_free = free_ac_wh
        best_uncovered_price = 0.0
        for fp, fl in future:
            if remaining_free >= fl:
                remaining_free -= fl
            else:
                # First hour not (fully) covered: this is where new charge goes
                best_uncovered_price = fp
                break

        if best_uncovered_price < break_even_price:
            # AC charging at this hour is economically unjustified.
            dc_wh = battery_max_charge_power_w * ac_factor
            ac_wh = dc_wh / max(inverter_ac_to_dc_efficiency, 1e-9)
            excess_cost_per_wh = break_even_price - best_uncovered_price
            total_penalty += ac_wh * excess_cost_per_wh * ac_penalty_factor

    return total_penalty
