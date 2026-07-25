"""GA-specific encoding/decoding functions for genetic algorithm.

This module contains pure functions for encoding and decoding GA individuals,
as well utilities for splitting and merging individual components.
These functions are solver-specific (DEAP GA) and separated from the
simulation engine to allow reuse across different optimizers.
"""

from typing import Optional

import numpy as np


def decode_charge_discharge(
    discharge_hours_bin: np.ndarray,
    bat_possible_charge_values: list[float],
    optimize_dc_charge: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode the input array into ac_charge, dc_charge, and discharge arrays.

    The encoding scheme uses integer state codes per hour:
    - Idle:       0 .. len_bat-1
    - Discharge:  len_bat .. 2*len_bat - 1
    - AC Charge:  2*len_bat .. 3*len_bat - 1  (maps to bat_possible_charge_values)
    - DC optional: 3*len_bat (not allowed), 3*len_bat + 1 (allowed)

    Args:
        discharge_hours_bin: Array of integer state codes (one per prediction hour).
        bat_possible_charge_values: List of allowed AC charge rates for the battery.
        optimize_dc_charge: If True, DC charge is encoded in state codes.
            If False, DC charge is always enabled (all ones).

    Returns:
        Tuple of (ac_charge, dc_charge, discharge) arrays.
        - ac_charge: Float array with charge rate factors per hour.
        - dc_charge: Float array (0 or 1) indicating DC charge enabled per hour.
        - discharge: Int array (0 or 1) indicating discharge enabled per hour.
    """
    discharge_hours_bin_np = np.array(discharge_hours_bin)
    len_bat = len(bat_possible_charge_values)

    # Idle states
    idle_mask = (discharge_hours_bin_np >= 0) & (discharge_hours_bin_np < len_bat)

    # Discharge states
    discharge_mask = (discharge_hours_bin_np >= len_bat) & (
        discharge_hours_bin_np < 2 * len_bat
    )

    # AC states
    ac_mask = (discharge_hours_bin_np >= 2 * len_bat) & (discharge_hours_bin_np < 3 * len_bat)
    ac_indices = (discharge_hours_bin_np[ac_mask] - 2 * len_bat).astype(int)

    # DC states (if enabled)
    if optimize_dc_charge:
        dc_not_allowed_state = 3 * len_bat
        dc_allowed_state = 3 * len_bat + 1
        dc_charge = np.where(discharge_hours_bin_np == dc_allowed_state, 1, 0)
    else:
        dc_charge = np.ones_like(discharge_hours_bin_np, dtype=float)

    # Generate the result arrays
    discharge = np.zeros_like(discharge_hours_bin_np, dtype=int)
    discharge[discharge_mask] = 1  # Set Discharge states to 1

    ac_charge = np.zeros_like(discharge_hours_bin_np, dtype=float)
    ac_charge[ac_mask] = [bat_possible_charge_values[i] for i in ac_indices]

    # Idle is just 0, already default.

    return ac_charge, dc_charge, discharge


def split_individual(
    individual: list[int],
    prediction_hours: int,
    optimize_ev: bool,
    optimize_home_appliance: bool,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[int]]:
    """Split the individual solution into its components.

    Components:
    1. Discharge hours (binary as int NumPy array),
    2. Electric vehicle charge hours (float as int NumPy array, if applicable),
    3. Dishwasher start time (integer if applicable).

    Args:
        individual: The full GA individual as a list of integers.
        prediction_hours: Number of prediction hours (length of each segment).
        optimize_ev: If True, EV charge indices are included in the individual.
        optimize_home_appliance: If True, home appliance start time is the last element.

    Returns:
        Tuple of (discharge_hours_bin, ev_charge_hours_index, washingstart_int).
    """
    # Discharge hours as a NumPy array of ints
    discharge_hours_bin = np.array(individual[:prediction_hours], dtype=int)

    # EV charge hours as a NumPy array of ints (if optimize_ev is True)
    ev_charge_hours_index: Optional[np.ndarray] = (
        np.array(
            individual[prediction_hours : prediction_hours * 2],
            dtype=int,
        )
        if optimize_ev
        else None
    )

    # Washing machine start time as an integer (if applicable)
    washingstart_int: Optional[int] = (
        int(individual[-1])
        if optimize_home_appliance
        else None
    )

    return discharge_hours_bin, ev_charge_hours_index, washingstart_int


def merge_individual(
    discharge_hours_bin: np.ndarray,
    ev_charge_hours_index: Optional[np.ndarray],
    washingstart_int: Optional[int],
    prediction_hours: int,
    optimize_ev: bool,
    optimize_home_appliance: bool,
) -> list[int]:
    """Merge the individual components back into a single solution list.

    Args:
        discharge_hours_bin: Binary discharge hours array.
        ev_charge_hours_index: EV charge hours as integers, or None.
        washingstart_int: Dishwasher start time as integer, or None.
        prediction_hours: Number of prediction hours.
        optimize_ev: If True, EV charge indices are included.
        optimize_home_appliance: If True, home appliance start time is appended.

    Returns:
        The merged individual solution as a list of integers.
    """
    # Start with the discharge hours
    individual = discharge_hours_bin.tolist()

    # Add EV charge hours if applicable
    if optimize_ev and ev_charge_hours_index is not None:
        individual.extend(ev_charge_hours_index.tolist())
    elif optimize_ev:
        # If optimize_ev is active but no EV data is available, append zeros
        individual.extend([0] * prediction_hours)

    # Add dishwasher start time if applicable
    if optimize_home_appliance and washingstart_int is not None:
        individual.append(washingstart_int)
    elif optimize_home_appliance:
        # If a home appliance is optimized but no start time is available
        individual.append(0)

    return individual


def compute_total_states(
    bat_possible_charge_values: list[float],
    optimize_dc_charge: bool,
) -> int:
    """Compute the total number of battery/discharge states.

    State layout:
    - Idle:      len(bat_possible_charge_values) states
    - Discharge: len(bat_possible_charge_values) states
    - AC-Charge: len(bat_possible_charge_values) states
    - With DC: + 2 additional states (DC not allowed + DC allowed)

    Args:
        bat_possible_charge_values: List of allowed AC charge rates.
        optimize_dc_charge: If True, includes 2 extra DC states.

    Returns:
        Total number of states for the battery/discharge encoding.
    """
    len_bat = len(bat_possible_charge_values)
    if optimize_dc_charge:
        return 3 * len_bat + 2
    return 3 * len_bat
