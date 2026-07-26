"""Simulation context dataclass for multi-phase simulation.

Provides shared state carried through simulation initialization,
hourly stepping, and finalization phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from akkudoktoreos.optimization.simulation.devices import (
        Battery,
        HomeAppliance,
        Inverter,
    )


@dataclass
class SimulationContext:
    """Mutable state carried through the multi-phase simulation.

    Holds bounds, action arrays, forecast data, device references,
    and pre-allocated result storage for a single simulation run.

    Used by both EnergySimulationEngine and legacy GeneticSimulation methods.
    """

    # Bounds
    start_hour: int = 0
    end_hour: int = 0
    total_hours: int = 0

    # Action arrays (length = prediction_hours)
    ac_charge_hours: Optional[np.ndarray] = None
    dc_charge_hours: Optional[np.ndarray] = None
    bat_discharge_hours: Optional[np.ndarray] = None
    ev_charge_hours: Optional[np.ndarray] = None
    ev_discharge_hours: Optional[np.ndarray] = None

    # Forecast arrays
    load_energy_array: Optional[np.ndarray] = None
    elect_price_hourly: Optional[np.ndarray] = None
    elect_revenue_per_hour: Optional[np.ndarray] = None
    pv_prediction_wh: Optional[np.ndarray] = None

    # Devices
    battery: Optional[Battery] = None
    ev: Optional[Battery] = None
    home_appliance: Optional[HomeAppliance] = None
    inverter: Optional[Inverter] = None

    # Inverter-derived scalars
    ac_to_dc_eff: float = 1.0
    dc_to_ac_eff: float = 1.0
    max_ac_charge_w: Optional[float] = None
    ac_charging_possible: bool = False

    # Battery LCOS / degradation cost
    price_per_wh_battery: float = 0.0

    # Home appliance
    home_appliance_enabled: bool = False

    # Pre-allocated result arrays (length = total_hours)
    loads_energy: list[float] = field(default_factory=list)
    feedin_energy: list[float] = field(default_factory=list)
    consumption_energy: list[float] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)
    revenue: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    electricity_price: list[float] = field(default_factory=list)
    soc_battery: list[float] = field(default_factory=list)
    soc_ev: list[float] = field(default_factory=list)
    home_appliance_wh: list[float] = field(default_factory=list)
