"""Single-step simulation result dataclass."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnergySimulationStep:
    """Immutable result of a single hourly simulation step.

    Attributes:
        consumption: Total consumption from grid + local [Wh].
        energy_feedin_grid: Energy fed to grid [Wh].
        energy_consumption_grid: Energy drawn from grid [Wh].
        losses: Total losses (EV + inverter + battery) [Wh].
        self_consumption: PV self-consumption [Wh].
        home_appliance_wh: Home appliance energy this hour [Wh].
        cost: Grid import cost [Euro].
        revenue: Grid export revenue [Euro].
        electricity_price: Current electricity price [Euro/Wh].
    """

    # Energy flows [Wh]
    consumption: float = 0.0
    energy_feedin_grid: float = 0.0
    energy_consumption_grid: float = 0.0
    losses: float = 0.0
    self_consumption: float = 0.0
    home_appliance_wh: float = 0.0

    # Financial [Euro]
    cost: float = 0.0
    revenue: float = 0.0
    electricity_price: float = 0.0
