"""Simulation result dataclass.

Provides a lightweight, solver-agnostic result container that mirrors
the fields of SimulationResultData without Pydantic dependencies.
Used internally by EnergySimulationEngine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SimulationResult:
    """Result of a full energy simulation run.

    Matches the field layout of SimulationResultData for drop-in
    compatibility. All arrays are length ``total_hours``.
    """

    # Per-hour arrays
    load_wh_per_hour: list[float] = field(default_factory=list)
    grid_feed_in_wh_per_hour: list[float] = field(default_factory=list)
    grid_consumption_wh_per_hour: list[float] = field(default_factory=list)
    costs_per_hour: list[float] = field(default_factory=list)
    revenue_per_hour: list[float] = field(default_factory=list)
    losses_per_hour: list[float] = field(default_factory=list)
    electricity_price: list[float] = field(default_factory=list)
    battery_soc_per_hour: list[float] = field(default_factory=list)
    ev_soc_per_hour: list[float] = field(default_factory=list)
    home_appliance_wh_per_hour: list[float] = field(default_factory=list)

    # Aggregates
    total_costs: float = 0.0
    total_revenue: float = 0.0
    total_balance: float = 0.0
    total_losses: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict matching SimulationResultData alias names.

        Returns a dictionary that can be passed directly to
        ``SimulationResultData(**result.to_dict())``.
        """
        return {
            "Last_Wh_pro_Stunde": self.load_wh_per_hour,
            "Netzeinspeisung_Wh_pro_Stunde": self.grid_feed_in_wh_per_hour,
            "Netzbezug_Wh_pro_Stunde": self.grid_consumption_wh_per_hour,
            "Kosten_Euro_pro_Stunde": self.costs_per_hour,
            "Einnahmen_Euro_pro_Stunde": self.revenue_per_hour,
            "Gesamtbilanz_Euro": self.total_balance,
            "EAuto_SoC_pro_Stunde": self.ev_soc_per_hour,
            "Gesamteinnahmen_Euro": self.total_revenue,
            "Gesamtkosten_Euro": self.total_costs,
            "Verluste_Pro_Stunde": self.losses_per_hour,
            "Gesamt_Verluste": self.total_losses,
            "akku_soc_pro_stunde": self.battery_soc_per_hour,
            "Home_appliance_wh_per_hour": self.home_appliance_wh_per_hour,
            "Electricity_price": self.electricity_price,
        }
