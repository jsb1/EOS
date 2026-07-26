"""Solver-agnostic simulation session class.

Provides a Pydantic data container with methods for preparing, resetting,
and running energy simulations. Designed to be reused by any optimization
solver (GA, DP, MPC, LP) without duplication.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from loguru import logger
from numpydantic import NDArray, Shape
from pydantic import ConfigDict, Field

from akkudoktoreos.core.pydantic import PydanticBaseModel
from akkudoktoreos.optimization.simulation.context import SimulationContext
from akkudoktoreos.optimization.simulation.devices import (
    Battery,
    HomeAppliance,
    Inverter,
)
from akkudoktoreos.optimization.simulation.step import EnergySimulationStep

if TYPE_CHECKING:
    from akkudoktoreos.optimization.simulation.parameters import (
        EnergyManagementParameters,
    )


class SimulationSession(PydanticBaseModel):
    """Solver-agnostic simulation session.

    Data container holding forecast arrays, device references, and action
    arrays, together with helper methods to prepare, reset, and simulate.
    The actual simulation delegates to [EnergySimulationEngine](akkudoktoreos.optimization.simulation.engine.EnergySimulationEngine).

    Subclasses may override ``prepare`` to accept solver-specific parameter
    types, but the core logic is identical for all solvers.
    """

    # Disable validation on assignment to speed up simulation runs.
    model_config = ConfigDict(
        validate_assignment=False,
    )

    start_hour: int = Field(
        default=0,
        ge=0,
        le=23,
        json_schema_extra={"description": "Starting hour on day for optimizations."},
    )

    optimization_hours: Optional[int] = Field(
        default=24,
        ge=0,
        json_schema_extra={"description": "Number of hours into the future for optimizations."},
    )

    prediction_hours: Optional[int] = Field(
        default=48,
        ge=0,
        json_schema_extra={"description": "Number of hours into the future for predictions"},
    )

    load_energy_array: Optional[NDArray[Shape["*"], float]] = Field(
        default=None,
        json_schema_extra={
            "description": "An array of floats representing the total load (consumption) in watts for different time intervals."
        },
    )
    pv_prediction_wh: Optional[NDArray[Shape["*"], float]] = Field(
        default=None,
        json_schema_extra={
            "description": "An array of floats representing the forecasted photovoltaic output in watts for different time intervals."
        },
    )
    elect_price_hourly: Optional[NDArray[Shape["*"], float]] = Field(
        default=None,
        json_schema_extra={
            "description": "An array of floats representing the electricity price per watt-hour for different time intervals."
        },
    )
    elect_revenue_per_hour_arr: Optional[NDArray[Shape["*"], float]] = Field(
        default=None,
        json_schema_extra={
            "description": "An array of floats representing the feed-in compensation per watt-hour."
        },
    )

    battery: Optional[Battery] = Field(default=None, json_schema_extra={"description": "TBD."})
    ev: Optional[Battery] = Field(default=None, json_schema_extra={"description": "TBD."})
    home_appliance: Optional[HomeAppliance] = Field(
        default=None, json_schema_extra={"description": "TBD."}
    )
    inverter: Optional[Inverter] = Field(default=None, json_schema_extra={"description": "TBD."})
    price_per_wh_battery: float = Field(
        default=0.0,
        ge=0,
        json_schema_extra={
            "description": "LCOS cost per Wh charged into the battery (battery degradation cost)."
        },
    )

    ac_charge_hours: Optional[NDArray[Shape["*"], float]] = Field(
        default=None, json_schema_extra={"description": "TBD"}
    )
    dc_charge_hours: Optional[NDArray[Shape["*"], float]] = Field(
        default=None, json_schema_extra={"description": "TBD"}
    )
    bat_discharge_hours: Optional[NDArray[Shape["*"], float]] = Field(
        default=None, json_schema_extra={"description": "TBD"}
    )
    ev_charge_hours: Optional[NDArray[Shape["*"], float]] = Field(
        default=None, json_schema_extra={"description": "TBD"}
    )
    ev_discharge_hours: Optional[NDArray[Shape["*"], float]] = Field(
        default=None, json_schema_extra={"description": "TBD"}
    )
    home_appliance_start_hour: Optional[int] = Field(
        default=None,
        json_schema_extra={"description": "Home appliance start hour - None denotes no start."},
    )

    # ── Public API ────────────────────────────────────────────────────
    def prepare(
        self,
        parameters: "EnergyManagementParameters",
        optimization_hours: int,
        prediction_hours: int,
        ev: Optional[Battery] = None,
        home_appliance: Optional[HomeAppliance] = None,
        inverter: Optional[Inverter] = None,
    ) -> None:
        """Prepare simulation runs.

        Populate internal arrays and device references used during simulation.

        Args:
            parameters: Energy management parameters (EMS forecasts).
            optimization_hours: Number of optimization hours.
            prediction_hours: Number of prediction hours.
            ev: Electric vehicle battery device.
            home_appliance: Home appliance device.
            inverter: Inverter device.
        """
        self.optimization_hours = optimization_hours
        self.prediction_hours = prediction_hours

        # Load arrays from provided EMS parameters
        self.load_energy_array = np.array(parameters.total_load, float)
        self.pv_prediction_wh = np.array(parameters.pv_forecast_wh, float)
        self.elect_price_hourly = np.array(parameters.electricity_price_per_wh, float)
        self.elect_revenue_per_hour_arr = (
            parameters.feed_in_tariff_per_wh
            if isinstance(parameters.feed_in_tariff_per_wh, list)
            else np.full(len(self.load_energy_array), parameters.feed_in_tariff_per_wh, float)
        )

        # Associate devices
        if inverter:
            self.battery = inverter.battery
        else:
            self.battery = None
        self.ev = ev
        self.home_appliance = home_appliance
        self.inverter = inverter
        self.price_per_wh_battery = parameters.price_per_wh_battery

        # Initialize per-hour action arrays for the prediction horizon
        self.ac_charge_hours = np.full(self.prediction_hours, 0.0)
        self.dc_charge_hours = np.full(self.prediction_hours, 0.0)
        self.bat_discharge_hours = np.full(self.prediction_hours, 0.0)
        self.ev_charge_hours = np.full(self.prediction_hours, 0.0)
        self.ev_discharge_hours = np.full(self.prediction_hours, 0.0)
        self.home_appliance_start_hour = None

    def reset(self) -> None:
        """Reset device states and home appliance scheduling."""
        if self.ev:
            self.ev.reset()
        if self.battery:
            self.battery.reset()
        self.home_appliance_start_hour = None

    def simulate(self, start_hour: int) -> dict[str, Any]:
        """Simulate energy usage and costs for the given start hour.

        Delegates to [EnergySimulationEngine](akkudoktoreos.optimization.simulation.engine.EnergySimulationEngine),
        the solver-agnostic simulation engine.

        battery_soc_per_hour begin of the hour, initial hour state!
        load_wh_per_hour integral of last hour (end state)

        Args:
            start_hour: Simulation start hour (0-23).

        Returns:
            Dictionary matching the SimulationResultData schema.
        """
        from akkudoktoreos.optimization.simulation.devices import SimulationDevices
        from akkudoktoreos.optimization.simulation.engine import (
            EnergySimulationEngine,
            SimulationConfig,
        )

        prediction_h = self.prediction_hours if self.prediction_hours is not None else 48

        # Build SimulationConfig from session fields
        sim_config = SimulationConfig(
            prediction_hours=prediction_h,
            optimization_hours=self.optimization_hours if self.optimization_hours is not None else 24,
            start_hour=start_hour,
            load_energy_array=self.load_energy_array,
            pv_prediction_wh=self.pv_prediction_wh,
            elect_price_hourly=self.elect_price_hourly,
            elect_revenue_per_hour=self.elect_revenue_per_hour_arr,
        )

        # Build SimulationDevices from session device references
        sim_devices = SimulationDevices(
            battery=self.battery,
            ev=self.ev,
            inverter=self.inverter,
            home_appliance=self.home_appliance,
        )

        # Create engine and run
        engine = EnergySimulationEngine(devices=sim_devices, sim_config=sim_config)

        result = engine.run(
            ac_charge=self.ac_charge_hours if self.ac_charge_hours is not None else np.full(prediction_h, 0.0),
            dc_charge=self.dc_charge_hours if self.dc_charge_hours is not None else np.full(prediction_h, 0.0),
            discharge=self.bat_discharge_hours if self.bat_discharge_hours is not None else np.full(prediction_h, 0.0),
            ev_charge=self.ev_charge_hours,
            home_appliance_start=self.home_appliance_start_hour,
        )

        # Update home_appliance_start_hour from engine (may be adjusted by set_starting_time)
        if engine._home_appliance_start_hour is not None:
            self.home_appliance_start_hour = engine._home_appliance_start_hour

        return result.to_dict()

    # ── Legacy three-phase simulation ─────────────────────────────────
    def _simulate_init(self, start_hour: int) -> SimulationContext:
        """Initialize simulation context with validation and device state.

        This is the first phase of the three-phase simulation. It validates input
        arrays, configures device charge/discharge arrays, and pre-allocates result
        storage. The returned context carries all mutable state through the hourly
        steps and finalization.

        Args:
            start_hour: Simulation start hour (0-23).

        Returns:
            Fully initialized SimulationContext ready for hourly stepping.
        """
        self.start_hour = start_hour
        ctx = SimulationContext(start_hour=start_hour)

        # Fast-local references
        ctx.load_energy_array = self.load_energy_array
        ctx.ev_charge_hours = self.ev_charge_hours
        ctx.ev_discharge_hours = self.ev_discharge_hours
        ctx.ac_charge_hours = self.ac_charge_hours
        ctx.dc_charge_hours = self.dc_charge_hours
        ctx.bat_discharge_hours = self.bat_discharge_hours
        ctx.elect_price_hourly = self.elect_price_hourly
        ctx.elect_revenue_per_hour = self.elect_revenue_per_hour_arr
        ctx.pv_prediction_wh = self.pv_prediction_wh
        ctx.battery = self.battery
        ctx.ev = self.ev
        ctx.home_appliance = self.home_appliance
        ctx.inverter = self.inverter

        # --- Validation ---
        if (
            ctx.load_energy_array is None
            or ctx.pv_prediction_wh is None
            or ctx.elect_price_hourly is None
            or ctx.ev_charge_hours is None
            or ctx.ac_charge_hours is None
            or ctx.dc_charge_hours is None
            or ctx.elect_revenue_per_hour is None
            or ctx.bat_discharge_hours is None
            or ctx.ev_discharge_hours is None
        ):
            missing = []
            if ctx.load_energy_array is None:
                missing.append("Load Energy Array")
            if ctx.pv_prediction_wh is None:
                missing.append("PV Prediction Wh")
            if ctx.elect_price_hourly is None:
                missing.append("Electricity Price Hourly")
            if ctx.ev_charge_hours is None:
                missing.append("EV Charge Hours")
            if ctx.ac_charge_hours is None:
                missing.append("AC Charge Hours")
            if ctx.dc_charge_hours is None:
                missing.append("DC Charge Hours")
            if ctx.elect_revenue_per_hour is None:
                missing.append("Electricity Revenue Per Hour")
            if ctx.bat_discharge_hours is None:
                missing.append("Battery Discharge Hours")
            if ctx.ev_discharge_hours is None:
                missing.append("EV Discharge Hours")
            msg = ", ".join(missing)
            logger.error("Mandatory data missing - %s", msg)
            raise ValueError(f"Mandatory data missing: {msg}")

        if not (
            len(ctx.load_energy_array)
            == len(ctx.pv_prediction_wh)
            == len(ctx.elect_price_hourly)
        ):
            error_msg = (
                f"Array sizes do not match: Load Curve = {len(ctx.load_energy_array)}, "
                f"PV Forecast = {len(ctx.pv_prediction_wh)}, "
                f"Electricity Price = {len(ctx.elect_price_hourly)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        ctx.end_hour = len(ctx.load_energy_array)
        ctx.total_hours = ctx.end_hour - start_hour

        # --- Battery initialization ---
        if ctx.battery:
            ctx.soc_battery = [0.0] * ctx.total_hours
            ctx.soc_battery[0] = ctx.battery.current_soc_percentage()

            if ctx.inverter:
                ctx.ac_to_dc_eff = ctx.inverter.ac_to_dc_efficiency
                ctx.dc_to_ac_eff = ctx.inverter.dc_to_ac_efficiency
                ctx.max_ac_charge_w = ctx.inverter.max_ac_charge_power_w
            else:
                ctx.ac_to_dc_eff = 1.0
                ctx.dc_to_ac_eff = 1.0
                ctx.max_ac_charge_w = None

            ctx.ac_charging_possible = ctx.ac_to_dc_eff > 0 and (
                ctx.max_ac_charge_w is None or ctx.max_ac_charge_w > 0
            )

            if not ctx.ac_charging_possible:
                ctx.ac_charge_hours = np.zeros_like(ctx.ac_charge_hours)

            # Zero out charge arrays outside simulation window
            ctx.dc_charge_hours[0:start_hour] = 0
            ctx.dc_charge_hours[ctx.end_hour:] = 0
            ctx.ac_charge_hours[0:start_hour] = 0
            ctx.ac_charge_hours[ctx.end_hour:] = 0
            ctx.battery.charge_array = np.where(
                ctx.ac_charge_hours != 0, ctx.ac_charge_hours, ctx.dc_charge_hours
            )

            ctx.bat_discharge_hours[0:start_hour] = 0
            ctx.bat_discharge_hours[ctx.end_hour:] = 0
            ctx.battery.discharge_array = ctx.bat_discharge_hours
        else:
            ctx.soc_battery = [0.0] * ctx.total_hours
            ctx.ac_to_dc_eff = 1.0
            ctx.dc_to_ac_eff = 1.0
            ctx.max_ac_charge_w = None
            ctx.ac_charging_possible = False

        # --- EV initialization ---
        if ctx.ev:
            ctx.soc_ev = [0.0] * ctx.total_hours
            ctx.soc_ev[0] = ctx.ev.current_soc_percentage()

            ctx.ev_charge_hours[0:start_hour] = 0
            ctx.ev_charge_hours[ctx.end_hour:] = 0
            ctx.ev.charge_array = ctx.ev_charge_hours

            ctx.ev_discharge_hours[0:start_hour] = 0
            ctx.ev_discharge_hours[ctx.end_hour:] = 0
            ctx.ev.discharge_array = ctx.ev_discharge_hours
        else:
            ctx.soc_ev = [0.0] * ctx.total_hours

        # --- Home appliance initialization ---
        if ctx.home_appliance and self.home_appliance_start_hour is not None:
            ctx.home_appliance_enabled = True
            self.home_appliance_start_hour = ctx.home_appliance.set_starting_time(
                self.home_appliance_start_hour, start_hour
            )
        else:
            ctx.home_appliance_enabled = False

        # Pre-allocate result lists
        ctx.loads_energy = [0.0] * ctx.total_hours
        ctx.feedin_energy = [0.0] * ctx.total_hours
        ctx.consumption_energy = [0.0] * ctx.total_hours
        ctx.costs = [0.0] * ctx.total_hours
        ctx.revenue = [0.0] * ctx.total_hours
        ctx.losses = [0.0] * ctx.total_hours
        ctx.electricity_price = [0.0] * ctx.total_hours
        ctx.home_appliance_wh = [0.0] * ctx.total_hours

        return ctx

    def _simulate_hourly_step(self, hour: int, ctx: SimulationContext) -> EnergySimulationStep:
        """Process a single simulation hour.

        This is the core simulation logic that runs for each hour in the
        simulation window. It handles load consumption, home appliance scheduling,
        EV charging, battery SOC tracking, inverter energy processing, AC battery
        charging, and financial calculations.

        Args:
            hour: Absolute hour index (0-based from prediction start).
            ctx: Simulation context carrying device references and state.

        Returns:
            EnergySimulationStep with all values computed for this hour.
        """
        assert ctx.load_energy_array is not None
        assert ctx.ev_charge_hours is not None
        assert ctx.ac_charge_hours is not None
        assert ctx.elect_price_hourly is not None
        assert ctx.elect_revenue_per_hour is not None
        assert ctx.pv_prediction_wh is not None

        hour_idx = hour - ctx.start_hour
        result = EnergySimulationStep()

        # --- Base consumption from load ---
        consumption = ctx.load_energy_array[hour]

        # --- Home appliance load ---
        if ctx.home_appliance_enabled and ctx.home_appliance is not None:
            ha_load = ctx.home_appliance.get_load_for_hour(hour)
            consumption += ha_load
            result.home_appliance_wh = ha_load

        # --- EV charging ---
        if ctx.ev is not None:
            ctx.soc_ev[hour_idx] = ctx.ev.current_soc_percentage()
            if ctx.ev_charge_hours[hour] > 0:
                loaded_energy_ev, ev_charge_losses = ctx.ev.charge_energy(
                    wh=None, hour=hour, charge_factor=ctx.ev_charge_hours[hour]
                )
                consumption += loaded_energy_ev
                result.losses += ev_charge_losses

        # --- Record battery SOC (begin-of-interval state) ---
        if ctx.battery is not None:
            ctx.soc_battery[hour_idx] = ctx.battery.current_soc_percentage()

        # --- Inverter energy processing (PV -> grid/consumption) ---
        energy_feedin_grid_actual = 0.0
        energy_consumption_grid_actual = 0.0

        if ctx.inverter is not None:
            energy_produced = ctx.pv_prediction_wh[hour]
            (
                energy_feedin_grid_actual,
                energy_consumption_grid_actual,
                inv_losses,
                result.self_consumption,
            ) = ctx.inverter.process_energy(energy_produced, consumption, hour)
            result.losses += inv_losses

        # --- AC PV battery charging ---
        if ctx.battery is not None:
            hour_ac_charge = ctx.ac_charge_hours[hour]
            if hour_ac_charge > 0.0 and ctx.ac_charging_possible:
                effective_charge_factor = hour_ac_charge
                if ctx.max_ac_charge_w is not None and ctx.battery.max_charge_power_w > 0:
                    max_dc_factor = (
                        ctx.max_ac_charge_w * ctx.ac_to_dc_eff
                    ) / ctx.battery.max_charge_power_w
                    effective_charge_factor = min(effective_charge_factor, max_dc_factor)

                if effective_charge_factor > 0:
                    battery_charged_energy_actual, battery_losses_actual = (
                        ctx.battery.charge_energy(
                            None, hour, charge_factor=effective_charge_factor
                        )
                    )

                    dc_energy = battery_charged_energy_actual + battery_losses_actual
                    ac_energy = dc_energy / ctx.ac_to_dc_eff
                    inverter_charge_losses = ac_energy - dc_energy

                    consumption += ac_energy
                    energy_consumption_grid_actual += ac_energy
                    result.losses += battery_losses_actual + inverter_charge_losses

        # --- Financial calculations ---
        result.consumption = consumption
        result.energy_feedin_grid = energy_feedin_grid_actual
        result.energy_consumption_grid = energy_consumption_grid_actual
        result.electricity_price = ctx.elect_price_hourly[hour]
        result.cost = energy_consumption_grid_actual * ctx.elect_price_hourly[hour]
        result.revenue = energy_feedin_grid_actual * ctx.elect_revenue_per_hour[hour]

        return result

    def _simulate_finalize(self, ctx: SimulationContext) -> dict[str, Any]:
        """Aggregate hourly results into the final simulation output dictionary.

        This is the third and final phase. It converts the accumulated per-hour
        data in the context into numpy arrays and computes aggregate totals.

        Args:
            ctx: Simulation context with all hourly data accumulated.

        Returns:
            Dictionary matching the SimulationResultData schema.
        """
        total_hours = ctx.total_hours

        # Convert lists to numpy arrays for compatibility with existing consumers
        loads_energy_per_hour = np.array(ctx.loads_energy, float)
        feedin_energy_per_hour = np.array(ctx.feedin_energy, float)
        consumption_energy_per_hour = np.array(ctx.consumption_energy, float)
        costs_per_hour = np.array(ctx.costs, float)
        revenue_per_hour = np.array(ctx.revenue, float)
        losses_wh_per_hour = np.array(ctx.losses, float)
        electricity_price_per_hour = np.array(ctx.electricity_price, float)
        soc_per_hour = np.array(ctx.soc_battery, float)
        soc_ev_per_hour = np.array(ctx.soc_ev, float)
        home_appliance_wh_per_hour = np.array(ctx.home_appliance_wh, float)

        total_cost = np.nansum(costs_per_hour)
        total_losses = np.nansum(losses_wh_per_hour)
        total_revenue = np.nansum(revenue_per_hour)

        return {
            "Last_Wh_pro_Stunde": loads_energy_per_hour,
            "Netzeinspeisung_Wh_pro_Stunde": feedin_energy_per_hour,
            "Netzbezug_Wh_pro_Stunde": consumption_energy_per_hour,
            "Kosten_Euro_pro_Stunde": costs_per_hour,
            "akku_soc_pro_stunde": soc_per_hour,
            "Einnahmen_Euro_pro_Stunde": revenue_per_hour,
            "Gesamtbilanz_Euro": total_cost - total_revenue,
            "EAuto_SoC_pro_Stunde": soc_ev_per_hour,
            "Gesamteinnahmen_Euro": total_revenue,
            "Gesamtkosten_Euro": total_cost,
            "Verluste_Pro_Stunde": losses_wh_per_hour,
            "Gesamt_Verluste": total_losses,
            "Home_appliance_wh_per_hour": home_appliance_wh_per_hour,
            "Electricity_price": electricity_price_per_hour,
        }
