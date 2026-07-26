"""Solver-agnostic energy simulation engine.

Takes device parameters + forecast data + action sequence,
returns simulation results (costs, grid energy, SOC trajectory, etc.).

This engine is a drop-in replacement for GeneticSimulation.simulate()
and must produce bitwise-identical results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from akkudoktoreos.config.config import ConfigEOS
    from akkudoktoreos.optimization.simulation.parameters import (
        OptimizationParameters,
    )
    from akkudoktoreos.optimization.simulation.parameters import (
        EnergyManagementParameters,
    )

from akkudoktoreos.optimization.simulation.context import SimulationContext
from akkudoktoreos.optimization.simulation.devices import (
    Battery,
    DeviceFactory,
    HomeAppliance,
    Inverter,
    SimulationDevices,
)
from akkudoktoreos.optimization.simulation.result import SimulationResult
from akkudoktoreos.optimization.simulation.step import EnergySimulationStep


@dataclass
class SimulationConfig:
    """Immutable configuration for a simulation run.

    Holds all forecast arrays and horizon settings needed by the engine.
    """

    prediction_hours: int
    optimization_hours: int
    start_hour: int = 0

    # Forecast data
    load_energy_array: Optional[np.ndarray] = None
    pv_prediction_wh: Optional[np.ndarray] = None
    elect_price_hourly: Optional[np.ndarray] = None
    elect_revenue_per_hour: Optional[np.ndarray] = None
    temperature_forecast: Optional[np.ndarray] = None

    # Battery LCOS / degradation cost
    price_per_wh_battery: float = 0.0


class EnergySimulationEngine:
    """Solver-agnostic energy simulation engine.

    Takes device parameters + forecast data + action sequence,
    returns simulation results (costs, grid energy, SOC trajectory, etc.).

    Usage:
        engine = EnergySimulationEngine.create(parameters, config)
        result = engine.run(ac_charge, dc_charge, discharge, ev_charge, appliance_start)
    """

    def __init__(
        self,
        devices: SimulationDevices,
        sim_config: SimulationConfig,
    ) -> None:
        """Initialize engine with devices and configuration.

        Use the ``create()`` classmethod for the standard construction path.

        Args:
            devices: SimulationDevices container.
            sim_config: SimulationConfig with forecast data.
        """
        self.devices = devices
        self.sim_config = sim_config
        self._home_appliance_start_hour: Optional[int] = None

    # ── Factory ──────────────────────────────────────────────────────
    @classmethod
    def create(
        cls,
        parameters: OptimizationParameters,
        config: ConfigEOS,
        start_hour: int = 0,
    ) -> EnergySimulationEngine:
        """Create engine from optimization parameters + global config.

        Args:
            parameters: Full optimization parameters (EMS forecasts + device params).
            config: Global EOS configuration.
            start_hour: Simulation start hour (0-23).

        Returns:
            Fully initialized EnergySimulationEngine ready for simulation.
        """
        prediction_hours = config.prediction.hours if config.prediction.hours is not None else 24
        optimization_hours = (
            config.optimization.horizon_hours if config.optimization.horizon_hours is not None else 24
        )

        # Create devices
        devices = DeviceFactory.create_devices(
            parameters, config, prediction_hours, optimization_hours
        )

        # Extract forecast data from EMS parameters
        ems = parameters.ems
        load_energy_array = np.array(ems.total_load, float)
        pv_prediction_wh = np.array(ems.pv_forecast_wh, float)
        elect_price_hourly = np.array(ems.electricity_price_per_wh, float)

        if isinstance(ems.feed_in_tariff_per_wh, list):
            elect_revenue_per_hour = np.array(ems.feed_in_tariff_per_wh, float)
        else:
            elect_revenue_per_hour = np.full(len(load_energy_array), ems.feed_in_tariff_per_wh, float)

        sim_config = SimulationConfig(
            prediction_hours=prediction_hours,
            optimization_hours=optimization_hours,
            start_hour=start_hour,
            load_energy_array=load_energy_array,
            pv_prediction_wh=pv_prediction_wh,
            elect_price_hourly=elect_price_hourly,
            elect_revenue_per_hour=elect_revenue_per_hour,
            price_per_wh_battery=ems.price_per_wh_battery,
        )

        return cls(devices=devices, sim_config=sim_config)

    # ── Public API ───────────────────────────────────────────────────
    def run(
        self,
        ac_charge: np.ndarray,
        dc_charge: np.ndarray,
        discharge: np.ndarray,
        ev_charge: Optional[np.ndarray] = None,
        home_appliance_start: Optional[int] = None,
    ) -> SimulationResult:
        """Run full simulation for given action sequence.

        Executes the three-phase simulation:
        1. Init – reset devices, configure arrays
        2. Hourly steps – process each hour
        3. Finalize – convert to result

        Args:
            ac_charge: AC charge factor array (0=off, 0-1=fraction).
            dc_charge: DC (PV) charge allow array (0/1).
            discharge: Discharge allow array (0/1).
            ev_charge: EV charge factor array (optional).
            home_appliance_start: Home appliance start hour (optional).

        Returns:
            SimulationResult with all per-hour and aggregate data.
        """
        # Phase 1: Init
        ctx = self._init(ac_charge, dc_charge, discharge, ev_charge, home_appliance_start)

        # Phase 2: Hourly steps
        for hour in range(ctx.start_hour, ctx.end_hour):
            hour_idx = hour - ctx.start_hour
            step = self._hourly_step(hour, ctx)

            # Accumulate step results into context
            ctx.feedin_energy[hour_idx] = step.energy_feedin_grid
            ctx.consumption_energy[hour_idx] = step.energy_consumption_grid
            ctx.losses[hour_idx] = step.losses
            ctx.loads_energy[hour_idx] = step.consumption
            ctx.electricity_price[hour_idx] = step.electricity_price
            ctx.costs[hour_idx] = step.cost
            ctx.revenue[hour_idx] = step.revenue
            ctx.home_appliance_wh[hour_idx] = step.home_appliance_wh

        # Phase 3: Finalize
        return self._finalize(ctx)

    def step(
        self,
        hour: int,
        ac_charge: float,
        dc_charge: float,
        discharge: int,
        ev_charge: float = 0.0,
    ) -> EnergySimulationStep:
        """Run single hourly step. For iterative solvers (MPC).

        Requires ``run()`` or ``_init()`` to be called first.

        Args:
            hour: Absolute hour index (0-based from prediction start).
            ac_charge: AC charge factor for this hour.
            dc_charge: DC charge allow flag for this hour.
            discharge: Discharge allow flag for this hour.
            ev_charge: EV charge factor for this hour.

        Returns:
            EnergySimulationStep with results for this hour.
        """
        # This method needs a pre-initialized context.
        # For MPC usage, the caller manages the context.
        raise NotImplementedError(
            "step() requires a pre-initialized context. "
            "Use run() for full simulation or access _context after _init()."
        )

    # ── Internal ─────────────────────────────────────────────────────
    def _init(
        self,
        ac_charge: np.ndarray,
        dc_charge: np.ndarray,
        discharge: np.ndarray,
        ev_charge: Optional[np.ndarray],
        home_appliance_start: Optional[int],
    ) -> SimulationContext:
        """Initialize simulation context with validation and device state.

        Mirrors GeneticSimulation._simulate_init() exactly.

        Args:
            ac_charge: AC charge factor array.
            dc_charge: DC charge allow array.
            discharge: Discharge allow array.
            ev_charge: EV charge factor array.
            home_appliance_start: Home appliance start hour.

        Returns:
            Fully initialized _EngineContext ready for hourly stepping.
        """
        cfg = self.sim_config
        devices = self.devices
        start_hour = cfg.start_hour

        ctx = SimulationContext(start_hour=start_hour)

        # Action arrays
        ctx.ac_charge_hours = ac_charge
        ctx.dc_charge_hours = dc_charge
        ctx.bat_discharge_hours = discharge
        ctx.ev_charge_hours = ev_charge if ev_charge is not None else np.full(cfg.prediction_hours, 0.0)
        ctx.ev_discharge_hours = np.full(cfg.prediction_hours, 0.0)

        # Forecast arrays
        ctx.load_energy_array = cfg.load_energy_array
        ctx.elect_price_hourly = cfg.elect_price_hourly
        ctx.elect_revenue_per_hour = cfg.elect_revenue_per_hour
        ctx.pv_prediction_wh = cfg.pv_prediction_wh

        # Devices
        ctx.battery = devices.battery
        ctx.ev = devices.ev
        ctx.home_appliance = devices.home_appliance
        ctx.inverter = devices.inverter

        # Battery LCOS / degradation cost
        ctx.price_per_wh_battery = cfg.price_per_wh_battery

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
        if ctx.home_appliance and home_appliance_start is not None:
            ctx.home_appliance_enabled = True
            self._home_appliance_start_hour = ctx.home_appliance.set_starting_time(
                home_appliance_start, start_hour
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

    def _hourly_step(self, hour: int, ctx: SimulationContext) -> EnergySimulationStep:
        """Process a single simulation hour.

        Mirrors GeneticSimulation._simulate_hourly_step() exactly.

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

        # --- Inverter energy processing (PV → grid/consumption) ---
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
                    # LCOS cost for battery charging (degradation)
                    result.cost += battery_charged_energy_actual * ctx.price_per_wh_battery

        # --- Financial calculations ---
        result.consumption = consumption
        result.energy_feedin_grid = energy_feedin_grid_actual
        result.energy_consumption_grid = energy_consumption_grid_actual
        result.electricity_price = ctx.elect_price_hourly[hour]
        result.cost += energy_consumption_grid_actual * ctx.elect_price_hourly[hour]
        result.revenue = energy_feedin_grid_actual * ctx.elect_revenue_per_hour[hour]

        return result

    def _finalize(self, ctx: SimulationContext) -> SimulationResult:
        """Aggregate hourly results into final SimulationResult.

        Mirrors GeneticSimulation._simulate_finalize() exactly.

        Args:
            ctx: Simulation context with all hourly data accumulated.

        Returns:
            SimulationResult with all per-hour and aggregate data.
        """
        loads_energy_per_hour = np.array(ctx.loads_energy, float).tolist()
        feedin_energy_per_hour = np.array(ctx.feedin_energy, float).tolist()
        consumption_energy_per_hour = np.array(ctx.consumption_energy, float).tolist()
        costs_per_hour = np.array(ctx.costs, float).tolist()
        revenue_per_hour = np.array(ctx.revenue, float).tolist()
        losses_wh_per_hour = np.array(ctx.losses, float).tolist()
        electricity_price_per_hour = np.array(ctx.electricity_price, float).tolist()
        soc_per_hour = np.array(ctx.soc_battery, float).tolist()
        soc_ev_per_hour = np.array(ctx.soc_ev, float).tolist()
        home_appliance_wh_per_hour = np.array(ctx.home_appliance_wh, float).tolist()

        total_cost = float(np.nansum(costs_per_hour))
        total_losses = float(np.nansum(losses_wh_per_hour))
        total_revenue = float(np.nansum(revenue_per_hour))

        return SimulationResult(
            load_wh_per_hour=loads_energy_per_hour,
            grid_feed_in_wh_per_hour=feedin_energy_per_hour,
            grid_consumption_wh_per_hour=consumption_energy_per_hour,
            costs_per_hour=costs_per_hour,
            revenue_per_hour=revenue_per_hour,
            losses_per_hour=losses_wh_per_hour,
            electricity_price=electricity_price_per_hour,
            battery_soc_per_hour=soc_per_hour,
            ev_soc_per_hour=soc_ev_per_hour,
            home_appliance_wh_per_hour=home_appliance_wh_per_hour,
            total_costs=total_cost,
            total_revenue=total_revenue,
            total_balance=total_cost - total_revenue,
            total_losses=total_losses,
        )
