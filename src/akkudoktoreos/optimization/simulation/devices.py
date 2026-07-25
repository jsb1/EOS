"""Device factory, container and device classes for simulation.

Provides DeviceFactory for creating simulation devices from optimization
parameters + config, SimulationDevices as an immutable container,
and the Battery, Inverter, and HomeAppliance device classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Optional

import numpy as np

from akkudoktoreos.devices.devices import BATTERY_DEFAULT_CHARGE_RATES
from akkudoktoreos.prediction.interpolator import get_eos_load_interpolator
from akkudoktoreos.optimization.simulation.parameters import SolarPanelBatteryParameters

if TYPE_CHECKING:
    from akkudoktoreos.config.config import ConfigEOS
    from akkudoktoreos.optimization.simulation.parameters import (
        OptimizationParameters,
    )
    from akkudoktoreos.optimization.simulation.parameters import (
        BaseBatteryParameters,
        ElectricVehicleParameters,
        HomeApplianceParameters,
        InverterParameters,
    )


@dataclass(frozen=True)
class SimulationDevices:
    """Immutable container for all simulation devices.

    Attributes:
        battery: Main battery (PV-coupled).
        ev: Electric vehicle battery (optional).
        inverter: Inverter managing PV → battery/grid flows.
        home_appliance: Schedulable home appliance (e.g., dishwasher).
    """

    battery: Optional[Battery]
    ev: Optional[Battery]
    inverter: Optional[Inverter]
    home_appliance: Optional[HomeAppliance]


class DeviceFactory:
    """Factory for creating simulation devices from parameters + config."""

    @staticmethod
    def create_battery(
        pv_battery: Optional[SolarPanelBatteryParameters],
        prediction_hours: int,
    ) -> Optional[Battery]:
        """Create the main battery device.

        Args:
            pv_battery: Battery parameters from optimization parameters.
            prediction_hours: Total prediction horizon in hours.

        Returns:
            Battery instance or None if no battery configured.
        """
        if pv_battery is None:
            return None

        battery = Battery(pv_battery, prediction_hours=prediction_hours)
        battery.set_charge_per_hour(np.full(prediction_hours, 0))
        return battery

    @staticmethod
    def create_ev(
        ev_params: Optional[ElectricVehicleParameters],
        prediction_hours: int,
    ) -> Optional[Battery]:
        """Create the EV device.

        Args:
            ev_params: EV parameters from optimization parameters.
            prediction_hours: Total prediction horizon in hours.

        Returns:
            Battery instance (used as EV) or None.
        """
        if ev_params is None:
            return None

        ev = Battery(ev_params, prediction_hours=prediction_hours)
        ev.set_charge_per_hour(np.full(prediction_hours, 1))
        return ev

    @staticmethod
    def create_inverter(
        inverter_params: Optional[InverterParameters],
        battery: Optional[Battery],
    ) -> Optional[Inverter]:
        """Create the inverter device.

        Args:
            inverter_params: Inverter parameters from optimization parameters.
            battery: Battery to connect to the inverter.

        Returns:
            Inverter instance or None.
        """
        if inverter_params is None:
            return None

        return Inverter(inverter_params, battery=battery)

    @staticmethod
    def create_home_appliance(
        dishwasher_params: Optional[HomeApplianceParameters],
        optimization_hours: int,
        prediction_hours: int,
    ) -> Optional[HomeAppliance]:
        """Create the home appliance device.

        Args:
            dishwasher_params: Dishwasher parameters from optimization parameters.
            optimization_hours: Optimization horizon in hours.
            prediction_hours: Total prediction horizon in hours.

        Returns:
            HomeAppliance instance or None.
        """
        if dishwasher_params is None:
            return None

        return HomeAppliance(
            parameters=dishwasher_params,
            optimization_hours=optimization_hours,
            prediction_hours=prediction_hours,
        )

    @staticmethod
    def create_devices(
        parameters: OptimizationParameters,
        config: ConfigEOS,
        prediction_hours: int,
        optimization_hours: int,
    ) -> SimulationDevices:
        """Create all simulation devices from parameters + config.

        Args:
            parameters: Full optimization parameters.
            config: Global EOS configuration.
            prediction_hours: Total prediction horizon in hours.
            optimization_hours: Optimization horizon in hours.

        Returns:
            SimulationDevices container with all devices initialized.
        """
        battery = DeviceFactory.create_battery(parameters.pv_battery, prediction_hours)
        ev = DeviceFactory.create_ev(parameters.ev, prediction_hours)
        inverter = DeviceFactory.create_inverter(parameters.inverter, battery)
        home_appliance = DeviceFactory.create_home_appliance(
            parameters.dishwasher, optimization_hours, prediction_hours
        )

        return SimulationDevices(
            battery=battery,
            ev=ev,
            inverter=inverter,
            home_appliance=home_appliance,
        )


# ── Device Classes ──────────────────────────────────────────────────────
# Moved from src/akkudoktoreos/devices/genetic/ for co-location with
# EnergySimulationEngine and DeviceFactory.


class Battery:
    """Represents a battery device with methods to simulate energy charging and discharging."""

    def __init__(self, parameters: BaseBatteryParameters, prediction_hours: int):
        self.parameters = parameters
        self.prediction_hours = prediction_hours
        self._setup()

    def _setup(self) -> None:
        """Sets up the battery parameters based on provided parameters."""
        self.capacity_wh = self.parameters.capacity_wh
        self.initial_soc_percentage = self.parameters.initial_soc_percentage
        self.charging_efficiency = self.parameters.charging_efficiency
        self.discharging_efficiency = self.parameters.discharging_efficiency

        # Charge rates, in case of None use default
        self.charge_rates = np.array(BATTERY_DEFAULT_CHARGE_RATES, dtype=float)
        if self.parameters.charge_rates:
            charge_rates = np.array(self.parameters.charge_rates, dtype=float)
            charge_rates = np.unique(charge_rates)
            charge_rates.sort()
            self.charge_rates = charge_rates

        # Only assign for storage battery
        self.min_soc_percentage = (
            self.parameters.min_soc_percentage
            if isinstance(self.parameters, SolarPanelBatteryParameters)
            else 0
        )
        self.max_soc_percentage = self.parameters.max_soc_percentage

        # Initialize state of charge
        if self.parameters.max_charge_power_w is not None:
            self.max_charge_power_w = self.parameters.max_charge_power_w
        else:
            self.max_charge_power_w = self.capacity_wh  # TODO this should not be equal capacity_wh
        self.discharge_array = np.full(self.prediction_hours, 0)
        self.charge_array = np.full(self.prediction_hours, 0)
        self.soc_wh = (self.initial_soc_percentage / 100) * self.capacity_wh
        self.min_soc_wh = (self.min_soc_percentage / 100) * self.capacity_wh
        self.max_soc_wh = (self.max_soc_percentage / 100) * self.capacity_wh

    def _lower_charge_rates_desc(self, start_rate: float) -> Iterator[float]:
        """Yield all charge rates lower than a given rate in descending order.

        Args:
            charge_rates (np.ndarray): Sorted 1D array of available charge rates.
            start_rate (float): The reference charge rate.

        Yields:
            float: Charge rates lower than `start_rate`, in descending order.
        """
        charge_rates_fast = self.charge_rates

        # Find the insertion index for start_rate (left-most position)
        idx = np.searchsorted(charge_rates_fast, start_rate, side="left")

        # Yield values before idx in reverse (descending)
        return (charge_rates_fast[j] for j in range(idx - 1, -1, -1))

    def to_dict(self) -> dict[str, Any]:
        """Converts the object to a dictionary representation."""
        return {
            "device_id": self.parameters.device_id,
            "capacity_wh": self.capacity_wh,
            "initial_soc_percentage": self.initial_soc_percentage,
            "soc_wh": self.soc_wh,
            "hours": self.prediction_hours,
            "discharge_array": self.discharge_array,
            "charge_array": self.charge_array,
            "charging_efficiency": self.charging_efficiency,
            "discharging_efficiency": self.discharging_efficiency,
            "max_charge_power_w": self.max_charge_power_w,
        }

    def reset(self) -> None:
        """Resets the battery state to its initial values."""
        self.soc_wh = (self.initial_soc_percentage / 100) * self.capacity_wh
        self.soc_wh = min(self.soc_wh, self.max_soc_wh)  # Only clamp to max
        self.discharge_array = np.full(self.prediction_hours, 0)
        self.charge_array = np.full(self.prediction_hours, 0)

    def set_discharge_per_hour(self, discharge_array: np.ndarray) -> None:
        """Sets the discharge values for each hour."""
        if len(discharge_array) != self.prediction_hours:
            raise ValueError(
                f"Discharge array must have exactly {self.prediction_hours} elements. Got {len(discharge_array)} elements."
            )
        self.discharge_array = np.array(discharge_array)

    def set_charge_per_hour(self, charge_array: np.ndarray) -> None:
        """Sets the charge values for each hour."""
        if len(charge_array) != self.prediction_hours:
            raise ValueError(
                f"Charge array must have exactly {self.prediction_hours} elements. Got {len(charge_array)} elements."
            )
        self.charge_array = np.array(charge_array)

    def current_soc_percentage(self) -> float:
        """Calculates the current state of charge in percentage."""
        return (self.soc_wh / self.capacity_wh) * 100

    def discharge_energy(self, wh: float, hour: int) -> tuple[float, float]:
        """Discharge energy from the battery.

        Discharge is limited by:
        * Requested delivered energy
        * Remaining energy above minimum SoC
        * Maximum discharge power
        * Discharge efficiency

        Args:
            wh (float): Requested delivered energy in watt-hours.
            hour (int): Time index. If `self.discharge_array[hour] == 0`,
                no discharge occurs.

        Returns:
            tuple[float, float]:
                delivered_wh (float): Actual delivered energy [Wh].
                losses_wh (float): Conversion losses [Wh].
        """
        if self.discharge_array[hour] == 0:
            return 0.0, 0.0

        # Raw extractable energy above minimum SoC
        raw_available_wh = max(self.soc_wh - self.min_soc_wh, 0.0)

        # Maximum raw discharge due to power limit
        max_raw_wh = self.max_charge_power_w  # TODO rename to max_discharge_power_w

        # Actual raw withdrawal (internal)
        raw_withdrawal_wh = min(raw_available_wh, max_raw_wh)

        # Convert raw to delivered
        max_deliverable_wh = raw_withdrawal_wh * self.discharging_efficiency

        # Cap by requested delivered energy
        delivered_wh = min(wh, max_deliverable_wh)

        # Effective raw withdrawal based on what is delivered
        raw_used_wh = delivered_wh / self.discharging_efficiency

        # Update SoC
        self.soc_wh -= raw_used_wh
        self.soc_wh = max(self.soc_wh, self.min_soc_wh)

        # Losses
        losses_wh = raw_used_wh - delivered_wh

        return delivered_wh, losses_wh

    def charge_energy(
        self,
        wh: Optional[float],
        hour: int,
        charge_factor: float = 0.0,
    ) -> tuple[float, float]:
        """Charge energy into the battery.

        Two **exclusive** modes:

        **Mode 1:**

        - `wh is not None` and `charge_factor == 0`
        - The raw requested charge energy is `wh` (pre-efficiency).
        - If remaining capacity is insufficient, charging is automatically limited.
        - No exception is raised due to capacity limits.

        **Mode 2:**

        - `wh is None` and `charge_factor > 0`
        - The raw requested energy is `max_charge_power_w * charge_factor`.
        - If the request exceeds remaining capacity, the algorithm tries to find a lower
          `charge_factor` that is compatible. If such a charge factor exists, this hour's
          `charge_factor` is replaced.
        - If no charge factor can accommodate charging, the request is ignored (``(0.0, 0.0)`` is
          returned) and a penalty is applied elsewhere.

        Charging is constrained by:

        - Available SoC headroom (``max_soc_wh − soc_wh``)
        - ``max_charge_power_w``
        - ``charging_efficiency``

        Args:
            wh (float | None):
                Requested raw energy [Wh] before efficiency.
                Must be provided only for Mode 1 (charge_factor must be 0).

            hour (int):
                Time index. If charging is disabled at this hour (charge_array[hour] == 0),
                returns `(0.0, 0.0)`.

            charge_factor (float):
                Fraction (0–1) of max charge power.
                Must be >0 only in Mode 2 (`wh is None`).

        Returns:
            tuple[float, float]:
                stored_wh : float
                    Energy stored after efficiency [Wh].
                losses_wh : float
                    Conversion losses [Wh].

        Raises:
            ValueError:
                - If the mode is ambiguous (neither Mode 1 nor Mode 2).
                - If the final new SoC would exceed capacity_wh.

        Notes:
            stored_wh = raw_input_wh * charging_efficiency
            losses_wh = raw_input_wh − stored_wh
        """
        # Charging allowed in this hour?
        if hour is not None and self.charge_array[hour] == 0:
            return 0.0, 0.0

        # Provide fast (3x..5x) local read access (vs. self.xxx) for repetitive read access
        soc_wh_fast = self.soc_wh
        max_charge_power_w_fast = self.max_charge_power_w
        charging_efficiency_fast = self.charging_efficiency

        # Decide mode & determine raw_request_wh and raw_charge_wh
        if wh is not None and charge_factor == 0.0:  # mode 1
            raw_request_wh = wh
            raw_charge_wh = max(self.max_soc_wh - soc_wh_fast, 0.0) / charging_efficiency_fast
        elif wh is None and charge_factor > 0.0:  # mode 2
            raw_request_wh = max_charge_power_w_fast * charge_factor
            raw_charge_wh = max(self.max_soc_wh - soc_wh_fast, 0.0) / charging_efficiency_fast
            if raw_request_wh > raw_charge_wh:
                # Use a lower charge factor
                lower_charge_factors = self._lower_charge_rates_desc(charge_factor)
                for charge_factor in lower_charge_factors:
                    raw_request_wh = max_charge_power_w_fast * charge_factor
                    if raw_request_wh <= raw_charge_wh:
                        self.charge_array[hour] = charge_factor
                        break
                if raw_request_wh > raw_charge_wh:
                    # ignore request - penalty for missing SoC will be applied
                    self.charge_array[hour] = 0
                    return 0.0, 0.0
        else:
            raise ValueError(
                f"{self.parameters.device_id}: charge_energy must be called either "
                "with wh != None and charge_factor == 0, or with wh == None and charge_factor > 0."
            )

        # Remaining capacity
        max_raw_wh = min(raw_charge_wh, max_charge_power_w_fast)

        # Actual raw intake
        raw_input_wh = raw_request_wh if raw_request_wh < max_raw_wh else max_raw_wh

        # Apply efficiency
        stored_wh = raw_input_wh * charging_efficiency_fast
        new_soc = soc_wh_fast + stored_wh

        if new_soc > self.capacity_wh:
            raise ValueError(
                f"{self.parameters.device_id}: SoC {new_soc} Wh exceeds capacity {self.capacity_wh} Wh"
            )

        self.soc_wh = new_soc
        losses_wh = raw_input_wh - stored_wh

        return stored_wh, losses_wh

    def current_energy_content(self) -> float:
        """Returns the current usable energy in the battery."""
        usable_energy = (self.soc_wh - self.min_soc_wh) * self.discharging_efficiency
        return max(usable_energy, 0.0)


class Inverter:
    """Inverter device for processing PV energy through battery/grid."""

    def __init__(
        self,
        parameters: InverterParameters,
        battery: Optional[Battery] = None,
    ):
        self.parameters: InverterParameters = parameters
        self.battery: Optional[Battery] = battery
        self._setup()

    def _setup(self) -> None:
        from loguru import logger

        if self.battery and self.parameters.battery_id != self.battery.parameters.device_id:
            error_msg = f"Battery ID mismatch - {self.parameters.battery_id} is configured; got {self.battery.parameters.device_id}."
            logger.error(error_msg)
            raise ValueError(error_msg)
        self.self_consumption_predictor = get_eos_load_interpolator()
        self.max_power_wh = (
            self.parameters.max_power_wh
        )  # Maximum power that the inverter can handle
        self.dc_to_ac_efficiency = self.parameters.dc_to_ac_efficiency
        self.ac_to_dc_efficiency = self.parameters.ac_to_dc_efficiency
        self.max_ac_charge_power_w = self.parameters.max_ac_charge_power_w

    def process_energy(
        self, generation: float, consumption: float, hour: int
    ) -> tuple[float, float, float, float]:
        losses = 0.0
        grid_export = 0.0
        grid_import = 0.0
        self_consumption = 0.0

        # Cache inverter DC→AC efficiency for discharge path
        dc_to_ac_eff = self.dc_to_ac_efficiency

        if generation >= consumption:
            if consumption > self.max_power_wh:
                # If consumption exceeds maximum inverter power
                losses += generation - self.max_power_wh
                remaining_power = self.max_power_wh - consumption
                grid_import = -remaining_power  # Negative indicates feeding into the grid
                self_consumption = self.max_power_wh
            else:
                # Calculate scr using cached results per energy management/optimization run
                scr = self.self_consumption_predictor.calculate_self_consumption(
                    consumption, generation
                )

                # Remaining power after consumption
                remaining_power = (generation - consumption) * scr  # EVQ
                # Remaining load Self Consumption not perfect
                remaining_load_evq = (generation - consumption) * (1.0 - scr)

                if remaining_load_evq > 0:
                    # The battery must cover the remaining consumption
                    if self.battery:
                        # Request more DC from battery to account for DC→AC conversion loss
                        dc_request = remaining_load_evq / dc_to_ac_eff
                        from_battery_dc, discharge_losses = self.battery.discharge_energy(
                            dc_request, hour
                        )
                        # Convert DC output to AC
                        from_battery_ac = from_battery_dc * dc_to_ac_eff
                        inverter_discharge_losses = from_battery_dc - from_battery_ac
                        remaining_load_evq -= from_battery_ac
                        losses += discharge_losses + inverter_discharge_losses
                    else:
                        from_battery_ac = 0.0

                    # If the battery cannot fully cover the remaining consumption, the rest is drawn from the grid
                    if remaining_load_evq > 0:
                        grid_import += remaining_load_evq
                        remaining_load_evq = 0
                else:
                    from_battery_ac = 0.0

                if remaining_power > 0:
                    # Load battery with excess energy (DC path, no inverter conversion needed)
                    charge_losses = 0.0
                    if self.battery:
                        charged_energie, charge_losses = self.battery.charge_energy(
                            remaining_power, hour
                        )
                        remaining_surplus = remaining_power - (charged_energie + charge_losses)
                    else:
                        remaining_surplus = remaining_power

                    # Feed-in to the grid based on remaining capacity
                    if remaining_surplus > self.max_power_wh - consumption:
                        grid_export = self.max_power_wh - consumption
                        losses += remaining_surplus - grid_export
                    else:
                        grid_export = remaining_surplus

                    losses += charge_losses
                self_consumption = (
                    consumption + from_battery_ac
                )  # Self-consumption is equal to the load

        else:
            # Case 2: Insufficient generation, cover shortfall
            shortfall = consumption - generation
            available_ac_power = max(self.max_power_wh - generation, 0)

            # Discharge battery to cover shortfall, if possible
            if self.battery:
                # Need shortfall in AC, request more DC from battery for DC→AC conversion
                ac_needed = min(shortfall, available_ac_power)
                dc_request = ac_needed / dc_to_ac_eff
                battery_discharge_dc, discharge_losses = self.battery.discharge_energy(
                    dc_request, hour
                )
                # Convert DC output to AC
                battery_discharge_ac = battery_discharge_dc * dc_to_ac_eff
                inverter_discharge_losses = battery_discharge_dc - battery_discharge_ac
                losses += discharge_losses + inverter_discharge_losses
            else:
                battery_discharge_ac = 0

            # Draw remaining required power from the grid (discharge_losses are already subtracted in the battery)
            grid_import = shortfall - battery_discharge_ac
            self_consumption = generation + battery_discharge_ac

        return grid_export, grid_import, losses, self_consumption


class HomeAppliance:
    """Schedulable home appliance (e.g., dishwasher) with load curve generation."""

    def __init__(
        self,
        parameters: HomeApplianceParameters,
        optimization_hours: int,
        prediction_hours: int,
    ):
        self.parameters: HomeApplianceParameters = parameters
        self.prediction_hours = prediction_hours
        self._setup()

    def _setup(self) -> None:
        """Sets up the home appliance parameters based provided parameters."""
        from akkudoktoreos.config.configabc import TimeWindow, TimeWindowSequence
        from akkudoktoreos.utils.datetimeutil import to_datetime, to_duration, to_time

        self.load_curve = np.zeros(self.prediction_hours)  # Initialize the load curve with zeros
        self.duration_h = self.parameters.duration_h
        self.consumption_wh = self.parameters.consumption_wh
        # setup possible start times
        if self.parameters.time_windows is None:
            self.parameters.time_windows = TimeWindowSequence(
                windows=[
                    TimeWindow(
                        start_time=to_time("00:00"),
                        duration=to_duration(f"{self.prediction_hours} hours"),
                    ),
                ]
            )
        start_datetime = to_datetime().set(hour=0, minute=0, second=0)
        duration = to_duration(f"{self.duration_h} hours")
        self.start_allowed: list[bool] = []
        for hour in range(0, self.prediction_hours):
            self.start_allowed.append(
                self.parameters.time_windows.contains(
                    start_datetime.add(hours=hour), duration=duration
                )
            )
        start_earliest = self.parameters.time_windows.earliest_start_time(duration, start_datetime)
        if start_earliest:
            self.start_earliest = start_earliest.hour
        else:
            self.start_earliest = 0
        start_latest = self.parameters.time_windows.latest_start_time(duration, start_datetime)
        if start_latest:
            self.start_latest = start_latest.hour
        else:
            self.start_latest = 23

    def set_starting_time(self, start_hour: int, global_start_hour: int = 0) -> int:
        """Sets the start time of the device and generates the corresponding load curve.

        :param start_hour: The hour at which the device should start.
        """
        if not self.start_allowed[start_hour]:
            # It is not allowed (by the time windows) to start the application at this time
            if global_start_hour <= self.start_latest:
                # There is a time window left to start the appliance. Use it
                start_hour = self.start_latest
            else:
                # There is no time window left to run the application
                # Set the start into tomorrow
                start_hour = self.start_earliest + 24

        self.reset_load_curve()

        # Calculate power per hour based on total consumption and duration
        power_per_hour = self.consumption_wh / self.duration_h  # Convert to watt-hours

        # Set the power for the duration of use in the load curve array
        if start_hour < len(self.load_curve):
            end_hour = min(start_hour + self.duration_h, self.prediction_hours)
            self.load_curve[start_hour:end_hour] = power_per_hour

        return start_hour

    def reset_load_curve(self) -> None:
        """Resets the load curve."""
        self.load_curve = np.zeros(self.prediction_hours)

    def get_load_curve(self) -> np.ndarray:
        """Returns the current load curve."""
        return self.load_curve

    def get_load_for_hour(self, hour: int) -> float:
        """Returns the load for a specific hour.

        :param hour: The hour for which the load is queried.
        :return: The load in watts for the specified hour.
        """
        if hour < 0 or hour >= self.prediction_hours:
            raise ValueError(
                f"The specified hour {hour} is outside the available time frame {self.prediction_hours}."
            )

        return self.load_curve[hour]


# ── Legacy Re-exports ──────────────────────────────────────────────────
# Keep backward compatibility for existing import paths.

