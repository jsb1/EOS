"""Simulation parameter base class and device parameter models.

Provides a solver-agnostic base class for all simulation parameters,
along with device-specific parameter classes for Battery, EV, Home
Appliance, and Inverter.
"""

from __future__ import annotations

from typing import Optional, Union

from pydantic import (
    AliasChoices,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from loguru import logger

from akkudoktoreos.core.coreabc import (
    ConfigMixin,
    MeasurementMixin,
    PredictionMixin,
)
from akkudoktoreos.core.pydantic import PydanticBaseModel
from akkudoktoreos.utils.datetimeutil import to_duration


class SimulationParametersBaseModel(PydanticBaseModel):
    """Pydantic base model for simulation parameters.

    Subclass this for any parameter model that should forbid extra fields.
    """

    model_config = ConfigDict(extra="forbid")


# ── Helper factories ──────────────────────────────────────────────────────


def _max_charging_power_field(description: Optional[str] = None) -> float:
    if description is None:
        description = "Maximum charging power in watts."
    return Field(default=5000, gt=0, json_schema_extra={"description": description})


def _initial_soc_percentage_field(description: str) -> int:
    return Field(
        default=0, ge=0, le=100, json_schema_extra={"description": description, "examples": [42]}
    )


def _discharging_efficiency_field(default_value: float) -> float:
    return Field(
        default=default_value,
        gt=0,
        le=1,
        json_schema_extra={
            "description": "A float representing the discharge efficiency of the battery."
        },
    )


# ── Device parameter classes ──────────────────────────────────────────────


class BaseBatteryParameters(SimulationParametersBaseModel):
    """Battery Device Simulation Configuration."""

    device_id: str = Field(
        json_schema_extra={"description": "ID of battery", "examples": ["battery1"]}
    )
    hours: Optional[int] = Field(
        default=None,
        gt=0,
        json_schema_extra={
            "description": "Number of prediction hours. Defaults to global config prediction hours.",
            "examples": [None],
        },
    )
    capacity_wh: int = Field(
        gt=0,
        json_schema_extra={
            "description": "An integer representing the capacity of the battery in watt-hours.",
            "examples": [8000],
        },
    )
    charging_efficiency: float = Field(
        default=0.88,
        gt=0,
        le=1,
        json_schema_extra={
            "description": "A float representing the charging efficiency of the battery."
        },
    )
    discharging_efficiency: float = _discharging_efficiency_field(0.88)
    max_charge_power_w: Optional[float] = _max_charging_power_field()
    initial_soc_percentage: int = _initial_soc_percentage_field(
        "An integer representing the state of charge of the battery at the **start** of the current hour (not the current state)."
    )
    min_soc_percentage: int = Field(
        default=0,
        ge=0,
        le=100,
        json_schema_extra={
            "description": "An integer representing the minimum state of charge (SOC) of the battery in percentage.",
            "examples": [10],
        },
    )
    max_soc_percentage: int = Field(
        default=100,
        ge=0,
        le=100,
        json_schema_extra={
            "description": "An integer representing the maximum state of charge (SOC) of the battery in percentage."
        },
    )
    charge_rates: Optional[list[float]] = Field(
        default=None,
        json_schema_extra={
            "description": "Charge rates as factor of maximum charging power [0.00 ... 1.00]. None denotes all charge rates are available.",
            "examples": [[0.0, 0.25, 0.5, 0.75, 1.0], None],
        },
    )


class SolarPanelBatteryParameters(BaseBatteryParameters):
    """PV battery device simulation configuration."""

    max_charge_power_w: Optional[float] = _max_charging_power_field()


class ElectricVehicleParameters(BaseBatteryParameters):
    """Battery Electric Vehicle Device Simulation Configuration."""

    device_id: str = Field(
        json_schema_extra={"description": "ID of electric vehicle", "examples": ["ev1"]}
    )
    discharging_efficiency: float = _discharging_efficiency_field(1.0)
    initial_soc_percentage: int = _initial_soc_percentage_field(
        "An integer representing the current state of charge (SOC) of the battery in percentage."
    )


class HomeApplianceParameters(SimulationParametersBaseModel):
    """Home Appliance Device Simulation Configuration."""

    device_id: str = Field(
        json_schema_extra={"description": "ID of home appliance", "examples": ["dishwasher"]}
    )
    hours: Optional[int] = Field(
        default=None,
        gt=0,
        json_schema_extra={
            "description": "Number of prediction hours. Defaults to global config prediction hours.",
            "examples": [None],
        },
    )
    consumption_wh: int = Field(
        gt=0,
        json_schema_extra={
            "description": "An integer representing the energy consumption of a household device in watt-hours.",
            "examples": [2000],
        },
    )
    duration_h: int = Field(
        gt=0,
        json_schema_extra={
            "description": "An integer representing the usage duration of a household device in hours.",
            "examples": [3],
        },
    )
    time_windows: Optional["TimeWindowSequence"] = Field(  # noqa: F821
        default=None,
        json_schema_extra={
            "description": "List of allowed time windows. Defaults to optimization general time window.",
            "examples": [
                [
                    {"start_time": "10:00", "duration": "3 hours"},
                ],
            ],
        },
    )


# Resolve forward reference to TimeWindowSequence
from akkudoktoreos.config.configabc import TimeWindowSequence  # noqa: E402

HomeApplianceParameters.model_rebuild()


class InverterParameters(SimulationParametersBaseModel):
    """Inverter Device Simulation Configuration."""

    device_id: str = Field(
        json_schema_extra={"description": "ID of inverter", "examples": ["inverter1"]}
    )
    hours: Optional[int] = Field(
        default=None,
        gt=0,
        json_schema_extra={
            "description": "Number of prediction hours. Defaults to global config prediction hours.",
            "examples": [None],
        },
    )
    max_power_wh: float = Field(gt=0, json_schema_extra={"examples": [10000]})
    battery_id: Optional[str] = Field(
        default=None,
        json_schema_extra={"description": "ID of battery", "examples": [None, "battery1"]},
    )
    ac_to_dc_efficiency: float = Field(
        default=1.0,
        ge=0,
        le=1,
        json_schema_extra={
            "description": (
                "Efficiency of AC to DC conversion (for AC/grid charging of battery). "
                "Set to 0 to disable AC charging via inverter. "
                "Default 1.0 for backward compatibility (no additional inverter loss)."
            ),
            "examples": [0.95, 1.0, 0.0],
        },
    )
    dc_to_ac_efficiency: float = Field(
        default=1.0,
        gt=0,
        le=1,
        json_schema_extra={
            "description": (
                "Efficiency of DC to AC conversion (for battery discharging to AC load/grid). "
                "Default 1.0 for backward compatibility (no additional inverter loss)."
            ),
            "examples": [0.95, 1.0],
        },
    )
    max_ac_charge_power_w: Optional[float] = Field(
        default=None,
        ge=0,
        json_schema_extra={
            "description": (
                "Maximum AC charging power in watts. "
                "None means no additional limit (battery's own max_charge_power_w applies). "
                "Set to 0 to disable AC charging."
            ),
            "examples": [None, 0, 5000],
        },
    )


# ── Energy management parameters ──────────────────────────────────────────


class EnergyManagementParameters(SimulationParametersBaseModel):
    """Encapsulates energy-related forecasts and costs for simulation.

    Solver-agnostic: usable by Genetic, DP, LP, or any other optimization algorithm.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    pv_forecast_wh: list[float] = Field(
        validation_alias=AliasChoices("pv_forecast_wh", "pv_prognose_wh"),
        json_schema_extra={
            "description": "An array of floats representing the forecasted photovoltaic output in watts for different time intervals."
        },
    )
    electricity_price_per_wh: list[float] = Field(
        validation_alias=AliasChoices("electricity_price_per_wh", "strompreis_euro_pro_wh"),
        json_schema_extra={
            "description": "An array of floats representing the electricity price per watt-hour for different time intervals."
        },
    )
    feed_in_tariff_per_wh: Union[list[float], float] = Field(
        validation_alias=AliasChoices("feed_in_tariff_per_wh", "einspeiseverguetung_euro_pro_wh"),
        json_schema_extra={
            "description": "A float or array of floats representing the feed-in compensation per watt-hour."
        },
    )
    price_per_wh_battery: float = Field(
        validation_alias=AliasChoices("price_per_wh_battery", "preis_euro_pro_wh_akku"),
        json_schema_extra={
            "description": "A float representing the cost of battery energy per watt-hour."
        },
    )
    total_load: list[float] = Field(
        validation_alias=AliasChoices("total_load", "gesamtlast"),
        json_schema_extra={
            "description": "An array of floats representing the total load (consumption) in watts for different time intervals."
        },
    )

    # Computed fields for backward compatibility (deprecated German names)
    @computed_field(json_schema_extra={"deprecated": True})
    def pv_prognose_wh(self) -> list[float]:
        """Deprecated: Use pv_forecast_wh instead."""
        return self.pv_forecast_wh

    @computed_field(json_schema_extra={"deprecated": True})
    def strompreis_euro_pro_wh(self) -> list[float]:
        """Deprecated: Use electricity_price_per_wh instead."""
        return self.electricity_price_per_wh

    @computed_field(json_schema_extra={"deprecated": True})
    def einspeiseverguetung_euro_pro_wh(self) -> Union[list[float], float]:
        """Deprecated: Use feed_in_tariff_per_wh instead."""
        return self.feed_in_tariff_per_wh

    @computed_field(json_schema_extra={"deprecated": True})
    def preis_euro_pro_wh_akku(self) -> float:
        """Deprecated: Use price_per_wh_battery instead."""
        return self.price_per_wh_battery

    @computed_field(json_schema_extra={"deprecated": True})
    def gesamtlast(self) -> list[float]:
        """Deprecated: Use total_load instead."""
        return self.total_load

    @model_validator(mode="after")
    def validate_list_length(self) -> Self:
        """Validate that all input lists are of the same length.

        Raises:
            ValueError: If input list lengths differ.
        """
        pv_forecast_length = len(self.pv_forecast_wh)
        if (
            pv_forecast_length != len(self.electricity_price_per_wh)
            or pv_forecast_length != len(self.total_load)
            or (
                isinstance(self.feed_in_tariff_per_wh, list)
                and pv_forecast_length != len(self.feed_in_tariff_per_wh)
            )
        ):
            raise ValueError("Input lists have different lengths")
        return self


# ── Optimization parameters ───────────────────────────────────────────────


class OptimizationParameters(
    ConfigMixin,
    MeasurementMixin,
    PredictionMixin,
    SimulationParametersBaseModel,
):
    """Solver-agnostic optimization parameters.

    Base class holding all parameters common to any optimisation solver
    (GA, DP, MPC, LP). Subclasses add solver-specific fields or behaviour.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    ems: EnergyManagementParameters
    pv_battery: Optional[SolarPanelBatteryParameters] = Field(
        validation_alias=AliasChoices("pv_battery", "pv_akku"),
        json_schema_extra={"description": "PV battery parameters."},
    )
    inverter: Optional[InverterParameters]
    ev: Optional[ElectricVehicleParameters] = Field(
        validation_alias=AliasChoices("ev", "eauto"),
        json_schema_extra={"description": "Electric vehicle parameters."},
    )
    dishwasher: Optional[HomeApplianceParameters] = None
    temperature_forecast: Optional[list[Optional[float]]] = Field(
        default=None,
        json_schema_extra={
            "description": "An array of floats representing the temperature forecast in degrees Celsius for different time intervals."
        },
    )
    start_solution: Optional[list[float]] = Field(
        default=None,
        json_schema_extra={
            "description": "Can be `null` or contain a previous solution (if available)."
        },
    )

    # Computed fields for backward compatibility (deprecated German names)
    @computed_field(json_schema_extra={"deprecated": True})
    def pv_akku(self) -> Optional[SolarPanelBatteryParameters]:
        """Deprecated: Use pv_battery instead."""
        return self.pv_battery

    @computed_field(json_schema_extra={"deprecated": True})
    def eauto(self) -> Optional[ElectricVehicleParameters]:
        """Deprecated: Use ev instead."""
        return self.ev

    @model_validator(mode="after")
    def validate_list_length(self) -> Self:
        """Ensure that temperature forecast list matches the PV forecast length.

        Raises:
            ValueError: If list lengths mismatch.
        """
        arr_length = len(self.ems.pv_forecast_wh)
        if self.temperature_forecast is not None and arr_length != len(self.temperature_forecast):
            raise ValueError("Input lists have different lengths")
        return self

    @field_validator("start_solution")
    def validate_start_solution(
        cls, start_solution: Optional[list[float]]
    ) -> Optional[list[float]]:
        """Validate that the starting solution has at least two elements."""
        if start_solution is not None and len(start_solution) < 2:
            raise ValueError("Requires at least two values.")
        return start_solution

    @classmethod
    async def prepare(cls) -> "Optional[OptimizationParameters]":
        """Prepare optimization parameters from config, forecast and measurement data.

        Fills in values needed for optimization from available configuration, predictions and
        measurements. If some data is missing, default or demo values are used.

        Returns:
            OptimizationParameters: The fully prepared optimization parameters.

        Raises:
            ValueError: If required configuration values like start time are missing.
        """
        from akkudoktoreos.core.coreabc import get_ems

        ems = get_ems()

        # Check for run definitions
        if ems.start_datetime is None:
            error_msg = "Start datetime unknown."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Check for general predictions conditions
        if cls.config.general.latitude is None:
            default_latitude = 52.52
            logger.info(f"Latitude unknown - defaulting to {default_latitude}.")
            cls.config.general.latitude = default_latitude
        if cls.config.general.longitude is None:
            default_longitude = 13.405
            logger.info(f"Longitude unknown - defaulting to {default_longitude}.")
            cls.config.general.longitude = default_longitude
        if cls.config.prediction.hours is None:
            logger.info("Prediction hours unknown - defaulting to 48 hours.")
            cls.config.prediction.hours = 48
        if cls.config.prediction.historic_hours is None:
            logger.info("Prediction historic hours unknown - defaulting to 24 hours.")
            cls.config.prediction.historic_hours = 24
        # Check optimization definitions
        if cls.config.optimization.horizon_hours is None:
            logger.info("Optimization horizon unknown - defaulting to 24 hours.")
            cls.config.optimization.horizon_hours = 24
        if cls.config.optimization.interval is None:
            logger.info("Optimization interval unknown - defaulting to 3600 seconds.")
            cls.config.optimization.interval = 3600
        if cls.config.optimization.interval != 3600:
            logger.info(
                "Optimization interval '{}' seconds not supported - forced to 3600 seconds."
            )
            cls.config.optimization.interval = 3600

        # Get start solution from last run
        start_solution = None
        last_solution = ems.genetic_solution()
        if last_solution and last_solution.start_solution:
            start_solution = last_solution.start_solution

        # Prepare solver-specific config defaults (e.g. GA penalties)
        await cls._prepare_solver_config()

        # Collect forecast and device data
        data = await cls._collect_forecast_and_device_data(ems, start_solution)
        if data is None:
            return None

        (
            pvforecast_ac_power,
            elecprice_marketprice_wh,
            feed_in_tariff_wh,
            loadforecast_power_w,
            weather_temp_air,
            battery_params,
            battery_lcos_kwh,
            electric_vehicle_params,
            inverter_params,
            home_appliance_params,
        ) = data

        return cls(
            ems=EnergyManagementParameters(
                pv_forecast_wh=pvforecast_ac_power,
                electricity_price_per_wh=elecprice_marketprice_wh,
                feed_in_tariff_per_wh=feed_in_tariff_wh,
                total_load=loadforecast_power_w,
                price_per_wh_battery=battery_lcos_kwh / 1000,
            ),
            temperature_forecast=weather_temp_air,
            pv_battery=battery_params,
            ev=electric_vehicle_params,
            inverter=inverter_params,
            dishwasher=home_appliance_params,
            start_solution=start_solution,
        )

    @classmethod
    async def _prepare_solver_config(cls) -> None:
        """Prepare solver-specific config defaults. Override in subclasses."""
        pass

    @classmethod
    async def _collect_forecast_and_device_data(
        cls, ems, start_solution: Optional[list[float]]
    ) -> Optional[tuple]:
        """Collect forecast and device data from predictions and measurements."""
        interval = to_duration(cls.config.optimization.interval)
        power_to_energy_per_interval_factor = cls.config.optimization.interval / 3600
        parameter_start_datetime = ems.start_datetime.set(hour=0, second=0, microsecond=0)
        parameter_end_datetime = parameter_start_datetime.add(hours=cls.config.prediction.hours)
        max_retries = 10

        for attempt in range(1, max_retries + 1):
            if attempt > max_retries:
                error_msg = f"Maximum retries {max_retries} for parameter collection exceeded. Parameter preparation attempt {attempt}."
                logger.error(error_msg)
                raise ValueError(error_msg)

            await cls.prediction.update_data()

            # PV forecast
            try:
                array = await cls.prediction.key_to_array(
                    key="pvforecast_ac_power",
                    start_datetime=parameter_start_datetime,
                    end_datetime=parameter_end_datetime,
                    interval=interval,
                    fill_method="linear",
                )
                pvforecast_ac_power = (array * power_to_energy_per_interval_factor).tolist()
            except Exception as e:
                logger.info(
                    "No PV forecast data available - defaulting to demo data. Parameter preparation attempt {}: {}",
                    attempt,
                    e,
                )
                cls.config.merge_settings_from_dict(
                    {
                        "pvforecast": {
                            "provider": "PVForecastAkkudoktor",
                            "max_planes": 4,
                            "planes": [
                                {
                                    "peakpower": 5.0,
                                    "surface_azimuth": 170,
                                    "surface_tilt": 7,
                                    "userhorizon": [20, 27, 22, 20],
                                    "inverter_paco": 10000,
                                },
                                {
                                    "peakpower": 4.8,
                                    "surface_azimuth": 90,
                                    "surface_tilt": 7,
                                    "userhorizon": [30, 30, 30, 50],
                                    "inverter_paco": 10000,
                                },
                                {
                                    "peakpower": 1.4,
                                    "surface_azimuth": 140,
                                    "surface_tilt": 60,
                                    "userhorizon": [60, 30, 0, 30],
                                    "inverter_paco": 2000,
                                },
                                {
                                    "peakpower": 1.6,
                                    "surface_azimuth": 185,
                                    "surface_tilt": 45,
                                    "userhorizon": [45, 25, 30, 60],
                                    "inverter_paco": 1400,
                                },
                            ],
                        },
                    }
                )
                continue

            # Electricity market price
            try:
                array = await cls.prediction.key_to_array(
                    key="elecprice_marketprice_wh",
                    start_datetime=parameter_start_datetime,
                    end_datetime=parameter_end_datetime,
                    interval=interval,
                    fill_method="ffill",
                )
                elecprice_marketprice_wh = array.tolist()
            except Exception as e:
                logger.info(
                    "No Electricity Marketprice forecast data available - defaulting to demo data. Parameter preparation attempt {}: {}",
                    attempt,
                    e,
                )
                cls.config.elecprice.provider = "ElecPriceAkkudoktor"
                continue

            # Load forecast
            try:
                array = await cls.prediction.key_to_array(
                    key="loadforecast_power_w",
                    start_datetime=parameter_start_datetime,
                    end_datetime=parameter_end_datetime,
                    interval=interval,
                    fill_method="ffill",
                )
                loadforecast_power_w = array.tolist()
            except Exception as e:
                logger.info(
                    "No Load forecast data available - defaulting to demo data. Parameter preparation attempt {}: {}",
                    attempt,
                    e,
                )
                cls.config.merge_settings_from_dict(
                    {
                        "load": {
                            "provider": "LoadAkkudoktor",
                            "loadakkudoktor": {
                                "loadakkudoktor_year_energy_kwh": "3000",
                            },
                        },
                    }
                )
                continue

            # Feed-in tariff
            try:
                array = await cls.prediction.key_to_array(
                    key="feed_in_tariff_wh",
                    start_datetime=parameter_start_datetime,
                    end_datetime=parameter_end_datetime,
                    interval=interval,
                    fill_method="ffill",
                )
                feed_in_tariff_wh = array.tolist()
            except Exception as e:
                logger.info(
                    "No feed in tariff forecast data available - defaulting to demo data. Parameter preparation attempt {}: {}",
                    attempt,
                    e,
                )
                cls.config.merge_settings_from_dict(
                    {
                        "feedintariff": {
                            "provider": "FeedInTariffFixed",
                            "feedintarifffixed": {
                                "feed_in_tariff_kwh": 0.078,
                            },
                        },
                    }
                )
                continue

            # Weather forecast
            try:
                array = await cls.prediction.key_to_array(
                    key="weather_temp_air",
                    start_datetime=parameter_start_datetime,
                    end_datetime=parameter_end_datetime,
                    interval=interval,
                    fill_method="ffill",
                )
                weather_temp_air = array.tolist()
            except Exception as e:
                logger.info(
                    "No weather forecast data available - defaulting to demo data. Parameter preparation attempt {}: {}",
                    attempt,
                    e,
                )
                cls.config.weather.provider = "BrightSky"
                continue

            # Batteries
            if cls.config.devices.max_batteries is None:
                logger.info("Number of battery devices not configured - defaulting to 1.")
                cls.config.devices.max_batteries = 1
            if cls.config.devices.max_batteries == 0:
                battery_params = None
                battery_lcos_kwh = 0
            else:
                if cls.config.devices.batteries is None:
                    logger.info("No battery device data available - defaulting to demo data.")
                    cls.config.devices.batteries = [{"device_id": "battery1", "capacity_wh": 8000}]
                try:
                    battery_config = cls.config.devices.batteries[0]
                    battery_params = SolarPanelBatteryParameters(
                        device_id=battery_config.device_id,
                        capacity_wh=battery_config.capacity_wh,
                        charging_efficiency=battery_config.charging_efficiency,
                        discharging_efficiency=battery_config.discharging_efficiency,
                        max_charge_power_w=battery_config.max_charge_power_w,
                        min_soc_percentage=battery_config.min_soc_percentage,
                        max_soc_percentage=battery_config.max_soc_percentage,
                        charge_rates=battery_config.charge_rates,
                    )
                except Exception as e:
                    logger.info(
                        "No battery device data available - defaulting to demo data. Parameter preparation attempt {}: {}",
                        attempt,
                        e,
                    )
                    cls.config.devices.batteries = [{"device_id": "battery1", "capacity_wh": 8000}]
                    continue
                if battery_config.levelized_cost_of_storage_kwh is None:
                    logger.info(
                        "No battery device LCOS data available - defaulting to 0 [amount/kWh]. Parameter preparation attempt {}.",
                        attempt,
                    )
                    battery_config.levelized_cost_of_storage_kwh = 0
                battery_lcos_kwh = battery_config.levelized_cost_of_storage_kwh
                try:
                    initial_soc_factor = await cls.measurement.key_to_value(
                        key=battery_config.measurement_key_soc_factor,
                        target_datetime=ems.start_datetime,
                        time_window=to_duration(to_duration("48 hours")),
                    )
                    if initial_soc_factor > 1.0 or initial_soc_factor < 0.0:
                        logger.error(
                            f"Invalid battery initial SoC factor {initial_soc_factor} - defaulting to 0.0."
                        )
                        initial_soc_factor = 0.0
                    initial_soc_percentage = int(initial_soc_factor * 100)
                except Exception:
                    initial_soc_percentage = None
                if initial_soc_percentage is None:
                    logger.info(
                        f"No battery device SoC data (measurement key = '{battery_config.measurement_key_soc_factor}') available - defaulting to 0."
                    )
                    initial_soc_percentage = 0
                battery_params.initial_soc_percentage = initial_soc_percentage

            # Electric Vehicles
            if cls.config.devices.max_electric_vehicles is None:
                logger.info("Number of electric_vehicle devices not configured - defaulting to 1.")
                cls.config.devices.max_electric_vehicles = 1
            if cls.config.devices.max_electric_vehicles == 0:
                electric_vehicle_params = None
            else:
                if cls.config.devices.electric_vehicles is None:
                    logger.info(
                        "No electric vehicle device data available - defaulting to demo data."
                    )
                    cls.config.devices.max_electric_vehicles = 1
                    cls.config.devices.electric_vehicles = [
                        {
                            "device_id": "ev11",
                            "capacity_wh": 50000,
                            "charge_rates": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                            "min_soc_percentage": 70,
                        }
                    ]
                try:
                    electric_vehicle_config = cls.config.devices.electric_vehicles[0]
                    electric_vehicle_params = ElectricVehicleParameters(
                        device_id=electric_vehicle_config.device_id,
                        capacity_wh=electric_vehicle_config.capacity_wh,
                        charging_efficiency=electric_vehicle_config.charging_efficiency,
                        discharging_efficiency=electric_vehicle_config.discharging_efficiency,
                        charge_rates=electric_vehicle_config.charge_rates,
                        max_charge_power_w=electric_vehicle_config.max_charge_power_w,
                        min_soc_percentage=electric_vehicle_config.min_soc_percentage,
                        max_soc_percentage=electric_vehicle_config.max_soc_percentage,
                    )
                except Exception as e:
                    logger.info(
                        "No electric_vehicle device data available - defaulting to demo data. Parameter preparation attempt {}: {}",
                        attempt,
                        e,
                    )
                    cls.config.devices.max_electric_vehicles = 1
                    cls.config.devices.electric_vehicles = [
                        {
                            "device_id": "ev12",
                            "capacity_wh": 50000,
                            "charge_rates": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                            "min_soc_percentage": 70,
                        }
                    ]
                    continue
                try:
                    initial_soc_factor = await cls.measurement.key_to_value(
                        key=electric_vehicle_config.measurement_key_soc_factor,
                        target_datetime=ems.start_datetime,
                        time_window=to_duration(to_duration("48 hours")),
                    )
                    if initial_soc_factor > 1.0 or initial_soc_factor < 0.0:
                        logger.error(
                            f"Invalid electric vehicle initial SoC factor {initial_soc_factor} - defaulting to 0.0."
                        )
                        initial_soc_factor = 0.0
                    initial_soc_percentage = int(initial_soc_factor * 100)
                except Exception:
                    initial_soc_percentage = None
                if initial_soc_percentage is None:
                    logger.info(
                        f"No electric vehicle device SoC data (measurement key = '{electric_vehicle_config.measurement_key_soc_factor}') available - defaulting to 0."
                    )
                    initial_soc_percentage = 0
                electric_vehicle_params.initial_soc_percentage = initial_soc_percentage

            # Inverters
            if cls.config.devices.max_inverters is None:
                logger.info("Number of inverter devices not configured - defaulting to 1.")
                cls.config.devices.max_inverters = 1
            if cls.config.devices.max_inverters == 0:
                inverter_params = None
            else:
                if cls.config.devices.inverters is None:
                    logger.info("No inverter device data available - defaulting to demo data.")
                    cls.config.devices.inverters = [
                        {
                            "device_id": "inverter1",
                            "max_power_w": 10000,
                            "battery_id": battery_config.device_id,
                        }
                    ]
                try:
                    inverter_config = cls.config.devices.inverters[0]
                    inverter_params = InverterParameters(
                        device_id=inverter_config.device_id,
                        max_power_wh=inverter_config.max_power_w,
                        battery_id=inverter_config.battery_id,
                        ac_to_dc_efficiency=inverter_config.ac_to_dc_efficiency,
                        dc_to_ac_efficiency=inverter_config.dc_to_ac_efficiency,
                        max_ac_charge_power_w=inverter_config.max_ac_charge_power_w,
                    )
                except Exception as e:
                    logger.info(
                        "No inverter device data available - defaulting to demo data. Parameter preparation attempt {}: {}",
                        attempt,
                        e,
                    )
                    cls.config.devices.inverters = [
                        {
                            "device_id": "inverter1",
                            "max_power_w": 10000,
                            "battery_id": battery_config.device_id,
                        }
                    ]
                    continue

            # Home Appliances
            if cls.config.devices.max_home_appliances is None:
                logger.info("Number of home appliance devices not configured - defaulting to 1.")
                cls.config.devices.max_home_appliances = 1
            if cls.config.devices.max_home_appliances == 0:
                home_appliance_params = None
            else:
                home_appliance_params = None
                if cls.config.devices.home_appliances is None:
                    logger.info(
                        "No home appliance device data available - defaulting to demo data."
                    )
                    cls.config.devices.home_appliances = [
                        {
                            "device_id": "dishwasher1",
                            "consumption_wh": 2000,
                            "duration_h": 3.0,
                            "time_windows": {
                                "windows": [
                                    {
                                        "start_time": "08:00",
                                        "duration": "5 hours",
                                    },
                                    {
                                        "start_time": "15:00",
                                        "duration": "3 hours",
                                    },
                                ],
                            },
                        }
                    ]
                try:
                    home_appliance_config = cls.config.devices.home_appliances[0]
                    home_appliance_params = HomeApplianceParameters(
                        device_id=home_appliance_config.device_id,
                        consumption_wh=home_appliance_config.consumption_wh,
                        duration_h=home_appliance_config.duration_h,
                        time_windows=home_appliance_config.time_windows,
                    )
                except Exception as e:
                    logger.info(
                        "No home appliance device data available - defaulting to demo data. Parameter preparation attempt {}: {}",
                        attempt,
                        e,
                    )
                    cls.config.devices.home_appliances = [
                        {
                            "device_id": "dishwasher1",
                            "consumption_wh": 2000,
                            "duration_h": 3.0,
                            "time_windows": None,
                        }
                    ]
                    continue

            return (
                pvforecast_ac_power,
                elecprice_marketprice_wh,
                feed_in_tariff_wh,
                loadforecast_power_w,
                weather_temp_air,
                battery_params,
                battery_lcos_kwh,
                electric_vehicle_params,
                inverter_params,
                home_appliance_params,
            )

        return None


__all__ = [
    "SimulationParametersBaseModel",
    "BaseBatteryParameters",
    "SolarPanelBatteryParameters",
    "ElectricVehicleParameters",
    "HomeApplianceParameters",
    "InverterParameters",
    "EnergyManagementParameters",
    "OptimizationParameters",
]
