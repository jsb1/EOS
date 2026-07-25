"""Identity tests for EnergySimulationEngine vs SimulationSession.

Verifies that EnergySimulationEngine produces bitwise-identical results
to SimulationSession.simulate() using the exact same device parameters
and forecast data.
"""

from __future__ import annotations

import numpy as np
import pytest

from akkudoktoreos.optimization.simulation.devices import (
    Battery,
    HomeAppliance,
    Inverter,
)
from akkudoktoreos.optimization.simulation.session import SimulationSession
from akkudoktoreos.optimization.simulation.parameters import (
    ElectricVehicleParameters,
    EnergyManagementParameters,
    HomeApplianceParameters,
    InverterParameters,
    SolarPanelBatteryParameters,
)
from akkudoktoreos.optimization.simulation.devices import SimulationDevices
from akkudoktoreos.optimization.simulation.engine import (
    EnergySimulationEngine,
    SimulationConfig,
)

start_hour = 1


# ── Test data (identical to test_geneticsimulation.py) ──────────────

PV_PROGNOSE_WH = [
    0, 0, 0, 0, 0, 0, 0, 8.05, 352.91, 728.51, 930.28, 1043.25,
    1106.74, 1161.69, 6018.82, 5519.07, 3969.88, 3017.96, 1943.07,
    1007.17, 319.67, 7.88, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.04,
    335.59, 705.32, 1121.12, 1604.79, 2157.38, 1433.25, 5718.49,
    4553.96, 3027.55, 2574.46, 1720.4, 963.4, 383.3, 0, 0, 0,
]

STROMPREIS_EURO_PRO_WH = [
    0.0003384, 0.0003318, 0.0003284, 0.0003283, 0.0003289, 0.0003334,
    0.0003290, 0.0003302, 0.0003042, 0.0002430, 0.0002280, 0.0002212,
    0.0002093, 0.0001879, 0.0001838, 0.0002004, 0.0002198, 0.0002270,
    0.0002997, 0.0003195, 0.0003081, 0.0002969, 0.0002921, 0.0002780,
    0.0003384, 0.0003318, 0.0003284, 0.0003283, 0.0003289, 0.0003334,
    0.0003290, 0.0003302, 0.0003042, 0.0002430, 0.0002280, 0.0002212,
    0.0002093, 0.0001879, 0.0001838, 0.0002004, 0.0002198, 0.0002270,
    0.0002997, 0.0003195, 0.0003081, 0.0002969, 0.0002921, 0.0002780,
]

EINSPEISEVERGUETUNG_EURO_PRO_WH = 0.00007
PREIS_EURO_PRO_WH_AKKU = 0.0001

GESAMTLAST = [
    676.71, 876.19, 527.13, 468.88, 531.38, 517.95, 483.15, 472.28,
    1011.68, 995.00, 1053.07, 1063.91, 1320.56, 1132.03, 1163.67,
    1176.82, 1216.22, 1103.78, 1129.12, 1178.71, 1050.98, 988.56,
    912.38, 704.61, 516.37, 868.05, 694.34, 608.79, 556.31, 488.89,
    506.91, 804.89, 1141.98, 1056.97, 992.46, 1155.99, 827.01,
    1257.98, 1232.67, 871.26, 860.88, 1158.03, 1222.72, 1221.04,
    949.99, 987.01, 733.99, 592.97,
]

PREDICTION_HOURS = 48
OPTIMIZATION_HOURS = 24
HOME_APPLIANCE_START = 2


def _create_battery():
    """Create a fresh battery device."""
    akku = Battery(
        SolarPanelBatteryParameters(
            device_id="battery1",
            capacity_wh=5000,
            initial_soc_percentage=80,
            min_soc_percentage=10,
        ),
        prediction_hours=PREDICTION_HOURS,
    )
    akku.reset()
    return akku


def _create_inverter(akku):
    """Create a fresh inverter device."""
    return Inverter(
        InverterParameters(
            device_id="inverter1",
            max_power_wh=10000,
            battery_id=akku.parameters.device_id,
        ),
        battery=akku,
    )


def _create_home_appliance():
    """Create a fresh home appliance device."""
    return HomeAppliance(
        HomeApplianceParameters(
            device_id="dishwasher1",
            consumption_wh=2000,
            duration_h=2,
            time_windows=None,
        ),
        optimization_hours=OPTIMIZATION_HOURS,
        prediction_hours=PREDICTION_HOURS,
    )


def _create_ev():
    """Create a fresh EV device."""
    eauto = Battery(
        ElectricVehicleParameters(
            device_id="ev1",
            capacity_wh=26400,
            initial_soc_percentage=10,
            min_soc_percentage=10,
        ),
        prediction_hours=PREDICTION_HOURS,
    )
    eauto.set_charge_per_hour(np.full(PREDICTION_HOURS, 1))
    return eauto


def _create_session(akku, inverter, home_appliance, eauto, config_eos):
    """Create SimulationSession with standard setup."""
    config_eos.merge_settings_from_dict(
        {"prediction": {"hours": PREDICTION_HOURS}, "optimization": {"hours": OPTIMIZATION_HOURS}}
    )

    sim = SimulationSession()
    sim.prepare(
        EnergyManagementParameters(
            pv_forecast_wh=PV_PROGNOSE_WH,
            electricity_price_per_wh=STROMPREIS_EURO_PRO_WH,
            feed_in_tariff_per_wh=EINSPEISEVERGUETUNG_EURO_PRO_WH,
            price_per_wh_battery=PREIS_EURO_PRO_WH_AKKU,
            total_load=GESAMTLAST,
        ),
        optimization_hours=OPTIMIZATION_HOURS,
        prediction_hours=PREDICTION_HOURS,
        inverter=inverter,
        ev=eauto,
        home_appliance=home_appliance,
    )
    return sim


def _create_engine(akku, inverter, home_appliance, eauto, config_eos):
    """Create EnergySimulationEngine with standard setup."""
    config_eos.merge_settings_from_dict(
        {"prediction": {"hours": PREDICTION_HOURS}, "optimization": {"hours": OPTIMIZATION_HOURS}}
    )

    sim_devices = SimulationDevices(
        battery=akku,
        ev=eauto,
        inverter=inverter,
        home_appliance=home_appliance,
    )

    sim_config = SimulationConfig(
        prediction_hours=PREDICTION_HOURS,
        optimization_hours=OPTIMIZATION_HOURS,
        start_hour=start_hour,
        load_energy_array=np.array(GESAMTLAST, float),
        pv_prediction_wh=np.array(PV_PROGNOSE_WH, float),
        elect_price_hourly=np.array(STROMPREIS_EURO_PRO_WH, float),
        elect_revenue_per_hour=np.full(PREDICTION_HOURS, EINSPEISEVERGUETUNG_EURO_PRO_WH, float),
    )

    return EnergySimulationEngine(devices=sim_devices, sim_config=sim_config)


def _set_actions(sim: SimulationSession, ac_val=1.0, dc_val=1.0, dis_val=1.0, ev_val=1.0):
    """Set action arrays on a SimulationSession instance."""
    # mypy doesn't know these are ndarray after prepare()
    sim.ac_charge_hours[start_hour] = ac_val  # type: ignore[index]
    sim.dc_charge_hours[start_hour] = dc_val  # type: ignore[index]
    sim.bat_discharge_hours[start_hour] = dis_val  # type: ignore[index]
    sim.ev_charge_hours[start_hour] = ev_val  # type: ignore[index]


def _make_action_arrays(ac_val=1.0, dc_val=1.0, dis_val=1.0, ev_val=1.0):
    """Create action arrays for the engine."""
    ac_charge = np.full(PREDICTION_HOURS, 0.0)
    dc_charge = np.full(PREDICTION_HOURS, 0.0)
    discharge = np.full(PREDICTION_HOURS, 0.0)
    ev_charge = np.full(PREDICTION_HOURS, 0.0)

    ac_charge[start_hour] = ac_val
    dc_charge[start_hour] = dc_val
    discharge[start_hour] = dis_val
    ev_charge[start_hour] = ev_val

    return ac_charge, dc_charge, discharge, ev_charge


def _compare_results(session_result, engine_result, label=""):
    """Compare all result fields between SimulationSession and engine."""
    msg_prefix = f"{label}: " if label else ""

    # Load
    np.testing.assert_array_almost_equal(
        np.array(session_result["Last_Wh_pro_Stunde"]),
        np.array(engine_result.load_wh_per_hour),
        decimal=10,
        err_msg=f"{msg_prefix}load_wh_per_hour mismatch",
    )

    # Grid feed-in
    np.testing.assert_array_almost_equal(
        np.array(session_result["Netzeinspeisung_Wh_pro_Stunde"]),
        np.array(engine_result.grid_feed_in_wh_per_hour),
        decimal=10,
        err_msg=f"{msg_prefix}grid_feed_in_wh_per_hour mismatch",
    )

    # Grid consumption
    np.testing.assert_array_almost_equal(
        np.array(session_result["Netzbezug_Wh_pro_Stunde"]),
        np.array(engine_result.grid_consumption_wh_per_hour),
        decimal=10,
        err_msg=f"{msg_prefix}grid_consumption_wh_per_hour mismatch",
    )

    # Costs per hour
    np.testing.assert_array_almost_equal(
        np.array(session_result["Kosten_Euro_pro_Stunde"]),
        np.array(engine_result.costs_per_hour),
        decimal=10,
        err_msg=f"{msg_prefix}costs_per_hour mismatch",
    )

    # Revenue per hour
    np.testing.assert_array_almost_equal(
        np.array(session_result["Einnahmen_Euro_pro_Stunde"]),
        np.array(engine_result.revenue_per_hour),
        decimal=10,
        err_msg=f"{msg_prefix}revenue_per_hour mismatch",
    )

    # Battery SoC
    np.testing.assert_array_almost_equal(
        np.array(session_result["akku_soc_pro_stunde"]),
        np.array(engine_result.battery_soc_per_hour),
        decimal=10,
        err_msg=f"{msg_prefix}battery_soc_per_hour mismatch",
    )

    # EV SoC
    np.testing.assert_array_almost_equal(
        np.array(session_result["EAuto_SoC_pro_Stunde"]),
        np.array(engine_result.ev_soc_per_hour),
        decimal=10,
        err_msg=f"{msg_prefix}ev_soc_per_hour mismatch",
    )

    # Losses per hour
    np.testing.assert_array_almost_equal(
        np.array(session_result["Verluste_Pro_Stunde"]),
        np.array(engine_result.losses_per_hour),
        decimal=10,
        err_msg=f"{msg_prefix}losses_per_hour mismatch",
    )

    # Home appliance
    np.testing.assert_array_almost_equal(
        np.array(session_result["Home_appliance_wh_per_hour"]),
        np.array(engine_result.home_appliance_wh_per_hour),
        decimal=10,
        err_msg=f"{msg_prefix}home_appliance_wh_per_hour mismatch",
    )

    # Electricity price
    np.testing.assert_array_almost_equal(
        np.array(session_result["Electricity_price"]),
        np.array(engine_result.electricity_price),
        decimal=10,
        err_msg=f"{msg_prefix}electricity_price mismatch",
    )

    # Aggregate values
    assert abs(session_result["Gesamtkosten_Euro"] - engine_result.total_costs) < 1e-10, (
        f"{msg_prefix}total_costs: {session_result['Gesamtkosten_Euro']} "
        f"vs {engine_result.total_costs}"
    )
    assert abs(session_result["Gesamteinnahmen_Euro"] - engine_result.total_revenue) < 1e-10, (
        f"{msg_prefix}total_revenue: {session_result['Gesamteinnahmen_Euro']} "
        f"vs {engine_result.total_revenue}"
    )
    assert abs(session_result["Gesamtbilanz_Euro"] - engine_result.total_balance) < 1e-10, (
        f"{msg_prefix}total_balance: {session_result['Gesamtbilanz_Euro']} "
        f"vs {engine_result.total_balance}"
    )
    assert abs(session_result["Gesamt_Verluste"] - engine_result.total_losses) < 1e-10, (
        f"{msg_prefix}total_losses: {session_result['Gesamt_Verluste']} "
        f"vs {engine_result.total_losses}"
    )


class TestEngineIdentity:
    """Verify EnergySimulationEngine produces identical results to SimulationSession."""

    def test_identity_with_all_actions(self, config_eos) -> None:
        """Engine must produce bitwise-identical results to SimulationSession."""
        # Create independent devices for each simulation
        akku_g = _create_battery()
        inv_g = _create_inverter(akku_g)
        ha_g = _create_home_appliance()
        ev_g = _create_ev()

        akku_e = _create_battery()
        inv_e = _create_inverter(akku_e)
        ha_e = _create_home_appliance()
        ev_e = _create_ev()

        # Setup SimulationSession
        sim_g = _create_session(akku_g, inv_g, ha_g, ev_g, config_eos)
        _set_actions(sim_g, ac_val=1.0, dc_val=1.0, dis_val=1.0, ev_val=1.0)
        sim_g.home_appliance_start_hour = HOME_APPLIANCE_START

        # Setup Engine
        engine = _create_engine(akku_e, inv_e, ha_e, ev_e, config_eos)
        ac_charge, dc_charge, discharge, ev_charge = _make_action_arrays(1.0, 1.0, 1.0, 1.0)

        # Run
        genetic_result = sim_g.simulate(start_hour=start_hour)
        engine_result = engine.run(
            ac_charge=ac_charge,
            dc_charge=dc_charge,
            discharge=discharge,
            ev_charge=ev_charge,
            home_appliance_start=HOME_APPLIANCE_START,
        )

        _compare_results(genetic_result, engine_result, "all_actions")

    def test_identity_no_home_appliance(self, config_eos) -> None:
        """Engine must match SimulationSession when home appliance is disabled."""
        akku_g = _create_battery()
        inv_g = _create_inverter(akku_g)
        ha_g = _create_home_appliance()
        ev_g = _create_ev()

        akku_e = _create_battery()
        inv_e = _create_inverter(akku_e)
        ha_e = _create_home_appliance()
        ev_e = _create_ev()

        sim_g = _create_session(akku_g, inv_g, ha_g, ev_g, config_eos)
        _set_actions(sim_g, ac_val=1.0, dc_val=1.0, dis_val=1.0, ev_val=1.0)
        sim_g.home_appliance_start_hour = None  # Disabled

        engine = _create_engine(akku_e, inv_e, ha_e, ev_e, config_eos)
        ac_charge, dc_charge, discharge, ev_charge = _make_action_arrays(1.0, 1.0, 1.0, 1.0)

        session_result = sim_g.simulate(start_hour=start_hour)
        engine_result = engine.run(
            ac_charge=ac_charge,
            dc_charge=dc_charge,
            discharge=discharge,
            ev_charge=ev_charge,
            home_appliance_start=None,
        )

        _compare_results(session_result, engine_result, "no_ha")

    def test_identity_all_zeros_actions(self, config_eos) -> None:
        """Engine must match SimulationSession with all-zero action arrays."""
        akku_g = _create_battery()
        inv_g = _create_inverter(akku_g)
        ha_g = _create_home_appliance()
        ev_g = _create_ev()

        akku_e = _create_battery()
        inv_e = _create_inverter(akku_e)
        ha_e = _create_home_appliance()
        ev_e = _create_ev()

        sim_g = _create_session(akku_g, inv_g, ha_g, ev_g, config_eos)
        _set_actions(sim_g, ac_val=0.0, dc_val=0.0, dis_val=0.0, ev_val=0.0)
        sim_g.home_appliance_start_hour = None

        engine = _create_engine(akku_e, inv_e, ha_e, ev_e, config_eos)
        ac_charge, dc_charge, discharge, ev_charge = _make_action_arrays(0.0, 0.0, 0.0, 0.0)

        session_result = sim_g.simulate(start_hour=start_hour)
        engine_result = engine.run(
            ac_charge=ac_charge,
            dc_charge=dc_charge,
            discharge=discharge,
            ev_charge=ev_charge,
            home_appliance_start=None,
        )

        _compare_results(session_result, engine_result, "zero_actions")

    def test_result_array_lengths(self, config_eos) -> None:
        """Verify result arrays have correct length (total_hours = end_hour - start_hour)."""
        akku_e = _create_battery()
        inv_e = _create_inverter(akku_e)
        ha_e = _create_home_appliance()
        ev_e = _create_ev()

        engine = _create_engine(akku_e, inv_e, ha_e, ev_e, config_eos)
        total_hours = len(GESAMTLAST) - start_hour

        ac_charge, dc_charge, discharge, ev_charge = _make_action_arrays()
        engine_result = engine.run(
            ac_charge=ac_charge,
            dc_charge=dc_charge,
            discharge=discharge,
            ev_charge=ev_charge,
            home_appliance_start=HOME_APPLIANCE_START,
        )

        assert len(engine_result.load_wh_per_hour) == total_hours
        assert len(engine_result.grid_feed_in_wh_per_hour) == total_hours
        assert len(engine_result.grid_consumption_wh_per_hour) == total_hours
        assert len(engine_result.costs_per_hour) == total_hours
        assert len(engine_result.revenue_per_hour) == total_hours
        assert len(engine_result.battery_soc_per_hour) == total_hours
        assert len(engine_result.ev_soc_per_hour) == total_hours
        assert len(engine_result.losses_per_hour) == total_hours
        assert len(engine_result.home_appliance_wh_per_hour) == total_hours
        assert len(engine_result.electricity_price) == total_hours

    def test_to_dict_compatibility(self, config_eos) -> None:
        """Verify engine result.to_dict() can be consumed by SimulationResult."""
        from akkudoktoreos.optimization.simulation.solution import SimulationResult

        akku_e = _create_battery()
        inv_e = _create_inverter(akku_e)
        ha_e = _create_home_appliance()
        ev_e = _create_ev()

        engine = _create_engine(akku_e, inv_e, ha_e, ev_e, config_eos)
        ac_charge, dc_charge, discharge, ev_charge = _make_action_arrays()

        engine_result = engine.run(
            ac_charge=ac_charge,
            dc_charge=dc_charge,
            discharge=discharge,
            ev_charge=ev_charge,
            home_appliance_start=HOME_APPLIANCE_START,
        )

        # Verify to_dict() produces a valid SimulationResult
        gsr = SimulationResult(**engine_result.to_dict())
        assert gsr is not None
        assert len(gsr.load_wh_per_hour) == len(GESAMTLAST) - start_hour

    def test_identity_discharge_only(self, config_eos) -> None:
        """Engine must match SimulationSession with discharge-only actions."""
        akku_g = _create_battery()
        inv_g = _create_inverter(akku_g)
        ha_g = _create_home_appliance()
        ev_g = _create_ev()

        akku_e = _create_battery()
        inv_e = _create_inverter(akku_e)
        ha_e = _create_home_appliance()
        ev_e = _create_ev()

        sim_g = _create_session(akku_g, inv_g, ha_g, ev_g, config_eos)
        _set_actions(sim_g, ac_val=0.0, dc_val=0.0, dis_val=1.0, ev_val=0.0)
        sim_g.home_appliance_start_hour = None

        engine = _create_engine(akku_e, inv_e, ha_e, ev_e, config_eos)
        ac_charge, dc_charge, discharge, ev_charge = _make_action_arrays(0.0, 0.0, 1.0, 0.0)

        session_result = sim_g.simulate(start_hour=start_hour)
        engine_result = engine.run(
            ac_charge=ac_charge,
            dc_charge=dc_charge,
            discharge=discharge,
            ev_charge=ev_charge,
            home_appliance_start=None,
        )

        _compare_results(session_result, engine_result, "discharge_only")

    def test_identity_ev_charging_only(self, config_eos) -> None:
        """Engine must match SimulationSession with EV charging only."""
        akku_g = _create_battery()
        inv_g = _create_inverter(akku_g)
        ha_g = _create_home_appliance()
        ev_g = _create_ev()

        akku_e = _create_battery()
        inv_e = _create_inverter(akku_e)
        ha_e = _create_home_appliance()
        ev_e = _create_ev()

        sim_g = _create_session(akku_g, inv_g, ha_g, ev_g, config_eos)
        _set_actions(sim_g, ac_val=0.0, dc_val=0.0, dis_val=0.0, ev_val=1.0)
        sim_g.home_appliance_start_hour = None

        engine = _create_engine(akku_e, inv_e, ha_e, ev_e, config_eos)
        ac_charge, dc_charge, discharge, ev_charge = _make_action_arrays(0.0, 0.0, 0.0, 1.0)

        session_result = sim_g.simulate(start_hour=start_hour)
        engine_result = engine.run(
            ac_charge=ac_charge,
            dc_charge=dc_charge,
            discharge=discharge,
            ev_charge=ev_charge,
            home_appliance_start=None,
        )

        _compare_results(session_result, engine_result, "ev_only")
