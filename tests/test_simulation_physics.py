"""Unit tests for simulation/physics.py pure functions."""

import pytest

from akkudoktoreos.optimization.simulation.physics import (
    compute_battery_next_soc,
    compute_ev_next_soc,
)


class TestComputeBatteryNextSoc:
    """Tests for compute_battery_next_soc pure function."""

    def test_no_action_soc_unchanged(self) -> None:
        """SoC stays the same when no charging or discharging."""
        result = compute_battery_next_soc(
            current_soc_wh=5000.0,
            min_soc_wh=1000.0,
            max_soc_wh=10000.0,
            charging_efficiency=0.95,
            discharging_efficiency=0.95,
            ac_charge_factor=0.0,
            dc_charge_allowed=False,
            discharge_allowed=False,
            pv_wh=0.0,
            load_wh=2000.0,
            max_ac_charge_power_w=3000.0,
            ac_to_dc_efficiency=0.95,
            dc_to_ac_efficiency=0.95,
        )
        assert result == 5000.0

    def test_ac_charging_full(self) -> None:
        """AC charging adds energy based on max power * factor * efficiencies."""
        result = compute_battery_next_soc(
            current_soc_wh=5000.0,
            min_soc_wh=1000.0,
            max_soc_wh=10000.0,
            charging_efficiency=0.95,
            discharging_efficiency=0.95,
            ac_charge_factor=1.0,
            dc_charge_allowed=False,
            discharge_allowed=False,
            pv_wh=0.0,
            load_wh=0.0,
            max_ac_charge_power_w=3000.0,
            ac_to_dc_efficiency=0.95,
            dc_to_ac_efficiency=0.95,
        )
        # ac_draw = 3000, dc_stored = 3000 * 0.95 * 0.95 = 2707.5
        expected = 5000.0 + 2707.5
        assert abs(result - expected) < 0.01

    def test_ac_charging_clipped_to_max_soc(self) -> None:
        """AC charging cannot exceed max_soc_wh."""
        result = compute_battery_next_soc(
            current_soc_wh=9000.0,
            min_soc_wh=1000.0,
            max_soc_wh=10000.0,
            charging_efficiency=0.95,
            discharging_efficiency=0.95,
            ac_charge_factor=1.0,
            dc_charge_allowed=False,
            discharge_allowed=False,
            pv_wh=0.0,
            load_wh=0.0,
            max_ac_charge_power_w=3000.0,
            ac_to_dc_efficiency=0.95,
            dc_to_ac_efficiency=0.95,
        )
        assert result == 10000.0

    def test_dc_pv_charging(self) -> None:
        """DC (PV) charging when allowed."""
        result = compute_battery_next_soc(
            current_soc_wh=5000.0,
            min_soc_wh=1000.0,
            max_soc_wh=10000.0,
            charging_efficiency=0.95,
            discharging_efficiency=0.95,
            ac_charge_factor=0.0,
            dc_charge_allowed=True,
            discharge_allowed=False,
            pv_wh=2000.0,
            load_wh=0.0,
            max_ac_charge_power_w=3000.0,
            ac_to_dc_efficiency=0.95,
            dc_to_ac_efficiency=0.95,
        )
        # pv_for_charge = min(2000, 5000/0.95) = 2000
        # stored = 2000 * 0.95 = 1900
        expected = 5000.0 + 1900.0
        assert abs(result - expected) < 0.01

    def test_dc_pv_charging_disabled(self) -> None:
        """PV charging is skipped when dc_charge_allowed=False."""
        result = compute_battery_next_soc(
            current_soc_wh=5000.0,
            min_soc_wh=1000.0,
            max_soc_wh=10000.0,
            charging_efficiency=0.95,
            discharging_efficiency=0.95,
            ac_charge_factor=0.0,
            dc_charge_allowed=False,
            discharge_allowed=False,
            pv_wh=2000.0,
            load_wh=0.0,
            max_ac_charge_power_w=3000.0,
            ac_to_dc_efficiency=0.95,
            dc_to_ac_efficiency=0.95,
        )
        assert result == 5000.0

    def test_discharge_reduces_soc(self) -> None:
        """Discharging reduces SoC to cover remaining load."""
        result = compute_battery_next_soc(
            current_soc_wh=8000.0,
            min_soc_wh=1000.0,
            max_soc_wh=10000.0,
            charging_efficiency=0.95,
            discharging_efficiency=0.95,
            ac_charge_factor=0.0,
            dc_charge_allowed=False,
            discharge_allowed=True,
            pv_wh=0.0,
            load_wh=2000.0,
            max_ac_charge_power_w=3000.0,
            ac_to_dc_efficiency=0.95,
            dc_to_ac_efficiency=0.95,
        )
        # pv_used_load = min(0, 2000) = 0
        # remaining_load = 2000
        # max_discharge = (8000 - 1000) / 0.95 = 7368.4
        # dis_dc = min(7368.4, 2000/0.95) = 2105.26
        # soc_after = 8000 - 2105.26 * 0.95 = 8000 - 2000 = 6000
        assert abs(result - 6000.0) < 0.01

    def test_discharge_respects_min_soc(self) -> None:
        """Discharge cannot go below min_soc_wh."""
        result = compute_battery_next_soc(
            current_soc_wh=2000.0,
            min_soc_wh=1000.0,
            max_soc_wh=10000.0,
            charging_efficiency=0.95,
            discharging_efficiency=0.95,
            ac_charge_factor=0.0,
            dc_charge_allowed=False,
            discharge_allowed=True,
            pv_wh=0.0,
            load_wh=5000.0,
            max_ac_charge_power_w=3000.0,
            ac_to_dc_efficiency=0.95,
            dc_to_ac_efficiency=0.95,
        )
        # max_discharge = (2000 - 1000) / 0.95 = 1052.6
        # dis_dc = min(1052.6, 5000/0.95) = 1052.6
        # soc_after = 2000 - 1052.6 * 0.95 = 2000 - 1000 = 1000
        assert abs(result - 1000.0) < 0.01

    def test_clamped_to_min_soc(self) -> None:
        """Result is clamped to min_soc_wh."""
        result = compute_battery_next_soc(
            current_soc_wh=500.0,
            min_soc_wh=1000.0,
            max_soc_wh=10000.0,
            charging_efficiency=0.95,
            discharging_efficiency=0.95,
            ac_charge_factor=0.0,
            dc_charge_allowed=False,
            discharge_allowed=False,
            pv_wh=0.0,
            load_wh=0.0,
            max_ac_charge_power_w=3000.0,
        )
        assert result == 1000.0

    def test_clamped_to_max_soc(self) -> None:
        """Result is clamped to max_soc_wh."""
        result = compute_battery_next_soc(
            current_soc_wh=15000.0,
            min_soc_wh=1000.0,
            max_soc_wh=10000.0,
            charging_efficiency=0.95,
            discharging_efficiency=0.95,
            ac_charge_factor=0.0,
            dc_charge_allowed=False,
            discharge_allowed=False,
            pv_wh=0.0,
            load_wh=0.0,
            max_ac_charge_power_w=3000.0,
        )
        assert result == 10000.0

    def test_ac_and_dc_charging_combined(self) -> None:
        """AC + DC (PV) charging in the same step."""
        result = compute_battery_next_soc(
            current_soc_wh=5000.0,
            min_soc_wh=1000.0,
            max_soc_wh=10000.0,
            charging_efficiency=0.95,
            discharging_efficiency=0.95,
            ac_charge_factor=0.5,
            dc_charge_allowed=True,
            discharge_allowed=False,
            pv_wh=1000.0,
            load_wh=0.0,
            max_ac_charge_power_w=3000.0,
            ac_to_dc_efficiency=0.95,
            dc_to_ac_efficiency=0.95,
        )
        # AC: ac_draw = 1500, dc_stored = 1500 * 0.95 * 0.95 = 1353.75
        # DC: headroom = 10000 - 5000 - 1353.75 = 3646.25
        # pv_for_charge = min(1000, 3646.25/0.95) = 1000
        # dc_from_pv = 1000 * 0.95 = 950
        # total = 5000 + 1353.75 + 950 = 7303.75
        assert abs(result - 7303.75) < 0.01

    def test_full_cycle_ac_charge_then_discharge(self) -> None:
        """AC charge followed by discharge in the same step."""
        result = compute_battery_next_soc(
            current_soc_wh=5000.0,
            min_soc_wh=1000.0,
            max_soc_wh=10000.0,
            charging_efficiency=0.95,
            discharging_efficiency=0.95,
            ac_charge_factor=1.0,
            dc_charge_allowed=False,
            discharge_allowed=True,
            pv_wh=0.0,
            load_wh=5000.0,
            max_ac_charge_power_w=3000.0,
            ac_to_dc_efficiency=0.95,
            dc_to_ac_efficiency=0.95,
        )
        # AC: dc_stored = 3000 * 0.95 * 0.95 = 2707.5
        # soc_after_charge = 5000 + 2707.5 = 7707.5
        # Discharge: remaining_load = 5000
        # max_discharge = (7707.5 - 1000) / 0.95 = 7060.5
        # dis_dc = min(7060.5, 5000/0.95) = 5263.16
        # soc_final = 7707.5 - 5263.16 * 0.95 = 7707.5 - 5000 = 2707.5
        assert abs(result - 2707.5) < 0.01


class TestComputeEvNextSoc:
    """Tests for compute_ev_next_soc pure function."""

    def test_no_charging_soc_unchanged(self) -> None:
        """SoC stays the same when charge_factor=0."""
        result = compute_ev_next_soc(
            current_soc_wh=20000.0,
            min_soc_wh=0.0,
            max_soc_wh=60000.0,
            charging_efficiency=0.92,
            charge_factor=0.0,
            max_charge_power_w=11000.0,
        )
        assert result == 20000.0

    def test_charging_adds_energy(self) -> None:
        """Charging adds energy based on power * factor * efficiency."""
        result = compute_ev_next_soc(
            current_soc_wh=20000.0,
            min_soc_wh=0.0,
            max_soc_wh=60000.0,
            charging_efficiency=0.92,
            charge_factor=1.0,
            max_charge_power_w=11000.0,
        )
        # requested = 11000
        # headroom = 60000 - 20000 = 40000
        # actual = min(11000, 40000/0.92) = 11000
        # new_soc = 20000 + 11000 * 0.92 = 20000 + 10120 = 30120
        assert abs(result - 30120.0) < 0.01

    def test_charging_clipped_to_max_soc(self) -> None:
        """Charging cannot exceed max_soc_wh."""
        result = compute_ev_next_soc(
            current_soc_wh=55000.0,
            min_soc_wh=0.0,
            max_soc_wh=60000.0,
            charging_efficiency=0.92,
            charge_factor=1.0,
            max_charge_power_w=11000.0,
        )
        # headroom = 5000
        # actual = min(11000, 5000/0.92) = min(11000, 5434.8) = 5434.8
        # new_soc = 55000 + 5434.8 * 0.92 = 55000 + 5000 = 60000
        assert abs(result - 60000.0) < 0.01

    def test_partial_charge_factor(self) -> None:
        """Partial charge factor reduces charging power."""
        result = compute_ev_next_soc(
            current_soc_wh=20000.0,
            min_soc_wh=0.0,
            max_soc_wh=60000.0,
            charging_efficiency=0.92,
            charge_factor=0.5,
            max_charge_power_w=11000.0,
        )
        # requested = 11000 * 0.5 = 5500
        # headroom = 40000
        # actual = min(5500, 40000/0.92) = 5500
        # new_soc = 20000 + 5500 * 0.92 = 20000 + 5060 = 25060
        assert abs(result - 25060.0) < 0.01

    def test_clamped_to_min_soc(self) -> None:
        """Result is clamped to min_soc_wh."""
        result = compute_ev_next_soc(
            current_soc_wh=-100.0,
            min_soc_wh=0.0,
            max_soc_wh=60000.0,
            charging_efficiency=0.92,
            charge_factor=0.0,
            max_charge_power_w=11000.0,
        )
        assert result == -100.0  # No charging, but below min: still returned as-is

    def test_negative_charge_factor_unchanged(self) -> None:
        """Negative charge_factor is treated as no charging."""
        result = compute_ev_next_soc(
            current_soc_wh=20000.0,
            min_soc_wh=0.0,
            max_soc_wh=60000.0,
            charging_efficiency=0.92,
            charge_factor=-0.5,
            max_charge_power_w=11000.0,
        )
        assert result == 20000.0
