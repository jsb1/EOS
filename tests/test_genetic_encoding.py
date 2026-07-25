"""Unit tests for genetic/encoding.py pure functions."""

import numpy as np
import pytest

from akkudoktoreos.optimization.genetic.encoding import (
    compute_total_states,
    decode_charge_discharge,
    merge_individual,
    split_individual,
)


class TestComputeTotalStates:
    """Tests for compute_total_states pure function."""

    def test_single_charge_level_no_dc(self) -> None:
        """Single charge level without DC = 3 states."""
        assert compute_total_states([1.0], False) == 3

    def test_single_charge_level_with_dc(self) -> None:
        """Single charge level with DC = 5 states (3 + 2)."""
        assert compute_total_states([1.0], True) == 5

    def test_multiple_charge_levels_no_dc(self) -> None:
        """Two charge levels without DC = 6 states (3 * 2)."""
        assert compute_total_states([0.5, 1.0], False) == 6

    def test_multiple_charge_levels_with_dc(self) -> None:
        """Two charge levels with DC = 8 states (3 * 2 + 2)."""
        assert compute_total_states([0.5, 1.0], True) == 8

    def test_three_charge_levels(self) -> None:
        """Three charge levels without DC = 9 states."""
        assert compute_total_states([0.33, 0.66, 1.0], False) == 9

    def test_three_charge_levels_with_dc(self) -> None:
        """Three charge levels with DC = 11 states."""
        assert compute_total_states([0.33, 0.66, 1.0], True) == 11


class TestDecodeChargeDischarge:
    """Tests for decode_charge_discharge pure function."""

    def test_all_idle_no_dc(self) -> None:
        """All idle states produce zero arrays."""
        states = np.array([0, 0, 0, 0])
        ac_charge, dc_charge, discharge = decode_charge_discharge(states, [1.0], False)

        assert np.all(ac_charge == 0.0)
        assert np.all(dc_charge == 1.0)  # DC always on when not optimized
        assert np.all(discharge == 0)

    def test_all_discharge_no_dc(self) -> None:
        """All discharge states (len_bat .. 2*len_bat-1)."""
        states = np.array([1, 1, 1, 1])  # len_bat=1, discharge state = 1
        ac_charge, dc_charge, discharge = decode_charge_discharge(states, [1.0], False)

        assert np.all(ac_charge == 0.0)
        assert np.all(discharge == 1)

    def test_all_ac_charge_no_dc(self) -> None:
        """All AC charge states (2*len_bat .. 3*len_bat-1)."""
        states = np.array([2, 2, 2, 2])  # len_bat=1, AC state = 2
        ac_charge, dc_charge, discharge = decode_charge_discharge(states, [1.0], False)

        assert np.all(ac_charge == 1.0)
        assert np.all(discharge == 0)

    def test_mixed_states_single_level(self) -> None:
        """Mixed idle, discharge, AC charge with single charge level."""
        states = np.array([0, 1, 2, 0])  # idle, discharge, ac, idle
        ac_charge, dc_charge, discharge = decode_charge_discharge(states, [1.0], False)

        assert ac_charge[0] == 0.0
        assert discharge[0] == 0
        assert ac_charge[1] == 0.0
        assert discharge[1] == 1
        assert ac_charge[2] == 1.0
        assert discharge[2] == 0
        assert ac_charge[3] == 0.0
        assert discharge[3] == 0

    def test_multiple_charge_levels(self) -> None:
        """Multiple charge levels map correctly to AC charge rates."""
        charge_levels = [0.5, 1.0]  # len_bat = 2
        # AC states: 4 (index 0 → 0.5), 5 (index 1 → 1.0)
        states = np.array([4, 5, 0, 2])  # ac(0.5), ac(1.0), idle, idle
        ac_charge, dc_charge, discharge = decode_charge_discharge(states, charge_levels, False)

        assert ac_charge[0] == pytest.approx(0.5)
        assert ac_charge[1] == pytest.approx(1.0)
        assert ac_charge[2] == 0.0
        assert ac_charge[3] == 0.0

    def test_dc_charge_optimized_allowed(self) -> None:
        """DC charge allowed state (3*len_bat + 1) enables DC."""
        states = np.array([3, 4])  # len_bat=1: dc_not_allowed=3, dc_allowed=4
        ac_charge, dc_charge, discharge = decode_charge_discharge(states, [1.0], True)

        assert dc_charge[0] == 0  # DC not allowed
        assert dc_charge[1] == 1  # DC allowed

    def test_dc_charge_optimized_disabled(self) -> None:
        """DC charge not allowed state disables DC."""
        states = np.array([0, 3])  # idle + dc_not_allowed
        ac_charge, dc_charge, discharge = decode_charge_discharge(states, [1.0], True)

        assert dc_charge[0] == 0  # Not DC-allowed state → 0
        assert dc_charge[1] == 0  # DC not allowed state → 0

    def test_dc_not_optimized_all_ones(self) -> None:
        """When DC is not optimized, dc_charge is all ones."""
        states = np.array([0, 1, 2])
        ac_charge, dc_charge, discharge = decode_charge_discharge(states, [1.0], False)

        assert np.all(dc_charge == 1.0)

    def test_discharge_with_multiple_levels(self) -> None:
        """Discharge states with multiple charge levels (2..3 for len_bat=2)."""
        charge_levels = [0.5, 1.0]  # len_bat = 2
        # Discharge states: 2, 3
        states = np.array([2, 3, 0])
        ac_charge, dc_charge, discharge = decode_charge_discharge(states, charge_levels, False)

        assert discharge[0] == 1
        assert discharge[1] == 1
        assert discharge[2] == 0
        assert np.all(ac_charge == 0.0)


class TestSplitIndividual:
    """Tests for split_individual pure function."""

    def test_basic_split_no_ev_no_ha(self) -> None:
        """Split individual without EV or home appliance."""
        individual = [0, 1, 2, 0, 1]
        discharge, ev_charge, ha_start = split_individual(
            individual, prediction_hours=5, optimize_ev=False, optimize_home_appliance=False
        )

        assert np.array_equal(discharge, np.array([0, 1, 2, 0, 1]))
        assert ev_charge is None
        assert ha_start is None

    def test_split_with_ev(self) -> None:
        """Split individual with EV charge indices."""
        individual = [0, 1, 2, 0, 1, 0, 1, 0, 1, 0]
        discharge, ev_charge, ha_start = split_individual(
            individual, prediction_hours=5, optimize_ev=True, optimize_home_appliance=False
        )

        assert np.array_equal(discharge, np.array([0, 1, 2, 0, 1]))
        assert np.array_equal(ev_charge, np.array([0, 1, 0, 1, 0]))  # type: ignore[arg-type]
        assert ha_start is None

    def test_split_with_home_appliance(self) -> None:
        """Split individual with home appliance start time."""
        individual = [0, 1, 2, 10]  # 3 discharge states + 1 HA start
        discharge, ev_charge, ha_start = split_individual(
            individual, prediction_hours=3, optimize_ev=False, optimize_home_appliance=True
        )

        assert np.array_equal(discharge, np.array([0, 1, 2]))
        assert ev_charge is None
        assert ha_start == 10

    def test_split_with_ev_and_ha(self) -> None:
        """Split individual with both EV and home appliance."""
        individual = [0, 1, 2, 0, 1, 0, 15]  # 3 discharge + 3 EV + 1 HA
        discharge, ev_charge, ha_start = split_individual(
            individual, prediction_hours=3, optimize_ev=True, optimize_home_appliance=True
        )

        assert np.array_equal(discharge, np.array([0, 1, 2]))
        assert np.array_equal(ev_charge, np.array([0, 1, 0]))  # type: ignore[arg-type]
        assert ha_start == 15

    def test_split_returns_int_dtype(self) -> None:
        """Discharge array should be int dtype."""
        individual = [0, 1, 2]
        discharge, _, _ = split_individual(
            individual, prediction_hours=3, optimize_ev=False, optimize_home_appliance=False
        )
        assert discharge.dtype == np.int64 or discharge.dtype == np.int32


class TestMergeIndividual:
    """Tests for merge_individual pure function."""

    def test_basic_merge_no_ev_no_ha(self) -> None:
        """Merge without EV or home appliance."""
        discharge = np.array([0, 1, 2, 0])
        individual = merge_individual(
            discharge, None, None, prediction_hours=4, optimize_ev=False, optimize_home_appliance=False
        )

        assert individual == [0, 1, 2, 0]

    def test_merge_with_ev(self) -> None:
        """Merge with EV charge indices."""
        discharge = np.array([0, 1, 2])
        ev_charge = np.array([1, 0, 1])
        individual = merge_individual(
            discharge, ev_charge, None, prediction_hours=3, optimize_ev=True, optimize_home_appliance=False
        )

        assert individual == [0, 1, 2, 1, 0, 1]

    def test_merge_with_ev_none_data(self) -> None:
        """Merge with optimize_ev=True but ev_charge_hours_index=None → zeros."""
        discharge = np.array([0, 1])
        individual = merge_individual(
            discharge, None, None, prediction_hours=2, optimize_ev=True, optimize_home_appliance=False
        )

        assert individual == [0, 1, 0, 0]

    def test_merge_with_home_appliance(self) -> None:
        """Merge with home appliance start time."""
        discharge = np.array([0, 1, 2])
        individual = merge_individual(
            discharge, None, 10, prediction_hours=3, optimize_ev=False, optimize_home_appliance=True
        )

        assert individual == [0, 1, 2, 10]

    def test_merge_with_home_appliance_none(self) -> None:
        """Merge with optimize_home_appliance=True but washingstart_int=None → 0."""
        discharge = np.array([0, 1])
        individual = merge_individual(
            discharge, None, None, prediction_hours=2, optimize_ev=False, optimize_home_appliance=True
        )

        assert individual == [0, 1, 0]

    def test_merge_with_ev_and_ha(self) -> None:
        """Merge with both EV and home appliance."""
        discharge = np.array([0, 1, 2])
        ev_charge = np.array([1, 0, 1])
        individual = merge_individual(
            discharge, ev_charge, 15, prediction_hours=3, optimize_ev=True, optimize_home_appliance=True
        )

        assert individual == [0, 1, 2, 1, 0, 1, 15]

    def test_split_merge_roundtrip(self) -> None:
        """Split then merge should produce the original individual."""
        original = [2, 1, 0, 1, 0, 1, 0, 1, 12]
        prediction_hours = 4

        # Split
        discharge, ev_charge, ha_start = split_individual(
            original, prediction_hours=prediction_hours, optimize_ev=True, optimize_home_appliance=True
        )

        # Merge
        merged = merge_individual(
            discharge, ev_charge, ha_start, prediction_hours=prediction_hours,
            optimize_ev=True, optimize_home_appliance=True,
        )

        assert merged == original

    def test_split_merge_roundtrip_no_ev(self) -> None:
        """Roundtrip without EV."""
        original = [1, 0, 2, 8]
        prediction_hours = 3

        discharge, ev_charge, ha_start = split_individual(
            original, prediction_hours=prediction_hours, optimize_ev=False, optimize_home_appliance=True
        )

        merged = merge_individual(
            discharge, ev_charge, ha_start, prediction_hours=prediction_hours,
            optimize_ev=False, optimize_home_appliance=True,
        )

        assert merged == original
