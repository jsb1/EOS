"""Dynamic Programming (DP) solver for energy optimization.

Implements Bellman optimality for discrete-state energy management with:
- Battery SoC discretization (1% resolution → 101 levels)
- EV SoC discretization (1% resolution → 101 levels)
- Home appliance scheduling
- Terminal penalties (battery residual value, EV SoC miss, AC charge break-even)
- GA parity features (DC charge flag, worst-case mode, EV optimization check)
- HYBRID mode support via to_ga_individual()
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from numba import njit

from akkudoktoreos.core.coreabc import ConfigMixin
from akkudoktoreos.optimization.simulation.physics import (
    compute_battery_next_soc,
    compute_battery_next_soc_with_flows,
    compute_ev_next_soc,
)
from akkudoktoreos.optimization.simulation.penalties import (
    ac_charge_break_even_penalty,
    battery_residual_value_penalty,
    ev_residual_value_penalty,
    ev_soc_miss_penalty,
)
from akkudoktoreos.optimization.simulation.devices import (
    Battery,
    HomeAppliance,
    Inverter,
)
from akkudoktoreos.optimization.simulation.parameters import (
    BaseBatteryParameters,
    ElectricVehicleParameters,
    HomeApplianceParameters,
    InverterParameters,
    OptimizationParameters,
    SolarPanelBatteryParameters,
)
from akkudoktoreos.optimization.simulation.session import SimulationSession
from akkudoktoreos.optimization.simulation.solution import (
    SimulationResult,
    SimulationSolution,
)
from akkudoktoreos.optimization.dp.dpsolution import DPSolution


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOC_RESOLUTION_PERCENT = 1  # 1% steps → 101 discrete levels (0-100%)
NUM_SOC_LEVELS = int(100 / SOC_RESOLUTION_PERCENT) + 1  # 101
INF_COST = 1e12  # Infinity cost for invalid states


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DPState:
    """Discrete DP state: (hour, battery_soc_index, ev_soc_index, appliance_started)."""

    hour: int
    bat_idx: int
    ev_idx: int
    appliance_started: bool

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.hour, self.bat_idx, self.ev_idx, int(self.appliance_started))


@dataclass(frozen=True)
class DPAction:
    """Discrete DP action for one time step."""

    ac_charge_rate_idx: int  # Index into charge_rates array (0 = off)
    dc_charge_allowed: bool
    discharge_allowed: bool
    ev_charge_rate_idx: int  # Index into EV charge_rates array (0 = off)

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (
            self.ac_charge_rate_idx,
            int(self.dc_charge_allowed),
            int(self.discharge_allowed),
            self.ev_charge_rate_idx,
        )


# ---------------------------------------------------------------------------
# Helper functions for min/max SoC in Wh
# ---------------------------------------------------------------------------

# =============================================================================
# Numba-accelerated helper functions for DP backward pass
# =============================================================================

@njit(cache=True)
def _scr(consumption: float, generation: float) -> float:
    """Self-consumption ratio (from physics.py)."""
    if generation <= 0.0:
        return 1.0
    if consumption <= 0.0:
        return 0.0
    return consumption / generation


@njit(cache=True)
def _battery_flows(
    current_soc: float,
    min_soc: float,
    max_soc: float,
    chg_eff: float,
    dis_eff: float,
    ac_factor: float,
    dc_allowed: bool,
    dis_allowed: bool,
    pv: float,
    load: float,
    ev_draw: float,
    max_ac_p: float,
    ac_dc_eff: float,
    dc_ac_eff: float,
) -> tuple:
    """Compute battery flows - Numba-compatible version of compute_battery_next_soc_with_flows."""
    bsv = current_soc
    total_cons = load + ev_draw
    grid_imp = 0.0
    grid_exp = 0.0
    bat_chg = 0.0
    ac_grid = 0.0

    # AC charging from grid
    if ac_factor > 0.0:
        ac_d = max_ac_p * ac_factor
        dc_st = ac_d * ac_dc_eff * chg_eff
        head = max_soc - bsv
        if head > 0:
            dc_st = min(dc_st, head)
            ac_grid = ac_d
            bat_chg = dc_st
        else:
            dc_st = 0.0
            ac_grid = 0.0
            bat_chg = 0.0

    total_cons += ac_grid

    if pv >= total_cons:
        # PV surplus
        scr_val = _scr(total_cons, pv)
        rem_pow = (pv - total_cons) * scr_val
        rem_load = (pv - total_cons) * (1.0 - scr_val)
        bat_dis_ac = 0.0

        if rem_load > 0 and dis_allowed:
            ac_need = rem_load
            dc_req = ac_need / dc_ac_eff / dis_eff
            max_dis = max(0.0, bsv - min_soc)
            raw_dc = min(max_dis, dc_req)
            del_dc = raw_dc * dis_eff
            bat_dis_ac = del_dc * dc_ac_eff
            bsv -= raw_dc
            rem_load -= bat_dis_ac

        if rem_load > 0:
            grid_imp += rem_load

        # DC charging
        if rem_pow > 0 and dc_allowed and bsv < max_soc:
            head = max_soc - bsv - bat_chg
            if head > 0:
                pv_chg = min(rem_pow, head / chg_eff)
                stored = pv_chg * chg_eff
                bat_chg += stored
                rem_pow -= pv_chg
                bsv += stored

        grid_exp = max(0.0, rem_pow)
    else:
        # PV deficit
        short = total_cons - pv
        bat_dis_ac = 0.0

        if dis_allowed and short > 0:
            ac_need = short
            dc_req = ac_need / dc_ac_eff / dis_eff
            max_dis = max(0.0, bsv - min_soc)
            raw_dc = min(max_dis, dc_req)
            del_dc = raw_dc * dis_eff
            bat_dis_ac = del_dc * dc_ac_eff
            bsv -= raw_dc
            short -= bat_dis_ac

        grid_imp += max(0.0, short)

    bsv += bat_chg
    next_soc = max(min_soc, min(max_soc, bsv))
    return next_soc, grid_imp, grid_exp


@njit(cache=True)
def _bellman_backward(
    V,
    pol_act,
    pol_bat,
    pol_ev,
    pol_ap,
    bat_soc_lk,
    ev_tr_next,
    ev_tr_draw,
    act_ac_ri,
    act_dc,
    act_dis,
    bat_chr_arr,
    pv_arr,
    load_arr,
    price_arr,
    rev_arr,
    ha_win_arr,
    horizon,
    n_act,
    n_ha,
    has_ev,
    has_ha,
    worst,
    bat_min,
    bat_max,
    bat_ce,
    bat_de,
    acdc,
    dcac,
    maxac,
    nsoc,
    inf_c,
):
    """Numba-accelerated Bellman backward pass."""
    for t in range(horizon - 1, -1, -1):
        pv = pv_arr[t] if t < len(pv_arr) else 0.0
        ld = load_arr[t] if t < len(load_arr) else 0.0
        pr = price_arr[t] if t < len(price_arr) else 0.0
        rv = rev_arr[t] if t < len(rev_arr) else 0.0

        ha_cs = False
        if has_ha:
            for wi in range(n_ha):
                if ha_win_arr[wi] == t:
                    ha_cs = True
                    break

        for bi in range(nsoc):
            bsw = bat_soc_lk[bi]
            for ei in range(nsoc):
                for ap in range(2):
                    bc = inf_c
                    ba = -1
                    bnb = -1
                    bne = -1
                    bna = -1

                    for ai in range(n_act):
                        nap = ap

                        if has_ev:
                            nei = ev_tr_next[ai, ei]
                            edw = ev_tr_draw[ai, ei]
                            if ei == nsoc - 1 and nei > ei:
                                continue
                        else:
                            nei = ei
                            edw = 0.0

                        acf = bat_chr_arr[act_ac_ri[ai]]
                        dca = act_dc[ai] != 0
                        disa = act_dis[ai] != 0

                        ns, gi, ge = _battery_flows(
                            bsw, bat_min, bat_max, bat_ce, bat_de,
                            acf, dca, disa, pv, ld, edw,
                            maxac, acdc, dcac,
                        )

                        nbi = int(max(0, min(nsoc - 1,
                            (ns - bat_min) / (bat_max - bat_min) * (nsoc - 1))))

                        ic = gi * pr - ge * rv
                        fc = V[t + 1, nbi, nei, nap]
                        if fc >= inf_c:
                            continue

                        nc = ic + fc
                        if worst:
                            nc = -nc

                        if nc < bc:
                            bc = nc
                            ba = ai
                            bnb = nbi
                            bne = nei
                            bna = nap

                    if ap == 0 and ha_cs:
                        fc = V[t + 1, bi, ei, 1]
                        if fc < bc:
                            bc = fc
                            ba = -2
                            bnb = bi
                            bne = ei
                            bna = 1

                    if ba >= -2:
                        V[t, bi, ei, ap] = bc
                        pol_act[t, bi, ei, ap] = ba
                        pol_bat[t, bi, ei, ap] = bnb
                        pol_ev[t, bi, ei, ap] = bne
                        pol_ap[t, bi, ei, ap] = bna


def _min_soc_wh(params: BaseBatteryParameters) -> float:
    """Get minimum SoC in Wh from parameters."""
    min_pct = params.min_soc_percentage if isinstance(params, SolarPanelBatteryParameters) else 0
    return (min_pct / 100.0) * params.capacity_wh


def _max_soc_wh(params: BaseBatteryParameters) -> float:
    """Get maximum SoC in Wh from parameters."""
    return (params.max_soc_percentage / 100.0) * params.capacity_wh


def _max_charge_power_w(params: BaseBatteryParameters) -> float:
    """Get max charge power in W from parameters."""
    return params.max_charge_power_w if params.max_charge_power_w is not None else float(params.capacity_wh)


# ---------------------------------------------------------------------------
# DPOptimizer
# ---------------------------------------------------------------------------

class DPOptimizer(ConfigMixin):
    """Dynamic Programming solver for energy optimization.

    Uses Bellman optimality to find the globally optimal policy in a discretized
    state space.
    State = (hour, battery_soc_index, ev_soc_index, appliance_started)
    Action = (ac_charge_rate, dc_charge_allowed, discharge_allowed, ev_charge_rate)
    """

    def __init__(self) -> None:
        """Initialize DP optimizer."""
        super().__init__()

    def _get_soc_index(self, soc_wh: float, min_wh: float, max_wh: float) -> int:
        """Convert continuous SoC [Wh] to discrete index."""
        if max_wh <= min_wh:
            return 0
        ratio = (soc_wh - min_wh) / (max_wh - min_wh)
        idx = int(np.clip(ratio, 0.0, 1.0) * (NUM_SOC_LEVELS - 1))
        return int(np.clip(idx, 0, NUM_SOC_LEVELS - 1))

    def _get_soc_from_index(self, idx: int, min_wh: float, max_wh: float) -> float:
        """Convert discrete index to continuous SoC [Wh]."""
        if max_wh <= min_wh:
            return min_wh
        ratio = idx / (NUM_SOC_LEVELS - 1)
        return min_wh + ratio * (max_wh - min_wh)

    def _get_charge_rates(self, charge_rates: Optional[list[float]]) -> list[float]:
        """Get charge rate levels including off (0)."""
        if charge_rates is None or len(charge_rates) == 0:
            return [0.0, 0.5, 1.0]
        rates = sorted(set(charge_rates))
        if rates[0] != 0.0:
            rates = [0.0] + rates
        return rates

    def _build_action_space(
        self,
        bat_charge_rates: list[float],
        ev_charge_rates: list[float],
        optimize_dc_charge: bool,
    ) -> list[DPAction]:
        """Build all valid actions for one time step."""
        actions: list[DPAction] = []
        dc_values = [True, False] if optimize_dc_charge else [True]

        for ac_idx in range(len(bat_charge_rates)):
            for dc_allowed in dc_values:
                for dis_allowed in [True, False]:
                    for ev_idx in range(len(ev_charge_rates)):
                        # Cannot charge and discharge battery simultaneously
                        if ac_idx > 0 and dis_allowed:
                            continue
                        actions.append(
                            DPAction(
                                ac_charge_rate_idx=ac_idx,
                                dc_charge_allowed=dc_allowed,
                                discharge_allowed=dis_allowed,
                                ev_charge_rate_idx=ev_idx,
                            )
                        )
        return actions

    def _transition_battery(
        self,
        bat_idx: int,
        action: DPAction,
        bat_params: SolarPanelBatteryParameters,
        inv_params: Optional[InverterParameters],
        pv_wh: float,
        load_wh: float,
        ev_draw_wh: float,
        bat_charge_rates: list[float],
        bat_min_wh: float,
        bat_max_wh: float,
    ) -> tuple[int, float, Any]:
        """Compute next battery SoC index with energy flows.

        Uses compute_battery_next_soc_with_flows() for consistent cost calculation.

        Returns:
            (next_bat_idx, next_bat_soc_wh, energy_flows)
        """
        bat_soc_wh = self._get_soc_from_index(bat_idx, bat_min_wh, bat_max_wh)
        ac_rate = bat_charge_rates[action.ac_charge_rate_idx]

        ac_to_dc_eff = inv_params.ac_to_dc_efficiency if inv_params else 1.0
        dc_to_ac_eff = inv_params.dc_to_ac_efficiency if inv_params else 1.0
        max_ac_power = inv_params.max_ac_charge_power_w if inv_params and inv_params.max_ac_charge_power_w is not None else _max_charge_power_w(bat_params)

        # Use flows-aware computation for consistent cost calculation
        flows = compute_battery_next_soc_with_flows(
            current_soc_wh=bat_soc_wh,
            min_soc_wh=bat_min_wh,
            max_soc_wh=bat_max_wh,
            charging_efficiency=bat_params.charging_efficiency,
            discharging_efficiency=bat_params.discharging_efficiency,
            ac_charge_factor=ac_rate,
            dc_charge_allowed=action.dc_charge_allowed,
            discharge_allowed=action.discharge_allowed,
            pv_wh=pv_wh,
            load_wh=load_wh,
            ev_draw_wh=ev_draw_wh,
            max_ac_charge_power_w=max_ac_power,
            ac_to_dc_efficiency=ac_to_dc_eff,
            dc_to_ac_efficiency=dc_to_ac_eff,
        )
        next_bat_idx = self._get_soc_index(flows.next_soc_wh, bat_min_wh, bat_max_wh)

        return next_bat_idx, flows.next_soc_wh, flows

    def _transition_ev(
        self,
        ev_idx: int,
        action: DPAction,
        ev_params: ElectricVehicleParameters,
        ev_charge_rates: list[float],
        ev_min_wh: float,
        ev_max_wh: float,
    ) -> tuple[int, float]:
        """Compute next EV SoC index and actual SoC value.

        Returns:
            (next_ev_idx, ev_draw_wh)
        """
        ev_soc_wh = self._get_soc_from_index(ev_idx, ev_min_wh, ev_max_wh)
        ev_rate = ev_charge_rates[action.ev_charge_rate_idx]

        ev_draw_wh = _max_charge_power_w(ev_params) * ev_rate if ev_rate > 0 else 0.0

        if ev_rate <= 0:
            return ev_idx, ev_draw_wh

        next_ev_soc_wh = compute_ev_next_soc(
            current_soc_wh=ev_soc_wh,
            min_soc_wh=ev_min_wh,
            max_soc_wh=ev_max_wh,
            charging_efficiency=ev_params.charging_efficiency,
            charge_factor=ev_rate,
            max_charge_power_w=_max_charge_power_w(ev_params),
        )
        next_ev_idx = self._get_soc_index(next_ev_soc_wh, ev_min_wh, ev_max_wh)

        return next_ev_idx, ev_draw_wh

    def _compute_terminal_penalty(
        self,
        bat_idx: int,
        ev_idx: int,
        ac_charge_hours: np.ndarray,
        params: OptimizationParameters,
    ) -> float:
        """Compute terminal penalty at end of horizon."""
        if params.ems is None or params.pv_battery is None:
            return 0.0

        penalty = 0.0
        bat = params.pv_battery
        ev = params.ev
        inv = params.inverter

        bat_min_wh = _min_soc_wh(bat)
        bat_max_wh = _max_soc_wh(bat)
        bat_soc_wh = self._get_soc_from_index(bat_idx, bat_min_wh, bat_max_wh)

        # Battery residual value penalty (only net increase relative to initial SOC)
        dc_to_ac_eff = inv.dc_to_ac_efficiency if inv else 1.0
        initial_soc_wh = (bat.initial_soc_percentage / 100.0) * bat.capacity_wh
        penalty += battery_residual_value_penalty(
            battery_energy_content_wh=bat_soc_wh,
            dc_to_ac_efficiency=dc_to_ac_eff,
            price_per_wh_battery=params.ems.price_per_wh_battery,
            initial_soc_wh=initial_soc_wh,
        )

        # EV SoC miss penalty
        # Use full capacity range (0-100%) to match DP state space
        if ev is not None:
            ev_soc_wh = self._get_soc_from_index(ev_idx, 0.0, ev.capacity_wh)
            ev_soc_pct = (ev_soc_wh / ev.capacity_wh) * 100.0 if ev.capacity_wh > 0 else 0.0

            try:
                penalty_factor = float(
                    self.config.optimization.genetic.penalties.get("ev_soc_miss", 10.0)
                )
            except Exception:
                penalty_factor = 10.0

            ev_penalty = ev_soc_miss_penalty(
                ev_soc_percentage=ev_soc_pct,
                min_soc_percentage=ev.min_soc_percentage,
                max_soc_percentage=ev.max_soc_percentage,
                penalty_factor=penalty_factor,
            )
            penalty += ev_penalty

        # AC charge break-even penalty
        initial_soc_wh = (bat.initial_soc_percentage / 100.0) * bat.capacity_wh
        inv_eff_dc_ac = inv.dc_to_ac_efficiency if inv else 1.0
        inv_eff_ac_dc = inv.ac_to_dc_efficiency if inv else 1.0

        try:
            ac_penalty_factor = float(
                self.config.optimization.genetic.penalties.get("ac_charge_break_even", 1.0)
            )
        except Exception:
            ac_penalty_factor = 1.0

        penalty += ac_charge_break_even_penalty(
            ac_charge_hours=ac_charge_hours,
            electricity_prices=np.array(params.ems.electricity_price_per_wh),
            load_wh_per_hour=np.array(params.ems.total_load),
            start_hour=0,
            initial_soc_wh=initial_soc_wh,
            min_soc_wh=_min_soc_wh(bat),
            battery_charging_efficiency=bat.charging_efficiency,
            battery_discharging_efficiency=bat.discharging_efficiency,
            inverter_dc_to_ac_efficiency=inv_eff_dc_ac,
            inverter_ac_to_dc_efficiency=inv_eff_ac_dc,
            battery_max_charge_power_w=_max_charge_power_w(bat),
            ac_penalty_factor=ac_penalty_factor,
        )

        return penalty

    def optimize(
        self,
        params: OptimizationParameters,
        ha_params: Optional[HomeApplianceParameters] = None,
        start_hour: int = 0,
        worst_case: bool = False,
        optimize_ev: bool = True,
        optimize_dc_charge: bool = True,
    ) -> DPSolution:
        """Run DP optimization.

        Implements Bellman optimality with backward pass and policy extraction.

        Args:
            params: Optimization parameters.
            ha_params: Home appliance parameters or None.
            start_hour: Start hour index.
            worst_case: If True, invert cost (maximize instead of minimize).
            optimize_ev: Whether to optimize EV charging.
            optimize_dc_charge: Whether dc_charge_allowed is a decision variable.

        Returns:
            DPSolution with optimal decision variables.
        """
        t_start = time.perf_counter()

        if params.ems is None or params.pv_battery is None:
            raise ValueError("EMS and battery parameters required for DP optimization")

        bat = params.pv_battery
        ev = params.ev
        inv = params.inverter

        has_ev = ev is not None and optimize_ev
        has_ha = ha_params is not None

        horizon = len(params.ems.electricity_price_per_wh)
        pv_array = np.array(params.ems.pv_forecast_wh)
        load_array = np.array(params.ems.total_load)
        price_array = np.array(params.ems.electricity_price_per_wh)

        # Handle feed_in_tariff_per_wh which can be list[float] or float
        if isinstance(params.ems.feed_in_tariff_per_wh, list):
            revenue_array = np.array(params.ems.feed_in_tariff_per_wh)
        else:
            revenue_array = np.full(horizon, float(params.ems.feed_in_tariff_per_wh))

        # Charge rates
        bat_charge_rates = self._get_charge_rates(bat.charge_rates)
        ev_charge_rates = self._get_charge_rates(ev.charge_rates if ev else None)

        # Build action space
        actions = self._build_action_space(
            bat_charge_rates, ev_charge_rates, optimize_dc_charge
        )
        num_actions = len(actions)

        # Battery SoC bounds
        bat_min_wh = _min_soc_wh(bat)
        bat_max_wh = _max_soc_wh(bat)
        initial_bat_soc_wh = (bat.initial_soc_percentage / 100.0) * bat.capacity_wh
        initial_bat_idx = self._get_soc_index(initial_bat_soc_wh, bat_min_wh, bat_max_wh)

        # EV SoC bounds: use full capacity range (0-100%), not min/max_soc_percentage
        # min_soc_percentage is the TARGET at end of horizon, not the operational minimum
        initial_ev_idx = 0
        ev_min_wh, ev_max_wh = 0.0, 0.0
        if has_ev and ev is not None:
            ev_min_wh = 0.0  # EV can be empty during optimization
            ev_max_wh = ev.capacity_wh  # EV can be fully charged
            initial_ev_soc_wh = (ev.initial_soc_percentage / 100.0) * ev.capacity_wh
            initial_ev_idx = self._get_soc_index(initial_ev_soc_wh, ev_min_wh, ev_max_wh)

        # Precompute SOC index lookup arrays (avoids repeated _get_soc_from_index calls)
        bat_soc_lookup = np.array(
            [self._get_soc_from_index(i, bat_min_wh, bat_max_wh) for i in range(NUM_SOC_LEVELS)],
            dtype=np.float64
        )
        ev_soc_lookup = np.array(
            [self._get_soc_from_index(i, ev_min_wh, ev_max_wh) for i in range(NUM_SOC_LEVELS)]
            if has_ev and ev is not None else np.zeros(NUM_SOC_LEVELS, dtype=np.float64),
            dtype=np.float64
        )

        # Precompute EV transitions: ev_trans[action_idx, ev_idx] = (next_ev_idx, ev_draw_wh)
        # EV transition depends only on (ev_idx, action), not on time
        ev_trans_next = np.full((num_actions, NUM_SOC_LEVELS), -1, dtype=np.int32)
        ev_trans_draw = np.zeros((num_actions, NUM_SOC_LEVELS), dtype=np.float64)
        if has_ev and ev is not None:
            ev_max_charge_w = _max_charge_power_w(ev)
            for a_idx, action in enumerate(actions):
                ev_rate = ev_charge_rates[action.ev_charge_rate_idx]
                if ev_rate <= 0:
                    ev_trans_next[a_idx, :] = np.arange(NUM_SOC_LEVELS, dtype=np.int32)
                else:
                    ev_draw_wh = ev_max_charge_w * ev_rate
                    ev_trans_draw[a_idx, :] = ev_draw_wh
                    for ev_idx in range(NUM_SOC_LEVELS):
                        if ev_idx == NUM_SOC_LEVELS - 1:
                            # At 100%, cannot charge further
                            ev_trans_next[a_idx, ev_idx] = ev_idx
                            ev_trans_draw[a_idx, ev_idx] = 0.0
                        else:
                            ev_soc_wh = ev_soc_lookup[ev_idx]
                            next_ev_soc_wh = compute_ev_next_soc(
                                current_soc_wh=ev_soc_wh,
                                min_soc_wh=ev_min_wh,
                                max_soc_wh=ev_max_wh,
                                charging_efficiency=ev.charging_efficiency,
                                charge_factor=ev_rate,
                                max_charge_power_w=ev_max_charge_w,
                            )
                            next_ev_idx = self._get_soc_index(next_ev_soc_wh, ev_min_wh, ev_max_wh)
                            ev_trans_next[a_idx, ev_idx] = next_ev_idx

        # Appliance time windows
        ha_start_windows: list[int] = []
        if has_ha and ha_params is not None and ha_params.time_windows is not None:
            for tw in ha_params.time_windows:
                for h in range(tw.start, tw.end + 1):
                    if h < horizon:
                        ha_start_windows.append(h)
        elif has_ha:
            ha_start_windows = list(range(horizon))

        # Precompute terminal penalty components (independent of state)
        # ac_charge_break_even_penalty doesn't depend on bat_idx/ev_idx, compute once
        inv = params.inverter
        bat = params.pv_battery
        inv_eff_dc_ac = inv.dc_to_ac_efficiency if inv else 1.0
        inv_eff_ac_dc = inv.ac_to_dc_efficiency if inv else 1.0
        initial_soc_wh = (bat.initial_soc_percentage / 100.0) * bat.capacity_wh

        try:
            ac_penalty_factor = float(
                self.config.optimization.genetic.penalties.get("ac_charge_break_even", 1.0)
            )
        except Exception:
            ac_penalty_factor = 1.0

        # Precompute ac_charge_break_even_penalty once (with zero ac_charge_hours)
        ac_charge_hours_zero = np.zeros(horizon)
        ac_penalty_base = ac_charge_break_even_penalty(
            ac_charge_hours=ac_charge_hours_zero,
            electricity_prices=np.array(params.ems.electricity_price_per_wh),
            load_wh_per_hour=np.array(params.ems.total_load),
            start_hour=0,
            initial_soc_wh=initial_soc_wh,
            min_soc_wh=_min_soc_wh(bat),
            battery_charging_efficiency=bat.charging_efficiency,
            battery_discharging_efficiency=bat.discharging_efficiency,
            inverter_dc_to_ac_efficiency=inv_eff_dc_ac,
            inverter_ac_to_dc_efficiency=inv_eff_ac_dc,
            battery_max_charge_power_w=_max_charge_power_w(bat),
            ac_penalty_factor=ac_penalty_factor,
        )

        # Precompute EV penalty factor
        ev = params.ev
        ev_penalty_factor = 10.0
        if ev is not None:
            try:
                ev_penalty_factor = float(
                    self.config.optimization.genetic.penalties.get("ev_soc_miss", 10.0)
                )
            except Exception:
                ev_penalty_factor = 10.0

        # Value function: V[hour, bat_idx, ev_idx, appliance_started]
        V = np.full(
            (horizon + 1, NUM_SOC_LEVELS, NUM_SOC_LEVELS, 2),
            INF_COST,
            dtype=np.float64,
        )

        # Initialize V[horizon] with terminal penalties (vectorized)
        # Precompute battery residual values for all bat_idx
        bat_min_wh = _min_soc_wh(bat)
        bat_max_wh = _max_soc_wh(bat)
        initial_soc_wh = (bat.initial_soc_percentage / 100.0) * bat.capacity_wh
        bat_resid_values = np.zeros(NUM_SOC_LEVELS, dtype=np.float64)
        for bat_idx in range(NUM_SOC_LEVELS):
            bat_soc_wh = self._get_soc_from_index(bat_idx, bat_min_wh, bat_max_wh)
            bat_resid_values[bat_idx] = battery_residual_value_penalty(
                battery_energy_content_wh=bat_soc_wh,
                dc_to_ac_efficiency=inv_eff_dc_ac,
                price_per_wh_battery=params.ems.price_per_wh_battery,
                initial_soc_wh=initial_soc_wh,
                electricity_prices=price_array,
                feed_in_tariffs=revenue_array,
            )

        # Precompute EV penalties for all ev_idx
        # Includes both residual value penalty (energy cost) and SOC miss penalty (requirement)
        ev_penalties = np.zeros(NUM_SOC_LEVELS, dtype=np.float64)
        if has_ev and ev is not None:
            ev_initial_soc_wh = (ev.initial_soc_percentage / 100.0) * ev.capacity_wh
            for ev_idx in range(NUM_SOC_LEVELS):
                ev_soc_wh = self._get_soc_from_index(ev_idx, 0.0, ev.capacity_wh)
                ev_soc_pct = (ev_soc_wh / ev.capacity_wh) * 100.0 if ev.capacity_wh > 0 else 0.0

                # EV residual value penalty: cost of energy invested in EV beyond initial SOC
                ev_resid = ev_residual_value_penalty(
                    ev_energy_content_wh=ev_soc_wh,
                    initial_soc_wh=ev_initial_soc_wh,
                    electricity_prices=price_array,
                    feed_in_tariffs=revenue_array,
                )

                # EV SOC miss penalty: penalty for not reaching minimum SOC requirement
                ev_miss = ev_soc_miss_penalty(
                    ev_soc_percentage=ev_soc_pct,
                    min_soc_percentage=ev.min_soc_percentage,
                    max_soc_percentage=ev.max_soc_percentage,
                    penalty_factor=ev_penalty_factor,
                )

                ev_penalties[ev_idx] = ev_resid + ev_miss

        # Vectorized terminal penalty assignment
        # V[horizon, bat_idx, ev_idx, ap_started] = bat_resid[bat_idx] + ev_penalty[ev_idx] + ac_penalty_base
        for ap_started in [0, 1]:
            # Use broadcasting: (NUM_SOC_LEVELS, 1) + (1, NUM_SOC_LEVELS) + scalar
            V[horizon, :, :, ap_started] = (
                bat_resid_values[:, np.newaxis]
                + ev_penalties[np.newaxis, :]
                + ac_penalty_base
            )

        # Policy storage
        policy_action = np.full(
            (horizon, NUM_SOC_LEVELS, NUM_SOC_LEVELS, 2),
            -1,
            dtype=np.int32,
        )
        policy_bat = np.full(
            (horizon, NUM_SOC_LEVELS, NUM_SOC_LEVELS, 2),
            -1,
            dtype=np.int32,
        )
        policy_ev = np.full(
            (horizon, NUM_SOC_LEVELS, NUM_SOC_LEVELS, 2),
            -1,
            dtype=np.int32,
        )
        policy_appliance = np.full(
            (horizon, NUM_SOC_LEVELS, NUM_SOC_LEVELS, 2),
            -1,
            dtype=np.int8,
        )

        total_states_explored = 0

        # Precompute battery parameters for inline computation
        bat_charging_eff = bat.charging_efficiency
        bat_discharging_eff = bat.discharging_efficiency
        ac_to_dc_eff = inv.ac_to_dc_efficiency if inv else 1.0
        dc_to_ac_eff = inv.dc_to_ac_efficiency if inv else 1.0
        max_ac_power = inv.max_ac_charge_power_w if inv and inv.max_ac_charge_power_w is not None else _max_charge_power_w(bat)

        # Precompute action properties as arrays for fast access
        action_ac_rate_idx = np.array([a.ac_charge_rate_idx for a in actions], dtype=np.int32)
        action_dc_charge = np.array([a.dc_charge_allowed for a in actions], dtype=np.int32)
        action_discharge = np.array([a.discharge_allowed for a in actions], dtype=np.int32)
        action_ev_rate_idx = np.array([a.ev_charge_rate_idx for a in actions], dtype=np.int32)

        # Precompute bat_charge_rates as array
        bat_charge_rates_arr = np.array(bat_charge_rates, dtype=np.float64)

        # Precompute ha_start_windows as numpy array for numba
        ha_start_windows_arr = np.array(ha_start_windows, dtype=np.int32)

        # Backward pass: Bellman optimality from horizon-1 to 0 using Numba JIT
        # V[horizon] already contains terminal penalties
        _bellman_backward(
            V,
            policy_action,
            policy_bat,
            policy_ev,
            policy_appliance,
            bat_soc_lookup,
            ev_trans_next,
            ev_trans_draw,
            action_ac_rate_idx,
            action_dc_charge,
            action_discharge,
            bat_charge_rates_arr,
            pv_array,
            load_array,
            price_array,
            revenue_array,
            ha_start_windows_arr,
            horizon,
            num_actions,
            len(ha_start_windows),
            has_ev,
            has_ha,
            worst_case,
            bat_min_wh,
            bat_max_wh,
            bat_charging_eff,
            bat_discharging_eff,
            ac_to_dc_eff,
            dc_to_ac_eff,
            max_ac_power,
            NUM_SOC_LEVELS,
            INF_COST,
        )

        # Total states explored (for reporting)
        total_states_explored = horizon * NUM_SOC_LEVELS * NUM_SOC_LEVELS * 2

        # Optimal cost from start state is V[0, initial_state]
        optimal_cost = V[0, initial_bat_idx, initial_ev_idx, 0]
        if optimal_cost >= INF_COST:
            raise RuntimeError("DP optimization failed: start state cannot reach any valid terminal state")

        # Forward pass to extract optimal policy from start state
        # policy_*[t, s] stores the NEXT state chosen from state s at time t
        bat_idx, ev_idx, ap_started = initial_bat_idx, initial_ev_idx, 0

        ac_charge_factors: list[float] = [0.0] * horizon
        dc_charge_flags: list[float] = [0.0] * horizon
        discharge_flags: list[int] = [0] * horizon
        ev_charge_factors: list[float] = [0.0] * horizon
        appliance_start_hour: Optional[int] = None

        for t in range(horizon):
            a_idx = policy_action[t, bat_idx, ev_idx, ap_started]
            next_bat = policy_bat[t, bat_idx, ev_idx, ap_started]
            next_ev = policy_ev[t, bat_idx, ev_idx, ap_started]
            next_ap = policy_appliance[t, bat_idx, ev_idx, ap_started]

            if a_idx == -1 or next_bat == -1:
                # No valid action from this state
                break

            if a_idx == -2:
                # Home appliance start
                appliance_start_hour = t
            elif a_idx >= 0:
                action = actions[a_idx]
                ac_charge_factors[t] = bat_charge_rates[action.ac_charge_rate_idx]
                dc_charge_flags[t] = float(action.dc_charge_allowed)
                discharge_flags[t] = int(action.discharge_allowed)
                ev_charge_factors[t] = ev_charge_rates[action.ev_charge_rate_idx]
            # else: default (no action)

            bat_idx = next_bat
            ev_idx = next_ev
            ap_started = next_ap

        # Best terminal state is the final state reached
        best_terminal_state = (bat_idx, ev_idx, ap_started)

        t_end = time.perf_counter()
        computation_time_ms = (t_end - t_start) * 1000.0

        # Build solution
        return self._create_solution(
            params=params,
            ha_params=ha_params,
            start_hour=start_hour,
            ac_charge_factors=ac_charge_factors,
            dc_charge_flags=dc_charge_flags,
            discharge_flags=discharge_flags,
            ev_charge_factors=ev_charge_factors if has_ev else None,
            appliance_start_hour=appliance_start_hour,
            optimal_cost=optimal_cost,
            total_states_explored=total_states_explored,
            computation_time_ms=computation_time_ms,
            start_soc_index=initial_bat_idx,
            end_soc_index=best_terminal_state[0] if best_terminal_state else initial_bat_idx,
        )

    def _run_simulation(
        self,
        params: OptimizationParameters,
        ha_params: Optional[HomeApplianceParameters],
        ac_charge_factors: list[float],
        dc_charge_flags: list[float],
        discharge_flags: list[int],
        ev_charge_factors: Optional[list[float]],
        appliance_start_hour: Optional[int],
        start_hour: int,
    ) -> tuple[SimulationResult, Optional[Battery]]:
        """Run simulation with DP solution to obtain SimulationResult.

        Follows the same pattern as GA's evaluate_inner(): create devices,
        prepare SimulationSession, set action arrays, run simulation.

        Args:
            params: Optimization parameters.
            ha_params: Home appliance parameters or None.
            ac_charge_factors: AC charge factors from DP solution.
            dc_charge_flags: DC charge flags from DP solution.
            discharge_flags: Discharge flags from DP solution.
            ev_charge_factors: EV charge factors from DP solution or None.
            appliance_start_hour: Home appliance start hour or None.
            start_hour: Simulation start hour.

        Returns:
            Tuple of (SimulationResult, Battery instance for ev_obj) or (SimulationResult, None).
        """
        if params.ems is None or params.pv_battery is None:
            raise ValueError("EMS and battery parameters required for DP simulation")

        horizon = len(params.ems.electricity_price_per_wh)

        # Create battery device
        battery = Battery(
            parameters=params.pv_battery,
            prediction_hours=horizon,
        )

        # Create EV device
        ev: Optional[Battery] = None
        if params.ev is not None:
            ev = Battery(
                parameters=params.ev,
                prediction_hours=horizon,
            )

        # Create inverter
        inverter: Optional[Inverter] = None
        if params.inverter is not None:
            inverter = Inverter(
                params.inverter,
                battery=battery,
            )

        # Create home appliance
        home_appliance: Optional[HomeAppliance] = None
        if ha_params is not None:
            home_appliance = HomeAppliance(
                parameters=ha_params,
                optimization_hours=self.config.optimization.horizon_hours,
                prediction_hours=horizon,
            )

        # Create and prepare simulation session
        simulation = SimulationSession(
            start_hour=start_hour,
            optimization_hours=self.config.optimization.horizon_hours,
            prediction_hours=horizon,
        )
        simulation.prepare(
            parameters=params.ems,
            optimization_hours=self.config.optimization.horizon_hours,
            prediction_hours=horizon,
            inverter=inverter,
            ev=ev,
            home_appliance=home_appliance,
        )

        # Set action arrays from DP solution
        simulation.ac_charge_hours = np.array(ac_charge_factors[:horizon], dtype=float)
        simulation.dc_charge_hours = np.array(dc_charge_flags[:horizon], dtype=float)
        simulation.bat_discharge_hours = np.array(discharge_flags[:horizon], dtype=float)

        if ev_charge_factors is not None:
            simulation.ev_charge_hours = np.array(ev_charge_factors[:horizon], dtype=float)
        else:
            simulation.ev_charge_hours = np.full(horizon, 0.0)

        if appliance_start_hour is not None and home_appliance is not None:
            simulation.home_appliance_start_hour = appliance_start_hour

        # Run simulation
        simulation_result_dict = simulation.simulate(start_hour)
        simulation_result = SimulationResult(**simulation_result_dict)

        # Apply terminal penalties to match DP optimization
        if params.pv_battery and params.inverter:
            battery_energy_content = battery.current_energy_content()
            initial_soc_wh = (params.pv_battery.initial_soc_percentage / 100.0) * params.pv_battery.capacity_wh
            penalty = battery_residual_value_penalty(
                battery_energy_content_wh=battery_energy_content,
                dc_to_ac_efficiency=params.inverter.dc_to_ac_efficiency,
                price_per_wh_battery=params.ems.price_per_wh_battery,
                initial_soc_wh=initial_soc_wh,
                electricity_prices=np.array(params.ems.electricity_price_per_wh),
                feed_in_tariffs=np.array(params.ems.feed_in_tariff_per_wh) if isinstance(params.ems.feed_in_tariff_per_wh, list) else np.full(len(params.ems.electricity_price_per_wh), params.ems.feed_in_tariff_per_wh),
            )
            simulation_result.total_balance += penalty

        return simulation_result, ev

    def _create_solution(
        self,
        params: OptimizationParameters,
        ha_params: Optional[HomeApplianceParameters],
        start_hour: int,
        ac_charge_factors: list[float],
        dc_charge_flags: list[float],
        discharge_flags: list[int],
        ev_charge_factors: Optional[list[float]],
        appliance_start_hour: Optional[int],
        optimal_cost: float,
        total_states_explored: int,
        computation_time_ms: float,
        start_soc_index: int,
        end_soc_index: int,
    ) -> DPSolution:
        """Create DPSolution from optimization results.

        Runs a simulation with the optimal decision variables to obtain
        the required SimulationResult (same pattern as GA).
        """
        # Run simulation to get SimulationResult
        simulation_result, ev_instance = self._run_simulation(
            params=params,
            ha_params=ha_params,
            ac_charge_factors=ac_charge_factors,
            dc_charge_flags=dc_charge_flags,
            discharge_flags=discharge_flags,
            ev_charge_factors=ev_charge_factors,
            appliance_start_hour=appliance_start_hour,
            start_hour=start_hour,
        )

        # ev_obj uses Battery instance; validator converts to ElectricVehicleResult
        ev_obj = ev_instance
        solution_dict: dict[str, Any] = {
            "ems": params.ems.model_dump() if params.ems else None,
            "pv_battery": params.pv_battery.model_dump() if params.pv_battery else None,
            "ev": params.ev.model_dump() if params.ev else None,
            "inverter": params.inverter.model_dump() if params.inverter else None,
            "ac_charge": ac_charge_factors,
            "dc_charge": dc_charge_flags,
            "discharge_allowed": discharge_flags,
            "ev_charge_hours_float": ev_charge_factors,
            "result": simulation_result,
            "ev_obj": ev_obj,
            "washingstart": appliance_start_hour,
        }

        solution = DPSolution(**solution_dict)
        solution.total_states_explored = total_states_explored
        solution.computation_time_ms = computation_time_ms
        solution.dp_start_soc_index = start_soc_index
        solution.dp_end_soc_index = end_soc_index

        return solution

    def to_ga_individual(self, solution: DPSolution) -> list[int]:
        """Convert DP solution to GA individual format for HYBRID mode.

        Maps DP decision variables to GA encoding format:
        [discharge_states..., ev_charge_indices..., appliance_start]

        Args:
            solution: DP solution to convert.

        Returns:
            GA-compatible individual as list of integers.
        """
        horizon = len(solution.ac_charge)

        # Get charge rates (default to common set)
        bat_charge_rates = self._get_charge_rates(None)
        ev_charge_rates = self._get_charge_rates(None)

        optimize_dc_charge = True
        num_dc_options = 2 if optimize_dc_charge else 1

        # Encode discharge states
        individual: list[int] = []
        for t in range(horizon):
            ac_factor = solution.ac_charge[t]
            dc_flag = bool(solution.dc_charge[t])
            dis_flag = bool(solution.discharge_allowed[t])

            # Find matching charge rate index
            ac_idx = 0
            for i, rate in enumerate(bat_charge_rates):
                if abs(rate - ac_factor) < 1e-6:
                    ac_idx = i
                    break

            # Encode into single integer
            state = ac_idx
            if dc_flag and optimize_dc_charge:
                state += len(bat_charge_rates)
            if dis_flag:
                state += len(bat_charge_rates) * num_dc_options

            individual.append(state)

        # Add EV charge indices
        if solution.ev_charge_hours_float is not None:
            for t in range(horizon):
                ev_factor = solution.ev_charge_hours_float[t]
                ev_idx = 0
                for i, rate in enumerate(ev_charge_rates):
                    if abs(rate - ev_factor) < 1e-6:
                        ev_idx = i
                        break
                individual.append(ev_idx)

        # Add appliance start
        if solution.washingstart is not None:
            individual.append(solution.washingstart)

        return individual