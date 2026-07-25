"""Genetic algorithm."""

from __future__ import annotations

import random
import time
from typing import Any, Optional

import numpy as np
from deap import algorithms, base, creator, tools
from loguru import logger
from pydantic import ConfigDict, Field

from akkudoktoreos.optimization.simulation.devices import (
    Battery,
    HomeAppliance,
    Inverter,
)
from akkudoktoreos.optimization.simulation.session import SimulationSession
from akkudoktoreos.optimization.genetic.geneticparams import (
    GeneticEnergyManagementParameters,
    GeneticOptimizationParameters,
)
from akkudoktoreos.optimization.genetic.geneticsolution import GeneticSolution
from akkudoktoreos.optimization.simulation.solution import SimulationResult
from akkudoktoreos.optimization.optimizationabc import OptimizationBase


class GeneticSimulation(SimulationSession):
    """Device simulation for GENETIC optimization algorithm.

    Thin subclass of [SimulationSession](akkudoktoreos.optimization.simulation.session.SimulationSession)
    that overrides [prepare](akkudoktoreos.optimization.simulation.session.SimulationSession.prepare)
    to accept solver-specific parameter types.
    """

    # Disable validation on assignment to speed up simulation runs.
    model_config = ConfigDict(
        validate_assignment=False,
    )

    def prepare(
        self,
        parameters: GeneticEnergyManagementParameters,
        optimization_hours: int,
        prediction_hours: int,
        ev: Optional[Battery] = None,
        home_appliance: Optional[HomeAppliance] = None,
        inverter: Optional[Inverter] = None,
    ) -> None:
        """Prepare simulation runs.

        Populate internal arrays and device references used during simulation.

        Args:
            parameters: Genetic energy management parameters (EMS forecasts).
            optimization_hours: Number of optimization hours.
            prediction_hours: Number of prediction hours.
            ev: Electric vehicle battery device.
            home_appliance: Home appliance device.
            inverter: Inverter device.
        """
        super().prepare(
            parameters,
            optimization_hours=optimization_hours,
            prediction_hours=prediction_hours,
            ev=ev,
            home_appliance=home_appliance,
            inverter=inverter,
        )


class GeneticOptimization(OptimizationBase):
    """GENETIC algorithm to solve energy optimization."""

    def __init__(
        self,
        verbose: bool = False,
        fixed_seed: Optional[int] = None,
    ):
        """Initialize the optimization problem with the required parameters."""
        self.opti_param: dict[str, Any] = {}
        self.fixed_ev_hours = self.config.prediction.hours - self.config.optimization.horizon_hours
        self.ev_possible_charge_values: list[float] = [1.0]
        # Separate charge-level list for battery AC charging (independent of EV rates).
        # Populated from parameters.pv_battery.charge_rates in optimize_ems.
        self.bat_possible_charge_values: list[float] = [1.0]
        self.verbose = verbose
        self.fix_seed = fixed_seed
        self.optimize_ev = True
        self.optimize_dc_charge = False
        self.fitness_history: dict[str, Any] = {}

        # Set a fixed seed for random operations if provided or in debug mode
        if self.fix_seed is not None:
            random.seed(self.fix_seed)
        elif logger.level == "DEBUG":
            self.fix_seed = random.randint(1, 100000000000)  # noqa: S311
            random.seed(self.fix_seed)

        # Create Simulation
        self.simulation = GeneticSimulation()

    def decode_charge_discharge(
        self, discharge_hours_bin: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode the input array into ac_charge, dc_charge, and discharge arrays.

        Delegates to the pure function in [encoding](akkudoktoreos.optimization.genetic.encoding).
        """
        from akkudoktoreos.optimization.genetic.encoding import decode_charge_discharge as _decode

        return _decode(
            discharge_hours_bin,
            self.bat_possible_charge_values,
            self.optimize_dc_charge,
        )

    def mutate(self, individual: list[int]) -> tuple[list[int]]:
        """Custom mutation function for the individual."""
        from akkudoktoreos.optimization.genetic.encoding import compute_total_states

        total_states = compute_total_states(
            self.bat_possible_charge_values,
            self.optimize_dc_charge,
        )

        # 1. Mutating the charge_discharge part
        charge_discharge_part = individual[: self.config.prediction.hours]
        (charge_discharge_mutated,) = self.toolbox.mutate_charge_discharge(charge_discharge_part)

        # Instead of a fixed clamping to 0..8 or 0..6 dynamically:
        charge_discharge_mutated = np.clip(charge_discharge_mutated, 0, total_states - 1)
        individual[: self.config.prediction.hours] = charge_discharge_mutated

        # 2. Mutating the EV charge part, if active
        if self.optimize_ev:
            ev_charge_part = individual[
                self.config.prediction.hours : self.config.prediction.hours * 2
            ]
            (ev_charge_part_mutated,) = self.toolbox.mutate_ev_charge_index(ev_charge_part)
            ev_charge_part_mutated[self.config.prediction.hours - self.fixed_ev_hours :] = [
                0
            ] * self.fixed_ev_hours
            individual[self.config.prediction.hours : self.config.prediction.hours * 2] = (
                ev_charge_part_mutated
            )

        # 3. Mutating the appliance start time, if applicable
        if self.opti_param["home_appliance"] > 0:
            appliance_part = [individual[-1]]
            (appliance_part_mutated,) = self.toolbox.mutate_hour(appliance_part)
            individual[-1] = appliance_part_mutated[0]

        return (individual,)

    # Method to create an individual based on the conditions
    def create_individual(self) -> list[int]:
        # Start with discharge states for the individual
        individual_components = [
            self.toolbox.attr_discharge_state() for _ in range(self.config.prediction.hours)
        ]

        # Add EV charge index values if optimize_ev is True
        if self.optimize_ev:
            individual_components += [
                self.toolbox.attr_ev_charge_index() for _ in range(self.config.prediction.hours)
            ]

        # Add the start time of the household appliance if it's being optimized
        if self.opti_param["home_appliance"] > 0:
            individual_components += [self.toolbox.attr_int()]

        return creator.Individual(individual_components)

    def merge_individual(
        self,
        discharge_hours_bin: np.ndarray,
        ev_charge_hours_index: Optional[np.ndarray],
        washingstart_int: Optional[int],
    ) -> list[int]:
        """Merge the individual components back into a single solution list.

        Delegates to the pure function in [encoding](akkudoktoreos.optimization.genetic.encoding).
        """
        from akkudoktoreos.optimization.genetic.encoding import merge_individual as _merge

        optimize_ha = self.opti_param.get("home_appliance", 0) > 0
        return _merge(
            discharge_hours_bin,
            ev_charge_hours_index,
            washingstart_int,
            self.config.prediction.hours,
            self.optimize_ev,
            optimize_ha,
        )

    def split_individual(
        self, individual: list[int]
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[int]]:
        """Split the individual solution into its components.

        Delegates to the pure function in [encoding](akkudoktoreos.optimization.genetic.encoding).

        Components:
        1. Discharge hours (binary as int NumPy array),
        2. Electric vehicle charge hours (float as int NumPy array, if applicable),
        3. Dishwasher start time (integer if applicable).
        """
        from akkudoktoreos.optimization.genetic.encoding import split_individual as _split

        optimize_ha = self.opti_param and self.opti_param.get("home_appliance", 0) > 0
        return _split(
            individual,
            self.config.prediction.hours,
            self.optimize_ev,
            optimize_ha,
        )

    def setup_deap_environment(self, opti_param: dict[str, Any], start_hour: int) -> None:
        """Set up the DEAP environment with fitness and individual creation rules."""
        from akkudoktoreos.optimization.genetic.encoding import compute_total_states

        self.opti_param = opti_param

        # Remove existing definitions if any
        for attr in ["FitnessMin", "Individual"]:
            if attr in creator.__dict__:
                del creator.__dict__[attr]

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        # Battery state space uses bat_possible_charge_values; EV index space uses ev_possible_charge_values.
        len_ev = len(self.ev_possible_charge_values)

        total_states = compute_total_states(
            self.bat_possible_charge_values,
            self.optimize_dc_charge,
        )

        # State space: 0 .. (total_states - 1)
        self.toolbox.register("attr_discharge_state", random.randint, 0, total_states - 1)

        # EV attributes (separate index space)
        if self.optimize_ev:
            self.toolbox.register(
                "attr_ev_charge_index",
                random.randint,
                0,
                len_ev - 1,
            )

        # Household appliance start time
        self.toolbox.register("attr_int", random.randint, start_hour, 23)

        self.toolbox.register("individual", self.create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)

        # Mutation operator for battery charge/discharge states
        self.toolbox.register(
            "mutate_charge_discharge", tools.mutUniformInt, low=0, up=total_states - 1, indpb=0.2
        )

        # Mutation operator for EV states (separate index space)
        self.toolbox.register(
            "mutate_ev_charge_index",
            tools.mutUniformInt,
            low=0,
            up=len_ev - 1,
            indpb=0.2,
        )

        # Mutation for household appliance
        self.toolbox.register("mutate_hour", tools.mutUniformInt, low=start_hour, up=23, indpb=0.2)

        # Custom mutate function remains unchanged
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def evaluate_inner(self, individual: list[int]) -> dict[str, Any]:
        """Simulates the energy management system (EMS) using the provided individual solution.

        This is an internal function.
        """
        self.simulation.reset()
        discharge_hours_bin, ev_charge_hours_index, washingstart_int = self.split_individual(
            individual
        )

        if self.opti_param.get("home_appliance", 0) > 0 and washingstart_int:
            # Set start hour for appliance
            self.simulation.home_appliance_start_hour = washingstart_int

        ac_charge_hours, dc_charge_hours, discharge = self.decode_charge_discharge(
            discharge_hours_bin
        )

        self.simulation.bat_discharge_hours = discharge
        # Set DC charge hours only if DC optimization is enabled
        if self.optimize_dc_charge:
            self.simulation.dc_charge_hours = dc_charge_hours
        else:
            self.simulation.dc_charge_hours = np.full(self.config.prediction.hours, 1)
        self.simulation.ac_charge_hours = ac_charge_hours

        if ev_charge_hours_index is not None:
            ev_charge_hours_float = np.array(
                [self.ev_possible_charge_values[i] for i in ev_charge_hours_index],
                float,
            )
            # discharge is set to 0 by default
            self.simulation.ev_charge_hours = ev_charge_hours_float
        else:
            # discharge is set to 0 by default
            self.simulation.ev_charge_hours = np.full(self.config.prediction.hours, 0)

        # Do the simulation and return result.
        return self.simulation.simulate(self.ems.start_datetime.hour)

    def evaluate(
        self,
        individual: list[int],
        parameters: GeneticOptimizationParameters,
        start_hour: int,
        worst_case: bool,
    ) -> tuple[float]:
        """Evaluate the fitness score of a single individual in the DEAP genetic algorithm.

        This method runs a simulation based on the provided individual genome and
        optimization parameters. The resulting performance is converted into a
        fitness score compatible with DEAP (i.e., returned as a 1-tuple).

        Args:
            individual (list[int]):
                The genome representing one candidate solution.
            parameters (GeneticOptimizationParameters):
                Optimization parameters that influence simulation behavior,
                constraints, and scoring logic.
            start_hour (int):
                The simulation start hour (0–23 or domain-specific).
                Used to initialize time-based scheduling or constraints.
            worst_case (bool):
                If True, evaluates the solution under worst-case assumptions
                (e.g., pessimistic forecasts or boundary conditions).
                If False, uses nominal assumptions.

        Returns:
            tuple[float]:
                A single-element tuple containing the computed fitness score.
                Lower score is better: "FitnessMin".

        Raises:
            ValueError: If input arguments are invalid or the individual structure
                is not compatible with the simulation.
            RuntimeError: If the simulation fails or cannot produce results.

        Notes:
            The resulting score should match DEAP's expected format: a tuple, even
            if only a single scalar fitness value is returned.
        """
        try:
            simulation_result = self.evaluate_inner(individual)
        except Exception as e:
            # Return bad fitness score ("FitnessMin") in case of an exception
            return (100000.0,)

        total_balance = simulation_result["Gesamtbilanz_Euro"] * (-1.0 if worst_case else 1.0)

        # EV 100% & charge not allowed
        if self.optimize_ev:
            discharge_hours_bin, ev_charge_hours_index, washingstart_int = self.split_individual(
                individual
            )

            ev_soc_per_hour = np.array(
                simulation_result.get("EAuto_SoC_pro_Stunde", [])
            )  # Beispielkey

            if ev_soc_per_hour is None or ev_charge_hours_index is None:
                raise ValueError("ev_soc_per_hour or ev_charge_hours_index is None")
            min_length = min(ev_soc_per_hour.size, ev_charge_hours_index.size)
            ev_soc_per_hour_tail = ev_soc_per_hour[-min_length:]
            ev_charge_hours_index_tail = ev_charge_hours_index[-min_length:]

            # Mask
            invalid_charge_mask = (ev_soc_per_hour_tail == 100) & (ev_charge_hours_index_tail > 0)

            if np.any(invalid_charge_mask):
                invalid_indices = np.where(invalid_charge_mask)[0]
                if len(invalid_indices) > 1:
                    ev_charge_hours_index_tail[invalid_indices] = 0

                ev_charge_hours_index[-min_length:] = ev_charge_hours_index_tail.tolist()

                adjusted_individual = self.merge_individual(
                    discharge_hours_bin, ev_charge_hours_index, washingstart_int
                )

                individual[:] = adjusted_individual

        # New check: Activate discharge when battery SoC is 0
        # battery_soc_per_hour = np.array(
        #     o.get("akku_soc_pro_stunde", [])
        # )  # Example key for battery SoC

        # if battery_soc_per_hour is not None:
        #     if battery_soc_per_hour is None or discharge_hours_bin is None:
        #         raise ValueError("battery_soc_per_hour or discharge_hours_bin is None")
        #     min_length = min(battery_soc_per_hour.size, discharge_hours_bin.size)
        #     battery_soc_per_hour_tail = battery_soc_per_hour[-min_length:]
        #     discharge_hours_bin_tail = discharge_hours_bin[-min_length:]
        #     len_ac = len(self.config.optimization.ev_available_charge_rates_percent)

        #     # # Find hours where battery SoC is 0
        #     # zero_soc_mask = battery_soc_per_hour_tail == 0
        #     # discharge_hours_bin_tail[zero_soc_mask] = (
        #     #     len_ac + 2
        #     # )  # Activate discharge for these hours

        #     # When Battery SoC then set the Discharge randomly to 0 or 1. otherwise it's very
        #     # unlikely to get a state where a battery can store energy for a longer time
        #     # Find hours where battery SoC is 0
        #     zero_soc_mask = battery_soc_per_hour_tail == 0
        #     # discharge_hours_bin_tail[zero_soc_mask] = (
        #     # len_ac + 2
        #     # )  # Activate discharge for these hours
        #     set_to_len_ac_plus_2 = np.random.rand() < 0.5  # True with 50% probability

        #     # Werte setzen basierend auf der zufälligen Entscheidung
        #     value_to_set = len_ac + 2 if set_to_len_ac_plus_2 else 0
        #     discharge_hours_bin_tail[zero_soc_mask] = value_to_set

        #     # Merge the updated discharge_hours_bin back into the individual
        #     adjusted_individual = self.merge_individual(
        #         discharge_hours_bin, ev_charge_hours_index, washingstart_int
        #     )
        #     individual[:] = adjusted_individual

        # More metrics
        individual.extra_data = (  # type: ignore[attr-defined]
            simulation_result["Gesamtbilanz_Euro"],
            simulation_result["Gesamt_Verluste"],
            parameters.ev.min_soc_percentage - self.simulation.ev.current_soc_percentage()
            if parameters.ev and self.simulation.ev
            else 0,
        )

        # --- Penalty functions (solver-agnostic) ---
        from akkudoktoreos.optimization.simulation.penalties import (
            ac_charge_break_even_penalty,
            battery_residual_value_penalty,
            ev_soc_miss_penalty,
        )

        # Battery residual value penalty
        if self.simulation.battery:
            battery_energy_content = self.simulation.battery.current_energy_content()
            dc_to_ac_eff = (
                self.simulation.inverter.dc_to_ac_efficiency
                if self.simulation.inverter
                else 1.0
            )
            total_balance += battery_residual_value_penalty(
                battery_energy_content_wh=battery_energy_content,
                dc_to_ac_efficiency=dc_to_ac_eff,
                price_per_wh_battery=parameters.ems.price_per_wh_battery,
            )

        # AC charging break-even penalty
        if (
            self.simulation.battery
            and self.simulation.inverter
            and self.simulation.ac_charge_hours is not None
            and self.simulation.elect_price_hourly is not None
            and self.simulation.load_energy_array is not None
        ):
            inv = self.simulation.inverter
            bat = self.simulation.battery
            initial_soc_wh = (bat.initial_soc_percentage / 100.0) * bat.capacity_wh

            try:
                ac_penalty_factor = float(
                    self.config.optimization.genetic.penalties["ac_charge_break_even"]
                )
            except Exception:
                ac_penalty_factor = 1.0

            total_balance += ac_charge_break_even_penalty(
                ac_charge_hours=self.simulation.ac_charge_hours,
                electricity_prices=self.simulation.elect_price_hourly,
                load_wh_per_hour=self.simulation.load_energy_array,
                start_hour=start_hour,
                initial_soc_wh=initial_soc_wh,
                min_soc_wh=bat.min_soc_wh,
                battery_charging_efficiency=bat.charging_efficiency,
                battery_discharging_efficiency=bat.discharging_efficiency,
                inverter_dc_to_ac_efficiency=inv.dc_to_ac_efficiency,
                inverter_ac_to_dc_efficiency=inv.ac_to_dc_efficiency,
                battery_max_charge_power_w=bat.max_charge_power_w,
                ac_penalty_factor=ac_penalty_factor,
            )

        # EV SOC miss penalty
        if self.optimize_ev and parameters.ev and self.simulation.ev:
            try:
                penalty_factor = self.config.optimization.genetic.penalties["ev_soc_miss"]
            except Exception:
                penalty_factor = 10
                logger.error(
                    "Penalty function parameter `ev_soc_miss` not configured, using {}.",
                    penalty_factor,
                )
            ev_soc_percentage = self.simulation.ev.current_soc_percentage()
            total_balance += ev_soc_miss_penalty(
                ev_soc_percentage=ev_soc_percentage,
                min_soc_percentage=parameters.ev.min_soc_percentage,
                max_soc_percentage=parameters.ev.max_soc_percentage,
                penalty_factor=penalty_factor,
            )

        return (total_balance,)

    def optimize(
        self,
        start_solution: Optional[list[float]] = None,
        ngen: int = 200,
    ) -> tuple[Any, dict[str, list[Any]]]:
        """Run the optimization process using a genetic algorithm.

        @TODO: optimize() ngen default (200) is different from optimize_ems() ngen default (400).
        """
        # Set the number of inviduals in a generation
        try:
            individuals = self.config.optimization.genetic.individuals
            if individuals is None:
                raise
        except Exception:
            individuals = 300
            logger.error("Individuals not configured. Using {}.", individuals)

        population = self.toolbox.population(n=individuals)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        stats.register("avg", np.mean)
        stats.register("max", np.max)

        logger.debug("Start optimize: {}", start_solution)

        # Insert the start solution into the population if provided
        if start_solution is not None:
            for _ in range(10):
                population.insert(0, creator.Individual(start_solution))

        # Run the evolutionary algorithm
        pop, log = algorithms.eaMuPlusLambda(
            population,
            self.toolbox,
            mu=100,
            lambda_=150,
            cxpb=0.6,
            mutpb=0.4,
            ngen=ngen,
            stats=stats,
            halloffame=hof,
            verbose=self.verbose,
        )

        # Store fitness history
        self.fitness_history = {
            "gen": log.select("gen"),  # Generation numbers (X-axis)
            "avg": log.select("avg"),  # Average fitness for each generation (Y-axis)
            "max": log.select("max"),  # Maximum fitness for each generation (Y-axis)
            "min": log.select("min"),  # Minimum fitness for each generation (Y-axis)
        }

        member: dict[str, list[float]] = {"balance": [], "losses": [], "constraints": []}
        for ind in population:
            if hasattr(ind, "extra_data"):
                extra_value1, extra_value2, extra_value3 = ind.extra_data
                member["balance"].append(extra_value1)
                member["losses"].append(extra_value2)
                member["constraints"].append(extra_value3)

        return hof[0], member

    def optimize_ems(
        self,
        parameters: GeneticOptimizationParameters,
        start_hour: Optional[int] = None,
        worst_case: bool = False,
        ngen: Optional[int] = None,
    ) -> GeneticSolution:
        """Perform EMS (Energy Management System) optimization and visualize results."""
        if start_hour is None:
            start_hour = self.ems.start_datetime.hour
        # Start hour has to be in sync with energy management
        if start_hour != self.ems.start_datetime.hour:
            raise ValueError(
                f"Start hour not synced. EMS {self.ems.start_datetime.hour} vs. GENETIC {start_hour}."
            )

        # Set the number of generations
        generations = ngen
        if generations is None:
            try:
                generations = self.config.optimization.genetic.generations
            except Exception:
                generations = 400
                logger.error("Generations not configured. Using {}.", generations)

        self.simulation.reset()

        # Initialize PV and EV batteries
        battery: Optional[Battery] = None
        if parameters.pv_battery:
            battery = Battery(
                parameters.pv_battery,
                prediction_hours=self.config.prediction.hours,
            )
            battery.set_charge_per_hour(np.full(self.config.prediction.hours, 0))

        ev: Optional[Battery] = None
        if parameters.ev:
            ev = Battery(
                parameters.ev,
                prediction_hours=self.config.prediction.hours,
            )
            ev.set_charge_per_hour(np.full(self.config.prediction.hours, 1))
            self.optimize_ev = (
                parameters.ev.min_soc_percentage - parameters.ev.initial_soc_percentage >= 0
            )
            # electrical vehicle charge rates
            if parameters.ev.charge_rates is not None:
                self.ev_possible_charge_values = parameters.ev.charge_rates
            elif (
                self.config.devices.electric_vehicles
                and self.config.devices.electric_vehicles[0]
                and self.config.devices.electric_vehicles[0].charge_rates is not None
            ):
                self.ev_possible_charge_values = self.config.devices.electric_vehicles[
                    0
                ].charge_rates
            else:
                warning_msg = "No charge rates provided for electric vehicle - using default."
                logger.warning(warning_msg)
                self.ev_possible_charge_values = [
                    0.0,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1.0,
                ]
        else:
            self.optimize_ev = False

        # Battery AC charge rates — use the battery's configured charge_rates so the
        # optimizer can select partial AC charge power (e.g. 10 %, 50 %, 100 %) instead
        # of always forcing full power.  Falls back to [1.0] when not configured.
        if parameters.pv_battery and parameters.pv_battery.charge_rates:
            self.bat_possible_charge_values = [
                r for r in parameters.pv_battery.charge_rates if r > 0.0
            ] or [1.0]
        elif (
            self.config.devices.batteries
            and self.config.devices.batteries[0]
            and self.config.devices.batteries[0].charge_rates
        ):
            self.bat_possible_charge_values = [
                r for r in self.config.devices.batteries[0].charge_rates if r > 0.0
            ] or [1.0]
        else:
            self.bat_possible_charge_values = [1.0]
        logger.debug("Battery AC charge levels: {}", self.bat_possible_charge_values)

        # Initialize household appliance if applicable
        dishwasher = (
            HomeAppliance(
                parameters=parameters.dishwasher,
                optimization_hours=self.config.optimization.horizon_hours,
                prediction_hours=self.config.prediction.hours,
            )
            if parameters.dishwasher is not None
            else None
        )

        # Initialize the inverter and energy management system
        inverter: Optional[Inverter] = None
        if parameters.inverter:
            inverter = Inverter(
                parameters.inverter,
                battery=battery,
            )

        # Prepare device simulation
        self.simulation.prepare(
            parameters=parameters.ems,
            optimization_hours=self.config.optimization.horizon_hours,
            prediction_hours=self.config.prediction.hours,
            inverter=inverter,  # battery is part of inverter
            ev=ev,
            home_appliance=dishwasher,
        )

        # Setup the DEAP environment and optimization process
        self.setup_deap_environment({"home_appliance": 1 if dishwasher else 0}, start_hour)
        self.toolbox.register(
            "evaluate",
            lambda ind: self.evaluate(ind, parameters, start_hour, worst_case),
        )

        start_time = time.time()
        start_solution, extra_data = self.optimize(parameters.start_solution, ngen=generations)
        elapsed_time = time.time() - start_time
        logger.debug(f"Time evaluate inner: {elapsed_time:.4f} sec.")

        # Perform final evaluation on the best solution
        simulation_result = self.evaluate_inner(start_solution)

        # Prepare results
        discharge_hours_bin, ev_charge_hours_index, washingstart_int = self.split_individual(
            start_solution
        )
        # home appliance may have choosen a different appliance start hour
        if self.simulation.home_appliance:
            washingstart_int = self.simulation.home_appliance_start_hour

        ev_charge_hours_float = (
            [self.ev_possible_charge_values[i] for i in ev_charge_hours_index]
            if ev_charge_hours_index is not None
            else None
        )

        # Simulation may have changed something, use simulation values
        ac_charge_hours = self.simulation.ac_charge_hours
        if ac_charge_hours is None:
            ac_charge_hours = []
        else:
            ac_charge_hours = ac_charge_hours.tolist()
        dc_charge_hours = self.simulation.dc_charge_hours
        if dc_charge_hours is None:
            dc_charge_hours = []
        else:
            dc_charge_hours = dc_charge_hours.tolist()
        discharge = self.simulation.bat_discharge_hours
        if discharge is None:
            discharge = []
        else:
            discharge = discharge.tolist()

        # Visualize the results in PDF
        try:
            from akkudoktoreos.utils.visualize import prepare_visualize

            visualize = {
                "ac_charge": ac_charge_hours,
                "dc_charge": dc_charge_hours,
                "discharge_allowed": discharge,
                "ev_charge_hours_float": ev_charge_hours_float,
                "result": SimulationResult(**simulation_result).model_dump(),
                "ev_obj": self.simulation.ev.to_dict() if self.simulation.ev else None,
                "start_solution": start_solution,
                "washingstart": washingstart_int,
                "extra_data": extra_data,
                "fitness_history": self.fitness_history,
                "fixed_seed": self.fix_seed,
            }

            prepare_visualize(parameters, visualize, start_hour=start_hour)

        except Exception as ex:
            error_msg = f"Visualization failed: {ex}"
            logger.error(error_msg)

        return GeneticSolution(
            **{
                "ac_charge": ac_charge_hours,
                "dc_charge": dc_charge_hours,
                "discharge_allowed": discharge,
                "ev_charge_hours_float": ev_charge_hours_float,
                "result": SimulationResult(**simulation_result),
                "ev_obj": self.simulation.ev,
                "start_solution": start_solution,
                "washingstart": washingstart_int,
            }
        )
