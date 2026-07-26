"""GENETIC algorithm paramters.

This module defines the Pydantic-based configuration and input parameter models
used in the energy optimization routines, including photovoltaic forecasts,
electricity pricing, and system component parameters.

It also provides a method to assemble these parameters from predictions,
forecasts, and fallback defaults, preparing them for optimization runs.
"""

from typing import Optional

from loguru import logger

from akkudoktoreos.optimization.simulation.parameters import (
    EnergyManagementParameters,
    OptimizationParameters,
)

# Do not import directly from akkudoktoreos.core.coreabc
# EnergyManagementSystemMixin - Creates circular dependency with ems.py
# StartMixin                  - Creates circular dependency with ems.py


# Backward-compatible alias: EnergyManagementParameters lives in simulation/parameters.py
GeneticEnergyManagementParameters = EnergyManagementParameters


class GeneticOptimizationParameters(OptimizationParameters):
    """Genetic algorithm optimisation parameters.

    Extends the solver-agnostic `OptimizationParameters` with GA-specific
    config defaults (individuals, generations, penalties).
    """

    @classmethod
    async def _prepare_solver_config(cls) -> None:
        """Set GA-specific config defaults."""
        if cls.config.optimization.genetic.individuals is None:
            logger.info("Genetic individuals unknown - defaulting to 300.")
            cls.config.optimization.genetic.individuals = 300
        if cls.config.optimization.genetic.generations is None:
            logger.info("Genetic generations unknown - defaulting to 400.")
            cls.config.optimization.genetic.generations = 400
        if "ev_soc_miss" not in cls.config.optimization.genetic.penalties:
            logger.info("Genetic penalties unknown - defaulting to ev_soc_miss = 10.")
            cls.config.optimization.genetic.penalties["ev_soc_miss"] = 10

    @classmethod
    async def prepare(cls) -> "Optional[GeneticOptimizationParameters]":
        """Prepare GA optimization parameters from config, forecast and measurement data.

        Fills in values needed for optimization from available configuration, predictions and
        measurements. If some data is missing, default or demo values are used.

        Parameters start by definition of the genetic algorithm at hour 0 of the actual date
        (not at start datetime of energy management run)

        Returns:
            GeneticOptimizationParameters: The fully prepared optimization parameters.

        Raises:
            ValueError: If required configuration values like start time are missing.
        """
        return await super().prepare()  # type: ignore[return-value]
