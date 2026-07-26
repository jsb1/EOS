"""DP algorithm parameters.

This module defines the Pydantic-based configuration model for the DP solver.
"""

from typing import Optional

from loguru import logger

from akkudoktoreos.optimization.simulation.parameters import (
    OptimizationParameters,
)


class DPOptimizationParameters(OptimizationParameters):
    """DP algorithm optimisation parameters.

    Extends the solver-agnostic `OptimizationParameters` with DP-specific
    config (SoC resolution, charge rates handling).
    """

    @classmethod
    async def _prepare_solver_config(cls) -> None:
        """Set DP-specific config defaults."""
        # DP uses its own resolution parameter; no population/generation config needed.
        logger.info("DP solver config prepared with 1% SoC resolution (101 steps).")

    @classmethod
    async def prepare(cls) -> "Optional[DPOptimizationParameters]":
        """Prepare DP optimization parameters from config, forecast and measurement data.

        Returns:
            DPOptimizationParameters: The fully prepared optimization parameters.

        Raises:
            ValueError: If required configuration values like start time are missing.
        """
        return await super().prepare()  # type: ignore[return-value]
