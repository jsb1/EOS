"""DP algorithm optimisation solution."""

from typing import Optional

from pydantic import Field

from akkudoktoreos.optimization.simulation.solution import SimulationSolution


class DPSolution(SimulationSolution):
    """DP algorithm solution extending the solver-agnostic base.

    Adds DP-specific metadata (states explored, computation time).
    """

    optimizer: str = Field(default="DP")
    total_states_explored: int = Field(
        default=0,
        json_schema_extra={
            "description": "Total number of states evaluated during DP."
        },
    )
    computation_time_ms: float = Field(
        default=0.0,
        json_schema_extra={
            "description": "Computation time in milliseconds."
        },
    )
    dp_start_soc_index: Optional[int] = Field(
        default=None,
        json_schema_extra={
            "description": "Discretized start SoC index of the battery."
        },
    )
    dp_end_soc_index: Optional[int] = Field(
        default=None,
        json_schema_extra={
            "description": "Discretized end SoC index of the battery (optimal terminal state)."
        },
    )
