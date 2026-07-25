"""Genetic algorithm optimisation solution."""

from akkudoktoreos.optimization.simulation.solution import SimulationSolution


class GeneticSolution(SimulationSolution):
    """Genetic algorithm solution extending the solver-agnostic base.

    Currently a thin alias for :class:`SimulationSolution` to maintain
    backward compatibility. Subclasses can add GA-specific extensions.
    """

    pass
