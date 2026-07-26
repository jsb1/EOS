"""Dynamic Programming solver for energy optimization.

This module provides a DP-based optimizer that finds the globally optimal
energy management policy via Bellman optimality over a discretized SoC space.

Features:
- 1% SoC resolution (101 discrete levels) by default
- Battery + EV + Home Appliance optimization
- Full GA parity (charge rates, DC flag, visualization, worst-case mode)
- DP-as-GA-warmup (HYBRID mode)
"""

from akkudoktoreos.optimization.dp.dpparams import DPOptimizationParameters
from akkudoktoreos.optimization.dp.dpoptimizer import DPOptimizer
from akkudoktoreos.optimization.dp.dpsolution import DPSolution

__all__ = [
    "DPOptimizer",
    "DPOptimizationParameters",
    "DPSolution",
]
