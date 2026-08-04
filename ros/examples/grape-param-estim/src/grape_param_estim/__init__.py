"""Sparse batch estimation and posterior sampling for Grape flights.

The package root deliberately exports no estimator implementation symbols.
Applications import the specific ROS-free submodule they use so importing a
wire-format or progress helper never initializes SciPy, ROS, or a solver.
"""

__all__ = []
