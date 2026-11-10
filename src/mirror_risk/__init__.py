"""Finite-size simulations for mirror, direct-gate, and linear GD dynamics."""

from .dynamics import (
    ConvergenceConfig,
    DynamicsConfig,
    DynamicsRun,
    inverse_mirror_map,
    project_l1_ball,
)
from .experiment import ExperimentConfig, prepare_problems, run_experiment, write_summaries
from .problem import LinearRegressionProblem, ProblemSpec

__all__ = [
    "ConvergenceConfig",
    "DynamicsConfig",
    "DynamicsRun",
    "ExperimentConfig",
    "LinearRegressionProblem",
    "ProblemSpec",
    "inverse_mirror_map",
    "project_l1_ball",
    "prepare_problems",
    "run_experiment",
    "write_summaries",
]

__version__ = "0.1.0"
