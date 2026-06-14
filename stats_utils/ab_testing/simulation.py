"""Simulation helpers for A/B testing workflows.

This module provides the modern import path for the legacy
`ab_testing_utils.sim_utils` implementation.
"""

from ab_testing_utils.sim_utils import SimulateABTest, SimulateSkewedContinuous

__all__ = ["SimulateABTest", "SimulateSkewedContinuous"]
