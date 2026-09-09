"""Experiment orchestration: configuration in, results out."""

from .pipeline import (
    ExperimentResult, Pipeline, STAGES,
)

__all__ = ["Pipeline", "ExperimentResult", "STAGES"]
