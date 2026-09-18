"""Exception hierarchy.

The scientific policy of this platform (spec sections 8, 12, 154) is that
the software never silently changes the physics a user asked for.  Every
condition that would require altering grid spacing, bandwidth, recording
length, acquisition or model extent raises instead of being auto-fixed,
and the exception carries the defensible alternatives.
"""

from __future__ import annotations


class Sim3DError(Exception):
    """Base class for all sim3d errors."""


class ConfigError(Sim3DError):
    """Malformed or self-inconsistent configuration."""


class UnitsError(Sim3DError):
    """An attempt to mix or misdeclare units."""


class ValidationError(Sim3DError):
    """A model or state failed a QC check that makes it unphysical."""


class StabilityError(Sim3DError):
    """The requested numerical setup violates a stability criterion."""


class InfeasibleExperiment(Sim3DError):
    """The requested experiment is computationally unreasonable.

    Raised instead of silently coarsening the grid, lowering the frequency,
    dropping shots or cropping the model.  ``alternatives`` holds the
    scientifically defensible options that the user must choose between
    explicitly.
    """

    def __init__(self, message: str, alternatives: list[str] | None = None):
        self.alternatives = list(alternatives or [])
        if self.alternatives:
            bullets = "\n".join(f"  - {a}" for a in self.alternatives)
            message = f"{message}\n\nScientifically defensible alternatives:\n{bullets}"
        super().__init__(message)


class BackendError(Sim3DError):
    """A compute backend is unavailable or was used incorrectly."""
