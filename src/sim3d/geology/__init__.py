"""Parametric 3D geological model building (spec sections 17-25)."""

from .surfaces import (
    Anticline, Composite, Dipping, Flat, Surface, Syncline,
)
from .heterogeneity import GaussianField, HeterogeneitySpec, random_field
from .faults import Fault, FaultSet
from .facies import FACIES, Facies
from .builder import GeologyModel, Layer, build_geology
from .templates import TEMPLATES, template

__all__ = [
    "Surface", "Flat", "Dipping", "Anticline", "Syncline", "Composite",
    "GaussianField", "HeterogeneitySpec", "random_field",
    "Fault", "FaultSet", "FACIES", "Facies",
    "GeologyModel", "Layer", "build_geology", "TEMPLATES", "template",
]
