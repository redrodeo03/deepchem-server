"""
Primitives package for DeepChem Server.

This package contains modularized primitive classes that inherit from a base Primitive class.
Each primitive implements the 'run' method to execute its specific functionality.
"""

from .base import Primitive
from .evaluate import Evaluate
from .featurize import Featurize
from .infer import Infer
from .train import Train

__all__ = [
    "Primitive",
    "Featurize",
    "Train",
    "Evaluate",
    "Infer",
]
