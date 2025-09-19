"""
DeepChem Server Client Package

A Python client for interacting with the DeepChem server API.
Provides Primitives client for computation tasks and Data client for data operations.
"""

__version__ = "0.1.0"

from .base import BaseClient
from .data import Data
from .primitives.base import Primitive
from .primitives.featurize import Featurize
from .primitives.train import Train
from .primitives.evaluate import Evaluate
from .primitives.infer import Infer
from .settings import Settings

__all__ = [
    "Settings",
    "Data",
    "BaseClient",
    "Primitive",
    "Featurize",
    "Train",
    "Evaluate",
    "Infer",
]
