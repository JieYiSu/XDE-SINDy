"""Public API for the XDE-SINDy optimizer."""

from .common import _rms_distance_matrix, latin_hypercube
from .backend import ComputeBackend
from .dictionary import PolynomialLibrary
from .core import XDESindy
from .policy import SparsePolicyModel
from .sparse import _ridge_fit
from .surrogate import EnsembleSparseRegressor

__all__ = [
    "latin_hypercube",
    "PolynomialLibrary",
    "EnsembleSparseRegressor",
    "SparsePolicyModel",
    "XDESindy",
    "ComputeBackend",
]
