"""Interpretable candidate dictionaries used by XDE-SINDy.

The dictionary is deliberately explicit: every column has a symbolic name and
an operator specification.  This makes the surrogate auditable and keeps the
search model separate from the numerical regression implementation.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .backend import ComputeBackend

Array = np.ndarray


class PolynomialLibrary:
    """Hierarchical polynomial and harmonic candidate dictionary.

    Bounded inputs are mapped to ``z in [-1, 1]``.  The default dictionary
    contains a constant, linear terms, and squares. Optional aggregate terms
    provide explicit radial, periodic, and adjacent-coupling summaries for
    nonseparable landscapes. Interactions are added only when requested,
    avoiding an unnecessary quadratic expansion in high dimension.
    """

    def __init__(
        self,
        degree: int = 2,
        include_sine: bool = False,
        include_cosine: bool = False,
        feature_names: Optional[Sequence[str]] = None,
        active_features: Optional[Sequence[int]] = None,
        interaction_pairs: Optional[Sequence[Tuple[int, int]]] = None,
        bounds: Optional[Sequence[Tuple[float, float]]] = None,
        include_interactions: bool = True,
        include_all_main_effects: bool = False,
        harmonics: Sequence[int] = (1,),
        include_aggregate_terms: bool = False,
        aggregate_harmonics: Sequence[int] = (1,),
        compute_backend: Optional[ComputeBackend] = None,
    ) -> None:
        if degree < 1:
            raise ValueError("degree must be at least 1")
        self.degree = int(degree)
        self.include_sine = bool(include_sine)
        self.include_cosine = bool(include_cosine)
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.active_features = (
            None
            if active_features is None
            else sorted(set(int(i) for i in active_features))
        )
        self.interaction_pairs = (
            None
            if interaction_pairs is None
            else {
                tuple(sorted((int(left), int(right))))
                for left, right in interaction_pairs
            }
        )
        self.include_interactions = bool(include_interactions)
        self.include_all_main_effects = bool(include_all_main_effects)
        self.harmonics = tuple(int(value) for value in harmonics if int(value) > 0)
        if not self.harmonics:
            self.harmonics = (1,)
        self.include_aggregate_terms = bool(include_aggregate_terms)
        self.aggregate_harmonics = tuple(
            int(value) for value in aggregate_harmonics if int(value) > 0
        )
        if not self.aggregate_harmonics:
            self.aggregate_harmonics = (1,)
        self.bounds = None if bounds is None else np.asarray(bounds, dtype=float)
        if self.bounds is not None and (
            self.bounds.ndim != 2 or self.bounds.shape[1] != 2
        ):
            raise ValueError("bounds must have shape (n_features, 2)")
        if self.bounds is not None and np.any(self.bounds[:, 1] <= self.bounds[:, 0]):
            raise ValueError("each upper bound must exceed its lower bound")
        self.names: List[str] = []
        self.n_features = 0
        self._specs: List[Tuple[Any, ...]] = []
        self.compute_backend = compute_backend or ComputeBackend("cpu")

    def set_compute_backend(self, backend: ComputeBackend) -> None:
        """Share one audited backend across all XDE-SINDy model layers."""
        self.compute_backend = backend

    def set_active_features(
        self,
        active_features: Optional[Sequence[int]],
        interaction_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> None:
        """Update variables and pairs allowed to form interaction terms."""
        self.active_features = (
            None
            if active_features is None
            else sorted(set(int(i) for i in active_features))
        )
        if interaction_pairs is not None:
            self.interaction_pairs = {
                tuple(sorted((int(left), int(right))))
                for left, right in interaction_pairs
            }
        self._specs = []

    def set_interaction_pairs(
        self, interaction_pairs: Optional[Sequence[Tuple[int, int]]]
    ) -> None:
        self.interaction_pairs = (
            None
            if interaction_pairs is None
            else {
                tuple(sorted((int(left), int(right))))
                for left, right in interaction_pairs
            }
        )
        self._specs = []

    def normalize(self, x: Array) -> Array:
        values = np.asarray(x, dtype=float)
        if self.bounds is None:
            return values
        center = (self.bounds[:, 0] + self.bounds[:, 1]) * 0.5
        half_range = (self.bounds[:, 1] - self.bounds[:, 0]) * 0.5
        return (values - center) / half_range

    def denormalize(self, z: Array) -> Array:
        values = np.asarray(z, dtype=float)
        if self.bounds is None:
            return values
        center = (self.bounds[:, 0] + self.bounds[:, 1]) * 0.5
        half_range = (self.bounds[:, 1] - self.bounds[:, 0]) * 0.5
        return center + values * half_range

    def _build_specs(self, n_features: int) -> None:
        active = (
            self.active_features
            if self.active_features is not None
            else list(range(n_features))
        )
        if any(index < 0 or index >= n_features for index in active):
            raise ValueError("active_features contains an invalid feature index")
        names = self.feature_names or [f"x{i}" for i in range(n_features)]
        if len(names) != n_features:
            raise ValueError("feature_names must match the number of input features")

        specs: List[Tuple[Any, ...]] = [("const",)]
        term_names = ["1"]
        main_features = list(range(n_features)) if self.include_all_main_effects else active
        if self.degree >= 1:
            for index in main_features:
                specs.append(("linear", index))
                term_names.append(names[index])
        if self.degree >= 2:
            for index in main_features:
                specs.append(("square", index))
                term_names.append(f"{names[index]}^2")
            if self.include_interactions:
                if self.interaction_pairs is None:
                    pairs = list(combinations(active, 2))
                else:
                    pairs = {
                        pair
                        for pair in self.interaction_pairs
                        if 0 <= pair[0] < n_features
                        and 0 <= pair[1] < n_features
                        and pair[0] != pair[1]
                    }
                    for index in range(n_features - 1):
                        pairs.add((index, index + 1))
                    pairs = sorted(pairs)
                for left, right in pairs:
                    specs.append(("interaction", left, right))
                    term_names.append(f"{names[left]}*{names[right]}")
        if self.include_sine:
            for harmonic in self.harmonics:
                for index in active:
                    specs.append(("sine", harmonic, index))
                    term_names.append(f"sin({harmonic}*pi*{names[index]})")
        if self.include_cosine:
            for harmonic in self.harmonics:
                for index in active:
                    specs.append(("cosine", harmonic, index))
                    term_names.append(f"cos({harmonic}*pi*{names[index]})")
        if self.include_aggregate_terms:
            # These features are explicit dictionary columns. Sparse regression
            # decides whether the global summaries are retained in the formula.
            specs.append(("rms_x",))
            term_names.append("rms(x)")
            specs.append(("radial_exp",))
            term_names.append("exp(-0.2*rms(x))")
            for harmonic in self.aggregate_harmonics:
                specs.append(("mean_cosine", harmonic))
                if harmonic == 1:
                    term_names.append("mean(cos(2*pi*x))")
                else:
                    term_names.append(f"mean(cos({2 * harmonic}*pi*x))")
            if n_features >= 2:
                specs.append(("chain_residual",))
                term_names.append("mean((x_i^2-x_{i+1})^2)")
        self._specs = specs
        self.names = term_names
        self.n_features = n_features

    def transform(self, x: Array) -> Array:
        values = np.asarray(x, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2:
            raise ValueError("x must be a 1D or 2D array")
        if not np.all(np.isfinite(values)):
            raise ValueError("x must contain only finite values")
        if self._specs == [] or self.n_features != values.shape[1]:
            self._build_specs(values.shape[1])
        return np.asarray(
            self.compute_backend.dictionary_transform(
                values,
                self._specs,
                self.bounds,
                return_device=False,
            ),
            dtype=float,
        )

    def transform_device(self, x: Array) -> Any:
        """Build the explicit dictionary on the selected numerical device."""
        device_input = self.compute_backend.is_device_array(x)
        values = x if device_input else np.asarray(x, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2:
            raise ValueError("x must be a 1D or 2D array")
        if device_input:
            if not self.compute_backend.scalar(
                self.compute_backend.array_module.all(
                    self.compute_backend.array_module.isfinite(values)
                )
            ):
                raise ValueError("x must contain only finite values")
        elif not np.all(np.isfinite(values)):
            raise ValueError("x must contain only finite values")
        if self._specs == [] or self.n_features != values.shape[1]:
            self._build_specs(values.shape[1])
        return self.compute_backend.dictionary_transform(
            values,
            self._specs,
            self.bounds,
            return_device=True,
        )


__all__ = ["Array", "PolynomialLibrary"]
