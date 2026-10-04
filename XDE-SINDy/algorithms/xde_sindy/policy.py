"""Interpretable state-to-parameter policy for the DE control layer."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .backend import ComputeBackend
from .dictionary import Array, PolynomialLibrary
from .surrogate import EnsembleSparseRegressor

class SparsePolicyModel:
    """Low-order interpretable map from search state to F and CR."""

    def __init__(
        self,
        random_state: Optional[int] = None,
        state_names: Optional[Sequence[str]] = None,
        target_transform: str = "robust_affine",
        target_clip: float = 8.0,
        local_correction: bool = True,
        local_k: int = 5,
        max_distance_reference: int = 256,
        compute_backend: Optional[ComputeBackend] = None,
    ) -> None:
        self.compute_backend = compute_backend or ComputeBackend("cpu")
        self.state_names = list(state_names or [
            "diversity", "stagnation", "improvement_rate", "uncertainty"
        ])
        library = PolynomialLibrary(
            degree=1,
            feature_names=self.state_names,
            include_interactions=False,
            compute_backend=self.compute_backend,
        )
        self.f_model = EnsembleSparseRegressor(
            library=library, n_models=4, threshold=0.05,
            max_terms=max(2, len(self.state_names)), random_state=random_state,
            target_transform=target_transform,
            target_clip=target_clip,
            local_correction=local_correction,
            local_k=local_k,
            max_distance_reference=max_distance_reference,
            compute_backend=self.compute_backend,
        )
        cr_library = PolynomialLibrary(
            degree=1,
            feature_names=self.state_names,
            include_interactions=False,
            compute_backend=self.compute_backend,
        )
        self.cr_model = EnsembleSparseRegressor(
            library=cr_library, n_models=4, threshold=0.05,
            max_terms=max(2, len(self.state_names)),
            random_state=None if random_state is None else random_state + 1,
            target_transform=target_transform,
            target_clip=target_clip,
            local_correction=local_correction,
            local_k=local_k,
            max_distance_reference=max_distance_reference,
            compute_backend=self.compute_backend,
        )
        self.fitted = False
        self.quality_score = 0.0
        self.f_range = (0.25, 1.0)
        self.cr_range = (0.05, 0.99)

    def fit(self, states: Array, f_values: Array, cr_values: Array) -> "SparsePolicyModel":
        states = np.asarray(states, dtype=float)
        f_values = np.asarray(f_values, dtype=float).reshape(-1)
        cr_values = np.asarray(cr_values, dtype=float).reshape(-1)
        if len(states) < 3:
            return self
        self.f_model.fit(states, f_values)
        self.cr_model.fit(states, cr_values)
        self.fitted = True
        self.quality_score = min(self.f_model.quality_score, self.cr_model.quality_score)
        return self

    def predict(self, state: Array) -> Tuple[float, float]:
        if not self.fitted:
            raise RuntimeError("fit must be called before prediction")
        f_value, _ = self.f_model.predict_with_uncertainty(np.asarray(state).reshape(1, -1))
        cr_value, _ = self.cr_model.predict_with_uncertainty(np.asarray(state).reshape(1, -1))
        return float(np.clip(f_value[0], 0.25, 1.0)), float(np.clip(cr_value[0], 0.05, 0.99))


__all__ = ["SparsePolicyModel"]

