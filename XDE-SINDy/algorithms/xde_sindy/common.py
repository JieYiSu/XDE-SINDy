"""Shared numerical utilities for the XDE-SINDy package."""

from __future__ import annotations

from typing import List, Mapping, Optional, Sequence

import numpy as np

from .backend import ComputeBackend

Array = np.ndarray


def _rms_distance_matrix(
    query: Array,
    reference: Array,
    compute_backend: ComputeBackend | None = None,
) -> Array:
    """Return pairwise root-mean-square coordinate distances."""
    if compute_backend is not None:
        return compute_backend.rms_distance_matrix(query, reference)
    query_values = np.asarray(query, dtype=float)
    reference_values = np.asarray(reference, dtype=float)
    if query_values.ndim != 2 or reference_values.ndim != 2:
        raise ValueError("query and reference must be 2D arrays")
    if query_values.shape[1] != reference_values.shape[1]:
        raise ValueError("query and reference dimensions must match")
    dimension = max(int(query_values.shape[1]), 1)
    query_norm = np.sum(query_values * query_values, axis=1, keepdims=True)
    reference_norm = np.sum(
        reference_values * reference_values, axis=1, keepdims=True
    ).T
    squared = (
        query_norm
        + reference_norm
        - 2.0 * (query_values @ reference_values.T)
    ) / dimension
    return np.sqrt(np.maximum(squared, 0.0))


def _resolve_metadata(
    values: Optional[Sequence[str] | Mapping[int, str]],
    dimension: int,
    default_prefix: str,
    field_name: str,
) -> List[str]:
    """Normalize labels, units, or descriptions to one value per variable."""
    if values is None:
        if default_prefix:
            return [f"{default_prefix}{index}" for index in range(dimension)]
        return ["" for _ in range(dimension)]
    if isinstance(values, Mapping):
        result = []
        for index in range(dimension):
            value = values.get(index, values.get(str(index), None))
            result.append(str(value) if value is not None else "")
        return result
    if isinstance(values, str):
        values = [values]
    result = [str(value) for value in values]
    if len(result) != dimension:
        raise ValueError(f"{field_name} must contain exactly {dimension} entries")
    return result


def latin_hypercube(
    n_samples: int,
    n_dimensions: int,
    rng: np.random.Generator,
) -> Array:
    """Draw a space-filling Latin-hypercube initial design."""
    if n_samples < 1 or n_dimensions < 1:
        raise ValueError("n_samples and n_dimensions must be positive")
    result = np.empty((n_samples, n_dimensions), dtype=float)
    for dimension in range(n_dimensions):
        permutation = rng.permutation(n_samples)
        result[:, dimension] = (permutation + rng.random(n_samples)) / n_samples
    return result


__all__ = ["Array", "_rms_distance_matrix", "_resolve_metadata", "latin_hypercube"]
