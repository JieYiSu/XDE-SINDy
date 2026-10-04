"""Common result type returned by every optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class OptimizationResult:
    """Terminal solution and the true-evaluation convergence record."""

    x: np.ndarray
    fun: float
    evaluations: int
    history: list[dict[str, Any]]


__all__ = ["OptimizationResult"]
