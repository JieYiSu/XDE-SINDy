"""Optional EvoX adapters for running XDE-SINDy through EvoX workflows.

The adapter keeps the existing XDE-SINDy engine and its FE accounting intact.
EvoX owns the Algorithm/Problem boundary and receives every true-evaluation
batch through its normal ``Algorithm.evaluate`` proxy. This module is
optional: the core NumPy/CuPy implementation does not depend on EvoX.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import torch

from .core import XDESindy

try:  # EvoX is an optional experiment dependency.
    from evox.core import Algorithm, Problem

    _EVOX_IMPORT_ERROR: Exception | None = None
except Exception as error:  # pragma: no cover - depends on the local environment
    _EVOX_IMPORT_ERROR = error

    class Algorithm:  # type: ignore[no-redef]
        """Import-time placeholder used when EvoX is not installed."""

    class Problem:  # type: ignore[no-redef]
        """Import-time placeholder used when EvoX is not installed."""


def _require_evox() -> None:
    if _EVOX_IMPORT_ERROR is not None:
        raise ImportError(
            "EvoX integration requires EvoX 1.4 or newer. "
            "Install the optional requirements-evox.txt dependencies."
        ) from _EVOX_IMPORT_ERROR


def _torch_to_cupy(value: torch.Tensor) -> Any:
    """Export a Torch CUDA tensor to CuPy without a host round trip."""
    if not value.is_cuda:
        return value.detach().cpu().numpy()
    try:
        import cupy as cp
    except ImportError as error:  # pragma: no cover - GPU environment specific
        raise ImportError(
            "EvoX XDE-SINDy integration needs CuPy for the existing backend"
        ) from error
    return cp.from_dlpack(value.detach())


class EvoXProblemAdapter(Problem):
    """Wrap a Torch batch evaluator as an EvoX ``Problem``.

    ``evaluator`` must accept a tensor shaped ``(batch, dimension)`` and return
    one scalar fitness per row. It should perform all objective arithmetic on
    the tensor's device.
    """

    def __init__(
        self,
        evaluator: Callable[[torch.Tensor], torch.Tensor],
        dimension: int,
    ) -> None:
        _require_evox()
        super().__init__()
        self.evaluator = evaluator
        self.dimension = int(dimension)
        self.evaluations = 0

    def evaluate(self, population: torch.Tensor) -> torch.Tensor:
        values = self.evaluator(population).reshape(-1)
        if values.ndim != 1 or values.shape[0] != population.shape[0]:
            raise ValueError(
                "EvoX problem evaluator must return one value per population row"
            )
        self.evaluations += int(population.shape[0])
        return values


class EvoXObjectiveBridge:
    """Connect an EvoX ``Algorithm.evaluate`` proxy to XDE-SINDy.

    XDE-SINDy still stores its archive and performs its white-box decisions in
    the existing engine. The bridge only converts the NumPy candidate batch to
    a Torch tensor and exports the returned Torch fitness through DLPack.
    """

    def __init__(
        self,
        device: str | torch.device = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self._evaluate_proxy: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
        self._backend: Any = None

    def bind(
        self,
        evaluate_proxy: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self._evaluate_proxy = evaluate_proxy

    def evaluate_batch_device(self, points: np.ndarray, *, backend: Any) -> Any:
        if self._evaluate_proxy is None:
            raise RuntimeError("EvoX objective bridge is not bound to a workflow")
        self._backend = backend
        tensor = torch.as_tensor(
            np.asarray(points, dtype=float),
            dtype=torch.float64,
            device=self.device,
        )
        fitness = self._evaluate_proxy(tensor).reshape(-1)
        if fitness.shape[0] != tensor.shape[0]:
            raise ValueError("EvoX evaluation returned the wrong batch size")
        return _torch_to_cupy(fitness)

    def __call__(self, point: np.ndarray) -> float:
        if self._backend is None:
            raise RuntimeError("EvoX objective bridge has not evaluated a batch yet")
        values = self.evaluate_batch_device(
            np.asarray(point, dtype=float).reshape(1, -1),
            backend=self._backend,
        )
        return float(self._backend.to_numpy(values).reshape(-1)[0])


class EvoXXDESindy(Algorithm):
    """Expose the existing XDE-SINDy engine through the EvoX API.

    The current engine is FE-driven and may select several small real batches
    inside one logical run. Therefore the adapter executes that complete run
    from EvoX ``init_step`` while every objective batch still passes through
    EvoX's ``Problem.evaluate`` method and monitor hooks. This preserves the
    current algorithm and accuracy while making it usable in EvoX experiment
    harnesses. A later native Torch port can split the engine into per-step
    updates without changing this public boundary.
    """

    def __init__(
        self,
        engine: XDESindy,
        objective_bridge: EvoXObjectiveBridge,
    ) -> None:
        _require_evox()
        super().__init__()
        self.engine = engine
        self.objective_bridge = objective_bridge
        self.result: Any = None
        self.completed = False
        self.pop = torch.empty((0, engine.dimension), dtype=torch.float64)
        self.fit = torch.empty((0,), dtype=torch.float64)

        # The existing engine must use the bridge's device-side batch path.
        self.engine.objective = objective_bridge
        self.engine.gpu_objective = True

    def init_step(self) -> None:
        self.objective_bridge.bind(self.evaluate)
        self.result = self.engine.minimize()
        self.completed = True
        device = self.objective_bridge.device
        self.pop = torch.as_tensor(
            np.asarray(self.result.x, dtype=float).reshape(1, -1),
            dtype=torch.float64,
            device=device,
        )
        self.fit = torch.as_tensor(
            [float(self.result.fun)],
            dtype=torch.float64,
            device=device,
        )

    def step(self) -> None:
        if not self.completed:
            self.init_step()

    def final_step(self) -> None:
        if not self.completed:
            self.init_step()

    def record_step(self) -> dict[str, torch.Tensor]:
        evaluations = 0 if self.result is None else int(self.result.evaluations)
        return {
            "pop": self.pop,
            "fit": self.fit,
            "evaluations": torch.tensor(
                evaluations,
                dtype=torch.int64,
                device=self.pop.device,
            ),
        }


__all__ = [
    "EvoXObjectiveBridge",
    "EvoXProblemAdapter",
    "EvoXXDESindy",
]
