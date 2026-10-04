"""Auditable NumPy/CuPy execution backend for XDE-SINDy.

The backend changes only where dense numerical kernels run. Symbolic term
definitions, sparse coefficients, selection rules, and all white-box audit
records remain identical across CPU and CUDA execution.
"""

from __future__ import annotations

from collections import Counter
from importlib import import_module
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence
import warnings

import numpy as np


class ComputeBackend:
    """Dispatch dense XDE-SINDy kernels to NumPy or CuPy with an audit trail."""

    ACCELERATED_COMPONENTS = (
        "initial_design_scaling",
        "de_candidate_generation",
        "candidate_feature_construction",
        "symbolic_dictionary",
        "sparse_regression_linear_algebra",
        "ensemble_prediction",
        "distance_uncertainty",
        "interaction_screening",
        "candidate_ranking",
    )

    def __init__(
        self,
        requested: str = "auto",
        device: int = 0,
        min_elements: int = 1,
    ) -> None:
        requested = str(requested).strip().lower()
        if requested not in {"auto", "cpu", "cuda"}:
            raise ValueError("compute backend must be auto, cpu, or cuda")
        if int(device) < 0:
            raise ValueError("GPU device must be nonnegative")
        if int(min_elements) < 1:
            raise ValueError("GPU minimum element count must be positive")

        self.requested = requested
        self.device_index = int(device)
        self.min_elements = int(min_elements)
        self.resolved = "cpu"
        self.fallback_reason: str | None = None
        self.device_name = "CPU"
        self.array_module = np
        self._cupy = None
        self._kernel_calls: Counter[str] = Counter()
        self._gpu_kernel_calls: Counter[str] = Counter()
        self._cpu_fallback_calls: Counter[str] = Counter()
        self._host_to_device_bytes = 0
        self._device_to_host_bytes = 0
        self._host_to_device_transfers = 0
        self._device_to_host_transfers = 0
        self._dictionary_plan_cache: dict[tuple[tuple[Any, ...], ...], dict[str, Any]] = {}
        self._dictionary_plan_builds = 0
        self._dictionary_plan_cache_hits = 0
        self._dictionary_bounds_cache: dict[tuple[int, str], tuple[np.ndarray, Any, Any]] = {}
        self._dictionary_bounds_cache_hits = 0

        if requested != "cpu":
            try:
                default_cache_root = Path(
                    os.environ.get("LOCALAPPDATA", str(Path.home() / ".cache"))
                ) / "XDE_SINDy_GPU"
                cache_dir = Path(
                    os.environ.get("XDE_GPU_CACHE_DIR", str(default_cache_root))
                ).resolve()
                temporary_dir = cache_dir / "tmp"
                cache_dir.mkdir(parents=True, exist_ok=True)
                temporary_dir.mkdir(parents=True, exist_ok=True)
                os.environ.setdefault("CUPY_CACHE_DIR", str(cache_dir))
                os.environ["TMP"] = str(temporary_dir)
                os.environ["TEMP"] = str(temporary_dir)
                tempfile.tempdir = str(temporary_dir)
                # The pip ``cupy-cuda12x`` wheel bundles the CUDA runtime in
                # several site-packages directories rather than exposing one
                # Toolkit root. CuPy consequently emits a benign path-detection
                # warning even when the driver and runtime are usable. Suppress
                # only this narrowly identified warning; initialization errors
                # remain visible and are handled by the normal fallback logic.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"CUDA path could not be detected\..*",
                        category=UserWarning,
                        module=r"cupy\._environment",
                    )
                    cupy = import_module("cupy")
                device_count = int(cupy.cuda.runtime.getDeviceCount())
                if self.device_index >= device_count:
                    raise RuntimeError(
                        f"CUDA device {self.device_index} is unavailable; "
                        f"detected {device_count} device(s)"
                    )
                cupy.cuda.Device(self.device_index).use()
                probe = cupy.zeros(1, dtype=cupy.float64)
                probe.sum().item()
                properties = cupy.cuda.runtime.getDeviceProperties(self.device_index)
                raw_name = properties.get("name", "CUDA device")
                if isinstance(raw_name, bytes):
                    raw_name = raw_name.decode("utf-8", errors="replace")
                self._cupy = cupy
                self.array_module = cupy
                self.resolved = "cuda"
                self.device_name = str(raw_name)
            except Exception as error:
                self.fallback_reason = f"{type(error).__name__}: {error}"
                if requested == "cuda":
                    raise RuntimeError(
                        "CUDA execution was requested but CuPy could not initialize. "
                        "Install a CUDA-matched CuPy package, for example cupy-cuda12x, "
                        "or set compute_backend to auto/cpu."
                    ) from error

    @property
    def cuda_enabled(self) -> bool:
        return self.resolved == "cuda"

    def uses_cuda(self, element_count: int, operation: str) -> bool:
        self._kernel_calls[str(operation)] += 1
        use_cuda = self.cuda_enabled and int(element_count) >= self.min_elements
        if use_cuda:
            self._gpu_kernel_calls[str(operation)] += 1
        else:
            self._cpu_fallback_calls[str(operation)] += 1
        return use_cuda

    def _to_device(self, value: Any, xp: Any) -> Any:
        if xp is np:
            return np.asarray(value)
        if isinstance(value, xp.ndarray):
            return value
        array = np.asarray(value)
        self._host_to_device_bytes += int(array.nbytes)
        self._host_to_device_transfers += 1
        return xp.asarray(array)

    def to_device(self, value: Any) -> Any:
        """Move an array to this backend's active device when CUDA is enabled."""
        return self._to_device(value, self.array_module)

    def to_numpy(self, value: Any) -> np.ndarray:
        if self._cupy is not None and isinstance(value, self._cupy.ndarray):
            self._device_to_host_bytes += int(value.nbytes)
            self._device_to_host_transfers += 1
            return self._cupy.asnumpy(value)
        return np.asarray(value)

    def _is_device_array(self, value: Any) -> bool:
        return self._cupy is not None and isinstance(value, self._cupy.ndarray)

    def is_device_array(self, value: Any) -> bool:
        """Return whether ``value`` already resides on the active CUDA device."""
        return self._is_device_array(value)

    def scalar(self, value: Any) -> float:
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)

    @staticmethod
    def _element_count(value: Any) -> int:
        size = getattr(value, "size", None)
        return int(size if size is not None else np.size(value))

    def xp_for(self, element_count: int, operation: str) -> Any:
        return self.array_module if self.uses_cuda(element_count, operation) else np

    def arrays_for(self, operation: str, *values: Any) -> tuple[Any, tuple[Any, ...]]:
        """Select one array module and co-locate all values for an operation."""
        if any(self._is_device_array(value) for value in values):
            self._kernel_calls[str(operation)] += 1
            self._gpu_kernel_calls[str(operation)] += 1
            xp = self.array_module
        else:
            element_count = sum(self._element_count(value) for value in values)
            xp = self.xp_for(element_count, operation)
        return xp, tuple(self._to_device(value, xp) for value in values)

    def matmul(
        self,
        left: Any,
        right: Any,
        *,
        operation: str = "matmul",
        return_device: bool = False,
    ) -> Any:
        size = self._element_count(left) + self._element_count(right)
        if self._is_device_array(left) or self._is_device_array(right):
            # A host fallback would both synchronize and reject a CuPy input.
            self._kernel_calls[str(operation)] += 1
            self._gpu_kernel_calls[str(operation)] += 1
            xp = self.array_module
        else:
            xp = self.xp_for(size, operation)
        result = self._to_device(left, xp) @ self._to_device(right, xp)
        return result if return_device else self.to_numpy(result)

    def rms_distance_matrix(self, query: Any, reference: Any) -> np.ndarray:
        query_values = np.asarray(query, dtype=float)
        reference_values = np.asarray(reference, dtype=float)
        if query_values.ndim != 2 or reference_values.ndim != 2:
            raise ValueError("query and reference must be 2D arrays")
        if query_values.shape[1] != reference_values.shape[1]:
            raise ValueError("query and reference dimensions must match")
        xp = self.xp_for(
            query_values.size + reference_values.size,
            "rms_distance",
        )
        q = self._to_device(query_values, xp)
        r = self._to_device(reference_values, xp)
        dimension = max(int(query_values.shape[1]), 1)
        query_norm = xp.sum(q * q, axis=1, keepdims=True)
        reference_norm = xp.sum(r * r, axis=1, keepdims=True).T
        squared = (query_norm + reference_norm - 2.0 * (q @ r.T)) / dimension
        return self.to_numpy(xp.sqrt(xp.maximum(squared, 0.0)))

    def distance_uncertainty(
        self,
        query: Any,
        reference: Any,
        residuals: Any,
        *,
        local_k: int,
        bandwidth: float,
        target_scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fuse distance, local correction, spread, and trust-weight kernels.

        The uncertainty path only needs four row-wise statistics. Keeping the
        pairwise distance matrix on the selected device avoids transferring the
        full ``n_query x n_reference`` matrix to the host before selecting the
        nearest neighbours and computing residual weights.
        """
        query_shape = tuple(getattr(query, "shape", np.shape(query)))
        reference_shape = tuple(getattr(reference, "shape", np.shape(reference)))
        residual_shape = tuple(getattr(residuals, "shape", np.shape(residuals)))
        if len(query_shape) != 2 or len(reference_shape) != 2:
            raise ValueError("query and reference must be 2D arrays")
        if query_shape[1] != reference_shape[1]:
            raise ValueError("query and reference dimensions must match")
        if len(residual_shape) not in {1, 2}:
            raise ValueError("residuals must be a one-dimensional array")
        if int(np.prod(residual_shape)) != reference_shape[0]:
            raise ValueError("residuals must align with reference rows")
        if int(local_k) < 1:
            raise ValueError("local_k must be positive")
        if reference_shape[0] < 1:
            raise ValueError("reference must contain at least one point")

        xp, (query_xp, reference_xp, residuals_xp) = self.arrays_for(
            "distance_uncertainty",
            query,
            reference,
            residuals,
        )
        query_xp = query_xp.astype(xp.float64, copy=False)
        reference_xp = reference_xp.astype(xp.float64, copy=False)
        residuals_xp = residuals_xp.reshape(-1).astype(xp.float64, copy=False)
        dimension = max(int(query_shape[1]), 1)
        query_norm = xp.sum(query_xp * query_xp, axis=1, keepdims=True)
        reference_norm = xp.sum(
            reference_xp * reference_xp,
            axis=1,
            keepdims=True,
        ).T
        squared = (
            query_norm
            + reference_norm
            - 2.0 * (query_xp @ reference_xp.T)
        ) / dimension
        pairwise = xp.sqrt(xp.maximum(squared, 0.0))

        distances = xp.min(pairwise, axis=1)
        neighbor_count = min(int(local_k), int(reference_shape[0]))
        neighbor_indices = xp.argpartition(
            pairwise,
            neighbor_count - 1,
            axis=1,
        )[:, :neighbor_count]
        neighbor_distances = xp.take_along_axis(
            pairwise,
            neighbor_indices,
            axis=1,
        )
        weights = xp.exp(-neighbor_distances / max(float(bandwidth), 1e-8))
        weights /= xp.maximum(xp.sum(weights, axis=1, keepdims=True), 1e-12)
        neighbor_residuals = residuals_xp[neighbor_indices]
        correction = xp.sum(weights * neighbor_residuals, axis=1)
        residual_scale = max(
            1e-8,
            self.scalar(xp.median(xp.abs(residuals_xp))),
            0.05 * abs(float(target_scale)),
        )
        correction = xp.clip(
            correction,
            -2.0 * residual_scale,
            2.0 * residual_scale,
        )
        local_spread = xp.sqrt(
            xp.maximum(
                xp.sum(
                    weights
                    * (neighbor_residuals - correction[:, None]) ** 2,
                    axis=1,
                ),
                0.0,
            )
        )
        dimension_penalty = xp.sqrt(float(dimension) / 8.0)
        bandwidth_value = max(float(bandwidth), 1e-8)
        local_weight = 0.75 * xp.exp(
            -distances * dimension_penalty / bandwidth_value
        )
        trust_radius = min(0.35, max(0.08, float(bandwidth)))
        local_weight = xp.where(
            distances <= trust_radius,
            local_weight,
            0.05 * local_weight,
        )
        statistics = xp.stack(
            (distances, correction, local_spread, local_weight),
            axis=0,
        )
        statistics_host = np.asarray(self.to_numpy(statistics), dtype=float)
        return tuple(statistics_host[index] for index in range(4))

    def _dictionary_plan(
        self,
        specs: Sequence[Sequence[Any]],
    ) -> dict[str, Any]:
        """Compile stable dictionary column metadata once per specification set."""
        key = tuple(tuple(spec) for spec in specs)
        cached = self._dictionary_plan_cache.get(key)
        if cached is not None:
            self._dictionary_plan_cache_hits += 1
            return cached

        positions: dict[str, list[tuple[int, tuple[Any, ...]]]] = {}
        for index, raw_spec in enumerate(specs):
            spec = tuple(raw_spec)
            positions.setdefault(str(spec[0]), []).append((index, spec))

        supported = {
            "const", "linear", "square", "interaction", "sine", "cosine",
            "rms_x", "radial_exp", "mean_cosine", "chain_residual",
        }
        unsupported = [kind for kind in positions if kind not in supported]
        if unsupported:
            raise ValueError(f"unsupported dictionary term kind: {unsupported[0]}")

        def one_index(kind: str) -> tuple[int, ...]:
            return tuple(index for index, _ in positions.get(kind, ()))

        def one_feature(kind: str) -> tuple[int, ...]:
            return tuple(int(spec[1]) for _, spec in positions.get(kind, ()))

        plan: dict[str, Any] = {
            "const": one_index("const"),
            "linear": (one_index("linear"), one_feature("linear")),
            "square": (one_index("square"), one_feature("square")),
            "interaction": (
                one_index("interaction"),
                tuple(int(spec[1]) for _, spec in positions.get("interaction", ())),
                tuple(int(spec[2]) for _, spec in positions.get("interaction", ())),
            ),
            "sine": (
                one_index("sine"),
                tuple(int(spec[1]) for _, spec in positions.get("sine", ())),
                tuple(int(spec[2]) for _, spec in positions.get("sine", ())),
            ),
            "cosine": (
                one_index("cosine"),
                tuple(int(spec[1]) for _, spec in positions.get("cosine", ())),
                tuple(int(spec[2]) for _, spec in positions.get("cosine", ())),
            ),
            "rms_x": one_index("rms_x"),
            "radial_exp": one_index("radial_exp"),
            "mean_cosine": (
                one_index("mean_cosine"),
                tuple(int(spec[1]) for _, spec in positions.get("mean_cosine", ())),
            ),
            "chain_residual": one_index("chain_residual"),
        }
        self._dictionary_plan_cache[key] = plan
        self._dictionary_plan_builds += 1
        return plan

    def _dictionary_normalization(
        self,
        bounds: np.ndarray,
        xp: Any,
    ) -> tuple[Any, Any]:
        """Return cached bound-normalization parameters on the requested device."""
        bounds_host = np.asarray(bounds, dtype=float)
        # A small host-array transform may populate this cache before the same
        # bounds are used by a device-resident CuPy batch. Keep one normalized
        # entry per array module so NumPy vectors are never reused in CuPy
        # arithmetic (or vice versa).
        cache_key = (id(bounds), "cpu" if xp is np else "cuda")
        cached = self._dictionary_bounds_cache.get(cache_key)
        if (
            cached is None
            or cached[0].shape != bounds_host.shape
            or not np.array_equal(cached[0], bounds_host)
        ):
            center = 0.5 * (bounds_host[:, 0] + bounds_host[:, 1])
            half_range = 0.5 * (bounds_host[:, 1] - bounds_host[:, 0])
            if xp is np:
                center_xp, half_range_xp = center, half_range
            else:
                center_xp = self._to_device(center, xp)
                half_range_xp = self._to_device(half_range, xp)
            self._dictionary_bounds_cache[cache_key] = (
                bounds_host.copy(),
                center_xp,
                half_range_xp,
            )
            return center_xp, half_range_xp
        self._dictionary_bounds_cache_hits += 1
        return cached[1], cached[2]

    def dictionary_transform(
        self,
        values: np.ndarray,
        specs: Sequence[Sequence[Any]],
        bounds: np.ndarray | None,
        *,
        return_device: bool = False,
    ) -> Any:
        device_input = self._is_device_array(values)
        if device_input:
            value_shape = tuple(values.shape)
            if len(value_shape) != 2:
                raise ValueError("values must be a 2D array")
            xp = self.array_module
            self._kernel_calls["symbolic_dictionary"] += 1
            self._gpu_kernel_calls["symbolic_dictionary"] += 1
            if not self.scalar(xp.all(xp.isfinite(values))):
                raise ValueError("values must contain only finite values")
        else:
            values = np.asarray(values, dtype=float)
            if values.ndim != 2:
                raise ValueError("values must be a 2D array")
            if not np.all(np.isfinite(values)):
                raise ValueError("values must contain only finite values")
            value_shape = values.shape
            xp = self.xp_for(
                values.size * max(len(specs), 1),
                "symbolic_dictionary",
            )
        x = self._to_device(values, xp)
        if bounds is None:
            normalized = x
        else:
            center, half_range = self._dictionary_normalization(bounds, xp)
            normalized = (x - center) / half_range
        plan = self._dictionary_plan(specs)
        if not specs:
            return xp.empty((value_shape[0], 0), dtype=xp.float64) if return_device else np.empty(
                (value_shape[0], 0), dtype=float
            )

        # Build complete columns first and assemble them once. Repeated
        # indexed writes launch many small CuPy elementwise kernels.
        columns: list[Any] = [None] * len(specs)

        const_indices = plan["const"]
        if const_indices:
            constant = xp.ones(value_shape[0], dtype=xp.float64)
            for index in const_indices:
                columns[index] = constant
        linear_indices, linear_features = plan["linear"]
        if linear_indices:
            for index, feature in zip(linear_indices, linear_features):
                columns[index] = normalized[:, feature]
        square_indices, square_features = plan["square"]
        if square_indices:
            square_values = normalized[:, square_features]
            for offset, index in enumerate(square_indices):
                columns[index] = square_values[:, offset] * square_values[:, offset]
        interaction_indices, left_features, right_features = plan["interaction"]
        if interaction_indices:
            for index, left, right in zip(
                interaction_indices, left_features, right_features
            ):
                columns[index] = normalized[:, left] * normalized[:, right]
        for kind in ("sine", "cosine"):
            indices, harmonics_host, features = plan[kind]
            if not indices:
                continue
            harmonics = xp.asarray(harmonics_host, dtype=xp.float64)
            feature_values = normalized[:, features]
            angles = xp.pi * feature_values * harmonics[None, :]
            values_for_kind = xp.sin(angles) if kind == "sine" else xp.cos(angles)
            for offset, index in enumerate(indices):
                columns[index] = values_for_kind[:, offset]

        rms = None
        for index in plan["rms_x"]:
            if rms is None:
                rms = xp.sqrt(xp.mean(x * x, axis=1))
            columns[index] = rms
        for index in plan["radial_exp"]:
            if rms is None:
                rms = xp.sqrt(xp.mean(x * x, axis=1))
            columns[index] = xp.exp(-0.2 * rms)
        mean_cosine_indices, mean_cosine_harmonics_host = plan["mean_cosine"]
        if mean_cosine_indices:
            harmonics = xp.asarray(
                mean_cosine_harmonics_host, dtype=xp.float64
            )
            angles = 2.0 * xp.pi * x[:, :, None] * harmonics[None, None, :]
            mean_cosines = xp.mean(xp.cos(angles), axis=1)
            for offset, index in enumerate(mean_cosine_indices):
                columns[index] = mean_cosines[:, offset]
        for index in plan["chain_residual"]:
            if value_shape[1] < 2:
                columns[index] = xp.zeros(value_shape[0], dtype=xp.float64)
            else:
                chain = x[:, :-1] * x[:, :-1] - x[:, 1:]
                columns[index] = xp.mean(chain * chain, axis=1)
        if any(column is None for column in columns):
            raise RuntimeError("dictionary specification did not produce all columns")
        result = xp.stack(columns, axis=1)
        return result if return_device else self.to_numpy(result)

    def ridge_fit(
        self,
        x: Any,
        y: Any,
        alpha: float,
        *,
        return_device: bool = False,
    ) -> Any:
        x_shape = tuple(getattr(x, "shape", np.shape(x)))
        if len(x_shape) != 2 or x_shape[1] == 0:
            return np.empty(0, dtype=float)
        xp, (design, target) = self.arrays_for(
            "sparse_ridge",
            x,
            y,
        )
        design = design.astype(xp.float64, copy=False)
        target = target.astype(xp.float64, copy=False)
        try:
            if design.shape[1] > design.shape[0]:
                kernel = design @ design.T
                kernel.flat[:: kernel.shape[0] + 1] += float(alpha)
                beta = design.T @ xp.linalg.solve(kernel, target)
            else:
                gram = design.T @ design
                gram.flat[:: gram.shape[0] + 1] += float(alpha)
                beta = xp.linalg.solve(gram, design.T @ target)
        except xp.linalg.LinAlgError:
            if design.shape[1] > design.shape[0]:
                beta = design.T @ (xp.linalg.pinv(kernel) @ target)
            else:
                beta = xp.linalg.pinv(gram) @ (design.T @ target)
        if return_device:
            return beta
        return np.asarray(self.to_numpy(beta), dtype=float)

    def ridge_fit_from_normal_equations(
        self,
        gram: Any,
        rhs: Any,
        alpha: float,
        *,
        return_device: bool = False,
    ) -> Any:
        """Solve a ridge system when ``X.T @ X`` and ``X.T @ y`` are cached."""
        xp, (gram_xp, rhs_xp) = self.arrays_for(
            "sparse_ridge",
            gram,
            rhs,
        )
        gram_xp = gram_xp.astype(xp.float64, copy=True)
        rhs_xp = rhs_xp.astype(xp.float64, copy=False)
        if gram_xp.ndim != 2 or gram_xp.shape[0] != gram_xp.shape[1]:
            raise ValueError("cached ridge Gram matrix must be square")
        gram_xp.flat[:: gram_xp.shape[0] + 1] += float(alpha)
        try:
            beta = xp.linalg.solve(gram_xp, rhs_xp)
        except xp.linalg.LinAlgError:
            beta = xp.linalg.pinv(gram_xp) @ rhs_xp
        if return_device:
            return beta
        return np.asarray(self.to_numpy(beta), dtype=float)

    def candidate_features(
        self,
        population: np.ndarray,
        trials: np.ndarray,
        parent_indices: np.ndarray,
        f_values: np.ndarray,
        cr_values: np.ndarray,
        strategy_values: np.ndarray,
        state: np.ndarray,
        population_values: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        objective_scale: float,
        strategy_count: int,
    ) -> np.ndarray:
        element_count = int(population.size + trials.size)
        xp = self.xp_for(element_count, "candidate_feature_construction")
        pop = self._to_device(population, xp)
        trial = self._to_device(trials, xp)
        parents = self._to_device(parent_indices, xp).astype(xp.int64)
        f_xp = self._to_device(f_values, xp)
        cr_xp = self._to_device(cr_values, xp)
        strategy = self._to_device(strategy_values, xp).astype(xp.int64)
        values = self._to_device(population_values, xp)
        span = self._to_device(upper - lower, xp)
        parent_x = pop[parents]
        step = xp.linalg.norm((trial - parent_x) / span, axis=1) / xp.sqrt(
            float(population.shape[1])
        )
        best = xp.min(values)
        parent_gap = xp.maximum(values[parents] - best, 0.0) / float(objective_scale)
        order = xp.argsort(values)
        ranks_all = xp.empty_like(order, dtype=xp.float64)
        ranks_all[order] = xp.arange(len(order), dtype=xp.float64)
        ranks = ranks_all[parents] / max(len(order) - 1, 1)
        tiled_state = xp.tile(self._to_device(state, xp), (len(trials), 1))
        one_hot = xp.column_stack(
            [(strategy == index).astype(xp.float64) for index in range(1, strategy_count + 1)]
        )
        strategy_code = (strategy.astype(xp.float64) - 1.0) / max(
            strategy_count - 1, 1
        )
        result = xp.column_stack(
            [
                xp.clip(step, 0.0, 2.0),
                xp.clip(parent_gap, 0.0, 4.0),
                ranks,
                f_xp,
                cr_xp,
                tiled_state,
                strategy_code,
                one_hot,
                one_hot * f_xp[:, None],
                one_hot * cr_xp[:, None],
            ]
        )
        return np.asarray(self.to_numpy(result), dtype=float)

    def scale_unit_points(
        self,
        unit_points: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> np.ndarray:
        xp = self.xp_for(np.size(unit_points), "initial_design_scaling")
        points = self._to_device(unit_points, xp)
        lower_xp = self._to_device(lower, xp)
        upper_xp = self._to_device(upper, xp)
        return np.asarray(
            self.to_numpy(lower_xp + points * (upper_xp - lower_xp)),
            dtype=float,
        )

    def de_trials(
        self,
        parents: np.ndarray,
        donor_r1: np.ndarray,
        donor_r2: np.ndarray,
        donor_r3: np.ndarray,
        pbest: np.ndarray,
        best: np.ndarray,
        local_donors: np.ndarray,
        f_values: np.ndarray,
        strategies: np.ndarray,
        crossover_mask: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> np.ndarray:
        """Generate all DE donor equations and crossover trials in one kernel graph."""
        xp = self.xp_for(np.size(parents) * 5, "de_candidate_generation")
        parent_xp = self._to_device(parents, xp)
        r1 = self._to_device(donor_r1, xp)
        r2 = self._to_device(donor_r2, xp)
        r3 = self._to_device(donor_r3, xp)
        pbest_xp = self._to_device(pbest, xp)
        best_xp = self._to_device(best, xp)
        local_xp = self._to_device(local_donors, xp)
        f_xp = self._to_device(f_values, xp)[:, None]
        strategy_xp = self._to_device(strategies, xp)
        donors = parent_xp.copy()
        donors = xp.where(
            (strategy_xp == 1)[:, None],
            parent_xp + f_xp * (pbest_xp - parent_xp) + f_xp * (r1 - r2),
            donors,
        )
        donors = xp.where(
            (strategy_xp == 2)[:, None],
            r1 + f_xp * (r2 - r3),
            donors,
        )
        donors = xp.where(
            (strategy_xp == 3)[:, None],
            best_xp[None, :] + f_xp * (r1 - r2),
            donors,
        )
        donors = xp.where((strategy_xp == 4)[:, None], local_xp, donors)
        crossover = self._to_device(crossover_mask, xp)
        trials = xp.where(crossover, donors, parent_xp)
        trials = xp.clip(
            trials,
            self._to_device(lower, xp),
            self._to_device(upper, xp),
        )
        return np.asarray(self.to_numpy(trials), dtype=float)

    def argsort(self, values: Any, *, descending: bool = False) -> np.ndarray:
        xp = self.xp_for(np.size(values), "candidate_ranking")
        array = self._to_device(values, xp)
        order = xp.argsort(-array if descending else array)
        return np.asarray(self.to_numpy(order), dtype=int)

    def audit(self) -> dict[str, Any]:
        cupy_version = None
        cuda_runtime_version = None
        if self._cupy is not None:
            cupy_version = str(getattr(self._cupy, "__version__", "unknown"))
            try:
                cuda_runtime_version = int(self._cupy.cuda.runtime.runtimeGetVersion())
            except Exception:
                cuda_runtime_version = None
        return {
            "requested": self.requested,
            "resolved": self.resolved,
            "device_index": self.device_index,
            "device_name": self.device_name,
            "gpu_min_elements": self.min_elements,
            "cupy_version": cupy_version,
            "cuda_runtime_version": cuda_runtime_version,
            "fallback_reason": self.fallback_reason,
            "kernel_calls": dict(sorted(self._kernel_calls.items())),
            "gpu_kernel_calls": dict(sorted(self._gpu_kernel_calls.items())),
            "cpu_fallback_calls": dict(sorted(self._cpu_fallback_calls.items())),
            "host_to_device_bytes": int(self._host_to_device_bytes),
            "device_to_host_bytes": int(self._device_to_host_bytes),
            "host_to_device_transfers": int(self._host_to_device_transfers),
            "device_to_host_transfers": int(self._device_to_host_transfers),
            "dictionary_plan_builds": int(self._dictionary_plan_builds),
            "dictionary_plan_cache_hits": int(self._dictionary_plan_cache_hits),
            "dictionary_bounds_cache_hits": int(self._dictionary_bounds_cache_hits),
            "accelerated_components": list(self.ACCELERATED_COMPONENTS),
            "semantics": (
                "The numerical backend changes execution hardware only; symbolic "
                "dictionary terms, sparse coefficients, model-selection rules, and "
                "true-evaluation decisions remain auditable and hardware invariant."
            ),
        }


__all__ = ["ComputeBackend"]
