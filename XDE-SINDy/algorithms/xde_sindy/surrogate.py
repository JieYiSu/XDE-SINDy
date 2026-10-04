"""Ensemble sparse surrogate with auditable uncertainty and trust-region tools."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .backend import ComputeBackend
from .common import _rms_distance_matrix
from .dictionary import Array, PolynomialLibrary
from .sparse import _fit_hierarchical_additive_coefficients, _fit_sparse_coefficients

class EnsembleSparseRegressor:
    """Ensemble-SINDy regressor with target transform, sparse selection, and distance-aware uncertainty."""

    def __init__(
        self,
        library: Optional[PolynomialLibrary] = None,
        n_models: int = 8,
        sample_ratio: float = 0.8,
        ridge_alpha: float = 1e-3,
        threshold: float = 0.03,
        random_state: Optional[int] = None,
        max_terms: int = 32,
        normalize: bool = True,
        target_transform: str = "robust_affine",
        target_clip: float = 8.0,
        local_correction: bool = True,
        local_k: int = 5,
        max_distance_reference: int = 256,
        hierarchical_additive_backbone: bool = False,
        adaptive_structure_selection: bool = False,
        compute_backend: Optional[ComputeBackend] = None,
    ) -> None:
        if n_models < 1:
            raise ValueError("n_models must be positive")
        if not 0.0 < sample_ratio <= 1.0:
            raise ValueError("sample_ratio must be in (0, 1]")
        self.compute_backend = compute_backend or ComputeBackend("cpu")
        self.library = library or PolynomialLibrary(
            compute_backend=self.compute_backend
        )
        self.library.set_compute_backend(self.compute_backend)
        self.n_models = int(n_models)
        self.sample_ratio = float(sample_ratio)
        self.ridge_alpha = float(ridge_alpha)
        self.threshold = float(threshold)
        self.max_terms = int(max_terms)
        self.normalize = bool(normalize)
        self.target_transform_name = str(target_transform).lower()
        if self.target_transform_name not in {"robust_affine", "asinh"}:
            raise ValueError("target_transform must be robust_affine or asinh")
        if target_clip <= 0.0:
            raise ValueError("target_clip must be positive")
        self.target_clip = float(target_clip)
        if local_k < 1:
            raise ValueError("local_k must be positive")
        if max_distance_reference < 3:
            raise ValueError("max_distance_reference must be at least 3")
        self.local_correction_enabled = bool(local_correction)
        self.local_k = int(local_k)
        self.max_distance_reference = int(max_distance_reference)
        self.hierarchical_additive_backbone = bool(hierarchical_additive_backbone)
        self.adaptive_structure_selection = bool(adaptive_structure_selection)
        self.adaptive_structure_selection_active = False
        self.structure_selection_minimum_samples = 0
        self.rng = np.random.default_rng(random_state)
        self.coefficients: Optional[Array] = None
        self.fitted = False
        self.normalized = False
        self.training_x: Optional[Array] = None
        self.training_y: Optional[Array] = None
        self.training_u: Optional[Array] = None
        self.training_formula_prediction: Optional[Array] = None
        self.training_residuals: Optional[Array] = None
        self.distance_reference_indices = np.empty(0, dtype=int)
        self.distance_reference_u: Optional[Array] = None
        self.distance_reference_residuals: Optional[Array] = None
        self.local_bandwidth = 0.25
        self.last_local_correction = np.empty(0, dtype=float)
        self.last_local_spread = np.empty(0, dtype=float)
        self.last_local_weight = np.empty(0, dtype=float)
        self.target_center = 0.0
        self.target_scale = 1.0
        self.cv_rmse = np.inf
        self.in_sample_error = np.inf
        self.quality_score = 0.0
        # Transparent online calibration learned from real prediction errors.
        self.uncertainty_calibration = 1.0
        self.uncertainty_calibration_updates = 0
        self.calibration_history: List[float] = []
        self.last_distance = np.empty(0, dtype=float)
        self.fit_count = 0
        self.selection_audit_records: List[Dict[str, Any]] = []
        self.dictionary_audit: Dict[str, Any] = {}
        self._quadratic_cache: Optional[Dict[str, Any]] = None

    def _target_forward(self, y: Array) -> Array:
        scaled = (np.asarray(y, dtype=float) - self.target_center) / self.target_scale
        if self.target_transform_name == "asinh":
            return np.arcsinh(scaled)
        return np.clip(scaled, -self.target_clip, self.target_clip)

    def _target_inverse(self, y: Array) -> Array:
        values = np.asarray(y, dtype=float)
        if self.target_transform_name == "asinh":
            return self.target_center + self.target_scale * np.sinh(np.clip(values, -20.0, 20.0))
        return self.target_center + self.target_scale * np.clip(
            values, -self.target_clip, self.target_clip
        )

    def _formula_predict_samples(self, x: Array) -> Array:
        coefficients = self._check_fitted()
        dictionary = self.library.transform_device(x)
        transformed = self.compute_backend.matmul(
            dictionary,
            coefficients.T,
            operation="ensemble_prediction",
        )
        return self._target_inverse(transformed)

    def predict_formula_samples(self, x: Array) -> Array:
        """Return ensemble predictions of the sparse formula, excluding local residual correction."""
        return self._formula_predict_samples(x)

    def _build_distance_reference(self) -> None:
        """Build a compact reference set for distance risk and local residual correction.

        All labelled samples still enter the SINDy fit. Distance calculations use a
        mixture of uniform, recent, and incumbent samples to avoid a full-archive
        distance matrix for every candidate.
        """
        if self.training_u is None or self.training_residuals is None:
            self.distance_reference_indices = np.empty(0, dtype=int)
            self.distance_reference_u = None
            self.distance_reference_residuals = None
            return
        sample_count = len(self.training_u)
        if sample_count <= self.max_distance_reference:
            indices = np.arange(sample_count, dtype=int)
        else:
            quota = max(1, self.max_distance_reference // 3)
            evenly = np.linspace(0, sample_count - 1, quota, dtype=int)
            recent = np.arange(max(0, sample_count - quota), sample_count, dtype=int)
            best = np.argsort(self.training_y)[:quota]
            indices = np.unique(np.concatenate([evenly, recent, best]))
            if len(indices) > self.max_distance_reference:
                indices = indices[: self.max_distance_reference]
        self.distance_reference_indices = np.asarray(indices, dtype=int)
        self.distance_reference_u = self.training_u[self.distance_reference_indices]
        self.distance_reference_residuals = self.training_residuals[self.distance_reference_indices]
        if len(self.distance_reference_u) < 2:
            self.local_bandwidth = 0.25
            return
        pairwise = _rms_distance_matrix(
            self.distance_reference_u,
            self.distance_reference_u,
            self.compute_backend,
        )
        np.fill_diagonal(pairwise, np.inf)
        nearest = np.min(pairwise, axis=1)
        finite_nearest = nearest[np.isfinite(nearest)]
        self.local_bandwidth = max(
            0.05,
            2.0 * float(np.median(finite_nearest)) if len(finite_nearest) else 0.25,
        )

    def _local_residual_prediction(self, query_u: Array) -> Tuple[Array, Array, Array, Array]:
        if (
            not self.local_correction_enabled
            or self.distance_reference_u is None
            or self.distance_reference_residuals is None
            or len(self.distance_reference_u) == 0
        ):
            zeros = np.zeros(len(query_u), dtype=float)
            return zeros, zeros, zeros, zeros
        return self.compute_backend.distance_uncertainty(
            query_u,
            self.distance_reference_u,
            self.distance_reference_residuals,
            local_k=self.local_k,
            bandwidth=self.local_bandwidth,
            target_scale=self.target_scale,
        )

    def fit(self, x: Array, y: Array) -> "EnsembleSparseRegressor":
        values = np.asarray(x, dtype=float)
        target = np.asarray(y, dtype=float).reshape(-1)
        if values.ndim != 2:
            raise ValueError("x must be a 2D array")
        if values.shape[0] != target.shape[0]:
            raise ValueError("x and y must contain the same number of rows")
        if values.shape[0] < 3:
            raise ValueError("at least three samples are required")
        finite = np.all(np.isfinite(values), axis=1) & np.isfinite(target)
        values = values[finite]
        target = target[finite]
        if len(target) < 3:
            raise ValueError("at least three finite samples are required")

        self._quadratic_cache = None

        self.training_x = values.copy()
        self.training_y = target.copy()
        self.training_u = np.asarray(self.library.normalize(values), dtype=float).copy()
        order = np.argsort(target)
        n_best = max(8, len(target) // 2)
        n_recent = max(8, min(len(target), max(32, len(target) // 3)))
        focus = np.concatenate([target[order[:n_best]], target[-n_recent:]])
        self.target_center = float(np.median(focus))
        mad = float(np.median(np.abs(focus - self.target_center)))
        robust_scale = 1.4826 * mad
        self.target_scale = max(robust_scale, 0.5 * float(np.std(focus)), 1e-8)
        transformed_target = self._target_forward(target)
        dictionary = self.library.transform_device(values)
        dictionary_on_device = self.compute_backend.is_device_array(dictionary)
        fit_target = (
            self.compute_backend.to_device(transformed_target)
            if dictionary_on_device
            else transformed_target
        )

        def backend_indices(indices: Array) -> Any:
            if not dictionary_on_device:
                return indices
            return self.compute_backend.array_module.asarray(
                indices,
                dtype=self.compute_backend.array_module.int64,
            )

        self.dictionary_audit = {
            "term_count": int(dictionary.shape[1]),
            "terms": list(self.library.names),
            "specifications": [
                {"index": int(index), "type": str(spec[0]), "arguments": [int(value) if isinstance(value, (int, np.integer)) else value for value in spec[1:]]}
                for index, spec in enumerate(self.library._specs)
            ],
            "coordinate_system": "z in [-1, 1] after bound normalization" if self.library.bounds is not None else "raw input coordinates",
        }
        sample_size = max(3, int(np.ceil(self.sample_ratio * len(values))))
        self.structure_selection_minimum_samples = max(
            40,
            4 * int(values.shape[1]),
        )
        self.adaptive_structure_selection_active = bool(
            self.adaptive_structure_selection
            and len(values) >= self.structure_selection_minimum_samples
        )
        models: List[Array] = []
        self.selection_audit_records = []
        oob_sum = np.zeros(len(values), dtype=float)
        oob_count = np.zeros(len(values), dtype=float)

        for _ in range(self.n_models):
            indices = self.rng.integers(0, len(values), size=sample_size)
            fit_indices = backend_indices(indices)
            bootstrap_dictionary = dictionary[fit_indices]
            bootstrap_target = fit_target[fit_indices]
            in_bag = np.zeros(len(values), dtype=bool)
            in_bag[indices] = True
            oob = ~in_bag
            if self.hierarchical_additive_backbone:
                backbone_coefficients, _, backbone_audit = _fit_hierarchical_additive_coefficients(
                    bootstrap_dictionary,
                    bootstrap_target,
                    self.ridge_alpha,
                    self.threshold,
                    self.max_terms,
                    self.library.names,
                    retain_all_main_effect_groups=(
                        self.adaptive_structure_selection
                        and not self.adaptive_structure_selection_active
                    ),
                    compute_backend=self.compute_backend,
                )
                coefficients = backbone_coefficients
                selection_audit = backbone_audit
                structure_candidates: Dict[str, Dict[str, Any]] = {}
                selected_structure = "hierarchical_additive_backbone"
                if self.adaptive_structure_selection_active:
                    sparse_coefficients, _, sparse_audit = _fit_sparse_coefficients(
                        bootstrap_dictionary,
                        bootstrap_target,
                        self.ridge_alpha,
                        self.threshold,
                        self.max_terms,
                        return_audit=True,
                        term_names=self.library.names,
                        polynomial_first=True,
                        compute_backend=self.compute_backend,
                    )
                    candidate_models = {
                        "hierarchical_additive_backbone": (
                            backbone_coefficients,
                            backbone_audit,
                        ),
                        "standard_sparse_dictionary": (
                            sparse_coefficients,
                            sparse_audit,
                        ),
                    }
                    for structure_name, (candidate_coefficients, _) in candidate_models.items():
                        nonzero = int(np.sum(np.abs(candidate_coefficients[1:]) > 1e-10))
                        if np.any(oob):
                            validation_indices = np.flatnonzero(oob)
                            candidate_prediction = self._target_inverse(
                                self.compute_backend.matmul(
                                    dictionary[backend_indices(validation_indices)],
                                    candidate_coefficients,
                                    operation="ensemble_prediction",
                                )
                            )
                            candidate_rmse = float(np.sqrt(np.mean(
                                (candidate_prediction - target[oob]) ** 2
                            )))
                        else:
                            candidate_prediction = self._target_inverse(
                                self.compute_backend.matmul(
                                    bootstrap_dictionary,
                                    candidate_coefficients,
                                    operation="ensemble_prediction",
                                )
                            )
                            candidate_rmse = float(np.sqrt(np.mean(
                                (candidate_prediction - target[indices]) ** 2
                            )))
                        complexity_multiplier = 1.0 + 0.002 * nonzero
                        structure_candidates[structure_name] = {
                            "oob_rmse": candidate_rmse,
                            "nonzero_term_count": nonzero,
                            "complexity_multiplier": complexity_multiplier,
                            "selection_score": candidate_rmse * complexity_multiplier,
                            "validation_source": "out_of_bag" if np.any(oob) else "bootstrap_in_sample",
                        }
                    selected_structure = min(
                        structure_candidates,
                        key=lambda name: structure_candidates[name]["selection_score"],
                    )
                    coefficients, selection_audit = candidate_models[selected_structure]
                selection_audit = {
                    **selection_audit,
                    "selected_structure": selected_structure,
                    "structure_candidates": structure_candidates,
                    "structure_selection_rule": (
                        "minimize validation_RMSE*(1+0.002*nonzero_terms) on the "
                        "same bootstrap out-of-bag samples"
                        if self.adaptive_structure_selection_active
                        else (
                            "retain the hierarchical additive backbone until at least "
                            "four true-labelled samples per input variable are available"
                            if self.adaptive_structure_selection
                            else "hierarchical additive backbone fixed by configuration"
                        )
                    ),
                }
            else:
                coefficients, _, selection_audit = _fit_sparse_coefficients(
                    bootstrap_dictionary, bootstrap_target,
                    self.ridge_alpha, self.threshold, self.max_terms,
                    return_audit=True,
                    term_names=self.library.names,
                    polynomial_first=True,
                    compute_backend=self.compute_backend,
                )
            models.append(coefficients)
            self.selection_audit_records.append({
                "model_index": len(models),
                "sample_size": int(sample_size),
                "sample_ratio": float(self.sample_ratio),
                "bootstrap_indices": [int(index) for index in indices],
                **selection_audit,
            })
            if np.any(oob):
                oob_indices = np.flatnonzero(oob)
                oob_sum[oob] += self.compute_backend.matmul(
                    dictionary[backend_indices(oob_indices)],
                    coefficients,
                    operation="ensemble_prediction",
                )
                oob_count[oob] += 1.0

        self.coefficients = np.vstack(models)
        # Mark the model as fitted before calling public prediction methods.
        # Otherwise in-sample diagnostics would recurse into an unfitted-state error.
        self.fitted = True
        self.normalized = bool(self.library.bounds is not None)
        formula_predictions = self._target_inverse(
            self.compute_backend.matmul(
                dictionary,
                self.coefficients.T,
                operation="ensemble_prediction",
            )
        )
        self.training_formula_prediction = np.mean(formula_predictions, axis=1)
        self.training_residuals = target - self.training_formula_prediction
        self._build_distance_reference()
        oob_mask = oob_count > 0
        if np.any(oob_mask):
            oob_prediction = self._target_inverse(oob_sum[oob_mask] / oob_count[oob_mask])
            self.cv_rmse = float(np.sqrt(np.mean((oob_prediction - target[oob_mask]) ** 2)))
        else:
            self.cv_rmse = float("inf")
        in_prediction, _ = self.predict_with_uncertainty(values, include_distance=False)
        self.in_sample_error = float(np.sqrt(np.mean((in_prediction - target) ** 2)))
        denominator = max(float(np.std(target)), self.target_scale, 1e-8)
        normalized_error = self.cv_rmse / denominator if np.isfinite(self.cv_rmse) else 10.0
        self.quality_score = float(1.0 / (1.0 + normalized_error))
        self.fit_count += 1
        return self

    def update_uncertainty_calibration(
        self,
        predicted: float,
        observed: float,
        predicted_uncertainty: float,
    ) -> float:
        '''Update uncertainty scale from one newly observed true value.

        The update is auditable: abs(predicted-observed) divided by the
        predicted uncertainty is clipped and applied as an EMA.
        '''
        error = abs(float(predicted) - float(observed))
        uncertainty = max(abs(float(predicted_uncertainty)), 1e-8)
        ratio = float(np.clip(error / uncertainty, 0.25, 4.0))
        self.uncertainty_calibration = float(np.clip(
            0.90 * self.uncertainty_calibration + 0.10 * ratio,
            0.50, 5.0,
        ))
        self.uncertainty_calibration_updates += 1
        self.calibration_history.append(float(self.uncertainty_calibration))
        return self.uncertainty_calibration

    def _check_fitted(self) -> Array:
        if not self.fitted or self.coefficients is None:
            raise RuntimeError("fit must be called before prediction")
        return self.coefficients

    def predict_samples(self, x: Array) -> Array:
        return self._formula_predict_samples(x)

    def _predict_components(
        self,
        x: Array,
        include_distance: bool = True,
    ) -> Tuple[Array, Array, Array]:
        predictions = self.predict_formula_samples(x)
        formula_mean = np.mean(predictions, axis=1)
        mean = formula_mean.copy()
        spread = np.std(predictions, axis=1)
        if not include_distance or self.training_u is None:
            self.last_distance = np.zeros(len(mean), dtype=float)
            self.last_local_correction = np.zeros(len(mean), dtype=float)
            self.last_local_spread = np.zeros(len(mean), dtype=float)
            self.last_local_weight = np.zeros(len(mean), dtype=float)
            return formula_mean, mean, spread
        query_u = np.asarray(self.library.normalize(np.asarray(x, dtype=float)), dtype=float)
        if query_u.ndim == 1:
            query_u = query_u.reshape(1, -1)
        distances, correction, local_spread, local_weight = self._local_residual_prediction(query_u)
        self.last_distance = distances
        self.last_local_correction = correction
        self.last_local_spread = local_spread
        self.last_local_weight = local_weight
        mean = mean + local_weight * correction
        spread = np.sqrt(spread ** 2 + (local_weight * local_spread) ** 2)
        distance_risk = np.clip(distances / 0.25, 0.0, 4.0)
        residual_scale = max(
            self.target_scale * 0.05,
            self.cv_rmse if np.isfinite(self.cv_rmse) else self.target_scale,
        )
        calibrated_spread = spread * float(self.uncertainty_calibration)
        return formula_mean, mean, calibrated_spread + residual_scale * distance_risk

    def predict_with_uncertainty(
        self,
        x: Array,
        include_distance: bool = True,
    ) -> Tuple[Array, Array]:
        _, mean, uncertainty = self._predict_components(x, include_distance=include_distance)
        return mean, uncertainty

    def predict_with_details(self, x: Array) -> Tuple[Array, Array, Array]:
        mean, uncertainty = self.predict_with_uncertainty(x, include_distance=True)
        return mean, uncertainty, self.last_distance.copy()

    def predict_with_formula_details(
        self,
        x: Array,
    ) -> Tuple[Array, Array, Array, Array]:
        """Return formula mean, corrected mean, uncertainty, and distance in one pass."""
        formula, mean, uncertainty = self._predict_components(x, include_distance=True)
        return formula, mean, uncertainty, self.last_distance.copy()

    def selection_frequency(self) -> Array:
        coefficients = self._check_fitted()
        return np.mean(np.abs(coefficients) > 1e-10, axis=0)

    def selected_terms(self, min_frequency: float = 0.5) -> List[str]:
        frequencies = self.selection_frequency()
        return [
            name for name, frequency in zip(self.library.names, frequencies)
            if frequency >= min_frequency
        ]

    def coefficient_intervals(self) -> Tuple[Array, Array]:
        return np.percentile(self._check_fitted(), 2.5, axis=0), np.percentile(
            self._check_fitted(), 97.5, axis=0
        )

    def mean_coefficients(self) -> Array:
        return np.mean(self._check_fitted(), axis=0)

    def _sparse_linear_hessian(self) -> Tuple[Array, Array, Dict[str, int]]:
        """Extract the linear vector and Hessian of the mean sparse quadratic."""
        if self._quadratic_cache is not None:
            return (
                self._quadratic_cache["linear"],
                self._quadratic_cache["hessian"],
                self._quadratic_cache["counts"],
            )
        dimension = int(self.library.n_features)
        coefficients = self.mean_coefficients()
        linear = np.zeros(dimension, dtype=float)
        hessian = np.zeros((dimension, dimension), dtype=float)
        counts = {"linear": 0, "square": 0, "interaction": 0}
        for coefficient, spec in zip(coefficients, self.library._specs):
            value = float(coefficient)
            if abs(value) <= 1e-12:
                continue
            kind = spec[0]
            if kind == "linear":
                linear[int(spec[1])] += value
                counts["linear"] += 1
            elif kind == "square":
                index = int(spec[1])
                hessian[index, index] += 2.0 * value
                counts["square"] += 1
            elif kind == "interaction":
                left = int(spec[1])
                right = int(spec[2])
                hessian[left, right] += value
                hessian[right, left] += value
                counts["interaction"] += 1
        eigvals = np.linalg.eigvalsh(hessian) if hessian.size else np.zeros(1, dtype=float)
        min_eigenvalue = float(np.min(eigvals))
        unconstrained_z = None
        unconstrained_reason = "hessian_not_positive_definite"
        if min_eigenvalue > 1e-8:
            try:
                unconstrained_z = np.linalg.solve(hessian, -linear)
            except np.linalg.LinAlgError:
                unconstrained_reason = "singular_hessian"
            else:
                if not np.all(np.isfinite(unconstrained_z)):
                    unconstrained_z = None
                    unconstrained_reason = "non_finite_unconstrained_vertex"
                else:
                    unconstrained_reason = "available"
        self._quadratic_cache = {
            "linear": linear,
            "hessian": hessian,
            "counts": counts,
            "min_eigenvalue": min_eigenvalue,
            "unconstrained_z": unconstrained_z,
            "unconstrained_reason": unconstrained_reason,
        }
        return linear, hessian, counts

    def _solve_ucpe_step(
        self,
        gradient: Array,
        hessian: Array,
        radii: Array,
        *,
        min_eigenvalue: Optional[float] = None,
        unconstrained_step: Optional[Array] = None,
    ) -> Tuple[Array, str, float, bool]:
        """Minimize a local quadratic inside the unit ball y=(z-c)/r."""
        scale = np.asarray(radii, dtype=float)
        g_y = scale * np.asarray(gradient, dtype=float)
        min_eig = (
            float(min_eigenvalue)
            if min_eigenvalue is not None
            else float(np.min(np.linalg.eigvalsh(hessian)))
        )
        if min_eig >= 0.0:
            min_eig_y = min_eig * float(np.min(scale) ** 2)
        else:
            min_eig_y = min_eig * float(np.max(scale) ** 2)
        g_norm = float(np.linalg.norm(g_y))
        dimension = int(g_y.size)
        if g_norm < 1e-12:
            return np.zeros(dimension, dtype=float), "stationary_at_center", min_eig_y, False
        if min_eig > 1e-8:
            y_newton = None if unconstrained_step is None else np.asarray(unconstrained_step, dtype=float)
            if y_newton is not None and np.all(np.isfinite(y_newton)):
                newton_norm = float(np.linalg.norm(y_newton))
                if newton_norm <= 1.0 + 1e-10:
                    return y_newton, "interior_newton", min_eig_y, True
                return y_newton / newton_norm, "boundary_truncated_newton", min_eig_y, True
        direction = -g_y / g_norm
        linear_term = float(g_y @ direction)
        scaled_direction = scale * direction
        curvature = float(scaled_direction @ hessian @ scaled_direction)
        if curvature > 1e-12:
            alpha = min(1.0, -linear_term / curvature)
        else:
            alpha = 1.0
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return alpha * direction, "cauchy_point", min_eig_y, False

    def trust_region_acquisition(
        self,
        center_x: Array,
        population: Array,
        uncertainty: Optional[float] = None,
        quality_score: Optional[float] = None,
        radius_scale: float = 1.0,
    ) -> Dict[str, Any]:
        """Minimize the sparse formula inside an uncertainty-calibrated ellipsoid.

        The trust region is the population (or elite-neighborhood) covariance
        ellipsoid in bound-normalized coordinates, contracted by surrogate
        quality and predictive uncertainty. If the Hessian is not positive
        definite, only a Cauchy point is taken; an unconstrained stationary
        point is never injected.
        """
        if not self.fitted or self.coefficients is None:
            return {
                "usable": False,
                "reason": "model_not_fitted",
                "acquisition_name": "uncertainty-calibrated population ellipsoid",
            }
        if not self.library._specs or self.library.n_features <= 0:
            return {
                "usable": False,
                "reason": "empty_dictionary",
                "acquisition_name": "uncertainty-calibrated population ellipsoid",
            }

        center = np.asarray(center_x, dtype=float).reshape(-1)
        pop = np.asarray(population, dtype=float)
        if pop.ndim == 1:
            pop = pop.reshape(1, -1)
        if center.size != int(self.library.n_features):
            return {
                "usable": False,
                "reason": "center_dimension_mismatch",
                "acquisition_name": "uncertainty-calibrated population ellipsoid",
            }
        if pop.size == 0:
            pop = center.reshape(1, -1)

        z_center = np.asarray(self.library.normalize(center.reshape(1, -1)), dtype=float).reshape(-1)
        z_pop = np.asarray(self.library.normalize(pop), dtype=float)
        distances = np.linalg.norm(z_pop - z_center[None, :], axis=1)
        neighbor_count = max(4, min(len(z_pop), int(np.ceil(0.40 * len(z_pop)))))
        nearest = z_pop[np.argsort(distances)[:neighbor_count]]
        sigma = np.std(nearest, axis=0)
        sigma = np.maximum(sigma, 1e-3)
        u_norm = 0.0
        if uncertainty is not None and np.isfinite(float(uncertainty)):
            u_norm = float(uncertainty) / max(float(self.target_scale), 1e-8)
        quality = float(self.quality_score if quality_score is None else quality_score)
        gamma = float(np.clip(quality / (1.0 + u_norm), 0.35, 1.0))
        adaptive_scale = float(np.clip(radius_scale, 0.35, 1.80))
        radii = np.clip(0.75 * sigma * gamma * adaptive_scale, 0.02, 0.35)

        linear, hessian, counts = self._sparse_linear_hessian()
        gradient = linear + hessian @ z_center
        quadratic_cache = self._quadratic_cache or {}
        min_eig = float(quadratic_cache.get("min_eigenvalue", 0.0))
        unconstrained_z = quadratic_cache.get("unconstrained_z")
        unconstrained_inside = False
        unconstrained_mahalanobis = None
        unconstrained_reason = str(
            quadratic_cache.get("unconstrained_reason", "hessian_not_positive_definite")
        )
        if unconstrained_z is not None:
            y_uncon = (unconstrained_z - z_center) / radii
            unconstrained_mahalanobis = float(np.sqrt(np.sum(y_uncon * y_uncon)))
            unconstrained_inside = bool(unconstrained_mahalanobis <= 1.0 + 1e-8)
            unconstrained_reason = (
                "inside_ellipsoid" if unconstrained_inside else "outside_ellipsoid"
            )

        unconstrained_step = (
            None
            if unconstrained_z is None
            else (unconstrained_z - z_center) / radii
        )
        step, step_type, min_eig_y, newton_used = self._solve_ucpe_step(
            gradient,
            hessian,
            radii,
            min_eigenvalue=min_eig,
            unconstrained_step=unconstrained_step,
        )
        z_candidate = np.clip(z_center + radii * step, -1.0, 1.0)
        x_candidate = np.asarray(self.library.denormalize(z_candidate.reshape(1, -1)), dtype=float).reshape(-1)
        mahalanobis = float(np.sqrt(np.sum(((z_candidate - z_center) / radii) ** 2)))
        formula_samples = self.predict_formula_samples(
            np.vstack((x_candidate, center))
        )
        formula = float(np.mean(formula_samples[0]))
        center_formula = float(np.mean(formula_samples[1]))
        predicted_reduction = float(center_formula - formula)
        moved = bool(mahalanobis > 1e-8)
        usable = bool(moved and np.all(np.isfinite(x_candidate)) and np.isfinite(formula))
        reason = step_type if usable else ("no_local_descent" if step_type == "stationary_at_center" else "non_finite_or_stationary")
        return {
            "usable": usable,
            "reason": reason,
            "acquisition_name": "uncertainty-calibrated population ellipsoid",
            "x": x_candidate.tolist(),
            "z": z_candidate.tolist(),
            "center_x": center.tolist(),
            "center_z": z_center.tolist(),
            "radii": radii.tolist(),
            "mahalanobis": mahalanobis,
            "step_type": step_type,
            "newton_used": bool(newton_used),
            "min_eigenvalue": min_eig,
            "min_eigenvalue_y": float(min_eig_y),
            "uncertainty_scale": u_norm,
            "quality_score": quality,
            "radius_shrinkage": gamma,
            "adaptive_radius_scale": adaptive_scale,
            "neighbor_count": int(neighbor_count),
            "unconstrained_rejected": bool((unconstrained_z is None) or (not unconstrained_inside)),
            "unconstrained_inside_ellipsoid": bool(unconstrained_inside),
            "unconstrained_reason": unconstrained_reason,
            "unconstrained_mahalanobis": unconstrained_mahalanobis,
            "unconstrained_z": None if unconstrained_z is None else unconstrained_z.tolist(),
            "predicted_formula": formula,
            "center_predicted_formula": center_formula,
            "predicted_reduction": predicted_reduction,
            "square_term_count": int(counts["square"]),
            "linear_term_count": int(counts["linear"]),
            "interaction_term_count": int(counts["interaction"]),
            "solve_rule": (
                "Minimize the identified sparse quadratic inside the uncertainty-"
                "calibrated population ellipsoid {z: ||(z-z_c)/r||_2 <= 1}. "
                "r_i = clip(0.75 * neighborhood_std_i * quality/(1+uncertainty) "
                "* adaptive_radius_scale, 0.02, 0.35). The adaptive scale is updated "
                "from agreement between predicted and realized improvement. "
                "A truncated Newton step is used only when the Hessian is positive "
                "definite; otherwise the Cauchy point is taken. An unconstrained "
                "stationary point H z = -g is recorded for audit and is never injected "
                "when it lies outside the ellipsoid or the Hessian is not PD."
            ),
        }

    def in_sample_rmse(self) -> float:
        return float(self.in_sample_error)

    def diagnostics(self) -> Dict[str, float]:
        return {
            "cv_rmse": float(self.cv_rmse),
            "in_sample_rmse": float(self.in_sample_error),
            "quality_score": float(self.quality_score),
            "target_center": float(self.target_center),
            "target_scale": float(self.target_scale),
            "fit_count": float(self.fit_count),
            "local_correction": float(self.local_correction_enabled),
            "local_k": float(self.local_k),
            "local_bandwidth": float(self.local_bandwidth),
            "reference_sample_count": float(len(self.distance_reference_indices)),
            "uncertainty_calibration": float(self.uncertainty_calibration),
            "uncertainty_calibration_updates": float(self.uncertainty_calibration_updates),
            "hierarchical_additive_backbone": float(
                self.hierarchical_additive_backbone
            ),
            "adaptive_structure_selection": float(
                self.adaptive_structure_selection
            ),
            "adaptive_structure_selection_active": float(
                self.adaptive_structure_selection_active
            ),
            "structure_selection_minimum_samples": float(
                self.structure_selection_minimum_samples
            ),
        }

    def selection_audit(self) -> Dict[str, Any]:
        """Return the white-box audit of the surrogate.

        The record describes the candidate dictionary, bootstrap ensemble, stepwise
        sparse selection, ridge refit, stability rule, and the uncertainty formula.
        """
        coefficients = self.mean_coefficients() if self.fitted else np.empty(0, dtype=float)
        frequencies = self.selection_frequency() if self.fitted else np.empty(0, dtype=float)
        stable_terms = []
        if self.fitted:
            for index, (name, coefficient, frequency) in enumerate(
                zip(self.library.names, coefficients, frequencies)
            ):
                stable_terms.append({
                    "index": int(index),
                    "term": name,
                    "coefficient": float(coefficient),
                    "selection_frequency": float(frequency),
                    "selected": bool(abs(float(coefficient)) > 1e-10 and frequency >= 0.5),
                })
        ensemble_models = deepcopy(self.selection_audit_records)
        for record in ensemble_models:
            selected_indices = [int(index) for index in record.get("selected_indices", [])]
            record["selected_terms"] = [
                self.library.names[index]
                for index in selected_indices
                if 0 <= index < len(self.library.names)
            ]
            for path_item in record.get("selection_path", []):
                candidate_index = int(path_item.get("candidate_index", -1))
                path_item["candidate_term"] = (
                    self.library.names[candidate_index]
                    if 0 <= candidate_index < len(self.library.names)
                    else None
                )
                path_item["selected_terms"] = [
                    self.library.names[index]
                    for index in path_item.get("selected_indices", [])
                    if 0 <= int(index) < len(self.library.names)
                ]
        if (
            self.hierarchical_additive_backbone
            and self.adaptive_structure_selection
            and self.adaptive_structure_selection_active
        ):
            selection_algorithm = (
                "out-of-bag selection between a joint linear/quadratic main-effect "
                "backbone and a standard sparse dictionary; the selected formula is "
                "refit within each bootstrap ensemble member"
            )
            hierarchy_rule = (
                "the additive candidate uses variable-group hierarchy; the standard "
                "candidate uses weak polynomial hierarchy"
            )
            complexity_label = (
                "validation RMSE multiplied by 1+0.002 times the nonzero-term count"
            )
        elif (
            self.hierarchical_additive_backbone
            and self.adaptive_structure_selection
        ):
            selection_algorithm = (
                "warm-up ridge fit of all linear/quadratic main-effect groups; "
                "out-of-bag structure selection is deferred until four true-labelled "
                "samples per input variable are available"
            )
            hierarchy_rule = (
                "all variable groups are retained during warm-up; ridge shrinkage "
                "controls coefficient magnitude before data-supported group deletion"
            )
            complexity_label = (
                "all main-effect groups plus the residual-term budget during warm-up"
            )
        elif self.hierarchical_additive_backbone:
            selection_algorithm = (
                "joint ridge screening of linear/quadratic main-effect groups, "
                "followed by residual-correlation sparse selection"
            )
            hierarchy_rule = (
                "linear and quadratic main effects share one variable-group budget; "
                "residual interactions include any missing linear parents"
            )
            complexity_label = "maximum number of main-effect groups plus residual terms"
        else:
            selection_algorithm = (
                "residual-correlation forward selection followed by ridge refit"
            )
            hierarchy_rule = (
                "linear and quadratic terms compete on the same standardized scale; "
                "selecting a square or interaction includes missing parent linear terms"
            )
            complexity_label = "maximum number of selected dictionary terms"
        return {
            "hierarchical_additive_backbone": bool(
                self.hierarchical_additive_backbone
            ),
            "adaptive_structure_selection": bool(
                self.adaptive_structure_selection
            ),
            "adaptive_structure_selection_active": bool(
                self.adaptive_structure_selection_active
            ),
            "structure_selection_minimum_samples": int(
                self.structure_selection_minimum_samples
            ),
            "dictionary_terms": list(self.library.names),
            "dictionary_definition": deepcopy(self.dictionary_audit),
            "selection_rule": {
                "algorithm": selection_algorithm,
                "threshold": float(self.threshold),
                "max_terms": int(self.max_terms),
                "complexity_budget": complexity_label,
                "ridge_alpha": float(self.ridge_alpha),
                "standardization": "non-constant columns are centered and scaled before correlation scoring",
                "score": "abs(column dot current residual) / n_samples",
                "hierarchy": hierarchy_rule,
                "stop_rules": [
                    "maximum number of terms reached",
                    "best residual correlation below threshold times target scale",
                    "relative residual reduction stagnated after five selections",
                ],
                "final_refit": "selected standardized columns are refit with ridge regression and transformed back to original column scale",
            },
            "ensemble_models": ensemble_models,
            "target_transform_rule": {
                "name": self.target_transform_name,
                "center": float(self.target_center),
                "scale": float(self.target_scale),
                "clip": float(self.target_clip),
                "forward": "u=clip((y-target_center)/target_scale, -target_clip, target_clip)" if self.target_transform_name == "robust_affine" else "u=asinh((y-target_center)/target_scale)",
                "inverse": "y=target_center+target_scale*clip(u, -target_clip, target_clip)" if self.target_transform_name == "robust_affine" else "y=target_center+target_scale*sinh(u)",
            },
            "final_stable_terms": stable_terms,
            "stability_rule": {
                "selected_when": "absolute mean coefficient > 1e-10 and selection frequency >= 0.5",
                "selection_frequency": "number of ensemble models selecting a term divided by n_models",
                "coefficient_interval": "2.5th and 97.5th percentiles across ensemble coefficients",
            },
            "uncertainty_rule": {
                "formula": "calibration * [sqrt(ensemble_prediction_variance + local_weight^2 * local_residual_spread^2) + residual_scale * distance_risk]",
                "components": [
                    "ensemble disagreement",
                    "distance from reference true samples",
                    "kernel-weighted local residual correction spread",
                    "out-of-bag validation residual scale",
                ],
                "local_correction_is_not_a_formula_term": True,
                "local_correction_rule": {
                    "reference_set": "uniformly spaced, recent, and best true samples, capped by max_distance_reference",
                    "neighbor_count": int(self.local_k),
                    "bandwidth": float(self.local_bandwidth),
                    "kernel": "w_i=exp(-distance_i/local_bandwidth), normalized over the k nearest reference samples",
                    "correction": "local_correction=sum_i(w_i*training_residual_i), clipped to +/-2*median(|reference residual|); weight is reduced in high dimension and outside the local trust radius",
                    "trust_weight": "local_weight=0.75*exp(-nearest_distance/local_bandwidth)",
                    "spread": "sqrt(sum_i(w_i*(residual_i-local_correction)^2))",
                    "interpretation": "transparent local residual interpolation; it corrects formula error near observed samples and is not treated as a discovered global mechanism",
                },
            },
        }


__all__ = ["EnsembleSparseRegressor"]

