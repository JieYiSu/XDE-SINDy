"""Interpretable Surrogate-Assisted Differential Evolution for Expensive Optimization

Design principles
-----------------
1. Every candidate offspring is scored by the surrogate and recorded in the
   candidate ledger. Only true-evaluated samples re-enter supervised fitting,
   so bootstrap pseudo-labels cannot contaminate the archive.
2. The objective surrogate uses bound-normalized coordinates, a hierarchical
   dictionary, and sparse regression. Linear and quadratic terms compete on a
   common standardized residual-correlation scale; weak hierarchy includes parent
   linear terms after a square or interaction is selected. Candidate generation
   minimizes the sparse formula inside an uncertainty-calibrated population
   ellipsoid rather than injecting an unconstrained stationary point.
3. Trust is defined jointly by ensemble disagreement, hold-out residual scale,
   and distance from true-labelled samples.
4. An offspring-improvement model predicts the next-generation gain of a
   (strategy, F, CR) combination; the selected F and CR values are injected
   into the candidate pool.
5. One true evaluation is reserved at termination to confirm the
   surrogate-selected best candidate.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .backend import ComputeBackend
from .common import _resolve_metadata, _rms_distance_matrix, latin_hypercube
from .dictionary import PolynomialLibrary
from .policy import SparsePolicyModel
from algorithms.common.result import OptimizationResult
from .sparse import _ridge_fit
from .surrogate import EnsembleSparseRegressor

Array = np.ndarray


class XDESindy:
    """Interpretable, uncertainty-aware surrogate-assisted DE for expensive single-objective problems."""

    def __init__(
        self,
        objective: Callable[[Array], float],
        bounds: Sequence[Tuple[float, float]],
        variable_labels: Optional[Sequence[str] | Mapping[int, str]] = None,
        variable_units: Optional[Sequence[str] | Mapping[int, str]] = None,
        variable_descriptions: Optional[Sequence[str] | Mapping[int, str]] = None,
        population_size: int = 20,
        max_evaluations: int = 500,
        surrogate_start: int = 40,
        real_batch_size: Optional[int] = None,
        surrogate_refit_interval: int = 5,
        max_surrogate_samples: Optional[int] = None,
        surrogate_samples_per_dimension: float = 2.0,
        improvement_degree: int = 1,
        max_active_variables: int = 32,
        block_mutation_size: Optional[int] = 32,
        uncertainty_weight: float = 0.2,
        trust_threshold: float = 0.15,
        library: Optional[PolynomialLibrary] = None,
        n_ensemble_models: int = 8,
        constraint: Optional[Callable[[Array], Array]] = None,
        random_state: Optional[int] = None,
        candidate_pool_multiplier: int = 4,
        max_surrogate_terms: int = 32,
        max_interaction_terms: int = 16,
        min_model_quality: float = 0.35,
        reserve_final_evaluation: bool = True,
        exploration_fraction: float = 0.30,
        target_transform: str = "robust_affine",
        target_clip: float = 8.0,
        local_correction: bool = True,
        hierarchical_additive_backbone: bool = True,
        adaptive_structure_selection: bool = True,
        local_k: int = 5,
        max_distance_reference: int = 256,
        de_p_best_rate: float = 0.20,
        de_archive_multiplier: int = 2,
        de_memory_size: int = 8,
        de_strategy_learning_rate: float = 0.15,
        de_high_dimension_threshold: int = 30,
        de_strategy_probability_floor: float = 0.10,
        de_local_strategy_max_probability: float = 0.55,
        relative_improvement_mode: str = "auto",
        relative_improvement_tolerance: float = 5.0e-4,
        de_policy_probe_count: int = 4,
        de_policy_update_interval: int = 5,
        de_global_candidate_fraction: float = 0.35,
        max_screening_audit_records: int = 8,
        max_candidate_audit_records: int = 8192,
        interaction_screening_interval: int = 5,
        uncertainty_candidate_fraction: float = 1.0,
        uncertainty_evaluation_interval: int = 1,
        min_improvement_validation_samples: int = 48,
        improvement_policy_weight: float = 0.25,
        compute_backend: str = "auto",
        gpu_device: int = 0,
        gpu_min_elements: int = 250000,
        gpu_objective: bool = False,
    ) -> None:
        self.objective = objective
        self.constraint = constraint
        self.gpu_objective = bool(gpu_objective)
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (n_features, 2)")
        if np.any(self.bounds[:, 1] <= self.bounds[:, 0]):
            raise ValueError("each upper bound must exceed its lower bound")
        if population_size < 4:
            raise ValueError("population_size must be at least 4")
        if max_evaluations < population_size:
            raise ValueError("max_evaluations must cover the initial population")
        if surrogate_refit_interval < 1:
            raise ValueError("surrogate_refit_interval must be positive")
        if candidate_pool_multiplier < 1:
            raise ValueError("candidate_pool_multiplier must be positive")
        if max_surrogate_terms < 1:
            raise ValueError("max_surrogate_terms must be positive")
        if not 0.0 < de_p_best_rate <= 1.0:
            raise ValueError("de_p_best_rate must be in (0, 1]")
        if de_archive_multiplier < 1:
            raise ValueError("de_archive_multiplier must be positive")
        if de_memory_size < 1:
            raise ValueError("de_memory_size must be positive")
        if not 0.0 <= de_strategy_learning_rate <= 1.0:
            raise ValueError("de_strategy_learning_rate must be in [0, 1]")
        if surrogate_samples_per_dimension < 0.0:
            raise ValueError("surrogate_samples_per_dimension must be non-negative")
        if de_high_dimension_threshold < 1:
            raise ValueError("de_high_dimension_threshold must be positive")
        if not 0.0 < de_strategy_probability_floor < 1.0 / 4.0:
            raise ValueError("de_strategy_probability_floor must be in (0, 0.25)")
        if not de_strategy_probability_floor <= de_local_strategy_max_probability < 1.0:
            raise ValueError(
                "de_local_strategy_max_probability must be in [floor, 1)"
            )
        if str(relative_improvement_mode).strip().lower() not in {"auto", "fixed"}:
            raise ValueError("relative_improvement_mode must be auto or fixed")
        if relative_improvement_tolerance <= 0.0:
            raise ValueError("relative_improvement_tolerance must be positive")
        if de_policy_probe_count < 1:
            raise ValueError("de_policy_probe_count must be positive")
        if de_policy_update_interval < 1:
            raise ValueError("de_policy_update_interval must be positive")
        if not 0.0 < de_global_candidate_fraction <= 1.0:
            raise ValueError("de_global_candidate_fraction must be in (0, 1]")
        if max_screening_audit_records < 1:
            raise ValueError("max_screening_audit_records must be positive")
        if max_candidate_audit_records < 1:
            raise ValueError("max_candidate_audit_records must be positive")
        if interaction_screening_interval < 1:
            raise ValueError("interaction_screening_interval must be positive")
        if not 0.0 < uncertainty_candidate_fraction <= 1.0:
            raise ValueError("uncertainty_candidate_fraction must be in (0, 1]")
        if uncertainty_evaluation_interval < 1:
            raise ValueError("uncertainty_evaluation_interval must be positive")
        if min_improvement_validation_samples < 4:
            raise ValueError("min_improvement_validation_samples must be at least 4")
        if not 0.0 <= improvement_policy_weight <= 1.0:
            raise ValueError("improvement_policy_weight must be in [0, 1]")

        self.dimension = int(self.bounds.shape[0])
        self.compute_backend = ComputeBackend(
            compute_backend,
            device=gpu_device,
            min_elements=gpu_min_elements,
        )
        self._variable_labels_provided = variable_labels is not None
        self.variable_labels = _resolve_metadata(
            variable_labels, self.dimension, "x", "variable_labels"
        )
        self.variable_units = _resolve_metadata(
            variable_units, self.dimension, "", "variable_units"
        )
        self.variable_descriptions = _resolve_metadata(
            variable_descriptions, self.dimension, "", "variable_descriptions"
        )
        self.population_size = int(population_size)
        self.max_evaluations = int(max_evaluations)
        self.surrogate_start = max(3, int(surrogate_start))
        self.real_batch_size = int(real_batch_size or max(1, population_size // 4))
        self.surrogate_refit_interval = int(surrogate_refit_interval)
        self.de_high_dimension_threshold = int(de_high_dimension_threshold)
        if self.dimension >= 100:
            self.effective_surrogate_refit_interval = max(
                self.surrogate_refit_interval, 20
            )
        elif self.dimension >= 30:
            self.effective_surrogate_refit_interval = max(
                self.surrogate_refit_interval, 10
            )
        else:
            self.effective_surrogate_refit_interval = self.surrogate_refit_interval
        # Keep the configured cap for the bounded surrogate training view, but
        # give high-dimensional models a small number of labelled samples per
        # variable. The complete true-evaluation archive remains available for
        # auditing and the FE budget is unchanged.
        self.surrogate_samples_per_dimension = float(surrogate_samples_per_dimension)
        if max_surrogate_samples is None:
            self.max_surrogate_samples = None
        elif self.dimension >= self.de_high_dimension_threshold:
            dimension_target = int(
                np.ceil(self.surrogate_samples_per_dimension * self.dimension)
            )
            self.max_surrogate_samples = min(
                self.max_evaluations,
                max(int(max_surrogate_samples), dimension_target),
            )
        else:
            self.max_surrogate_samples = int(max_surrogate_samples)
        self.max_active_variables = max(2, int(max_active_variables))
        self.block_mutation_size = block_mutation_size
        self.uncertainty_weight = float(uncertainty_weight)
        self.trust_threshold = float(trust_threshold)
        self.candidate_pool_multiplier = int(candidate_pool_multiplier)
        self.max_surrogate_terms = int(max_surrogate_terms)
        self.max_interaction_terms = int(max_interaction_terms)
        self.min_model_quality = float(min_model_quality)
        self.reserve_final_evaluation = bool(reserve_final_evaluation)
        self.exploration_fraction = float(np.clip(exploration_fraction, 0.05, 0.8))
        self.target_transform = str(target_transform).lower()
        self.target_clip = float(target_clip)
        self.local_correction = bool(local_correction)
        self.hierarchical_additive_backbone = bool(
            hierarchical_additive_backbone
        )
        self.adaptive_structure_selection = bool(
            adaptive_structure_selection
        )
        self.local_k = int(local_k)
        self.max_distance_reference = int(max_distance_reference)
        self.de_p_best_rate = float(de_p_best_rate)
        self.de_archive_multiplier = int(de_archive_multiplier)
        self.de_memory_size = int(de_memory_size)
        self.de_strategy_learning_rate = float(de_strategy_learning_rate)
        self.de_strategy_probability_floor = float(de_strategy_probability_floor)
        self.de_local_strategy_max_probability = float(
            de_local_strategy_max_probability
        )
        self.relative_improvement_mode = str(relative_improvement_mode).strip().lower()
        self.relative_improvement_tolerance = float(relative_improvement_tolerance)
        self.de_policy_probe_count = int(de_policy_probe_count)
        self.de_policy_update_interval = int(de_policy_update_interval)
        self.de_global_candidate_fraction = float(de_global_candidate_fraction)
        self.max_screening_audit_records = int(max_screening_audit_records)
        self.max_candidate_audit_records = int(max_candidate_audit_records)
        self.interaction_screening_interval = int(interaction_screening_interval)
        self.uncertainty_candidate_fraction = float(uncertainty_candidate_fraction)
        self.uncertainty_evaluation_interval = int(uncertainty_evaluation_interval)
        self.min_improvement_validation_samples = int(
            min_improvement_validation_samples
        )
        self.improvement_policy_weight = float(improvement_policy_weight)
        self.rng = np.random.default_rng(random_state)

        if library is None:
            self.library = PolynomialLibrary(
                degree=2,
                include_sine=False,
                include_cosine=False,
                harmonics=(1, 2),
                feature_names=[f"z{i}" for i in range(self.dimension)],
                bounds=self.bounds,
                include_interactions=False,
                include_all_main_effects=True,
                include_aggregate_terms=True,
                aggregate_harmonics=(1, 2),
                compute_backend=self.compute_backend,
            )
        else:
            self.library = library
        objective_complexity_budget = self.max_surrogate_terms
        if self.hierarchical_additive_backbone:
            objective_complexity_budget = max(
                objective_complexity_budget,
                self.dimension + self.max_interaction_terms,
            )
        self.surrogate = EnsembleSparseRegressor(
            library=self.library,
            n_models=max(1, int(n_ensemble_models)),
            threshold=0.035,
            max_terms=objective_complexity_budget,
            random_state=random_state,
            target_transform=self.target_transform,
            target_clip=self.target_clip,
            local_correction=self.local_correction,
            local_k=self.local_k,
            max_distance_reference=self.max_distance_reference,
            hierarchical_additive_backbone=(
                self.hierarchical_additive_backbone
            ),
            adaptive_structure_selection=(
                self.adaptive_structure_selection
            ),
            compute_backend=self.compute_backend,
        )

        self.state_names = ["diversity", "stagnation", "improvement_rate", "uncertainty"]
        if self.constraint is not None:
            self.state_names.insert(3, "feasible_ratio")
        self.de_strategy_names = {
            1: "current-to-pbest/1",
            2: "rand/1",
            3: "best/1",
            4: "gaussian-local",
        }
        self.de_strategy_feature_names = {
            1: "strategy_current_to_pbest_1",
            2: "strategy_rand_1",
            3: "strategy_best_1",
            4: "strategy_gaussian_local",
        }
        # The state block in the improvement features must match state_names exactly.
        # Constrained problems append feasible_ratio, so a fixed-length list is unsafe.
        improvement_names = [
            "step_size", "parent_gap", "parent_rank", "F", "CR",
            *self.state_names,
            "strategy_code",
            *[self.de_strategy_feature_names[index] for index in self.de_strategy_names],
            *[
                f"F_x_{self.de_strategy_feature_names[index]}"
                for index in self.de_strategy_names
            ],
            *[
                f"CR_x_{self.de_strategy_feature_names[index]}"
                for index in self.de_strategy_names
            ],
        ]
        self.improvement_library = PolynomialLibrary(
            degree=max(1, min(int(improvement_degree), 2)),
            feature_names=improvement_names,
            include_interactions=False,
            compute_backend=self.compute_backend,
        )
        self.improvement_model = EnsembleSparseRegressor(
            library=self.improvement_library,
            n_models=max(1, int(n_ensemble_models)),
            threshold=0.04,
            max_terms=14,
            random_state=None if random_state is None else random_state + 202,
            target_transform=self.target_transform,
            target_clip=self.target_clip,
            local_correction=self.local_correction,
            local_k=self.local_k,
            max_distance_reference=self.max_distance_reference,
            compute_backend=self.compute_backend,
        )
        self.policy_model = SparsePolicyModel(
            random_state=None if random_state is None else random_state + 303,
            state_names=self.state_names,
            target_transform=self.target_transform,
            target_clip=self.target_clip,
            local_correction=self.local_correction,
            local_k=self.local_k,
            max_distance_reference=self.max_distance_reference,
            compute_backend=self.compute_backend,
        )
        self.constraint_model = EnsembleSparseRegressor(
            library=PolynomialLibrary(
                degree=1,
                feature_names=[f"z{i}" for i in range(self.dimension)],
                bounds=self.bounds,
                include_interactions=False,
                include_all_main_effects=True,
                compute_backend=self.compute_backend,
            ),
            n_models=max(1, int(n_ensemble_models)),
            threshold=0.04,
            max_terms=min(self.max_surrogate_terms, 24),
            random_state=None if random_state is None else random_state + 101,
            target_transform=self.target_transform,
            target_clip=self.target_clip,
            local_correction=self.local_correction,
            local_k=self.local_k,
            max_distance_reference=self.max_distance_reference,
            compute_backend=self.compute_backend,
        )

        self.archive_x: List[Array] = []
        self.archive_y: List[float] = []
        self.archive_violation: List[float] = []
        self.improvement_archive_x: List[Array] = []
        self.improvement_archive_y: List[float] = []
        self.improvement_validation_records: List[Dict[str, float]] = []
        self.improvement_validation_count = 0
        self.policy_state_archive: List[Array] = []
        self.policy_f_archive: List[float] = []
        self.policy_cr_archive: List[float] = []
        # Policy supervision stores only true-evaluated offspring with a positive improvement.
        # This keeps the F/CR model from being trained on surrogate labels or failed trials.
        self.policy_feedback_archive: List[Dict[str, Any]] = []
        self.candidate_archive: List[Dict[str, Any]] = []
        self.candidate_archive_total_count = 0
        self.candidate_archive_dropped_count = 0
        self.history: List[Dict[str, Any]] = []
        self.validation_errors: List[float] = []
        self.last_policy: Dict[str, Any] = {}
        self.active_dimensions: List[int] = list(range(self.dimension))
        self.surrogate_fit_count = 0
        self.auxiliary_fit_count = 0
        self.final_confirmation = False
        self.final_confirmation_record: Dict[str, Any] = {}
        self.initial_design_size = 0
        self.screening_audit_records: List[Dict[str, Any]] = []
        self.screening_fit_count = 0
        self.real_selection_audit_records: List[Dict[str, Any]] = []
        self._stagnation = 0
        self.periodic_terms_admitted = False
        self._best_seen = np.inf
        self._last_improvement_rate = 0.0
        self.last_stagnation_diagnostics: Dict[str, Any] = {}
        self._last_pool: Optional[Dict[str, Any]] = None
        self.de_strategy_policy_evaluation: List[Dict[str, Any]] = []
        self.last_improvement_calibration: Dict[str, Any] = {}

        # SHADE-like memory is the robust parameter baseline.
        self.shade_m_f = np.full(self.de_memory_size, 0.70, dtype=float)
        self.shade_m_cr = np.full(self.de_memory_size, 0.80, dtype=float)
        self.shade_index = 0
        self.shade_success_log: List[Dict[str, float]] = []
        self.elite_x: Optional[Array] = None
        self.elite_value = np.inf
        self.elite_violation = np.inf
        self.last_trust_region_report: Dict[str, Any] = {}
        self.trust_region_scale = 1.0
        self.trust_region_feedback: List[Dict[str, Any]] = []
        # XDE-SINDy's DE layer keeps a separate direction archive.
        # Only successful true offspring update the strategy distribution.
        self.de_strategy_probabilities = np.asarray([0.42, 0.28, 0.15, 0.15], dtype=float)
        self.de_strategy_selected = np.zeros(4, dtype=float)
        self.de_strategy_successful = np.zeros(4, dtype=float)
        self.de_strategy_gain = np.zeros(4, dtype=float)
        self.de_strategy_update_log: List[Dict[str, Any]] = []
        self.de_archive_x: List[Array] = []
        self.de_archive_capacity = max(self.de_archive_multiplier * self.population_size, 2)
        self._cached_policy_selection: Optional[Dict[str, Any]] = None
        self._last_policy_probe_generation = -self.de_policy_update_interval

    def _shade_baseline_policy(self) -> Tuple[float, float]:
        """Return the current SHADE-like baseline parameters."""
        return (
            float(np.clip(np.mean(self.shade_m_f), 0.25, 1.0)),
            float(np.clip(np.mean(self.shade_m_cr), 0.05, 0.99)),
        )

    def _blend_improvement_policy(
        self,
        fallback_f: float,
        fallback_cr: float,
        proposed_f: float,
        proposed_cr: float,
    ) -> Tuple[float, float]:
        """Blend a validated model proposal with the robust SHADE anchor."""
        weight = self.improvement_policy_weight
        f_value = (1.0 - weight) * float(fallback_f) + weight * float(proposed_f)
        cr_value = (1.0 - weight) * float(fallback_cr) + weight * float(proposed_cr)
        return (
            float(np.clip(f_value, 0.25, 1.0)),
            float(np.clip(cr_value, 0.10, 0.99)),
        )

    def _update_trust_region_scale(
        self,
        predicted_gain: float,
        actual_gain: float,
    ) -> Dict[str, Any]:
        """Update the ellipsoid radius from transparent prediction agreement.

        The agreement ratio follows the trust-region principle, but the region
        remains the uncertainty-calibrated population ellipsoid. A reliable
        positive gain expands the next local search; a failed or badly
        over-predicted step contracts it.
        """
        predicted = float(predicted_gain)
        actual = float(actual_gain)
        denominator = max(abs(predicted), self.surrogate.target_scale * 1e-8, 1e-12)
        agreement = float(actual / denominator)
        old_scale = float(self.trust_region_scale)
        if actual <= 0.0 or agreement < 0.10:
            new_scale = 0.65 * old_scale
            action = "contract"
        elif agreement >= 0.75:
            new_scale = 1.20 * old_scale
            action = "expand"
        elif agreement < 0.25:
            new_scale = 0.85 * old_scale
            action = "contract"
        else:
            new_scale = old_scale
            action = "retain"
        self.trust_region_scale = float(np.clip(new_scale, 0.35, 1.80))
        record = {
            "generation": int(len(self.history)),
            "predicted_gain": predicted,
            "actual_gain": actual,
            "agreement_ratio": agreement,
            "scale_before": old_scale,
            "scale_after": float(self.trust_region_scale),
            "action": action,
            "update_rule": (
                "expand by 1.20 when actual_gain>0 and agreement>=0.75; "
                "contract by 0.65 after failure or agreement<0.10; contract by "
                "0.85 when agreement<0.25; otherwise retain; clip to [0.35,1.80]"
            ),
        }
        self.trust_region_feedback.append(record)
        return record

    def _update_shade_memory(
        self,
        f_values: Array,
        cr_values: Array,
        improvements: Array,
    ) -> None:
        """Update SHADE memory using successful real offspring only."""
        f_values = np.asarray(f_values, dtype=float).reshape(-1)
        cr_values = np.asarray(cr_values, dtype=float).reshape(-1)
        improvements = np.asarray(improvements, dtype=float).reshape(-1)
        mask = (
            np.isfinite(f_values)
            & np.isfinite(cr_values)
            & np.isfinite(improvements)
            & (improvements > 0.0)
        )
        if not np.any(mask):
            return
        f_success = np.clip(f_values[mask], 0.25, 1.0)
        cr_success = np.clip(cr_values[mask], 0.05, 0.99)
        weights = improvements[mask]
        weighted_f = float(
            np.sum(weights * f_success * f_success)
            / max(np.sum(weights * f_success), 1e-12)
        )
        weighted_cr = float(
            np.sum(weights * cr_success) / max(np.sum(weights), 1e-12)
        )
        index = int(self.shade_index % len(self.shade_m_f))
        self.shade_m_f[index] = float(
            np.clip(0.90 * self.shade_m_f[index] + 0.10 * weighted_f, 0.25, 1.0)
        )
        self.shade_m_cr[index] = float(
            np.clip(0.90 * self.shade_m_cr[index] + 0.10 * weighted_cr, 0.05, 0.99)
        )
        self.shade_index = (index + 1) % len(self.shade_m_f)
        self.shade_success_log.append({
            "memory_index": float(index),
            "weighted_f": weighted_f,
            "weighted_cr": weighted_cr,
            "total_improvement": float(np.sum(weights)),
            "success_count": float(np.sum(mask)),
        })

    def _append_de_archive(self, point: Array) -> None:
        """Insert a replaced parent into the bounded DE direction archive."""
        self.de_archive_x.append(np.asarray(point, dtype=float).copy())
        while len(self.de_archive_x) > self.de_archive_capacity:
            remove_index = int(self.rng.integers(0, len(self.de_archive_x)))
            del self.de_archive_x[remove_index]

    @staticmethod
    def _project_strategy_probabilities(values: Array, floor: float = 0.05) -> Array:
        """Project a strategy distribution onto the simplex with a strictly positive floor."""
        projected = np.maximum(np.asarray(values, dtype=float).reshape(-1), 0.0)
        total = float(np.sum(projected))
        if total <= 0.0:
            projected = np.full(projected.size, 1.0 / projected.size, dtype=float)
        else:
            projected = projected / total
        dimension = int(projected.size)
        if dimension * floor >= 1.0:
            return np.full(dimension, 1.0 / dimension, dtype=float)
        for _ in range(dimension):
            below = projected < floor - 1e-12
            if not np.any(below):
                break
            projected[below] = floor
            remaining = 1.0 - floor * float(np.sum(below))
            free = ~below
            free_total = float(np.sum(projected[free]))
            if free_total <= 0.0:
                projected[free] = remaining / max(int(np.sum(free)), 1)
            else:
                projected[free] = remaining * projected[free] / free_total
        return projected / np.sum(projected)

    def _stabilize_de_strategy_probabilities(self, values: Array) -> Array:
        """Keep high-dimensional DE search from collapsing onto local mutation."""
        floor = self.de_strategy_probability_floor
        projected = self._project_strategy_probabilities(values, floor=floor)
        if self.dimension < self.de_high_dimension_threshold:
            return projected

        local_index = len(projected) - 1
        cap = self.de_local_strategy_max_probability
        if projected[local_index] <= cap + 1e-12:
            return projected

        excess = float(projected[local_index] - cap)
        projected[local_index] = cap
        remainder = projected[:-1]
        free = np.maximum(remainder - floor, 0.0)
        free_total = float(np.sum(free))
        if free_total > 1e-12:
            remainder += excess * free / free_total
        else:
            remainder += excess / max(len(remainder), 1)
        projected[:-1] = remainder
        return projected / np.sum(projected)

    def _update_de_strategy_probabilities(
        self,
        selected_counts: Array,
        successful_gains: Array,
        generation: int,
    ) -> None:
        """Update strategy probabilities from true offspring improvements."""
        selected_counts = np.asarray(selected_counts, dtype=float).reshape(-1)
        successful_gains = np.asarray(successful_gains, dtype=float).reshape(-1)
        quality = np.divide(
            successful_gains,
            np.maximum(selected_counts, 1.0),
            out=np.zeros_like(successful_gains),
            where=selected_counts > 0.0,
        )
        old = self.de_strategy_probabilities.copy()
        if np.any(quality > 0.0):
            target = (quality + 1e-12) / np.sum(quality + 1e-12)
            learning_rate = self.de_strategy_learning_rate
            updated = (1.0 - learning_rate) * old + learning_rate * target
            self.de_strategy_probabilities = self._stabilize_de_strategy_probabilities(updated)
        self.de_strategy_update_log.append({
            "generation": float(generation),
            "selected_counts": selected_counts.tolist(),
            "successful_gains": successful_gains.tolist(),
            "quality": quality.tolist(),
            "probabilities_before": old.tolist(),
            "probabilities_after": self.de_strategy_probabilities.tolist(),
            "update_rule": (
                "p_new=high_dimensional_floor_cap((1-lambda)*p_old+lambda*q); "
                "q=successful_gain/selected_count; "
                f"lambda={self.de_strategy_learning_rate:.6g}"
            ),
        })

    @property
    def _lower(self) -> Array:
        return self.bounds[:, 0]

    @property
    def _upper(self) -> Array:
        return self.bounds[:, 1]

    def _sample_population(self) -> Array:
        design = latin_hypercube(self.population_size, self.dimension, self.rng)
        return self.compute_backend.scale_unit_points(
            design, self._lower, self._upper
        )

    def _constraint_violation(self, point: Array) -> float:
        if self.constraint is None:
            return 0.0
        result = np.asarray(self.constraint(np.asarray(point, dtype=float)), dtype=float)
        if result.size == 0:
            return 0.0
        return float(np.sum(np.maximum(result.reshape(-1), 0.0)))

    def _evaluate(self, point: Array) -> Tuple[float, float]:
        value = float(self.objective(np.asarray(point, dtype=float)))
        if not np.isfinite(value):
            value = float("inf")
        violation = self._constraint_violation(point)
        self.archive_x.append(np.asarray(point, dtype=float).copy())
        self.archive_y.append(value)
        self.archive_violation.append(violation)
        return value, violation

    def _evaluate_batch(self, points: Array) -> List[Tuple[float, float]]:
        """Evaluate real points in one call when the objective supports it."""
        values_x = np.asarray(points, dtype=float)
        if values_x.ndim != 2 or values_x.shape[1] != self.dimension:
            raise ValueError("points must have shape (n_points, n_features)")
        if len(values_x) == 0:
            return []

        device_evaluator = getattr(self.objective, "evaluate_batch_device", None)
        if self.gpu_objective and callable(device_evaluator):
            raw_values = device_evaluator(
                values_x,
                backend=self.compute_backend,
            )
            objective_values = np.asarray(
                self.compute_backend.to_numpy(raw_values), dtype=float
            ).reshape(-1)
        else:
            batch_evaluator = getattr(self.objective, "evaluate_batch", None)
            if callable(batch_evaluator):
                objective_values = np.asarray(
                    batch_evaluator(values_x), dtype=float
                ).reshape(-1)
            else:
                objective_values = np.asarray(
                    [self.objective(point) for point in values_x], dtype=float
                ).reshape(-1)
        if len(objective_values) != len(values_x):
            raise ValueError("batch objective returned the wrong number of values")

        results: List[Tuple[float, float]] = []
        for point, raw_value in zip(values_x, objective_values):
            value = float(raw_value)
            if not np.isfinite(value):
                value = float("inf")
            violation = self._constraint_violation(point)
            self.archive_x.append(point.copy())
            self.archive_y.append(value)
            self.archive_violation.append(violation)
            results.append((value, violation))
        return results

    def _nearest_archive_distance(
        self,
        point: Array,
        *,
        max_block_size: int = 1024,
        scale: Array | None = None,
    ) -> float:
        """Return the nearest archive distance without materializing the archive."""
        if not self.archive_x:
            return float("inf")
        query = np.asarray(point, dtype=float).reshape(-1)
        divisor = np.ones_like(query) if scale is None else np.asarray(scale, dtype=float)
        divisor = np.maximum(divisor, 1e-12)
        block_size = max(1, int(max_block_size))
        nearest = float("inf")
        for start in range(0, len(self.archive_x), block_size):
            block = np.asarray(self.archive_x[start : start + block_size], dtype=float)
            distances = np.linalg.norm((block - query) / divisor, axis=1)
            nearest = min(nearest, float(np.min(distances)))
        return nearest

    def _surrogate_training_data(self) -> Tuple[Array, Array, Array]:
        """Build a bounded, auditable training view of the true archive."""
        full_y = np.asarray(self.archive_y, dtype=float)
        limit = self.max_surrogate_samples
        if limit is None or len(full_y) <= limit:
            indices = np.arange(len(full_y), dtype=int)
            x = np.asarray(self.archive_x, dtype=float)
        else:
            limit = int(limit)
            elite_count = max(1, limit // 2)
            elite = np.argsort(full_y, kind="stable")[:elite_count]
            recent = np.arange(max(0, len(full_y) - (limit - elite_count)), len(full_y))
            indices = np.unique(np.concatenate([elite, recent]))
            if len(indices) < limit:
                remaining = np.setdiff1d(
                    np.arange(len(full_y)), indices, assume_unique=False
                )
                indices = np.concatenate([indices, remaining[: limit - len(indices)]])
            indices = np.asarray(indices[:limit], dtype=int)
            x = np.asarray([self.archive_x[int(index)] for index in indices], dtype=float)
        return x, full_y[indices], np.asarray(indices, dtype=int)

    @staticmethod
    def _is_better(value: float, violation: float, incumbent_value: float, incumbent_violation: float) -> bool:
        if violation < incumbent_violation - 1e-14:
            return True
        if violation > incumbent_violation + 1e-14:
            return False
        return value < incumbent_value

    def _is_meaningful_improvement(
        self, current_best: float, previous_best: float
    ) -> bool:
        """Require observed, model-aware progress before resetting stagnation."""
        if not np.isfinite(current_best) or not np.isfinite(previous_best):
            return bool(np.isfinite(current_best) and not np.isfinite(previous_best))
        if current_best >= previous_best:
            self._stagnation_tolerance(current_best, previous_best)
            return False
        tolerance = self._stagnation_tolerance(current_best, previous_best)
        return (float(previous_best) - float(current_best)) > tolerance

    def _stagnation_tolerance(
        self, current_best: float, previous_best: float
    ) -> float:
        """Estimate a meaningful improvement from observed values and model error.

        The fixed relative tolerance remains a lower bound. In automatic mode,
        recent true objective dispersion and the fitted SINDy ensemble's
        out-of-fold RMSE raise that bound when the observed resolution or model
        error makes a smaller change unreliable.
        """
        objective_scale = max(1.0, abs(float(previous_best)))
        relative_floor = self.relative_improvement_tolerance * objective_scale
        if self.relative_improvement_mode == "fixed":
            tolerance = max(1.0e-12, relative_floor)
            self.last_stagnation_diagnostics = {
                "mode": "fixed",
                "tolerance": float(tolerance),
                "relative_floor": float(relative_floor),
                "observed_iqr": None,
                "surrogate_cv_rmse": None,
                "surrogate_quality": None,
            }
            return float(tolerance)

        finite_values = np.asarray(
            [value for value in self.archive_y if np.isfinite(value)], dtype=float
        )
        recent_values = finite_values[-32:]
        if recent_values.size >= 4:
            q25, q75 = np.percentile(recent_values, [25.0, 75.0])
            observed_iqr = max(0.0, float(q75 - q25))
        else:
            observed_iqr = 0.0

        cv_rmse = float(self.surrogate.cv_rmse)
        if not self.surrogate.fitted or not np.isfinite(cv_rmse):
            model_error = 0.0
            reported_cv_rmse = None
            reported_quality = None
        else:
            quality = float(np.clip(self.surrogate.quality_score, 0.0, 1.0))
            # Low-quality models receive a larger error floor. The true
            # objective, rather than a surrogate prediction, still decides
            # whether progress occurred.
            model_error = cv_rmse * (1.0 + (1.0 - quality))
            reported_cv_rmse = cv_rmse
            reported_quality = quality

        observed_component = 0.01 * observed_iqr
        tolerance = max(
            1.0e-12,
            relative_floor,
            observed_component,
            model_error,
        )
        self.last_stagnation_diagnostics = {
            "mode": "auto",
            "tolerance": float(tolerance),
            "relative_floor": float(relative_floor),
            "observed_iqr": float(observed_iqr),
            "observed_component": float(observed_component),
            "surrogate_cv_rmse": reported_cv_rmse,
            "surrogate_model_error_floor": float(model_error),
            "surrogate_quality": reported_quality,
            "current_best": float(current_best),
            "previous_best": float(previous_best),
        }
        return float(tolerance)

    def _best_population_index(self, values: Array, violations: Array) -> int:
        feasible = np.where(violations <= 0.0, values, np.inf)
        if np.any(np.isfinite(feasible)):
            return int(np.argmin(feasible))
        return int(np.argmin(violations))

    def _worst_population_index(self, values: Array, violations: Array) -> int:
        """Return the population member that is most suitable for replacement by an initial-design point."""
        feasible = np.where(violations <= 0.0)[0]
        if len(feasible):
            return int(feasible[np.argmax(values[feasible])])
        return int(np.lexsort((values, violations))[-1])

    def _best_archive_index(self) -> int:
        violations = np.asarray(self.archive_violation, dtype=float)
        values = np.asarray(self.archive_y, dtype=float)
        feasible = np.where(violations <= 0.0, values, np.inf)
        if np.any(np.isfinite(feasible)):
            return int(np.argmin(feasible))
        return int(np.argmin(violations))

    def _state_features(self, population: Array) -> Array:
        normalized = np.asarray((population - self._lower) / (self._upper - self._lower), dtype=float)
        diversity = float(np.mean(np.std(normalized, axis=0) * 2.0))
        diversity = float(np.clip(diversity, 0.0, 1.0))
        stagnation = float(np.clip(self._stagnation / 20.0, 0.0, 1.0))
        uncertainty = 0.0
        if self.surrogate.fitted:
            _, spread, distance = self.surrogate.predict_with_details(population)
            scale = max(self.surrogate.target_scale, float(np.std(self.archive_y)), 1e-8)
            uncertainty = float(np.mean(spread) / scale + np.mean(distance))
            uncertainty = float(np.clip(uncertainty, 0.0, 1.0))
        values = [diversity, stagnation, self._last_improvement_rate]
        if self.constraint is not None:
            feasible = 1.0
            if self.archive_violation:
                feasible = float(np.mean(np.asarray(self.archive_violation[-self.population_size:]) <= 0.0))
            values.append(feasible)
        values.append(uncertainty)
        return np.asarray(values, dtype=float)

    def _safe_policy(self, state: Array) -> Tuple[float, float]:
        diversity = float(state[0])
        stagnation = float(state[1])
        improvement = float(state[2])
        uncertainty = float(state[-1])
        f_value = 0.42 + 0.30 * stagnation + 0.10 * uncertainty + 0.16 * diversity
        cr_value = 0.74 + 0.10 * (1.0 - diversity) + 0.10 * improvement - 0.08 * uncertainty
        return float(np.clip(f_value, 0.30, 0.95)), float(np.clip(cr_value, 0.20, 0.98))

    def _policy(self, population: Array) -> Tuple[float, float, Array]:
        state = self._state_features(population)
        safe_f, safe_cr = self._safe_policy(state)
        shade_f, shade_cr = self._shade_baseline_policy()
        # Keep a stable SHADE-like anchor when the sparse policy is noisy.
        fallback_f = 0.45 * safe_f + 0.55 * shade_f
        fallback_cr = 0.45 * safe_cr + 0.55 * shade_cr
        f_value, cr_value = fallback_f, fallback_cr
        source = 0.0
        if self.policy_model.fitted and self.policy_model.quality_score >= 0.65:
            learned_f, learned_cr = self.policy_model.predict(state)
            f_value = 0.35 * fallback_f + 0.65 * learned_f
            cr_value = 0.35 * fallback_cr + 0.65 * learned_cr
            source = 1.0
        self.last_policy = {
            "F": float(f_value),
            "CR": float(cr_value),
            "diversity": float(state[0]),
            "stagnation": float(state[1]),
            "improvement_rate": float(state[2]),
            "uncertainty_signal": float(state[-1]),
            "policy_model_source": source,
            "shade_baseline_F": float(shade_f),
            "shade_baseline_CR": float(shade_cr),
            "improvement_policy_used": 0.0,
            "improvement_policy_accepted": 0.0,
        }
        if self.constraint is not None:
            self.last_policy["feasible_ratio"] = float(state[3])
        return float(f_value), float(cr_value), state

    def _improvement_reference_scale(self) -> float:
        """Return a robust scale for comparing predicted and observed gains."""
        values = np.asarray(self.improvement_archive_y, dtype=float)
        values = values[np.isfinite(values)]
        if values.size > 1:
            center = float(np.median(values))
            mad = float(np.median(np.abs(values - center)))
            robust = 1.4826 * mad
            spread = float(np.std(values))
        else:
            robust = 0.0
            spread = 0.0
        model_scale = (
            float(self.improvement_model.target_scale)
            if self.improvement_model.fitted
            else 1.0
        )
        return max(robust, spread, model_scale, 1e-8)

    def _improvement_model_reliability(self) -> Dict[str, Any]:
        """Audit whether predicted offspring gains are trustworthy for decisions."""
        recent = self.improvement_validation_records[-64:]
        finite = [
            row for row in recent
            if np.isfinite(row.get("predicted_gain", np.nan))
            and np.isfinite(row.get("actual_gain", np.nan))
        ]
        sample_count = len(finite)
        base = {
            "reliable": False,
            "sample_count": int(sample_count),
            "required_samples": int(self.min_improvement_validation_samples),
            "sign_accuracy": 0.0,
            "positive_precision": 0.0,
            "normalized_median_absolute_error": None,
            "reason": "insufficient_true_validation",
            "validation_window": 64,
        }
        if sample_count < self.min_improvement_validation_samples:
            return base
        predicted = np.asarray(
            [row["predicted_gain"] for row in finite], dtype=float
        )
        actual = np.asarray(
            [row["actual_gain"] for row in finite], dtype=float
        )
        sign_accuracy = float(np.mean((predicted > 0.0) == (actual > 0.0)))
        predicted_positive = predicted > 0.0
        positive_precision = (
            float(np.mean(actual[predicted_positive] > 0.0))
            if np.any(predicted_positive)
            else 0.0
        )
        actual_scale = max(
            float(np.median(np.abs(actual))),
            0.25 * float(np.std(actual)),
            1e-8,
        )
        normalized_mae = float(
            np.median(np.abs(predicted - actual)) / actual_scale
        )
        reliable = bool(
            self.improvement_model.fitted
            and self.improvement_model.quality_score >= self.min_model_quality
            and sign_accuracy >= 0.60
            and positive_precision >= 0.55
            and normalized_mae <= 0.75
        )
        return {
            **base,
            "reliable": reliable,
            "sign_accuracy": sign_accuracy,
            "positive_precision": positive_precision,
            "normalized_median_absolute_error": normalized_mae,
            "reason": "empirically_reliable" if reliable else "empirical_validation_failed",
            "thresholds": {
                "minimum_sign_accuracy": 0.60,
                "minimum_positive_precision": 0.55,
                "maximum_normalized_median_absolute_error": 0.75,
            },
        }

    def _calibrate_improvement_predictions(
        self,
        predictions: Array,
    ) -> Tuple[Array, Dict[str, Any]]:
        """Clip extrapolated gain predictions to an auditable empirical range."""
        raw = np.asarray(predictions, dtype=float).reshape(-1)
        values = np.asarray(self.improvement_archive_y, dtype=float)
        values = values[np.isfinite(values)]
        scale = self._improvement_reference_scale()
        if values.size >= 3:
            center = float(np.median(values))
            lower_quantile, upper_quantile = np.quantile(values, [0.05, 0.95])
            lower = float(min(lower_quantile - 2.0 * scale, center - 2.0 * scale))
            upper = float(max(upper_quantile + 2.0 * scale, center + 2.0 * scale))
        else:
            center = 0.0
            lower = -2.0 * scale
            upper = 2.0 * scale
        finite = np.nan_to_num(raw, nan=center, posinf=upper, neginf=lower)
        calibrated = np.clip(finite, lower, upper)
        audit = {
            "center": float(center),
            "scale": float(scale),
            "lower": float(lower),
            "upper": float(upper),
            "clipped": bool(np.any(np.abs(calibrated - finite) > 1e-12)),
            "rule": (
                "gain=clip(raw_gain, [min(q05-2s, median-2s), "
                "max(q95+2s, median+2s)]); s=max(robust_MAD, std, model_scale)"
            ),
            "observed_sample_count": int(values.size),
        }
        self.last_improvement_calibration = audit
        return calibrated, audit

    def _improvement_features(
        self,
        population: Array,
        trials: Array,
        parent_indices: Array,
        f_values: Array,
        cr_values: Array,
        strategy_values: Array,
        state: Array,
        population_values: Array,
    ) -> Array:
        archive_scale = (
            float(np.std(self.archive_y))
            if len(self.archive_y) > 1
            else 0.0
        )
        scale = max(
            archive_scale,
            self.surrogate.target_scale if self.surrogate.fitted else 1.0,
            1e-8,
        )
        strategy_ids = np.asarray(strategy_values, dtype=int).reshape(-1)
        if len(strategy_ids) != len(trials):
            raise ValueError("strategy_values must match the number of trials")
        return self.compute_backend.candidate_features(
            population,
            trials,
            np.asarray(parent_indices, dtype=int),
            np.asarray(f_values, dtype=float),
            np.asarray(cr_values, dtype=float),
            strategy_ids,
            np.asarray(state, dtype=float),
            np.asarray(population_values, dtype=float),
            self._lower,
            self._upper,
            scale,
            len(self.de_strategy_names),
        )

    def _select_policy_by_improvement(
        self,
        population: Array,
        population_values: Array,
        state: Array,
        fallback_f: float,
        fallback_cr: float,
    ) -> Tuple[float, float, Dict[str, Any]]:
        probability_strategy = int(np.argmax(self.de_strategy_probabilities)) + 1
        metadata: Dict[str, Any] = {
            "improvement_policy_used": 0.0,
            "improvement_policy_accepted": 0.0,
            "policy_guard_triggered": 0.0,
            "predicted_policy_gain": 0.0,
            "predicted_policy_uncertainty": 0.0,
            "policy_candidates": 0.0,
            "de_strategy_selected": probability_strategy,
            "de_strategy_name": self.de_strategy_names[probability_strategy],
            "de_strategy_score": 0.0,
            "de_strategy_predicted_gain": 0.0,
            "de_strategy_uncertainty": 0.0,
            "de_strategy_model_ready": float(self.improvement_model.fitted),
            "policy_probe_reused": 0.0,
            "policy_probe_refreshed": 0.0,
        }

        parameter_candidates: List[Tuple[float, float]] = [
            (fallback_f, fallback_cr),
            (self._safe_policy(state)[0], self._safe_policy(state)[1]),
            self._shade_baseline_policy(),
            (0.45, 0.80),
            (0.60, 0.75),
            (0.75, 0.70),
            (0.90, 0.55),
        ]
        unique_candidates: List[Tuple[float, float]] = []
        for f_value, cr_value in parameter_candidates:
            candidate = (
                float(np.clip(f_value, 0.25, 1.0)),
                float(np.clip(cr_value, 0.10, 0.99)),
            )
            if not any(
                abs(candidate[0] - old[0]) < 1e-12
                and abs(candidate[1] - old[1]) < 1e-12
                for old in unique_candidates
            ):
                unique_candidates.append(candidate)

        reliability = self._improvement_model_reliability()
        model_ready = bool(reliability["reliable"])
        metadata["improvement_model_reliable"] = float(model_ready)
        metadata["improvement_validation_samples"] = float(
            reliability["sample_count"]
        )
        metadata["improvement_sign_accuracy"] = float(
            reliability["sign_accuracy"]
        )
        if not model_ready:
            self.de_strategy_policy_evaluation.append({
                "generation": int(len(self.history)),
                "model_ready": False,
                "model_quality": float(self.improvement_model.quality_score),
                "empirical_reliability": dict(reliability),
                "probe_count": 0,
                "parameter_candidates": [
                    {"F": float(f_value), "CR": float(cr_value)}
                    for f_value, cr_value in unique_candidates
                ],
                "strategy_scores": {str(index): None for index in self.de_strategy_names},
                "selected_strategy": probability_strategy,
                "selected_strategy_name": self.de_strategy_names[probability_strategy],
                "selected_F": float(fallback_f),
                "selected_CR": float(fallback_cr),
                "selection_source": "strategy_probability_prior",
                "selection_rule": (
                    "before the improvement model passes true-outcome reliability checks, "
                    "use the highest-probability DE strategy and the SHADE-like parameter fallback"
                ),
            })
            return fallback_f, fallback_cr, metadata

        generation = int(len(self.history))
        if (
            self._cached_policy_selection is not None
            and generation - self._last_policy_probe_generation < self.de_policy_update_interval
        ):
            cached = self._cached_policy_selection
            cached_metadata = dict(cached["metadata"])
            metadata.update(cached_metadata)
            metadata["policy_probe_reused"] = 1.0
            metadata["policy_probe_refreshed"] = 0.0
            self.de_strategy_policy_evaluation.append({
                "generation": generation,
                "model_ready": True,
                "model_quality": float(self.improvement_model.quality_score),
                "probe_count": 0,
                "probe_reused": True,
                "source_generation": int(cached["generation"]),
                "strategy_scores": dict(cached["strategy_scores"]),
                "strategy_model_scores": dict(cached["strategy_model_scores"]),
                "strategy_empirical_scores": dict(cached["strategy_empirical_scores"]),
                "strategy_blended_scores": dict(cached["strategy_blended_scores"]),
                "selected_strategy": int(metadata["de_strategy_selected"]),
                "selected_strategy_name": str(metadata["de_strategy_name"]),
                "selected_F": float(cached["F"]),
                "selected_CR": float(cached["CR"]),
                "selection_rule": (
                    "reuse the last fully audited improvement-model decision inside "
                    "the policy refresh interval"
                ),
            })
            return float(cached["F"]), float(cached["CR"]), metadata

        # The operator identity follows accumulated true gains. The improvement
        # ensemble probes executable F/CR settings only for that empirical arm.
        probe_count = min(self.population_size, self.de_policy_probe_count)
        strategy_scores: Dict[str, float | None] = {}
        strategy_gains: Dict[str, float | None] = {}
        strategy_uncertainties: Dict[str, float | None] = {}
        best_by_strategy: Dict[str, Dict[str, float]] = {}
        strategy_details: Dict[str, List[Dict[str, float]]] = {}
        strategy_model_scores: Dict[str, float | None] = {}
        strategy_empirical_scores: Dict[str, float] = {}
        strategy_blended_scores: Dict[str, float | None] = {}
        gain_scale = self._improvement_reference_scale()
        # The operator arm is selected from true realized gains. The sparse
        # improvement model is responsible only for proposing F/CR on that
        # empirically supported arm, which avoids a second, model-only strategy
        # controller and reduces virtual probing by a factor of four.
        strategy_indices = [probability_strategy]
        for strategy_index in strategy_indices:
            empirical_gain = self.de_strategy_gain[strategy_index - 1] / max(
                self.de_strategy_selected[strategy_index - 1], 1.0
            )
            strategy_empirical_scores[str(strategy_index)] = float(
                empirical_gain / gain_scale
            )
        for strategy_index in strategy_indices:
            arm_scores: List[Dict[str, float]] = []
            for candidate_index, (f_value, cr_value) in enumerate(unique_candidates):
                probe_pool = self._generate_candidate_pool(
                    population,
                    population_values,
                    f_value,
                    cr_value,
                    state,
                    strategy_override=strategy_index,
                    count=probe_count,
                    record=False,
                    probe_mode=True,
                )
                gain = np.asarray(
                    probe_pool.get("predicted_gain", np.zeros(probe_count)),
                    dtype=float,
                )
                spread = np.asarray(
                    probe_pool.get("gain_uncertainty", np.zeros(probe_count)),
                    dtype=float,
                )
                normalized_spread = np.asarray(spread, dtype=float) / max(
                    self.improvement_model.target_scale, 1e-8
                )
                finite = np.isfinite(gain) & np.isfinite(normalized_spread)
                if np.any(finite):
                    mean_gain = float(np.mean(gain[finite]))
                    mean_uncertainty = float(np.mean(normalized_spread[finite]))
                    model_score = float(
                        mean_gain / gain_scale
                        - self.uncertainty_weight * mean_uncertainty
                    )
                    empirical_score = strategy_empirical_scores[str(strategy_index)]
                    score = float(0.75 * model_score + 0.25 * empirical_score)
                else:
                    mean_gain = 0.0
                    mean_uncertainty = float("inf")
                    model_score = float("-inf")
                    empirical_score = strategy_empirical_scores[str(strategy_index)]
                    score = float("-inf")
                arm_scores.append({
                    "candidate_index": float(candidate_index),
                    "F": float(f_value),
                    "CR": float(cr_value),
                    "score": score,
                    "predicted_gain": mean_gain,
                    "predicted_uncertainty": mean_uncertainty,
                    "model_score": model_score,
                    "empirical_score": empirical_score,
                    "blended_score": score,
                })
            strategy_details[str(strategy_index)] = arm_scores
            best_arm = max(arm_scores, key=lambda item: item["score"])
            strategy_model_scores[str(strategy_index)] = (
                None
                if not np.isfinite(best_arm["model_score"])
                else float(best_arm["model_score"])
            )
            strategy_blended_scores[str(strategy_index)] = (
                None
                if not np.isfinite(best_arm["blended_score"])
                else float(best_arm["blended_score"])
            )
            strategy_scores[str(strategy_index)] = (
                None if not np.isfinite(best_arm["score"]) else float(best_arm["score"])
            )
            strategy_gains[str(strategy_index)] = float(best_arm["predicted_gain"])
            strategy_uncertainties[str(strategy_index)] = (
                None
                if not np.isfinite(best_arm["predicted_uncertainty"])
                else float(best_arm["predicted_uncertainty"])
            )
            best_by_strategy[str(strategy_index)] = dict(best_arm)

        finite_strategy_scores = {
            int(index): float(score)
            for index, score in strategy_scores.items()
            if score is not None and np.isfinite(score)
        }
        if not finite_strategy_scores:
            selected_strategy = probability_strategy
            selected_f, selected_cr = fallback_f, fallback_cr
            proposed_f, proposed_cr = fallback_f, fallback_cr
            best_score = float("-inf")
            best_gain = 0.0
            best_uncertainty = float("inf")
            baseline_score = float("-inf")
            baseline_gain = 0.0
            guard_margin = 1.0
            guard_triggered = True
        else:
            # Strategy identity remains controlled by true-gain probabilities.
            # The validated sparse model proposes only F/CR for that empirical arm.
            selected_strategy = probability_strategy
            best_arm = best_by_strategy[str(probability_strategy)]
            best_score = float(best_arm["score"])
            best_gain = float(best_arm["predicted_gain"])
            best_uncertainty = float(best_arm["predicted_uncertainty"])
            baseline_arm = min(
                strategy_details[str(probability_strategy)],
                key=lambda item: abs(item["F"] - fallback_f) + abs(item["CR"] - fallback_cr),
            )
            baseline_score = float(baseline_arm["score"])
            baseline_gain = float(baseline_arm["predicted_gain"])
            guard_margin = 0.01 * max(abs(best_gain) / gain_scale, 1.0)
            guard_triggered = (
                not np.isfinite(best_score)
                or not np.isfinite(baseline_score)
                or best_score <= baseline_score + guard_margin
                or best_gain < -0.50 * gain_scale
            )
            if guard_triggered:
                selected_strategy = probability_strategy
                selected_f, selected_cr = float(fallback_f), float(fallback_cr)
                proposed_f, proposed_cr = float(best_arm["F"]), float(best_arm["CR"])
            else:
                proposed_f = float(best_arm["F"])
                proposed_cr = float(best_arm["CR"])
                selected_f, selected_cr = self._blend_improvement_policy(
                    fallback_f,
                    fallback_cr,
                    proposed_f,
                    proposed_cr,
                )

        metadata["improvement_policy_used"] = 1.0
        metadata["policy_probe_refreshed"] = 1.0
        metadata["policy_candidates"] = float(len(unique_candidates))
        metadata["predicted_policy_gain"] = float(best_gain)
        metadata["predicted_policy_uncertainty"] = (
            0.0 if not np.isfinite(best_uncertainty) else float(best_uncertainty)
        )
        metadata["policy_guard_triggered"] = float(guard_triggered)
        metadata["de_strategy_selected"] = int(selected_strategy)
        metadata["de_strategy_name"] = self.de_strategy_names[int(selected_strategy)]
        metadata["de_strategy_score"] = (
            0.0 if not np.isfinite(best_score) else float(best_score)
        )
        metadata["de_strategy_predicted_gain"] = float(best_gain)
        metadata["de_strategy_uncertainty"] = (
            0.0 if not np.isfinite(best_uncertainty) else float(best_uncertainty)
        )
        metadata["model_proposed_F"] = float(proposed_f)
        metadata["model_proposed_CR"] = float(proposed_cr)
        metadata["improvement_policy_weight"] = float(
            self.improvement_policy_weight
        )
        if guard_triggered:
            metadata["policy_guard_triggered"] = 1.0
            selected_f, selected_cr = float(fallback_f), float(fallback_cr)
        else:
            metadata["improvement_policy_accepted"] = 1.0

        audit_record = {
            "generation": int(len(self.history)),
            "model_ready": True,
            "model_quality": float(self.improvement_model.quality_score),
            "probe_count": int(probe_count),
            "parameter_candidates": [
                {"index": int(index), "F": float(f_value), "CR": float(cr_value)}
                for index, (f_value, cr_value) in enumerate(unique_candidates)
            ],
            "strategy_scores": strategy_scores,
            "strategy_model_scores": strategy_model_scores,
            "strategy_empirical_scores": strategy_empirical_scores,
            "strategy_blended_scores": strategy_blended_scores,
            "strategy_predicted_gain": strategy_gains,
            "strategy_predicted_uncertainty": strategy_uncertainties,
            "best_parameters_by_strategy": best_by_strategy,
            "selected_strategy": int(selected_strategy),
            "selected_strategy_name": self.de_strategy_names[int(selected_strategy)],
            "selected_F": float(selected_f),
            "selected_CR": float(selected_cr),
            "baseline_strategy": int(probability_strategy),
            "baseline_score": None if not np.isfinite(baseline_score) else float(baseline_score),
            "baseline_predicted_gain": float(baseline_gain),
            "best_score": None if not np.isfinite(best_score) else float(best_score),
            "gain_reference_scale": float(gain_scale),
            "guard_margin": float(guard_margin),
            "guard_triggered": bool(guard_triggered),
            "selection_rule": (
                "select the operator arm from realized true-evaluation gains; probe only "
                "that empirically supported arm with the sparse improvement ensemble; "
                "clip extrapolated gains to the observed range and accept the proposed "
                "F/CR adjustment only when it exceeds the SHADE parameter baseline"
            ),
        }
        self.de_strategy_policy_evaluation.append(audit_record)
        self._last_policy_probe_generation = generation
        self._cached_policy_selection = {
            "generation": generation,
            "F": float(selected_f),
            "CR": float(selected_cr),
            "metadata": dict(metadata),
            "strategy_scores": dict(strategy_scores),
            "strategy_model_scores": dict(strategy_model_scores),
            "strategy_empirical_scores": dict(strategy_empirical_scores),
            "strategy_blended_scores": dict(strategy_blended_scores),
        }
        return float(selected_f), float(selected_cr), metadata

    def _draw_candidate_indices(self, count: int) -> Tuple[Array, Array, Array, Array]:
        parents = np.arange(count, dtype=int) % self.population_size
        self.rng.shuffle(parents)
        # Random keys give each row a uniform permutation without a Python loop.
        # Setting the parent key to +inf removes it before selecting the first
        # three keys, while argpartition avoids sorting every population row.
        random_keys = self.rng.random((count, self.population_size))
        random_keys[np.arange(count), parents] = np.inf
        sampled = np.argpartition(random_keys, kth=2, axis=1)[:, :3]
        r1, r2, r3 = sampled.T.astype(int, copy=False)
        strategies = self.rng.choice(
            np.array([1, 2, 3, 4], dtype=int),
            size=count,
            p=self.de_strategy_probabilities,
        )
        return parents, r1, r2, r3, strategies

    def _reserve_candidate_audit_space(self, incoming_count: int) -> None:
        """Drop old audit rows while preserving every row in the incoming batch."""
        incoming_count = max(0, int(incoming_count))
        retention = max(self.max_candidate_audit_records, incoming_count)
        overflow = len(self.candidate_archive) + incoming_count - retention
        if overflow > 0:
            del self.candidate_archive[:overflow]
            self.candidate_archive_dropped_count += overflow

    def _generate_candidate_pool(
        self,
        population: Array,
        population_values: Array,
        f_value: float,
        cr_value: float,
        state: Array,
        strategy_override: Optional[int] = None,
        *,
        count: Optional[int] = None,
        record: bool = True,
        probe_mode: bool = False,
    ) -> Dict[str, Any]:
        if strategy_override is not None and int(strategy_override) not in self.de_strategy_names:
            raise ValueError("strategy_override must identify one of the configured DE strategies")
        if count is None:
            count = max(self.population_size, self.population_size * self.candidate_pool_multiplier)
        count = max(1, int(count))
        parents, r1, r2, r3, strategies = self._draw_candidate_indices(count)
        if float(state[1]) >= 0.40:
            n_explore = max(1, int(np.ceil(0.30 * count)))
            strategies[-n_explore:] = 2
        policy_strategy_forced = np.zeros(count, dtype=bool)
        if strategy_override is not None:
            forced_count = count if probe_mode else max(1, int(np.ceil(0.35 * count)))
            strategies[:forced_count] = int(strategy_override)
            policy_strategy_forced[:forced_count] = True
        memory_indices = self.rng.integers(0, len(self.shade_m_f), size=count)
        if probe_mode:
            # Strategy-arm comparison holds F/CR fixed so parameter noise does not mask operator differences.
            f_values = np.full(count, np.clip(f_value, 0.25, 1.0), dtype=float)
            cr_values = np.full(count, np.clip(cr_value, 0.10, 0.99), dtype=float)
        else:
            shade_mask = self.rng.random(count) < 0.50
            f_values = np.clip(self.rng.normal(f_value, 0.12, size=count), 0.25, 1.0)
            cr_values = np.clip(self.rng.normal(cr_value, 0.12, size=count), 0.10, 0.99)
            if np.any(shade_mask):
                f_values[shade_mask] = np.clip(
                    self.rng.normal(self.shade_m_f[memory_indices[shade_mask]], 0.10),
                    0.25, 1.0,
                )
                cr_values[shade_mask] = np.clip(
                    self.rng.normal(self.shade_m_cr[memory_indices[shade_mask]], 0.10),
                    0.10, 0.99,
                )
            f_values[: max(1, count // 4)] = f_value
            cr_values[: max(1, count // 4)] = cr_value
        trials = np.empty((count, self.dimension), dtype=float)
        pbest_count = max(2, int(np.ceil(self.de_p_best_rate * self.population_size)))
        ranked = np.argsort(population_values)
        pbest_indices = ranked[:pbest_count]
        best = population[ranked[0]]
        # Uncertainty-calibrated population ellipsoid: a local model step, not a global vertex.
        tr_report = {"usable": False}
        trust_region_candidate = np.zeros(count, dtype=bool)
        trust_region_predicted_gain = np.zeros(count, dtype=float)
        if self.surrogate.fitted and not probe_mode:
            reference = self.elite_x if self.elite_x is not None else best
            _, center_uncertainty = self.surrogate.predict_with_uncertainty(
                np.asarray(reference, dtype=float).reshape(1, -1)
            )
            uncertainty_at_center = float(center_uncertainty[0])
            tr_report = self.surrogate.trust_region_acquisition(
                reference,
                population,
                uncertainty=uncertainty_at_center,
                radius_scale=self.trust_region_scale,
            )
            if not probe_mode:
                self.last_trust_region_report = tr_report
        parents_x = population[parents]
        # The direction archive contains replaced parents from successful
        # true evaluations. It is separate from archive_x, which stores all
        # supervised observations for the sparse surrogate.
        donor_r1 = population[r1].copy()
        donor_r2 = population[r2].copy()
        donor_r3 = population[r3].copy()
        archive_used = np.zeros(count, dtype=bool)
        if self.de_archive_x:
            direction_archive = np.asarray(self.de_archive_x, dtype=float)
            archive_rate = 0.25
            for donor in (donor_r1, donor_r2, donor_r3):
                use_archive = self.rng.random(count) < archive_rate
                if np.any(use_archive):
                    archive_indices = self.rng.integers(
                        0, len(direction_archive), size=int(np.sum(use_archive))
                    )
                    donor[use_archive] = direction_archive[archive_indices]
                    archive_used |= use_archive
        strategy_four = strategies == 4
        if not probe_mode:
            n_elite_local = max(1, count // 6)
            strategies[-n_elite_local:] = 4
            strategy_four = strategies == 4
            cr_values[strategy_four] = np.clip(np.maximum(cr_values[strategy_four], 0.85), 0.10, 0.99)
            f_values[strategy_four] = np.clip(f_values[strategy_four], 0.25, 0.70)
        pbest_targets = parents_x.copy()
        strategy_one = strategies == 1
        if np.any(strategy_one):
            pbest_targets[strategy_one] = population[
                pbest_indices[
                    self.rng.integers(
                        0,
                        pbest_count,
                        size=int(np.sum(strategy_one)),
                    )
                ]
            ]
        local_donors = parents_x.copy()
        if np.any(strategy_four):
            span = np.maximum(self._upper - self._lower, 1e-12)
            pop_std = np.maximum(np.std(population, axis=0), 1e-4 * span)
            local_center = self.elite_x if self.elite_x is not None else best
            if tr_report.get("usable") and float(tr_report.get("mahalanobis", np.inf)) <= 1.0 + 1e-8:
                local_center = np.asarray(tr_report["x"], dtype=float)
            sigma = 0.50 * f_values[strategy_four, None] * pop_std[None, :]
            local_donors[strategy_four] = local_center[None, :] + self.rng.normal(
                0.0,
                1.0,
                size=(int(np.sum(strategy_four)), self.dimension),
            ) * sigma
        crossover_mask = self.rng.random((count, self.dimension)) < cr_values[:, None]
        block_ids = np.full(count, -1, dtype=int)
        block_dimension_count = np.full(count, self.dimension, dtype=int)
        global_candidate = np.ones(count, dtype=bool)
        use_blocks = (
            self.dimension >= self.de_high_dimension_threshold
            and self.block_mutation_size is not None
            and int(self.block_mutation_size) > 1
        )
        if use_blocks:
            global_count = max(
                1,
                int(np.ceil(self.de_global_candidate_fraction * count)),
            )
            global_candidate[global_count:] = False
            configured_block_size = int(self.block_mutation_size)
            # A value such as 32 is useful for low dimensions but must not
            # silently disable block search at exactly 30 dimensions.
            if configured_block_size >= self.dimension:
                block_size = max(
                    2,
                    min(self.dimension - 1, int(np.ceil(2.0 * np.sqrt(self.dimension)))),
                )
            else:
                block_size = min(configured_block_size, self.dimension - 1)
            block_count = int(np.ceil(self.dimension / block_size))
            block_ids = self.rng.integers(0, block_count, size=count)
            block_ids[:global_count] = -1
            block_dimension_count[:] = np.minimum(
                block_size,
                self.dimension - block_ids * block_size,
            )
            block_dimension_count[:global_count] = self.dimension
            dimensions = np.arange(self.dimension, dtype=int)[None, :]
            block_starts = block_ids[:, None] * block_size
            block_stops = np.minimum(
                self.dimension,
                block_starts + block_size,
            )
            block_mask = (dimensions >= block_starts) & (dimensions < block_stops)
            block_mask[:global_count, :] = True
            crossover_mask &= block_mask
            forced_dimensions = np.empty(count, dtype=int)
            forced_dimensions[:global_count] = self.rng.integers(
                0, self.dimension, size=global_count
            )
            local_count = count - global_count
            if local_count:
                forced_dimensions[global_count:] = (
                    block_ids[global_count:] * block_size
                    + self.rng.integers(0, block_dimension_count[global_count:])
                )
            # The global arm keeps the complete DE donor. This whole-vector
            # mutation preserves coordinated directions that block crossover
            # cannot express, especially on rotated or coupled landscapes.
            crossover_mask[:global_count] = True
        else:
            forced_dimensions = self.rng.integers(0, self.dimension, size=count)
        crossover_mask[np.arange(count), forced_dimensions] = True
        mutated_dimension_count = np.sum(crossover_mask, axis=1).astype(int)
        trials = self.compute_backend.de_trials(
            parents_x,
            donor_r1,
            donor_r2,
            donor_r3,
            pbest_targets,
            best,
            local_donors,
            f_values,
            strategies,
            crossover_mask,
            self._lower,
            self._upper,
        )
        if (not probe_mode) and tr_report.get("usable"):
            tr_x = np.clip(np.asarray(tr_report["x"], dtype=float), self._lower, self._upper)
            too_close = False
            if self.archive_x:
                span = np.maximum(self._upper - self._lower, 1e-12)
                block_size = max(1, 1_000_000 // max(self.dimension, 1))
                nearest = self._nearest_archive_distance(
                    tr_x,
                    max_block_size=block_size,
                    scale=span,
                )
                too_close = nearest < 1e-4
            mahalanobis = float(tr_report.get("mahalanobis", np.inf))
            inside_ellipsoid = np.isfinite(mahalanobis) and mahalanobis <= 1.0 + 1e-8
            if inside_ellipsoid and (not too_close):
                trials[-1] = tr_x
                trust_region_candidate[-1] = True
                trust_region_predicted_gain[-1] = float(
                    tr_report.get("predicted_reduction", 0.0)
                )
                strategies[-1] = 4
                parents[-1] = int(ranked[0])
                cr_values[-1] = 0.99
                f_values[-1] = float(np.clip(f_values[-1], 0.25, 0.55))

        objective_proxy_used = bool(self.surrogate.fitted and not probe_mode)
        if objective_proxy_used:
            formula_predicted, predicted, uncertainty, distance = (
                self.surrogate.predict_with_formula_details(trials)
            )
            if self.archive_y:
                y_best = float(np.min(self.archive_y))
                y_ref = float(np.percentile(self.archive_y, 80))
                margin = max(1e-8, 0.25 * abs(y_ref - y_best), 0.05 * abs(y_best))
                predicted = np.clip(predicted, y_best - margin, y_ref + abs(y_ref - y_best) + margin)
        elif probe_mode:
            formula_predicted = population_values[parents].copy()
            predicted = population_values[parents].copy()
            uncertainty = np.zeros(count, dtype=float)
            distance = np.zeros(count, dtype=float)
        else:
            formula_predicted = population_values[parents].copy()
            predicted = population_values[parents].copy()
            uncertainty = np.full(count, np.inf)
            distance = np.full(count, np.inf)

        improvement_features = self._improvement_features(
            population, trials, parents, f_values, cr_values, strategies,
            state, population_values,
        )
        if self.improvement_model.fitted:
            gain_samples = self.improvement_model.predict_formula_samples(
                improvement_features
            )
            raw_predicted_gain = np.mean(gain_samples, axis=1)
            gain_uncertainty = np.std(gain_samples, axis=1)
            improvement_uncertainty_mode = "ensemble_only"
            raw_predicted_gain = np.asarray(raw_predicted_gain, dtype=float)
            gain_uncertainty = np.asarray(gain_uncertainty, dtype=float)
        else:
            raw_predicted_gain = np.zeros(count, dtype=float)
            gain_uncertainty = np.zeros(count, dtype=float)
            improvement_uncertainty_mode = "model_not_fitted"
        predicted_gain, gain_calibration = self._calibrate_improvement_predictions(
            raw_predicted_gain
        )
        parent_predicted_gain = np.asarray(
            population_values[parents] - formula_predicted,
            dtype=float,
        )

        objective_scale = max(
            float(np.std(self.archive_y)) if len(self.archive_y) > 1 else 0.0,
            self.surrogate.target_scale if self.surrogate.fitted else 1.0,
            1e-8,
        )
        gain_signal = np.clip(
            np.nan_to_num(
                predicted_gain,
                nan=0.0,
                posinf=2.0 * objective_scale,
                neginf=-2.0 * objective_scale,
            ),
            -2.0 * objective_scale,
            2.0 * objective_scale,
        )
        model_weight = self.uncertainty_weight if self.surrogate.quality_score >= self.min_model_quality else 0.05
        improvement_reliability = self._improvement_model_reliability()
        gain_weight = (
            0.25 * model_weight
            if improvement_reliability["reliable"]
            else 0.0
        )
        acquisition = (
            predicted
            - model_weight * np.nan_to_num(uncertainty, nan=objective_scale, posinf=objective_scale * 10.0)
            - gain_weight * gain_signal
        )
        if not self.surrogate.fitted:
            acquisition = self.rng.random(count)

        pool = {
            "x": trials,
            "parents": parents,
            "F": f_values,
            "CR": cr_values,
            "strategy": strategies,
            "predicted": predicted,
            "formula_predicted": formula_predicted,
            "uncertainty": uncertainty,
            "distance": distance,
            "raw_predicted_gain": raw_predicted_gain,
            "predicted_gain": predicted_gain,
            "parent_predicted_gain": parent_predicted_gain,
            "gain_uncertainty": gain_uncertainty,
            "gain_signal": gain_signal,
            "gain_calibration": gain_calibration,
            "improvement_uncertainty_mode": improvement_uncertainty_mode,
            "improvement_reliability": improvement_reliability,
            "acquisition": acquisition,
            "improvement_features": improvement_features,
            "memory_index": memory_indices,
            "block_id": block_ids,
            "block_dimension_count": block_dimension_count,
            "mutated_dimension_count": mutated_dimension_count,
            "archive_used": archive_used,
            "strategy_probabilities": self.de_strategy_probabilities[strategies - 1],
            "policy_strategy": None if strategy_override is None else int(strategy_override),
            "policy_strategy_forced": policy_strategy_forced,
            "trust_region_candidate": trust_region_candidate,
            "trust_region_predicted_gain": trust_region_predicted_gain,
            "global_candidate": global_candidate,
            "objective_proxy_used": objective_proxy_used,
        }
        ledger_indices = np.full(count, -1, dtype=int)
        if record:
            self._reserve_candidate_audit_space(count)
            generation = len(self.history)
            for i in range(count):
                ledger_indices[i] = len(self.candidate_archive)
                self.candidate_archive.append({
                    "generation": int(generation),
                    "candidate_index": int(i),
                    "parent_index": int(parents[i]),
                    "strategy": int(strategies[i]),
                    "F": float(f_values[i]),
                    "CR": float(cr_values[i]),
                    "memory_index": int(memory_indices[i]),
                    "block_id": int(block_ids[i]),
                    "block_dimension_count": int(block_dimension_count[i]),
                    "mutated_dimension_count": int(mutated_dimension_count[i]),
                    "archive_used": bool(archive_used[i]),
                    "policy_strategy": None if strategy_override is None else int(strategy_override),
                    "policy_strategy_forced": bool(policy_strategy_forced[i]),
                    "strategy_probability": float(self.de_strategy_probabilities[int(strategies[i]) - 1]),
                    "predicted": float(predicted[i]),
                    "formula_predicted": float(formula_predicted[i]),
                    "uncertainty": float(uncertainty[i]) if np.isfinite(uncertainty[i]) else None,
                    "distance": float(distance[i]) if np.isfinite(distance[i]) else None,
                    "raw_predicted_gain": float(raw_predicted_gain[i]),
                    "predicted_gain": float(predicted_gain[i]),
                    "parent_predicted_gain": float(parent_predicted_gain[i]),
                    "acquisition": float(acquisition[i]),
                    "surrogate_ready": bool(self.surrogate.fitted),
                    "proxy_evaluated": bool(self.surrogate.fitted),
                    "true_evaluated": False,
                    "true_value": None,
                    "constraint_violation": None,
                    "trust_region_candidate": bool(trust_region_candidate[i]),
                    "trust_region_predicted_gain": float(trust_region_predicted_gain[i]),
                    "global_candidate": bool(global_candidate[i]),
                })
            self.candidate_archive_total_count += count
        pool["ledger_indices"] = ledger_indices
        if record:
            self._last_pool = pool
        return pool

    def _mark_candidate_evaluated(
        self,
        pool: Dict[str, Any],
        candidate_index: int,
        value: float,
        violation: float,
        improvement: float,
        final_confirmation: bool = False,
    ) -> None:
        """Write the true label back to the candidate ledger for pointwise comparison."""
        ledger_index = int(pool["ledger_indices"][int(candidate_index)])
        record = self.candidate_archive[ledger_index]
        record.update({
            # High-dimensional virtual trials are not duplicated in the audit
            # ledger. The complete vector is retained only once the expensive
            # objective has actually been queried.
            "x": np.asarray(pool["x"][int(candidate_index)], dtype=float).tolist(),
            "true_evaluated": True,
            "true_value": float(value),
            "constraint_violation": float(violation),
            "actual_improvement": float(improvement),
            "final_confirmation": bool(final_confirmation),
        })


    def _select_real_indices(self, pool: Dict[str, Any], remaining: int) -> Tuple[Array, Dict[str, float]]:
        """Select true evaluations with hard risk-coverage safeguards."""
        count = min(self.real_batch_size, int(remaining), len(pool["x"]))
        if count <= 0:
            return np.empty(0, dtype=int), {"trust_outside": 0.0, "trust_fraction": 1.0}

        predicted = np.asarray(
            pool.get("predicted", pool.get("acquisition")), dtype=float
        )
        ranking_predicted = np.asarray(
            pool.get("formula_predicted", predicted), dtype=float
        )
        if not np.any(np.isfinite(ranking_predicted)):
            ranking_predicted = predicted
        acquisition = np.asarray(pool.get("acquisition", predicted), dtype=float)
        uncertainty = np.nan_to_num(
            np.asarray(pool.get("uncertainty", np.zeros(len(pool["x"]))), dtype=float),
            nan=0.0, posinf=1e12,
        )
        distance = np.nan_to_num(
            np.asarray(pool.get("distance", np.zeros(len(pool["x"]))), dtype=float),
            nan=1e12, posinf=1e12,
        )
        gain = np.nan_to_num(
            np.asarray(pool.get("predicted_gain", np.zeros(len(pool["x"]))), dtype=float),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        parent_gain = np.nan_to_num(
            np.asarray(pool.get("parent_predicted_gain", gain), dtype=float),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        global_mask = np.asarray(
            pool.get("global_candidate", np.zeros(len(pool["x"]), dtype=bool)),
            dtype=bool,
        )
        finite_predicted = np.isfinite(ranking_predicted)
        predicted_order = self.compute_backend.argsort(
            np.where(finite_predicted, ranking_predicted, np.inf)
        )
        if not np.any(finite_predicted):
            predicted_order = self.compute_backend.argsort(acquisition)

        trust_scale = max(
            self.surrogate.target_scale if self.surrogate.fitted else 1.0,
            float(np.std(self.archive_y)) if self.archive_y else 1.0,
            1e-8,
        )
        normalized_uncertainty = uncertainty / trust_scale
        in_trust = (normalized_uncertainty <= self.trust_threshold) & (distance <= 0.35)

        # Untrusted points are removed from proxy exploitation. They remain
        # eligible for explicit confirmation or risk exploration.
        proxy_exploitation_order = self.compute_backend.argsort(
            np.where(in_trust, acquisition, np.inf)
        )
        gain_order = self.compute_backend.argsort(gain, descending=True)
        global_order = self.compute_backend.argsort(parent_gain, descending=True)
        risk_order = self.compute_backend.argsort(
            uncertainty + distance, descending=True
        )
        selected: List[int] = []

        def add(index: int) -> None:
            if len(selected) < count and int(index) not in selected:
                selected.append(int(index))

        # Hard coverage rules: always confirm the predicted best, and reserve
        # a true evaluation for the highest-risk candidate when possible.
        predicted_best = int(predicted_order[0])
        competitive_count = max(
            2,
            int(np.ceil(self.uncertainty_candidate_fraction * len(pool["x"]))),
        )
        competitive_indices = np.asarray(
            predicted_order[:competitive_count], dtype=int
        )
        high_uncertainty = int(
            competitive_indices[np.argmax(uncertainty[competitive_indices])]
        )
        high_risk = int(risk_order[0])
        generation = int(len(self.history))
        calibration_trigger = bool(
            self.surrogate.fitted
            and self.surrogate.uncertainty_calibration >= 1.50
        )
        quality_trigger = bool(
            (not self.surrogate.fitted)
            or self.surrogate.quality_score < self.min_model_quality
        )
        uncertainty_audit_due = bool(
            generation % self.uncertainty_evaluation_interval == 0
            or calibration_trigger
            or quality_trigger
        )
        add(predicted_best)
        # Trust-region candidates compete by predicted value and acquisition;
        # they are not forced into the true-evaluation batch.
        forced_predicted_best = predicted_best in selected
        if count >= 2 and uncertainty_audit_due:
            # This slot is reserved for the candidate with the largest
            # calibrated uncertainty when the periodic or calibration trigger fires.
            add(high_uncertainty)
        forced_high_uncertainty = bool(
            uncertainty_audit_due and high_uncertainty in selected
        )

        # Preserve one full-dimensional DE trial in the real-evaluation batch.
        # Block mutation improves scalability, but without this safeguard it can
        # discard coordinated moves that are essential on rotated landscapes.
        global_candidates = [int(index) for index in global_order if global_mask[int(index)]]
        global_champion = global_candidates[0] if global_candidates else None
        global_runner_up = global_candidates[1] if len(global_candidates) >= 2 else None
        forced_global_champion = False
        if count >= 2 and global_champion is not None:
            add(global_champion)
            forced_global_champion = global_champion in selected

        if count >= 3 and (not uncertainty_audit_due) and global_runner_up is not None:
            add(global_runner_up)

        if count >= 3:
            add(int(gain_order[0]))

        # Only trusted points can consume exploitation slots.
        exploitation_quota = max(0, count - len(selected))
        for index in proxy_exploitation_order:
            if exploitation_quota <= 0:
                break
            index = int(index)
            if in_trust[index] and np.isfinite(acquisition[index]):
                before = len(selected)
                add(index)
                exploitation_quota -= max(0, len(selected) - before)

        for index in gain_order:
            if len(selected) >= count:
                break
            add(int(index))
        for index in risk_order:
            if len(selected) >= count:
                break
            add(int(index))

        # Max-min diversity fill keeps the batch from collapsing onto one point.
        while len(selected) < count:
            candidates = [index for index in range(len(pool["x"])) if index not in selected]
            if not candidates:
                break
            if not selected:
                add(candidates[0])
                continue
            selected_points = pool["x"][np.asarray(selected)]
            separations = [
                float(np.min(np.linalg.norm(
                    (pool["x"][index] - selected_points) / (self._upper - self._lower),
                    axis=1,
                )))
                for index in candidates
            ]
            add(candidates[int(np.argmax(separations))])

        selected_array = np.asarray(selected[:count], dtype=int)
        in_trust_indices = [int(index) for index in np.flatnonzero(in_trust)]
        proxy_indices = [
            int(index) for index in proxy_exploitation_order
            if in_trust[int(index)] and np.isfinite(acquisition[int(index)])
        ]
        selected_records = []
        for index in selected_array:
            index = int(index)
            selected_records.append({
                "candidate_index": index,
                "acquisition": float(acquisition[index]) if np.isfinite(acquisition[index]) else None,
                "predicted": float(predicted[index]) if np.isfinite(predicted[index]) else None,
                "predicted_gain": float(gain[index]),
                "uncertainty": float(uncertainty[index]),
                "distance": float(distance[index]),
                "trustworthy": bool(in_trust[index]),
            })
        self.real_selection_audit_records.append({
            "generation": generation,
            "requested_count": int(count),
            "quota_rule": {
                "mandatory_predicted_best": int(predicted_best in selected),
                "mandatory_high_uncertainty": int(forced_high_uncertainty),
                "uncertainty_audit_due": int(uncertainty_audit_due),
                "predicted_gain": int(min(1, count >= 3)),
                "proxy_exploitation": int(max(
                    0,
                    len(selected) - int(predicted_best in selected) - int(high_risk in selected),
                )),
            },
            "selected_indices": [int(index) for index in selected_array],
            "selected_candidates": selected_records,
            "predicted_best_index": predicted_best,
            "high_uncertainty_index": high_uncertainty,
            "uncertainty_competitive_indices": [
                int(index) for index in competitive_indices
            ],
            "high_risk_index": high_risk,
            "global_champion_index": global_champion,
            "global_runner_up_index": global_runner_up,
            "forced_predicted_best": bool(forced_predicted_best),
            "forced_high_uncertainty": bool(forced_high_uncertainty),
            "forced_global_champion": bool(forced_global_champion),
            "uncertainty_audit_due": bool(uncertainty_audit_due),
            "uncertainty_evaluation_interval": int(self.uncertainty_evaluation_interval),
            "uncertainty_calibration_triggered": calibration_trigger,
            "surrogate_quality_triggered": quality_trigger,
            "proxy_exploitation_indices": proxy_indices,
            "in_trust_indices": in_trust_indices,
            "trust_scale": float(trust_scale),
            "trust_threshold": float(self.trust_threshold),
            "uncertainty_candidate_fraction": float(self.uncertainty_candidate_fraction),
            "trust_rule": "proxy exploitation requires normalized uncertainty <= trust_threshold and normalized distance <= 0.35; uncertainty exploration is restricted to the best half by formula prediction",
            "proxy_gate": "untrusted candidates enter only by mandatory confirmation, risk exploration, gain ranking, or diversity fill",
            "trust_outside": float(np.sum(~in_trust)),
            "trust_fraction": float(np.mean(in_trust)),
        })
        return selected_array, {
            "trust_outside": float(np.sum(~in_trust)),
            "trust_fraction": float(np.mean(in_trust)),
            "forced_predicted_best": float(forced_predicted_best),
            "forced_high_uncertainty": float(forced_high_uncertainty),
            "forced_global_champion": float(forced_global_champion),
            "uncertainty_audit_due": float(uncertainty_audit_due),
        }

    def _record_improvement(
        self,
        feature: Array,
        improvement: float,
    ) -> None:
        self.improvement_archive_x.append(np.asarray(feature, dtype=float).copy())
        self.improvement_archive_y.append(float(improvement))

    def _current_surrogate_refit_interval(self) -> int:
        """Use frequent warm-up updates, then the dimension-aware interval."""
        if self.dimension >= 100:
            return int(self.effective_surrogate_refit_interval)
        warmup_evaluations = max(500, 2 * self.surrogate_start)
        if len(self.archive_y) <= warmup_evaluations:
            return int(self.surrogate_refit_interval)
        return int(self.effective_surrogate_refit_interval)

    def _screen_interactions(self, x: Array, y: Array) -> None:
        """Select a few interaction terms from main-effect residuals while keeping the dictionary hierarchical."""
        # Restore all main effects before screening so a previous active set cannot
        # permanently drop a variable. Screening constrains only nonlinear expansions.
        self.library.set_active_features(list(range(self.dimension)), [])
        terms_before_screening = list(self.library.names)
        base_model = EnsembleSparseRegressor(
            library=self.library,
            n_models=2,
            threshold=0.04,
            max_terms=min(self.max_surrogate_terms, 24),
            random_state=17 + self.surrogate_fit_count,
            compute_backend=self.compute_backend,
        )
        base_model.fit(x, y)
        base_prediction, _ = base_model.predict_with_uncertainty(x, include_distance=False)
        target = np.asarray(y, dtype=float)
        residual = base_prediction - target
        normalized = self.library.normalize(x)
        centered_residual = residual - float(np.mean(residual))
        centered_target = target - float(np.mean(target))
        residual_score = np.maximum(
            np.abs(self.compute_backend.matmul(
                normalized.T,
                centered_residual,
                operation="interaction_screening",
            )),
            np.abs(self.compute_backend.matmul(
                (normalized ** 2).T,
                centered_residual,
                operation="interaction_screening",
            )),
        )
        direct_score = np.maximum(
            np.abs(self.compute_backend.matmul(
                normalized.T,
                centered_target,
                operation="interaction_screening",
            )),
            np.abs(self.compute_backend.matmul(
                (normalized ** 2).T,
                centered_target,
                operation="interaction_screening",
            )),
        )
        # Combine main-effect coefficients and residual correlation so a single noisy
        # screening criterion is less likely to drop a relevant variable.
        main_score = np.zeros(self.dimension, dtype=float)
        for coefficient, spec in zip(base_model.mean_coefficients(), self.library._specs):
            if spec[0] in {"linear", "square"}:
                main_score[int(spec[1])] += abs(float(coefficient))
        variable_score = main_score + residual_score + 0.25 * direct_score
        active_count = min(self.max_active_variables, self.dimension)
        active = np.argsort(-variable_score)[:active_count]
        self.active_dimensions = sorted(int(index) for index in active)
        pair_scores: List[Tuple[float, Tuple[int, int]]] = []
        pairs: List[Tuple[int, int]] = []
        if len(active) >= 2 and self.max_interaction_terms > 0:
            # Do not materialize every pair dictionary column at once.  With
            # 2,000 active variables this is 1,999,000 columns and can exceed
            # 6 GiB before the regression even starts.  Stream bounded chunks
            # and retain only the top interaction scores needed downstream.
            batch_size = max(
                1024,
                min(8192, 1_000_000 // max(int(normalized.shape[0]), 1)),
            )
            candidate_pairs = combinations(active.tolist(), 2)
            batch: List[Tuple[int, int]] = []

            def score_pair_batch(batch_pairs: List[Tuple[int, int]]) -> None:
                if not batch_pairs:
                    return
                left = np.asarray([item[0] for item in batch_pairs], dtype=int)
                right = np.asarray([item[1] for item in batch_pairs], dtype=int)
                pair_dictionary = normalized[:, left] * normalized[:, right]
                scores = np.abs(self.compute_backend.matmul(
                    pair_dictionary.T,
                    residual,
                    operation="interaction_screening",
                ))
                pair_scores.extend(
                    (float(score), (int(pair[0]), int(pair[1])))
                    for score, pair in zip(scores, batch_pairs)
                )
                pair_scores.sort(key=lambda item: item[0], reverse=True)
                del pair_scores[self.max_interaction_terms:]

            for pair in candidate_pairs:
                batch.append((int(pair[0]), int(pair[1])))
                if len(batch) >= batch_size:
                    score_pair_batch(batch)
                    batch = []
            score_pair_batch(batch)
            pair_scores.sort(key=lambda item: item[0], reverse=True)
            pairs = [pair for score, pair in pair_scores[: self.max_interaction_terms] if score > 1e-8]
        # Write the screening result into the dictionary. Updating active_dimensions alone
        # would still allow transform() to build the full high-dimensional candidate set.
        adjacent = [(index, index + 1) for index in range(self.dimension - 1)]
        merged = []
        seen = set()
        for pair in list(pairs) + adjacent:
            key = (int(pair[0]), int(pair[1]))
            if key[0] == key[1] or key in seen:
                continue
            seen.add(key)
            merged.append(key)
        pairs = merged
        self.library.include_interactions = True
        self.library.set_active_features(self.active_dimensions, pairs)
        self.library.transform(x)
        self.screening_audit_records.append({
            "fit_index": int(self.screening_fit_count + 1),
            "base_dictionary_terms": terms_before_screening,
            "base_model_audit": base_model.selection_audit(),
            "variable_score_formula": "main_effect_coefficient_abs + max(abs(z dot residual), abs(z^2 dot residual)) + 0.25*max(abs(z dot target), abs(z^2 dot target))",
            "variable_scores": [float(value) for value in variable_score],
            "active_variable_limit": int(active_count),
            "selected_active_variables": [int(index) for index in self.active_dimensions],
            "pair_score_formula": "abs((z_i*z_j) dot residual)",
            "pair_screening_batch_size": (
                int(batch_size) if len(active) >= 2 and self.max_interaction_terms > 0 else 0
            ),
            "pair_score_retention": "top max_interaction_terms only",
            "pair_scores": [
                {"pair": [int(left), int(right)], "score": float(score)}
                for score, (left, right) in pair_scores
            ],
            "selected_interaction_pairs": [[int(left), int(right)] for left, right in pairs],
            "dictionary_terms_after_screening": list(self.library.names),
        })
        self.screening_fit_count += 1
        if len(self.screening_audit_records) > self.max_screening_audit_records:
            self.screening_audit_records = self.screening_audit_records[
                -self.max_screening_audit_records:
            ]

    def _fit_models(self, force: bool = False) -> None:
        if len(self.archive_y) < self.surrogate_start and not force:
            return
        if len(self.archive_y) < 3:
            return
        current_refit_interval = self._current_surrogate_refit_interval()
        if (
            not force
            and self.history
            and len(self.history) % current_refit_interval != 0
        ):
            return
        x, y, training_indices = self._surrogate_training_data()
        # The bounded training view keeps high-dimensional refits predictable.
        # Dictionary structure is refreshed less often because repeating variable
        # and interaction discovery on nearly identical archives is expensive and
        # adds structural jitter to the explanation.
        refresh_structure = (
            self.screening_fit_count == 0
            or (
                self.dimension >= self.de_high_dimension_threshold
                and self.dimension < 100
                and len(self.archive_y) <= 500
            )
            or self.surrogate_fit_count % self.interaction_screening_interval == 0
        )
        if self.hierarchical_additive_backbone:
            self.surrogate.max_terms = max(
                self.max_surrogate_terms,
                self.dimension + self.max_interaction_terms,
            )
        else:
            self.surrogate.max_terms = max(
                self.max_surrogate_terms,
                min(64, self.dimension + 16),
            )
        if refresh_structure:
            self.library.include_sine = False
            self.library.include_cosine = False
            self.library._specs = []
            self._screen_interactions(x, y)
            self.surrogate.fit(x, y)
            polynomial_quality = float(self.surrogate.quality_score)
            if polynomial_quality < 0.55:
                self.library.include_sine = True
                self.library.include_cosine = True
                self.library._specs = []
                self.library.set_active_features(
                    self.active_dimensions, self.library.interaction_pairs
                )
                self.surrogate.fit(x, y)
                self.periodic_terms_admitted = True
            else:
                self.periodic_terms_admitted = False
        else:
            self.surrogate.fit(x, y)
        self.surrogate_fit_count += 1
        if self.constraint is not None and self.archive_violation:
            violation = np.asarray(self.archive_violation, dtype=float)[training_indices]
            if np.any(violation > 0.0):
                self.constraint_model.fit(x, violation)
        if len(self.improvement_archive_y) >= 6:
            self.improvement_model.fit(
                np.asarray(self.improvement_archive_x, dtype=float),
                np.asarray(self.improvement_archive_y, dtype=float),
            )
            self.auxiliary_fit_count += 1
        if len(self.policy_state_archive) >= 6:
            self.policy_model.fit(
                np.asarray(self.policy_state_archive, dtype=float),
                np.asarray(self.policy_f_archive, dtype=float),
                np.asarray(self.policy_cr_archive, dtype=float),
            )
            self.auxiliary_fit_count += 1

    def _final_candidate(
        self,
        population: Array,
        population_values: Array,
        state: Array,
    ) -> Tuple[Array, float, float, int, Dict[str, Any]]:
        # The last budget slot is a true confirmation, so the candidate is
        # explicitly selected and then evaluated by the real objective.
        f_value, cr_value = self._shade_baseline_policy()
        strategy_override = self.last_policy.get("de_strategy_selected")
        if strategy_override is not None:
            strategy_override = int(strategy_override)
            if strategy_override not in self.de_strategy_names:
                strategy_override = None
        pool = self._generate_candidate_pool(
            population,
            population_values,
            f_value,
            cr_value,
            state,
            strategy_override=strategy_override,
        )
        predicted = np.asarray(pool["predicted"], dtype=float)
        ranking = np.asarray(pool.get("formula_predicted", predicted), dtype=float)
        uncertainty = np.nan_to_num(
            np.asarray(pool["uncertainty"], dtype=float), nan=np.inf, posinf=np.inf
        )
        if self.surrogate.fitted and np.any(np.isfinite(ranking)):
            index = int(np.nanargmin(ranking))
        elif self.surrogate.fitted and np.any(np.isfinite(predicted)):
            index = int(np.nanargmin(predicted))
        else:
            index = int(np.argmin(pool["acquisition"]))
        return (
            pool["x"][index].copy(),
            float(predicted[index]),
            float(uncertainty[index]),
            index,
            pool,
        )


    def _maybe_restart_stagnated_search(
        self,
        population: Array,
        population_values: Array,
        population_violations: Array,
        loop_limit: int,
    ) -> Dict[str, float]:
        """Re-sample a subset of coordinates of the worst individuals after stall."""
        if self._stagnation < 12 or self._stagnation % 12 != 0:
            return {"restarted": 0.0, "restart_evaluations": 0.0}
        remaining = loop_limit - len(self.archive_y)
        if remaining <= 1:
            return {"restarted": 0.0, "restart_evaluations": 0.0}
        n_restart = min(remaining - 1, max(1, self.population_size // 8))
        worst = np.argsort(population_values)[-n_restart:]
        self._reserve_candidate_audit_space(len(worst))
        n_coords = max(1, int(np.ceil(0.15 * self.dimension)))
        span = np.maximum(self._upper - self._lower, 1e-12)
        pop_std = np.maximum(np.std(population, axis=0), 1e-4 * span)
        center = self.elite_x if self.elite_x is not None else population[int(np.argmin(population_values))]
        tr_report = self.last_trust_region_report or {}
        if tr_report.get("usable") and float(tr_report.get("mahalanobis", np.inf)) <= 1.0 + 1e-8:
            center = np.clip(np.asarray(tr_report["x"], dtype=float), self._lower, self._upper)
        accepted = 0
        evaluations = 0
        for index in worst:
            if len(self.archive_y) >= loop_limit:
                break
            trial = np.asarray(center, dtype=float).copy()
            coords = np.asarray(self.rng.choice(self.dimension, size=n_coords, replace=False), dtype=int)
            trial[coords] = np.clip(
                trial[coords] + self.rng.normal(0.0, 1.0, size=coords.size) * (1.5 * pop_std[coords]),
                self._lower[coords],
                self._upper[coords],
            )
            value, violation = self._evaluate(trial)
            evaluations += 1
            self.candidate_archive.append({
                "generation": int(len(self.history)),
                "candidate_index": -1,
                "parent_index": int(index),
                "strategy": 4,
                "F": 0.35,
                "CR": 0.99,
                "true_evaluated": True,
                "true_value": float(value),
                "constraint_violation": float(violation),
                "stagnation_restart": True,
                "trust_region_candidate": False,
                "surrogate_ready": bool(self.surrogate.fitted),
                "proxy_evaluated": False,
                "x": trial.tolist(),
            })
            self.candidate_archive_total_count += 1
            if self._is_better(
                value,
                violation,
                population_values[int(index)],
                population_violations[int(index)],
            ):
                population[int(index)] = trial
                population_values[int(index)] = value
                population_violations[int(index)] = violation
                accepted += 1
        boosted = np.array([0.15, 0.45, 0.15, 0.25], dtype=float)
        mixed = 0.70 * self.de_strategy_probabilities + 0.30 * boosted
        self.de_strategy_probabilities = self._stabilize_de_strategy_probabilities(mixed)
        return {"restarted": float(accepted), "restart_evaluations": float(evaluations)}

    def minimize(self) -> OptimizationResult:
        reserve = 1 if self.reserve_final_evaluation and self.max_evaluations > self.population_size else 0
        loop_limit = self.max_evaluations - reserve
        population = self._sample_population()
        initial = self._evaluate_batch(population)
        population_values = np.asarray([item[0] for item in initial], dtype=float)
        population_violations = np.asarray([item[1] for item in initial], dtype=float)

        # Complete the initial design before surrogate-assisted candidate screening.
        # Extra design points enter the labelled archive; only competitive ones replace population members.
        initial_target = min(loop_limit, max(self.population_size, self.surrogate_start))
        extra_count = max(0, initial_target - self.population_size)
        if extra_count:
            extra_design = latin_hypercube(extra_count, self.dimension, self.rng)
            extra_points = self.compute_backend.scale_unit_points(
                extra_design, self._lower, self._upper
            )
            extra_results = self._evaluate_batch(extra_points)
            for point, (value, violation) in zip(extra_points, extra_results):
                worst_index = self._worst_population_index(
                    population_values, population_violations
                )
                if self._is_better(
                    value,
                    violation,
                    population_values[worst_index],
                    population_violations[worst_index],
                ):
                    population[worst_index] = point
                    population_values[worst_index] = value
                    population_violations[worst_index] = violation
        self.initial_design_size = len(self.archive_y)
        best_index = self._best_population_index(population_values, population_violations)
        self._best_seen = float(population_values[best_index])
        self.elite_x = population[best_index].copy()
        self.elite_value = float(population_values[best_index])
        self.elite_violation = float(population_violations[best_index])
        self._fit_models(force=False)

        while len(self.archive_y) < loop_limit:
            fallback_f, fallback_cr, state = self._policy(population)
            f_value, cr_value, policy_meta = self._select_policy_by_improvement(
                population, population_values, state, fallback_f, fallback_cr
            )
            self.last_policy.update(policy_meta)
            self.last_policy["F"] = f_value
            self.last_policy["CR"] = cr_value
            strategy_override = int(policy_meta.get("de_strategy_selected", 0)) or None
            pool = self._generate_candidate_pool(
                population,
                population_values,
                f_value,
                cr_value,
                state,
                strategy_override=strategy_override,
            )
            remaining = loop_limit - len(self.archive_y)
            selected, trust_info = self._select_real_indices(pool, remaining)
            improved = 0
            prediction_errors: List[float] = []
            successful_f: List[float] = []
            successful_cr: List[float] = []
            successful_improvements: List[float] = []
            trust_feedback = None
            selected_strategy_counts = np.zeros(4, dtype=float)
            successful_strategy_counts = np.zeros(4, dtype=float)
            successful_strategy_gains = np.zeros(4, dtype=float)
            selected_points = np.asarray(
                [pool["x"][candidate_index] for candidate_index in selected],
                dtype=float,
            )
            selected_results = (
                self._evaluate_batch(selected_points) if len(selected) else []
            )
            for candidate_index, (value, violation) in zip(selected, selected_results):
                parent_index = int(pool["parents"][candidate_index])
                parent_value = float(population_values[parent_index])
                strategy_index = int(pool["strategy"][candidate_index]) - 1
                selected_strategy_counts[strategy_index] += 1.0
                if self.surrogate.fitted and np.isfinite(pool["predicted"][candidate_index]):
                    prediction_error = abs(float(pool["predicted"][candidate_index]) - value)
                    prediction_errors.append(prediction_error)
                    self.surrogate.update_uncertainty_calibration(
                        float(pool["predicted"][candidate_index]),
                        value,
                        float(pool["uncertainty"][candidate_index]),
                    )
                improvement = parent_value - value
                if self.improvement_model.fitted:
                    predicted_improvement = float(
                        pool["predicted_gain"][candidate_index]
                    )
                    if np.isfinite(predicted_improvement) and np.isfinite(improvement):
                        self.improvement_validation_count += 1
                        self.improvement_validation_records.append({
                            "generation": float(len(self.history)),
                            "candidate_index": float(candidate_index),
                            "predicted_gain": predicted_improvement,
                            "actual_gain": float(improvement),
                        })
                        if len(self.improvement_validation_records) > 256:
                            self.improvement_validation_records = (
                                self.improvement_validation_records[-256:]
                            )
                self._record_improvement(
                    pool["improvement_features"][candidate_index], improvement
                )
                self._mark_candidate_evaluated(
                    pool,
                    int(candidate_index),
                    value,
                    violation,
                    improvement,
                    final_confirmation=False,
                )
                if bool(pool["trust_region_candidate"][candidate_index]):
                    trust_feedback = self._update_trust_region_scale(
                        float(pool["trust_region_predicted_gain"][candidate_index]),
                        float(improvement),
                    )
                parent_accepted = self._is_better(
                    value, violation, population_values[parent_index], population_violations[parent_index]
                )
                worst_index = self._worst_population_index(population_values, population_violations)
                worst_accepted = (
                    not parent_accepted
                    and worst_index != parent_index
                    and self._is_better(
                        value, violation, population_values[worst_index], population_violations[worst_index]
                    )
                )
                if parent_accepted or worst_accepted:
                    replace_index = parent_index if parent_accepted else int(worst_index)
                    replaced_member = population[replace_index].copy()
                    replaced_value = float(population_values[replace_index])
                    population[replace_index] = pool["x"][candidate_index]
                    population_values[replace_index] = value
                    population_violations[replace_index] = violation
                    improved += 1
                    population_gain = replaced_value - value
                    if population_gain > 0.0:
                        self._append_de_archive(replaced_member)
                        successful_strategy_counts[strategy_index] += 1.0
                        successful_strategy_gains[strategy_index] += float(population_gain)
                        successful_f.append(float(pool["F"][candidate_index]))
                        successful_cr.append(float(pool["CR"][candidate_index]))
                        successful_improvements.append(float(population_gain))
                        if parent_accepted:
                            # Supervise the F/CR policy only with true-evaluated, parent-beating offspring.
                            self.policy_state_archive.append(state.copy())
                            self.policy_f_archive.append(float(pool["F"][candidate_index]))
                            self.policy_cr_archive.append(float(pool["CR"][candidate_index]))
                            self.policy_feedback_archive.append({
                                "true_evaluated": True,
                                "actual_improvement": float(improvement),
                                "F": float(pool["F"][candidate_index]),
                                "CR": float(pool["CR"][candidate_index]),
                                "strategy": int(strategy_index + 1),
                                "generation": int(len(self.history)),
                                "candidate_index": int(candidate_index),
                                "parent_index": int(parent_index),
                                "state": state.copy(),
                            })
            self._update_shade_memory(
                np.asarray(successful_f, dtype=float),
                np.asarray(successful_cr, dtype=float),
                np.asarray(successful_improvements, dtype=float),
            )
            self.de_strategy_selected += selected_strategy_counts
            self.de_strategy_successful += successful_strategy_counts
            self.de_strategy_gain += successful_strategy_gains
            self._update_de_strategy_probabilities(
                selected_strategy_counts,
                successful_strategy_gains,
                generation=len(self.history),
            )
            if prediction_errors:
                self.validation_errors.extend(prediction_errors)
            current_best_index = self._best_population_index(population_values, population_violations)
            current_best = float(population_values[current_best_index])
            current_best_violation = float(population_violations[current_best_index])
            elite_preserved = False
            if self.elite_x is None or self._is_better(
                current_best, current_best_violation,
                self.elite_value, self.elite_violation,
            ):
                self.elite_x = population[current_best_index].copy()
                self.elite_value = current_best
                self.elite_violation = current_best_violation
            else:
                elite_present = any(
                    np.allclose(individual, self.elite_x)
                    for individual in population
                )
                if not elite_present:
                    worst_index = self._worst_population_index(
                        population_values, population_violations
                    )
                    population[worst_index] = self.elite_x.copy()
                    population_values[worst_index] = self.elite_value
                    population_violations[worst_index] = self.elite_violation
                    elite_preserved = True
                    current_best_index = self._best_population_index(
                        population_values, population_violations
                    )
                    current_best = float(population_values[current_best_index])
            if self._is_meaningful_improvement(current_best, self._best_seen):
                self._best_seen = current_best
                self._stagnation = 0
            else:
                self._stagnation += 1
            self._last_improvement_rate = float(improved / max(len(selected), 1))
            restart_info = self._maybe_restart_stagnated_search(
                population, population_values, population_violations, loop_limit
            )
            if prediction_errors:
                self.validation_errors.extend([])
            self._fit_models(force=False)
            row: Dict[str, Any] = {
                "generation": float(len(self.history)),
                "evaluations": float(len(self.archive_y)),
                "best": current_best,
                "F": float(f_value),
                "CR": float(cr_value),
                "diversity": float(state[0]),
                "stagnation": float(state[1]),
                "stagnation_tolerance": self.last_stagnation_diagnostics.get(
                    "tolerance"
                ),
                "stagnation_observed_iqr": self.last_stagnation_diagnostics.get(
                    "observed_iqr"
                ),
                "stagnation_surrogate_error": self.last_stagnation_diagnostics.get(
                    "surrogate_model_error_floor"
                ),
                "improvement_rate": float(state[2]),
                "uncertainty_signal": float(state[-1]),
                "real_evaluations": float(len(selected)),
                "improved_trials": float(improved),
                "candidate_pool_size": float(len(pool["x"])),
                "surrogate_candidates": float(len(pool["x"])),
                "surrogate_quality": float(self.surrogate.quality_score),
                "surrogate_cv_rmse": float(self.surrogate.cv_rmse) if np.isfinite(self.surrogate.cv_rmse) else None,
                "trust_outside": trust_info["trust_outside"],
                "trust_fraction": trust_info["trust_fraction"],
                "forced_predicted_best": trust_info.get("forced_predicted_best", 0.0),
                "forced_high_uncertainty": trust_info.get("forced_high_uncertainty", 0.0),
                "forced_global_champion": trust_info.get("forced_global_champion", 0.0),
                "uncertainty_audit_due": trust_info.get("uncertainty_audit_due", 0.0),
                "trust_region_scale": float(self.trust_region_scale),
                "trust_region_feedback_action": (
                    None if trust_feedback is None else trust_feedback.get("action")
                ),
                "elite_preserved": float(elite_preserved),
                "stagnation_restarts": restart_info.get("restarted", 0.0),
                "stagnation_restart_evaluations": restart_info.get("restart_evaluations", 0.0),
                "periodic_terms_admitted": float(self.periodic_terms_admitted),
                "shade_memory_F": float(np.mean(self.shade_m_f)),
                "shade_memory_CR": float(np.mean(self.shade_m_cr)),
                "de_archive_size": float(len(self.de_archive_x)),
                "de_archive_usage_fraction": float(np.mean(pool["archive_used"])),
                "policy_strategy": (
                    None if strategy_override is None else int(strategy_override)
                ),
                "policy_strategy_forced_fraction": float(
                    np.mean(pool["policy_strategy_forced"])
                ),
                "strategy_1_probability": float(self.de_strategy_probabilities[0]),
                "strategy_2_probability": float(self.de_strategy_probabilities[1]),
                "strategy_3_probability": float(self.de_strategy_probabilities[2]),
                "strategy_4_probability": float(self.de_strategy_probabilities[3]),
                "strategy_1_successful_gain": float(successful_strategy_gains[0]),
                "strategy_2_successful_gain": float(successful_strategy_gains[1]),
                "strategy_3_successful_gain": float(successful_strategy_gains[2]),
                "strategy_4_successful_gain": float(successful_strategy_gains[3]),
                "surrogate_uncertainty_calibration": float(self.surrogate.uncertainty_calibration),
                "surrogate_ready": float(self.surrogate.fitted),
                "improvement_model_ready": float(self.improvement_model.fitted),
                "policy_model_ready": float(self.policy_model.fitted),
                "final_confirmation": 0.0,
                **policy_meta,
            }
            if self.constraint is not None:
                row["feasible_ratio"] = float(state[3])
            self.history.append(row)

        # Reserve one true evaluation so the surrogate-selected best candidate is confirmed.
        if reserve and len(self.archive_y) < self.max_evaluations:
            state = self._state_features(population)
            point, predicted, uncertainty, candidate_index, final_pool = self._final_candidate(
                population, population_values, state
            )
            value, violation = self._evaluate(point)
            self._mark_candidate_evaluated(
                final_pool,
                candidate_index,
                value,
                violation,
                0.0,
                final_confirmation=True,
            )
            if self.elite_x is None or self._is_better(
                value, violation, self.elite_value, self.elite_violation
            ):
                self.elite_x = point.copy()
                self.elite_value = float(value)
                self.elite_violation = float(violation)
                worst_index = self._worst_population_index(population_values, population_violations)
                population[worst_index] = point.copy()
                population_values[worst_index] = float(value)
                population_violations[worst_index] = float(violation)
            self.final_confirmation = True
            self.final_confirmation_record = {
                "x": point.tolist(),
                "predicted": predicted,
                "uncertainty": uncertainty,
                "true_value": value,
                "prediction_error": abs(predicted - value) if np.isfinite(predicted) else None,
            }
            self.history.append({
                "generation": float(len(self.history)),
                "evaluations": float(len(self.archive_y)),
                "best": float(np.min(self.archive_y)),
                "real_evaluations": 1.0,
                "candidate_pool_size": 0.0,
                "surrogate_candidates": 0.0,
                "final_confirmation": 1.0,
                "final_predicted": predicted,
                "final_uncertainty": uncertainty,
                "final_true_value": value,
            })
            # The confirmed point is added to the labelled archive before the final refit.
            self._fit_models(force=True)
        else:
            self.final_confirmation = bool(self.archive_y)

        best_archive_index = self._best_archive_index()
        return OptimizationResult(
            x=self.archive_x[best_archive_index].copy(),
            fun=float(self.archive_y[best_archive_index]),
            evaluations=len(self.archive_y),
            history=list(self.history),
        )

    @staticmethod
    def _format_formula_number(value: float) -> str:
        """Format a formula coefficient without trailing noise digits."""
        return f"{float(value):.6g}"

    def _formula_text(self, model: EnsembleSparseRegressor) -> str:
        """Render a readable formula consistent with the target transform."""
        if not model.fitted:
            return ""
        argument = "z(x)" if model.library.bounds is not None else "x"
        coefficients = model.mean_coefficients()
        frequencies = model.selection_frequency()
        selected = [
            index for index, (coefficient, frequency) in enumerate(zip(coefficients, frequencies))
            if abs(float(coefficient)) > 1e-10 and float(frequency) >= 0.5
        ]
        if not selected:
            selected = [
                index for index, coefficient in enumerate(coefficients)
                if abs(float(coefficient)) > 1e-10
            ]
        if 0 not in selected:
            selected.insert(0, 0)
        selected = sorted(set(selected))
        body = self._format_formula_number(coefficients[0])
        for index in selected:
            if index == 0:
                continue
            coefficient = float(coefficients[index])
            sign = "+" if coefficient >= 0.0 else "-"
            body += f" {sign} {self._format_formula_number(abs(coefficient))}*{model.library.names[index]}"
        if model.target_transform_name == "asinh":
            return (
                f"f_hat({argument}) = {self._format_formula_number(model.target_center)} + "
                f"{self._format_formula_number(model.target_scale)}*sinh(clip({body}, "
                f"-{self._format_formula_number(model.target_clip)}, "
                f"{self._format_formula_number(model.target_clip)}))"
            )
        return (
            f"f_hat({argument}) = {self._format_formula_number(model.target_center)} + "
            f"{self._format_formula_number(model.target_scale)}*clip({body}, "
            f"-{self._format_formula_number(model.target_clip)}, "
            f"{self._format_formula_number(model.target_clip)})"
        )

    def _model_labels(
        self,
        model: EnsembleSparseRegressor,
    ) -> Tuple[List[str], List[str], List[str]]:
        """Return variable labels, units, and descriptions for one model."""
        if model is self.surrogate or model is self.constraint_model:
            return self.variable_labels, self.variable_units, self.variable_descriptions
        feature_names = model.library.feature_names
        if feature_names is None:
            feature_names = [f"v{index}" for index in range(model.library.n_features)]
        labels = [str(value) for value in feature_names]
        return labels, [""] * len(labels), [""] * len(labels)

    def _term_interpretation(
        self,
        model: EnsembleSparseRegressor,
        index: int,
        coefficient: float,
        frequency: float,
        point_features: Optional[Array] = None,
    ) -> Dict[str, Any]:
        """Translate one dictionary term into a compact, evidence-based statement."""
        labels, units, descriptions = self._model_labels(model)
        name = model.library.names[index]
        spec = model.library._specs[index]
        kind = str(spec[0])
        selected = bool(float(frequency) >= 0.5 and abs(float(coefficient)) > 1e-10)
        item: Dict[str, Any] = {
            "term": name,
            "type": kind,
            "coefficient": float(coefficient),
            "selection_frequency": float(frequency),
            "selected": selected,
            "coefficient_space": (
                "transformed-objective space; the sparse linear combination is formed "
                "after the monotone target transform"
            ),
            "variables": [],
        }

        def variable(index_value: int) -> Dict[str, Any]:
            label = labels[index_value] if index_value < len(labels) else f"v{index_value}"
            unit = units[index_value] if index_value < len(units) else ""
            description = descriptions[index_value] if index_value < len(descriptions) else ""
            return {
                "index": int(index_value),
                "coordinate": f"z{index_value}" if model.library.bounds is not None else label,
                "label": label,
                "unit": unit or None,
                "description": description or None,
            }

        if kind == "const":
            item["interpretation"] = (
                "Intercept of the surrogate in the transformed-objective space. "
                "It is the reference response at the bound-normalized origin z = 0 "
                "and is not by itself a physical offset of the original objective."
            )
        elif kind == "linear":
            item["variables"] = [variable(int(spec[1]))]
            direction = (
                "increases" if coefficient > 0.0 else "decreases" if coefficient < 0.0 else "is locally insensitive to"
            )
            item["interpretation"] = (
                f"A first-order main effect of {item['variables'][0]['label']}. "
                f"When the term is retained, increasing z{int(spec[1])} {direction} "
                "the predicted objective in the transformed space."
            )
        elif kind == "square":
            item["variables"] = [variable(int(spec[1]))]
            if coefficient > 0.0:
                detail = (
                    "A positive coefficient is consistent with locally convex, bowl-shaped curvature: "
                    "deviations from the origin increase the predicted objective."
                )
            elif coefficient < 0.0:
                detail = (
                    "A negative coefficient is consistent with locally concave curvature. "
                    "This association is conditional on the current sample and bound constraints."
                )
            else:
                detail = "The quadratic coefficient is negligible relative to the selection threshold."
            item["interpretation"] = (
                f"A quadratic curvature term in {item['variables'][0]['label']}. {detail}"
            )
        elif kind == "interaction":
            left, right = int(spec[1]), int(spec[2])
            item["variables"] = [variable(left), variable(right)]
            if coefficient > 0.0:
                detail = (
                    "Co-directed deviations of the two coordinates are associated with a higher predicted objective; "
                    "opposite-signed deviations can offset that contribution."
                )
            elif coefficient < 0.0:
                detail = (
                    "Co-directed deviations of the two coordinates are associated with a lower predicted objective."
                )
            else:
                detail = "The interaction coefficient is negligible relative to the selection threshold."
            item["interpretation"] = (
                f"A bilinear interaction between {item['variables'][0]['label']} and "
                f"{item['variables'][1]['label']}. {detail} "
                "The term is not reducible to the sum of the two main effects."
            )
        elif kind in {"sine", "cosine"}:
            harmonic, feature_index = int(spec[1]), int(spec[2])
            item["variables"] = [variable(feature_index)]
            period = 2.0 / max(harmonic, 1)
            item["frequency"] = harmonic
            item["normalized_period"] = period
            item["interpretation"] = (
                f"A Fourier candidate of harmonic {harmonic} in {item['variables'][0]['label']} "
                f"({name}), with normalized period {period:.4g}. "
                "Retention indicates a repeatable association in the current sample, "
                "not a verified physical oscillation."
            )
        elif kind == "rms_x":
            item["interpretation"] = (
                "A global radial-amplitude summary, rms(x) = "
                "sqrt(mean_i(x_i^2)). A retained positive coefficient indicates "
                "that increasing the overall coordinate magnitude raises the "
                "transformed objective, conditional on the other selected terms."
            )
        elif kind == "radial_exp":
            item["interpretation"] = (
                "An explicit radial envelope exp(-0.2*rms(x)). This term captures "
                "a decaying response with distance from the coordinate origin and "
                "is retained only when supported by sparse regression."
            )
        elif kind == "mean_cosine":
            harmonic = int(spec[1])
            item["frequency"] = harmonic
            item["interpretation"] = (
                f"A global periodic summary mean(cos(2*pi*{harmonic}*x_i)) "
                "over all coordinates. It represents coherent oscillatory structure "
                "rather than an opaque nonlinear latent feature."
            )
        elif kind == "chain_residual":
            item["interpretation"] = (
                "A chain-coupling residual mean((x_i^2-x_{i+1})^2), which measures "
                "the violation of adjacent-coordinate quadratic consistency. It is "
                "a transparent nonseparable feature and not a black-box embedding."
            )
        else:
            item["interpretation"] = (
                "An unregistered dictionary term; interpretation is deferred to the selection audit."
            )

        if point_features is not None and index < len(point_features):
            item["feature_value"] = float(point_features[index])
            item["transformed_contribution"] = float(point_features[index] * coefficient)
        return item

    def _variable_semantics(self) -> List[Dict[str, Any]]:
        """Map bound-normalized coordinates to user-supplied or generic labels."""
        semantics = []
        for index, (lower, upper) in enumerate(self.bounds):
            center = (float(lower) + float(upper)) * 0.5
            half_range = (float(upper) - float(lower)) * 0.5
            label = self.variable_labels[index]
            unit = self.variable_units[index] or None
            description = self.variable_descriptions[index] or None
            if self._variable_labels_provided:
                meaning = (
                    f"User-supplied domain label {label}. The surrogate is identified in z{index}; "
                    "the label is used only to render the formula in domain language and does not "
                    "constitute an independent physical identification."
                )
                status = "user_provided_label"
            else:
                meaning = (
                    f"{label} is the {index}-th bound-normalized design variable of the benchmark. "
                    "No temperature, pressure, or other physical identity is assigned automatically."
                )
                status = "generic_benchmark_variable"
            semantics.append({
                "index": index,
                "coordinate": f"z{index}",
                "label": label,
                "unit": unit,
                "description": description,
                "bounds": [float(lower), float(upper)],
                "normalization": f"z{index}=({label}-{center:.6g})/{half_range:.6g}",
                "meaning": meaning,
                "physical_status": status,
            })
        return semantics

    @staticmethod
    def _search_semantics() -> Dict[str, Any]:
        """Define the search quantities that appear in the white-box audit."""
        return {
            "F": {
                "name": "differential scaling factor",
                "meaning": (
                    "Differential scaling factor: the step-length multiplier of the difference vector. "
                    "Larger values increase the search radius."
                ),
            },
            "CR": {
                "name": "crossover inheritance probability",
                "meaning": (
                    "Crossover inheritance probability: the expected fraction of coordinates inherited "
                    "from the donor vector."
                ),
            },
            "diversity": {
                "name": "population diversity",
                "meaning": (
                    "Mean coordinate-wise spread of the current population after bound normalization. "
                    "A low value is consistent with concentration or stall."
                ),
            },
            "stagnation": {
                "name": "stagnation level",
                "meaning": "Normalized count of consecutive batches that did not refresh the incumbent.",
            },
            "improvement_rate": {
                "name": "offspring improvement rate",
                "meaning": (
                    "Fraction of true-evaluated offspring that replaced their parents, measuring the "
                    "empirical success of the current difference operators."
                ),
            },
            "uncertainty": {
                "name": "surrogate uncertainty",
                "meaning": (
                    "Composite credibility signal from ensemble disagreement, local residual spread, "
                    "and distance to the labelled archive."
                ),
            },
            "predicted_gain": {
                "name": "predicted improvement",
                "meaning": (
                    "Predicted objective decrease of an offspring relative to its parent, used for "
                    "candidate ranking and true-evaluation allocation."
                ),
            },
            "strategies": {
                "1": "current-to-pbest/1: move toward a p-best parent and add an archive-aware difference.",
                "2": "rand/1: random difference vectors for broader exploration.",
                "3": "best/1: mutation around the current global best for intensified exploitation.",
                "4": "gaussian-local: Gaussian perturbation of the elite or in-ellipsoid trust-region point, scaled by the current population standard deviation.",
            },
            "trust_region_acquisition": {
                "name": "uncertainty-calibrated population ellipsoid",
                "meaning": (
                    "Local minimizer of the identified sparse formula inside the population "
                    "confidence ellipsoid, contracted by surrogate quality and predictive "
                    "uncertainty. An unconstrained stationary point is not used as a global jump."
                ),
            },
        }

    def _model_report(self, model: EnsembleSparseRegressor, point: Optional[Array] = None) -> Dict[str, Any]:
        if not model.fitted:
            return {
                "selected_terms": [],
                "coefficients": {},
                "selection_frequency": {},
                "formula_text": "",
                "term_interpretations": [],
            }
        coefficients = model.mean_coefficients()
        low, high = model.coefficient_intervals()
        frequencies = model.selection_frequency()
        point_features = None
        if point is not None:
            point_features = model.library.transform(np.asarray(point).reshape(1, -1))[0]
        report: Dict[str, Any] = {
            "selected_terms": model.selected_terms(),
            "coefficients": dict(zip(model.library.names, coefficients.tolist())),
            "coefficient_intervals": {
                name: [float(left), float(right)]
                for name, left, right in zip(model.library.names, low, high)
            },
            "selection_frequency": dict(zip(model.library.names, model.selection_frequency().tolist())),
            "diagnostics": model.diagnostics(),
            "target_transform": model.target_transform_name,
            "normalized_coordinates": bool(model.normalized),
            "selection_audit": model.selection_audit(),
            "formula_text": self._formula_text(model),
            "term_interpretations": [
                self._term_interpretation(
                    model, index, float(coefficient), float(frequency), point_features
                )
                for index, (coefficient, frequency) in enumerate(zip(coefficients, frequencies))
            ],
        }
        if point is not None:
            features = point_features
            contributions = features * coefficients
            prediction, uncertainty = model.predict_with_uncertainty(np.asarray(point).reshape(1, -1))
            report["transformed_local_contributions"] = dict(zip(model.library.names, contributions.tolist()))
            if model.target_transform_name == "robust_affine":
                objective_contributions = contributions * float(model.target_scale)
                contribution_note = "Contributions are reported on the original objective scale; the intercept still requires target_center."
            else:
                objective_contributions = contributions
                contribution_note = "The inverse asinh map is nonlinear; the listed contributions remain in the transformed space."
            report["local_contributions"] = dict(zip(model.library.names, objective_contributions.tolist()))
            report["contribution_note"] = contribution_note
            report["prediction"] = float(prediction[0])
            report["uncertainty"] = float(uncertainty[0])
            report["formula_prediction"] = float(
                np.mean(model.predict_formula_samples(np.asarray(point).reshape(1, -1)))
            )
            report["local_correction"] = float(model.last_local_correction[0])
            report["local_correction_weight"] = float(model.last_local_weight[0])
            report["local_correction_spread"] = float(model.last_local_spread[0])
        return report

    def explain(self, x: Optional[Array] = None) -> Dict[str, Any]:
        if not self.archive_y:
            raise RuntimeError("minimize must be called before explain()")
        point = np.asarray(
            x if x is not None else self.archive_x[self._best_archive_index()],
            dtype=float,
        ).reshape(-1)
        objective_report = self._model_report(self.surrogate, point)
        improvement_report = self._model_report(self.improvement_model)
        constraint_report = (
            self._model_report(self.constraint_model)
            if self.constraint_model.fitted else {}
        )
        validation_mae = float(np.mean(self.validation_errors)) if self.validation_errors else None
        trust_fractions = [
            float(row["trust_fraction"])
            for row in self.history
            if "trust_fraction" in row and np.isfinite(row["trust_fraction"])
        ]
        trust_report = {
            "trust_threshold": self.trust_threshold,
            "validation_mae": validation_mae,
            "surrogate_rmse": self.surrogate.in_sample_rmse() if self.surrogate.fitted else None,
            "surrogate_cv_rmse": float(self.surrogate.cv_rmse) if np.isfinite(self.surrogate.cv_rmse) else None,
            "surrogate_quality": float(self.surrogate.quality_score),
            "surrogate_training_samples": int(
                len(self.surrogate.training_y) if self.surrogate.training_y is not None else 0
            ),
            "max_surrogate_samples": (
                None
                if self.max_surrogate_samples is None
                else int(self.max_surrogate_samples)
            ),
            "surrogate_samples_per_dimension": float(
                self.surrogate_samples_per_dimension
            ),
            "relative_improvement_mode": self.relative_improvement_mode,
            "relative_improvement_tolerance": float(
                self.relative_improvement_tolerance
            ),
            "last_stagnation_diagnostics": deepcopy(
                self.last_stagnation_diagnostics
            ),
            "mean_trust_fraction": (
                float(np.mean(trust_fractions)) if trust_fractions else None
            ),
            "high_uncertainty_generations": int(sum(
                row.get("trust_outside", 0.0) > 0 for row in self.history
            )),
            "final_confirmation": bool(self.final_confirmation),
            "final_confirmation_record": dict(self.final_confirmation_record),
            "uncertainty_calibration": float(self.surrogate.uncertainty_calibration),
            "uncertainty_calibration_updates": int(self.surrogate.uncertainty_calibration_updates),
        }
        strategy_probability_map = {
            str(index): float(self.de_strategy_probabilities[index - 1])
            for index in self.de_strategy_names
        }
        strategy_name_map = {
            str(index): name for index, name in self.de_strategy_names.items()
        }
        strategy_selected_map = {
            self.de_strategy_names[index]: float(self.de_strategy_selected[index - 1])
            for index in self.de_strategy_names
        }
        strategy_successful_map = {
            self.de_strategy_names[index]: float(self.de_strategy_successful[index - 1])
            for index in self.de_strategy_names
        }
        strategy_gain_map = {
            self.de_strategy_names[index]: float(self.de_strategy_gain[index - 1])
            for index in self.de_strategy_names
        }
        policy_report = {
            "learned": bool(self.improvement_model.fitted),
            "active": bool(any(row.get("improvement_policy_accepted", 0.0) > 0 for row in self.history)),
            "state_names": list(self.state_names),
            "last_observation": dict(self.last_policy),
            "de_strategy_names": [
                self.de_strategy_names[index] for index in self.de_strategy_names
            ],
            "de_strategy_name_map": strategy_name_map,
            "de_strategy_probabilities": strategy_probability_map,
            "de_strategy_probability_vector": [
                float(value) for value in self.de_strategy_probabilities
            ],
            "de_strategy_probability_floor": float(
                self.de_strategy_probability_floor
            ),
            "de_local_strategy_max_probability": float(
                self.de_local_strategy_max_probability
            ),
            "de_strategy_selected": strategy_selected_map,
            "de_strategy_successful": strategy_successful_map,
            "de_strategy_gain": strategy_gain_map,
            "de_strategy_update_log": deepcopy(self.de_strategy_update_log),
            "de_strategy_evaluation": deepcopy(self.de_strategy_policy_evaluation),
            "de_archive_size": int(len(self.de_archive_x)),
            "de_archive_capacity": int(self.de_archive_capacity),
                "de_policy_probe_count": int(self.de_policy_probe_count),
                "de_policy_update_interval": int(self.de_policy_update_interval),
                "global_candidate_fraction": float(self.de_global_candidate_fraction),
            "policy_feedback_sample_count": int(len(self.policy_feedback_archive)),
            "policy_supervision_rule": (
                "F/CR policy supervision uses only offspring with true_evaluated=True and "
                "actual_improvement>0; failed or surrogate-only candidates are excluded"
            ),
            "de_strategy_selection_rule": (
                "the operator arm is selected from realized true-evaluation gains; "
                "the reliability-gated sparse improvement ensemble probes F/CR settings "
                "only for that empirically supported arm, and SHADE remains the anchor"
            ),
            "shade_memory": {
                "F": [float(value) for value in self.shade_m_f],
                "CR": [float(value) for value in self.shade_m_cr],
                "next_memory_index": int(self.shade_index),
                "successful_updates": deepcopy(self.shade_success_log),
                "update_rule": "successful true offspring only; weighted Lehmer mean for F and weighted arithmetic mean for CR",
            },
            "F_terms": self.policy_model.f_model.selected_terms() if self.policy_model.fitted else [],
            "CR_terms": self.policy_model.cr_model.selected_terms() if self.policy_model.fitted else [],
            "F_formula_text": self._formula_text(self.policy_model.f_model) if self.policy_model.fitted else "",
            "CR_formula_text": self._formula_text(self.policy_model.cr_model) if self.policy_model.fitted else "",
            "F_selection_audit": self.policy_model.f_model.selection_audit() if self.policy_model.fitted else {},
            "CR_selection_audit": self.policy_model.cr_model.selection_audit() if self.policy_model.fitted else {},
            "fallback_formula": "F=clip(0.42+0.30*stagnation+0.10*uncertainty+0.16*diversity); CR=clip(0.74+0.10*(1-diversity)+0.10*improvement_rate-0.08*uncertainty)",
        }
        global_report = {
            "objective_terms": objective_report.get("selected_terms", []),
            "improvement_terms": improvement_report.get("selected_terms", []),
            "objective_model": objective_report,
            "improvement_model": improvement_report,
            "constraint_model": constraint_report,
        }
        surrogate_decision_audit = {
            "candidate_generation": {
                "pool_size_formula": "max(population_size, population_size*candidate_pool_multiplier)",
                "parent_rule": "shuffle repeated population indices so each candidate has a parent",
                "parameter_sampling": (
                    "half of the candidates use the SHADE-like memory and the remainder "
                    "are sampled around the selected policy; F is clipped to [0.25, 1.0], "
                    "CR to [0.10, 0.99], and the first quarter uses the selected values"
                ),
                "strategy_names": strategy_name_map,
                "strategy_probabilities": strategy_probability_map,
                "strategy_probability_update_rule": (
                    "p_new=high_dimensional_floor_cap((1-lambda)*p_old+lambda*q); "
                    "q=successful_true_gain/selected_true_trials"
                ),
                "relative_improvement_tolerance": float(
                    self.relative_improvement_tolerance
                ),
                "policy_strategy_rule": (
                    "the selected strategy arm is forced into 35% of the next candidate "
                    "pool; the remaining candidates follow the adaptive probability distribution"
                ),
                "policy_strategy_evaluation": deepcopy(self.de_strategy_policy_evaluation),
                "policy_probe_count": int(self.de_policy_probe_count),
                "mutation_rules": {
                    "1": "parent + F*(pbest-parent) + F*(r1-r2)",
                    "2": "r1 + F*(r2-r3)",
                    "3": "best + F*(r1-r2)",
                    "4": "elite or in-ellipsoid trust-region point + Normal(0, 0.50*F*population_std); CR is raised so the local trial is inherited",
                },
                "trust_region_rule": (
                    "minimize the sparse formula inside the uncertainty-calibrated "
                    "population ellipsoid around the incumbent; truncated Newton is used "
                    "only when the Hessian is positive definite, otherwise a Cauchy point "
                    "is taken. The unconstrained vertex H z = -g is never injected when it "
                    "lies outside the ellipsoid or the Hessian is not PD"
                ),
                "crossover_rule": "each coordinate takes donor when Uniform(0,1)<CR; one coordinate is forced to donor",
                "boundary_rule": "clip every trial to variable bounds",
                "block_rule": (
                    f"for dimension >= {self.de_high_dimension_threshold}, crossover is "
                    "restricted to one contiguous variable block; configured block size "
                    "is adapted when it would equal the full dimension"
                ),
            },
            "candidate_prediction": {
                "objective_prediction": "mean over the fitted sparse ensemble formula predictions",
                "uncertainty": self.surrogate.selection_audit().get("uncertainty_rule", {}),
                "improvement_prediction": (
                    "EnsembleSparseRegressor applied to step_size, parent_gap, parent_rank, "
                    "F, CR, state variables, explicit one-hot strategy indicators, and "
                    "strategy-specific F/CR interaction features"
                ),
                "acquisition_formula": (
                    "predicted - model_weight*uncertainty - gain_weight*calibrated_gain_signal"
                ),
                "model_weight_rule": "uncertainty_weight when objective surrogate quality >= min_model_quality, otherwise 0.05",
                "gain_weight_rule": "0.25*model_weight when improvement surrogate quality >= min_model_quality, otherwise 0",
            },
            "real_evaluation_selection": {
                "explicit_quota": "mandatory predicted-best confirmation + mandatory high-risk confirmation, then predicted-gain, trusted proxy exploitation, risk ranking, and diversity fill",
                "quota_formula": "predicted_best is always selected; high_risk is selected when batch>=2; untrusted candidates are excluded from proxy exploitation",
                "trust_rule": "proxy exploitation requires normalized uncertainty <= trust_threshold and normalized distance <= 0.35",
                "selection_records": deepcopy(self.real_selection_audit_records),
            },
            "feedback_rule": (
                "only true objective evaluations enter archive_y; a trial replaces its "
                "parent or, if it loses the parent contest, the current worst member; "
                "SHADE memory and the F/CR policy are updated only from parent-beating "
                "true offspring; the trust-region point is a local acquisition candidate "
                "and is not a mandatory true-evaluation slot"
            ),
            "trust_region_acquisition": deepcopy(self.last_trust_region_report),
            "trust_region_feedback": deepcopy(self.trust_region_feedback[-32:]),
            "final_confirmation": {
                "rule": "reserve one real evaluation for the best surrogate acquisition candidate",
                "refit_after_confirmation": True,
                "record": dict(self.final_confirmation_record),
            },
        }
        local_report = {
            "x": point.tolist(),
            "objective_contributions": objective_report.get("local_contributions", {}),
            "transformed_objective_contributions": objective_report.get("transformed_local_contributions", {}),
            "contribution_note": objective_report.get("contribution_note", ""),
            "formula_text": objective_report.get("formula_text", ""),
            "predicted_objective": objective_report.get("prediction"),
            "formula_prediction": objective_report.get("formula_prediction"),
            "local_correction": objective_report.get("local_correction"),
            "local_correction_weight": objective_report.get("local_correction_weight"),
            "local_correction_spread": objective_report.get("local_correction_spread"),
            "objective_uncertainty": objective_report.get("uncertainty"),
        }
        return {
            "compute_backend": self.compute_backend.audit(),
            "gpu_objective": {
                "requested": bool(self.gpu_objective),
                "available": callable(
                    getattr(self.objective, "evaluate_batch_device", None)
                ),
                "active": bool(
                    self.gpu_objective
                    and self.compute_backend.cuda_enabled
                    and callable(
                        getattr(self.objective, "evaluate_batch_device", None)
                    )
                ),
                "batch_size": int(self.real_batch_size),
            },
            "selected_terms": objective_report.get("selected_terms", []),
            "mean_coefficients": objective_report.get("coefficients", {}),
            "local_contributions": objective_report.get("local_contributions", {}),
            "prediction": objective_report.get("prediction"),
            "uncertainty": objective_report.get("uncertainty"),
            "formula_text": objective_report.get("formula_text", ""),
            "variable_semantics": self._variable_semantics(),
            "term_interpretations": objective_report.get("term_interpretations", []),
            "interpretation_scope": (
                "The sparse formula is identified on bound-normalized coordinates z in [-1, 1]. "
                "User-supplied labels, if present, map z_i to domain language. "
                "Retained terms describe stable associations in the surrogate and the implemented "
                "search mechanism; they are not claimed as causal physical laws."
            ),
            "search_semantics": self._search_semantics(),
            "surrogate_decision_audit": surrogate_decision_audit,
            "global": global_report,
            "local": local_report,
            "search_process": {
                "history": list(self.history),
                "candidate_count": int(self.candidate_archive_total_count),
                "candidate_records_retained": len(self.candidate_archive),
                "candidate_records_dropped": int(self.candidate_archive_dropped_count),
                "candidate_audit_retention": f"latest {self.max_candidate_audit_records} records",
            },
            "trustworthiness": trust_report,
            "policy": policy_report,
            "screening": {
                "dimension": self.dimension,
                "active_variables": list(self.active_dimensions),
                "active_variable_count": len(self.active_dimensions),
                "max_active_variables": self.max_active_variables,
                "block_mutation_size": self.block_mutation_size,
                "surrogate_fit_count": self.surrogate_fit_count,
                "auxiliary_fit_count": self.auxiliary_fit_count,
                "candidate_count": int(self.candidate_archive_total_count),
                "candidate_records_retained": len(self.candidate_archive),
                "candidate_records_dropped": int(self.candidate_archive_dropped_count),
                "supervised_sample_count": len(self.archive_y),
                "screening_fit_count": int(self.screening_fit_count),
                "screening_audit_retention": (
                    f"latest {self.max_screening_audit_records} fits"
                ),
                "screening_audit": deepcopy(self.screening_audit_records),
            },
        }

__all__ = [
    'XDESindy',
]

