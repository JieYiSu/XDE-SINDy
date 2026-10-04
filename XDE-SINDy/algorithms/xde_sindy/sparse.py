"""Sparse regression and hierarchy rules for interpretable surrogates.

The routines in this module perform standardized forward selection followed by
ridge refitting.  Their audit records expose the selected terms, scores,
hierarchy additions, and stopping conditions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .backend import ComputeBackend
from .dictionary import Array

def _ridge_fit(
    x: Array,
    y: Array,
    alpha: float,
    compute_backend: ComputeBackend | None = None,
    *,
    return_device: bool = False,
) -> Any:
    """Solve ridge regression in the smaller feature or sample space."""
    if compute_backend is not None:
        return compute_backend.ridge_fit(
            x,
            y,
            alpha,
            return_device=return_device,
        )
    if x.shape[1] == 0:
        return np.empty(0, dtype=float)
    if x.shape[1] > x.shape[0]:
        kernel = x @ x.T
        kernel.flat[:: kernel.shape[0] + 1] += float(alpha)
        try:
            return x.T @ np.linalg.solve(kernel, y)
        except np.linalg.LinAlgError:
            return x.T @ (np.linalg.pinv(kernel) @ y)
    gram = x.T @ x
    gram.flat[:: gram.shape[0] + 1] += float(alpha)
    rhs = x.T @ y
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram) @ rhs


def _is_periodic_term(name: str) -> bool:
    return name.startswith("sin(") or name.startswith("cos(")


def _is_interaction_term(name: str) -> bool:
    return "*" in name


def _is_square_term(name: str) -> bool:
    return "^2" in name


def _is_linear_term(name: str) -> bool:
    return (
        name != "1"
        and not _is_square_term(name)
        and not _is_interaction_term(name)
        and not _is_periodic_term(name)
    )


def _parent_term_names(name: str) -> List[str]:
    """Return linear parents of a square or pairwise product term."""
    if _is_square_term(name):
        return [name[:-2]]
    if _is_interaction_term(name):
        return [part for part in name.split("*") if part]
    return []


def _fit_sparse_coefficients(
    dictionary: Array,
    target: Array,
    alpha: float,
    threshold: float,
    max_terms: int,
    return_audit: bool = False,
    term_names: Optional[Sequence[str]] = None,
    polynomial_first: bool = True,
    compute_backend: ComputeBackend | None = None,
) -> Tuple[Array, Array] | Tuple[Array, Array, Dict[str, Any]]:
    """Select terms by residual correlation and refit with ridge regression.

    Columns are standardized before scoring so large-scale terms are not preferred
    merely by magnitude. Linear and quadratic dictionary columns compete on this
    common scale. After a square or interaction is accepted, missing parent linear
    terms are included (weak hierarchy). Periodic terms may be delayed until the
    polynomial residual remains large. The returned coefficients are mapped back
    to the original column scale; the selected mask excludes the constant column.
    """
    n_samples, n_terms = dictionary.shape
    if compute_backend is None:
        xp = np
        dictionary_xp = np.asarray(dictionary, dtype=float)
        target_xp = np.asarray(target, dtype=float)
    else:
        xp, (dictionary_xp, target_xp) = compute_backend.arrays_for(
            "sparse_selection",
            dictionary,
            target,
        )
        dictionary_xp = dictionary_xp.astype(xp.float64, copy=False)
        target_xp = target_xp.astype(xp.float64, copy=False)
    coefficients_xp = xp.zeros(n_terms, dtype=xp.float64)
    if n_terms <= 1:
        mean_target = xp.mean(target_xp)
        coefficients_xp[0] = mean_target
        coefficients = (
            np.asarray(coefficients_xp, dtype=float)
            if compute_backend is None
            else np.asarray(compute_backend.to_numpy(coefficients_xp), dtype=float)
        )
        audit = {
            "selection_path": [],
            "stop_reason": "constant_only_dictionary",
            "candidate_count": 0,
            "max_terms": int(max_terms),
            "threshold": float(threshold),
        }
        return (coefficients, np.empty(0, dtype=int), audit) if return_audit else (coefficients, np.empty(0, dtype=int))

    features = dictionary_xp[:, 1:]
    means = xp.mean(features, axis=0)
    scales = xp.std(features, axis=0)
    scales = xp.where((~xp.isfinite(scales)) | (scales < 1e-8), 1.0, scales)
    standardized = (features - means) / scales
    target_mean = xp.mean(target_xp)
    centered_target = target_xp - target_mean
    # All forward-selection ridge systems use subsets of this same
    # standardized design. Cache the normal equations once per fit instead of
    # recomputing X_selected.T @ X_selected for every selected term.
    cached_gram = standardized.T @ standardized
    cached_rhs = standardized.T @ centered_target

    def cached_ridge(indices: Any) -> Any:
        indices_xp = (
            indices
            if hasattr(indices, "dtype") and getattr(indices, "ndim", 0) == 1
            else xp.asarray(indices, dtype=xp.int64)
        )
        local_gram = cached_gram[indices_xp[:, None], indices_xp[None, :]]
        local_rhs = cached_rhs[indices_xp]
        if compute_backend is not None:
            return compute_backend.ridge_fit_from_normal_equations(
                local_gram,
                local_rhs,
                alpha,
                return_device=(xp is not np),
            )
        local_gram = np.asarray(local_gram, dtype=float).copy()
        local_rhs = np.asarray(local_rhs, dtype=float)
        local_gram.flat[:: local_gram.shape[0] + 1] += float(alpha)
        try:
            return np.linalg.solve(local_gram, local_rhs)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(local_gram) @ local_rhs

    residual = centered_target.copy()
    limit = min(int(max_terms), n_terms - 1, max(1, n_samples - 1))
    selected: List[int] = []
    raw_target_scale = xp.std(centered_target)
    target_scale = max(
        float(raw_target_scale) if compute_backend is None else compute_backend.scalar(raw_target_scale),
        1e-8,
    )
    audit: Dict[str, Any] = {
        "selection_path": [],
        "stop_reason": "max_terms_reached",
        "candidate_count": int(n_terms - 1),
        "max_terms": int(limit),
        "threshold": float(threshold),
        "target_scale": float(target_scale),
        "standardization": "each non-constant dictionary column is centered and scaled by its sample standard deviation",
        "score_definition": "absolute correlation between standardized candidate column and current residual",
        "selected_indices": [],
        "polynomial_first": bool(polynomial_first),
        "square_first": False,
        "periodic_terms_admitted": False,
        "periodic_admission_reason": "not_required",
        "interaction_terms_admitted": True,
        "interaction_admission_reason": "polynomial_terms_compete_from_the_first_step",
        "linear_terms_admitted": True,
        "linear_admission_reason": "linear_and_quadratic_compete_from_the_first_step",
        "hierarchy_rule": (
            "weak hierarchy: linear and quadratic terms compete on a standardized "
            "residual-correlation scale; selecting z_i^2 or z_i z_j includes any "
            "missing parent linear terms before the ridge refit"
        ),
    }
    names = list(term_names) if term_names is not None else []
    periodic_mask = np.zeros(n_terms - 1, dtype=bool)
    name_to_feature: Dict[str, int] = {}
    if names:
        for index, name in enumerate(names[1:n_terms]):
            label = str(name)
            periodic_mask[index] = _is_periodic_term(label)
            name_to_feature[label] = index
    admit_periodic = (not polynomial_first) or (not np.any(periodic_mask))
    if admit_periodic:
        audit["periodic_terms_admitted"] = True
        audit["periodic_admission_reason"] = "periodic_terms_compete_from_the_first_step" if not polynomial_first else "not_required"
    initial_norm_value = xp.linalg.norm(residual)
    initial_residual_norm = (
        float(initial_norm_value)
        if compute_backend is None
        else compute_backend.scalar(initial_norm_value)
    )

    for step in range(limit):
        if len(selected) >= limit:
            audit["stop_reason"] = "max_terms_reached"
            break
        correlations = xp.abs(standardized.T @ residual) / max(n_samples, 1)
        if selected:
            correlations[xp.asarray(selected, dtype=xp.int64)] = -xp.inf
        if not admit_periodic:
            correlations[xp.asarray(periodic_mask)] = -xp.inf
        best_value = xp.argmax(correlations)
        best = int(best_value) if compute_backend is None else int(compute_backend.scalar(best_value))
        score_value = correlations[best]
        best_score = (
            float(score_value)
            if compute_backend is None
            else compute_backend.scalar(score_value)
        )
        if not np.isfinite(best_score) or best_score < threshold * target_scale:
            residual_norm_value = xp.linalg.norm(residual)
            residual_norm = (
                float(residual_norm_value)
                if compute_backend is None
                else compute_backend.scalar(residual_norm_value)
            )
            residual_ratio = float(
                residual_norm / max(initial_residual_norm, 1e-12)
            )
            if not admit_periodic and residual_ratio > 0.40:
                admit_periodic = True
                audit["periodic_terms_admitted"] = True
                audit["periodic_admission_reason"] = "polynomial_residual_ratio_above_0.40"
                continue
            audit["stop_reason"] = "correlation_below_threshold"
            audit["last_max_score"] = best_score if np.isfinite(best_score) else None
            audit["threshold_score"] = float(threshold * target_scale)
            break
        selected.append(best)
        heredity_parents: List[int] = []
        if names and (best + 1) < len(names):
            for parent in _parent_term_names(str(names[best + 1])):
                parent_idx = name_to_feature.get(parent)
                if (
                    parent_idx is not None
                    and parent_idx not in selected
                    and len(selected) < limit
                ):
                    selected.append(parent_idx)
                    heredity_parents.append(parent_idx)
        beta = cached_ridge(selected)
        new_residual = centered_target - standardized[:, selected] @ beta
        old_norm_value = xp.linalg.norm(residual)
        new_norm_value = xp.linalg.norm(new_residual)
        if compute_backend is None:
            old_norm = float(old_norm_value)
            new_norm = float(new_norm_value)
        else:
            old_norm = compute_backend.scalar(old_norm_value)
            new_norm = compute_backend.scalar(new_norm_value)
        reduction = max(0.0, old_norm - new_norm)
        audit["selection_path"].append({
            "step": int(step + 1),
            "candidate_index": int(best + 1),
            "selection_score": best_score,
            "threshold_score": float(threshold * target_scale),
            "selected_indices": [0] + [int(index + 1) for index in selected],
            "heredity_parent_indices": [int(index + 1) for index in heredity_parents],
            "residual_norm_before": old_norm,
            "residual_norm_after": new_norm,
            "relative_residual_reduction": reduction / max(old_norm, 1e-12),
        })
        residual = new_residual
        if step >= 4 and old_norm > 1e-12 and new_norm / old_norm > 0.995:
            audit["stop_reason"] = "residual_reduction_stagnated"
            break

    if not selected:
        coefficients_xp[0] = target_mean
        coefficients = (
            np.asarray(coefficients_xp, dtype=float)
            if compute_backend is None
            else np.asarray(compute_backend.to_numpy(coefficients_xp), dtype=float)
        )
        if audit["stop_reason"] == "max_terms_reached":
            audit["stop_reason"] = "no_candidate_passed_threshold"
        return (coefficients, np.empty(0, dtype=int), audit) if return_audit else (coefficients, np.empty(0, dtype=int))

    selected_array = np.asarray(selected, dtype=int)
    selected_xp = xp.asarray(selected_array, dtype=xp.int64)
    beta = cached_ridge(selected_xp)
    coefficients_xp[1 + selected_xp] = beta / scales[selected_xp]
    coefficients_xp[0] = target_mean - xp.sum(
        beta * means[selected_xp] / scales[selected_xp]
    )
    coefficients = (
        np.asarray(coefficients_xp, dtype=float)
        if compute_backend is None
        else np.asarray(compute_backend.to_numpy(coefficients_xp), dtype=float)
    )
    audit["selected_indices"] = [0] + [int(index + 1) for index in selected_array]
    return (coefficients, selected_array, audit) if return_audit else (coefficients, selected_array)


def _fit_hierarchical_additive_coefficients(
    dictionary: Array,
    target: Array,
    alpha: float,
    threshold: float,
    max_groups: int,
    term_names: Sequence[str],
    retain_all_main_effect_groups: bool = False,
    compute_backend: ComputeBackend | None = None,
) -> Tuple[Array, Array, Dict[str, Any]]:
    """Fit a white-box additive backbone followed by sparse residual terms.

    Each variable contributes one hierarchical group containing its linear and
    quadratic columns. Groups compete by their standardized fitted contribution,
    so a quadratic main effect never consumes a second model-complexity slot or
    appears without its linear parent. Interactions and periodic terms are then
    admitted only when they explain residual structure left by the additive model.
    """
    n_samples, n_terms = dictionary.shape
    names = [str(name) for name in term_names[:n_terms]]
    if n_terms <= 1 or len(names) != n_terms:
        result = _fit_sparse_coefficients(
            dictionary,
            target,
            alpha,
            threshold,
            max_groups,
            return_audit=True,
            term_names=term_names,
            compute_backend=compute_backend,
        )
        coefficients, selected, audit = result
        audit["hierarchical_additive_backbone"] = False
        audit["backbone_fallback_reason"] = "main-effect term metadata unavailable"
        return coefficients, selected, audit

    if compute_backend is None:
        xp = np
        dictionary_xp = np.asarray(dictionary, dtype=float)
        target_xp = np.asarray(target, dtype=float)
    else:
        xp, (dictionary_xp, target_xp) = compute_backend.arrays_for(
            "sparse_selection",
            dictionary,
            target,
        )
        dictionary_xp = dictionary_xp.astype(xp.float64, copy=False)
        target_xp = target_xp.astype(xp.float64, copy=False)

    def scalar(value: Any) -> float:
        if compute_backend is None:
            return float(value)
        return compute_backend.scalar(value)

    def to_numpy(value: Any) -> np.ndarray:
        if compute_backend is None:
            return np.asarray(value)
        return compute_backend.to_numpy(value)

    features = dictionary_xp[:, 1:]
    means = xp.mean(features, axis=0)
    scales = xp.std(features, axis=0)
    scales = xp.where((~xp.isfinite(scales)) | (scales < 1e-8), 1.0, scales)
    standardized = (features - means) / scales
    target_mean = xp.mean(target_xp)
    centered_target = target_xp - target_mean
    target_scale = max(scalar(xp.std(centered_target)), 1e-8)

    groups: Dict[str, List[int]] = {}
    optional: List[int] = []
    for feature_index, name in enumerate(names[1:]):
        if _is_linear_term(name):
            groups.setdefault(name, []).append(feature_index)
        elif _is_square_term(name):
            groups.setdefault(name[:-2], []).append(feature_index)
        else:
            optional.append(feature_index)
    groups = {
        name: sorted(set(indices))
        for name, indices in groups.items()
        if any(_is_linear_term(names[index + 1]) for index in indices)
    }
    if not groups:
        coefficients, selected, audit = _fit_sparse_coefficients(
            dictionary,
            target,
            alpha,
            threshold,
            max_groups,
            return_audit=True,
            term_names=term_names,
            compute_backend=compute_backend,
        )
        audit["hierarchical_additive_backbone"] = False
        audit["backbone_fallback_reason"] = "no complete linear-parent groups"
        return coefficients, selected, audit

    group_names = list(groups)
    all_main_indices = sorted({
        index for indices in groups.values() for index in indices
    })
    all_main_xp = xp.asarray(all_main_indices, dtype=xp.int64)
    group_scores: List[Tuple[float, str]] = []
    if retain_all_main_effect_groups:
        group_scoring = "marginal_standardized_correlation"
        correlations = (
            xp.abs(standardized[:, all_main_xp].T @ centered_target)
            / max(n_samples, 1)
        )
        correlation_values = np.asarray(to_numpy(correlations), dtype=float)
        correlation_lookup = {
            feature_index: float(correlation_values[position])
            for position, feature_index in enumerate(all_main_indices)
        }
        for group_name in group_names:
            score = float(np.linalg.norm([
                correlation_lookup[index] for index in groups[group_name]
            ]))
            group_scores.append((score, group_name))
    else:
        group_scoring = "joint_ridge_contribution"
        pilot_beta = _ridge_fit(
            standardized[:, all_main_xp], centered_target, alpha,
            compute_backend,
            return_device=(xp is not np),
        )
        main_positions = {
            feature_index: position
            for position, feature_index in enumerate(all_main_indices)
        }
        max_group_width = max(len(groups[name]) for name in group_names)
        group_features = np.zeros(
            (len(group_names), max_group_width),
            dtype=int,
        )
        group_beta_positions = np.zeros_like(group_features)
        group_mask = np.zeros_like(group_features, dtype=float)
        for row, group_name in enumerate(group_names):
            indices = groups[group_name]
            width = len(indices)
            group_features[row, :width] = indices
            group_beta_positions[row, :width] = [
                main_positions[index] for index in indices
            ]
            group_mask[row, :width] = 1.0
        group_features_xp = xp.asarray(group_features, dtype=xp.int64)
        group_beta_positions_xp = xp.asarray(
            group_beta_positions,
            dtype=xp.int64,
        )
        group_mask_xp = xp.asarray(group_mask, dtype=xp.float64)
        group_beta = pilot_beta[group_beta_positions_xp] * group_mask_xp
        contributions = xp.sum(
            standardized[:, group_features_xp] * group_beta[None, :, :],
            axis=2,
        )
        score_values = xp.sqrt(xp.mean(contributions * contributions, axis=0))
        score_values_np = np.asarray(to_numpy(score_values), dtype=float)
        for score, group_name in zip(score_values_np, group_names):
            group_scores.append((score, group_name))
    group_scores.sort(key=lambda item: item[0], reverse=True)
    group_limit = min(max(1, int(max_groups)), len(group_scores))
    score_threshold = float(threshold * target_scale)
    if retain_all_main_effect_groups:
        selected_groups = [name for _, name in group_scores[:group_limit]]
    else:
        selected_groups = [
            name for score, name in group_scores[:group_limit]
            if score >= score_threshold
        ]
    if not selected_groups:
        selected_groups = [group_scores[0][1]]

    group_score_lookup = {
        name: float(score) for score, name in group_scores
    }
    selected: List[int] = []
    selection_path: List[Dict[str, Any]] = []
    for step, group_name in enumerate(selected_groups, start=1):
        group_indices = groups[group_name]
        selected.extend(index for index in group_indices if index not in selected)
        selection_path.append({
            "step": int(step),
            "stage": "additive_backbone",
            "candidate_index": int(group_indices[0] + 1),
            "candidate_group": group_name,
            "group_indices": [int(index + 1) for index in group_indices],
            "selection_score": group_score_lookup[group_name],
            "threshold_score": score_threshold,
            "selected_indices": [0] + [int(index + 1) for index in selected],
            "heredity_parent_indices": [],
        })

    selected_xp = xp.asarray(selected, dtype=xp.int64)
    beta = _ridge_fit(
        standardized[:, selected_xp],
        centered_target,
        alpha,
        compute_backend,
        return_device=(xp is not np),
    )
    residual = centered_target - standardized[:, selected_xp] @ beta
    optional_budget = max(0, int(max_groups) - len(selected_groups))
    optional_selected = 0
    stop_reason = "additive_backbone_complete"
    while optional_selected < optional_budget and optional:
        optional_xp = xp.asarray(optional, dtype=xp.int64)
        correlations = (
            xp.abs(standardized[:, optional_xp].T @ residual)
            / max(n_samples, 1)
        )
        best_position = int(scalar(xp.argmax(correlations)))
        best_score = scalar(correlations[best_position])
        if not np.isfinite(best_score) or best_score < score_threshold:
            stop_reason = "residual_correlation_below_threshold"
            break
        best = int(optional.pop(best_position))
        heredity_parents: List[int] = []
        for parent_name in _parent_term_names(names[best + 1]):
            for parent_index in groups.get(parent_name, []):
                if parent_index not in selected:
                    selected.append(parent_index)
                    heredity_parents.append(parent_index)
        if best not in selected:
            selected.append(best)
        old_norm = scalar(xp.linalg.norm(residual))
        selected_xp = xp.asarray(selected, dtype=xp.int64)
        beta = _ridge_fit(
            standardized[:, selected_xp],
            centered_target,
            alpha,
            compute_backend,
            return_device=(xp is not np),
        )
        residual = centered_target - standardized[:, selected_xp] @ beta
        new_norm = scalar(xp.linalg.norm(residual))
        optional_selected += 1
        selection_path.append({
            "step": int(len(selection_path) + 1),
            "stage": "residual_dictionary",
            "candidate_index": int(best + 1),
            "selection_score": best_score,
            "threshold_score": score_threshold,
            "selected_indices": [0] + [int(index + 1) for index in selected],
            "heredity_parent_indices": [int(index + 1) for index in heredity_parents],
            "residual_norm_before": old_norm,
            "residual_norm_after": new_norm,
            "relative_residual_reduction": max(0.0, old_norm - new_norm) / max(old_norm, 1e-12),
        })
        if old_norm > 1e-12 and new_norm / old_norm > 0.995:
            stop_reason = "residual_reduction_stagnated"
            break
    if optional_selected >= optional_budget and optional_budget > 0:
        stop_reason = "residual_group_budget_reached"

    selected_array = np.asarray(sorted(set(selected)), dtype=int)
    selected_xp = xp.asarray(selected_array, dtype=xp.int64)
    beta = _ridge_fit(
        standardized[:, selected_xp],
        centered_target,
        alpha,
        compute_backend,
        return_device=(xp is not np),
    )
    coefficients_xp = xp.zeros(n_terms, dtype=xp.float64)
    coefficients_xp[1 + selected_xp] = beta / scales[selected_xp]
    coefficients_xp[0] = target_mean - xp.sum(
        beta * means[selected_xp] / scales[selected_xp]
    )
    coefficients = np.asarray(to_numpy(coefficients_xp), dtype=float)
    audit = {
        "selection_path": selection_path,
        "stop_reason": stop_reason,
        "candidate_count": int(n_terms - 1),
        "max_terms": int(max_groups),
        "max_main_effect_groups": int(group_limit),
        "threshold": float(threshold),
        "target_scale": target_scale,
        "standardization": "each non-constant dictionary column is centered and scaled by its sample standard deviation",
        "score_definition": "RMS contribution of each jointly fitted linear/quadratic variable group; residual terms use absolute standardized residual correlation",
        "selected_indices": [0] + [int(index + 1) for index in selected_array],
        "selected_main_effect_groups": list(selected_groups),
        "main_effect_group_scoring": group_scoring,
        "retain_all_main_effect_groups": bool(
            retain_all_main_effect_groups
        ),
        "main_effect_group_scores": {
            name: float(score) for score, name in group_scores
        },
        "residual_term_count": int(optional_selected),
        "hierarchical_additive_backbone": True,
        "polynomial_first": True,
        "square_first": False,
        "periodic_terms_admitted": bool(optional_selected > 0),
        "interaction_terms_admitted": bool(optional_selected > 0),
        "linear_terms_admitted": True,
        "hierarchy_rule": (
            "strong group hierarchy for main effects: z_i and z_i^2 share one "
            "variable-group budget; residual interactions add missing linear parents"
        ),
    }
    return coefficients, selected_array, audit


__all__ = [
    "_ridge_fit",
    "_fit_sparse_coefficients",
    "_fit_hierarchical_additive_coefficients",
]



