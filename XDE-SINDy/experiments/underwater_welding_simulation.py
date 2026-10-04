"""Reduced-order digital-twin experiment for an underwater welding robot.

The previous version was only a hand-written objective proxy.  This version
contains an explicit, calibratable simulation chain:

* a six-axis serial robot with standard DH forward kinematics;
* joint-limit, workspace, workpiece-clearance, and trajectory constraints;
* a moving heat-source temperature field with water-side convective cooling;
* melt-pool width/depth, heat-input, arc-stability, and porosity indicators;
* a constrained single-objective quality/energy/trajectory score.

It is a reduced-order digital twin, so the coefficients must be calibrated
against the actual robot, torch, material, water temperature/flow, and weld
measurements before the results are described as validated engineering data.
The embedded ``RUN_*`` configuration makes the file directly runnable from an
IDE or by double-clicking it; edit those constants for a different run.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.special import ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algorithms.xde_sindy.core import XDESindy  # noqa: E402


@dataclass(frozen=True)
class DigitalTwinConfig:
    """Calibratable reduced-order physics and robot parameters."""

    ambient_temperature_c: float = 4.0
    liquidus_temperature_c: float = 1450.0
    thermal_conductivity_w_mk: float = 25.0
    density_kg_m3: float = 7800.0
    heat_capacity_j_kgk: float = 600.0
    water_convection_w_m2k: float = 1500.0
    arc_efficiency: float = 0.72
    rosenthal_scale: float = 0.18
    weld_length_m: float = 0.16
    path_points: int = 24
    workpiece_x_m: float = 0.55
    workpiece_y_m: float = -0.28
    workpiece_z_m: float = 0.16
    desired_tool_axis_y: float = -1.0
    target_width_mm: float = 4.5
    target_depth_mm: float = 2.4
    max_trajectory_error_mm: float = 2.0
    collision_radius_m: float = 0.045
    workspace_radius_m: float = 0.82


# The first six variables are robot joints. The remaining variables are
# process/controller variables. Replace bounds with the calibrated robot data.
VARIABLES = [
    ("q1_rad", -np.pi, np.pi),
    ("q2_rad", -1.45, 1.45),
    ("q3_rad", -1.60, 1.60),
    ("q4_rad", -np.pi, np.pi),
    ("q5_rad", -1.45, 1.45),
    ("q6_rad", -np.pi, np.pi),
    ("current_A", 80.0, 220.0),
    ("voltage_V", 16.0, 32.0),
    ("travel_speed_mm_s", 1.0, 12.0),
    ("wire_feed_mm_s", 20.0, 160.0),
    ("torch_angle_deg", -20.0, 20.0),
    ("shielding_flow_L_min", 8.0, 30.0),
    ("stand_off_mm", 8.0, 20.0),
    ("weave_amplitude_mm", 0.0, 8.0),
]
LABELS = [name for name, _low, _high in VARIABLES]
BOUNDS = [(float(low), float(high)) for _name, low, high in VARIABLES]
JOINT_LOWER = np.asarray([b[0] for b in BOUNDS[:6]], dtype=float)
JOINT_UPPER = np.asarray([b[1] for b in BOUNDS[:6]], dtype=float)

ALGORITHM_REFERENCES = {
    "xde_sindy": "This work: interpretable XDE-SINDy",
    "gl_sade": "GL-SADE, IEEE TCYB 2022, DOI 10.1109/TCYB.2022.3175533 (reproducible global/local RBF-DE implementation)",
    "ego_gp": "EGO, Jones, Schonlau & Welch, Journal of Global Optimization 1998, DOI 10.1023/A:1008306431147",
    "turbo_lgp": "TuRBO, Eriksson et al., NeurIPS 2019 (single local trust-region implementation)",
    "random_search": "Budget-matched random-search sanity baseline",
}

# Double-click/IDE run configuration. Edit these constants when a different
# experiment is needed; no command-line switches are required.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_REPEATS = 20
RUN_EVALUATIONS = 300
RUN_ALGORITHMS = {"xde_sindy", "gl_sade", "ego_gp", "turbo_lgp", "random_search"}
RUN_CONFIG_PATH = PROJECT_ROOT / "configs" / "underwater_welding_digital_twin.json"
RUN_OUTPUT_PATH = PROJECT_ROOT / "data" / "underwater_welding_comparison.csv"


def _dh(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def _melt_dimensions(
    lateral: np.ndarray,
    depth: np.ndarray,
    temperature: np.ndarray,
    threshold: float,
) -> tuple[float, float]:
    """Estimate melt-pool width/depth from interpolated threshold crossings.

    The previous implementation reported the outermost grid cell classified
    as molten.  That made the engineering metrics jump in 2 mm/0.5 mm steps
    and produced a staircase objective for the surrogate.  Linear crossing
    interpolation retains the same temperature-field definition while
    returning continuous dimensions between grid nodes.
    """
    lateral = np.asarray(lateral, dtype=float)
    depth = np.asarray(depth, dtype=float)
    temperature = np.asarray(temperature, dtype=float)
    if temperature.shape != (len(depth), len(lateral)):
        raise ValueError("temperature must have shape (len(depth), len(lateral))")
    if float(np.max(temperature)) < threshold:
        return 0.0, 0.0

    def crossing(left_value: float, right_value: float, left_axis: float, right_axis: float) -> float:
        denominator = right_value - left_value
        if abs(denominator) <= 1e-12:
            return float((left_axis + right_axis) / 2.0)
        fraction = np.clip((threshold - left_value) / denominator, 0.0, 1.0)
        return float(left_axis + fraction * (right_axis - left_axis))

    width_m = 0.0
    for row in temperature:
        molten = np.flatnonzero(row >= threshold)
        if molten.size == 0:
            continue
        first, last = int(molten[0]), int(molten[-1])
        left = float(lateral[first])
        right = float(lateral[last])
        if first > 0:
            left = crossing(row[first - 1], row[first], lateral[first - 1], lateral[first])
        if last < len(lateral) - 1:
            right = crossing(row[last], row[last + 1], lateral[last], lateral[last + 1])
        width_m = max(width_m, right - left)

    depth_m = 0.0
    for column in temperature.T:
        molten = np.flatnonzero(column >= threshold)
        if molten.size == 0:
            continue
        last = int(molten[-1])
        bottom = float(depth[last])
        if last < len(depth) - 1:
            bottom = crossing(column[last], column[last + 1], depth[last], depth[last + 1])
        depth_m = max(depth_m, bottom)

    return float(width_m * 1000.0), float(depth_m * 1000.0)


class UnderwaterWeldingDigitalTwin:
    """One-segment robot/welding digital twin with calibrated interfaces."""

    # Compact six-axis arm geometry. Replace with the actual DH table when
    # the robot model is known; the rest of the simulation stays unchanged.
    _a = np.asarray([0.0, 0.30, 0.25, 0.0, 0.0, 0.0], dtype=float)
    _alpha = np.asarray([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0], dtype=float)
    _d = np.asarray([0.28, 0.0, 0.0, 0.18, 0.12, 0.10], dtype=float)

    def __init__(self, config: DigitalTwinConfig | None = None):
        self.config = config or DigitalTwinConfig()
        self._lateral = np.linspace(-0.014, 0.014, 29)
        self._depth = np.linspace(0.0, 0.012, 25)

    def forward_kinematics(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        transform = np.eye(4)
        points = [transform[:3, 3].copy()]
        for a, alpha, d, theta in zip(self._a, self._alpha, self._d, q):
            transform = transform @ _dh(a, alpha, d, theta)
            points.append(transform[:3, 3].copy())
        return transform, np.asarray(points)

    def _robot_constraints(self, q: np.ndarray, tip: np.ndarray, link_points: np.ndarray) -> dict[str, float]:
        cfg = self.config
        joint_violation = float(np.sum(np.maximum(JOINT_LOWER - q, 0.0) ** 2 + np.maximum(q - JOINT_UPPER, 0.0) ** 2))
        workspace_violation = float(max(0.0, np.linalg.norm(tip) - cfg.workspace_radius_m) ** 2)
        # The first point is the robot mounting origin, whose z=0 coordinate
        # is a frame reference rather than a moving link that can hit the
        # floor.  Exclude it from the link-clearance penalty.
        arm_points = link_points[1:]
        floor_violation = float(np.sum(np.maximum(0.025 - arm_points[:, 2], 0.0) ** 2))
        workpiece_axis = np.asarray([self.config.workpiece_x_m, self.config.workpiece_y_m, self.config.workpiece_z_m])
        # The final point is the torch tip, which is intentionally placed at
        # the weld path.  Its stand-off is handled by the process variable;
        # only the robot links are checked for workpiece collision.
        collision_points = link_points[:-1]
        radial = np.linalg.norm(collision_points[:, :2] - workpiece_axis[:2], axis=1)
        collision_violation = float(np.sum(np.maximum(cfg.collision_radius_m - radial, 0.0) ** 2))
        return {"joint_violation": joint_violation, "workspace_violation": workspace_violation,
                "floor_violation": floor_violation, "collision_violation": collision_violation}

    def _target_path(self, weave_amplitude_mm: float) -> np.ndarray:
        cfg = self.config
        s = np.linspace(0.0, 1.0, cfg.path_points)
        return np.column_stack([
            cfg.workpiece_x_m + cfg.weld_length_m * (s - 0.5),
            cfg.workpiece_y_m + weave_amplitude_mm * 1e-3 * np.sin(2.0 * np.pi * 2.0 * s),
            np.full_like(s, cfg.workpiece_z_m),
        ])

    def _trajectory_metrics(self, x: np.ndarray, transform: np.ndarray) -> dict[str, float]:
        cfg = self.config
        path = self._target_path(float(x[13]))
        tip = transform[:3, 3]
        s = np.linspace(0.0, 1.0, self.config.path_points)
        # The IK pose is the centre waypoint; the controller interpolates a
        # straight Cartesian segment over the weld length and superimposes the
        # commanded weave. This exposes both pose and trajectory error.
        commanded = tip[None, :] + np.column_stack([
            self.config.weld_length_m * (s - 0.5),
            float(x[13]) * 1e-3 * np.sin(2.0 * np.pi * 2.0 * s),
            np.zeros_like(s),
        ])
        position_error = np.linalg.norm(path - commanded, axis=1)
        tool_axis = transform[:3, 2]
        desired_axis = np.asarray([0.0, cfg.desired_tool_axis_y, 0.0])
        orientation_error = float(1.0 - np.clip(np.dot(tool_axis, desired_axis), -1.0, 1.0))
        return {"trajectory_error_mm": float(np.mean(position_error) * 1000.0),
                "trajectory_max_error_mm": float(np.max(position_error) * 1000.0),
                "orientation_error": orientation_error}

    def _thermal_metrics(self, x: np.ndarray) -> dict[str, float]:
        cfg = self.config
        current, voltage, speed_mm_s = float(x[6]), float(x[7]), float(x[8])
        wire, torch_angle, flow, stand_off = map(float, (x[9], x[10], x[11], x[12]))
        speed_m_s = max(speed_mm_s * 1e-3, 1e-5)
        power_w = cfg.arc_efficiency * current * voltage
        conductivity = max(cfg.thermal_conductivity_w_mk, 1e-6)
        diffusivity = conductivity / (cfg.density_kg_m3 * cfg.heat_capacity_j_kgk)
        lateral, depth = np.meshgrid(self._lateral, self._depth)
        distance = np.sqrt(lateral**2 + depth**2 + (0.45e-3 + stand_off * 1e-4) ** 2)
        delta_t = cfg.rosenthal_scale * power_w / (2.0 * np.pi * conductivity * distance)
        delta_t *= np.exp(-speed_m_s * np.maximum(lateral, 0.0) / (2.0 * diffusivity + 1e-12))
        water_cooling = 1.0 / (1.0 + cfg.water_convection_w_m2k * distance / (conductivity * 1000.0))
        temperature = cfg.ambient_temperature_c + delta_t * water_cooling
        width_mm, depth_mm = _melt_dimensions(
            self._lateral, self._depth, temperature, cfg.liquidus_temperature_c
        )
        heat_input_j_mm = power_w / max(speed_mm_s, 1e-6)
        wire_match = abs(wire - 0.62 * current) / max(0.62 * current, 1.0)
        voltage_match = abs(voltage - (18.0 + 0.35 * speed_mm_s)) / 12.0
        arc_instability = float(np.clip(0.55 * wire_match + 0.45 * voltage_match + 0.001 * abs(torch_angle), 0.0, 2.0))
        heat_ratio = heat_input_j_mm / 100.0
        porosity = float(np.clip(0.015 + 0.65 * np.exp(-flow / 16.0) + 0.18 * max(0.0, heat_ratio - 1.2) + 0.22 * max(0.0, 0.65 - heat_ratio), 0.0, 1.0))
        return {"power_w": float(power_w), "heat_input_j_mm": float(heat_input_j_mm),
                "melt_width_mm": width_mm, "melt_depth_mm": depth_mm,
                "arc_instability": arc_instability, "porosity_index": porosity,
                "water_cooling_factor": float(np.mean(water_cooling))}

    def evaluate(self, x: np.ndarray) -> dict[str, float]:
        x = np.asarray(x, dtype=float)
        transform, link_points = self.forward_kinematics(x[:6])
        robot = self._robot_constraints(x[:6], transform[:3, 3], link_points)
        trajectory = self._trajectory_metrics(x, transform)
        thermal = self._thermal_metrics(x)
        cfg = self.config
        quality_error = ((thermal["melt_width_mm"] - cfg.target_width_mm) / cfg.target_width_mm) ** 2
        depth_error = ((thermal["melt_depth_mm"] - cfg.target_depth_mm) / cfg.target_depth_mm) ** 2
        path_error = (trajectory["trajectory_error_mm"] / cfg.max_trajectory_error_mm) ** 2
        energy = thermal["power_w"] * cfg.weld_length_m / max(x[8] * 1e-3, 1e-5) / 10000.0
        stand_off_error = max(0.0, abs(x[12] - 13.0) - 5.0) ** 2 / 25.0
        constraint_violation = float(
            sum(robot.values())
            + max(0.0, trajectory["trajectory_max_error_mm"] - cfg.max_trajectory_error_mm) ** 2 / 100.0
        )
        score = float(3.0 * quality_error + 3.0 * depth_error + 2.0 * thermal["porosity_index"] ** 2
                      + 1.5 * thermal["arc_instability"] ** 2 + 1.5 * path_error + 0.5 * energy
                      + 0.4 * stand_off_error + 100.0 * constraint_violation)
        result = {"objective": score, "constraint_violation": constraint_violation}
        result.update(robot); result.update(trajectory); result.update(thermal)
        return result

    def objective(self, x: np.ndarray) -> float:
        return float(self.evaluate(x)["objective"])


def _normalise(x: np.ndarray) -> np.ndarray:
    lower = np.asarray([b[0] for b in BOUNDS], dtype=float)
    span = np.asarray([b[1] - b[0] for b in BOUNDS], dtype=float)
    return (np.asarray(x, dtype=float) - lower) / span


def _denormalise(z: np.ndarray) -> np.ndarray:
    lower = np.asarray([b[0] for b in BOUNDS], dtype=float)
    span = np.asarray([b[1] - b[0] for b in BOUNDS], dtype=float)
    return lower + np.asarray(z, dtype=float) * span


def _latin_hypercube(n: int, dimension: int, rng: np.random.Generator) -> np.ndarray:
    result = np.empty((n, dimension), dtype=float)
    for j in range(dimension):
        result[:, j] = (rng.permutation(n) + rng.random(n)) / n
    return result


def _evaluate_design(twin: UnderwaterWeldingDigitalTwin, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = _denormalise(z)
    y = np.asarray([twin.objective(point) for point in x], dtype=float)
    return np.asarray(z, dtype=float), y


def _fit_gp(z: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(
        length_scale=np.ones(z.shape[1]), length_scale_bounds="fixed", nu=2.5
    ) + WhiteKernel(noise_level=1e-6, noise_level_bounds="fixed")
    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, optimizer=None, random_state=17)
    model.fit(z, y)
    return model


def _expected_improvement(mean: np.ndarray, std: np.ndarray, incumbent: float) -> np.ndarray:
    std = np.maximum(np.asarray(std, dtype=float), 1e-12)
    improvement = incumbent - np.asarray(mean, dtype=float)
    z = improvement / std
    return improvement * ndtr(z) + std * np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def ego_gp(twin: UnderwaterWeldingDigitalTwin, rng: np.random.Generator, evaluations: int) -> tuple[float, np.ndarray]:
    """EGO-style GP + expected-improvement baseline (Jones et al., 1998)."""
    dimension = len(BOUNDS)
    n_init = min(evaluations, max(2 * dimension, 24))
    z, y = _evaluate_design(twin, _latin_hypercube(n_init, dimension, rng))
    batch = min(4, max(1, evaluations // 30))
    while len(y) < evaluations:
        model = _fit_gp(z, y)
        pool = rng.random((4096, dimension))
        mean, std = model.predict(pool, return_std=True)
        score = _expected_improvement(mean, std, float(np.min(y)))
        chosen = np.argsort(score)[-min(batch, evaluations - len(y)):]
        z_new, y_new = _evaluate_design(twin, pool[chosen])
        z, y = np.vstack([z, z_new]), np.concatenate([y, y_new])
    best = int(np.argmin(y))
    return float(y[best]), _denormalise(z[best])


def turbo_lgp(twin: UnderwaterWeldingDigitalTwin, rng: np.random.Generator, evaluations: int) -> tuple[float, np.ndarray]:
    """Single-trust-region TuRBO-style local GP baseline."""
    dimension = len(BOUNDS)
    n_init = min(evaluations, max(2 * dimension, 24))
    z, y = _evaluate_design(twin, _latin_hypercube(n_init, dimension, rng))
    length = 0.8
    batch = min(4, max(1, evaluations // 30))
    while len(y) < evaluations:
        incumbent = int(np.argmin(y))
        center = z[incumbent]
        distance = np.linalg.norm(z - center, axis=1)
        local = z[distance <= length]
        local_y = y[distance <= length]
        if len(local) < max(8, dimension + 2):
            nearest = np.argsort(distance)[:min(len(z), max(8, dimension + 2))]
            local, local_y = z[nearest], y[nearest]
        model = _fit_gp(local, local_y)
        low = np.maximum(0.0, center - length / 2.0)
        high = np.minimum(1.0, center + length / 2.0)
        pool = low + (high - low) * rng.random((4096, dimension))
        mean, std = model.predict(pool, return_std=True)
        score = _expected_improvement(mean, std, float(np.min(y)))
        chosen = np.argsort(score)[-min(batch, evaluations - len(y)):]
        z_new, y_new = _evaluate_design(twin, pool[chosen])
        previous_best = float(np.min(y))
        z, y = np.vstack([z, z_new]), np.concatenate([y, y_new])
        if float(np.min(y)) < previous_best:
            length = min(1.6, length * 1.25)
        else:
            length *= 0.75
            if length < 0.05:
                length = 0.8
    best = int(np.argmin(y))
    return float(y[best]), _denormalise(z[best])


def _rbf_predictor(z: np.ndarray, y: np.ndarray):
    neighbors = min(64, len(y))
    if len(y) < 4:
        return lambda query: np.full(len(query), float(np.mean(y)))

    def build(degree: int):
        return RBFInterpolator(
            z,
            y,
            kernel="thin_plate_spline",
            neighbors=neighbors,
            smoothing=1e-6,
            degree=degree,
        )

    try:
        # Keep SciPy's standard linear polynomial augmentation when the local
        # design has full rank. A locally selected population can become
        # rank-deficient in one or more coordinates, so prediction is guarded
        # and falls back to the same thin-plate kernel without polynomial
        # terms. This preserves the GL-SADE RBF model while avoiding a run
        # failure caused solely by a degenerate local sample geometry.
        model = build(degree=1)
    except (ValueError, np.linalg.LinAlgError):
        model = None

    def predict(query: np.ndarray) -> np.ndarray:
        nonlocal model
        if model is not None:
            try:
                return np.asarray(model(query), dtype=float).reshape(-1)
            except (ValueError, np.linalg.LinAlgError):
                model = None
        try:
            model = build(degree=-1)
            return np.asarray(model(query), dtype=float).reshape(-1)
        except (ValueError, np.linalg.LinAlgError):
            return np.full(len(query), float(np.mean(y)))

    return predict


def gl_sade(twin: UnderwaterWeldingDigitalTwin, rng: np.random.Generator, evaluations: int) -> tuple[float, np.ndarray]:
    """Global/local RBF-assisted DE baseline corresponding to GL-SADE."""
    dimension = len(BOUNDS)
    population_size = min(28, max(12, 2 * dimension))
    n_init = min(evaluations, population_size)
    population, fitness = _evaluate_design(twin, _latin_hypercube(n_init, dimension, rng))
    evaluated = len(fitness)
    while evaluated < evaluations:
        global_model = _rbf_predictor(population, fitness)
        best_index = int(np.argmin(fitness))
        distances = np.linalg.norm(population - population[best_index], axis=1)
        local_indices = np.argsort(distances)[:min(64, len(fitness))]
        local_model = _rbf_predictor(population[local_indices], fitness[local_indices])
        candidates = []
        for _ in range(max(4 * population_size, 64)):
            a, b, c = rng.choice(len(population), 3, replace=False)
            trial = population[a] + 0.65 * (population[b] - population[c])
            mask = rng.random(dimension) < 0.85
            mask[rng.integers(0, dimension)] = True
            trial = np.where(mask, trial, population[best_index])
            candidates.append(np.clip(trial, 0.0, 1.0))
        candidates = np.asarray(candidates)
        global_prediction = global_model(candidates)
        local_prediction = local_model(candidates)
        prediction = 0.55 * global_prediction + 0.45 * local_prediction
        chosen = np.argsort(prediction)[:min(4, evaluations - evaluated)]
        z_new, y_new = _evaluate_design(twin, candidates[chosen])
        evaluated += len(y_new)
        population = np.vstack([population, z_new])
        fitness = np.concatenate([fitness, y_new])
        keep = np.argsort(fitness)[:population_size]
        population, fitness = population[keep], fitness[keep]
    best = int(np.argmin(fitness))
    return float(fitness[best]), _denormalise(population[best])


def random_search(twin: UnderwaterWeldingDigitalTwin, rng: np.random.Generator, evaluations: int) -> tuple[float, np.ndarray]:
    lower = np.asarray([b[0] for b in BOUNDS]); upper = np.asarray([b[1] for b in BOUNDS])
    best_value = float("inf"); best_x = None
    for _ in range(evaluations):
        x = rng.uniform(lower, upper)
        value = twin.objective(x)
        if value < best_value:
            best_value, best_x = value, x.copy()
    return best_value, np.asarray(best_x)


def _run_xde(twin: UnderwaterWeldingDigitalTwin, rng: np.random.Generator, evaluations: int) -> tuple[float, np.ndarray, int]:
    optimizer = XDESindy(objective=twin.objective, bounds=BOUNDS, variable_labels=LABELS,
                          population_size=min(28, max(12, evaluations // 8)), max_evaluations=evaluations,
                          surrogate_start=min(48, max(16, evaluations // 4)), real_batch_size=4,
                          max_surrogate_samples=min(256, evaluations), surrogate_samples_per_dimension=2.0,
                          random_state=int(rng.integers(0, 2**31 - 1)), relative_improvement_mode="auto")
    result = optimizer.minimize()
    return float(result.fun), np.asarray(result.x), int(result.evaluations)


def _with_budget(result: tuple[float, np.ndarray], evaluations: int) -> tuple[float, np.ndarray, int]:
    value, point = result
    return float(value), np.asarray(point), int(evaluations)


def load_config(path: str | None) -> DigitalTwinConfig:
    if not path:
        return DigitalTwinConfig()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    valid = set(DigitalTwinConfig.__dataclass_fields__)
    unknown = sorted(set(payload) - valid)
    if unknown:
        raise ValueError(f"Unknown digital-twin parameters: {unknown}")
    return DigitalTwinConfig(**payload)


def run(repeats: int, evaluations: int, output: Path, algorithms: set[str], config: DigitalTwinConfig) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    twin = UnderwaterWeldingDigitalTwin(config)
    rows = []
    for repeat in range(1, repeats + 1):
        seed = 7000 + repeat
        methods = {
            "xde_sindy": lambda rng: _run_xde(twin, rng, evaluations),
            "gl_sade": lambda rng: _with_budget(gl_sade(twin, rng, evaluations), evaluations),
            "ego_gp": lambda rng: _with_budget(ego_gp(twin, rng, evaluations), evaluations),
            "turbo_lgp": lambda rng: _with_budget(turbo_lgp(twin, rng, evaluations), evaluations),
            "random_search": lambda rng: _with_budget(random_search(twin, rng, evaluations), evaluations),
        }
        for algorithm in sorted(algorithms):
            if algorithm not in methods:
                raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from {sorted(methods)}")
            started = time.perf_counter()
            value, best_x, used_evaluations = methods[algorithm](np.random.default_rng(seed))
            metrics = twin.evaluate(best_x)
            rows.append({"algorithm": algorithm, "repeat": repeat, "best_f": value,
                         "evaluations": used_evaluations, "seconds": time.perf_counter() - started,
                         "constraint_violation": metrics["constraint_violation"], "melt_width_mm": metrics["melt_width_mm"],
                         "melt_depth_mm": metrics["melt_depth_mm"], "porosity_index": metrics["porosity_index"],
                         "trajectory_error_mm": metrics["trajectory_error_mm"], "best_x": json.dumps(best_x.tolist())})
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    output.with_name(output.stem + "_model.json").write_text(json.dumps({
        "model": "reduced_order_underwater_welding_digital_twin", "calibration_required": True,
        "config": asdict(config), "variables": [{"name": n, "lower": lo, "upper": hi} for n, lo, hi in VARIABLES],
        "algorithms": {name: ALGORITHM_REFERENCES[name] for name in sorted(algorithms)},
        "evaluation_protocol": "Each algorithm receives the same true-FE budget; algorithms share a seed within each repeat for paired tests, while repeats use independent seeds; surrogate evaluations are free.",
    }, indent=2), encoding="utf-8")


def main() -> None:
    run(repeats=RUN_REPEATS, evaluations=RUN_EVALUATIONS, output=RUN_OUTPUT_PATH,
        algorithms=RUN_ALGORITHMS, config=load_config(str(RUN_CONFIG_PATH)))
    print(f"Wrote digital-twin simulation results to {RUN_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
