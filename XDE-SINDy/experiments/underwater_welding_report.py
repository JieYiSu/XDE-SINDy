"""Generate statistical tables and publication figures for the welding study.

The script is intentionally parameter-free for direct IDE execution.  It
reads ``code/data/underwater_welding_comparison.csv`` and writes all tables
and figures to ``code/data/underwater_welding_report``.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_INPUT_PATH = PROJECT_ROOT / "data" / "underwater_welding_comparison.csv"
RUN_OUTPUT_DIR = PROJECT_ROOT / "data" / "underwater_welding_report"

ALGORITHM_ORDER = ["xde_sindy", "gl_sade", "turbo_lgp", "ego_gp", "random_search"]
ALGORITHM_LABELS = {
    "xde_sindy": "XDE-SINDy",
    "gl_sade": "GL-SADE",
    "turbo_lgp": "TuRBO-LGP",
    "ego_gp": "EGO-GP",
    "random_search": "Random Search",
}
COLORS = {
    # Palette extracted from the supplied convergence figure:
    # XDE-SINDy, SADE-SS, SGDLCO, SAGPE, and BGP-SAEA.
    # The five underwater methods keep the same left-to-right color order.
    "xde_sindy": "#10739C",
    "gl_sade": "#08ADE7",
    "turbo_lgp": "#B5DE8C",
    "ego_gp": "#31A529",
    "random_search": "#FF9C9C",
}
METRICS = [
    "best_f", "constraint_violation", "melt_width_mm", "melt_depth_mm",
    "trajectory_error_mm", "porosity_index",
]
LOWER_IS_BETTER = True


def _holm(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    if not p_values.size:
        return p_values.copy()
    order = np.argsort(p_values, kind="stable")
    sorted_values = p_values[order]
    adjusted_sorted = np.maximum.accumulate(
        (len(sorted_values) - np.arange(len(sorted_values))) * sorted_values
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def _paired_p(x: np.ndarray, y: np.ndarray) -> float:
    difference = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    if np.allclose(difference, 0.0):
        return 1.0
    try:
        return float(wilcoxon(x, y, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def _read(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"algorithm", "repeat", *METRICS, "best_x"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    frame["repeat"] = frame["repeat"].astype(int)
    frame["algorithm"] = pd.Categorical(
        frame["algorithm"], categories=ALGORITHM_ORDER, ordered=True
    )
    return frame.sort_values(["repeat", "algorithm"]).reset_index(drop=True)


def _write(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _statistics(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    algorithms = [a for a in ALGORITHM_ORDER if (frame["algorithm"] == a).any()]
    metrics = [m for m in METRICS if m in frame.columns]

    median_rows = []
    for algorithm in algorithms:
        subset = frame[frame["algorithm"] == algorithm]
        row = {"algorithm": ALGORITHM_LABELS[algorithm], "n": len(subset)}
        for metric in metrics:
            q1, median, q3 = subset[metric].quantile([0.25, 0.5, 0.75])
            row[f"{metric}_median"] = float(median)
            row[f"{metric}_q1"] = float(q1)
            row[f"{metric}_q3"] = float(q3)
            row[f"{metric}_iqr"] = float(q3 - q1)
        row["strict_feasible_rate"] = float(np.mean(subset["constraint_violation"] <= 1e-9))
        row["practical_feasible_rate_cv_le_1"] = float(np.mean(subset["constraint_violation"] <= 1.0))
        median_rows.append(row)
    median_iqr = pd.DataFrame(median_rows)

    wide = frame.pivot(index="repeat", columns="algorithm", values=metrics)
    # pandas creates a metric/algorithm MultiIndex. Keep only complete paired blocks.
    complete = wide.dropna()
    rank_rows = []
    for repeat, values in complete.iterrows():
        row = {"repeat": int(repeat)}
        for metric in metrics:
            data = np.asarray([values[(metric, algorithm)] for algorithm in algorithms], dtype=float)
            ranks = rankdata(data, method="average")
            for algorithm, rank in zip(algorithms, ranks):
                row[f"{metric}_{algorithm}_rank"] = float(rank)
        rank_rows.append(row)
    rank_by_repeat = pd.DataFrame(rank_rows)

    average_rows = []
    for metric in metrics:
        rank_matrix = np.asarray([
            [rankdata(
                [complete.loc[repeat, (metric, algorithm)] for algorithm in algorithms],
                method="average",
            )[i] for i in range(len(algorithms))]
            for repeat in complete.index
        ], dtype=float)
        winners = np.argmin(
            complete[metric].to_numpy(dtype=float), axis=1
        )
        for index, algorithm in enumerate(algorithms):
            average_rows.append({
                "metric": metric,
                "algorithm": ALGORITHM_LABELS[algorithm],
                "average_rank": float(np.mean(rank_matrix[:, index])),
                "median_rank": float(np.median(rank_matrix[:, index])),
                "rank1_count": int(np.sum(winners == index)),
                "rank1_rate": float(np.mean(winners == index)),
            })
    average_ranks = pd.DataFrame(average_rows)

    friedman_rows = []
    for metric in metrics:
        columns = [complete[(metric, algorithm)].to_numpy(dtype=float) for algorithm in algorithms]
        statistic, p_value = friedmanchisquare(*columns)
        rank_matrix = np.asarray([
            rankdata([complete.loc[repeat, (metric, a)] for a in algorithms], method="average")
            for repeat in complete.index
        ])
        row = {
            "metric": metric,
            "n_blocks": len(complete),
            "friedman_chi2": float(statistic),
            "p_raw": float(p_value),
        }
        for index, algorithm in enumerate(algorithms):
            row[f"{algorithm}_average_rank"] = float(np.mean(rank_matrix[:, index]))
        friedman_rows.append(row)
    friedman = pd.DataFrame(friedman_rows)
    friedman["p_holm_metrics"] = _holm(friedman["p_raw"].to_numpy())
    friedman["sign_holm_metrics"] = friedman["p_holm_metrics"] < 0.05

    pair_rows = []
    for metric in metrics:
        for first, second in combinations(algorithms, 2):
            x = complete[(metric, first)].to_numpy(dtype=float)
            y = complete[(metric, second)].to_numpy(dtype=float)
            pair_rows.append({
                "metric": metric,
                "algorithm_a": ALGORITHM_LABELS[first],
                "algorithm_b": ALGORITHM_LABELS[second],
                "n_pairs": len(x),
                "median_delta_a_minus_b": float(np.median(x - y)),
                "p_raw": _paired_p(x, y),
            })
    holm = pd.DataFrame(pair_rows)
    # Correct each metric's family of pairwise tests separately.  This keeps
    # the objective comparison interpretable without letting unrelated
    # engineering metrics make the objective correction unnecessarily harsh.
    holm["p_holm"] = np.nan
    for metric, indices in holm.groupby("metric", sort=False).groups.items():
        holm.loc[indices, "p_holm"] = _holm(holm.loc[indices, "p_raw"].to_numpy())
    holm["sign_holm"] = holm["p_holm"] < 0.05

    return {
        "median_iqr": median_iqr,
        "average_ranks": average_ranks,
        "friedman": friedman,
        "holm": holm,
        "rank_by_repeat": rank_by_repeat,
    }


def _save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _trajectory_figure(frame: pd.DataFrame, output: Path) -> None:
    # Import the same digital-twin kinematics used to produce the CSV.
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    import underwater_welding_simulation as simulation

    twin = simulation.UnderwaterWeldingDigitalTwin(
        simulation.load_config(str(simulation.RUN_CONFIG_PATH))
    )
    best = frame.loc[frame.groupby("algorithm", observed=True)["best_f"].idxmin()]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=False, sharey=False)
    for axis, projection in zip(axes, ("xy", "xz")):
        target = twin._target_path(0.0) * 1000.0
        axis.plot(
            target[:, 0] if projection == "xy" else target[:, 0],
            target[:, 1] if projection == "xy" else target[:, 2],
            "k--", linewidth=1.6, label="Target path",
        )
        for algorithm in ALGORITHM_ORDER:
            if algorithm not in set(frame["algorithm"].astype(str)):
                continue
            row = best[best["algorithm"].astype(str) == algorithm].iloc[0]
            point = np.asarray(json.loads(row["best_x"]), dtype=float)
            transform, _ = twin.forward_kinematics(point[:6])
            s = np.linspace(0.0, 1.0, twin.config.path_points)
            tip = transform[:3, 3]
            commanded = tip[None, :] + np.column_stack([
                twin.config.weld_length_m * (s - 0.5),
                np.zeros_like(s),
                np.zeros_like(s),
            ])
            commanded *= 1000.0
            axis.plot(
                commanded[:, 0],
                commanded[:, 1] if projection == "xy" else commanded[:, 2],
                color=COLORS[algorithm], linewidth=1.4,
                label=ALGORITHM_LABELS[algorithm],
            )
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)" if projection == "xy" else "z (mm)")
        axis.grid(alpha=0.2)
        axis.set_title("XY projection" if projection == "xy" else "XZ projection")
    axes[0].legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle("Best-run torch trajectories (n=20 per algorithm)", fontsize=10)
    fig.tight_layout()
    _save_figure(fig, output, "underwater_welding_trajectory_map")


def _metric_figure(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))
    plot_specs = [
        ("best_f", "log10(best_f + 1)", True),
        ("constraint_violation", "Constraint violation", False),
        ("trajectory_error_mm", "Trajectory error (mm)", False),
        ("porosity_index", "Porosity index", False),
    ]
    for axis, (metric, label, log_transform) in zip(axes.flat, plot_specs):
        values = []
        labels = []
        for algorithm in ALGORITHM_ORDER:
            subset = frame[frame["algorithm"] == algorithm][metric].to_numpy(dtype=float)
            if log_transform:
                subset = np.log10(subset + 1.0)
            values.append(subset)
            labels.append(ALGORITHM_LABELS[algorithm])
        box = axis.boxplot(values, patch_artist=True, widths=0.58, showfliers=False)
        for patch, algorithm in zip(box["boxes"], ALGORITHM_ORDER):
            patch.set_facecolor(COLORS[algorithm])
            patch.set_alpha(0.65)
        for index, (data, algorithm) in enumerate(zip(values, ALGORITHM_ORDER), start=1):
            jitter = np.linspace(-0.08, 0.08, len(data))
            axis.scatter(index + jitter, data, s=10, color=COLORS[algorithm], alpha=0.7, edgecolor="none")
        axis.set_xticks(range(1, len(labels) + 1), labels, rotation=35, ha="right", fontsize=7)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Engineering and objective distributions (20 paired runs)", fontsize=10)
    fig.tight_layout()
    _save_figure(fig, output, "underwater_welding_metric_distributions")


def _rank_figure(stats: dict[str, pd.DataFrame], output: Path) -> None:
    ranks = stats["average_ranks"]
    subset = ranks[ranks["metric"] == "best_f"].copy()
    subset = subset.sort_values("average_rank")
    fig, axis = plt.subplots(figsize=(5.2, 3.0))
    bars = axis.bar(
        subset["algorithm"], subset["average_rank"],
        color=[COLORS[next(a for a, label in ALGORITHM_LABELS.items() if label == x)] for x in subset["algorithm"]],
    )
    axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    axis.set_ylabel("Average rank (lower is better)")
    axis.set_title("Average rank by composite objective")
    axis.tick_params(axis="x", rotation=30, labelsize=8)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save_figure(fig, output, "underwater_welding_average_rank")


def _repeat_figure(frame: pd.DataFrame, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.2, 3.4))
    for algorithm in ALGORITHM_ORDER:
        subset = frame[frame["algorithm"] == algorithm].sort_values("repeat")
        axis.plot(
            subset["repeat"], np.log10(subset["best_f"] + 1.0),
            color=COLORS[algorithm], linewidth=1.2, marker="o", markersize=3.5,
            label=ALGORITHM_LABELS[algorithm],
        )
    axis.set_xlabel("Repeat")
    axis.set_ylabel("log10(best_f + 1)")
    axis.set_title("Paired repeated-run objective values")
    axis.set_xticks(sorted(frame["repeat"].unique()))
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    _save_figure(fig, output, "underwater_welding_repeat_objectives")


def build_report(input_path: str | Path = RUN_INPUT_PATH, output_dir: str | Path = RUN_OUTPUT_DIR) -> dict[str, pd.DataFrame]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = _read(input_path)
    stats = _statistics(frame)
    for name, table in stats.items():
        _write(table, output / f"underwater_welding_{name}.csv")
    _trajectory_figure(frame, output)
    _metric_figure(frame, output)
    _rank_figure(stats, output)
    _repeat_figure(frame, output)
    metadata = {
        "source": str(Path(input_path).resolve()),
        "algorithms": [ALGORITHM_LABELS[a] for a in ALGORITHM_ORDER if (frame["algorithm"] == a).any()],
        "repeats_per_algorithm": int(frame.groupby("algorithm", observed=True).size().min()),
        "paired_test": "two-sided Wilcoxon signed-rank",
        "multiple_comparison_correction": "Holm step-down",
        "friedman_lower_is_better": LOWER_IS_BETTER,
        "figures": [
            "underwater_welding_trajectory_map.png/pdf",
            "underwater_welding_metric_distributions.png/pdf",
            "underwater_welding_average_rank.png/pdf",
            "underwater_welding_repeat_objectives.png/pdf",
        ],
    }
    (output / "underwater_welding_report_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return stats


def main() -> None:
    stats = build_report()
    global_row = stats["friedman"].loc[stats["friedman"]["metric"] == "best_f"].iloc[0]
    print(
        f"Wrote underwater welding report to {RUN_OUTPUT_DIR}; "
        f"Friedman chi2={global_row['friedman_chi2']:.4f}, p={global_row['p_raw']:.4g}"
    )


if __name__ == "__main__":
    main()
