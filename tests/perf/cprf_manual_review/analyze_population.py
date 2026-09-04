#!/usr/bin/env python3
"""Cluster high-confidence Real cases and characterize the full candidate population."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.preprocessing import RobustScaler


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parents[2]
EXPECTED_REVIEW_CASES = 1993
DEFAULT_BATCH = PROJECT_DIR / "runs/batch_20260823_140527/candidates_reviewed.all.csv"
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "runs/batch_20260823_140527/analysis/cprf_population_v1"
)
SEED = 20260903

INK = "#172121"
PAPER = "#f7f3e8"
GRID = "#d7d0c0"
HIGH_REAL = "#0f766e"
OTHER_REAL = "#d19a26"
PURE_FP = "#cf5c3f"
POPULATION = "#6f7c80"
CLUSTER_COLORS = (
    "#005f73",
    "#ca6702",
    "#0a9396",
    "#bb3e03",
    "#6a4c93",
    "#588157",
    "#9b2226",
    "#3d5a80",
)

NUMERIC_COLUMNS = (
    "channel",
    "freq_mhz",
    "t0_rec",
    "t1_rec",
    "dur_rec",
    "period_rec",
    "p_span_rec",
    "p_bins",
    "noise_sigma",
    "cpro_thr",
    "cprf_thr",
    "shape_mean",
    "shape_max",
    "pelt_z_mean",
    "pelt_z_max",
    "ridge_peak",
    "ridge_int",
    "band_conc",
    "band_persist",
    "local_contrast",
    "h2",
    "h3",
    "harm_n",
    "core_score",
    "score",
)
MODEL_FEATURES = (
    "period_rec",
    "dur_rec",
    "band_conc",
    "band_persist",
    "local_contrast",
    "ridge_int",
    "h2",
    "h3",
    "shape_fill",
)
BASIC_FEATURES = (
    "freq_mhz",
    "period_rec",
    "dur_rec",
    "p_span_rec",
    "p_bins",
    "noise_sigma",
    "shape_mean",
    "shape_max",
    "pelt_z_mean",
    "pelt_z_max",
    "ridge_peak",
    "ridge_int",
    "band_conc",
    "band_persist",
    "local_contrast",
    "score",
)
SECONDARY_FEATURES = (
    "cpro_thr",
    "cprf_thr",
    "shape_fill",
    "h2",
    "h3",
    "harm_n",
    "core_score",
)
LOG_FEATURES = {
    "period_rec",
    "dur_rec",
    "p_span_rec",
    "noise_sigma",
    "cpro_thr",
    "cprf_thr",
    "shape_mean",
    "shape_max",
    "pelt_z_mean",
    "pelt_z_max",
    "ridge_peak",
    "ridge_int",
    "local_contrast",
    "core_score",
    "score",
}
FEATURE_LABELS = {
    "freq_mhz": "Frequency (MHz)",
    "period_rec": "Period (records)",
    "dur_rec": "Duration (records)",
    "p_span_rec": "Period-band span (records)",
    "p_bins": "Period-band bins",
    "noise_sigma": "First-difference noise sigma",
    "cpro_thr": "CPRO normalization threshold",
    "cprf_thr": "CPRF normalization threshold",
    "shape_mean": "CPRO shape mean",
    "shape_max": "CPRO shape peak",
    "pelt_z_mean": "PELT z mean",
    "pelt_z_max": "PELT z peak",
    "band_conc": "Band concentration",
    "band_persist": "Band persistence",
    "local_contrast": "Local contrast",
    "ridge_peak": "Ridge peak",
    "ridge_int": "Integrated strength",
    "h2": "2f response / main",
    "h3": "3f response / main",
    "harm_n": "Supported harmonics",
    "core_score": "CPRF core score",
    "score": "Exported score",
    "shape_fill": "CPRO mean / peak",
}
GROUP_COLORS = {
    "High-confidence Real": HIGH_REAL,
    "Other Real": OTHER_REAL,
    "Pure FP": PURE_FP,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--selection", type=Path, default=BASE_DIR / "selection.csv")
    parser.add_argument("--labels", type=Path, default=BASE_DIR / "labels.csv")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs/cwt_default.json")
    parser.add_argument("--review-images", type=Path, default=BASE_DIR / "artifacts/review")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot-sample", type=int, default=60_000)
    parser.add_argument("--min-clusters", type=int, default=2)
    parser.add_argument("--max-clusters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": "#fffdf7",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.45,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "xtick.color": INK,
            "ytick.color": INK,
            "savefig.facecolor": PAPER,
            "savefig.bbox": "tight",
        }
    )


def load_population(path: Path) -> pd.DataFrame:
    columns = ["run_id", "candidate_id", "candidate_status", *NUMERIC_COLUMNS]
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    if frame.empty:
        raise RuntimeError(f"candidate table is empty: {path}")
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["shape_fill"] = frame["shape_mean"] / np.maximum(frame["shape_max"], 1e-12)
    frame["candidate_key"] = frame["run_id"].astype(str) + "\x1f" + frame["candidate_id"]
    if frame["candidate_key"].duplicated().any():
        raise RuntimeError("full candidate keys are not unique")
    if not np.isfinite(frame[list(MODEL_FEATURES)].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("full candidate feature matrix contains non-finite values")
    return frame


def load_manual(selection_path: Path, labels_path: Path) -> pd.DataFrame:
    selection = pd.read_csv(selection_path, low_memory=False)
    with labels_path.open(newline="", encoding="utf-8") as handle:
        labels = list(csv.DictReader(handle))
    if len(selection) != EXPECTED_REVIEW_CASES or len(labels) != EXPECTED_REVIEW_CASES:
        raise RuntimeError("analysis requires the complete fixed 1,993-case review set")
    annotations = {row["raw_key"]: json.loads(row["intervals"]) for row in labels}
    if set(selection["raw_key"]) != set(annotations):
        raise RuntimeError("selection and labels do not contain the same raw keys")

    groups: list[str] = []
    case_types: list[str] = []
    interval_counts: list[int] = []
    for key in selection["raw_key"]:
        items = annotations[str(key)]
        kinds = {str(item.get("label", "")) for item in items}
        if not items or "" in kinds or not kinds <= {"keep", "fp", "uncertain"}:
            raise RuntimeError(f"incomplete or invalid manual label: {key}")
        high_real = any(
            item.get("label") == "keep" and item.get("conf") == "high" for item in items
        )
        any_real = any(item.get("label") == "keep" for item in items)
        pure_fp = kinds == {"fp"}
        if high_real:
            group = "High-confidence Real"
        elif any_real:
            group = "Other Real"
        elif pure_fp:
            group = "Pure FP"
        else:
            raise RuntimeError(f"unsupported manual label combination: {key} {sorted(kinds)}")
        groups.append(group)
        case_types.append(next(iter(kinds)) if len(kinds) == 1 else "mixed")
        interval_counts.append(len(items))

    selection["manual_group"] = groups
    selection["case_type"] = case_types
    selection["interval_count"] = interval_counts
    selection["candidate_id"] = selection["candidate_id"].astype(str)
    selection["candidate_key"] = (
        selection["run_id"].astype(str) + "\x1f" + selection["candidate_id"]
    )
    for column in NUMERIC_COLUMNS:
        selection[column] = pd.to_numeric(selection[column], errors="raise")
    selection["shape_fill"] = selection["shape_mean"] / np.maximum(
        selection["shape_max"], 1e-12
    )
    return selection


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        (
            np.log10(np.maximum(frame["period_rec"].to_numpy(dtype=np.float64), 1e-12)),
            np.log10(np.maximum(frame["dur_rec"].to_numpy(dtype=np.float64), 1e-12)),
            frame["band_conc"].to_numpy(dtype=np.float64),
            frame["band_persist"].to_numpy(dtype=np.float64),
            np.log10(1.0 + np.maximum(frame["local_contrast"].to_numpy(dtype=np.float64), 0.0)),
            np.log10(1.0 + np.maximum(frame["ridge_int"].to_numpy(dtype=np.float64), 0.0)),
            frame["h2"].to_numpy(dtype=np.float64),
            frame["h3"].to_numpy(dtype=np.float64),
            frame["shape_fill"].to_numpy(dtype=np.float64),
        )
    )


def fit_clusters(
    high: pd.DataFrame,
    population: pd.DataFrame,
    minimum: int,
    maximum: int,
    seed: int,
) -> dict[str, object]:
    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    high_z = scaler.fit_transform(feature_matrix(high))
    population_z = scaler.transform(feature_matrix(population))
    scores: list[dict[str, object]] = []
    models: dict[int, KMeans] = {}
    upper = min(maximum, len(high) - 1)
    for clusters in range(max(2, minimum), upper + 1):
        model = KMeans(n_clusters=clusters, n_init=50, random_state=seed).fit(high_z)
        sizes = np.bincount(model.labels_, minlength=clusters)
        score = float(silhouette_score(high_z, model.labels_))
        scores.append({"clusters": clusters, "silhouette": score, "sizes": sizes.tolist()})
        models[clusters] = model
    if not scores:
        raise RuntimeError("cluster range does not contain a valid solution")
    chosen = max(scores, key=lambda row: float(row["silhouette"]))
    model = models[int(chosen["clusters"])]

    raw_high_labels = model.labels_
    harmonic_medians = [
        float(
            np.median(
                high.iloc[np.flatnonzero(raw_high_labels == cluster)][["h2", "h3"]].sum(axis=1)
            )
        )
        for cluster in range(model.n_clusters)
    ]
    raw_order = np.argsort(harmonic_medians)
    remap = np.empty(model.n_clusters, dtype=np.int64)
    for new_label, raw_label in enumerate(raw_order, 1):
        remap[int(raw_label)] = new_label
    high_labels = remap[raw_high_labels]
    population_raw_labels = model.predict(population_z)
    population_labels = remap[population_raw_labels]
    population_distances = model.transform(population_z)[
        np.arange(len(population_z)), population_raw_labels
    ]
    high_distances = model.transform(high_z)[np.arange(len(high_z)), raw_high_labels]
    radii = {
        cluster: float(np.quantile(high_distances[high_labels == cluster], 0.95))
        for cluster in range(1, model.n_clusters + 1)
    }
    within = np.asarray(
        [
            distance <= radii[int(cluster)]
            for distance, cluster in zip(population_distances, population_labels)
        ],
        dtype=bool,
    )
    pca = PCA(n_components=2, random_state=seed).fit(high_z)
    return {
        "high_z": high_z,
        "population_z": population_z,
        "high_pca": pca.transform(high_z),
        "population_pca": pca.transform(population_z),
        "high_labels": high_labels,
        "population_labels": population_labels,
        "high_distances": high_distances,
        "population_distances": population_distances,
        "within": within,
        "radii": radii,
        "scores": scores,
        "chosen": chosen,
    }


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=190)
    plt.close(fig)


def clipped_values(frame: pd.DataFrame, feature: str, low: float, high: float) -> np.ndarray:
    values = frame[feature].to_numpy(dtype=np.float64)
    return values[np.isfinite(values) & (values >= low) & (values <= high)]


def distribution_panel(
    ax: plt.Axes,
    population: pd.DataFrame,
    manual: pd.DataFrame,
    feature: str,
) -> None:
    all_values = population[feature].to_numpy(dtype=np.float64)
    finite = all_values[np.isfinite(all_values)]
    low, high = np.quantile(finite, [0.001, 0.999])
    if high <= low:
        low, high = float(np.min(finite)), float(np.max(finite)) + 1.0
    if feature in {"p_bins", "harm_n"}:
        minimum = int(np.floor(low))
        maximum = int(np.ceil(high))
        bins = np.arange(minimum - 0.5, maximum + 1.5)
    elif feature in LOG_FEATURES and low > 0:
        bins = np.geomspace(low, high, 60)
    else:
        bins = np.linspace(low, high, 60)
    ax.hist(
        clipped_values(population, feature, low, high),
        bins=bins,
        density=True,
        color=POPULATION,
        alpha=0.34,
        label="Full population",
    )
    for group, color in (("High-confidence Real", HIGH_REAL), ("Pure FP", PURE_FP)):
        values = clipped_values(manual[manual["manual_group"] == group], feature, low, high)
        if values.size:
            ax.hist(values, bins=bins, density=True, histtype="step", linewidth=1.7, color=color, label=group)
    if feature in LOG_FEATURES and low > 0:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(low, high)
    ax.set_title(FEATURE_LABELS[feature], fontsize=10.5)
    ax.set_ylabel("Density")


def plot_basic_distributions(
    population: pd.DataFrame,
    manual: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(18, 15.5))
    fig.suptitle("Basic candidate distributions", fontsize=19, fontweight="bold", y=0.985)
    for ax, feature in zip(axes.flat, BASIC_FEATURES):
        distribution_panel(ax, population, manual, feature)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.subplots_adjust(
        left=0.055, right=0.99, top=0.94, bottom=0.075,
        hspace=0.34, wspace=0.20,
    )
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.012),
        ncol=3, frameon=False,
    )
    save_figure(fig, output / "01_basic_candidate_distributions.png")

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    fig.suptitle("Threshold, harmonic and score distributions", fontsize=19, fontweight="bold")
    for ax, feature in zip(axes.flat, SECONDARY_FEATURES):
        distribution_panel(ax, population, manual, feature)
    axes.flat[-1].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False)
    save_figure(fig, output / "02_threshold_harmonic_score_distributions.png")


def plot_population(
    population: pd.DataFrame,
    manual: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    fig.suptitle("Full batch candidate population", fontsize=18, fontweight="bold")
    hb = axes[0, 0].hexbin(
        population["freq_mhz"], population["period_rec"], gridsize=(90, 55),
        bins="log", mincnt=1, cmap="cividis",
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set(xlabel="Frequency (MHz)", ylabel="Candidate period (records)")
    axes[0, 0].set_title("Where exported candidates occur")
    fig.colorbar(hb, ax=axes[0, 0], label="log10 candidates / hex")

    hb = axes[0, 1].hexbin(
        population["period_rec"], population["dur_rec"], gridsize=(65, 65),
        bins="log", mincnt=1, cmap="magma_r", xscale="log", yscale="log",
    )
    axes[0, 1].set(xlabel="Candidate period (records)", ylabel="Candidate duration (records)")
    axes[0, 1].set_title("Period-duration morphology")
    fig.colorbar(hb, ax=axes[0, 1], label="log10 candidates / hex")

    run_counts = population.groupby("run_id").size().sort_values(ascending=False)
    axes[1, 0].hist(run_counts, bins=35, color="#577590", edgecolor="white")
    axes[1, 0].axvline(
        run_counts.median(), color="#9b2226", linestyle="--",
        label=f"median {run_counts.median():.0f}",
    )
    axes[1, 0].set(xlabel="Candidates per observation", ylabel="Observations")
    axes[1, 0].set_title(f"Observation multiplicity (n={len(run_counts)})")
    axes[1, 0].legend(frameon=False)

    bins = np.linspace(
        float(population["freq_mhz"].min()), float(population["freq_mhz"].max()), 90
    )
    axes[1, 1].hist(
        population["freq_mhz"], bins=bins, density=True, histtype="stepfilled",
        alpha=0.35, color=POPULATION, label=f"All candidates ({len(population):,})",
    )
    high = manual[manual["manual_group"] == "High-confidence Real"]
    axes[1, 1].hist(
        high["freq_mhz"], bins=bins, density=True, histtype="step",
        linewidth=2.0, color=HIGH_REAL, label=f"High-confidence Real ({len(high):,})",
    )
    axes[1, 1].set(xlabel="Frequency (MHz)", ylabel="Normalized density")
    axes[1, 1].set_title("Frequency distribution and review support")
    axes[1, 1].legend(frameon=False)
    save_figure(fig, output / "03_population_overview.png")


def ecdf(ax: plt.Axes, values: np.ndarray, *, label: str, color: str) -> None:
    values = np.sort(values[np.isfinite(values)])
    if values.size:
        ax.plot(
            values, np.arange(1, values.size + 1) / values.size,
            label=label, color=color, linewidth=1.8,
        )


def plot_manual_distributions(manual: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(16, 13.5))
    fig.suptitle(
        "Manual-review feature distributions", fontsize=18, fontweight="bold", y=0.985
    )
    for ax, feature in zip(axes.flat, MODEL_FEATURES):
        for group, color in GROUP_COLORS.items():
            ecdf(
                ax,
                manual.loc[manual["manual_group"] == group, feature].to_numpy(dtype=np.float64),
                label=group,
                color=color,
            )
        if feature in {"period_rec", "dur_rec", "local_contrast", "ridge_int"}:
            ax.set_xscale("log")
        values = manual[feature].to_numpy(dtype=np.float64)
        low, high = np.quantile(values, [0.002, 0.998])
        if feature not in {"band_conc", "band_persist", "h2", "h3", "shape_fill"}:
            ax.set_xlim(max(low, np.nextafter(0.0, 1.0)), high)
        ax.set(title=FEATURE_LABELS[feature], ylabel="Empirical CDF")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.subplots_adjust(
        left=0.06, right=0.99, top=0.94, bottom=0.08,
        hspace=0.32, wspace=0.18,
    )
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.015),
        ncol=3, frameon=False,
    )
    save_figure(fig, output / "04_manual_feature_distributions.png")


def plot_score_plane(
    population: pd.DataFrame,
    manual: pd.DataFrame,
    detection: dict[str, object],
    output: Path,
) -> None:
    max_contrast = 20.0
    visible = population[population["local_contrast"] <= max_contrast]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    fig.suptitle("CPRF concentration-contrast score plane", fontsize=18, fontweight="bold")
    hb = axes[0].hexbin(
        visible["band_conc"], visible["local_contrast"], gridsize=(80, 70),
        bins="log", mincnt=1, cmap="Greys", linewidths=0,
    )
    fig.colorbar(hb, ax=axes[0], label="log10 candidates / hex")
    axes[0].text(
        0.02, 0.97, f"{len(population) - len(visible):,} candidates above y=20 omitted",
        transform=axes[0].transAxes, va="top", fontsize=9,
    )
    for group, color in GROUP_COLORS.items():
        rows = manual[
            (manual["manual_group"] == group) & (manual["local_contrast"] <= max_contrast)
        ]
        axes[1].scatter(
            rows["band_conc"], rows["local_contrast"],
            s=19 if group != "High-confidence Real" else 28,
            color=color, alpha=0.78, edgecolors="white", linewidths=0.25,
            label=f"{group} ({len(rows)})",
        )
    concentration = float(detection["cprf_min_band_concentration"])
    contrast = float(detection["cprf_min_local_contrast"])
    for ax in axes:
        ax.axvline(
            concentration, color="#17324d", linestyle="--", linewidth=1.7,
            label="batch CPRF gate",
        )
        ax.axhline(contrast, color="#17324d", linestyle="--", linewidth=1.7)
        ax.axhline(
            1.80, color="#8f5d00", linestyle=":", linewidth=1.8,
            label="manual-selection floor",
        )
        ax.set_xlim(
            0.0,
            min(0.92, max(0.82, float(population["band_conc"].quantile(0.999)))),
        )
        ax.set_ylim(0.0, max_contrast)
        ax.yaxis.set_major_locator(MultipleLocator(1.0))
        ax.set(xlabel="Main-band energy concentration", ylabel="Local sideband contrast")
    axes[0].set_title(f"All {len(population):,} pipeline candidates")
    axes[1].set_title("Fixed 1,993-case manual review")
    handles, labels = axes[1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[1].legend(unique.values(), unique.keys(), frameon=True, fontsize=9)
    save_figure(fig, output / "05_cprf_score_plane.png")


def plot_cprf_diagnostics(
    population: pd.DataFrame,
    manual: pd.DataFrame,
    detection: dict[str, object],
    sample_indices: np.ndarray,
    output: Path,
) -> None:
    sampled = population.iloc[sample_indices]
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    fig.suptitle("CPRF response diagnostics", fontsize=18, fontweight="bold")
    axes[0, 0].hexbin(
        sampled["h2"], sampled["h3"], gridsize=65, bins="log", mincnt=1, cmap="Blues"
    )
    for group, color in GROUP_COLORS.items():
        rows = manual[manual["manual_group"] == group]
        axes[0, 0].scatter(
            rows["h2"], rows["h3"], s=16, color=color, alpha=0.55, label=group
        )
    harmonic = float(detection["cprf_harmonic_min_relative"])
    axes[0, 0].axvline(harmonic, color=INK, linestyle="--")
    axes[0, 0].axhline(harmonic, color=INK, linestyle="--")
    axes[0, 0].set(
        xlabel=FEATURE_LABELS["h2"], ylabel=FEATURE_LABELS["h3"],
        title="Harmonic response plane",
    )
    axes[0, 0].legend(frameon=False, fontsize=8)

    hb = axes[0, 1].hexbin(
        sampled["band_persist"], np.maximum(sampled["ridge_int"], 1e-4),
        gridsize=65, bins="log", mincnt=1, cmap="YlGnBu", yscale="log",
    )
    axes[0, 1].axvline(
        float(detection["cprf_min_band_persistence"]), color=INK, linestyle="--"
    )
    axes[0, 1].set(
        xlabel="Band persistence", ylabel="Integrated ridge strength",
        title="Persistence-strength response",
    )
    fig.colorbar(hb, ax=axes[0, 1], label="sample density")

    hb = axes[1, 0].hexbin(
        np.maximum(sampled["ridge_peak"], 1e-4),
        np.maximum(sampled["ridge_int"], 1e-4),
        gridsize=65, bins="log", mincnt=1, cmap="inferno_r", xscale="log", yscale="log",
    )
    axes[1, 0].axvline(
        float(detection["cprf_min_peak_strength"]), color=INK, linestyle="--"
    )
    axes[1, 0].set(
        xlabel="Peak ridge strength", ylabel="Integrated ridge strength",
        title="Peak versus integrated strength",
    )
    fig.colorbar(hb, ax=axes[1, 0], label="sample density")

    harm_counts = population["harm_n"].round().astype(int).value_counts().sort_index()
    colors = ["#577590", "#ee9b00", "#bb3e03"]
    axes[1, 1].bar(
        harm_counts.index.astype(str), harm_counts.values,
        color=[colors[min(index, len(colors) - 1)] for index in range(len(harm_counts))],
    )
    for position, value in enumerate(harm_counts.values):
        axes[1, 1].text(position, value, f"{value:,}", ha="center", va="bottom", fontsize=9)
    axes[1, 1].set(
        xlabel="Supported auxiliary harmonics", ylabel="Candidates",
        title="Full-population harmonic support",
    )
    save_figure(fig, output / "06_cprf_diagnostics.png")


def threshold_grid(
    values: np.ndarray,
    current: float,
    *,
    upper: float | None = None,
) -> np.ndarray:
    finite = values[np.isfinite(values)]
    high = float(np.quantile(finite, 0.995))
    if upper is not None:
        high = min(high, upper)
    low = min(current, float(np.min(finite)))
    if high <= low:
        high = low + max(abs(low) * 0.1, 1e-3)
    return np.linspace(low, high, 140)


def make_gate_sweep(
    population: pd.DataFrame,
    manual: pd.DataFrame,
    detection: dict[str, object],
) -> pd.DataFrame:
    high = manual[manual["manual_group"] == "High-confidence Real"]
    false = manual[manual["manual_group"] == "Pure FP"]
    specifications = (
        ("ridge_peak", "cprf_min_peak_strength", None),
        ("ridge_int", "cprf_min_integrated_strength", None),
        ("band_persist", "cprf_min_band_persistence", 1.0),
        ("band_conc", "cprf_min_band_concentration", 1.0),
        ("local_contrast", "cprf_min_local_contrast", 20.0),
    )
    rows: list[dict[str, float | str]] = []
    for feature, config_key, upper in specifications:
        current = float(detection[config_key])
        grid = threshold_grid(
            population[feature].to_numpy(dtype=np.float64), current, upper=upper
        )
        for threshold in grid:
            rows.append(
                {
                    "feature": feature,
                    "threshold": float(threshold),
                    "population_retention": float(np.mean(population[feature] >= threshold)),
                    "high_real_retention": float(np.mean(high[feature] >= threshold)),
                    "pure_fp_retention": float(np.mean(false[feature] >= threshold)),
                    "current_threshold": current,
                }
            )
    return pd.DataFrame(rows)


def plot_gate_sweep(sweep: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    fig.suptitle("One-at-a-time CPRF tightening response", fontsize=18, fontweight="bold")
    for ax, feature in zip(axes.flat, sweep["feature"].drop_duplicates()):
        rows = sweep[sweep["feature"] == feature]
        ax.plot(
            rows["threshold"], rows["population_retention"],
            color=POPULATION, label="Full candidate population",
        )
        ax.plot(
            rows["threshold"], rows["high_real_retention"],
            color=HIGH_REAL, linewidth=2.2, label="High-confidence Real",
        )
        ax.plot(
            rows["threshold"], rows["pure_fp_retention"],
            color=PURE_FP, linewidth=2.2, label="Pure FP",
        )
        ax.axvline(
            float(rows["current_threshold"].iloc[0]), color=INK,
            linestyle="--", linewidth=1.4,
        )
        if feature in {"ridge_peak", "ridge_int", "local_contrast"}:
            ax.set_xscale("symlog", linthresh=0.25)
        ax.set_ylim(-0.02, 1.02)
        ax.set(
            xlabel=f"Minimum {FEATURE_LABELS[feature].lower()}",
            ylabel="Retained fraction",
            title=FEATURE_LABELS[feature],
        )
    axes[1, 2].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[1, 2].legend(handles, labels, loc="center", frameon=False, fontsize=11)
    axes[1, 2].text(
        0.5, 0.27,
        "Each panel tightens one gate only.\nCurves are conditional on candidates already\nexported by this batch.",
        ha="center", va="center", transform=axes[1, 2].transAxes, color="#625d52",
    )
    save_figure(fig, output / "07_cprf_tightening_response.png")


def make_joint_surface(
    population: pd.DataFrame,
    manual: pd.DataFrame,
    detection: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    high = manual[manual["manual_group"] == "High-confidence Real"]
    false = manual[manual["manual_group"] == "Pure FP"]
    x0 = float(detection["cprf_min_band_concentration"])
    y0 = float(detection["cprf_min_local_contrast"])
    concentrations = np.linspace(
        x0, min(0.85, float(population["band_conc"].quantile(0.999))), 85
    )
    contrast_high = min(
        20.0,
        max(
            float(population["local_contrast"].quantile(0.999)),
            float(high["local_contrast"].quantile(0.999)),
            float(false["local_contrast"].quantile(0.999)),
        ),
    )
    contrasts = np.linspace(y0, max(y0 + 0.1, contrast_high), 150)
    high_surface = np.empty((len(contrasts), len(concentrations)), dtype=np.float64)
    false_surface = np.empty_like(high_surface)
    rows: list[dict[str, float]] = []
    high_concentration = high["band_conc"].to_numpy()
    false_concentration = false["band_conc"].to_numpy()
    for yi, contrast in enumerate(contrasts):
        high_contrast = high["local_contrast"].to_numpy() >= contrast
        false_contrast = false["local_contrast"].to_numpy() >= contrast
        for xi, concentration in enumerate(concentrations):
            high_rate = float(np.mean(high_contrast & (high_concentration >= concentration)))
            false_rate = float(np.mean(false_contrast & (false_concentration >= concentration)))
            high_surface[yi, xi] = high_rate
            false_surface[yi, xi] = false_rate
            rows.append(
                {
                    "min_band_concentration": float(concentration),
                    "min_local_contrast": float(contrast),
                    "high_real_retention": high_rate,
                    "pure_fp_retention": false_rate,
                }
            )
    return concentrations, contrasts, high_surface, false_surface, pd.DataFrame(rows)


def plot_joint_surface(
    concentrations: np.ndarray,
    contrasts: np.ndarray,
    high_surface: np.ndarray,
    false_surface: np.ndarray,
    detection: dict[str, object],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    fig.suptitle("Joint CPRF concentration-contrast tightening", fontsize=18, fontweight="bold")
    extent = [concentrations[0], concentrations[-1], contrasts[0], contrasts[-1]]
    for ax, surface, title, cmap in (
        (axes[0], high_surface, "High-confidence Real retained", "YlGn"),
        (axes[1], false_surface, "Pure FP retained", "OrRd"),
    ):
        image = ax.imshow(
            surface, origin="lower", aspect="auto", extent=extent,
            vmin=0.0, vmax=1.0, cmap=cmap,
        )
        contours = ax.contour(
            concentrations, contrasts, surface,
            levels=[0.1, 0.25, 0.5, 0.75, 0.9], colors=INK,
            linewidths=0.65, alpha=0.7,
        )
        ax.clabel(contours, inline=True, fontsize=8, fmt="%.2f")
        ax.scatter(
            [float(detection["cprf_min_band_concentration"])],
            [float(detection["cprf_min_local_contrast"])],
            marker="x", s=90, linewidths=2.2, color="#17324d", label="batch gate",
        )
        ax.set(
            xlabel="Minimum band concentration", ylabel="Minimum local contrast", title=title
        )
        ax.legend(frameon=False)
        fig.colorbar(image, ax=ax, label="Retained fraction")
    save_figure(fig, output / "08_cprf_joint_gate_surface.png")


def select_joint_operating_points(surface: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for target in (0.90, 0.80, 0.70, 0.60, 0.50):
        eligible = surface[surface["high_real_retention"] >= target]
        minimum_fp = eligible["pure_fp_retention"].min()
        best = eligible[eligible["pure_fp_retention"] == minimum_fp].sort_values(
            ["high_real_retention", "min_band_concentration", "min_local_contrast"],
            ascending=[False, False, False],
        )
        selected = best.iloc[0].copy()
        selected["target_high_real_retention"] = target
        rows.append(selected)
    return pd.DataFrame(rows)[
        [
            "target_high_real_retention",
            "min_band_concentration",
            "min_local_contrast",
            "high_real_retention",
            "pure_fp_retention",
        ]
    ]


def plot_cluster_map(
    cluster: dict[str, object],
    sample_indices: np.ndarray,
    output: Path,
) -> None:
    high_pca = np.asarray(cluster["high_pca"])
    population_pca = np.asarray(cluster["population_pca"])
    high_labels = np.asarray(cluster["high_labels"])
    population_labels = np.asarray(cluster["population_labels"])
    within = np.asarray(cluster["within"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    fig.suptitle(
        f"High-confidence Real phenotypes (k={int(cluster['chosen']['clusters'])}, silhouette={float(cluster['chosen']['silhouette']):.3f})",
        fontsize=18,
        fontweight="bold",
    )
    for label in sorted(np.unique(high_labels)):
        selected = high_labels == label
        axes[0].scatter(
            high_pca[selected, 0], high_pca[selected, 1], s=34,
            color=CLUSTER_COLORS[label - 1], alpha=0.82,
            edgecolors="white", linewidths=0.4,
            label=f"Cluster {label} (n={selected.sum()})",
        )
    axes[0].set(
        xlabel="HQ PCA 1", ylabel="HQ PCA 2",
        title="Clusters fitted only on high-confidence Real",
    )
    axes[0].legend(frameon=False)

    sampled = np.asarray(sample_indices)
    axes[1].scatter(
        population_pca[sampled, 0], population_pca[sampled, 1],
        s=4, color="#aeb7b7", alpha=0.11, rasterized=True,
    )
    for label in sorted(np.unique(high_labels)):
        projected = sampled[(population_labels[sampled] == label) & within[sampled]]
        axes[1].scatter(
            population_pca[projected, 0], population_pca[projected, 1],
            s=5, color=CLUSTER_COLORS[label - 1], alpha=0.22, rasterized=True,
        )
        selected = high_labels == label
        axes[1].scatter(
            high_pca[selected, 0], high_pca[selected, 1], s=30,
            color=CLUSTER_COLORS[label - 1], edgecolors="white", linewidths=0.4,
        )
    axes[1].set(
        xlabel="HQ PCA 1", ylabel="HQ PCA 2",
        title="Full population projected into the HQ feature space",
    )
    axes[1].set_xlim(*np.quantile(population_pca[sampled, 0], [0.005, 0.995]))
    axes[1].set_ylim(*np.quantile(population_pca[sampled, 1], [0.005, 0.995]))
    axes[1].text(
        0.98, 0.97,
        "Color = inside the cluster's 95% HQ radius\nGray = outside all HQ phenotype envelopes",
        transform=axes[1].transAxes, va="top", ha="right", fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35", "facecolor": "#fffdf7",
            "edgecolor": GRID, "alpha": 0.94,
        },
    )
    save_figure(fig, output / "09_high_quality_cluster_map.png")


def cluster_descriptors(high: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
    descriptors: dict[int, str] = {}
    global_duration = float(high["dur_rec"].median())
    for cluster_id in sorted(np.unique(labels)):
        rows = high.iloc[np.flatnonzero(labels == cluster_id)]
        harmonic = float((rows["h2"] + rows["h3"]).median())
        if harmonic >= 0.80:
            descriptor = "harmonic-rich"
        elif float(rows["dur_rec"].median()) >= 1.8 * global_duration:
            descriptor = "long-duration fundamental"
        else:
            descriptor = "fundamental-dominant"
        descriptors[int(cluster_id)] = descriptor
    return descriptors


def make_cluster_summary(
    high: pd.DataFrame,
    manual: pd.DataFrame,
    manual_population_indices: np.ndarray,
    cluster: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    high_labels = np.asarray(cluster["high_labels"])
    population_labels = np.asarray(cluster["population_labels"])
    distances = np.asarray(cluster["population_distances"])
    within = np.asarray(cluster["within"])
    descriptors = cluster_descriptors(high, high_labels)
    summary_rows: list[dict[str, object]] = []
    for cluster_id in sorted(np.unique(high_labels)):
        high_rows = high.iloc[np.flatnonzero(high_labels == cluster_id)]
        population_mask = population_labels == cluster_id
        within_mask = population_mask & within
        manual_clusters = population_labels[manual_population_indices]
        manual_within = within[manual_population_indices]
        row: dict[str, object] = {
            "cluster": int(cluster_id),
            "phenotype": descriptors[int(cluster_id)],
            "high_real_seeds": int(len(high_rows)),
            "hq_radius_95": float(cluster["radii"][int(cluster_id)]),
            "full_nearest_cluster": int(np.count_nonzero(population_mask)),
            "full_inside_hq_envelope": int(np.count_nonzero(within_mask)),
            "full_inside_hq_fraction": float(np.mean(within_mask)),
        }
        for group in GROUP_COLORS:
            group_mask = manual["manual_group"].to_numpy() == group
            key = group.lower().replace("-", "_").replace(" ", "_")
            row[f"manual_{key}_inside"] = int(
                np.count_nonzero(
                    group_mask & (manual_clusters == cluster_id) & manual_within
                )
            )
            row[f"manual_{key}_total"] = int(np.count_nonzero(group_mask))
        for feature in MODEL_FEATURES:
            row[f"median_{feature}"] = float(high_rows[feature].median())
        summary_rows.append(row)

    assignments = manual[
        [
            "review_rank", "raw_key", "run_id", "candidate_id",
            "manual_group", "case_type", *MODEL_FEATURES,
        ]
    ].copy()
    assignments["cluster"] = population_labels[manual_population_indices]
    assignments["cluster_distance"] = distances[manual_population_indices]
    assignments["inside_hq_envelope"] = within[manual_population_indices].astype(int)
    assignments["phenotype"] = [
        descriptors[int(value)] for value in assignments["cluster"]
    ]
    return pd.DataFrame(summary_rows), assignments


def plot_cluster_profiles(
    cluster: dict[str, object],
    cluster_summary: pd.DataFrame,
    output: Path,
) -> None:
    labels = np.asarray(cluster["high_labels"])
    z = np.asarray(cluster["high_z"])
    ids = sorted(np.unique(labels))
    profiles = np.asarray([np.median(z[labels == cluster_id], axis=0) for cluster_id in ids])
    fig, axes = plt.subplots(
        1, 2, figsize=(17, 6.5), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.5, 1.0]},
    )
    fig.suptitle("High-confidence Real cluster profiles", fontsize=18, fontweight="bold")
    limit = max(1.0, float(np.quantile(np.abs(profiles), 0.98)))
    image = axes[0].imshow(
        profiles, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit
    )
    axes[0].set_xticks(
        range(len(MODEL_FEATURES)),
        [FEATURE_LABELS[name] for name in MODEL_FEATURES],
        rotation=35,
        ha="right",
    )
    axes[0].set_yticks(range(len(ids)), [f"Cluster {value}" for value in ids])
    axes[0].set_title("Median robust-standardized feature profile")
    for row in range(profiles.shape[0]):
        for column in range(profiles.shape[1]):
            value = profiles[row, column]
            axes[0].text(
                column, row, f"{value:.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(value) > 0.55 * limit else INK,
            )
    fig.colorbar(image, ax=axes[0], label="Median robust z")

    positions = np.arange(len(cluster_summary))
    axes[1].barh(
        positions + 0.18, cluster_summary["full_nearest_cluster"],
        height=0.32, color="#aeb7b7", label="Nearest cluster",
    )
    axes[1].barh(
        positions - 0.18, cluster_summary["full_inside_hq_envelope"],
        height=0.32,
        color=[CLUSTER_COLORS[index - 1] for index in cluster_summary["cluster"]],
        label="Inside 95% HQ envelope",
    )
    axes[1].set_yticks(
        positions,
        [f"C{row.cluster}: {row.phenotype}" for row in cluster_summary.itertuples()],
    )
    axes[1].set_xscale("log")
    axes[1].set(xlabel="Full-population candidates", title="Projection volume (log scale)")
    axes[1].legend(frameon=False)
    save_figure(fig, output / "10_cluster_profiles_and_population.png")


def plot_correlations(
    population: pd.DataFrame,
    sample_indices: np.ndarray,
    output: Path,
) -> None:
    features = [*MODEL_FEATURES, "ridge_peak", "score"]
    sample = population.iloc[sample_indices][features].to_numpy(dtype=np.float64)
    ranked = np.column_stack(
        [rankdata(sample[:, index], method="average") for index in range(sample.shape[1])]
    )
    correlation = np.corrcoef(ranked, rowvar=False)
    fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)
    image = ax.imshow(correlation, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    labels = [FEATURE_LABELS.get(name, name.replace("_", " ").title()) for name in features]
    ax.set_xticks(range(len(features)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(features)), labels)
    ax.set_title(
        f"Full-population Spearman feature correlation (sample n={len(sample_indices):,})",
        fontsize=16,
        fontweight="bold",
    )
    for row in range(len(features)):
        for column in range(len(features)):
            value = correlation[row, column]
            ax.text(
                column, row, f"{value:.2f}", ha="center", va="center", fontsize=7.5,
                color="white" if abs(value) > 0.62 else INK,
            )
    fig.colorbar(image, ax=ax, label="Spearman rho")
    save_figure(fig, output / "11_population_feature_correlations.png")


def plot_representatives(
    high: pd.DataFrame,
    cluster: dict[str, object],
    image_dir: Path,
    output: Path,
    count: int = 4,
) -> None:
    labels = np.asarray(cluster["high_labels"])
    distances = np.asarray(cluster["high_distances"])
    ids = sorted(np.unique(labels))
    fig, axes = plt.subplots(
        len(ids), count, figsize=(5.2 * count, 3.3 * len(ids)),
        squeeze=False, constrained_layout=True,
    )
    fig.suptitle(
        "Nearest-to-centroid high-confidence Real representatives",
        fontsize=18,
        fontweight="bold",
    )
    for row_index, cluster_id in enumerate(ids):
        members = np.flatnonzero(labels == cluster_id)
        representatives = members[np.argsort(distances[members])[:count]]
        for column, member in enumerate(representatives):
            item = high.iloc[int(member)]
            path = image_dir / f"{int(item['review_rank']):04d}_{item['raw_key']}.png"
            ax = axes[row_index, column]
            if path.exists():
                with Image.open(path) as image:
                    ax.imshow(np.asarray(image.convert("RGB")))
            else:
                ax.text(
                    0.5, 0.5, "image missing", ha="center", va="center",
                    transform=ax.transAxes,
                )
            ax.set_title(
                f"C{cluster_id} · rank {int(item['review_rank'])} · P={item['period_rec']:.1f} · D={item['dur_rec']:.0f}",
                fontsize=9,
            )
            ax.axis("off")
        for column in range(len(representatives), count):
            axes[row_index, column].axis("off")
    save_figure(fig, output / "12_cluster_representatives.png")


def isolated_real_candidates(
    population: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    isolated = assignments[
        (assignments["manual_group"] == "High-confidence Real")
        & (assignments["inside_hq_envelope"] == 0)
    ]
    population_by_key = population.set_index("candidate_key", drop=False)
    rows: list[dict[str, object]] = []
    period_tolerance = np.log(1.08)
    for assignment in isolated.itertuples(index=False):
        key = f"{assignment.run_id}\x1f{assignment.candidate_id}"
        candidate = population_by_key.loc[key]
        observation = population[population["run_id"] == assignment.run_id]
        overlap = np.maximum(
            0.0,
            np.minimum(observation["t1_rec"], candidate["t1_rec"])
            - np.maximum(observation["t0_rec"], candidate["t0_rec"]),
        )
        overlap_fraction = overlap / np.minimum(
            observation["dur_rec"], candidate["dur_rec"]
        )
        period_close = np.abs(
            np.log(observation["period_rec"] / candidate["period_rec"])
        ) <= period_tolerance
        family = observation[(overlap_fraction >= 0.50) & period_close]
        other_channels = family.loc[
            family["channel"] != candidate["channel"], "channel"
        ]
        rows.append(
            {
                "review_rank": int(assignment.review_rank),
                "raw_key": assignment.raw_key,
                "run_id": assignment.run_id,
                "candidate_id": assignment.candidate_id,
                "cluster": int(assignment.cluster),
                "phenotype": assignment.phenotype,
                "cluster_distance": float(assignment.cluster_distance),
                "channel": int(candidate["channel"]),
                "freq_mhz": float(candidate["freq_mhz"]),
                "t0_rec": int(candidate["t0_rec"]),
                "t1_rec": int(candidate["t1_rec"]),
                "period_rec": float(candidate["period_rec"]),
                "dur_rec": int(candidate["dur_rec"]),
                "band_conc": float(candidate["band_conc"]),
                "band_persist": float(candidate["band_persist"]),
                "local_contrast": float(candidate["local_contrast"]),
                "ridge_int": float(candidate["ridge_int"]),
                "h2": float(candidate["h2"]),
                "h3": float(candidate["h3"]),
                "family_candidates": int(len(family)),
                "family_channels": int(family["channel"].nunique()),
                "family_channel_span": int(
                    family["channel"].max() - family["channel"].min()
                ),
                "nearest_other_channel": (
                    int(np.min(np.abs(other_channels - candidate["channel"])))
                    if len(other_channels)
                    else -1
                ),
                "family_freq_span_mhz": float(
                    family["freq_mhz"].max() - family["freq_mhz"].min()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["family_channels", "cluster_distance"], ascending=[True, False]
    )


def plot_isolated_real_candidates(
    isolated: pd.DataFrame,
    image_dir: Path,
    output: Path,
) -> None:
    columns = 4
    rows = int(np.ceil(len(isolated) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(5.2 * columns, 3.35 * rows),
        squeeze=False, constrained_layout=True,
    )
    fig.suptitle(
        "High-confidence Real candidates outside the 95% phenotype envelopes",
        fontsize=18,
        fontweight="bold",
    )
    for ax, item in zip(axes.flat, isolated.itertuples(index=False)):
        path = image_dir / f"{item.review_rank:04d}_{item.raw_key}.png"
        with Image.open(path) as image:
            ax.imshow(np.asarray(image.convert("RGB")))
        companion = (
            "no matched channel companion"
            if item.family_channels == 1
            else f"matched channels={item.family_channels}, nearest dch={item.nearest_other_channel}"
        )
        ax.set_title(
            f"rank {item.review_rank} · C{item.cluster} · d={item.cluster_distance:.2f}\n"
            f"P={item.period_rec:.1f}, D={item.dur_rec} · {companion}",
            fontsize=8.5,
        )
        ax.axis("off")
    for ax in axes.flat[len(isolated):]:
        ax.axis("off")
    save_figure(fig, output / "13_isolated_high_quality_candidates.png")


def feature_auc(manual: pd.DataFrame) -> dict[str, dict[str, float | str]]:
    binary = manual[
        manual["manual_group"].isin(["High-confidence Real", "Pure FP"])
    ]
    target = (binary["manual_group"] == "High-confidence Real").astype(int).to_numpy()
    output: dict[str, dict[str, float | str]] = {}
    for feature in [*MODEL_FEATURES, "ridge_peak", "score"]:
        if binary[feature].nunique() < 2:
            auc = 0.5
        else:
            auc = float(roc_auc_score(target, binary[feature]))
        output[feature] = {
            "raw_auc": auc,
            "separation_auc": max(auc, 1.0 - auc),
            "real_direction": "higher" if auc >= 0.5 else "lower",
        }
    return output


def population_quantiles(population: pd.DataFrame) -> dict[str, dict[str, float]]:
    quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    return {
        feature: {
            f"q{int(quantile * 100):02d}": float(population[feature].quantile(quantile))
            for quantile in quantiles
        }
        for feature in ["freq_mhz", *MODEL_FEATURES, "ridge_peak", "score"]
    }


def markdown_cluster_table(cluster_summary: pd.DataFrame) -> list[str]:
    columns = (
        "cluster",
        "phenotype",
        "high_real_seeds",
        "full_inside_hq_envelope",
        "full_inside_hq_fraction",
        "median_period_rec",
        "median_dur_rec",
        "median_band_conc",
        "median_local_contrast",
    )
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    rows = [header, divider]
    for item in cluster_summary[list(columns)].itertuples(index=False, name=None):
        values = [f"{value:.4g}" if isinstance(value, float) else str(value) for value in item]
        rows.append("| " + " | ".join(values) + " |")
    return rows


def write_report(
    summary: dict[str, object],
    cluster_summary: pd.DataFrame,
    operating_points: pd.DataFrame,
    output: Path,
) -> None:
    groups = summary["manual_review"]["groups"]
    best_auc = max(
        summary["manual_feature_separation"].items(),
        key=lambda item: item[1]["separation_auc"],
    )
    lines = [
        "# Full Candidate Population and CPRF Analysis",
        "",
        f"- Full candidate rows: {summary['population']['candidates']:,}",
        f"- Observations: {summary['population']['observations']:,}",
        f"- Fixed manual-review cases: {summary['manual_review']['cases']:,}",
        f"- High-confidence Real clustering seeds: {groups['High-confidence Real']:,}",
        f"- Selected clusters: {summary['clustering']['selected_k']} (silhouette {summary['clustering']['silhouette']:.3f})",
        f"- Strongest single-feature separation: {best_auc[0]} (direction {best_auc[1]['real_direction']}, AUC {best_auc[1]['separation_auc']:.3f})",
        "",
        "## Cluster Summary",
        "",
        *markdown_cluster_table(cluster_summary),
        "",
        "## Manual Feature Separation",
        "",
        "| feature | separation AUC | Real direction |",
        "|---|---:|---|",
        *[
            f"| {feature} | {metrics['separation_auc']:.4f} | {metrics['real_direction']} |"
            for feature, metrics in sorted(
                summary["manual_feature_separation"].items(),
                key=lambda item: item[1]["separation_auc"],
                reverse=True,
            )
        ],
        "",
        "## Conditional CPRF Operating Points",
        "",
        "| target High Real retention | min band concentration | min local contrast | observed High Real retention | observed pure FP retention |",
        "|---:|---:|---:|---:|---:|",
        *[
            f"| {row.target_high_real_retention:.0%} | {row.min_band_concentration:.4f} | {row.min_local_contrast:.4f} | {row.high_real_retention:.2%} | {row.pure_fp_retention:.2%} |"
            for row in operating_points.itertuples(index=False)
        ],
        "",
        "## Interpretation Boundary",
        "",
        "Clusters are fitted only on high-confidence Real cases. Full-population assignment means nearest CPRF/morphology phenotype, not a Real classification. The 95% envelope is a descriptive similarity rule. Manual labels come from a deliberately filtered 1,993-case review set, so its Real/FP prevalence must not be extrapolated to the full batch.",
        "",
        "CPRF tightening plots are conditional on candidates already exported by the batch. They estimate stricter post-filters, not recovery of CPRF-rejected PELT windows.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_plotting()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    population = load_population(args.candidates)
    manual = load_manual(args.selection, args.labels)
    with args.config.open(encoding="utf-8") as handle:
        detection = json.load(handle)["detection"]

    population_lookup = pd.Series(
        np.arange(len(population)), index=population["candidate_key"]
    )
    if not set(manual["candidate_key"]) <= set(population_lookup.index):
        raise RuntimeError("manual review set is not a subset of the selected full batch")
    manual_population_indices = population_lookup.loc[
        manual["candidate_key"]
    ].to_numpy(dtype=np.int64)
    high = manual[
        manual["manual_group"] == "High-confidence Real"
    ].reset_index(drop=True)
    cluster = fit_clusters(
        high,
        population,
        args.min_clusters,
        args.max_clusters,
        args.seed,
    )
    cluster_summary, manual_assignments = make_cluster_summary(
        high, manual, manual_population_indices, cluster
    )
    isolated_candidates = isolated_real_candidates(population, manual_assignments)

    rng = np.random.default_rng(args.seed)
    sample_size = min(max(1, args.plot_sample), len(population))
    sample_indices = np.sort(
        rng.choice(len(population), size=sample_size, replace=False)
    )
    gate_sweep = make_gate_sweep(population, manual, detection)
    concentrations, contrasts, high_surface, false_surface, joint_surface = (
        make_joint_surface(population, manual, detection)
    )
    operating_points = select_joint_operating_points(joint_surface)

    plot_basic_distributions(population, manual, args.output_dir)
    plot_population(population, manual, args.output_dir)
    plot_manual_distributions(manual, args.output_dir)
    plot_score_plane(population, manual, detection, args.output_dir)
    plot_cprf_diagnostics(
        population, manual, detection, sample_indices, args.output_dir
    )
    plot_gate_sweep(gate_sweep, args.output_dir)
    plot_joint_surface(
        concentrations, contrasts, high_surface, false_surface, detection, args.output_dir
    )
    plot_cluster_map(cluster, sample_indices, args.output_dir)
    plot_cluster_profiles(cluster, cluster_summary, args.output_dir)
    plot_correlations(population, sample_indices, args.output_dir)
    plot_representatives(high, cluster, args.review_images, args.output_dir)
    plot_isolated_real_candidates(
        isolated_candidates, args.review_images, args.output_dir
    )

    cluster_summary.to_csv(args.output_dir / "cluster_summary.csv", index=False)
    manual_assignments.to_csv(args.output_dir / "manual_case_clusters.csv", index=False)
    isolated_candidates.to_csv(
        args.output_dir / "isolated_high_quality_candidates.csv", index=False
    )
    gate_sweep.to_csv(args.output_dir / "cprf_gate_sweep.csv", index=False)
    joint_surface.to_csv(args.output_dir / "cprf_joint_gate_surface.csv", index=False)
    operating_points.to_csv(
        args.output_dir / "cprf_joint_operating_points.csv", index=False
    )
    (args.output_dir / "cluster_model_selection.json").write_text(
        json.dumps(cluster["scores"], indent=2), encoding="utf-8"
    )

    status_counts = population["candidate_status"].fillna("unknown").value_counts().to_dict()
    group_counts = manual["manual_group"].value_counts().to_dict()
    summary: dict[str, object] = {
        "scope": {
            "candidate_source": str(args.candidates),
            "manual_source": str(args.labels),
            "manual_selection": str(args.selection),
            "seed": args.seed,
            "interpretation": "high-quality clusters are morphology prototypes, not a classifier",
            "selection_bias": "manual cases satisfy band_conc>=0.30, local_contrast>=1.80, ridge_int>=0 and are non-vetoed",
            "cprf_curve_scope": "one-at-a-time stricter post-filters among already exported candidates",
        },
        "population": {
            "candidates": int(len(population)),
            "observations": int(population["run_id"].nunique()),
            "channels": int(population["channel"].nunique()),
            "status": {str(key): int(value) for key, value in status_counts.items()},
            "quantiles": population_quantiles(population),
        },
        "manual_review": {
            "cases": int(len(manual)),
            "groups": {str(key): int(value) for key, value in group_counts.items()},
            "case_types": {
                str(key): int(value)
                for key, value in manual["case_type"].value_counts().to_dict().items()
            },
        },
        "clustering": {
            "features": list(MODEL_FEATURES),
            "scaling": "RobustScaler quantile_range=(10,90); log transforms for period, duration, contrast, integrated strength",
            "selection": cluster["scores"],
            "selected_k": int(cluster["chosen"]["clusters"]),
            "silhouette": float(cluster["chosen"]["silhouette"]),
            "full_inside_any_hq_envelope": int(np.count_nonzero(cluster["within"])),
            "full_inside_any_hq_envelope_fraction": float(np.mean(cluster["within"])),
        },
        "isolated_high_quality_candidates": {
            "definition": "High-confidence Real candidates outside the assigned cluster 95% robust-distance envelope.",
            "count": int(len(isolated_candidates)),
            "without_matched_channel_companion": int(
                (isolated_candidates["family_channels"] == 1).sum()
            ),
            "companion_match": "Same observation, period within 8%, and time overlap >= 50% of the shorter window.",
        },
        "manual_feature_separation": feature_auc(manual),
        "cprf_thresholds": {
            key: detection[key] for key in detection if key.startswith("cprf_")
        },
        "conditional_cprf_operating_points": operating_points.to_dict(orient="records"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(summary, cluster_summary, operating_points, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "population": len(population),
                "high_real": len(high),
                "selected_k": summary["clustering"]["selected_k"],
                "silhouette": summary["clustering"]["silhouette"],
                "full_inside_hq_envelope": summary["clustering"][
                    "full_inside_any_hq_envelope"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
