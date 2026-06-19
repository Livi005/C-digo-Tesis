#!/usr/bin/env python3
"""Se generan las tablas y gráficas descriptivas de los resultados oficiales."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_common import descriptive_table, find_project_root, output_dirs


def boxplot(frame: pd.DataFrame, groups: list[str], metric: str, title: str, path) -> None:
    labels: list[str] = []
    values: list[np.ndarray] = []
    for key, group in frame.groupby(groups, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        labels.append("\n".join(map(str, key_values)))
        values.append(pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy())
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 5))
    ax.boxplot(values, labels=labels, showfliers=True)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def barplot(frame: pd.DataFrame, label: str, value: str, title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(frame))
    ax.bar(x, frame[value])
    ax.set_xticks(x)
    ax.set_xticklabels(frame[label], rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(value)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = find_project_root()
    clean_dir, tables_dir, figures_dir = output_dirs(root)
    official_figures = figures_dir / "official"
    official_figures.mkdir(parents=True, exist_ok=True)

    variants = pd.read_csv(clean_dir / "summary_official_variants.csv")
    comparison = pd.read_csv(clean_dir / "summary_official_vns_ivns.csv")
    pairs = pd.read_csv(clean_dir / "pairs_official_vns_ivns.csv")

    variant_by_instance = descriptive_table(
        variants,
        ["instance", "ivns_profile", "vnd_mode"],
    )
    variant_by_instance.to_csv(tables_dir / "official_variants_descriptive.csv", index=False)

    ranking = (
        variants.groupby(["ivns_profile", "vnd_mode"])
        .agg(
            runs=("gap_percent", "size"),
            gap_mean=("gap_percent", "mean"),
            gap_std=("gap_percent", "std"),
            gap_median=("gap_percent", "median"),
            gap_min=("gap_percent", "min"),
            gap_max=("gap_percent", "max"),
            time_mean=("elapsed_time_seconds", "mean"),
            time_median=("elapsed_time_seconds", "median"),
        )
        .reset_index()
    )
    ranking["variant"] = ranking["ivns_profile"] + "-" + ranking["vnd_mode"]
    ranking = ranking.sort_values(["gap_median", "gap_mean", "time_median"]).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    ranking.to_csv(tables_dir / "official_variants_ranking.csv", index=False)

    comparison_descriptive = descriptive_table(comparison, ["instance", "algorithm"])
    comparison_descriptive.to_csv(tables_dir / "official_vns_ivns_descriptive.csv", index=False)

    pair_columns = [
        "instance",
        "seed",
        "delta_gap_ivns_minus_vns",
        "delta_time_ivns_minus_vns",
    ]
    pairs[pair_columns].to_csv(tables_dir / "official_vns_ivns_differences.csv", index=False)

    boxplot(
        comparison,
        ["instance", "algorithm"],
        "gap_percent",
        "Comparación oficial del gap",
        official_figures / "01_official_gap_boxplot.png",
    )
    boxplot(
        comparison,
        ["instance", "algorithm"],
        "elapsed_time_seconds",
        "Comparación oficial del tiempo",
        official_figures / "02_official_time_boxplot.png",
    )
    barplot(
        ranking,
        "variant",
        "gap_median",
        "Ranking oficial de variantes IVNS",
        official_figures / "03_official_variant_ranking.png",
    )
    boxplot(
        variants,
        ["ivns_profile", "vnd_mode"],
        "gap_percent",
        "Gap de las variantes IVNS",
        official_figures / "04_official_variants_gap.png",
    )

    print("Tablas oficiales guardadas en", tables_dir)
    print("Figuras oficiales guardadas en", official_figures)


if __name__ == "__main__":
    main()
