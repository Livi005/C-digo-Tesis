#!/usr/bin/env python3
"""Se analiza la comparación confirmatoria pareada de F-n135-k7."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_common import descriptive_table, find_project_root, output_dirs, paired_statistics


def boxplot(frame: pd.DataFrame, metric: str, title: str, path) -> None:
    labels: list[str] = []
    values: list[np.ndarray] = []
    for algorithm, group in frame.groupby("algorithm", sort=True):
        labels.append(algorithm)
        values.append(pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy())
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(values, labels=labels, showfliers=True)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def seed_plot(pairs: pd.DataFrame, metric: str, title: str, path) -> None:
    ordered = pairs.sort_values("seed")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ordered["seed"], ordered[metric], marker="o")
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Semilla")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = find_project_root()
    clean_dir, tables_dir, figures_dir = output_dirs(root)
    figure_dir = figures_dir / "confirmatory_f"
    figure_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(clean_dir / "summary_confirmatory_f.csv")
    pairs = pd.read_csv(clean_dir / "pairs_confirmatory_f.csv")

    descriptive = descriptive_table(frame, ["instance", "algorithm"])
    statistics = paired_statistics(pairs, "confirmatory_f", holm_by_metric=False)

    descriptive.to_csv(tables_dir / "confirmatory_f_descriptive.csv", index=False)
    pairs.to_csv(tables_dir / "confirmatory_f_paired_values.csv", index=False)
    statistics.to_csv(tables_dir / "confirmatory_f_wilcoxon_holm.csv", index=False)

    boxplot(frame, "gap_percent", "Confirmatorio F: distribución del gap", figure_dir / "16_confirmatory_f_gap_boxplot.png")
    boxplot(frame, "elapsed_time_seconds", "Confirmatorio F: distribución del tiempo", figure_dir / "17_confirmatory_f_time_boxplot.png")
    seed_plot(pairs, "delta_gap_ivns_minus_vns", "Confirmatorio F: diferencia de gap por semilla", figure_dir / "18_confirmatory_f_gap_by_seed.png")
    seed_plot(pairs, "delta_time_ivns_minus_vns", "Confirmatorio F: diferencia de tiempo por semilla", figure_dir / "19_confirmatory_f_time_by_seed.png")

    print("Pares confirmatorios F =", len(pairs))
    print("Tablas guardadas en", tables_dir)
    print("Figuras guardadas en", figure_dir)


if __name__ == "__main__":
    main()
