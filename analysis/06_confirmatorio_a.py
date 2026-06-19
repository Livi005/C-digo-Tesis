#!/usr/bin/env python3
"""Se analiza la comparación confirmatoria pareada de las instancias A."""

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
    for key, group in frame.groupby(["instance", "algorithm"], sort=True):
        labels.append("\n".join(map(str, key)))
        values.append(pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy())
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(values, labels=labels, showfliers=True)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def delta_boxplot(pairs: pd.DataFrame, metric: str, title: str, path) -> None:
    labels: list[str] = []
    values: list[np.ndarray] = []
    for instance, group in pairs.groupby("instance", sort=True):
        labels.append(instance)
        values.append(pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(values, labels=labels, showfliers=True)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = find_project_root()
    clean_dir, tables_dir, figures_dir = output_dirs(root)
    figure_dir = figures_dir / "confirmatory_a"
    figure_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(clean_dir / "summary_confirmatory_a.csv")
    pairs = pd.read_csv(clean_dir / "pairs_confirmatory_a.csv")

    descriptive = descriptive_table(frame, ["instance", "algorithm"])
    statistics = paired_statistics(pairs, "confirmatory_a")

    descriptive.to_csv(tables_dir / "confirmatory_a_descriptive.csv", index=False)
    pairs.to_csv(tables_dir / "confirmatory_a_paired_values.csv", index=False)
    statistics.to_csv(tables_dir / "confirmatory_a_wilcoxon_holm.csv", index=False)

    boxplot(frame, "gap_percent", "Confirmatorio A: distribución del gap", figure_dir / "09_confirmatory_a_gap_boxplot.png")
    boxplot(frame, "elapsed_time_seconds", "Confirmatorio A: distribución del tiempo", figure_dir / "10_confirmatory_a_time_boxplot.png")
    delta_boxplot(pairs, "delta_gap_ivns_minus_vns", "Confirmatorio A: diferencia de gap", figure_dir / "11_confirmatory_a_gap_delta.png")
    delta_boxplot(pairs, "delta_time_ivns_minus_vns", "Confirmatorio A: diferencia de tiempo", figure_dir / "12_confirmatory_a_time_delta.png")

    print("Pares confirmatorios A =", len(pairs))
    print("Tablas guardadas en", tables_dir)
    print("Figuras guardadas en", figure_dir)


if __name__ == "__main__":
    main()
