#!/usr/bin/env python3
"""Se resume la exploración de perfiles y tiempos en F-n135-k7."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_common import find_project_root, output_dirs


def ranking(frame: pd.DataFrame) -> pd.DataFrame:
    table = (
        frame.groupby(["configuration", "ivns_profile", "vnd_mode", "ivns_time"])
        .agg(
            runs=("gap_percent", "size"),
            gap_mean=("gap_percent", "mean"),
            gap_std=("gap_percent", "std"),
            gap_median=("gap_percent", "median"),
            gap_q1=("gap_percent", lambda values: values.quantile(0.25)),
            gap_q3=("gap_percent", lambda values: values.quantile(0.75)),
            gap_min=("gap_percent", "min"),
            gap_max=("gap_percent", "max"),
            time_mean=("elapsed_time_seconds", "mean"),
            time_median=("elapsed_time_seconds", "median"),
        )
        .reset_index()
        .sort_values(["gap_median", "gap_mean", "time_median"])
        .reset_index(drop=True)
    )
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table


def lineplot(frame: pd.DataFrame, group: str, title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, group_frame in frame.groupby(group, sort=True):
        ordered = group_frame.sort_values("ivns_time")
        ax.plot(ordered["ivns_time"], ordered["gap_median"], marker="o", label=label)
    ax.set_title(title)
    ax.set_xlabel("Tiempo por vecindad")
    ax.set_ylabel("Mediana del gap")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = find_project_root()
    clean_dir, tables_dir, figures_dir = output_dirs(root)
    figure_dir = figures_dir / "exploratory_f"
    figure_dir.mkdir(parents=True, exist_ok=True)

    pilot = pd.read_csv(clean_dir / "summary_f_pilot.csv")
    high = pd.read_csv(clean_dir / "summary_f_high.csv")

    pilot_ranking = ranking(pilot)
    high_ranking = ranking(high)
    high_best_by_time = (
        high_ranking.sort_values(["ivns_time", "gap_median", "gap_mean", "time_median"])
        .groupby("ivns_time", as_index=False)
        .first()
        .sort_values("ivns_time")
    )
    selected = high_ranking.iloc[[0]].copy()

    time_005 = pd.concat(
        [pilot, high[high["ivns_time"].round(2) == 0.05]],
        ignore_index=True,
    )
    time_005_ranking = ranking(time_005)

    pilot_ranking.to_csv(tables_dir / "f_pilot_ranking.csv", index=False)
    high_ranking.to_csv(tables_dir / "f_high_ranking.csv", index=False)
    high_best_by_time.to_csv(tables_dir / "f_high_best_by_time.csv", index=False)
    selected.to_csv(tables_dir / "f_selected_configuration.csv", index=False)
    time_005_ranking.to_csv(tables_dir / "f_profiles_time_0.05.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(time_005_ranking))
    ax.bar(x, time_005_ranking["gap_median"])
    ax.set_xticks(x)
    ax.set_xticklabels(time_005_ranking["configuration"], rotation=45, ha="right")
    ax.set_title("F-n135-k7: perfiles con tiempo 0.05")
    ax.set_ylabel("Mediana del gap")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "13_f_profiles_time_005.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    lineplot(high_ranking, "vnd_mode", "F-n135-k7: efg-high por modo y tiempo", figure_dir / "14_f_high_by_time.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(high_ranking["time_median"], high_ranking["gap_median"])
    for _, row in high_ranking.iterrows():
        ax.annotate(row["configuration"], (row["time_median"], row["gap_median"]), fontsize=7)
    ax.set_xlabel("Mediana del tiempo")
    ax.set_ylabel("Mediana del gap")
    ax.set_title("F-n135-k7: relación entre calidad y tiempo")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "15_f_gap_time.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("Configuración seleccionada F =", selected.iloc[0]["configuration"])
    print("Tablas guardadas en", tables_dir)
    print("Figuras guardadas en", figure_dir)


if __name__ == "__main__":
    main()
