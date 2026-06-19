#!/usr/bin/env python3
"""Se analiza la sensibilidad de IVNS a las variantes y tiempos en las instancias A."""

from __future__ import annotations

from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_common import (
    METRICS,
    apply_holm,
    cliffs_delta_lower_is_better,
    descriptive_table,
    find_project_root,
    mannwhitney,
    output_dirs,
    safe_kruskal,
)


def save_line(frame: pd.DataFrame, group: str, metric: str, title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, group_frame in frame.groupby(group, sort=True):
        ordered = group_frame.sort_values("ivns_time")
        ax.plot(ordered["ivns_time"], ordered[metric], marker="o", label=label)
    ax.set_title(title)
    ax.set_xlabel("Tiempo por vecindad")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_heatmap(frame: pd.DataFrame, path) -> None:
    pivot = frame.pivot(index="variant", columns="ivns_time", values="gap_median")
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(pivot.to_numpy(), aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{value:.2f}" for value in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Tiempo por vecindad")
    ax.set_title("Mediana del gap por variante y tiempo")
    fig.colorbar(image, ax=ax, label="Mediana del gap")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = find_project_root()
    clean_dir, tables_dir, figures_dir = output_dirs(root)
    figure_dir = figures_dir / "exploratory_a"
    figure_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(clean_dir / "summary_a_exploratory.csv")
    frame["variant"] = frame["ivns_profile"] + "-" + frame["vnd_mode"]
    frame["configuration"] = frame["variant"] + "-t" + frame["ivns_time"].map(lambda value: f"{value:.2f}")

    ranking = (
        frame.groupby(["configuration", "variant", "ivns_time"])
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
        .sort_values(["gap_median", "gap_mean", "time_median"])
        .reset_index(drop=True)
    )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    ranking.to_csv(tables_dir / "a_exploratory_ranking.csv", index=False)

    by_instance = descriptive_table(
        frame,
        ["instance", "configuration", "variant", "ivns_time"],
    )
    by_instance.to_csv(tables_dir / "a_exploratory_by_instance.csv", index=False)

    best_by_time = (
        ranking.sort_values(["ivns_time", "gap_median", "gap_mean", "time_median"])
        .groupby("ivns_time", as_index=False)
        .first()
        .sort_values("ivns_time")
    )
    best_by_time.to_csv(tables_dir / "a_exploratory_best_by_time.csv", index=False)

    selected = ranking.iloc[[0]].copy()
    selected.to_csv(tables_dir / "a_exploratory_selected_configuration.csv", index=False)
    selected_variant = str(selected.iloc[0]["variant"])

    global_rows: list[dict[str, object]] = []
    best_posthoc_rows: list[dict[str, object]] = []
    for (time_value, instance), group in frame.groupby(["ivns_time", "instance"], sort=True):
        variants = sorted(group["variant"].unique())
        best_variant = str(best_by_time.loc[best_by_time["ivns_time"] == time_value, "variant"].iloc[0])
        for metric in METRICS:
            statistic, p_value = safe_kruskal(
                [group.loc[group["variant"] == variant, metric] for variant in variants]
            )
            global_rows.append(
                {
                    "ivns_time": time_value,
                    "instance": instance,
                    "metric": metric,
                    "groups": len(variants),
                    "statistic": statistic,
                    "p_value": p_value,
                }
            )
            for other in variants:
                if other == best_variant:
                    continue
                x = group.loc[group["variant"] == best_variant, metric]
                y = group.loc[group["variant"] == other, metric]
                pair_statistic, pair_p = mannwhitney(x, y)
                best_posthoc_rows.append(
                    {
                        "ivns_time": time_value,
                        "instance": instance,
                        "metric": metric,
                        "best_variant": best_variant,
                        "other_variant": other,
                        "median_best": float(pd.to_numeric(x, errors="coerce").median()),
                        "median_other": float(pd.to_numeric(y, errors="coerce").median()),
                        "statistic": pair_statistic,
                        "p_value": pair_p,
                        "cliffs_delta_positive_favors_best": cliffs_delta_lower_is_better(x, y),
                    }
                )

    global_table = pd.DataFrame(global_rows)
    global_table["decision"] = global_table["p_value"].apply(
        lambda value: "significativo" if value < 0.05 else "no_significativo"
    )
    global_table.to_csv(tables_dir / "a_exploratory_variants_kruskal.csv", index=False)
    apply_holm(
        pd.DataFrame(best_posthoc_rows),
        ["ivns_time", "instance", "metric"],
    ).to_csv(tables_dir / "a_exploratory_best_vs_rest_holm.csv", index=False)

    selected_frame = frame[frame["variant"] == selected_variant].copy()
    time_global_rows: list[dict[str, object]] = []
    time_posthoc_rows: list[dict[str, object]] = []
    times = sorted(selected_frame["ivns_time"].unique())
    for instance, group in selected_frame.groupby("instance", sort=True):
        for metric in METRICS:
            statistic, p_value = safe_kruskal(
                [group.loc[group["ivns_time"] == time_value, metric] for time_value in times]
            )
            time_global_rows.append(
                {
                    "variant": selected_variant,
                    "instance": instance,
                    "metric": metric,
                    "groups": len(times),
                    "statistic": statistic,
                    "p_value": p_value,
                }
            )
            for left, right in combinations(times, 2):
                x = group.loc[group["ivns_time"] == left, metric]
                y = group.loc[group["ivns_time"] == right, metric]
                pair_statistic, pair_p = mannwhitney(x, y)
                time_posthoc_rows.append(
                    {
                        "variant": selected_variant,
                        "instance": instance,
                        "metric": metric,
                        "time_a": left,
                        "time_b": right,
                        "median_a": float(pd.to_numeric(x, errors="coerce").median()),
                        "median_b": float(pd.to_numeric(y, errors="coerce").median()),
                        "statistic": pair_statistic,
                        "p_value": pair_p,
                        "cliffs_delta_positive_favors_a": cliffs_delta_lower_is_better(x, y),
                    }
                )

    pd.DataFrame(time_global_rows).to_csv(tables_dir / "a_exploratory_selected_times_kruskal.csv", index=False)
    apply_holm(
        pd.DataFrame(time_posthoc_rows),
        ["instance", "metric"],
    ).to_csv(tables_dir / "a_exploratory_selected_times_posthoc_holm.csv", index=False)

    best_series = best_by_time.copy()
    best_series["series"] = "mejor configuración"
    save_line(
        best_series,
        "series",
        "gap_median",
        "Mejor mediana del gap por tiempo",
        figure_dir / "05_a_best_gap_by_time.png",
    )
    save_heatmap(ranking, figure_dir / "06_a_gap_heatmap.png")

    baseline = (
        frame[frame["ivns_profile"] == "baseline"]
        .groupby(["variant", "ivns_time"])
        .agg(gap_median=("gap_percent", "median"))
        .reset_index()
    )
    save_line(
        baseline,
        "variant",
        "gap_median",
        "Perfiles baseline por tiempo",
        figure_dir / "07_a_baseline_by_time.png",
    )

    selected_instance = (
        selected_frame.groupby(["instance", "ivns_time"])
        .agg(gap_median=("gap_percent", "median"))
        .reset_index()
    )
    save_line(
        selected_instance,
        "instance",
        "gap_median",
        f"{selected_variant}: gap por instancia y tiempo",
        figure_dir / "08_a_selected_by_instance.png",
    )

    print("Configuración seleccionada =", selected.iloc[0]["configuration"])
    print("Tablas guardadas en", tables_dir)
    print("Figuras guardadas en", figure_dir)


if __name__ == "__main__":
    main()
