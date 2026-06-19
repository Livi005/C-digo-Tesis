#!/usr/bin/env python3
"""Se comparan estadísticamente las variantes IVNS oficiales."""

from __future__ import annotations

from itertools import combinations

import pandas as pd

from analysis_common import (
    METRICS,
    apply_holm,
    find_project_root,
    mannwhitney,
    output_dirs,
    safe_kruskal,
    cliffs_delta_lower_is_better,
)


def main() -> None:
    root = find_project_root()
    clean_dir, tables_dir, _ = output_dirs(root)
    frame = pd.read_csv(clean_dir / "summary_official_variants.csv")
    frame["variant"] = frame["ivns_profile"] + "-" + frame["vnd_mode"]

    global_rows: list[dict[str, object]] = []
    posthoc_rows: list[dict[str, object]] = []

    for instance, instance_frame in frame.groupby("instance", sort=True):
        variants = sorted(instance_frame["variant"].unique())
        for metric in METRICS:
            groups = [instance_frame.loc[instance_frame["variant"] == variant, metric] for variant in variants]
            statistic, p_value = safe_kruskal(groups)
            global_rows.append(
                {
                    "instance": instance,
                    "metric": metric,
                    "groups": len(variants),
                    "statistic": statistic,
                    "p_value": p_value,
                }
            )
            for left, right in combinations(variants, 2):
                x = instance_frame.loc[instance_frame["variant"] == left, metric]
                y = instance_frame.loc[instance_frame["variant"] == right, metric]
                pair_statistic, pair_p = mannwhitney(x, y)
                posthoc_rows.append(
                    {
                        "instance": instance,
                        "metric": metric,
                        "variant_a": left,
                        "variant_b": right,
                        "median_a": float(pd.to_numeric(x, errors="coerce").median()),
                        "median_b": float(pd.to_numeric(y, errors="coerce").median()),
                        "statistic": pair_statistic,
                        "p_value": pair_p,
                        "cliffs_delta_positive_favors_a": cliffs_delta_lower_is_better(x, y),
                    }
                )

    global_table = pd.DataFrame(global_rows)
    global_table["decision"] = global_table["p_value"].apply(
        lambda value: "significativo" if value < 0.05 else "no_significativo"
    )
    posthoc_table = apply_holm(pd.DataFrame(posthoc_rows), ["instance", "metric"])

    global_table.to_csv(tables_dir / "official_variants_kruskal.csv", index=False)
    posthoc_table.to_csv(tables_dir / "official_variants_posthoc_holm.csv", index=False)

    print("Pruebas globales =", len(global_table))
    print("Comparaciones por pares =", len(posthoc_table))


if __name__ == "__main__":
    main()
