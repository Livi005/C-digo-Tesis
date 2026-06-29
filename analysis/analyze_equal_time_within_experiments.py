from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


EXPERIMENTS = {
    "local1": Path(
        "results/equal-time-best-vns-best-ivns-t10"
    ),
    "local2": Path(
        "results/equal-time-best-vns-best-ivns-global10-local2"
    ),
}

OUTPUT = Path(
    "results/analisis-equal-time-por-experimento"
)

INSTANCES = [
    "A-n33-k5",
    "A-n65-k9",
    "A-n80-k10",
]

BUDGET = 10.0


def holm_adjust(values):
    count = len(values)
    order = np.argsort(values)
    adjusted = np.ones(count)
    previous = 0.0

    for position, index in enumerate(order):
        value = min(
            1.0,
            (count - position)
            * values[index],
        )

        value = max(
            previous,
            value,
        )

        adjusted[index] = value
        previous = value

    return adjusted


def wilcoxon_effect(differences):
    values = np.asarray(
        differences,
        dtype=float,
    )

    values = values[
        ~np.isclose(
            values,
            0.0,
            atol=1e-12,
            rtol=0.0,
        )
    ]

    if len(values) == 0:
        return {
            "n_nonzero": 0,
            "statistic": 0.0,
            "p_value": 1.0,
            "rank_biserial": 0.0,
        }

    test = wilcoxon(
        values,
        zero_method="wilcox",
        alternative="two-sided",
        method="auto",
    )

    ranks = rankdata(
        np.abs(values)
    )

    positive = ranks[
        values > 0
    ].sum()

    negative = ranks[
        values < 0
    ].sum()

    total = positive + negative

    effect = (
        (positive - negative) / total
        if total > 0
        else 0.0
    )

    return {
        "n_nonzero": len(values),
        "statistic": float(
            test.statistic
        ),
        "p_value": float(
            test.pvalue
        ),
        "rank_biserial": float(
            effect
        ),
    }


OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)

all_cutoffs = []
validation_errors = []


for experiment, directory in EXPERIMENTS.items():

    summary_path = (
        directory
        / "summary-best-vns-vs-best-ivns.csv"
    )

    trace_path = (
        directory
        / "trace-best-vns-vs-best-ivns.csv"
    )

    summary = pd.read_csv(
        summary_path
    )

    trace = pd.read_csv(
        trace_path,
        usecols=[
            "run_id",
            "algorithm",
            "instance",
            "seed",
            "best_cost",
            "elapsed_time_seconds",
        ],
    )

    for frame in [
        summary,
        trace,
    ]:
        frame["algorithm"] = (
            frame["algorithm"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        frame["instance"] = (
            frame["instance"]
            .astype(str)
            .str.strip()
        )

        frame["seed"] = pd.to_numeric(
            frame["seed"],
            errors="raise",
        ).astype(int)

    if len(summary) != 180:
        validation_errors.append(
            f"{experiment}: "
            f"{len(summary)} corridas; "
            "se esperaban 180."
        )

    pair_signatures = summary.pivot(
        index=[
            "instance",
            "seed",
        ],
        columns="algorithm",
        values="initial_solution_signature",
    )

    if len(pair_signatures) != 90:
        validation_errors.append(
            f"{experiment}: "
            f"{len(pair_signatures)} pares; "
            "se esperaban 90."
        )

    if not (
        pair_signatures["vns_best"].astype(str)
        ==
        pair_signatures["ivns_best"].astype(str)
    ).all():
        validation_errors.append(
            f"{experiment}: "
            "VNS e IVNS no comparten "
            "la misma solución inicial."
        )

    trace["best_cost"] = pd.to_numeric(
        trace["best_cost"],
        errors="raise",
    )

    trace[
        "elapsed_time_seconds"
    ] = pd.to_numeric(
        trace["elapsed_time_seconds"],
        errors="raise",
    )

    eligible = trace[
        trace["elapsed_time_seconds"]
        <= BUDGET
    ]

    best_at_cutoff = (
        eligible
        .groupby(
            "run_id",
            as_index=False,
        )
        .agg(
            best_cost_at_10=(
                "best_cost",
                "min",
            ),
            last_trace_time=(
                "elapsed_time_seconds",
                "max",
            ),
        )
    )

    base = summary[
        [
            "run_id",
            "algorithm",
            "instance",
            "seed",
            "known_best",
            "initial_cost",
            "initial_solution_signature",
        ]
    ].copy()

    base["known_best"] = pd.to_numeric(
        base["known_best"],
        errors="raise",
    )

    base["initial_cost"] = pd.to_numeric(
        base["initial_cost"],
        errors="raise",
    )

    cutoff = base.merge(
        best_at_cutoff,
        on="run_id",
        how="left",
    )

    cutoff[
        "best_cost_at_10"
    ] = cutoff[
        "best_cost_at_10"
    ].fillna(
        cutoff["initial_cost"]
    )

    cutoff[
        "last_trace_time"
    ] = cutoff[
        "last_trace_time"
    ].fillna(0.0)

    cutoff["gap_at_10"] = (
        (
            cutoff["best_cost_at_10"]
            - cutoff["known_best"]
        )
        / cutoff["known_best"]
        * 100.0
    )

    cutoff["experiment"] = experiment

    all_cutoffs.append(
        cutoff
    )


if validation_errors:
    print("VALIDACIÓN: ERROR")

    for error in validation_errors:
        print("-", error)

    raise SystemExit(1)


cutoffs = pd.concat(
    all_cutoffs,
    ignore_index=True,
)

cutoffs.to_csv(
    OUTPUT / "resultados-a-10-segundos.csv",
    index=False,
)


descriptive_rows = []

for (
    experiment,
    instance,
    algorithm,
), group in cutoffs.groupby(
    [
        "experiment",
        "instance",
        "algorithm",
    ],
    sort=True,
):

    gaps = group["gap_at_10"]

    descriptive_rows.append(
        {
            "experiment": experiment,
            "instance": instance,
            "algorithm": algorithm,
            "n": len(group),
            "median_gap": float(
                gaps.median()
            ),
            "iqr_gap": float(
                gaps.quantile(0.75)
                - gaps.quantile(0.25)
            ),
            "minimum_gap": float(
                gaps.min()
            ),
            "maximum_gap": float(
                gaps.max()
            ),
        }
    )


descriptive = pd.DataFrame(
    descriptive_rows
)

descriptive.to_csv(
    OUTPUT / "descriptivo-a-10-segundos.csv",
    index=False,
)


comparison_rows = []

for experiment in EXPERIMENTS:

    indexes = []

    for instance in INSTANCES:

        selected = cutoffs[
            (
                cutoffs["experiment"]
                == experiment
            )
            &
            (
                cutoffs["instance"]
                == instance
            )
        ]

        pivot = selected.pivot(
            index="seed",
            columns="algorithm",
            values="gap_at_10",
        )

        difference = (
            pivot["ivns_best"]
            - pivot["vns_best"]
        )

        test = wilcoxon_effect(
            difference
        )

        indexes.append(
            len(comparison_rows)
        )

        comparison_rows.append(
            {
                "experiment": experiment,
                "instance": instance,
                "n_pairs": len(pivot),
                "median_difference_ivns_minus_vns": float(
                    difference.median()
                ),
                "wins_vns": int(
                    (
                        difference > 1e-12
                    ).sum()
                ),
                "wins_ivns": int(
                    (
                        difference < -1e-12
                    ).sum()
                ),
                "ties": int(
                    np.isclose(
                        difference,
                        0.0,
                        atol=1e-12,
                        rtol=0.0,
                    ).sum()
                ),
                "wilcoxon_statistic": (
                    test["statistic"]
                ),
                "p_value": (
                    test["p_value"]
                ),
                "holm_p": np.nan,
                "rank_biserial_positive_favors_vns": (
                    test["rank_biserial"]
                ),
            }
        )

    adjusted = holm_adjust(
        [
            comparison_rows[index][
                "p_value"
            ]
            for index in indexes
        ]
    )

    for index, value in zip(
        indexes,
        adjusted,
    ):
        comparison_rows[index][
            "holm_p"
        ] = value


comparisons = pd.DataFrame(
    comparison_rows
)

comparisons.to_csv(
    OUTPUT / "wilcoxon-vns-vs-ivns-a-10-segundos.csv",
    index=False,
)


# Comparación descriptiva entre los dos límites.
ivns = descriptive[
    descriptive["algorithm"]
    == "ivns_best"
].pivot(
    index="instance",
    columns="experiment",
    values="median_gap",
).reset_index()

ivns[
    "median_change_local2_minus_local1"
] = (
    ivns["local2"]
    - ivns["local1"]
)

ivns[
    "descriptive_better"
] = np.where(
    ivns[
        "median_change_local2_minus_local1"
    ] < 0,
    "local2",
    np.where(
        ivns[
            "median_change_local2_minus_local1"
        ] > 0,
        "local1",
        "tie",
    ),
)

ivns.to_csv(
    OUTPUT
    / "comparacion-descriptiva-ivns-local1-local2.csv",
    index=False,
)


print("=" * 94)
print("ANÁLISIS VÁLIDO POR EXPERIMENTO")
print("=" * 94)
print()
print(
    "Cada comparación VNS-IVNS usa "
    "soluciones iniciales compartidas."
)
print(
    "La comparación local1-local2 "
    "se presenta solo de forma descriptiva."
)
print()

print("VNS CONTRA IVNS A LOS 10 SEGUNDOS")
print(
    comparisons[
        [
            "experiment",
            "instance",
            "median_difference_ivns_minus_vns",
            "wins_vns",
            "wins_ivns",
            "ties",
            "holm_p",
            "rank_biserial_positive_favors_vns",
        ]
    ].to_string(
        index=False
    )
)

print()
print("MEDIANAS DE GAP A LOS 10 SEGUNDOS")
print(
    descriptive[
        [
            "experiment",
            "instance",
            "algorithm",
            "median_gap",
            "iqr_gap",
        ]
    ].to_string(
        index=False
    )
)

print()
print(
    "IVNS LOCAL-1 CONTRA LOCAL-2: "
    "COMPARACIÓN DESCRIPTIVA"
)
print(
    ivns.to_string(
        index=False
    )
)

print()
print("ARCHIVOS GUARDADOS EN:")
print(OUTPUT)
