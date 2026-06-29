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

EXPECTED_LOCAL_LIMIT = {
    "local1": 1.0,
    "local2": 2.0,
}

OUTPUT = Path(
    "results/analisis-equal-time-local1-vs-local2"
)

BUDGETS = [
    2.0,
    5.0,
    10.0,
]

INSTANCES = [
    "A-n33-k5",
    "A-n65-k9",
    "A-n80-k10",
]

ALGORITHMS = [
    "vns_best",
    "ivns_best",
]


def truth(value: object) -> bool:
    return (
        str(value)
        .strip()
        .lower()
        in {
            "true",
            "t",
            "1",
            "1.0",
            "yes",
        }
    )


def calculate_iqr(values: pd.Series) -> float:
    return float(
        values.quantile(0.75)
        - values.quantile(0.25)
    )


def holm_adjust(
    p_values: list[float],
) -> list[float]:
    count = len(p_values)

    order = sorted(
        range(count),
        key=lambda index: p_values[index],
    )

    adjusted = [1.0] * count
    previous = 0.0

    for position, original_index in enumerate(order):
        value = min(
            1.0,
            (count - position)
            * p_values[original_index],
        )

        value = max(
            previous,
            value,
        )

        adjusted[original_index] = value
        previous = value

    return adjusted


def wilcoxon_with_effect(
    differences: pd.Series,
) -> dict[str, float | int]:
    values = pd.to_numeric(
        differences,
        errors="coerce",
    ).dropna().to_numpy(dtype=float)

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

    result = wilcoxon(
        values,
        zero_method="wilcox",
        alternative="two-sided",
        method="auto",
    )

    ranks = rankdata(
        np.abs(values)
    )

    positive_rank = float(
        ranks[values > 0].sum()
    )

    negative_rank = float(
        ranks[values < 0].sum()
    )

    total_rank = (
        positive_rank
        + negative_rank
    )

    effect = (
        (
            positive_rank
            - negative_rank
        )
        / total_rank
        if total_rank
        else 0.0
    )

    return {
        "n_nonzero": len(values),
        "statistic": float(
            result.statistic
        ),
        "p_value": float(
            result.pvalue
        ),
        "rank_biserial": effect,
    }


OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)

errors: list[str] = []
summary_frames: dict[
    str,
    pd.DataFrame,
] = {}

cutoff_frames: list[
    pd.DataFrame
] = []


for experiment, directory in EXPERIMENTS.items():
    summary_path = (
        directory
        / "summary-best-vns-vs-best-ivns.csv"
    )

    trace_path = (
        directory
        / "trace-best-vns-vs-best-ivns.csv"
    )

    if not summary_path.exists():
        errors.append(
            f"No existe {summary_path}"
        )
        continue

    if not trace_path.exists():
        errors.append(
            f"No existe {trace_path}"
        )
        continue

    summary = pd.read_csv(
        summary_path
    )

    summary["algorithm"] = (
        summary["algorithm"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    summary["instance"] = (
        summary["instance"]
        .astype(str)
        .str.strip()
    )

    summary["seed"] = pd.to_numeric(
        summary["seed"],
        errors="raise",
    ).astype(int)

    if len(summary) != 180:
        errors.append(
            f"{experiment}: "
            f"{len(summary)} corridas; "
            "se esperaban 180."
        )

    if summary["run_id"].duplicated().any():
        errors.append(
            f"{experiment}: "
            "existen run_id duplicados."
        )

    if set(summary["algorithm"]) != set(
        ALGORITHMS
    ):
        errors.append(
            f"{experiment}: "
            "algoritmos inesperados."
        )

    if set(summary["instance"]) != set(
        INSTANCES
    ):
        errors.append(
            f"{experiment}: "
            "instancias inesperadas."
        )

    statuses = (
        summary["status"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if not (
        statuses == "ok"
    ).all():
        errors.append(
            f"{experiment}: "
            "existen estados distintos de OK."
        )

    if not summary["feasible"].map(
        truth
    ).all():
        errors.append(
            f"{experiment}: "
            "existen soluciones no factibles."
        )

    reasons = (
        summary["termination_reason"]
        .astype(str)
        .str.strip()
    )

    if not (
        reasons == "time_limit"
    ).all():
        errors.append(
            f"{experiment}: "
            "no todas las corridas terminaron "
            "por time_limit."
        )

    global_limits = set(
        pd.to_numeric(
            summary["time_limit_seconds"],
            errors="raise",
        ).round(9)
    )

    if global_limits != {10.0}:
        errors.append(
            f"{experiment}: "
            f"límites globales={global_limits}; "
            "se esperaba 10."
        )

    ivns_rows = summary[
        summary["algorithm"]
        == "ivns_best"
    ]

    local_limits = set(
        pd.to_numeric(
            ivns_rows[
                "ivns_max_neighborhood_seconds"
            ],
            errors="raise",
        ).round(9)
    )

    expected_limit = (
        EXPECTED_LOCAL_LIMIT[
            experiment
        ]
    )

    if local_limits != {
        expected_limit
    }:
        errors.append(
            f"{experiment}: "
            f"límites locales={local_limits}; "
            f"se esperaba {expected_limit}."
        )

    pairs = summary.pivot(
        index=[
            "instance",
            "seed",
        ],
        columns="algorithm",
        values=(
            "initial_solution_signature"
        ),
    )

    if len(pairs) != 90:
        errors.append(
            f"{experiment}: "
            f"{len(pairs)} pares; "
            "se esperaban 90."
        )

    if not (
        pairs["vns_best"].astype(str)
        ==
        pairs["ivns_best"].astype(str)
    ).all():
        errors.append(
            f"{experiment}: "
            "VNS e IVNS no comparten "
            "la misma solución inicial."
        )

    summary_frames[
        experiment
    ] = summary

    trace = pd.read_csv(
        trace_path,
        usecols=[
            "run_id",
            "algorithm",
            "instance",
            "seed",
            "best_cost",
            "elapsed_time_seconds",
            "status",
        ],
    )

    trace["algorithm"] = (
        trace["algorithm"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    trace["instance"] = (
        trace["instance"]
        .astype(str)
        .str.strip()
    )

    trace["seed"] = pd.to_numeric(
        trace["seed"],
        errors="raise",
    ).astype(int)

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

    for budget in BUDGETS:
        eligible = trace[
            trace[
                "elapsed_time_seconds"
            ]
            <= budget
        ]

        aggregated = (
            eligible
            .groupby(
                "run_id",
                as_index=False,
            )
            .agg(
                best_cost_at_cutoff=(
                    "best_cost",
                    "min",
                ),
                last_trace_time=(
                    "elapsed_time_seconds",
                    "max",
                ),
                trace_rows_at_cutoff=(
                    "best_cost",
                    "size",
                ),
            )
        )

        cutoff = base.merge(
            aggregated,
            on="run_id",
            how="left",
        )

        cutoff[
            "best_cost_at_cutoff"
        ] = cutoff[
            "best_cost_at_cutoff"
        ].fillna(
            cutoff["initial_cost"]
        )

        cutoff[
            "last_trace_time"
        ] = cutoff[
            "last_trace_time"
        ].fillna(0.0)

        cutoff[
            "trace_rows_at_cutoff"
        ] = cutoff[
            "trace_rows_at_cutoff"
        ].fillna(0).astype(int)

        cutoff[
            "gap_percent_at_cutoff"
        ] = (
            (
                cutoff[
                    "best_cost_at_cutoff"
                ]
                - cutoff["known_best"]
            )
            / cutoff["known_best"]
            * 100.0
        )

        cutoff[
            "experiment"
        ] = experiment

        cutoff[
            "budget_seconds"
        ] = budget

        cutoff_frames.append(
            cutoff
        )


if not errors:
    first = summary_frames[
        "local1"
    ][
        [
            "instance",
            "seed",
            "algorithm",
            "initial_solution_signature",
        ]
    ].copy()

    second = summary_frames[
        "local2"
    ][
        [
            "instance",
            "seed",
            "algorithm",
            "initial_solution_signature",
        ]
    ].copy()

    cross = first.merge(
        second,
        on=[
            "instance",
            "seed",
            "algorithm",
        ],
        suffixes=(
            "_local1",
            "_local2",
        ),
        how="outer",
        indicator=True,
    )

    if not (
        cross["_merge"] == "both"
    ).all():
        errors.append(
            "Los experimentos local1 y local2 "
            "no contienen las mismas corridas."
        )

    if not (
        cross[
            "initial_solution_signature_local1"
        ].astype(str)
        ==
        cross[
            "initial_solution_signature_local2"
        ].astype(str)
    ).all():
        errors.append(
            "Las soluciones iniciales cambian "
            "entre local1 y local2."
        )


if errors:
    validation = (
        "VALIDACIÓN: ERROR\n"
        + "\n".join(
            f"- {error}"
            for error in errors
        )
        + "\n"
    )

    (
        OUTPUT / "validacion.txt"
    ).write_text(
        validation,
        encoding="utf-8",
    )

    print(validation)

    raise SystemExit(1)


cutoffs = pd.concat(
    cutoff_frames,
    ignore_index=True,
)

cutoffs.to_csv(
    OUTPUT
    / "cutoffs-por-corrida.csv",
    index=False,
)


descriptive_rows = []

for (
    experiment,
    budget,
    instance,
    algorithm,
), group in cutoffs.groupby(
    [
        "experiment",
        "budget_seconds",
        "instance",
        "algorithm",
    ],
    sort=True,
):
    gaps = group[
        "gap_percent_at_cutoff"
    ]

    descriptive_rows.append(
        {
            "experiment": experiment,
            "budget_seconds": budget,
            "instance": instance,
            "algorithm": algorithm,
            "n": len(group),
            "mean_gap": float(
                gaps.mean()
            ),
            "median_gap": float(
                gaps.median()
            ),
            "iqr_gap": (
                calculate_iqr(gaps)
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
    OUTPUT
    / "descriptivo-checkpoints.csv",
    index=False,
)


within_rows = []

for experiment in EXPERIMENTS:
    for budget in BUDGETS:
        family_indexes = []

        for instance in INSTANCES:
            selected = cutoffs[
                (
                    cutoffs["experiment"]
                    == experiment
                )
                &
                (
                    cutoffs[
                        "budget_seconds"
                    ]
                    == budget
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
                values=(
                    "gap_percent_at_cutoff"
                ),
            )

            difference = (
                pivot["ivns_best"]
                - pivot["vns_best"]
            )

            result = (
                wilcoxon_with_effect(
                    difference
                )
            )

            family_indexes.append(
                len(within_rows)
            )

            within_rows.append(
                {
                    "experiment": experiment,
                    "budget_seconds": budget,
                    "instance": instance,
                    "n_pairs": len(pivot),
                    "median_difference_ivns_minus_vns": (
                        float(
                            difference.median()
                        )
                    ),
                    "wins_vns": int(
                        (
                            difference
                            > 1e-12
                        ).sum()
                    ),
                    "wins_ivns": int(
                        (
                            difference
                            < -1e-12
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
                    "n_nonzero": (
                        result["n_nonzero"]
                    ),
                    "wilcoxon_statistic": (
                        result["statistic"]
                    ),
                    "p_value": (
                        result["p_value"]
                    ),
                    "holm_p": np.nan,
                    "rank_biserial_positive_favors_vns": (
                        result[
                            "rank_biserial"
                        ]
                    ),
                }
            )

        adjusted = holm_adjust(
            [
                float(
                    within_rows[index][
                        "p_value"
                    ]
                )
                for index
                in family_indexes
            ]
        )

        for index, value in zip(
            family_indexes,
            adjusted,
        ):
            within_rows[index][
                "holm_p"
            ] = value


within = pd.DataFrame(
    within_rows
)

within.to_csv(
    OUTPUT
    / "wilcoxon-vns-vs-ivns.csv",
    index=False,
)


between_rows = []

for budget in BUDGETS:
    family_indexes = []

    for instance in INSTANCES:
        selected = cutoffs[
            (
                cutoffs["algorithm"]
                == "ivns_best"
            )
            &
            (
                cutoffs[
                    "budget_seconds"
                ]
                == budget
            )
            &
            (
                cutoffs["instance"]
                == instance
            )
        ]

        pivot = selected.pivot(
            index="seed",
            columns="experiment",
            values=(
                "gap_percent_at_cutoff"
            ),
        )

        # Positivo: local2 tiene menor gap.
        difference = (
            pivot["local1"]
            - pivot["local2"]
        )

        result = (
            wilcoxon_with_effect(
                difference
            )
        )

        family_indexes.append(
            len(between_rows)
        )

        between_rows.append(
            {
                "budget_seconds": budget,
                "instance": instance,
                "n_pairs": len(pivot),
                "median_improvement_local1_minus_local2": (
                    float(
                        difference.median()
                    )
                ),
                "wins_local2": int(
                    (
                        difference
                        > 1e-12
                    ).sum()
                ),
                "wins_local1": int(
                    (
                        difference
                        < -1e-12
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
                "n_nonzero": (
                    result["n_nonzero"]
                ),
                "wilcoxon_statistic": (
                    result["statistic"]
                ),
                "p_value": (
                    result["p_value"]
                ),
                "holm_p": np.nan,
                "rank_biserial_positive_favors_local2": (
                    result[
                        "rank_biserial"
                    ]
                ),
            }
        )

    adjusted = holm_adjust(
        [
            float(
                between_rows[index][
                    "p_value"
                ]
            )
            for index
            in family_indexes
        ]
    )

    for index, value in zip(
        family_indexes,
        adjusted,
    ):
        between_rows[index][
            "holm_p"
        ] = value


between = pd.DataFrame(
    between_rows
)

between.to_csv(
    OUTPUT
    / "wilcoxon-ivns-local2-vs-local1.csv",
    index=False,
)


time_rows = []

for experiment, summary in (
    summary_frames.items()
):
    for algorithm in ALGORITHMS:
        selected = summary[
            summary["algorithm"]
            == algorithm
        ]

        elapsed = pd.to_numeric(
            selected[
                "elapsed_time_seconds"
            ],
            errors="raise",
        )

        timeouts = pd.to_numeric(
            selected[
                "timeout_neighbors"
            ],
            errors="coerce",
        ).fillna(0)

        time_rows.append(
            {
                "experiment": experiment,
                "algorithm": algorithm,
                "runs": len(selected),
                "median_elapsed": float(
                    elapsed.median()
                ),
                "maximum_elapsed": float(
                    elapsed.max()
                ),
                "runs_over_10_seconds": int(
                    (
                        elapsed > 10.0
                    ).sum()
                ),
                "total_timeout_neighbors": int(
                    timeouts.sum()
                ),
                "maximum_timeouts_in_one_run": int(
                    timeouts.max()
                ),
            }
        )


times = pd.DataFrame(
    time_rows
)

times.to_csv(
    OUTPUT
    / "tiempos-y-timeouts.csv",
    index=False,
)


validation = (
    "VALIDACIÓN: TODO CORRECTO\n"
    "Experimentos: local1 y local2\n"
    "Corridas por experimento: 180\n"
    "Pares por experimento: 90\n"
    "Tiempo global nominal: 10 segundos\n"
    "Límite local IVNS local1: 1 segundo\n"
    "Límite local IVNS local2: 2 segundos\n"
    "Soluciones iniciales VNS-IVNS: iguales\n"
    "Soluciones iniciales entre experimentos: iguales\n"
    "Cortes reconstruidos desde trace: 2, 5 y 10 segundos\n"
)

(
    OUTPUT / "validacion.txt"
).write_text(
    validation,
    encoding="utf-8",
)


pd.set_option(
    "display.max_columns",
    None,
)

pd.set_option(
    "display.width",
    220,
)


print("=" * 88)
print(
    "COMPARACIÓN CON IGUAL PRESUPUESTO TEMPORAL"
)
print("=" * 88)
print()
print(validation)


print(
    "VNS CONTRA IVNS A LOS 10 SEGUNDOS"
)

print(
    within[
        within["budget_seconds"]
        == 10.0
    ][
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
print(
    "IVNS LOCAL-2 CONTRA IVNS LOCAL-1 "
    "A LOS 10 SEGUNDOS"
)

print(
    between[
        between["budget_seconds"]
        == 10.0
    ][
        [
            "instance",
            "median_improvement_local1_minus_local2",
            "wins_local2",
            "wins_local1",
            "ties",
            "holm_p",
            "rank_biserial_positive_favors_local2",
        ]
    ].to_string(
        index=False
    )
)


print()
print("MEDIANAS DE GAP A LOS 10 SEGUNDOS")

print(
    descriptive[
        descriptive["budget_seconds"]
        == 10.0
    ][
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
print("TIEMPOS Y TIMEOUTS")

print(
    times.to_string(
        index=False
    )
)


print()
print("ARCHIVOS GUARDADOS EN:")
print(OUTPUT)
