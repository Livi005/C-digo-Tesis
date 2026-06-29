from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(
    "results/p2-constructores-iniciales"
)

OUTPUT = Path(
    "results/analisis-p2-constructores-iniciales"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)

FILES = {
    "random": (
        ROOT
        / "random"
        / "summary-best-vns-vs-best-ivns.csv"
    ),
    "deterministic": (
        ROOT
        / "deterministic"
        / "summary-best-vns-vs-best-ivns.csv"
    ),
}

INSTANCES = [
    "A-n33-k5",
    "A-n65-k9",
    "A-n80-k10",
]

EXPECTED_SEEDS = set(
    range(91, 121)
)


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


def holm(
    values: list[float],
) -> list[float]:

    count = len(values)

    order = sorted(
        range(count),
        key=lambda index: values[index],
    )

    adjusted = [1.0] * count
    previous = 0.0

    for position, original_index in enumerate(
        order
    ):
        current = min(
            1.0,
            (count - position)
            * values[original_index],
        )

        current = max(
            previous,
            current,
        )

        adjusted[original_index] = current
        previous = current

    return adjusted


def wilcoxon_and_effect(
    differences: pd.Series,
) -> tuple[float, float, float]:

    values = pd.to_numeric(
        differences,
        errors="coerce",
    ).dropna()

    values = values[
        values.abs() > 1e-12
    ]

    if values.empty:
        return 0.0, 1.0, 0.0

    result = stats.wilcoxon(
        values,
        zero_method="wilcox",
        alternative="two-sided",
        method="auto",
    )

    ranks = stats.rankdata(
        values.abs()
    )

    positive = float(
        ranks[
            values.to_numpy() > 0
        ].sum()
    )

    negative = float(
        ranks[
            values.to_numpy() < 0
        ].sum()
    )

    effect = (
        (positive - negative)
        / (positive + negative)
    )

    return (
        float(result.statistic),
        float(result.pvalue),
        effect,
    )


errors: list[str] = []
pair_frames: list[pd.DataFrame] = []
raw_frames: list[pd.DataFrame] = []


for constructor, path in FILES.items():

    if not path.exists():
        errors.append(
            f"No existe {path}"
        )
        continue

    frame = pd.read_csv(path)

    frame.columns = [
        str(column)
        .strip()
        .lower()
        for column in frame.columns
    ]

    frame["constructor"] = constructor

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
        errors="coerce",
    ).astype("Int64")

    if len(frame) != 180:
        errors.append(
            f"{constructor}: "
            f"{len(frame)} filas; "
            "se esperaban 180."
        )

    if set(frame["algorithm"]) != {
        "vns_best",
        "ivns_best",
    }:
        errors.append(
            f"{constructor}: "
            "algoritmos inesperados."
        )

    if set(frame["instance"]) != set(
        INSTANCES
    ):
        errors.append(
            f"{constructor}: "
            "instancias inesperadas."
        )

    actual_seeds = set(
        frame["seed"]
        .dropna()
        .astype(int)
    )

    if actual_seeds != EXPECTED_SEEDS:
        errors.append(
            f"{constructor}: "
            "semillas inesperadas."
        )

    statuses_ok = (
        frame["status"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "ok"
    )

    if not statuses_ok.all():
        errors.append(
            f"{constructor}: "
            "existen estados distintos de OK."
        )

    if not frame["feasible"].map(
        truth
    ).all():
        errors.append(
            f"{constructor}: "
            "existen soluciones no factibles."
        )

    if frame.duplicated(
        [
            "instance",
            "seed",
            "algorithm",
        ]
    ).any():
        errors.append(
            f"{constructor}: "
            "existen corridas duplicadas."
        )

    for column in [
        "initial_cost",
        "gap_percent",
        "elapsed_time_seconds",
        "best_cost",
    ]:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    pivot = frame.pivot(
        index=[
            "instance",
            "seed",
        ],
        columns="algorithm",
    )

    pairs = pd.DataFrame(
        index=pivot.index
    ).reset_index()

    for column in [
        "initial_cost",
        "initial_solution_signature",
        "gap_percent",
        "elapsed_time_seconds",
        "best_cost",
    ]:
        for algorithm in [
            "vns_best",
            "ivns_best",
        ]:
            pairs[
                f"{column}_{algorithm}"
            ] = pivot[
                (column, algorithm)
            ].to_numpy()

    pairs["constructor"] = constructor

    pairs["same_signature"] = (
        pairs[
            "initial_solution_signature_vns_best"
        ].astype(str)
        ==
        pairs[
            "initial_solution_signature_ivns_best"
        ].astype(str)
    )

    pairs["same_initial_cost"] = np.isclose(
        pairs["initial_cost_vns_best"],
        pairs["initial_cost_ivns_best"],
    )

    if len(pairs) != 90:
        errors.append(
            f"{constructor}: "
            f"{len(pairs)} pares; "
            "se esperaban 90."
        )

    if not pairs[
        "same_signature"
    ].all():
        errors.append(
            f"{constructor}: "
            "existen firmas iniciales diferentes."
        )

    if not pairs[
        "same_initial_cost"
    ].all():
        errors.append(
            f"{constructor}: "
            "existen costos iniciales diferentes."
        )

    pairs[
        "delta_gap_ivns_minus_vns"
    ] = (
        pairs["gap_percent_ivns_best"]
        - pairs["gap_percent_vns_best"]
    )

    pairs[
        "delta_time_ivns_minus_vns"
    ] = (
        pairs[
            "elapsed_time_seconds_ivns_best"
        ]
        - pairs[
            "elapsed_time_seconds_vns_best"
        ]
    )

    pair_frames.append(pairs)
    raw_frames.append(frame)


if errors:

    validation_text = (
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
        validation_text,
        encoding="utf-8",
    )

    print(
        f"VALIDACIÓN: ERROR "
        f"({len(errors)} problemas)"
    )

    for error in errors:
        print("-", error)

    raise SystemExit(1)


all_pairs = pd.concat(
    pair_frames,
    ignore_index=True,
)

all_raw = pd.concat(
    raw_frames,
    ignore_index=True,
)

all_pairs.to_csv(
    OUTPUT / "pares-p2.csv",
    index=False,
)


# Soluciones iniciales.
initial_rows = []

for (
    constructor,
    instance,
), group in all_pairs.groupby(
    [
        "constructor",
        "instance",
    ],
    sort=True,
):

    initial_rows.append(
        {
            "constructor": constructor,
            "instance": instance,
            "n": len(group),
            "unique_initial_signatures": (
                group[
                    "initial_solution_signature_vns_best"
                ].nunique()
            ),
            "unique_initial_costs": (
                group[
                    "initial_cost_vns_best"
                ].nunique()
            ),
            "median_initial_cost": (
                group[
                    "initial_cost_vns_best"
                ].median()
            ),
            "minimum_initial_cost": (
                group[
                    "initial_cost_vns_best"
                ].min()
            ),
            "maximum_initial_cost": (
                group[
                    "initial_cost_vns_best"
                ].max()
            ),
        }
    )


pd.DataFrame(
    initial_rows
).to_csv(
    OUTPUT / "soluciones-iniciales.csv",
    index=False,
)


# Estadística descriptiva.
descriptive_rows = []

for (
    constructor,
    instance,
    algorithm,
), group in all_raw.groupby(
    [
        "constructor",
        "instance",
        "algorithm",
    ],
    sort=True,
):

    for metric in [
        "gap_percent",
        "elapsed_time_seconds",
    ]:

        values = group[metric]

        descriptive_rows.append(
            {
                "constructor": constructor,
                "instance": instance,
                "algorithm": algorithm,
                "metric": metric,
                "n": len(values),
                "mean": values.mean(),
                "median": values.median(),
                "q1": values.quantile(0.25),
                "q3": values.quantile(0.75),
                "iqr": (
                    values.quantile(0.75)
                    - values.quantile(0.25)
                ),
                "minimum": values.min(),
                "maximum": values.max(),
            }
        )


pd.DataFrame(
    descriptive_rows
).to_csv(
    OUTPUT / "resumen-descriptivo.csv",
    index=False,
)


# VNS contra IVNS dentro de cada constructor.
within_rows = []

for constructor in FILES:

    for metric, delta_column in [
        (
            "gap",
            "delta_gap_ivns_minus_vns",
        ),
        (
            "time",
            "delta_time_ivns_minus_vns",
        ),
    ]:

        family_indexes = []

        for instance in INSTANCES:

            group = all_pairs[
                (
                    all_pairs["constructor"]
                    == constructor
                )
                &
                (
                    all_pairs["instance"]
                    == instance
                )
            ]

            differences = group[
                delta_column
            ]

            (
                statistic,
                p_value,
                effect,
            ) = wilcoxon_and_effect(
                differences
            )

            family_indexes.append(
                len(within_rows)
            )

            within_rows.append(
                {
                    "constructor": constructor,
                    "metric": metric,
                    "instance": instance,
                    "n_pairs": len(group),
                    "median_difference_ivns_minus_vns": (
                        differences.median()
                    ),
                    "wins_vns": int(
                        (
                            differences
                            > 1e-12
                        ).sum()
                    ),
                    "wins_ivns": int(
                        (
                            differences
                            < -1e-12
                        ).sum()
                    ),
                    "ties": int(
                        (
                            differences.abs()
                            <= 1e-12
                        ).sum()
                    ),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "holm_p": np.nan,
                    "rank_biserial_positive_favors_vns": (
                        effect
                    ),
                }
            )

        adjusted = holm(
            [
                within_rows[index][
                    "p_value"
                ]
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
    / "vns-vs-ivns-por-constructor.csv",
    index=False,
)


# Comparar la ventaja entre constructores.
wide = all_pairs.pivot(
    index=[
        "instance",
        "seed",
    ],
    columns="constructor",
    values=[
        "delta_gap_ivns_minus_vns",
        "delta_time_ivns_minus_vns",
    ],
).reset_index()


interaction_rows = []

for metric, delta_column in [
    (
        "gap",
        "delta_gap_ivns_minus_vns",
    ),
    (
        "time",
        "delta_time_ivns_minus_vns",
    ),
]:

    family_indexes = []

    for instance in INSTANCES:

        group = wide[
            wide["instance"]
            == instance
        ]

        change = (
            group[
                (
                    delta_column,
                    "deterministic",
                )
            ]
            -
            group[
                (
                    delta_column,
                    "random",
                )
            ]
        )

        (
            statistic,
            p_value,
            effect,
        ) = wilcoxon_and_effect(
            change
        )

        family_indexes.append(
            len(interaction_rows)
        )

        interaction_rows.append(
            {
                "metric": metric,
                "instance": instance,
                "n_pairs": len(group),
                "median_change_det_minus_random": (
                    change.median()
                ),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "holm_p": np.nan,
                "rank_biserial": effect,
            }
        )

    adjusted = holm(
        [
            interaction_rows[index][
                "p_value"
            ]
            for index
            in family_indexes
        ]
    )

    for index, value in zip(
        family_indexes,
        adjusted,
    ):
        interaction_rows[index][
            "holm_p"
        ] = value


interaction = pd.DataFrame(
    interaction_rows
)

interaction.to_csv(
    OUTPUT / "efecto-del-constructor.csv",
    index=False,
)


(
    OUTPUT / "validacion.txt"
).write_text(
    "VALIDACIÓN P2: TODO CORRECTO\n"
    "Corridas totales: 360\n"
    "Pares VNS-IVNS: 180\n"
    "Pares por constructor: 90\n"
    "Pareo por instancia, semilla, "
    "costo y firma inicial: correcto\n",
    encoding="utf-8",
)


print("=" * 72)
print("ANÁLISIS P2: CONSTRUCTORES INICIALES")
print("=" * 72)

print()
print("VALIDACIÓN: TODO CORRECTO")
print("Corridas: 360")
print("Pares: 180")

print()
print("CALIDAD: VNS CONTRA IVNS")

print(
    within[
        within["metric"] == "gap"
    ][
        [
            "constructor",
            "instance",
            "wins_vns",
            "wins_ivns",
            "holm_p",
            "rank_biserial_positive_favors_vns",
        ]
    ].to_string(
        index=False
    )
)

print()
print(
    "¿CAMBIA LA DIFERENCIA "
    "SEGÚN EL CONSTRUCTOR?"
)

print(
    interaction[
        interaction["metric"] == "gap"
    ][
        [
            "instance",
            "median_change_det_minus_random",
            "holm_p",
            "rank_biserial",
        ]
    ].to_string(
        index=False
    )
)

print()
print(
    "Archivos guardados en:"
)
print(OUTPUT)
