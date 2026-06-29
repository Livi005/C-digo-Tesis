from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


INPUT_DIR = Path(
    "results/confirmatorio-pareado-best-vns-best-ivns"
)

SUMMARY_FILE = (
    INPUT_DIR / "summary-best-vns-vs-best-ivns.csv"
)

TRACE_FILE = (
    INPUT_DIR / "trace-best-vns-vs-best-ivns.csv"
)

MANIFEST_FILE = (
    INPUT_DIR / "configuracion-experimento.txt"
)

OUTPUT_DIR = Path(
    "results/analisis-confirmatorio-best-vns-best-ivns"
)

VALIDATION_FILE = OUTPUT_DIR / "validacion.txt"
PAIRS_FILE = OUTPUT_DIR / "pares-vns-ivns.csv"
DESCRIPTIVE_FILE = OUTPUT_DIR / "resumen-descriptivo.csv"
TESTS_FILE = OUTPUT_DIR / "pruebas-wilcoxon.csv"
REPORT_FILE = OUTPUT_DIR / "informe-estadistico.md"
HASH_FILE = OUTPUT_DIR / "SHA256SUMS.txt"
METADATA_FILE = OUTPUT_DIR / "metadatos-analisis.json"


EXPECTED_ALGORITHMS = {
    "vns_best",
    "ivns_best",
}

EXPECTED_INSTANCES = {
    "A-n33-k5",
    "A-n65-k9",
    "A-n80-k10",
}

EXPECTED_SEEDS = {
    str(seed)
    for seed in range(61, 91)
}

EXPECTED_ROWS = 180
EXPECTED_PAIRS = 90


def normalize(value: Any) -> str:
    return str(value).strip()


def normalize_algorithm(value: Any) -> str:
    return normalize(value).lower()


def parse_float(value: Any, field: str) -> float:
    text = normalize(value)

    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(
            f"No se pudo convertir {field}={text!r}"
        ) from exc

    if not math.isfinite(number):
        raise ValueError(
            f"{field} no es finito: {text!r}"
        )

    return number


def parse_bool(value: Any) -> bool:
    return normalize(value).lower() in {
        "true",
        "t",
        "1",
        "1.0",
        "yes",
    }


def quantile(
    values: list[float],
    probability: float,
) -> float:
    ordered = sorted(values)

    if not ordered:
        return math.nan

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    fraction = position - lower

    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def iqr(values: list[float]) -> float:
    return (
        quantile(values, 0.75)
        - quantile(values, 0.25)
    )


def rank_absolute_differences(
    differences: list[float],
) -> list[float]:
    absolute_values = [
        abs(value)
        for value in differences
    ]

    ordered_indices = sorted(
        range(len(absolute_values)),
        key=lambda index: (
            absolute_values[index],
            index,
        ),
    )

    ranks = [0.0] * len(differences)
    position = 0

    while position < len(ordered_indices):
        end = position + 1

        current_value = absolute_values[
            ordered_indices[position]
        ]

        while (
            end < len(ordered_indices)
            and math.isclose(
                absolute_values[
                    ordered_indices[end]
                ],
                current_value,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            end += 1

        first_rank = position + 1
        last_rank = end
        average_rank = (
            first_rank + last_rank
        ) / 2.0

        for offset in range(position, end):
            original_index = ordered_indices[offset]
            ranks[original_index] = average_rank

        position = end

    return ranks


def wilcoxon_signed_rank_exact(
    raw_differences: list[float],
) -> dict[str, float | int]:
    """
    Prueba de rangos con signo.

    Diferencia definida como IVNS - VNS.

    Valor positivo:
      VNS obtuvo un valor menor y, por tanto, mejor.

    Valor negativo:
      IVNS obtuvo un valor menor y mejor.
    """

    differences = [
        value
        for value in raw_differences
        if not math.isclose(
            value,
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]

    n_nonzero = len(differences)

    if n_nonzero == 0:
        return {
            "n_nonzero": 0,
            "w_plus": 0.0,
            "w_minus": 0.0,
            "statistic": 0.0,
            "p_value": 1.0,
            "rank_biserial": 0.0,
        }

    ranks = rank_absolute_differences(
        differences
    )

    w_plus = sum(
        rank
        for difference, rank
        in zip(differences, ranks)
        if difference > 0
    )

    w_minus = sum(
        rank
        for difference, rank
        in zip(differences, ranks)
        if difference < 0
    )

    statistic = min(
        w_plus,
        w_minus,
    )

    total_rank = w_plus + w_minus

    rank_biserial = (
        (w_plus - w_minus) / total_rank
        if total_rank
        else 0.0
    )

    # Los rangos medios pueden terminar en .5.
    # Se multiplican por dos para trabajar con enteros.
    scaled_ranks = [
        int(round(rank * 2.0))
        for rank in ranks
    ]

    observed_positive = int(
        round(w_plus * 2.0)
    )

    distribution: dict[int, int] = {
        0: 1,
    }

    for rank in scaled_ranks:
        updated = dict(distribution)

        for current_sum, count in distribution.items():
            new_sum = current_sum + rank
            updated[new_sum] = (
                updated.get(new_sum, 0)
                + count
            )

        distribution = updated

    total_assignments = 2 ** n_nonzero

    lower_count = sum(
        count
        for rank_sum, count in distribution.items()
        if rank_sum <= observed_positive
    )

    upper_count = sum(
        count
        for rank_sum, count in distribution.items()
        if rank_sum >= observed_positive
    )

    lower_probability = (
        lower_count / total_assignments
    )

    upper_probability = (
        upper_count / total_assignments
    )

    p_value = min(
        1.0,
        2.0
        * min(
            lower_probability,
            upper_probability,
        ),
    )

    return {
        "n_nonzero": n_nonzero,
        "w_plus": w_plus,
        "w_minus": w_minus,
        "statistic": statistic,
        "p_value": p_value,
        "rank_biserial": rank_biserial,
    }


def holm_adjust(
    p_values: list[float],
) -> list[float]:
    count = len(p_values)

    ordered = sorted(
        range(count),
        key=lambda index: p_values[index],
    )

    adjusted = [1.0] * count
    previous = 0.0

    for order_position, original_index in enumerate(
        ordered
    ):
        multiplier = count - order_position

        current = min(
            1.0,
            p_values[original_index]
            * multiplier,
        )

        current = max(
            previous,
            current,
        )

        adjusted[original_index] = current
        previous = current

    return adjusted


def winner_from_counts(
    wins_vns: int,
    wins_ivns: int,
) -> str:
    if wins_vns > wins_ivns:
        return "vns_best"

    if wins_ivns > wins_vns:
        return "ivns_best"

    return "tie"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

errors: list[str] = []


with SUMMARY_FILE.open(
    "r",
    encoding="utf-8",
    newline="",
) as stream:
    rows = list(csv.DictReader(stream))


if len(rows) != EXPECTED_ROWS:
    errors.append(
        f"El summary contiene {len(rows)} filas; "
        f"se esperaban {EXPECTED_ROWS}."
    )


algorithm_counts = Counter(
    normalize_algorithm(row["algorithm"])
    for row in rows
)

for algorithm in EXPECTED_ALGORITHMS:
    if algorithm_counts[algorithm] != 90:
        errors.append(
            f"{algorithm}: "
            f"{algorithm_counts[algorithm]} corridas; "
            "se esperaban 90."
        )


groups: dict[
    tuple[str, str],
    list[dict[str, str]],
] = defaultdict(list)

for row in rows:
    instance = normalize(row["instance"])
    seed = normalize(row["seed"])

    groups[(instance, seed)].append(row)


if len(groups) != EXPECTED_PAIRS:
    errors.append(
        f"Se encontraron {len(groups)} pares; "
        f"se esperaban {EXPECTED_PAIRS}."
    )


pair_rows: list[dict[str, Any]] = []


for key, pair in sorted(groups.items()):
    instance, seed = key

    if instance not in EXPECTED_INSTANCES:
        errors.append(
            f"Instancia inesperada: {instance}"
        )

    if seed not in EXPECTED_SEEDS:
        errors.append(
            f"Semilla inesperada: {seed}"
        )

    if len(pair) != 2:
        errors.append(
            f"{key}: contiene {len(pair)} filas."
        )
        continue

    by_algorithm = {
        normalize_algorithm(row["algorithm"]): row
        for row in pair
    }

    if set(by_algorithm) != EXPECTED_ALGORITHMS:
        errors.append(
            f"{key}: algoritmos={set(by_algorithm)}."
        )
        continue

    vns = by_algorithm["vns_best"]
    ivns = by_algorithm["ivns_best"]

    signatures = {
        normalize(vns["initial_solution_signature"]),
        normalize(ivns["initial_solution_signature"]),
    }

    initial_costs = {
        normalize(vns["initial_cost"]),
        normalize(ivns["initial_cost"]),
    }

    if len(signatures) != 1:
        errors.append(
            f"{key}: firmas iniciales diferentes."
        )

    if len(initial_costs) != 1:
        errors.append(
            f"{key}: costos iniciales diferentes."
        )

    for algorithm, row in by_algorithm.items():
        if normalize(row["status"]).lower() != "ok":
            errors.append(
                f"{key}, {algorithm}: "
                f"status={row['status']!r}."
            )

        if not parse_bool(row["feasible"]):
            errors.append(
                f"{key}, {algorithm}: "
                "solución no factible."
            )

        if not parse_bool(
            row["initial_solution_shared_p"]
        ):
            errors.append(
                f"{key}, {algorithm}: "
                "no figura como solución compartida."
            )

    vns_gap = parse_float(
        vns["gap_percent"],
        "vns_gap",
    )

    ivns_gap = parse_float(
        ivns["gap_percent"],
        "ivns_gap",
    )

    vns_time = parse_float(
        vns["elapsed_time_seconds"],
        "vns_time",
    )

    ivns_time = parse_float(
        ivns["elapsed_time_seconds"],
        "ivns_time",
    )

    gap_difference = ivns_gap - vns_gap
    time_difference = ivns_time - vns_time

    pair_rows.append(
        {
            "instance": instance,
            "seed": seed,
            "initial_cost": (
                parse_float(
                    vns["initial_cost"],
                    "initial_cost",
                )
            ),
            "initial_solution_signature": (
                normalize(
                    vns[
                        "initial_solution_signature"
                    ]
                )
            ),
            "vns_best_cost": (
                parse_float(
                    vns["best_cost"],
                    "vns_best_cost",
                )
            ),
            "ivns_best_cost": (
                parse_float(
                    ivns["best_cost"],
                    "ivns_best_cost",
                )
            ),
            "vns_gap_percent": vns_gap,
            "ivns_gap_percent": ivns_gap,
            "gap_difference_ivns_minus_vns": (
                gap_difference
            ),
            "gap_winner": (
                "vns_best"
                if gap_difference > 1e-12
                else (
                    "ivns_best"
                    if gap_difference < -1e-12
                    else "tie"
                )
            ),
            "vns_time_seconds": vns_time,
            "ivns_time_seconds": ivns_time,
            "time_difference_ivns_minus_vns": (
                time_difference
            ),
            "time_winner": (
                "vns_best"
                if time_difference > 1e-12
                else (
                    "ivns_best"
                    if time_difference < -1e-12
                    else "tie"
                )
            ),
        }
    )


validation_lines = [
    "VALIDACIÓN DEL EXPERIMENTO PAREADO",
    "=================================",
    "",
    f"Filas summary: {len(rows)}",
    f"Pares instancia-semilla: {len(groups)}",
    f"Corridas VNS: {algorithm_counts['vns_best']}",
    f"Corridas IVNS: {algorithm_counts['ivns_best']}",
    "",
]


if errors:
    validation_lines.append(
        f"VALIDACIÓN: ERROR ({len(errors)} problemas)"
    )

    validation_lines.extend(
        f"- {error}"
        for error in errors
    )

    VALIDATION_FILE.write_text(
        "\n".join(validation_lines) + "\n",
        encoding="utf-8",
    )

    print(
        f"VALIDACIÓN: ERROR ({len(errors)} problemas)"
    )

    for error in errors[:50]:
        print("-", error)

    raise SystemExit(1)


validation_lines.extend(
    [
        "Soluciones iniciales: pareadas correctamente",
        "Estados: todos OK",
        "Factibilidad: todas las soluciones factibles",
        "",
        "VALIDACIÓN: TODO CORRECTO",
    ]
)

VALIDATION_FILE.write_text(
    "\n".join(validation_lines) + "\n",
    encoding="utf-8",
)


pair_fields = [
    "instance",
    "seed",
    "initial_cost",
    "initial_solution_signature",
    "vns_best_cost",
    "ivns_best_cost",
    "vns_gap_percent",
    "ivns_gap_percent",
    "gap_difference_ivns_minus_vns",
    "gap_winner",
    "vns_time_seconds",
    "ivns_time_seconds",
    "time_difference_ivns_minus_vns",
    "time_winner",
]

write_csv(
    PAIRS_FILE,
    pair_rows,
    pair_fields,
)


# Resumen descriptivo.
descriptive_rows: list[dict[str, Any]] = []

for instance in sorted(EXPECTED_INSTANCES):
    instance_pairs = [
        row
        for row in pair_rows
        if row["instance"] == instance
    ]

    for algorithm in [
        "vns_best",
        "ivns_best",
    ]:
        gap_column = (
            "vns_gap_percent"
            if algorithm == "vns_best"
            else "ivns_gap_percent"
        )

        time_column = (
            "vns_time_seconds"
            if algorithm == "vns_best"
            else "ivns_time_seconds"
        )

        gaps = [
            float(row[gap_column])
            for row in instance_pairs
        ]

        times = [
            float(row[time_column])
            for row in instance_pairs
        ]

        descriptive_rows.append(
            {
                "instance": instance,
                "algorithm": algorithm,
                "n": len(instance_pairs),
                "median_gap_percent": (
                    statistics.median(gaps)
                ),
                "iqr_gap_percent": iqr(gaps),
                "mean_gap_percent": (
                    statistics.mean(gaps)
                ),
                "minimum_gap_percent": min(gaps),
                "maximum_gap_percent": max(gaps),
                "median_time_seconds": (
                    statistics.median(times)
                ),
                "iqr_time_seconds": iqr(times),
                "mean_time_seconds": (
                    statistics.mean(times)
                ),
                "minimum_time_seconds": min(times),
                "maximum_time_seconds": max(times),
            }
        )


descriptive_fields = [
    "instance",
    "algorithm",
    "n",
    "median_gap_percent",
    "iqr_gap_percent",
    "mean_gap_percent",
    "minimum_gap_percent",
    "maximum_gap_percent",
    "median_time_seconds",
    "iqr_time_seconds",
    "mean_time_seconds",
    "minimum_time_seconds",
    "maximum_time_seconds",
]

write_csv(
    DESCRIPTIVE_FILE,
    descriptive_rows,
    descriptive_fields,
)


# Pruebas de Wilcoxon.
test_rows: list[dict[str, Any]] = []


for metric in [
    "gap",
    "time",
]:
    for instance in sorted(EXPECTED_INSTANCES):
        instance_pairs = [
            row
            for row in pair_rows
            if row["instance"] == instance
        ]

        difference_column = (
            "gap_difference_ivns_minus_vns"
            if metric == "gap"
            else "time_difference_ivns_minus_vns"
        )

        winner_column = (
            "gap_winner"
            if metric == "gap"
            else "time_winner"
        )

        differences = [
            float(row[difference_column])
            for row in instance_pairs
        ]

        winner_counts = Counter(
            row[winner_column]
            for row in instance_pairs
        )

        result = wilcoxon_signed_rank_exact(
            differences
        )

        test_rows.append(
            {
                "metric": metric,
                "scope": instance,
                "n_pairs": len(instance_pairs),
                "n_nonzero": result["n_nonzero"],
                "w_plus": result["w_plus"],
                "w_minus": result["w_minus"],
                "wilcoxon_statistic": (
                    result["statistic"]
                ),
                "p_value": result["p_value"],
                "holm_adjusted_p": "",
                "rank_biserial": (
                    result["rank_biserial"]
                ),
                "median_difference_ivns_minus_vns": (
                    statistics.median(differences)
                ),
                "wins_vns": (
                    winner_counts["vns_best"]
                ),
                "wins_ivns": (
                    winner_counts["ivns_best"]
                ),
                "ties": winner_counts["tie"],
                "winner_by_counts": winner_from_counts(
                    winner_counts["vns_best"],
                    winner_counts["ivns_best"],
                ),
            }
        )

    # Prueba global sobre los 90 pares.
    difference_column = (
        "gap_difference_ivns_minus_vns"
        if metric == "gap"
        else "time_difference_ivns_minus_vns"
    )

    winner_column = (
        "gap_winner"
        if metric == "gap"
        else "time_winner"
    )

    differences = [
        float(row[difference_column])
        for row in pair_rows
    ]

    winner_counts = Counter(
        row[winner_column]
        for row in pair_rows
    )

    result = wilcoxon_signed_rank_exact(
        differences
    )

    test_rows.append(
        {
            "metric": metric,
            "scope": "overall-90-pairs",
            "n_pairs": len(pair_rows),
            "n_nonzero": result["n_nonzero"],
            "w_plus": result["w_plus"],
            "w_minus": result["w_minus"],
            "wilcoxon_statistic": (
                result["statistic"]
            ),
            "p_value": result["p_value"],
            "holm_adjusted_p": "",
            "rank_biserial": (
                result["rank_biserial"]
            ),
            "median_difference_ivns_minus_vns": (
                statistics.median(differences)
            ),
            "wins_vns": winner_counts["vns_best"],
            "wins_ivns": winner_counts["ivns_best"],
            "ties": winner_counts["tie"],
            "winner_by_counts": winner_from_counts(
                winner_counts["vns_best"],
                winner_counts["ivns_best"],
            ),
        }
    )


# Holm por familia:
# tres pruebas de instancia para gap y tres para tiempo.
for metric in [
    "gap",
    "time",
]:
    indices = [
        index
        for index, row in enumerate(test_rows)
        if (
            row["metric"] == metric
            and row["scope"] in EXPECTED_INSTANCES
        )
    ]

    p_values = [
        float(test_rows[index]["p_value"])
        for index in indices
    ]

    adjusted = holm_adjust(p_values)

    for index, adjusted_p in zip(
        indices,
        adjusted,
    ):
        test_rows[index][
            "holm_adjusted_p"
        ] = adjusted_p


test_fields = [
    "metric",
    "scope",
    "n_pairs",
    "n_nonzero",
    "w_plus",
    "w_minus",
    "wilcoxon_statistic",
    "p_value",
    "holm_adjusted_p",
    "rank_biserial",
    "median_difference_ivns_minus_vns",
    "wins_vns",
    "wins_ivns",
    "ties",
    "winner_by_counts",
]

write_csv(
    TESTS_FILE,
    test_rows,
    test_fields,
)


metadata = {
    "analysis_timestamp": datetime.now().isoformat(
        timespec="seconds"
    ),
    "design": "paired",
    "pairing_variables": [
        "instance",
        "seed",
        "initial_solution_signature",
        "initial_cost",
    ],
    "difference_definition": "IVNS - VNS",
    "positive_difference_interpretation": (
        "VNS obtuvo menor valor"
    ),
    "multiple_comparison_adjustment": (
        "Holm por separado para las tres "
        "instancias de cada métrica"
    ),
    "algorithms": {
        "vns_best": "extended-8-k3-best",
        "ivns_best": "baseline-k2-first-t1.00",
    },
    "pairs": len(pair_rows),
}

METADATA_FILE.write_text(
    json.dumps(
        metadata,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)


# Informe legible.
report_lines = [
    "# Análisis confirmatorio pareado",
    "",
    f"**Fecha:** "
    f"{datetime.now().isoformat(timespec='seconds')}",
    "",
    "## Configuraciones",
    "",
    "- VNS: `extended-8-k3-best`.",
    "- IVNS: `baseline-k2-first-t1.00`.",
    "- Pares: 90.",
    "- Corridas totales: 180.",
    "- Semillas: 61–90.",
    "- Constructor: aleatorio compartido.",
    "",
    "Cada pareja partió de la misma solución inicial. "
    "VNS e IVNS recibieron copias independientes.",
    "",
    "## Interpretación de las diferencias",
    "",
    "Las diferencias se calcularon como `IVNS − VNS`.",
    "",
    "- Diferencia positiva: VNS obtuvo un valor menor.",
    "- Diferencia negativa: IVNS obtuvo un valor menor.",
    "- Para gap y tiempo, un valor menor representa "
    "un mejor resultado.",
    "",
    "## Resultados descriptivos",
    "",
    "| Instancia | Algoritmo | Mediana gap | IQR gap | "
    "Mediana tiempo | IQR tiempo |",
    "|---|---|---:|---:|---:|---:|",
]

for row in descriptive_rows:
    report_lines.append(
        f"| {row['instance']} "
        f"| {row['algorithm']} "
        f"| {float(row['median_gap_percent']):.6f} "
        f"| {float(row['iqr_gap_percent']):.6f} "
        f"| {float(row['median_time_seconds']):.6f} "
        f"| {float(row['iqr_time_seconds']):.6f} |"
    )


report_lines.extend(
    [
        "",
        "## Pruebas pareadas",
        "",
        "| Métrica | Alcance | p | Holm | Efecto | "
        "VNS gana | IVNS gana | Empates |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
)

for row in test_rows:
    holm_value = row["holm_adjusted_p"]

    holm_text = (
        f"{float(holm_value):.8g}"
        if holm_value != ""
        else "—"
    )

    report_lines.append(
        f"| {row['metric']} "
        f"| {row['scope']} "
        f"| {float(row['p_value']):.8g} "
        f"| {holm_text} "
        f"| {float(row['rank_biserial']):.6f} "
        f"| {row['wins_vns']} "
        f"| {row['wins_ivns']} "
        f"| {row['ties']} |"
    )


report_lines.extend(
    [
        "",
        "## Convención del tamaño de efecto",
        "",
        "El rango biserial se calculó a partir de la "
        "diferencia `IVNS − VNS`:",
        "",
        "- valor positivo: ventaja para VNS;",
        "- valor negativo: ventaja para IVNS;",
        "- magnitud próxima a 1: diferencia muy marcada;",
        "- magnitud próxima a 0: diferencia pequeña.",
        "",
        "La corrección de Holm se aplicó de forma "
        "separada a la familia de pruebas de calidad "
        "y a la familia de pruebas de tiempo.",
        "",
    ]
)

REPORT_FILE.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)


hash_paths = [
    SUMMARY_FILE,
    TRACE_FILE,
    MANIFEST_FILE,
    Path(__file__),
    Path("analysis/paired_best_vns_ivns_core.lisp"),
    Path("analysis/run_best_vns_vs_best_ivns_paired_full.lisp"),
    Path("analysis/best_vns_config.lisp"),
    Path("vrp-suite/vrp-vns.org"),
    Path("vrp-suite/experiments-vns-ivns.org"),
    Path("vrp-suite/ivns-grammar.org"),
]

hash_lines = []

for path in hash_paths:
    if path.exists():
        hash_lines.append(
            f"{sha256_file(path)}  {path}"
        )

HASH_FILE.write_text(
    "\n".join(hash_lines) + "\n",
    encoding="utf-8",
)


print("=" * 76)
print("ANÁLISIS CONFIRMATORIO PAREADO")
print("=" * 76)
print()
print("VALIDACIÓN: TODO CORRECTO")
print(f"Pares: {len(pair_rows)}")
print(f"Corridas: {len(rows)}")
print()

for row in test_rows:
    if row["scope"] == "overall-90-pairs":
        print(
            f"{row['metric'].upper()}: "
            f"p={float(row['p_value']):.8g}, "
            f"efecto={float(row['rank_biserial']):.6f}, "
            f"VNS gana={row['wins_vns']}, "
            f"IVNS gana={row['wins_ivns']}, "
            f"empates={row['ties']}"
        )

print()
print("ARCHIVOS:")
print(f"  {VALIDATION_FILE}")
print(f"  {PAIRS_FILE}")
print(f"  {DESCRIPTIVE_FILE}")
print(f"  {TESTS_FILE}")
print(f"  {REPORT_FILE}")
print(f"  {METADATA_FILE}")
print(f"  {HASH_FILE}")
