from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


INPUT_DIR = Path("results/calibracion-vns-limpia")
OUTPUT_DIR = Path("results/analisis-calibracion-vns")

COMBINED_FILE = OUTPUT_DIR / "calibracion-vns-combinada.csv"
INSTANCE_SUMMARY_FILE = OUTPUT_DIR / "resumen-vns-por-instancia.csv"
RANKING_FILE = OUTPUT_DIR / "ranking-configuraciones-vns.csv"
WINNER_JSON_FILE = OUTPUT_DIR / "mejor-configuracion-vns.json"
REPORT_FILE = OUTPUT_DIR / "informe-calibracion-vns.md"
VALIDATION_FILE = OUTPUT_DIR / "validacion-calibracion-vns.txt"
HASH_FILE = OUTPUT_DIR / "SHA256SUMS.txt"

BEST_VNS_LISP = Path("analysis/best_vns_config.lisp")
BEST_VNS_LISP_COPY = OUTPUT_DIR / "best_vns_config.lisp"

EXPECTED_INSTANCES = {
    "A-n33-k5",
    "A-n65-k9",
    "A-n80-k10",
}

EXPECTED_SEEDS = {
    str(seed)
    for seed in range(1, 31)
}

EXPECTED_PROFILES = {
    "basic-3",
    "five-5",
    "extended-8",
}

EXPECTED_KMAX = {
    2,
    3,
    5,
}

EXPECTED_MODES = {
    "first",
    "best",
}

EXPECTED_RUNS_PER_CONFIGURATION = 90
EXPECTED_CONFIGURATIONS = 18
EXPECTED_TOTAL_RUNS = 1620

FILE_PATTERN = re.compile(
    r"summary-vns-"
    r"(basic-3|five-5|extended-8)"
    r"-k(2|3|5)"
    r"-(first|best)\.csv$"
)

PROFILE_TO_CRITERIA = {
    "basic-3": ":basic",
    "five-5": ":five",
    "extended-8": ":extended",
}

PROFILE_TO_COUNT = {
    "basic-3": 3,
    "five-5": 5,
    "extended-8": 8,
}

REQUIRED_COLUMNS = {
    "algorithm",
    "instance",
    "seed",
    "known_best",
    "max_iter",
    "kmax",
    "max_no_improve",
    "shake_max_tries",
    "vnd_mode",
    "vnd_max_rounds",
    "vnd_max_no_improve",
    "time_limit_seconds",
    "initial_cost",
    "initial_solution_signature",
    "best_cost",
    "gap_percent",
    "elapsed_time_seconds",
    "feasible",
    "status",
    "error",
}


def normalize(value: Any) -> str:
    return str(value).strip()


def normalize_symbol(value: Any) -> str:
    return normalize(value).lower().lstrip(":")


def parse_float(value: Any, field: str) -> float:
    text = normalize(value)

    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(
            f"No se pudo interpretar {field}={text!r}"
        ) from exc

    if not math.isfinite(number):
        raise ValueError(
            f"{field} no es finito: {text!r}"
        )

    return number


def parse_int(value: Any, field: str) -> int:
    number = parse_float(value, field)

    integer = int(number)

    if number != integer:
        raise ValueError(
            f"{field} no es entero: {number}"
        )

    return integer


def parse_bool(value: Any) -> bool:
    return normalize(value).lower() in {
        "true",
        "t",
        "1",
        "1.0",
        "yes",
    }


def quantile(values: list[float], probability: float) -> float:
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


def average_ranks(
    values: dict[str, float],
) -> dict[str, float]:
    """
    Asigna rango 1 al menor valor.

    Cuando existen empates, asigna el promedio de los
    rangos que ocuparían los valores empatados.
    """
    ordered = sorted(
        values.items(),
        key=lambda item: (item[1], item[0]),
    )

    result: dict[str, float] = {}

    index = 0

    while index < len(ordered):
        end = index + 1
        current_value = ordered[index][1]

        while (
            end < len(ordered)
            and math.isclose(
                ordered[end][1],
                current_value,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            end += 1

        first_rank = index + 1
        last_rank = end
        average_rank = (first_rank + last_rank) / 2.0

        for position in range(index, end):
            configuration = ordered[position][0]
            result[configuration] = average_rank

        index = end

    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def csv_write(
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
validation_lines: list[str] = []

summary_files = sorted(
    INPUT_DIR.glob("summary-vns-*.csv")
)

validation_lines.append(
    f"Archivos summary encontrados: {len(summary_files)}"
)

if len(summary_files) != EXPECTED_CONFIGURATIONS:
    errors.append(
        f"Se encontraron {len(summary_files)} summaries; "
        f"se esperaban {EXPECTED_CONFIGURATIONS}."
    )


records_by_configuration: dict[
    str,
    list[dict[str, Any]],
] = {}

records_by_block: dict[
    tuple[str, str],
    list[dict[str, Any]],
] = defaultdict(list)

all_records: list[dict[str, Any]] = []
original_fieldnames: list[str] | None = None
hash_paths: list[Path] = []

found_configurations: set[
    tuple[str, int, str]
] = set()


for path in summary_files:
    match = FILE_PATTERN.fullmatch(path.name)

    if match is None:
        errors.append(
            f"Nombre de summary no reconocido: {path.name}"
        )
        continue

    profile, kmax_text, mode = match.groups()
    kmax = int(kmax_text)

    configuration = (
        f"{profile}-k{kmax}-{mode}"
    )

    found_configurations.add(
        (profile, kmax, mode)
    )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    hash_paths.append(path)

    if original_fieldnames is None:
        original_fieldnames = list(fieldnames)

    missing_columns = (
        REQUIRED_COLUMNS.difference(fieldnames)
    )

    if missing_columns:
        errors.append(
            f"{path.name}: faltan columnas "
            f"{sorted(missing_columns)}."
        )
        continue

    if len(rows) != EXPECTED_RUNS_PER_CONFIGURATION:
        errors.append(
            f"{path.name}: contiene {len(rows)} corridas; "
            f"se esperaban {EXPECTED_RUNS_PER_CONFIGURATION}."
        )

    configuration_keys: set[
        tuple[str, str]
    ] = set()

    configuration_records: list[
        dict[str, Any]
    ] = []

    for row_number, row in enumerate(
        rows,
        start=2,
    ):
        instance = normalize(row["instance"])
        seed = normalize(row["seed"])

        key = (
            instance,
            seed,
        )

        if key in configuration_keys:
            errors.append(
                f"{path.name}: par duplicado "
                f"{instance}, seed={seed}."
            )

        configuration_keys.add(key)

        if instance not in EXPECTED_INSTANCES:
            errors.append(
                f"{path.name}:{row_number}: "
                f"instancia inesperada {instance!r}."
            )

        if seed not in EXPECTED_SEEDS:
            errors.append(
                f"{path.name}:{row_number}: "
                f"semilla inesperada {seed!r}."
            )

        algorithm = normalize_symbol(
            row["algorithm"]
        )

        if algorithm != "vns":
            errors.append(
                f"{path.name}:{row_number}: "
                f"algorithm={row['algorithm']!r}."
            )

        row_kmax = parse_int(
            row["kmax"],
            "kmax",
        )

        if row_kmax != kmax:
            errors.append(
                f"{path.name}:{row_number}: "
                f"kmax={row_kmax}, esperado={kmax}."
            )

        row_mode = normalize_symbol(
            row["vnd_mode"]
        )

        if row_mode != mode:
            errors.append(
                f"{path.name}:{row_number}: "
                f"vnd_mode={row_mode!r}, "
                f"esperado={mode!r}."
            )

        status = normalize(
            row["status"]
        ).lower()

        if status != "ok":
            errors.append(
                f"{path.name}:{row_number}: "
                f"status={status!r}."
            )

        if not parse_bool(row["feasible"]):
            errors.append(
                f"{path.name}:{row_number}: "
                "solución no factible."
            )

        error_text = normalize(
            row["error"]
        )

        if error_text:
            errors.append(
                f"{path.name}:{row_number}: "
                f"error={error_text!r}."
            )

        gap = parse_float(
            row["gap_percent"],
            "gap_percent",
        )

        elapsed_time = parse_float(
            row["elapsed_time_seconds"],
            "elapsed_time_seconds",
        )

        best_cost = parse_float(
            row["best_cost"],
            "best_cost",
        )

        initial_cost = parse_float(
            row["initial_cost"],
            "initial_cost",
        )

        signature = normalize(
            row["initial_solution_signature"]
        )

        if not signature:
            errors.append(
                f"{path.name}:{row_number}: "
                "firma inicial vacía."
            )

        record = dict(row)

        record.update(
            {
                "configuration": configuration,
                "profile": profile,
                "criteria_count": (
                    PROFILE_TO_COUNT[profile]
                ),
                "calibration_kmax": kmax,
                "calibration_vnd_mode": mode,
                "_gap": gap,
                "_elapsed_time": elapsed_time,
                "_best_cost": best_cost,
                "_initial_cost": initial_cost,
                "_initial_signature": signature,
            }
        )

        configuration_records.append(record)
        all_records.append(record)
        records_by_block[key].append(record)

    expected_keys = {
        (instance, seed)
        for instance in EXPECTED_INSTANCES
        for seed in EXPECTED_SEEDS
    }

    missing_keys = (
        expected_keys.difference(
            configuration_keys
        )
    )

    extra_keys = (
        configuration_keys.difference(
            expected_keys
        )
    )

    if missing_keys:
        errors.append(
            f"{path.name}: faltan "
            f"{len(missing_keys)} pares."
        )

    if extra_keys:
        errors.append(
            f"{path.name}: sobran "
            f"{len(extra_keys)} pares."
        )

    records_by_configuration[
        configuration
    ] = configuration_records

    validation_lines.append(
        f"{configuration}: "
        f"{len(configuration_records)} corridas"
    )


expected_configurations = {
    (profile, kmax, mode)
    for profile in EXPECTED_PROFILES
    for kmax in EXPECTED_KMAX
    for mode in EXPECTED_MODES
}

missing_configurations = (
    expected_configurations.difference(
        found_configurations
    )
)

if missing_configurations:
    errors.append(
        "Faltan configuraciones: "
        f"{sorted(missing_configurations)}."
    )


if len(all_records) != EXPECTED_TOTAL_RUNS:
    errors.append(
        f"Se reunieron {len(all_records)} corridas; "
        f"se esperaban {EXPECTED_TOTAL_RUNS}."
    )


# Comprobar el pareo de las 18 configuraciones.
shared_initial_ok = True

for key, block_records in sorted(
    records_by_block.items()
):
    if len(block_records) != EXPECTED_CONFIGURATIONS:
        errors.append(
            f"{key}: contiene "
            f"{len(block_records)} configuraciones; "
            f"se esperaban {EXPECTED_CONFIGURATIONS}."
        )

        shared_initial_ok = False
        continue

    signatures = {
        record["_initial_signature"]
        for record in block_records
    }

    initial_costs = {
        record["_initial_cost"]
        for record in block_records
    }

    if len(signatures) != 1:
        errors.append(
            f"{key}: las configuraciones no comparten "
            "la misma firma inicial."
        )

        shared_initial_ok = False

    if len(initial_costs) != 1:
        errors.append(
            f"{key}: las configuraciones no comparten "
            "el mismo costo inicial."
        )

        shared_initial_ok = False


validation_lines.append("")
validation_lines.append(
    f"Corridas totales: {len(all_records)}"
)
validation_lines.append(
    f"Bloques instancia-semilla: "
    f"{len(records_by_block)}"
)
validation_lines.append(
    "Misma solución inicial entre configuraciones: "
    f"{shared_initial_ok}"
)


if errors:
    validation_lines.append("")
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
        print(f"- {error}")

    print()
    print(
        f"Detalles guardados en {VALIDATION_FILE}"
    )

    raise SystemExit(1)


validation_lines.append("")
validation_lines.append(
    "VALIDACIÓN: TODO CORRECTO"
)

VALIDATION_FILE.write_text(
    "\n".join(validation_lines) + "\n",
    encoding="utf-8",
)


# Guardar las 1620 corridas en un solo CSV.
combined_fieldnames = [
    "configuration",
    "profile",
    "criteria_count",
    "calibration_kmax",
    "calibration_vnd_mode",
]

combined_fieldnames.extend(
    original_fieldnames or []
)

combined_rows: list[dict[str, Any]] = []

for record in all_records:
    combined_rows.append(
        {
            key: value
            for key, value in record.items()
            if not key.startswith("_")
        }
    )

csv_write(
    COMBINED_FILE,
    combined_rows,
    combined_fieldnames,
)


# Resumen por configuración e instancia.
instance_summary_rows: list[
    dict[str, Any]
] = []

for configuration, records in sorted(
    records_by_configuration.items()
):
    profile = records[0]["profile"]
    kmax = records[0]["calibration_kmax"]
    mode = records[0]["calibration_vnd_mode"]

    grouped_by_instance: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for record in records:
        grouped_by_instance[
            normalize(record["instance"])
        ].append(record)

    for instance in sorted(grouped_by_instance):
        instance_records = (
            grouped_by_instance[instance]
        )

        gaps = [
            record["_gap"]
            for record in instance_records
        ]

        times = [
            record["_elapsed_time"]
            for record in instance_records
        ]

        costs = [
            record["_best_cost"]
            for record in instance_records
        ]

        instance_summary_rows.append(
            {
                "configuration": configuration,
                "profile": profile,
                "criteria_count": (
                    PROFILE_TO_COUNT[profile]
                ),
                "kmax": kmax,
                "vnd_mode": mode,
                "instance": instance,
                "runs": len(instance_records),
                "median_gap_percent": (
                    statistics.median(gaps)
                ),
                "iqr_gap_percent": iqr(gaps),
                "mean_gap_percent": (
                    statistics.mean(gaps)
                ),
                "minimum_gap_percent": min(gaps),
                "maximum_gap_percent": max(gaps),
                "median_best_cost": (
                    statistics.median(costs)
                ),
                "median_time_seconds": (
                    statistics.median(times)
                ),
                "iqr_time_seconds": iqr(times),
                "total_time_seconds": sum(times),
            }
        )


instance_summary_fieldnames = [
    "configuration",
    "profile",
    "criteria_count",
    "kmax",
    "vnd_mode",
    "instance",
    "runs",
    "median_gap_percent",
    "iqr_gap_percent",
    "mean_gap_percent",
    "minimum_gap_percent",
    "maximum_gap_percent",
    "median_best_cost",
    "median_time_seconds",
    "iqr_time_seconds",
    "total_time_seconds",
]

csv_write(
    INSTANCE_SUMMARY_FILE,
    instance_summary_rows,
    instance_summary_fieldnames,
)


# Calcular rangos pareados en cada bloque instancia-semilla.
gap_ranks_by_configuration: dict[
    str,
    list[float],
] = defaultdict(list)

time_ranks_by_configuration: dict[
    str,
    list[float],
] = defaultdict(list)

wins_or_ties: dict[str, int] = defaultdict(int)
fractional_wins: dict[str, float] = defaultdict(float)

for key, block_records in sorted(
    records_by_block.items()
):
    gaps = {
        record["configuration"]: record["_gap"]
        for record in block_records
    }

    times = {
        record["configuration"]: record["_elapsed_time"]
        for record in block_records
    }

    gap_ranks = average_ranks(gaps)
    time_ranks = average_ranks(times)

    for configuration, rank in gap_ranks.items():
        gap_ranks_by_configuration[
            configuration
        ].append(rank)

    for configuration, rank in time_ranks.items():
        time_ranks_by_configuration[
            configuration
        ].append(rank)

    minimum_gap = min(gaps.values())

    winners = [
        configuration
        for configuration, gap in gaps.items()
        if math.isclose(
            gap,
            minimum_gap,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]

    for configuration in winners:
        wins_or_ties[configuration] += 1
        fractional_wins[configuration] += (
            1.0 / len(winners)
        )


summary_lookup = {
    (
        row["configuration"],
        row["instance"],
    ): row
    for row in instance_summary_rows
}


ranking_rows: list[dict[str, Any]] = []

for configuration, records in sorted(
    records_by_configuration.items()
):
    profile = records[0]["profile"]
    kmax = records[0]["calibration_kmax"]
    mode = records[0]["calibration_vnd_mode"]

    all_gaps = [
        record["_gap"]
        for record in records
    ]

    all_times = [
        record["_elapsed_time"]
        for record in records
    ]

    instance_medians = [
        float(
            summary_lookup[
                (configuration, instance)
            ]["median_gap_percent"]
        )
        for instance in sorted(EXPECTED_INSTANCES)
    ]

    instance_iqrs = [
        float(
            summary_lookup[
                (configuration, instance)
            ]["iqr_gap_percent"]
        )
        for instance in sorted(EXPECTED_INSTANCES)
    ]

    ranking_rows.append(
        {
            "configuration": configuration,
            "profile": profile,
            "criteria_count": (
                PROFILE_TO_COUNT[profile]
            ),
            "kmax": kmax,
            "vnd_mode": mode,
            "valid_runs": len(records),
            "mean_gap_rank": statistics.mean(
                gap_ranks_by_configuration[
                    configuration
                ]
            ),
            "median_gap_rank": statistics.median(
                gap_ranks_by_configuration[
                    configuration
                ]
            ),
            "wins_or_ties": (
                wins_or_ties[configuration]
            ),
            "fractional_wins": (
                fractional_wins[configuration]
            ),
            "mean_instance_median_gap": (
                statistics.mean(instance_medians)
            ),
            "worst_instance_median_gap": (
                max(instance_medians)
            ),
            "mean_instance_iqr_gap": (
                statistics.mean(instance_iqrs)
            ),
            "overall_median_gap_percent": (
                statistics.median(all_gaps)
            ),
            "overall_iqr_gap_percent": (
                iqr(all_gaps)
            ),
            "overall_mean_gap_percent": (
                statistics.mean(all_gaps)
            ),
            "overall_median_time_seconds": (
                statistics.median(all_times)
            ),
            "total_time_seconds": sum(all_times),
            "mean_time_rank": statistics.mean(
                time_ranks_by_configuration[
                    configuration
                ]
            ),
            "median_gap_A-n33-k5": (
                summary_lookup[
                    (configuration, "A-n33-k5")
                ]["median_gap_percent"]
            ),
            "median_gap_A-n65-k9": (
                summary_lookup[
                    (configuration, "A-n65-k9")
                ]["median_gap_percent"]
            ),
            "median_gap_A-n80-k10": (
                summary_lookup[
                    (configuration, "A-n80-k10")
                ]["median_gap_percent"]
            ),
        }
    )


# Regla de selección predefinida:
# 1. menor rango promedio de gap;
# 2. menor promedio de las medianas por instancia;
# 3. menor peor mediana entre instancias;
# 4. menor dispersión media;
# 5. menor tiempo mediano.
ranking_rows.sort(
    key=lambda row: (
        float(row["mean_gap_rank"]),
        float(row["mean_instance_median_gap"]),
        float(row["worst_instance_median_gap"]),
        float(row["mean_instance_iqr_gap"]),
        float(row["overall_median_time_seconds"]),
        str(row["configuration"]),
    )
)


for position, row in enumerate(
    ranking_rows,
    start=1,
):
    row["position"] = position


ranking_fieldnames = [
    "position",
    "configuration",
    "profile",
    "criteria_count",
    "kmax",
    "vnd_mode",
    "valid_runs",
    "mean_gap_rank",
    "median_gap_rank",
    "wins_or_ties",
    "fractional_wins",
    "mean_instance_median_gap",
    "worst_instance_median_gap",
    "mean_instance_iqr_gap",
    "overall_median_gap_percent",
    "overall_iqr_gap_percent",
    "overall_mean_gap_percent",
    "overall_median_time_seconds",
    "total_time_seconds",
    "mean_time_rank",
    "median_gap_A-n33-k5",
    "median_gap_A-n65-k9",
    "median_gap_A-n80-k10",
]

csv_write(
    RANKING_FILE,
    ranking_rows,
    ranking_fieldnames,
)


winner = ranking_rows[0]

winner_data = {
    "selection_timestamp": datetime.now().isoformat(
        timespec="seconds"
    ),
    "input_directory": str(INPUT_DIR),
    "selection_rule": [
        "menor rango promedio pareado del gap",
        "menor promedio de medianas del gap por instancia",
        "menor peor mediana del gap entre instancias",
        "menor promedio del IQR del gap",
        "menor tiempo mediano",
    ],
    "configuration": winner["configuration"],
    "profile": winner["profile"],
    "criteria_keyword": (
        PROFILE_TO_CRITERIA[
            str(winner["profile"])
        ]
    ),
    "criteria_count": winner["criteria_count"],
    "kmax": winner["kmax"],
    "vnd_mode": winner["vnd_mode"],
    "mean_gap_rank": winner["mean_gap_rank"],
    "mean_instance_median_gap": (
        winner["mean_instance_median_gap"]
    ),
    "worst_instance_median_gap": (
        winner["worst_instance_median_gap"]
    ),
    "mean_instance_iqr_gap": (
        winner["mean_instance_iqr_gap"]
    ),
    "overall_median_time_seconds": (
        winner["overall_median_time_seconds"]
    ),
    "runs": winner["valid_runs"],
}

WINNER_JSON_FILE.write_text(
    json.dumps(
        winner_data,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)


lisp_content = f"""(in-package :cl-user)

;; Configuración seleccionada automáticamente a partir de:
;; {INPUT_DIR}
;;
;; Regla principal:
;; menor rango promedio pareado del gap.

(defparameter *best-vns-configuration*
  "{winner['configuration']}")

(defparameter *best-vns-profile-name*
  "{winner['profile']}")

(defparameter *best-vns-criteria*
  {PROFILE_TO_CRITERIA[str(winner['profile'])]})

(defparameter *best-vns-criteria-count*
  {winner['criteria_count']})

(defparameter *best-vns-kmax*
  {winner['kmax']})

(defparameter *best-vns-vnd-mode*
  :{winner['vnd_mode']})

(defparameter *best-vns-mean-gap-rank*
  {float(winner['mean_gap_rank']):.12f})

(defparameter *best-vns-selection-source*
  "{RANKING_FILE}")
"""

BEST_VNS_LISP.write_text(
    lisp_content,
    encoding="utf-8",
)

BEST_VNS_LISP_COPY.write_text(
    lisp_content,
    encoding="utf-8",
)


common_parameters = {}

for column in [
    "max_iter",
    "max_no_improve",
    "shake_max_tries",
    "vnd_max_rounds",
    "vnd_max_no_improve",
    "time_limit_seconds",
]:
    values = sorted(
        {
            normalize(record[column])
            for record in all_records
        }
    )

    common_parameters[column] = values


report_lines: list[str] = []

report_lines.extend(
    [
        "# Informe de calibración de VNS",
        "",
        f"**Fecha del análisis:** "
        f"{datetime.now().isoformat(timespec='seconds')}",
        "",
        f"**Directorio de entrada:** `{INPUT_DIR}`",
        "",
        "## Validación de los datos",
        "",
        f"- Configuraciones: {len(records_by_configuration)}",
        f"- Corridas por configuración: "
        f"{EXPECTED_RUNS_PER_CONFIGURATION}",
        f"- Corridas totales: {len(all_records)}",
        f"- Bloques instancia-semilla: "
        f"{len(records_by_block)}",
        "- Estados: todos `ok`",
        "- Soluciones: todas factibles",
        "- Pares duplicados: ninguno",
        "- Solución inicial común entre las 18 "
        f"configuraciones: `{shared_initial_ok}`",
        "",
        "## Regla de selección",
        "",
        "La configuración se ordenó mediante los "
        "siguientes criterios:",
        "",
        "1. Menor rango promedio pareado del gap.",
        "2. Menor promedio de las medianas del gap "
        "por instancia.",
        "3. Menor peor mediana del gap entre instancias.",
        "4. Menor promedio del IQR del gap.",
        "5. Menor tiempo mediano como desempate.",
        "",
        "El rango se calculó dentro de cada bloque "
        "instancia-semilla. Por tanto, cada una de las "
        "18 configuraciones se comparó desde la misma "
        "solución inicial.",
        "",
        "## Configuración seleccionada",
        "",
        f"- **Configuración:** "
        f"`{winner['configuration']}`",
        f"- **Perfil:** `{winner['profile']}`",
        f"- **Cantidad de criterios:** "
        f"{winner['criteria_count']}",
        f"- **kmax:** {winner['kmax']}",
        f"- **Modo VND:** `{winner['vnd_mode']}`",
        f"- **Rango promedio del gap:** "
        f"{float(winner['mean_gap_rank']):.6f}",
        f"- **Promedio de medianas por instancia:** "
        f"{float(winner['mean_instance_median_gap']):.6f}",
        f"- **Peor mediana entre instancias:** "
        f"{float(winner['worst_instance_median_gap']):.6f}",
        f"- **Tiempo mediano:** "
        f"{float(winner['overall_median_time_seconds']):.6f} s",
        "",
        "## Parámetros comunes de la calibración",
        "",
    ]
)

for parameter, values in common_parameters.items():
    report_lines.append(
        f"- `{parameter}`: {', '.join(values)}"
    )


report_lines.extend(
    [
        "",
        "## Primeras diez configuraciones",
        "",
        "| Posición | Configuración | Rango gap | "
        "Mediana media | Peor mediana | "
        "Tiempo mediano |",
        "|---:|---|---:|---:|---:|---:|",
    ]
)

for row in ranking_rows[:10]:
    report_lines.append(
        f"| {row['position']} "
        f"| {row['configuration']} "
        f"| {float(row['mean_gap_rank']):.6f} "
        f"| {float(row['mean_instance_median_gap']):.6f} "
        f"| {float(row['worst_instance_median_gap']):.6f} "
        f"| {float(row['overall_median_time_seconds']):.6f} |"
    )


report_lines.extend(
    [
        "",
        "## Archivos producidos",
        "",
        f"- `{COMBINED_FILE}`",
        f"- `{INSTANCE_SUMMARY_FILE}`",
        f"- `{RANKING_FILE}`",
        f"- `{WINNER_JSON_FILE}`",
        f"- `{BEST_VNS_LISP}`",
        f"- `{VALIDATION_FILE}`",
        f"- `{HASH_FILE}`",
        "",
        "## Interpretación",
        "",
        "La configuración seleccionada constituye la "
        "mejor variante de VNS dentro de la rejilla "
        "evaluada. La selección pertenece a la fase "
        "exploratoria. La comparación final contra IVNS "
        "debe utilizar semillas nuevas que no participaron "
        "en esta calibración.",
        "",
    ]
)

REPORT_FILE.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)


# Guardar hashes de entradas, programas y fuentes.
additional_hash_paths = [
    Path(__file__),
    Path("analysis/run_vns_calibration_clean.lisp"),
    Path("vrp-suite/vrp-vns.org"),
    Path("vrp-suite/experiments-vns-ivns.org"),
    Path("src/vrp-vns.lisp"),
    Path("src/experiments-vns-ivns.lisp"),
]

for path in additional_hash_paths:
    if path.exists():
        hash_paths.append(path)

hash_lines = []

for path in sorted(
    set(hash_paths),
    key=lambda item: str(item),
):
    hash_lines.append(
        f"{sha256_file(path)}  {path}"
    )

HASH_FILE.write_text(
    "\n".join(hash_lines) + "\n",
    encoding="utf-8",
)


print("=" * 80)
print("ANÁLISIS DE CALIBRACIÓN VNS")
print("=" * 80)
print()
print("VALIDACIÓN: TODO CORRECTO")
print(f"Configuraciones: {len(records_by_configuration)}")
print(f"Corridas totales: {len(all_records)}")
print(
    "Soluciones iniciales compartidas entre configuraciones: "
    f"{shared_initial_ok}"
)
print()
print("MEJOR CONFIGURACIÓN:")
print(f"  {winner['configuration']}")
print(f"  Perfil: {winner['profile']}")
print(f"  Criterios: {winner['criteria_count']}")
print(f"  kmax: {winner['kmax']}")
print(f"  VND: {winner['vnd_mode']}")
print(
    "  Rango promedio de gap: "
    f"{float(winner['mean_gap_rank']):.6f}"
)
print()
print("ARCHIVOS GUARDADOS:")
print(f"  {REPORT_FILE}")
print(f"  {RANKING_FILE}")
print(f"  {INSTANCE_SUMMARY_FILE}")
print(f"  {WINNER_JSON_FILE}")
print(f"  {BEST_VNS_LISP}")
print(f"  {HASH_FILE}")
