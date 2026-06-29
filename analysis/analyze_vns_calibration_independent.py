from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


INPUT_DIR = Path("results/calibracion-vns-limpia")
OUTPUT_DIR = Path(
    "results/analisis-calibracion-vns-independiente"
)

COMBINED_FILE = OUTPUT_DIR / "calibracion-vns-combinada.csv"
INSTANCE_FILE = OUTPUT_DIR / "resumen-por-instancia.csv"
RANKING_FILE = OUTPUT_DIR / "ranking-configuraciones-vns.csv"
WINNER_FILE = OUTPUT_DIR / "mejor-configuracion-vns.json"
VALIDATION_FILE = OUTPUT_DIR / "validacion.txt"
REPORT_FILE = OUTPUT_DIR / "informe-calibracion-vns.md"
HASH_FILE = OUTPUT_DIR / "SHA256SUMS.txt"

BEST_LISP_FILE = Path("analysis/best_vns_config.lisp")
BEST_LISP_COPY = OUTPUT_DIR / "best_vns_config.lisp"


EXPECTED_INSTANCES = {
    "A-n33-k5",
    "A-n65-k9",
    "A-n80-k10",
}

EXPECTED_CONFIGURATIONS = 18
EXPECTED_RUNS_PER_CONFIGURATION = 90
EXPECTED_RUNS_PER_INSTANCE = 30
EXPECTED_TOTAL_RUNS = 1620

FILE_PATTERN = re.compile(
    r"summary-vns-"
    r"(basic-3|five-5|extended-8)"
    r"-k(2|3|5)"
    r"-(first|best)\.csv$"
)

PROFILE_TO_LISP = {
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
    "kmax",
    "vnd_mode",
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
            f"No se pudo convertir {field}={text!r}"
        ) from exc

    if not math.isfinite(number):
        raise ValueError(
            f"{field} no contiene un valor finito: {text!r}"
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


def average_ranks(
    values: dict[str, float],
) -> dict[str, float]:
    """
    Asigna rango 1 al menor valor.

    En caso de empate asigna el promedio de los rangos.
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
        rank = (first_rank + last_rank) / 2.0

        for position in range(index, end):
            configuration = ordered[position][0]
            result[configuration] = rank

        index = end

    return result


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


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
    f"Archivos summary: {len(summary_files)}"
)

if len(summary_files) != EXPECTED_CONFIGURATIONS:
    errors.append(
        f"Se encontraron {len(summary_files)} archivos; "
        f"se esperaban {EXPECTED_CONFIGURATIONS}."
    )


all_records: list[dict[str, Any]] = []

records_by_configuration: dict[
    str,
    list[dict[str, Any]],
] = {}

records_by_configuration_instance: dict[
    tuple[str, str],
    list[dict[str, Any]],
] = defaultdict(list)

original_columns: list[str] | None = None
input_files: list[Path] = []


for path in summary_files:
    match = FILE_PATTERN.fullmatch(path.name)

    if match is None:
        errors.append(
            f"Nombre de archivo no reconocido: {path.name}"
        )
        continue

    profile, kmax_text, mode = match.groups()
    kmax = int(kmax_text)

    configuration = (
        f"{profile}-k{kmax}-{mode}"
    )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    input_files.append(path)

    if original_columns is None:
        original_columns = list(fieldnames)

    missing = REQUIRED_COLUMNS.difference(fieldnames)

    if missing:
        errors.append(
            f"{path.name}: faltan columnas "
            f"{sorted(missing)}."
        )
        continue

    if len(rows) != EXPECTED_RUNS_PER_CONFIGURATION:
        errors.append(
            f"{path.name}: contiene {len(rows)} corridas; "
            f"se esperaban "
            f"{EXPECTED_RUNS_PER_CONFIGURATION}."
        )

    local_records: list[dict[str, Any]] = []
    local_keys: set[tuple[str, str]] = set()

    for row_number, row in enumerate(
        rows,
        start=2,
    ):
        instance = normalize(row["instance"])
        seed = normalize(row["seed"])

        key = (instance, seed)

        if key in local_keys:
            errors.append(
                f"{path.name}: corrida duplicada "
                f"{instance}, seed={seed}."
            )

        local_keys.add(key)

        if instance not in EXPECTED_INSTANCES:
            errors.append(
                f"{path.name}:{row_number}: "
                f"instancia inesperada {instance!r}."
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
                f"kmax={row_kmax}; esperado={kmax}."
            )

        row_mode = normalize_symbol(
            row["vnd_mode"]
        )

        if row_mode != mode:
            errors.append(
                f"{path.name}:{row_number}: "
                f"vnd_mode={row_mode!r}; "
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
        ).lower()

        if error_text not in {
            "",
            "nil",
            "none",
        }:
            errors.append(
                f"{path.name}:{row_number}: "
                f"error={row['error']!r}."
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
                "_gap": parse_float(
                    row["gap_percent"],
                    "gap_percent",
                ),
                "_time": parse_float(
                    row["elapsed_time_seconds"],
                    "elapsed_time_seconds",
                ),
                "_best_cost": parse_float(
                    row["best_cost"],
                    "best_cost",
                ),
            }
        )

        local_records.append(record)
        all_records.append(record)

        records_by_configuration_instance[
            (configuration, instance)
        ].append(record)

    instance_counts = Counter(
        normalize(row["instance"])
        for row in rows
    )

    for instance in EXPECTED_INSTANCES:
        count = instance_counts[instance]

        if count != EXPECTED_RUNS_PER_INSTANCE:
            errors.append(
                f"{path.name}: {instance} contiene "
                f"{count} corridas; se esperaban "
                f"{EXPECTED_RUNS_PER_INSTANCE}."
            )

        distinct_seeds = {
            normalize(row["seed"])
            for row in rows
            if normalize(row["instance"]) == instance
        }

        if len(distinct_seeds) != EXPECTED_RUNS_PER_INSTANCE:
            errors.append(
                f"{path.name}: {instance} contiene "
                f"{len(distinct_seeds)} semillas distintas; "
                f"se esperaban "
                f"{EXPECTED_RUNS_PER_INSTANCE}."
            )

    records_by_configuration[
        configuration
    ] = local_records

    validation_lines.append(
        f"{configuration}: {len(local_records)} corridas"
    )


if len(all_records) != EXPECTED_TOTAL_RUNS:
    errors.append(
        f"Se reunieron {len(all_records)} corridas; "
        f"se esperaban {EXPECTED_TOTAL_RUNS}."
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

    raise SystemExit(1)


validation_lines.extend(
    [
        "",
        f"Configuraciones: "
        f"{len(records_by_configuration)}",
        f"Corridas totales: {len(all_records)}",
        "",
        "Diseño de calibración: independiente",
        "No se exige solución inicial común "
        "entre configuraciones.",
        "",
        "VALIDACIÓN: TODO CORRECTO",
    ]
)

VALIDATION_FILE.write_text(
    "\n".join(validation_lines) + "\n",
    encoding="utf-8",
)


# CSV combinado.
combined_rows = [
    {
        key: value
        for key, value in record.items()
        if not key.startswith("_")
    }
    for record in all_records
]

combined_fields = [
    "configuration",
    "profile",
    "criteria_count",
    "calibration_kmax",
    "calibration_vnd_mode",
]

combined_fields.extend(
    original_columns or []
)

write_csv(
    COMBINED_FILE,
    combined_rows,
    combined_fields,
)


# Resumen por configuración e instancia.
instance_rows: list[dict[str, Any]] = []

for (
    configuration,
    instance,
), records in sorted(
    records_by_configuration_instance.items()
):
    profile = records[0]["profile"]
    kmax = records[0]["calibration_kmax"]
    mode = records[0]["calibration_vnd_mode"]

    gaps = [
        record["_gap"]
        for record in records
    ]

    times = [
        record["_time"]
        for record in records
    ]

    costs = [
        record["_best_cost"]
        for record in records
    ]

    instance_rows.append(
        {
            "configuration": configuration,
            "profile": profile,
            "criteria_count": (
                PROFILE_TO_COUNT[profile]
            ),
            "kmax": kmax,
            "vnd_mode": mode,
            "instance": instance,
            "runs": len(records),
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


instance_fields = [
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

write_csv(
    INSTANCE_FILE,
    instance_rows,
    instance_fields,
)


instance_lookup = {
    (
        row["configuration"],
        row["instance"],
    ): row
    for row in instance_rows
}


# Rangos independientes por instancia.
ranks_by_configuration: dict[
    str,
    list[float],
] = defaultdict(list)

for instance in sorted(EXPECTED_INSTANCES):
    medians = {
        configuration: float(
            instance_lookup[
                (configuration, instance)
            ]["median_gap_percent"]
        )
        for configuration in records_by_configuration
    }

    instance_ranks = average_ranks(medians)

    for configuration, rank in instance_ranks.items():
        ranks_by_configuration[
            configuration
        ].append(rank)


ranking_rows: list[dict[str, Any]] = []

for configuration, records in sorted(
    records_by_configuration.items()
):
    profile = records[0]["profile"]
    kmax = records[0]["calibration_kmax"]
    mode = records[0]["calibration_vnd_mode"]

    medians = [
        float(
            instance_lookup[
                (configuration, instance)
            ]["median_gap_percent"]
        )
        for instance in sorted(EXPECTED_INSTANCES)
    ]

    gap_iqrs = [
        float(
            instance_lookup[
                (configuration, instance)
            ]["iqr_gap_percent"]
        )
        for instance in sorted(EXPECTED_INSTANCES)
    ]

    times = [
        record["_time"]
        for record in records
    ]

    ranks = ranks_by_configuration[configuration]

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
            "mean_instance_rank": (
                statistics.mean(ranks)
            ),
            "best_instance_rank": min(ranks),
            "worst_instance_rank": max(ranks),
            "mean_instance_median_gap": (
                statistics.mean(medians)
            ),
            "worst_instance_median_gap": max(medians),
            "mean_instance_iqr_gap": (
                statistics.mean(gap_iqrs)
            ),
            "overall_median_time_seconds": (
                statistics.median(times)
            ),
            "median_gap_A-n33-k5": (
                instance_lookup[
                    (configuration, "A-n33-k5")
                ]["median_gap_percent"]
            ),
            "median_gap_A-n65-k9": (
                instance_lookup[
                    (configuration, "A-n65-k9")
                ]["median_gap_percent"]
            ),
            "median_gap_A-n80-k10": (
                instance_lookup[
                    (configuration, "A-n80-k10")
                ]["median_gap_percent"]
            ),
        }
    )


# Selección:
# 1. mejor rango medio entre las tres instancias;
# 2. menor promedio de medianas del gap;
# 3. menor peor mediana;
# 4. menor dispersión;
# 5. menor tiempo.
ranking_rows.sort(
    key=lambda row: (
        float(row["mean_instance_rank"]),
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


ranking_fields = [
    "position",
    "configuration",
    "profile",
    "criteria_count",
    "kmax",
    "vnd_mode",
    "valid_runs",
    "mean_instance_rank",
    "best_instance_rank",
    "worst_instance_rank",
    "mean_instance_median_gap",
    "worst_instance_median_gap",
    "mean_instance_iqr_gap",
    "overall_median_time_seconds",
    "median_gap_A-n33-k5",
    "median_gap_A-n65-k9",
    "median_gap_A-n80-k10",
]

write_csv(
    RANKING_FILE,
    ranking_rows,
    ranking_fields,
)


winner = ranking_rows[0]

winner_data = {
    "analysis_type": (
        "calibración exploratoria con muestras independientes"
    ),
    "analysis_timestamp": datetime.now().isoformat(
        timespec="seconds"
    ),
    "input_directory": str(INPUT_DIR),
    "selection_rule": [
        "menor rango medio de la mediana del gap "
        "entre las tres instancias",
        "menor promedio de las medianas del gap",
        "menor peor mediana del gap",
        "menor promedio del IQR",
        "menor tiempo mediano como desempate",
    ],
    "configuration": winner["configuration"],
    "profile": winner["profile"],
    "criteria_keyword": (
        PROFILE_TO_LISP[
            str(winner["profile"])
        ]
    ),
    "criteria_count": winner["criteria_count"],
    "kmax": winner["kmax"],
    "vnd_mode": winner["vnd_mode"],
    "mean_instance_rank": (
        winner["mean_instance_rank"]
    ),
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

WINNER_FILE.write_text(
    json.dumps(
        winner_data,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)


lisp_text = f"""(in-package :cl-user)

;; Mejor configuración de VNS obtenida mediante
;; calibración exploratoria con muestras independientes.
;;
;; Fuente:
;; {RANKING_FILE}

(defparameter *best-vns-configuration*
  "{winner['configuration']}")

(defparameter *best-vns-profile-name*
  "{winner['profile']}")

(defparameter *best-vns-criteria*
  {PROFILE_TO_LISP[str(winner['profile'])]})

(defparameter *best-vns-criteria-count*
  {winner['criteria_count']})

(defparameter *best-vns-kmax*
  {winner['kmax']})

(defparameter *best-vns-vnd-mode*
  :{winner['vnd_mode']})

(defparameter *best-vns-mean-instance-rank*
  {float(winner['mean_instance_rank']):.12f})
"""

BEST_LISP_FILE.write_text(
    lisp_text,
    encoding="utf-8",
)

BEST_LISP_COPY.write_text(
    lisp_text,
    encoding="utf-8",
)


report_lines = [
    "# Informe de calibración de VNS",
    "",
    f"**Fecha:** "
    f"{datetime.now().isoformat(timespec='seconds')}",
    "",
    "## Diseño experimental",
    "",
    "- Tipo de análisis: muestras independientes.",
    "- Configuraciones evaluadas: 18.",
    "- Instancias: 3.",
    "- Corridas por instancia y configuración: 30.",
    "- Corridas por configuración: 90.",
    "- Corridas totales: 1620.",
    "",
    "Las configuraciones no tenían que partir de la "
    "misma solución inicial. Cada configuración se "
    "evaluó mediante sus propias corridas aleatorias.",
    "",
    "## Regla de selección",
    "",
    "1. Menor rango medio de la mediana del gap "
    "entre las tres instancias.",
    "2. Menor promedio de las medianas del gap.",
    "3. Menor peor mediana entre las instancias.",
    "4. Menor promedio del IQR.",
    "5. Menor tiempo mediano como desempate.",
    "",
    "## Mejor configuración",
    "",
    f"- Configuración: `{winner['configuration']}`",
    f"- Perfil: `{winner['profile']}`",
    f"- Criterios: {winner['criteria_count']}",
    f"- kmax: {winner['kmax']}",
    f"- VND: `{winner['vnd_mode']}`",
    f"- Rango medio: "
    f"{float(winner['mean_instance_rank']):.6f}",
    f"- Promedio de medianas del gap: "
    f"{float(winner['mean_instance_median_gap']):.6f}",
    f"- Peor mediana del gap: "
    f"{float(winner['worst_instance_median_gap']):.6f}",
    f"- Tiempo mediano: "
    f"{float(winner['overall_median_time_seconds']):.6f} s",
    "",
    "## Primeras diez configuraciones",
    "",
    "| Pos. | Configuración | Rango medio | "
    "Gap mediano medio | Peor gap | Tiempo |",
    "|---:|---|---:|---:|---:|---:|",
]

for row in ranking_rows[:10]:
    report_lines.append(
        f"| {row['position']} "
        f"| {row['configuration']} "
        f"| {float(row['mean_instance_rank']):.6f} "
        f"| {float(row['mean_instance_median_gap']):.6f} "
        f"| {float(row['worst_instance_median_gap']):.6f} "
        f"| {float(row['overall_median_time_seconds']):.6f} |"
    )


report_lines.extend(
    [
        "",
        "## Uso posterior",
        "",
        "La configuración seleccionada se utilizará en "
        "el experimento confirmatorio contra la mejor "
        "configuración de IVNS. Ese experimento sí debe "
        "usar la misma solución inicial para VNS e IVNS "
        "en cada pareja instancia-semilla.",
        "",
    ]
)

REPORT_FILE.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)


hash_paths = list(input_files)

for path in [
    Path(__file__),
    Path("analysis/run_vns_calibration_clean.lisp"),
    Path("vrp-suite/vrp-vns.org"),
    Path("vrp-suite/experiments-vns-ivns.org"),
    Path("src/vrp-vns.lisp"),
    Path("src/experiments-vns-ivns.lisp"),
]:
    if path.exists():
        hash_paths.append(path)


hash_lines = [
    f"{sha256_file(path)}  {path}"
    for path in sorted(
        set(hash_paths),
        key=lambda item: str(item),
    )
]

HASH_FILE.write_text(
    "\n".join(hash_lines) + "\n",
    encoding="utf-8",
)


print("=" * 72)
print("ANÁLISIS INDEPENDIENTE DE LA CALIBRACIÓN VNS")
print("=" * 72)
print()
print("VALIDACIÓN: TODO CORRECTO")
print(f"Configuraciones: {len(records_by_configuration)}")
print(f"Corridas totales: {len(all_records)}")
print()
print("MEJOR CONFIGURACIÓN:")
print(f"  {winner['configuration']}")
print(f"  Perfil: {winner['profile']}")
print(f"  Criterios: {winner['criteria_count']}")
print(f"  kmax: {winner['kmax']}")
print(f"  VND: {winner['vnd_mode']}")
print(
    "  Rango medio entre instancias: "
    f"{float(winner['mean_instance_rank']):.6f}"
)
print()
print("ARCHIVOS GUARDADOS:")
print(f"  {VALIDATION_FILE}")
print(f"  {RANKING_FILE}")
print(f"  {INSTANCE_FILE}")
print(f"  {WINNER_FILE}")
print(f"  {REPORT_FILE}")
print(f"  {BEST_LISP_FILE}")
print(f"  {HASH_FILE}")
