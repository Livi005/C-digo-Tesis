from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("results/calibracion-vns")
OUTPUT = ROOT / "diagnostico-calibracion-vns.txt"
EXPECTED_SEEDS = {str(seed) for seed in range(1, 31)}
EXPECTED_ROWS = 90


def normalize(value: object) -> str:
    return str(value).strip()


def find_column(
    fieldnames: list[str],
    candidates: list[str],
) -> str | None:
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate

    return None


lines: list[str] = []


def report(text: str = "") -> None:
    print(text)
    lines.append(text)


summary_files = sorted(
    ROOT.glob("summary-vns-*.csv")
)

report("=" * 100)
report("DIAGNÓSTICO DE LA CALIBRACIÓN VNS")
report("=" * 100)
report()
report(f"Archivos summary encontrados: {len(summary_files)}")
report()


for path in summary_files:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    instance_column = find_column(
        fieldnames,
        ["instance", "instance_name"],
    )

    seed_column = find_column(
        fieldnames,
        ["seed"],
    )

    cost_column = find_column(
        fieldnames,
        ["best_cost", "cost"],
    )

    gap_column = find_column(
        fieldnames,
        ["gap_percent", "gap"],
    )

    time_column = find_column(
        fieldnames,
        [
            "elapsed_time_seconds",
            "time",
            "elapsed_time",
        ],
    )

    status_column = find_column(
        fieldnames,
        ["status"],
    )

    feasible_column = find_column(
        fieldnames,
        ["feasible"],
    )

    report("-" * 100)
    report(path.name)
    report("-" * 100)
    report(f"Filas leídas: {len(rows)}")
    report(f"Columnas: {len(fieldnames)}")
    report(f"Nombres de columnas: {fieldnames}")

    if instance_column is None or seed_column is None:
        report(
            "ERROR: no encuentro las columnas de instancia o semilla."
        )
        report()
        continue

    # Detectar si una cabecera se insertó como una fila de datos.
    header_rows = [
        row
        for row in rows
        if normalize(row.get(instance_column)) == instance_column
        or normalize(row.get(seed_column)) == seed_column
    ]

    clean_rows = [
        row
        for row in rows
        if row not in header_rows
    ]

    report(f"Cabeceras repetidas dentro del CSV: {len(header_rows)}")
    report(f"Filas de datos limpias: {len(clean_rows)}")

    keys = [
        (
            normalize(row.get(instance_column)),
            normalize(row.get(seed_column)),
        )
        for row in clean_rows
    ]

    key_counts = Counter(keys)
    unique_keys = set(keys)

    duplicate_keys = {
        key: count
        for key, count in key_counts.items()
        if count > 1
    }

    report(
        f"Pares únicos instancia-semilla: {len(unique_keys)}"
    )

    report(
        f"Pares repetidos: {len(duplicate_keys)}"
    )

    multiplicities = Counter(key_counts.values())

    report(
        "Multiplicidad de los pares "
        f"(veces que aparece cada uno): {dict(sorted(multiplicities.items()))}"
    )

    instance_counts = Counter(
        normalize(row.get(instance_column))
        for row in clean_rows
    )

    report(
        f"Filas por instancia: {dict(sorted(instance_counts.items()))}"
    )

    # Verificar las últimas 90 filas, que probablemente
    # pertenecen a la ejecución completa más reciente.
    last_rows = clean_rows[-EXPECTED_ROWS:]

    last_keys = [
        (
            normalize(row.get(instance_column)),
            normalize(row.get(seed_column)),
        )
        for row in last_rows
    ]

    last_instance_groups: dict[str, list[str]] = defaultdict(list)

    for instance, seed in last_keys:
        last_instance_groups[instance].append(seed)

    last_instance_counts = {
        instance: len(seeds)
        for instance, seeds in sorted(last_instance_groups.items())
    }

    last_seed_sets_ok = all(
        set(seeds) == EXPECTED_SEEDS
        for seeds in last_instance_groups.values()
    )

    last_90_complete = (
        len(last_rows) == EXPECTED_ROWS
        and len(set(last_keys)) == EXPECTED_ROWS
        and len(last_instance_groups) == 3
        and all(
            count == 30
            for count in last_instance_counts.values()
        )
        and last_seed_sets_ok
    )

    report(
        f"Últimas 90 filas: {len(last_rows)}"
    )

    report(
        "Pares únicos en las últimas 90 filas: "
        f"{len(set(last_keys))}"
    )

    report(
        "Distribución por instancia en las últimas 90: "
        f"{last_instance_counts}"
    )

    report(
        "Las últimas 90 forman una ejecución completa: "
        f"{last_90_complete}"
    )

    if last_rows:
        first_last = last_rows[0]
        final_last = last_rows[-1]

        report(
            "Primera clave de las últimas 90: "
            f"{normalize(first_last.get(instance_column))}, "
            f"seed={normalize(first_last.get(seed_column))}"
        )

        report(
            "Última clave de las últimas 90: "
            f"{normalize(final_last.get(instance_column))}, "
            f"seed={normalize(final_last.get(seed_column))}"
        )

    # Comprobar si las repeticiones de una misma instancia-semilla
    # dieron exactamente el mismo resultado.
    groups: dict[
        tuple[str, str],
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in clean_rows:
        key = (
            normalize(row.get(instance_column)),
            normalize(row.get(seed_column)),
        )
        groups[key].append(row)

    inconsistent_quality = 0
    inconsistent_time = 0
    examples: list[str] = []

    for key, group in groups.items():
        if len(group) < 2:
            continue

        quality_values = {
            (
                normalize(row.get(cost_column))
                if cost_column else "",
                normalize(row.get(gap_column))
                if gap_column else "",
                normalize(row.get(status_column))
                if status_column else "",
                normalize(row.get(feasible_column))
                if feasible_column else "",
            )
            for row in group
        }

        time_values = {
            normalize(row.get(time_column))
            if time_column else ""
            for row in group
        }

        if len(quality_values) > 1:
            inconsistent_quality += 1

            if len(examples) < 5:
                examples.append(
                    f"{key}: resultados={sorted(quality_values)}"
                )

        if len(time_values) > 1:
            inconsistent_time += 1

    report(
        "Pares repetidos con diferencias de costo/gap/estado: "
        f"{inconsistent_quality}"
    )

    report(
        "Pares repetidos con diferencias de tiempo: "
        f"{inconsistent_time}"
    )

    if examples:
        report("Ejemplos de diferencias:")

        for example in examples:
            report(f"  {example}")

    if status_column:
        statuses = Counter(
            normalize(row.get(status_column)).lower()
            for row in clean_rows
        )

        report(
            f"Estados encontrados: {dict(sorted(statuses.items()))}"
        )

    if feasible_column:
        feasible_values = Counter(
            normalize(row.get(feasible_column)).lower()
            for row in clean_rows
        )

        report(
            "Valores de factibilidad: "
            f"{dict(sorted(feasible_values.items()))}"
        )

    report()


report("=" * 100)
report("CONCLUSIÓN AUTOMÁTICA")
report("=" * 100)

all_last_90_complete = True

for path in summary_files:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = reader.fieldnames or []

    instance_column = find_column(
        fields,
        ["instance", "instance_name"],
    )

    seed_column = find_column(
        fields,
        ["seed"],
    )

    if instance_column is None or seed_column is None:
        all_last_90_complete = False
        continue

    rows = [
        row
        for row in rows
        if normalize(row.get(instance_column)) != instance_column
        and normalize(row.get(seed_column)) != seed_column
    ]

    last_rows = rows[-EXPECTED_ROWS:]

    keys = [
        (
            normalize(row.get(instance_column)),
            normalize(row.get(seed_column)),
        )
        for row in last_rows
    ]

    groups: dict[str, list[str]] = defaultdict(list)

    for instance, seed in keys:
        groups[instance].append(seed)

    complete = (
        len(last_rows) == EXPECTED_ROWS
        and len(set(keys)) == EXPECTED_ROWS
        and len(groups) == 3
        and all(len(seeds) == 30 for seeds in groups.values())
        and all(
            set(seeds) == EXPECTED_SEEDS
            for seeds in groups.values()
        )
    )

    if not complete:
        all_last_90_complete = False


if all_last_90_complete:
    report(
        "Las últimas 90 filas de los 18 summaries parecen formar "
        "una ejecución completa e independiente."
    )
    report(
        "Probablemente se pueden recuperar los resultados sin repetir "
        "las 1620 corridas."
    )
else:
    report(
        "Al menos un summary no contiene una ejecución completa en "
        "sus últimas 90 filas."
    )
    report(
        "No se debe limpiar automáticamente hasta revisar los archivos."
    )


OUTPUT.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

print()
print(f"Diagnóstico guardado en: {OUTPUT}")
