from __future__ import annotations

import csv
from pathlib import Path


FILE_NAME = "summary-vns-basic-3-k2-first.csv"

PATH_A = Path("results/repro-vns-a") / FILE_NAME
PATH_B = Path("results/repro-vns-b") / FILE_NAME

KEYS = [
    "instance",
    "seed",
]

COMPARE_COLUMNS = [
    "initial_cost",
    "initial_solution_signature",
    "best_cost",
    "final_cost",
    "gap_percent",
    "iterations",
    "accepted_improvements",
    "termination_reason",
    "feasible",
    "status",
]


def read_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        rows = list(csv.DictReader(stream))

    if len(rows) != 5:
        raise SystemExit(
            f"ERROR: {path} contiene {len(rows)} filas; "
            "se esperaban 5."
        )

    return {
        (
            row["instance"].strip(),
            row["seed"].strip(),
        ): row
        for row in rows
    }


rows_a = read_rows(PATH_A)
rows_b = read_rows(PATH_B)

if set(rows_a) != set(rows_b):
    raise SystemExit(
        "ERROR: las ejecuciones no contienen "
        "los mismos pares instancia-semilla."
    )


differences: list[str] = []

for key in sorted(rows_a):
    row_a = rows_a[key]
    row_b = rows_b[key]

    for column in COMPARE_COLUMNS:
        value_a = row_a[column].strip()
        value_b = row_b[column].strip()

        if value_a != value_b:
            differences.append(
                f"{key}, {column}: "
                f"{value_a!r} != {value_b!r}"
            )


if differences:
    print(
        f"REPRODUCIBILIDAD: ERROR "
        f"({len(differences)} diferencias)"
    )

    for difference in differences:
        print(f"- {difference}")

    raise SystemExit(1)


print("REPRODUCIBILIDAD: CORRECTA")
print(
    "Las dos ejecuciones produjeron los mismos "
    "costos, gaps y soluciones iniciales."
)
print(
    "El tiempo no se comparó porque puede variar "
    "entre ejecuciones."
)
