from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path("results/test-p2-constructores")

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

EXPECTED_ALGORITHMS = {
    "vns_best",
    "ivns_best",
}

errors: list[str] = []
signatures_by_constructor: dict[
    str,
    dict[str, str],
] = {}


for constructor, path in FILES.items():
    if not path.exists():
        errors.append(
            f"{constructor}: no existe {path}"
        )
        continue

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        rows = list(csv.DictReader(stream))

    if len(rows) != 4:
        errors.append(
            f"{constructor}: contiene {len(rows)} filas; "
            "se esperaban 4."
        )

    groups: dict[
        tuple[str, str],
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in rows:
        key = (
            row["instance"].strip(),
            row["seed"].strip(),
        )
        groups[key].append(row)

    if len(groups) != 2:
        errors.append(
            f"{constructor}: contiene {len(groups)} pares; "
            "se esperaban 2."
        )

    constructor_signatures: dict[str, str] = {}

    for key, pair in groups.items():
        if len(pair) != 2:
            errors.append(
                f"{constructor}, {key}: "
                f"contiene {len(pair)} filas."
            )
            continue

        algorithms = {
            row["algorithm"].strip().lower()
            for row in pair
        }

        if algorithms != EXPECTED_ALGORITHMS:
            errors.append(
                f"{constructor}, {key}: "
                f"algoritmos={algorithms}."
            )

        signatures = {
            row["initial_solution_signature"].strip()
            for row in pair
        }

        costs = {
            row["initial_cost"].strip()
            for row in pair
        }

        if len(signatures) != 1:
            errors.append(
                f"{constructor}, {key}: "
                "VNS e IVNS tienen firmas iniciales diferentes."
            )
        else:
            constructor_signatures[key[1]] = next(
                iter(signatures)
            )

        if len(costs) != 1:
            errors.append(
                f"{constructor}, {key}: "
                "VNS e IVNS tienen costos iniciales diferentes."
            )

        for row in pair:
            algorithm = row["algorithm"].strip()

            if row["status"].strip().lower() != "ok":
                errors.append(
                    f"{constructor}, {key}, {algorithm}: "
                    f"status={row['status']!r}."
                )

            feasible = (
                row["feasible"]
                .strip()
                .lower()
            )

            if feasible not in {
                "true",
                "t",
                "1",
                "1.0",
            }:
                errors.append(
                    f"{constructor}, {key}, {algorithm}: "
                    "solución no factible."
                )

            shared = (
                row["initial_solution_shared_p"]
                .strip()
                .lower()
            )

            if shared not in {
                "true",
                "t",
                "1",
                "1.0",
            }:
                errors.append(
                    f"{constructor}, {key}, {algorithm}: "
                    "no figura como solución inicial compartida."
                )

    signatures_by_constructor[
        constructor
    ] = constructor_signatures


deterministic_signatures = set(
    signatures_by_constructor
    .get("deterministic", {})
    .values()
)

if (
    signatures_by_constructor.get("deterministic")
    and len(deterministic_signatures) != 1
):
    errors.append(
        "El constructor determinista produjo soluciones "
        "iniciales diferentes para las semillas 91 y 92."
    )


if errors:
    print(
        f"P2 SMOKE TEST: ERROR "
        f"({len(errors)} problemas)"
    )

    for error in errors:
        print(f"- {error}")

    raise SystemExit(1)


print("P2 SMOKE TEST: CORRECTO")
print("Constructores comprobados: random y deterministic")
print("Pares comprobados: 4")
print("Corridas comprobadas: 8")

random_unique = len(
    set(
        signatures_by_constructor
        .get("random", {})
        .values()
    )
)

deterministic_unique = len(
    deterministic_signatures
)

print(
    "Firmas aleatorias diferentes: "
    f"{random_unique}"
)

print(
    "Firmas deterministas diferentes: "
    f"{deterministic_unique}"
)
