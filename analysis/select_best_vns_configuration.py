from __future__ import annotations

import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


INPUT_DIR = Path("results/calibracion-vns")
OUTPUT_RANKING = Path("results/calibracion-vns/ranking-vns.csv")
OUTPUT_LISP = Path("analysis/best_vns_config.lisp")

EXPECTED_RUNS = 90

FILE_PATTERN = re.compile(
    r"summary-vns-"
    r"(basic-3|five-5|extended-8)"
    r"-k(\d+)"
    r"-(first|best)\.csv$"
)

PROFILE_TO_LISP = {
    "basic-3": ":basic",
    "five-5": ":five",
    "extended-8": ":extended",
}


def parse_float(value: str) -> float:
    text = str(value).strip()

    if text == "":
        return math.nan

    try:
        return float(text)
    except ValueError:
        return math.nan


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {
        "true",
        "t",
        "1",
        "1.0",
        "yes",
    }


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)

    if not ordered:
        return math.nan

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * q
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
    return quantile(values, 0.75) - quantile(values, 0.25)


ranking: list[dict[str, object]] = []

summary_files = sorted(INPUT_DIR.glob("summary-vns-*.csv"))

if len(summary_files) != 18:
    raise SystemExit(
        f"ERROR: se encontraron {len(summary_files)} summaries; "
        "se esperaban 18."
    )


for path in summary_files:
    match = FILE_PATTERN.fullmatch(path.name)

    if match is None:
        raise SystemExit(
            f"ERROR: nombre de archivo no reconocido: {path.name}"
        )

    profile, kmax_text, mode = match.groups()
    kmax = int(kmax_text)

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        rows = list(csv.DictReader(stream))

    if len(rows) != EXPECTED_RUNS:
        raise SystemExit(
            f"ERROR: {path.name} contiene {len(rows)} corridas; "
            f"se esperaban {EXPECTED_RUNS}."
        )

    valid_rows = []
    feasible_count = 0

    gaps_by_instance: dict[str, list[float]] = defaultdict(list)
    times: list[float] = []

    for row in rows:
        status = str(row.get("status", "")).strip().lower()
        feasible = parse_bool(row.get("feasible", ""))

        gap = parse_float(row.get("gap_percent", ""))
        elapsed = parse_float(
            row.get("elapsed_time_seconds", "")
        )

        if feasible:
            feasible_count += 1

        if (
            status == "ok"
            and feasible
            and math.isfinite(gap)
            and math.isfinite(elapsed)
        ):
            valid_rows.append(row)
            gaps_by_instance[row["instance"]].append(gap)
            times.append(elapsed)

    if not gaps_by_instance:
        raise SystemExit(
            f"ERROR: {path.name} no contiene resultados válidos."
        )

    instance_medians = {
        instance: statistics.median(values)
        for instance, values in gaps_by_instance.items()
    }

    instance_iqrs = {
        instance: iqr(values)
        for instance, values in gaps_by_instance.items()
    }

    mean_instance_median_gap = statistics.mean(
        instance_medians.values()
    )

    mean_instance_iqr_gap = statistics.mean(
        instance_iqrs.values()
    )

    median_time = statistics.median(times)

    ranking.append(
        {
            "configuration": (
                f"{profile}-k{kmax}-{mode}"
            ),
            "profile": profile,
            "kmax": kmax,
            "vnd_mode": mode,
            "total_runs": len(rows),
            "valid_runs": len(valid_rows),
            "feasible_runs": feasible_count,
            "mean_instance_median_gap": (
                mean_instance_median_gap
            ),
            "mean_instance_iqr_gap": (
                mean_instance_iqr_gap
            ),
            "median_time_seconds": median_time,
            "median_gap_A-n33-k5": (
                instance_medians.get("A-n33-k5", math.nan)
            ),
            "median_gap_A-n65-k9": (
                instance_medians.get("A-n65-k9", math.nan)
            ),
            "median_gap_A-n80-k10": (
                instance_medians.get("A-n80-k10", math.nan)
            ),
        }
    )


ranking.sort(
    key=lambda row: (
        -int(row["valid_runs"]),
        -int(row["feasible_runs"]),
        float(row["mean_instance_median_gap"]),
        float(row["mean_instance_iqr_gap"]),
        float(row["median_time_seconds"]),
    )
)


fieldnames = list(ranking[0].keys())

with OUTPUT_RANKING.open(
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(ranking)


winner = ranking[0]

profile = str(winner["profile"])
kmax = int(winner["kmax"])
mode = str(winner["vnd_mode"])

with OUTPUT_LISP.open(
    "w",
    encoding="utf-8",
) as stream:
    stream.write("(in-package :cl-user)\n\n")

    stream.write(
        f'(defparameter *best-vns-profile-name* '
        f'"{profile}")\n'
    )

    stream.write(
        f"(defparameter *best-vns-criteria* "
        f"{PROFILE_TO_LISP[profile]})\n"
    )

    stream.write(
        f"(defparameter *best-vns-kmax* {kmax})\n"
    )

    stream.write(
        f"(defparameter *best-vns-vnd-mode* "
        f":{mode})\n"
    )


print()
print("RANKING DE CONFIGURACIONES VNS")
print("=" * 90)

for position, row in enumerate(ranking, start=1):
    print(
        f"{position:2d}. "
        f"{row['configuration']:<28} "
        f"gap={row['mean_instance_median_gap']:.6f}  "
        f"IQR={row['mean_instance_iqr_gap']:.6f}  "
        f"tiempo={row['median_time_seconds']:.6f}  "
        f"válidas={row['valid_runs']}"
    )

print()
print("MEJOR CONFIGURACIÓN:")
print(winner["configuration"])

print()
print(f"Ranking guardado en: {OUTPUT_RANKING}")
print(f"Configuración Lisp guardada en: {OUTPUT_LISP}")
