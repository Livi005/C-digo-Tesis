#!/usr/bin/env python3
"""Se limpian y validan todos los CSV utilizados en el análisis final."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from analysis_common import (
    build_pairs,
    clean_frame,
    find_project_root,
    output_dirs,
    read_csv,
    truthy,
    validate_status,
)


A_TIMES = [0.20, 0.50, 1.00, 1.50]
A_PROFILES = ["baseline", "efg-medium", "efg-high"]
MODES = ["best", "first", "random"]
F_HIGH_TIMES = [0.05, 0.25, 0.50, 1.00, 1.50]
SUMMARY_PATTERN = re.compile(
    r"summary-ivns-(baseline|efg-medium|efg-high)-(best|first|random)-time-([0-9.]+)"
)


def add_report(rows: list[dict[str, object]], dataset: str, check: str, expected: object, found: object, ok: bool) -> None:
    rows.append(
        {
            "dataset": dataset,
            "check": check,
            "expected": expected,
            "found": found,
            "status": "ok" if ok else "review",
        }
    )


def require_count(frame: pd.DataFrame, expected: int, label: str, report: list[dict[str, object]]) -> None:
    add_report(report, label, "rows", expected, len(frame), len(frame) == expected)
    if len(frame) != expected:
        raise RuntimeError(f"{label}: se esperaban {expected} filas y se encontraron {len(frame)}")


def read_official_variants(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = root / "results" / "oficiales-ivns-variants"
    summary = read_csv(base / "summary-ivns-variants-combined.csv")
    if "ivns_profile" not in summary.columns and "profile" in summary.columns:
        summary["ivns_profile"] = summary["profile"]
    if "variant" not in summary.columns:
        summary["variant"] = summary["ivns_profile"] + "-" + summary["vnd_mode"]
    summary["experiment_group"] = "official_variants"

    traces: list[pd.DataFrame] = []
    for path in sorted(base.glob("trace-ivns-*.csv")):
        stem = path.stem.replace("trace-ivns-", "")
        profile, mode = stem.rsplit("-", 1)
        frame = read_csv(path)
        frame["ivns_profile"] = profile
        frame["vnd_mode"] = mode
        frame["variant"] = f"{profile}-{mode}"
        frame["experiment_group"] = "official_variants"
        traces.append(frame)
    trace = pd.concat(traces, ignore_index=True)
    return clean_frame(summary), clean_frame(trace)


def read_official_comparison(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = root / "results" / "oficiales-final"
    summary = read_csv(base / "summary-main-baseline-random.csv")
    trace = read_csv(base / "trace-main-baseline-random.csv")
    summary["experiment_group"] = "official_vns_ivns"
    trace["experiment_group"] = "official_vns_ivns"
    return clean_frame(summary), clean_frame(trace)


def read_a_exploratory(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = root / "results" / "pruebas-personales"
    summaries: list[pd.DataFrame] = []
    traces: list[pd.DataFrame] = []
    for time_value in A_TIMES:
        time_dir = base / f"ivns-variants-time-{time_value:.2f}"
        for path in sorted(time_dir.glob("*/summary-ivns-*.csv")):
            match = SUMMARY_PATTERN.search(path.stem)
            if not match:
                continue
            profile, mode, parsed_time = match.groups()
            frame = read_csv(path)
            frame["ivns_profile"] = profile
            frame["vnd_mode"] = mode
            frame["variant"] = f"{profile}-{mode}"
            frame["ivns_time"] = float(parsed_time)
            frame["configuration"] = f"{profile}-{mode}-t{float(parsed_time):.2f}"
            frame["experiment_group"] = "a_time_exploratory"
            summaries.append(frame)
        for path in sorted(time_dir.glob("*/trace-ivns-*.csv")):
            match = SUMMARY_PATTERN.search(path.stem.replace("trace-", "summary-"))
            if not match:
                continue
            profile, mode, parsed_time = match.groups()
            frame = read_csv(path)
            frame["ivns_profile"] = profile
            frame["vnd_mode"] = mode
            frame["variant"] = f"{profile}-{mode}"
            frame["ivns_time"] = float(parsed_time)
            frame["configuration"] = f"{profile}-{mode}-t{float(parsed_time):.2f}"
            frame["experiment_group"] = "a_time_exploratory"
            traces.append(frame)
    return clean_frame(pd.concat(summaries, ignore_index=True)), clean_frame(pd.concat(traces, ignore_index=True))


def read_f_pilot(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = root / "results" / "pruebas-personales" / "f-n135-k7-piloto-time-0.05"
    summaries: list[pd.DataFrame] = []
    traces: list[pd.DataFrame] = []
    for path in sorted(base.glob("*/summary-f-n135-k7-*.csv")):
        variant = path.parent.name
        profile, mode = variant.rsplit("-", 1)
        frame = read_csv(path)
        frame["ivns_profile"] = profile
        frame["vnd_mode"] = mode
        frame["variant"] = variant
        frame["configuration"] = f"{variant}-t0.05"
        frame["ivns_time"] = 0.05
        frame["experiment_group"] = "f_pilot"
        summaries.append(frame)
    for path in sorted(base.glob("*/trace-f-n135-k7-*.csv")):
        variant = path.parent.name
        profile, mode = variant.rsplit("-", 1)
        frame = read_csv(path)
        frame["ivns_profile"] = profile
        frame["vnd_mode"] = mode
        frame["variant"] = variant
        frame["configuration"] = f"{variant}-t0.05"
        frame["ivns_time"] = 0.05
        frame["experiment_group"] = "f_pilot"
        traces.append(frame)
    return clean_frame(pd.concat(summaries, ignore_index=True)), clean_frame(pd.concat(traces, ignore_index=True))


def read_f_high(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = root / "results" / "pruebas-personales" / "f-n135-k7-high-times"
    summaries: list[pd.DataFrame] = []
    traces: list[pd.DataFrame] = []
    for time_value in F_HIGH_TIMES:
        for mode in MODES:
            folder = base / f"time-{time_value:.2f}" / mode
            summary_path = folder / f"summary-f-n135-k7-efg-high-{mode}-time-{time_value:.2f}.csv"
            trace_path = folder / f"trace-f-n135-k7-efg-high-{mode}-time-{time_value:.2f}.csv"
            summary = read_csv(summary_path)
            trace = read_csv(trace_path)
            for frame in [summary, trace]:
                frame["ivns_profile"] = "efg-high"
                frame["vnd_mode"] = mode
                frame["variant"] = f"efg-high-{mode}"
                frame["configuration"] = f"efg-high-{mode}-t{time_value:.2f}"
                frame["ivns_time"] = time_value
                frame["experiment_group"] = "f_high_exploratory"
            summaries.append(summary)
            traces.append(trace)
    return clean_frame(pd.concat(summaries, ignore_index=True)), clean_frame(pd.concat(traces, ignore_index=True))


def read_confirmatory(root: Path, kind: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if kind == "a":
        base = root / "results" / "confirmatorios" / "a-vns-vs-ivns-baseline-first-t1.00"
        summary_name = "summary-vns-ivns-baseline-first-t1.00.csv"
        trace_name = "trace-vns-ivns-baseline-first-t1.00.csv"
        group = "confirmatory_a"
    else:
        base = root / "results" / "confirmatorios" / "f-n135-k7-vns-vs-ivns-efg-high-first-t1.50"
        summary_name = "summary-f-n135-k7-vns-ivns-efg-high-first-t1.50.csv"
        trace_name = "trace-f-n135-k7-vns-ivns-efg-high-first-t1.50.csv"
        group = "confirmatory_f"
    summary = read_csv(base / summary_name)
    trace = read_csv(base / trace_name)
    summary["experiment_group"] = group
    trace["experiment_group"] = group
    return clean_frame(summary), clean_frame(trace)


def validate_balance(frame: pd.DataFrame, columns: list[str], expected: int, label: str) -> None:
    counts = frame.groupby(columns).size()
    if len(counts) == 0 or not (counts == expected).all():
        raise RuntimeError(f"{label}: el balance esperado no se cumple")


def main() -> None:
    root = find_project_root()
    clean_dir, tables_dir, _ = output_dirs(root)
    report: list[dict[str, object]] = []

    datasets = {
        "official_variants": (*read_official_variants(root), 810, 8100),
        "official_vns_ivns": (*read_official_comparison(root), 180, 1800),
        "a_exploratory": (*read_a_exploratory(root), 3240, 32400),
        "f_pilot": (*read_f_pilot(root), 30, 300),
        "f_high": (*read_f_high(root), 75, 750),
        "confirmatory_a": (*read_confirmatory(root, "a"), 180, 1800),
        "confirmatory_f": (*read_confirmatory(root, "f"), 60, 600),
    }

    for label, (summary, trace, expected_summary, expected_trace) in datasets.items():
        require_count(summary, expected_summary, f"{label}_summary", report)
        require_count(trace, expected_trace, f"{label}_trace", report)
        validate_status(summary, label)
        add_report(report, label, "status_and_feasibility", "valid", "valid", True)
        summary.to_csv(clean_dir / f"summary_{label}.csv", index=False)
        trace.to_csv(clean_dir / f"trace_{label}.csv", index=False)

    validate_balance(datasets["official_variants"][0], ["instance", "ivns_profile", "vnd_mode"], 30, "official_variants")
    validate_balance(datasets["official_vns_ivns"][0], ["instance", "algorithm"], 30, "official_vns_ivns")
    validate_balance(datasets["a_exploratory"][0], ["ivns_time", "instance", "ivns_profile", "vnd_mode"], 30, "a_exploratory")
    validate_balance(datasets["f_pilot"][0], ["ivns_profile", "vnd_mode"], 5, "f_pilot")
    validate_balance(datasets["f_high"][0], ["ivns_time", "vnd_mode"], 5, "f_high")
    validate_balance(datasets["confirmatory_a"][0], ["instance", "algorithm"], 30, "confirmatory_a")
    validate_balance(datasets["confirmatory_f"][0], ["instance", "algorithm"], 30, "confirmatory_f")

    for label in ["official_vns_ivns", "confirmatory_a", "confirmatory_f"]:
        pairs = build_pairs(datasets[label][0], label)
        expected_pairs = {"official_vns_ivns": 90, "confirmatory_a": 90, "confirmatory_f": 30}[label]
        valid = len(pairs) == expected_pairs and pairs["paired_valid"].all()
        add_report(report, label, "valid_pairs", expected_pairs, int(pairs["paired_valid"].sum()), valid)
        if not valid:
            raise RuntimeError(f"{label}: el pareo no es válido")
        pairs.to_csv(clean_dir / f"pairs_{label}.csv", index=False)

    pd.DataFrame(report).to_csv(tables_dir / "validation_report.csv", index=False)

    print("PROJECT_ROOT =", root)
    print("CLEAN_DIR =", clean_dir)
    print("TABLES_DIR =", tables_dir)
    for label, (summary, trace, _, _) in datasets.items():
        print(label, "summary=", len(summary), "trace=", len(trace))
    print("Validación completada")


if __name__ == "__main__":
    main()
