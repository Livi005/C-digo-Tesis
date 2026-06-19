#!/usr/bin/env python3
"""Se reúnen funciones comunes de lectura, resumen y comparación."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


METRICS = ["gap_percent", "elapsed_time_seconds"]
TRUE_VALUES = {"true", "t", "1", "yes", "y", "si", "sí"}


def find_project_root() -> Path:
    current = Path.cwd().resolve()
    for base in [current, *current.parents]:
        if (base / "results").exists() and (base / "analysis").exists():
            return base
    return current


def output_dirs(root: Path) -> tuple[Path, Path, Path]:
    base = root / "results" / "analysis_final"
    clean = base / "clean"
    tables = base / "tables"
    figures = base / "figures"
    for path in [clean, tables, figures]:
        path.mkdir(parents=True, exist_ok=True)
    return clean, tables, figures


def normalize_column(name: object) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def normalize_source(series: pd.Series) -> pd.Series:
    return normalize_text(series).str.replace("-", "_", regex=False)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo requerido: {path}")
    frame = pd.read_csv(path)
    frame.columns = [normalize_column(column) for column in frame.columns]
    frame["source_path"] = str(path)
    return clean_frame(frame)


def truthy(series: pd.Series) -> pd.Series:
    return normalize_text(series).isin(TRUE_VALUES)


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    text_columns = [
        "algorithm",
        "status",
        "feasible",
        "error",
        "vnd_mode",
        "ivns_profile",
        "profile",
        "variant",
        "configuration",
        "initial_solution_signature",
        "initial_solution_shared_p",
        "initial_solution_source",
    ]
    for column in text_columns:
        if column in out.columns:
            out[column] = normalize_text(out[column])
    if "initial_solution_source" in out.columns:
        out["initial_solution_source"] = normalize_source(out["initial_solution_source"])
    if "instance" in out.columns:
        out["instance"] = out["instance"].fillna("").astype(str).str.strip()
    numeric_columns = [
        "seed",
        "known_best",
        "initial_cost",
        "best_cost",
        "final_cost",
        "gap_percent",
        "elapsed_time_seconds",
        "iterations",
        "accepted_improvements",
        "invalid_neighbors",
        "no_improvement_neighbors",
        "timeout_neighbors",
        "neighbors_evaluated",
        "generation_tries",
        "best_iteration",
        "stagnation_iterations",
        "ivns_time",
        "ivns_max_neighborhood_seconds",
        "ivns_complexity_avg",
        "ivns_strings_evaluated_total",
    ]
    for column in numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    if "seed" in out.columns and out["seed"].notna().all():
        out["seed"] = out["seed"].astype(int)
    return out


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise RuntimeError(f"{label}: faltan columnas {sorted(missing)}")


def validate_status(frame: pd.DataFrame, label: str) -> None:
    require_columns(frame, ["status", "feasible"], label)
    if set(normalize_text(frame["status"])) != {"ok"}:
        raise RuntimeError(f"{label}: existen estados distintos de ok")
    if not truthy(frame["feasible"]).all():
        raise RuntimeError(f"{label}: existen soluciones no factibles")
    if "error" in frame.columns:
        errors = frame["error"].fillna("").astype(str).str.strip()
        if (errors != "").any():
            raise RuntimeError(f"{label}: existen mensajes en la columna error")


def descriptive(values: pd.Series) -> dict[str, float | int]:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "iqr": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    q1 = float(x.quantile(0.25))
    q3 = float(x.quantile(0.75))
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "median": float(x.median()),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "min": float(x.min()),
        "max": float(x.max()),
    }


def descriptive_table(
    frame: pd.DataFrame,
    group_columns: list[str],
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    selected_metrics = metrics or METRICS
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(group_columns, dropna=False, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_columns, key_values))
        for metric in selected_metrics:
            row = dict(base)
            row["metric"] = metric
            row.update(descriptive(group[metric]))
            rows.append(row)
    return pd.DataFrame(rows)


def holm_adjust(pvalues: Iterable[float]) -> list[float]:
    values = [float(value) for value in pvalues]
    count = len(values)
    if count == 0:
        return []
    order = sorted(range(count), key=lambda index: values[index])
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def apply_holm(frame: pd.DataFrame, family_columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    out["p_value_holm"] = np.nan
    out["decision"] = ""
    if out.empty:
        return out
    if family_columns:
        grouping = family_columns[0] if len(family_columns) == 1 else family_columns
        groups = out.groupby(grouping, dropna=False).groups.values()
    else:
        groups = [out.index]
    for indexes in groups:
        idx = list(indexes)
        adjusted = holm_adjust(out.loc[idx, "p_value"].tolist())
        out.loc[idx, "p_value_holm"] = adjusted
        out.loc[idx, "decision"] = [
            "significativo" if value < 0.05 else "no_significativo"
            for value in adjusted
        ]
    return out


def cliffs_delta_lower_is_better(x: pd.Series, y: pd.Series) -> float:
    xv = pd.to_numeric(x, errors="coerce").dropna().to_numpy()
    yv = pd.to_numeric(y, errors="coerce").dropna().to_numpy()
    if len(xv) == 0 or len(yv) == 0:
        return float("nan")
    lower = sum(int(np.sum(value < yv)) for value in xv)
    higher = sum(int(np.sum(value > yv)) for value in xv)
    return float((lower - higher) / (len(xv) * len(yv)))


def rank_biserial_lower_is_better(differences: pd.Series) -> float:
    values = pd.to_numeric(differences, errors="coerce").dropna()
    values = values[values.abs() > 1e-12]
    if values.empty:
        return 0.0
    ranks = stats.rankdata(values.abs())
    favorable = float(ranks[values.to_numpy() < 0].sum())
    unfavorable = float(ranks[values.to_numpy() > 0].sum())
    total = favorable + unfavorable
    return 0.0 if total == 0 else (favorable - unfavorable) / total


def safe_wilcoxon(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    xv = pd.to_numeric(x, errors="coerce")
    yv = pd.to_numeric(y, errors="coerce")
    valid = xv.notna() & yv.notna()
    xv = xv[valid]
    yv = yv[valid]
    if len(xv) == 0 or np.allclose(xv.to_numpy(), yv.to_numpy()):
        return 0.0, 1.0
    result = stats.wilcoxon(
        xv,
        yv,
        zero_method="wilcox",
        alternative="two-sided",
        method="auto",
    )
    return float(result.statistic), float(result.pvalue)


def safe_kruskal(groups: list[pd.Series]) -> tuple[float, float]:
    arrays = [pd.to_numeric(group, errors="coerce").dropna().to_numpy() for group in groups]
    arrays = [array for array in arrays if len(array)]
    if len(arrays) < 2:
        return float("nan"), float("nan")
    result = stats.kruskal(*arrays)
    return float(result.statistic), float(result.pvalue)


def mannwhitney(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    xv = pd.to_numeric(x, errors="coerce").dropna()
    yv = pd.to_numeric(y, errors="coerce").dropna()
    result = stats.mannwhitneyu(xv, yv, alternative="two-sided", method="auto")
    return float(result.statistic), float(result.pvalue)


def pairwise_mannwhitney(
    frame: pd.DataFrame,
    group_column: str,
    metric: str,
    base: dict[str, object],
    selected_pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    labels = sorted(frame[group_column].dropna().astype(str).unique())
    pairs = selected_pairs or list(combinations(labels, 2))
    rows: list[dict[str, object]] = []
    for left, right in pairs:
        x = frame.loc[frame[group_column].astype(str) == left, metric]
        y = frame.loc[frame[group_column].astype(str) == right, metric]
        statistic, p_value = mannwhitney(x, y)
        row = dict(base)
        row.update(
            {
                "metric": metric,
                "group_a": left,
                "group_b": right,
                "n_a": int(pd.to_numeric(x, errors="coerce").notna().sum()),
                "n_b": int(pd.to_numeric(y, errors="coerce").notna().sum()),
                "median_a": float(pd.to_numeric(x, errors="coerce").median()),
                "median_b": float(pd.to_numeric(y, errors="coerce").median()),
                "statistic": statistic,
                "p_value": p_value,
                "cliffs_delta_positive_favors_a": cliffs_delta_lower_is_better(x, y),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_pairs(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    required = {
        "instance",
        "seed",
        "algorithm",
        "initial_solution_signature",
        "initial_cost",
        "initial_solution_shared_p",
        "initial_solution_source",
        "gap_percent",
        "elapsed_time_seconds",
    }
    require_columns(frame, required, label)
    work = clean_frame(frame)
    duplicated = work.duplicated(["instance", "seed", "algorithm"], keep=False)
    if duplicated.any():
        raise RuntimeError(f"{label}: existen filas duplicadas por instancia, semilla y algoritmo")
    if set(work["algorithm"]) != {"vns", "ivns"}:
        raise RuntimeError(f"{label}: no se encontraron VNS e IVNS")

    pairs = work.pivot(index=["instance", "seed"], columns="algorithm")
    result = pd.DataFrame(index=pairs.index).reset_index()
    fields = [
        "initial_solution_signature",
        "initial_cost",
        "initial_solution_shared_p",
        "initial_solution_source",
        "gap_percent",
        "elapsed_time_seconds",
        "best_cost",
    ]
    for field in fields:
        if field not in work.columns:
            continue
        for algorithm in ["vns", "ivns"]:
            result[f"{field}_{algorithm}"] = pairs[(field, algorithm)].to_numpy()

    signature_vns = normalize_text(result["initial_solution_signature_vns"])
    signature_ivns = normalize_text(result["initial_solution_signature_ivns"])
    shared_vns = truthy(result["initial_solution_shared_p_vns"])
    shared_ivns = truthy(result["initial_solution_shared_p_ivns"])
    source_vns = normalize_source(result["initial_solution_source_vns"])
    source_ivns = normalize_source(result["initial_solution_source_ivns"])

    result["same_initial_signature"] = (
        signature_vns.ne("")
        & signature_ivns.ne("")
        & signature_vns.eq(signature_ivns)
    )
    result["same_initial_cost"] = pd.to_numeric(
        result["initial_cost_vns"], errors="coerce"
    ).eq(pd.to_numeric(result["initial_cost_ivns"], errors="coerce"))
    result["shared_initial_solution"] = shared_vns & shared_ivns
    result["shared_initial_source"] = source_vns.eq("shared_by_seed") & source_ivns.eq(
        "shared_by_seed"
    )
    result["paired_valid"] = (
        result["same_initial_signature"]
        & result["same_initial_cost"]
        & result["shared_initial_solution"]
        & result["shared_initial_source"]
    )
    result["delta_gap_ivns_minus_vns"] = (
        result["gap_percent_ivns"] - result["gap_percent_vns"]
    )
    result["delta_time_ivns_minus_vns"] = (
        result["elapsed_time_seconds_ivns"]
        - result["elapsed_time_seconds_vns"]
    )
    return result


def paired_statistics(pairs: pd.DataFrame, dataset: str, holm_by_metric: bool = True) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    mappings = {
        "gap_percent": (
            "gap_percent_ivns",
            "gap_percent_vns",
            "delta_gap_ivns_minus_vns",
        ),
        "elapsed_time_seconds": (
            "elapsed_time_seconds_ivns",
            "elapsed_time_seconds_vns",
            "delta_time_ivns_minus_vns",
        ),
    }
    for instance, group in pairs.groupby("instance", sort=True):
        for metric, (ivns_column, vns_column, difference_column) in mappings.items():
            statistic, p_value = safe_wilcoxon(group[ivns_column], group[vns_column])
            differences = group[difference_column]
            rows.append(
                {
                    "dataset": dataset,
                    "instance": instance,
                    "metric": metric,
                    "n_pairs": int(len(group)),
                    "mean_ivns": float(group[ivns_column].mean()),
                    "mean_vns": float(group[vns_column].mean()),
                    "median_ivns": float(group[ivns_column].median()),
                    "median_vns": float(group[vns_column].median()),
                    "mean_difference_ivns_minus_vns": float(differences.mean()),
                    "median_difference_ivns_minus_vns": float(differences.median()),
                    "wins_ivns": int((differences < 0).sum()),
                    "wins_vns": int((differences > 0).sum()),
                    "ties": int((differences.abs() <= 1e-12).sum()),
                    "statistic": statistic,
                    "p_value": p_value,
                    "rank_biserial_positive_favors_ivns": rank_biserial_lower_is_better(
                        differences
                    ),
                }
            )
    table = pd.DataFrame(rows)
    return apply_holm(table, ["metric"] if holm_by_metric else [])
