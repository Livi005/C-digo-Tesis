#!/usr/bin/env python3
"""Se analiza la comparación oficial pareada entre VNS e IVNS."""

from __future__ import annotations

import pandas as pd

from analysis_common import find_project_root, output_dirs, paired_statistics


def main() -> None:
    root = find_project_root()
    clean_dir, tables_dir, _ = output_dirs(root)
    pairs = pd.read_csv(clean_dir / "pairs_official_vns_ivns.csv")

    statistics = paired_statistics(pairs, "official_vns_ivns")
    pairs.to_csv(tables_dir / "official_vns_ivns_paired_values.csv", index=False)
    statistics.to_csv(tables_dir / "official_vns_ivns_wilcoxon_holm.csv", index=False)

    print("Pares oficiales =", len(pairs))
    print("Comparaciones estadísticas =", len(statistics))


if __name__ == "__main__":
    main()
