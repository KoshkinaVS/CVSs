"""
plot_top_configs.py — рисует карты треков (tracks/ + tracks_1to1_best/) ТОЛЬКО
для N лучших конфигураций из уже посчитанной summary_all_configurations.csv, по
combined_score = 100*POD + median_overlap_percent + median_overlap_percent_1to1
(см. tracks_comparison_lib.sort_summary_by_combined_score).

Рабочий процесс, под который написан этот скрипт
--------------------------------------------------
1) Сначала полная сводная статистика по ВСЕЙ сетке БЕЗ картинок (быстро -
   huracanpy.assess.match всё равно самая дорогая часть, но без отрисовки
   сотен PNG по каждому шторму каждой комбинации):

     python 7_compare_TE_with_reference.py --reference ibtracs --years 2010

   (БЕЗ --plot-tracks) - на выходе отсортированная summary_all_configurations.csv
   + matches_raw.csv/matching_statistics.csv на диске для каждой посчитанной
   комбинации (внутри её combo-папки, см. докстринг tracks_comparison_lib.
   combo_output_dir).

2) Отдельно, только для N лучших (по combined_score) - карты:

     python plot_top_configs.py --reference ibtracs \
         --summary-csv /storage/.../TC_comparison_huracanpy/summary_all_configurations.csv \
         --top-n 20

Matching (huracanpy.assess.match) здесь НЕ пересчитывается - берётся из уже
посчитанных на шаге (1) matches_raw.csv/matching_statistics.csv. Для отрисовки
карт всё равно нужны сырые треки (lon/lat), которых нет в этих CSV (там только
id/статистика) - поэтому ref/te_tracks подгружаются заново, но это дёшево по
сравнению с самим matching.

Если для какой-то из top-N конфигураций matches_raw.csv ещё не существует
(комбинация не была посчитана на шаге 1 - например, попала в grid_run_log.csv
позже) - по умолчанию она пропускается с предупреждением; --force-recompute
досчитает её здесь же (полный run_combo с matching, а не только карты).

Пример запуска
--------------
python plot_top_configs.py --reference ibtracs \
    --summary-csv /storage/.../TC_comparison_huracanpy/summary_all_configurations.csv \
    --top-n 20

python plot_top_configs.py --reference syclops --syclops-type TC --syclops-region NA \
    --summary-csv /storage/.../SyCLoPS_comparison_huracanpy/TC_NA/summary_all_configurations.csv \
    --top-n 10 --force-recompute
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import tracks_comparison_lib as lib

seven = importlib.import_module("7_compare_TE_with_reference")


def main():
    parser = argparse.ArgumentParser(
        description="Рисует tracks/ + tracks_1to1_best/ только для N лучших (по combined_score) "
                     "конфигураций из уже посчитанной summary_all_configurations.csv - без "
                     "пересчёта matching для остальной сетки."
    )
    parser.add_argument("--reference", required=True, choices=["ibtracs", "syclops"])
    parser.add_argument("--syclops-type", choices=lib.SYCLOPS_TYPES, default=None,
                         help="Обязателен при --reference syclops")
    parser.add_argument("--syclops-region", default="NA")
    parser.add_argument("--syclops-on-multiple", choices=["newest", "all"], default="newest")
    parser.add_argument("--summary-csv", required=True, type=Path,
                         help="summary_all_configurations.csv от 7_compare_TE_with_reference.py "
                              "(тот же --reference/--syclops-*, что здесь)")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument(
        "--force-recompute", action="store_true",
        help="Если для какой-то из top-N конфигураций matches_raw.csv/matching_statistics.csv ещё "
             "нет - посчитать её здесь же (обычный run_combo с matching), а не просто пропустить."
    )
    args = parser.parse_args()

    if args.reference == "syclops" and args.syclops_type is None:
        parser.error("--reference syclops требует --syclops-type {TC,SS,PL}")

    summary_df = pd.read_csv(args.summary_csv)
    if "combined_score" in summary_df.columns:
        summary_df = summary_df.sort_values("combined_score", ascending=False, na_position="last").reset_index(drop=True)
    else:
        print(f"В {args.summary_csv} нет combined_score (старая версия таблицы?) - считаю здесь же.")
        summary_df = lib.sort_summary_by_combined_score(summary_df)

    top = summary_df.head(args.top_n)
    print(f"Топ-{len(top)} из {len(summary_df)} конфигураций по combined_score:\n")
    show_cols = [c for c in [
        "eps", "size_filter", "extr_type", "maxgap", "mintime", "prioritize",
        "POD", "median_overlap_percent", "median_overlap_percent_1to1", "combined_score",
    ] if c in top.columns]
    print(top[show_cols].to_string(index=False))
    print()

    log_df = lib.read_combo_log()
    n_plotted = 0
    n_recomputed = 0
    n_skipped = 0

    for _, row in top.iterrows():
        years = [int(y) for y in str(row["years"]).split(",")]
        combos = lib.find_combos(
            log_df, years=years, eps=int(row["eps"]), size_filter=int(row["size_filter"]),
            extr_type=row["extr_type"], maxgap=int(row["maxgap"]), mintime=int(row["mintime"]),
            prioritize=bool(row["prioritize"]),
        )
        if not combos:
            print(f"[{lib.combo_label(row)}] не найдено в grid_run_log.csv (tracks_txt отсутствует?) - пропуск")
            n_skipped += 1
            continue
        combo = combos[0]  # find_combos уже схлопнула дубли по году (см. её докстринг)

        label = lib.combo_label(combo)
        combo_dir = lib.combo_output_dir(
            combo, seven.reference_subpath(args.reference, args.syclops_type, args.syclops_region)
        )
        matches_csv = combo_dir / "matches_raw.csv"
        stats_csv = combo_dir / "matching_statistics.csv"

        if not stats_csv.exists() or not matches_csv.exists():
            if not args.force_recompute:
                missing = stats_csv if not stats_csv.exists() else matches_csv
                print(f"[{label}] {missing} не найден - комбинация ещё не посчитана "
                      f"(запустите 7_compare_TE_with_reference.py на неё, или добавьте "
                      f"--force-recompute) - пропуск")
                n_skipped += 1
                continue
            print(f"[{label}] статистика ещё не посчитана - считаю здесь же (--force-recompute)...")
            seven.run_combo(
                combo, args.reference, args.syclops_type, args.syclops_region,
                args.syclops_on_multiple, plot_tracks=True, force=True,
            )
            n_recomputed += 1
            n_plotted += 1
            continue

        print(f"[{label}] рисую карты (matching уже посчитан, не пересчитываю)...")
        stats_df = pd.read_csv(stats_csv)
        ok = seven.plot_combo_tracks_from_cache(
            combo, args.reference, args.syclops_type, args.syclops_region, args.syclops_on_multiple,
            combo_dir, matches_csv, stats_df, label,
        )
        if ok:
            n_plotted += 1
        else:
            n_skipped += 1

    print(
        f"\nГотово: карты нарисованы для {n_plotted} из {len(top)} топ-конфигураций "
        f"({n_recomputed} потребовали пересчёта matching, {n_skipped} пропущено)."
    )


if __name__ == "__main__":
    main()
