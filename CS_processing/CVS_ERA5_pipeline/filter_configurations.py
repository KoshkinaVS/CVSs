"""
filter_configurations.py — отбирает "хорошие" конфигурации из уже посчитанной
сводной таблицы (summary_all_configurations.csv, см. 7_compare_TE_with_reference.py
/ 8_compare_TE_with_EddyClicker.py) по порогам сразу на трёх метриках:

    POD                          - Probability of Detection (доля 0..1)
    median_overlap_percent       - медианное перекрытие (по числу точек трека)
    median_overlap_percent_1to1  - медианное перекрытие (по времени), но по
                                    ОДНОМУ, наиболее долго совпадающему TE-треку
                                    на референсный шторм (см. докстринг
                                    summarize_track_statistics в tracks_comparison_lib.py)

По умолчанию - строгий AND (конфигурация должна пройти ВСЕ три порога), см.
--any для OR. Пороги по умолчанию (POD>=0.5, оба overlap>=50%) - только
отправная точка, не "официальные" критерии - подберите под задачу флагами.

Пример запуска
--------------
# посмотреть, что пройдёт с порогами по умолчанию, ничего не сохранять
python filter_configurations.py \
    --summary-csv /storage/.../TC_comparison_huracanpy/summary_all_configurations.csv \
    --dry-run

# сохранить отфильтрованную таблицу + картинки по прошедшим конфигурациям
python filter_configurations.py \
    --summary-csv /storage/.../TC_comparison_huracanpy/summary_all_configurations.csv \
    --pod-min 0.5 --overlap-min 50 --overlap-1to1-min 50 \
    --output /storage/.../TC_comparison_huracanpy/summary_filtered.csv --plot

# ослабить до "хотя бы один критерий" (OR вместо AND)
python filter_configurations.py --summary-csv ... --any
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
# ВАЖНО: tracks_comparison_lib импортируется НЕ здесь (на уровне модуля), а
# только внутри main(), и только когда реально нужен (--plot) - сама lib.py
# тянет за собой cartopy/huracanpy/xarray на импорте (см. её докстринг), а
# простой фильтр/сортировка уже готовой CSV-таблицы этого не требует. Формула
# combined_score продублирована здесь как одна строка (см. sort_by_combined_score
# ниже) - если понадобится держать её в одном месте, см. lib.sort_summary_by_combined_score.

REQUIRED_COLUMNS = ["POD", "median_overlap_percent", "median_overlap_percent_1to1"]


def filter_summary(
    summary_df: pd.DataFrame,
    pod_min: float,
    overlap_min: float,
    overlap_1to1_min: float,
    require_all: bool = True,
) -> pd.DataFrame:
    """Возвращает подмножество строк summary_df, прошедших пороги.

    pod_min - в долях (0..1), overlap_min/overlap_1to1_min - в процентах
    (0..100), как и хранятся сами колонки в summary_all_configurations.csv.
    NaN в любой из трёх колонок (например, n_TE_tracks/POD не досчитаны для
    комбинации, взятой из кэша без --force) трактуется как "порог не
    пройден" по этой метрике - строка не проходит AND-фильтр по этому
    условию, но МОЖЕТ пройти OR-фильтр по остальным двум, если они посчитаны."""
    missing = [c for c in REQUIRED_COLUMNS if c not in summary_df.columns]
    if missing:
        raise ValueError(
            f"В таблице не хватает колонок: {missing} - похоже, это summary_all_configurations.csv "
            f"от старой версии 7_/8_compare_*.py (до 2026-09-02, когда добавили "
            f"median_overlap_percent_1to1). Пересчитайте с текущей версией скриптов."
        )

    pod_ok = summary_df["POD"] >= pod_min
    overlap_ok = summary_df["median_overlap_percent"] >= overlap_min
    overlap_1to1_ok = summary_df["median_overlap_percent_1to1"] >= overlap_1to1_min

    mask = (pod_ok & overlap_ok & overlap_1to1_ok) if require_all else (pod_ok | overlap_ok | overlap_1to1_ok)
    result = summary_df[mask].copy()
    return sort_by_combined_score(result)


def sort_by_combined_score(df: pd.DataFrame) -> pd.DataFrame:
    """Та же формула и та же сортировка, что и в summary_all_configurations.csv
    (см. tracks_comparison_lib.sort_summary_by_combined_score) - продублирована
    здесь одной строкой, чтобы не тянуть cartopy/huracanpy/xarray (импортируются
    при загрузке tracks_comparison_lib.py) только ради сортировки готовой таблицы."""
    df = df.copy()
    df["combined_score"] = 100 * df["POD"] + df["median_overlap_percent"] + df["median_overlap_percent_1to1"]
    return df.sort_values("combined_score", ascending=False, na_position="last").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Фильтрует summary_all_configurations.csv по порогам на POD/median_overlap_percent/"
                     "median_overlap_percent_1to1 - отбор конфигураций сетки, которые действительно "
                     "хорошо воспроизводят треки (не только детектируют шторм - POD, но и хорошо "
                     "повторяют его форму/длительность - оба overlap)."
    )
    parser.add_argument("--summary-csv", required=True, type=Path,
                         help="Путь к summary_all_configurations.csv (печатается в конце "
                              "прогона 7_compare_TE_with_reference.py / 8_compare_TE_with_EddyClicker.py)")
    parser.add_argument("--pod-min", type=float, default=0.5, help="Порог POD, доля 0..1 (по умолчанию 0.5)")
    parser.add_argument("--overlap-min", type=float, default=50.0,
                         help="Порог median_overlap_percent, % 0..100 (по умолчанию 50)")
    parser.add_argument("--overlap-1to1-min", type=float, default=50.0,
                         help="Порог median_overlap_percent_1to1, % 0..100 (по умолчанию 50)")
    parser.add_argument("--any", dest="require_all", action="store_false",
                         help="OR вместо AND - конфигурация проходит, если выполнен ХОТЯ БЫ ОДИН "
                              "из трёх порогов (по умолчанию нужны все три)")
    parser.add_argument("--output", type=Path, default=None,
                         help="Куда сохранить отфильтрованную таблицу CSV (по умолчанию - "
                              "рядом с --summary-csv, summary_filtered.csv)")
    parser.add_argument("--plot", action="store_true",
                         help="Дополнительно перерисовать summary_pod_bar.png/summary_pod_vs_overlap.png/"
                              "summary_configuration_grid.png ТОЛЬКО по отфильтрованным конфигурациям "
                              "(рядом с --output, с суффиксом _filtered)")
    parser.add_argument("--dry-run", action="store_true", help="Только показать результат, ничего не сохранять")
    args = parser.parse_args()

    summary_df = pd.read_csv(args.summary_csv)
    print(f"Загружено {len(summary_df)} строк из {args.summary_csv}")

    filtered = filter_summary(
        summary_df, pod_min=args.pod_min, overlap_min=args.overlap_min,
        overlap_1to1_min=args.overlap_1to1_min, require_all=args.require_all,
    )

    mode = "И (все три порога)" if args.require_all else "ИЛИ (хотя бы один порог)"
    print(
        f"\nПороги: POD >= {args.pod_min}, median_overlap_percent >= {args.overlap_min}%, "
        f"median_overlap_percent_1to1 >= {args.overlap_1to1_min}% ({mode})"
    )
    print(f"Прошло: {len(filtered)} из {len(summary_df)} конфигураций\n")

    show_cols = [c for c in [
        "eps", "size_filter", "extr_type", "maxgap", "mintime", "prioritize",
        "POD", "median_overlap_percent", "median_overlap_percent_1to1", "combined_score",
    ] if c in filtered.columns]
    print(filtered[show_cols].to_string(index=False))

    if args.dry_run:
        print("\n--dry-run: сохранение/графики пропущены")
        return

    output_path = args.output or (args.summary_csv.parent / "summary_filtered.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_path, index=False)
    print(f"\nСохранено: {output_path}")

    if args.plot:
        import importlib
        lib = importlib.import_module("tracks_comparison_lib")
        plot_dir = output_path.parent
        title = f"Filtered (POD>={args.pod_min}, overlap>={args.overlap_min}%, 1to1>={args.overlap_1to1_min}%)"
        if len(filtered) == 0:
            print("Отфильтрованная таблица пуста - графики не строятся.")
            return
        lib.plot_pod_bar(filtered, plot_dir / "summary_pod_bar_filtered.png", title=f"Top POD — {title}")
        if len(filtered) > 1:
            lib.plot_pod_vs_overlap_scatter(filtered, plot_dir / "summary_pod_vs_overlap_filtered.png", title=title)
            lib.plot_configuration_summary(filtered, plot_dir / "summary_configuration_grid_filtered.png", title=title)


if __name__ == "__main__":
    main()
