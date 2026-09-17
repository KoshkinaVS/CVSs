"""
Stage H (NAAD) — сравнение треков TempestExtremes (NAAD, по умолчанию LoRes)
с ручной разметкой EddyClicker — по lat/lon, тем же способом
(huracanpy.assess.match), что и Stage G (7_compare_TE_with_reference_NAAD.py).
Аналог 8_compare_TE_with_EddyClicker.py (ERA5), но с важным отличием:

Главное отличие от ERA5-версии — ЧТО с чем сравнивается
------------------------------------------------------------
ERA5-версия сравнивает TE-треки, посчитанные на ERA5 (широтно-долготная
сетка, другой регион/данные), с EddyClicker (ручная разметка NAAD LoRes) —
то есть кросс-датасетная методическая проверка. Здесь — сравниваются TE-треки,
посчитанные НА САМИХ ДАННЫХ NAAD (этим же пайплайном, 6_grid_runner_TE_NAAD.py),
с той же EddyClicker-разметкой NAAD LoRes — это прямая, "родная" проверка:
и то, и другое построено на одной и той же WRF-сетке/данных, просто одно -
автоматический трекинг, другое - ручная разметка. Именно поэтому здесь по
умолчанию `--data-type LoRes` (EddyClicker размечен на LoRes, см. докстринг
ERA5-версии) - HiRes можно передать явно, но тогда сравнение станет
кросс-разрешительным (LoRes-разметка vs HiRes-треки), это уже другой вопрос
(насколько согласуются LoRes и HiRes), не то, что этот скрипт проверяет по
умолчанию.

Формат EddyClicker-треков и остальная логика (matching/POD/статистика/графики)
— без изменений, см. докстринг ERA5-версии и tracks_comparison_lib.py.

Пример запуска
--------------
# все tracking_ok-комбинации LoRes из grid_run_log, карты по умолчанию
python 8_compare_TE_with_EddyClicker_NAAD.py

# одна конкретная комбинация
python 8_compare_TE_with_EddyClicker_NAAD.py --eps 2 --size-filter 10 --extr-type local \
    --maxgap 6 --mintime 18 --no-prioritize

# только посмотреть найденные комбинации и путь к EddyClicker-папке
python 8_compare_TE_with_EddyClicker_NAAD.py --dry-run

2026-09-05: НЕ прогонялось на реальных данных (нет доступа к серверу из этой
сессии) - логика 1:1 скопирована с уже рабочей ERA5-версии, изменено только
то, что описано в докстринге tracks_comparison_lib_NAAD.py и выше.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import tracks_comparison_lib_NAAD as lib
import naad_config as cfg

# === CONFIG (как в ERA5-версии - EddyClicker лежит на "сырой" стороне
# данных, /storage/.../data/LoRes/LoRes/..., НЕЗАВИСИМО от того, в какой
# combo-папке TempestExtremes лежат TE-треки) ===
PATH_INIT_RAW = "/storage/thalassa/users/vkoshkina"
PATH_DATA_DIR = f"{PATH_INIT_RAW}/data/LoRes/LoRes"
TRACKS_PARAMS_FOLDER = "EddyClicker_tracks_Egor_2010_15params_2028-08-10_r2d"
DEFAULT_EDDYCLICKER_DIR = Path(PATH_DATA_DIR) / TRACKS_PARAMS_FOLDER

REFERENCE_LABEL = "EddyClicker"
REFERENCE_ID_COLUMN = "id_EddyClicker"
REFERENCE_SUBPATH = "EddyClicker"


def run_combo(combo: dict, ec_tracks, plot_tracks: bool, force: bool) -> dict:
    label = lib.combo_label(combo)
    combo_dir = lib.combo_output_dir(combo, REFERENCE_SUBPATH)
    matches_csv = combo_dir / "matches_raw.csv"
    stats_csv = combo_dir / "matching_statistics.csv"

    if not force and stats_csv.exists():
        print(f"[{label}] {stats_csv} уже есть, пропуск (--force для пересчёта)")
        stats_df = pd.read_csv(stats_csv)
        summary = lib.summarize_track_statistics(
            stats_df, float(stats_df.detected.mean()) if len(stats_df) else float("nan"), len(stats_df), float("nan")
        )
        return {**{k: combo[k] for k in lib.COMBO_DIMS}, "years": ",".join(str(y) for y in combo["years"]), **summary}

    combo_dir.mkdir(parents=True, exist_ok=True)

    te_tracks = lib.load_te_tracks_for_combo(combo)
    n_reference = int(ec_tracks.track_id.hrcn.nunique())
    n_te = int(te_tracks.track_id.hrcn.nunique())
    print(f"[{label}] EddyClicker: {n_reference} треков, TE: {n_te} треков ({combo['years']})")

    matches = lib.calculate_matches(ec_tracks, te_tracks, REFERENCE_LABEL)
    matches.to_csv(matches_csv, index=False)

    pod = lib.calculate_pod(matches, ec_tracks, REFERENCE_LABEL)
    stats_df = lib.calculate_track_statistics(ec_tracks, te_tracks, matches, REFERENCE_ID_COLUMN)
    stats_df.to_csv(stats_csv, index=False)

    print(f"[{label}] POD={pod:.3f}, matched={int(stats_df.detected.sum())}/{len(stats_df)}")

    if plot_tracks:
        lib.plot_all_reference_tracks(ec_tracks, te_tracks, combo_dir / "tracks", REFERENCE_LABEL, REFERENCE_ID_COLUMN, matches)
        lib.plot_best_single_match_tracks(ec_tracks, te_tracks, stats_df, REFERENCE_LABEL, combo_dir / "tracks_1to1_best")

    summary = lib.summarize_track_statistics(stats_df, pod, n_reference, n_te)
    return {**{k: combo[k] for k in lib.COMBO_DIMS}, "years": ",".join(str(y) for y in combo["years"]), **summary}


def main():
    parser = argparse.ArgumentParser(
        description="Stage H (NAAD): TE (LoRes/HiRes) vs EddyClicker (ручная разметка NAAD LoRes)"
    )
    parser.add_argument("--data-type", choices=["LoRes", "HiRes"], default="LoRes",
                         help="LoRes по умолчанию - EddyClicker размечен на LoRes (см. докстринг модуля); "
                              "HiRes даёт кросс-разрешительное сравнение, не прямую проверку")
    parser.add_argument("--region", default=cfg.DEFAULT_REGION_NAME)
    parser.add_argument("--sigma", type=int, default=cfg.DEFAULT_SIGMA)
    parser.add_argument("--eddyclicker-dir", type=Path, default=DEFAULT_EDDYCLICKER_DIR,
                         help=f"Папка с одним CSV на трек (по умолчанию {DEFAULT_EDDYCLICKER_DIR})")
    parser.add_argument("--years", type=int, nargs="+", default=None,
                         help="По умолчанию - все годы, найденные в grid_run_log (EddyClicker обычно "
                              "размечен на один год - сузьте, если нужно)")
    parser.add_argument("--eps", type=int, default=None)
    parser.add_argument("--size-filter", type=int, default=None)
    parser.add_argument("--extr-type", choices=["global", "local"], default=None,
                         help="Без 'geom' - для NAAD такой Stage A не считался")
    parser.add_argument("--maxgap", type=int, default=None)
    parser.add_argument("--mintime", type=int, default=None)
    pr = parser.add_mutually_exclusive_group()
    pr.add_argument("--prioritize", dest="prioritize", action="store_true", default=None)
    pr.add_argument("--no-prioritize", dest="prioritize", action="store_false")
    parser.add_argument("--no-plot-tracks", dest="plot_tracks", action="store_false", default=True,
                         help="EddyClicker обычно маленький набор - карты по умолчанию ВКЛЮЧЕНЫ")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    lib.configure(data_type=args.data_type, region_name=args.region, sigma=args.sigma)
    output_root = lib.SIGMA_DIR / "EddyClicker_comparison_huracanpy"

    log_df = lib.read_combo_log()
    combos = lib.find_combos(
        log_df, years=args.years, eps=args.eps, size_filter=args.size_filter,
        extr_type=args.extr_type, maxgap=args.maxgap, mintime=args.mintime, prioritize=args.prioritize,
    )
    if not combos:
        print(f"Не найдено ни одной подходящей комбинации TE (NAAD {args.data_type}) в {lib.COMBO_LOG_PATH}.")
        return

    print(f"data_type={args.data_type} регион={args.region}")
    print(f"EddyClicker-папка: {args.eddyclicker_dir}")
    print(f"Найдено {len(combos)} комбинаций TE (после фильтров) в {lib.COMBO_LOG_PATH}:")
    for combo in combos:
        print(f"  {lib.combo_label(combo)} | years={combo['years']}")

    if args.dry_run:
        print("--dry-run: выполнение пропущено")
        return

    if not args.eddyclicker_dir.is_dir():
        print(f"ОШИБКА: папка {args.eddyclicker_dir} не найдена.")
        return

    ec_tracks, file_map = lib.load_eddyclicker_tracks(args.eddyclicker_dir)
    print(f"EddyClicker загружен: {int(ec_tracks.track_id.hrcn.nunique())} треков из {args.eddyclicker_dir}")

    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sorted(file_map.items()), columns=["track_id", "source_file"]).to_csv(
        output_root / "eddyclicker_file_map.csv", index=False
    )

    plot_tracks_effective = args.plot_tracks
    if args.plot_tracks and len(combos) > 5:
        print(f"\n--plot-tracks включён (по умолчанию) для {len(combos)} комбинаций — это "
              f"{len(combos)} x {int(ec_tracks.track_id.hrcn.nunique())} карт. Если это много, "
              f"сузьте фильтры или добавьте --no-plot-tracks.")

    summaries = []
    for combo in combos:
        try:
            summaries.append(run_combo(combo, ec_tracks, plot_tracks_effective, args.force))
        except Exception as e:
            print(f"ОШИБКА для {lib.combo_label(combo)}: {e}")
            continue

    summary_df = pd.DataFrame(summaries)
    summary_csv = output_root / "summary_all_configurations.csv"

    if summary_csv.exists():
        old_summary_df = pd.read_csv(summary_csv)
        n_old = len(old_summary_df)
        summary_df = lib.merge_summary_tables(old_summary_df, summary_df)
        print(f"\nСуществующая таблица {summary_csv} ({n_old} строк) слита с {len(summaries)} "
              f"строками этого запуска -> {len(summary_df)} строк итого (старые строки, не "
              f"затронутые текущими фильтрами, сохранены).")

    summary_df = lib.sort_summary_by_combined_score(summary_df)
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nСводная таблица (отсортирована по убыванию combined_score = "
          f"100*POD + median_overlap_percent + median_overlap_percent_1to1): {summary_csv}")
    print(summary_df.to_string(index=False))

    title = f"NAAD {args.data_type} vs EddyClicker"
    lib.plot_pod_bar(summary_df, output_root / "summary_pod_bar.png", title=f"Top POD — {title}")
    if len(summary_df) > 1:
        lib.plot_pod_vs_overlap_scatter(summary_df, output_root / "summary_pod_vs_overlap.png", title=title)
        lib.plot_configuration_summary(summary_df, output_root / "summary_configuration_grid.png", title=title)


if __name__ == "__main__":
    main()
