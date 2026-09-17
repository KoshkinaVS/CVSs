"""
Stage H (альтернативная валидация) — сравнение треков TempestExtremes (ERA5,
широтно-долготная сетка) с ручной разметкой EddyClicker (NAAD LoRes,
декартова сетка x/y) — по lat/lon, а не по x/y.

Почему по lat/lon, а не по декартовым координатам
--------------------------------------------------
Старый CS_processing/CVS_alt_tracking/TempestExtremes/find_closest_tracks_EddyClicker.py
сравнивал треки в декартовых координатах (x/y узлов сетки LoRes) — это работало,
только когда ОБА трека на одной и той же сетке. Здесь один трек (ERA5 TE) — на
широтно-долготной сетке, другой (EddyClicker) — на декартовой сетке NAAD LoRes,
поэтому напрямую x/y сравнивать нельзя. Но EddyClicker-треки в папке
tracks_params_folder уже содержат lat/lon (см. EddyClicker_tracks_processing/
add_latlon.py: latitude/longitude получены по индексам pxc_ind/pyc_ind из
XLAT/XLONG соответствующего WRF-файла NAAD) — поэтому сравнение здесь идёт по
lat/lon, тем же способом (huracanpy.assess.match), что и Stage G для
IBTrACS/SyCLoPS (7_compare_TE_with_reference.py). Вся общая логика
(matching/POD/статистика/графики) — из tracks_comparison_lib.py, этот файл
только про то, откуда взять EddyClicker-треки.

Формат EddyClicker-треков
--------------------------
tracks_params_folder — один CSV НА ТРЕК (см. EddyClicker_tracks_processing/
add_track_params_EddyClicker_2026_v1.py: файлы читаются как
sorted(glob.glob(f'{path_tracks_dir}/*.csv')), результат сохраняется под тем
же именем файла в новую папку). Колонки lat/lon называются по-разному в
разных версиях скриптов (add_latlon.py пишет 'latitude'/'longitude',
add_track_params_* ожидает 'lat'/'lon') — lib.load_eddyclicker_tracks()
подхватывает любое из этих имён автоматически (см. её докстринг).
track_id присваивается по порядку файлов (0..N-1) — сохранён вместе с
matching_statistics.csv как reference_ID, соответствие "какой track_id —
какой файл" сохраняется отдельно в eddyclicker_file_map.csv.

EddyClicker — маленький вручную размеченный набор (обычно один год/регион),
поэтому, в отличие от Stage G, карты по каждому треку (--plot-tracks)
включены по умолчанию — секунды на единицы-десятки треков, а не тысячи PNG.

Пример запуска
--------------
# все ok/pending_stage_e_prime комбинации ERA5 из grid_run_log.csv, карты по умолчанию
python 8_compare_TE_with_EddyClicker.py

# одна конкретная комбинация, своя папка EddyClicker
python 8_compare_TE_with_EddyClicker.py --eps 1 --size-filter 25 --extr-type global \
    --maxgap 3 --mintime 18 --no-prioritize \
    --eddyclicker-dir /storage/thalassa/users/vkoshkina/data/LoRes/LoRes/EddyClicker_tracks_Egor_2010_15params_2028-08-10_r2d

# только посмотреть найденные комбинации и путь к EddyClicker-папке
python 8_compare_TE_with_EddyClicker.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import tracks_comparison_lib as lib

# === CONFIG (как в примере пользователя) ===
PATH_INIT = "/storage/thalassa/users/vkoshkina"
PATH_DATA_DIR = f"{PATH_INIT}/data/LoRes/LoRes"
TRACKS_PARAMS_FOLDER = "EddyClicker_tracks_Egor_2010_15params_2028-08-10_r2d"
DEFAULT_EDDYCLICKER_DIR = Path(PATH_DATA_DIR) / TRACKS_PARAMS_FOLDER

# СВОДКА по ВСЕЙ сетке сразу (summary_all_configurations.csv + графики + общий
# eddyclicker_file_map.csv) - на уровне sigma_dir (lib.SIGMA_DIR), у неё нет
# "своей" combo-папки. Детальные результаты ОДНОЙ комбинации (matches_raw.csv,
# matching_statistics.csv, карты треков) - см. run_combo() - лежат ВНУТРИ её
# собственной combo-папки (lib.combo_output_dir), не здесь.
OUTPUT_ROOT = lib.SIGMA_DIR / "EddyClicker_comparison_huracanpy"

REFERENCE_LABEL = "EddyClicker"
REFERENCE_ID_COLUMN = "id_EddyClicker"
REFERENCE_SUBPATH = "EddyClicker"  # подпапка внутри data_comparison_huracanpy/ каждой combo-папки


def run_combo(combo: dict, ec_tracks, plot_tracks: bool, force: bool) -> dict:
    label = lib.combo_label(combo)
    combo_dir = lib.combo_output_dir(combo, REFERENCE_SUBPATH)
    matches_csv = combo_dir / "matches_raw.csv"
    stats_csv = combo_dir / "matching_statistics.csv"

    if not force and stats_csv.exists():
        print(f"[{label}] {stats_csv} уже есть, пропуск (--force для пересчёта)")
        stats_df = pd.read_csv(stats_csv)
        # 2026-09-02: n_te_tracks=float('nan'), не None - см. коммент у _summarize_row
        # в 7_compare_TE_with_reference.py про TypeError в plot_configuration_summary
        # panel (c), когда сюда попадал python None вместо числового NaN.
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
        # 2026-09-02: см. коммент в 7_compare_TE_with_reference.py - отдельные "1-к-1"
        # картинки (референс + РОВНО один, наиболее долго совпадающий TE-трек).
        lib.plot_best_single_match_tracks(ec_tracks, te_tracks, stats_df, REFERENCE_LABEL, combo_dir / "tracks_1to1_best")

    summary = lib.summarize_track_statistics(stats_df, pod, n_reference, n_te)
    return {**{k: combo[k] for k in lib.COMBO_DIMS}, "years": ",".join(str(y) for y in combo["years"]), **summary}


def main():
    parser = argparse.ArgumentParser(
        description="Stage H: TE (ERA5, lat/lon) vs EddyClicker (ручная разметка NAAD LoRes, тоже по lat/lon)"
    )
    parser.add_argument("--eddyclicker-dir", type=Path, default=DEFAULT_EDDYCLICKER_DIR,
                         help=f"Папка с одним CSV на трек (по умолчанию {DEFAULT_EDDYCLICKER_DIR})")
    parser.add_argument("--years", type=int, nargs="+", default=None,
                         help="По умолчанию — все годы, найденные в grid_run_log.csv (обычно EddyClicker размечен на один год — сузьте, если нужно)")
    parser.add_argument("--eps", type=int, default=None)
    parser.add_argument("--size-filter", type=int, default=None)
    parser.add_argument("--extr-type", choices=["global", "local", "geom"], default=None)
    parser.add_argument("--maxgap", type=int, default=None)
    parser.add_argument("--mintime", type=int, default=None)
    pr = parser.add_mutually_exclusive_group()
    pr.add_argument("--prioritize", dest="prioritize", action="store_true", default=None)
    pr.add_argument("--no-prioritize", dest="prioritize", action="store_false")
    parser.add_argument("--no-plot-tracks", dest="plot_tracks", action="store_false", default=True,
                         help="EddyClicker обычно маленький набор — карты по умолчанию ВКЛЮЧЕНЫ, в отличие от Stage G")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    log_df = lib.read_combo_log()
    combos = lib.find_combos(
        log_df, years=args.years, eps=args.eps, size_filter=args.size_filter,
        extr_type=args.extr_type, maxgap=args.maxgap, mintime=args.mintime, prioritize=args.prioritize,
    )
    if not combos:
        print("Не найдено ни одной подходящей комбинации TE (ERA5) в grid_run_log.csv.")
        return

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

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sorted(file_map.items()), columns=["track_id", "source_file"]).to_csv(
        OUTPUT_ROOT / "eddyclicker_file_map.csv", index=False
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
    summary_csv = OUTPUT_ROOT / "summary_all_configurations.csv"

    # 2026-09-03: слияние (upsert) с уже существующей на диске таблицей, а не
    # перезапись поверх - см. подробный коммент и докстринг
    # lib.merge_summary_tables в 7_compare_TE_with_reference.py (та же
    # проблема и то же решение: узкий фильтр по этой сетке раньше стирал все
    # остальные, ранее посчитанные строки).
    if summary_csv.exists():
        old_summary_df = pd.read_csv(summary_csv)
        n_old = len(old_summary_df)
        summary_df = lib.merge_summary_tables(old_summary_df, summary_df)
        print(f"\nСуществующая таблица {summary_csv} ({n_old} строк) слита с {len(summaries)} "
              f"строками этого запуска -> {len(summary_df)} строк итого (старые строки, не "
              f"затронутые текущими фильтрами, сохранены).")

    # 2026-09-02: см. коммент в 7_compare_TE_with_reference.py - сортировка по
    # убыванию combined_score (100*POD + median_overlap_percent + median_overlap_percent_1to1).
    summary_df = lib.sort_summary_by_combined_score(summary_df)
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nСводная таблица (отсортирована по убыванию combined_score = "
          f"100*POD + median_overlap_percent + median_overlap_percent_1to1): {summary_csv}")
    print(summary_df.to_string(index=False))

    lib.plot_pod_bar(summary_df, OUTPUT_ROOT / "summary_pod_bar.png", title="Top POD — TE vs EddyClicker")
    if len(summary_df) > 1:
        lib.plot_pod_vs_overlap_scatter(summary_df, OUTPUT_ROOT / "summary_pod_vs_overlap.png", title="TE vs EddyClicker")
        # см. коммент в 7_compare_TE_with_reference.py - идея plot_configuration_summary из ERA5_plot_test.ipynb
        lib.plot_configuration_summary(summary_df, OUTPUT_ROOT / "summary_configuration_grid.png", title="TE vs EddyClicker")


if __name__ == "__main__":
    main()
