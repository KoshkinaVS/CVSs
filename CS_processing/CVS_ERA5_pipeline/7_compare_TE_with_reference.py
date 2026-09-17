"""
Stage G — валидация треков TempestExtremes (ERA5) против глобальных
референсных баз (IBTrACS, SyCLoPS), по сетке комбинаций из grid_run_log.csv
(6_grid_runner_TE_ERA5.py).

2026-08-30: заменяет прежний 7_compare_TC_TE_with_IBTrACS.py — тот умел
только IBTrACS; здесь тот же движок обобщён на --reference {ibtracs,syclops}
(идея и большая часть кода matching/статистики/графиков — по образцу
CS_processing/CVS_stat_tracks/tracks_comparison/comparison_utils.py +
run_statistics.py, которые уже делали ровно это обобщение для СТАРОЙ схемы
путей; здесь то же самое, но TE-треки берутся из grid_run_log.csv, а
SyCLoPS — из актуального SyCLoPS/tracks_types_csv/ вместо предполагавшейся
там конвертации в TempestExtremes-txt, которая, похоже, не поддерживается в
рабочем состоянии — см. докстринг tracks_comparison_lib.py).

Вся общая логика (чтение grid_run_log.csv, huracanpy.assess.match/pod,
трек-статистика, графики) — в tracks_comparison_lib.py, этот файл только
CLI + загрузка конкретного референса.

Пример запуска
--------------
# IBTrACS, все ok/pending_stage_e_prime комбинации за 2010 год, без карт
python 7_compare_TE_with_reference.py --reference ibtracs --years 2010

# SyCLoPS TC, North Atlantic, одна комбинация, с картами по каждому треку
python 7_compare_TE_with_reference.py --reference syclops --syclops-type TC \
    --years 2010 --eps 1 --size-filter 25 --extr-type global \
    --maxgap 3 --mintime 18 --no-prioritize --plot-tracks

# посмотреть, какие комбинации найдены, ничего не считать
python 7_compare_TE_with_reference.py --reference ibtracs --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import tracks_comparison_lib as lib


_ibtracs_na_cache = None  # huracanpy.load(source="ibtracs") один раз на весь прогон, не на комбинацию
_syclops_cache: dict = {}  # (type, region, years-tuple) -> dataset, тоже не перезагружаем на каждую комбинацию


def _get_ibtracs_na():
    global _ibtracs_na_cache
    if _ibtracs_na_cache is None:
        import huracanpy
        print("Загрузка IBTrACS (один раз на весь прогон)...")
        all_tracks = huracanpy.load(source="ibtracs")
        _ibtracs_na_cache = all_tracks.where(all_tracks.basin == "NA", drop=True)
    return _ibtracs_na_cache


def load_reference_for_combo(reference: str, combo: dict, syclops_type: str | None, syclops_region: str,
                              syclops_on_multiple: str):
    """Возвращает (dataset, reference_label, reference_id_column). IBTrACS/SyCLoPS
    кэшируются — комбинации в сетке обычно делят один и тот же набор лет, качать/
    парсить референс заново на каждый лист StitchNodes-параметров не нужно."""
    if reference == "ibtracs":
        na_tracks = _get_ibtracs_na()
        ref = na_tracks.where(na_tracks.time.dt.year.isin(combo["years"]), drop=True)
        return ref, "IBTrACS", "id_IBTrACS"
    else:
        cache_key = (syclops_type, syclops_region, tuple(combo["years"]))
        if cache_key not in _syclops_cache:
            _syclops_cache[cache_key] = lib.load_syclops_tracks(
                syclops_type, combo["years"], region=syclops_region, on_multiple=syclops_on_multiple
            )
        return _syclops_cache[cache_key], "SyCLoPS", "id_SyCLoPS"


def output_root(reference: str, syclops_type: str | None, syclops_region: str) -> Path:
    # СВОДКА по ВСЕЙ сетке сразу (summary_all_configurations.csv + графики топ-POD) -
    # у неё нет "своей" combo-папки, т.к. она про много комбинаций одновременно,
    # поэтому остаётся на уровне sigma_dir (lib.SIGMA_DIR - та же папка, где лежат
    # все combo-папки eps-ms-sf_extr_type). Детальные результаты ОДНОЙ комбинации
    # (matches_raw.csv, matching_statistics.csv, карты треков) - см. run_combo() -
    # лежат ВНУТРИ её собственной combo-папки (lib.combo_output_dir), не здесь.
    if reference == "ibtracs":
        return lib.SIGMA_DIR / "TC_comparison_huracanpy"
    return lib.SIGMA_DIR / "SyCLoPS_comparison_huracanpy" / f"{syclops_type}_{syclops_region}"


def reference_subpath(reference: str, syclops_type: str | None, syclops_region: str) -> Path:
    """Подпуть внутри data_comparison_huracanpy/ КОНКРЕТНОЙ combo-папки (см.
    lib.combo_output_dir) - что именно сравнивали. Имя папки здесь намеренно
    короче/без суффикса '_comparison_huracanpy' (он уже есть в 'data_comparison_
    huracanpy' на уровень выше) - в отличие от output_root() выше, которая
    называет папки СВОДКИ на уровне sigma_dir."""
    if reference == "ibtracs":
        return Path("IBTrACS")
    return Path("SyCLoPS") / f"{syclops_type}_{syclops_region}"


def _plot_combo(ref, te_tracks, matches, stats_df, reference_label, reference_id_column, combo_dir) -> None:
    """Собственно отрисовка карт для ОДНОЙ уже посчитанной комбинации - вынесено
    из run_combo() в отдельную функцию, чтобы её можно было вызвать и без
    полного пересчёта matching (см. plot_combo_tracks_from_cache() ниже и
    plot_top_configs.py - "сначала вся статистика по сетке без картинок,
    потом отдельно точечная прорисовка N лучших по combined_score")."""
    lib.plot_all_reference_tracks(ref, te_tracks, combo_dir / "tracks", reference_label, reference_id_column, matches)
    # 2026-09-02: отдельные "1-к-1" картинки (референс + РОВНО один, наиболее
    # долго совпадающий TE-трек - см. докстринг plot_best_single_match_tracks) -
    # своя подпапка, не смешивается с картами выше (которые могут показывать
    # НЕСКОЛЬКО совпавших TE-треков на одного референса).
    lib.plot_best_single_match_tracks(ref, te_tracks, stats_df, reference_label, combo_dir / "tracks_1to1_best")


def plot_combo_tracks_from_cache(combo: dict, reference: str, syclops_type, syclops_region, syclops_on_multiple,
                                  combo_dir: Path, matches_csv: Path, stats_df: pd.DataFrame, label: str) -> bool:
    """2026-09-02: рисует tracks/ + tracks_1to1_best/ для комбинации, matching
    которой УЖЕ посчитан раньше (matches_raw.csv/matching_statistics.csv на
    диске) - БЕЗ повторного huracanpy.assess.match (это самая дорогая часть
    run_combo()). matches_raw.csv/matching_statistics.csv хранят только id/
    статистику, не геометрию (lon/lat) - поэтому ref/te_tracks всё равно
    грузятся заново (это дёшево по сравнению с matching), просто не
    пересчитывается сам матчинг.

    Используется из run_combo() (при повторном --plot-tracks на уже готовые
    комбинации - раньше это молча ничего не рисовало) и из отдельного
    plot_top_configs.py (по запросу - "сначала вся сводная статистика по
    сетке, потом отдельным скриптом для N лучших по combined_score - только
    картинки, без пересчёта всей сетки").

    Возвращает True, если карты нарисованы, False - если matches_csv не
    найден (комбинация ещё не посчитана вообще, тут делать нечего)."""
    if not matches_csv.exists():
        print(f"  [{label}] {matches_csv} не найден - matching для этой комбинации ещё не считался, "
              f"нечего рисовать (сначала обычный запуск без --force, чтобы посчитать matching).")
        return False

    matches = pd.read_csv(matches_csv)
    ref, reference_label, reference_id_column = load_reference_for_combo(
        reference, combo, syclops_type, syclops_region, syclops_on_multiple
    )
    te_tracks = lib.load_te_tracks_for_combo(combo)
    _plot_combo(ref, te_tracks, matches, stats_df, reference_label, reference_id_column, combo_dir)
    return True


def run_combo(combo: dict, reference: str, syclops_type, syclops_region, syclops_on_multiple,
              plot_tracks: bool, force: bool) -> dict:
    label = lib.combo_label(combo)
    combo_dir = lib.combo_output_dir(combo, reference_subpath(reference, syclops_type, syclops_region))
    matches_csv = combo_dir / "matches_raw.csv"
    stats_csv = combo_dir / "matching_statistics.csv"

    if not force and stats_csv.exists():
        print(f"[{label}] {stats_csv} уже есть, пропуск пересчёта статистики (--force для пересчёта)")
        stats_df = pd.read_csv(stats_csv)
        if plot_tracks:
            # 2026-09-02: раньше --plot-tracks на уже посчитанной (кэшированной)
            # комбинации молча НЕ рисовал ничего - ранний return срабатывал ДО
            # блока с картами. Теперь карты рисуются и здесь тоже (matching не
            # пересчитывается, только ref/te_tracks грузятся заново - см.
            # plot_combo_tracks_from_cache).
            plot_combo_tracks_from_cache(
                combo, reference, syclops_type, syclops_region, syclops_on_multiple,
                combo_dir, matches_csv, stats_df, label,
            )
        return _summarize_row(combo, stats_df, n_te_tracks=None, n_reference=None, pod=None)

    combo_dir.mkdir(parents=True, exist_ok=True)

    ref, reference_label, reference_id_column = load_reference_for_combo(
        reference, combo, syclops_type, syclops_region, syclops_on_multiple
    )
    te_tracks = lib.load_te_tracks_for_combo(combo)

    import numpy as np
    n_reference = int(ref.track_id.hrcn.nunique()) if hasattr(ref, "track_id") else len(np.unique(ref.track_id.values))
    n_te = int(te_tracks.track_id.hrcn.nunique())
    print(f"[{label}] {reference_label}: {n_reference} треков, TE: {n_te} треков ({combo['years']})")

    matches = lib.calculate_matches(ref, te_tracks, reference_label)
    matches.to_csv(matches_csv, index=False)

    pod = lib.calculate_pod(matches, ref, reference_label)
    stats_df = lib.calculate_track_statistics(ref, te_tracks, matches, reference_id_column)
    stats_df.to_csv(stats_csv, index=False)

    print(f"[{label}] POD={pod:.3f}, matched={int(stats_df.detected.sum())}/{len(stats_df)}")

    if plot_tracks:
        _plot_combo(ref, te_tracks, matches, stats_df, reference_label, reference_id_column, combo_dir)

    return _summarize_row(combo, stats_df, n_te_tracks=n_te, n_reference=n_reference, pod=pod)


def _summarize_row(combo, stats_df, n_te_tracks, n_reference, pod) -> dict:
    """2026-09-02: n_te_tracks теперь тоже получает явный fallback (был только
    у pod/n_reference) - при повторном запуске без --force, когда
    matching_statistics.csv для комбинации уже есть (см. run_combo(), ветка
    "уже пропуск"), te_tracks заново не грузятся (в этом и смысл skip-check -
    не тратить время на huracanpy.load), поэтому точное число TE-треков
    неизвестно без перезагрузки. Раньше здесь молча оставался python None,
    который потом попадал в summary_all_configurations.csv как n_TE_tracks=None
    и валил plot_configuration_summary() на panel (c): matplotlib bar()
    нормально рисует NaN-высоту (просто пропускает бар), но не умеет
    складывать int+None (TypeError на combo, чей результат был взят из кэша).
    float('nan') - корректное "неизвестно" для числового столбца, в отличие
    от None."""
    summary = lib.summarize_track_statistics(
        stats_df, pod if pod is not None else float(stats_df.detected.mean()) if len(stats_df) else float("nan"),
        n_reference if n_reference is not None else len(stats_df),
        n_te_tracks if n_te_tracks is not None else float("nan"),
    )
    return {**{k: combo[k] for k in lib.COMBO_DIMS}, "years": ",".join(str(y) for y in combo["years"]), **summary}


def main():
    parser = argparse.ArgumentParser(
        description="Stage G: TE (ERA5) vs IBTrACS/SyCLoPS, по комбинациям из grid_run_log.csv"
    )
    parser.add_argument("--reference", required=True, choices=["ibtracs", "syclops"])
    parser.add_argument("--syclops-type", choices=lib.SYCLOPS_TYPES, default=None,
                         help="Обязателен при --reference syclops")
    parser.add_argument("--syclops-region", default="NA")
    parser.add_argument("--syclops-on-multiple", choices=["newest", "all"], default="newest",
                         help="Если для (type,region,year) в tracks_types_csv найдено несколько файлов "
                              "(разные запуски filter_type_for_year_and_region*.py) — взять самый свежий "
                              "(по умолчанию) или объединить все.")
    parser.add_argument("--years", type=int, nargs="+", default=None)
    parser.add_argument("--eps", type=int, default=None)
    parser.add_argument("--size-filter", type=int, default=None)
    parser.add_argument("--extr-type", choices=["global", "local", "geom"], default=None)
    parser.add_argument("--maxgap", type=int, default=None)
    parser.add_argument("--mintime", type=int, default=None)
    pr = parser.add_mutually_exclusive_group()
    pr.add_argument("--prioritize", dest="prioritize", action="store_true", default=None)
    pr.add_argument("--no-prioritize", dest="prioritize", action="store_false")
    parser.add_argument("--plot-tracks", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.reference == "syclops" and args.syclops_type is None:
        parser.error("--reference syclops требует --syclops-type {TC,SS,PL}")

    log_df = lib.read_combo_log()
    combos = lib.find_combos(
        log_df, years=args.years, eps=args.eps, size_filter=args.size_filter,
        extr_type=args.extr_type, maxgap=args.maxgap, mintime=args.mintime, prioritize=args.prioritize,
    )
    if not combos:
        print("Не найдено ни одной подходящей комбинации в grid_run_log.csv.")
        return

    print(f"Найдено {len(combos)} комбинаций (после фильтров) в {lib.COMBO_LOG_PATH}:")
    for combo in combos:
        print(f"  {lib.combo_label(combo)} | years={combo['years']}")

    if args.dry_run:
        print("--dry-run: выполнение пропущено")
        return

    plot_tracks_effective = args.plot_tracks
    if args.plot_tracks:
        # 2026-08-31: раньше --plot-tracks работал только для РОВНО ОДНОЙ комбинации
        # (иначе комбинации x референсные треки грозили тысячами PNG). Теперь защита
        # - на уровне одного референсного трека (см. plot_all_reference_tracks в
        # tracks_comparison_lib.py: карта пропускается, если у референсного трека
        # >= 10 совпавших TE-треков - вырожденный случай), поэтому здесь можно
        # рисовать сразу по всем найденным комбинациям без искусственного лимита.
        print(f"\n--plot-tracks: карты будут построены по всем {len(combos)} найденным комбинациям "
              f"(по одной карте на референсный трек - все совпавшие ERA5-треки на ней; референсные "
              f"треки с >= 10 совпадениями пропускаются автоматически). Если комбинаций много и это "
              f"даёт слишком много PNG - сузьте фильтры (--eps/--size-filter/--extr-type/--maxgap/"
              f"--mintime/--prioritize).")

    out_root = output_root(args.reference, args.syclops_type, args.syclops_region)  # только для сводки, см. output_root()

    summaries = []
    for combo in combos:
        try:
            summaries.append(run_combo(
                combo, args.reference, args.syclops_type, args.syclops_region, args.syclops_on_multiple,
                plot_tracks_effective, args.force,
            ))
        except Exception as e:
            print(f"ОШИБКА для {lib.combo_label(combo)}: {e}")
            continue

    summary_df = pd.DataFrame(summaries)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_csv = out_root / "summary_all_configurations.csv"

    # 2026-09-03: слияние с уже существующей на диске таблицей (upsert), а не
    # перезапись поверх - раньше summary_df.to_csv() писался БЕЗУСЛОВНО поверх
    # summary_csv, что при запуске с узкими фильтрами (например, одна
    # конкретная конфигурация только чтобы перерисовать её карты) стирало все
    # остальные, ранее посчитанные строки на полной сетке. См. докстринг
    # lib.merge_summary_tables - строки вне текущего запроса сохраняются,
    # строки с тем же ключом комбинации разрешаются по полноте данных
    # (не просто "новое поверх старого"), чтобы точечный --plot-tracks
    # перезапуск без --force не затирал ранее полностью посчитанную строку
    # NaN-версией.
    if summary_csv.exists():
        old_summary_df = pd.read_csv(summary_csv)
        n_old = len(old_summary_df)
        summary_df = lib.merge_summary_tables(old_summary_df, summary_df)
        print(f"\nСуществующая таблица {summary_csv} ({n_old} строк) слита с {len(summaries)} "
              f"строками этого запуска -> {len(summary_df)} строк итого (старые строки, не "
              f"затронутые текущими фильтрами, сохранены).")

    # 2026-09-02: сортировка итоговой таблицы по убыванию combined_score
    # (POD*100 + median_overlap_percent + median_overlap_percent_1to1) - см.
    # докстринг lib.sort_summary_by_combined_score. Строки без combined_score
    # (например, --dry-run не запускался, а комбинация целиком провалилась
    # исключением в run_combo() - тогда её вообще нет в summaries) сюда не
    # попадают; NaN только у комбинаций, взятых из кэша без --force.
    summary_df = lib.sort_summary_by_combined_score(summary_df)
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nСводная таблица (отсортирована по убыванию combined_score = "
          f"100*POD + median_overlap_percent + median_overlap_percent_1to1): {summary_csv}")
    print(summary_df.to_string(index=False))

    title = f"TE vs {args.reference.upper()}" + (f" ({args.syclops_type}/{args.syclops_region})" if args.reference == "syclops" else "")
    lib.plot_pod_bar(summary_df, out_root / "summary_pod_bar.png", title=f"Top POD — {title}")
    if len(summary_df) > 1:
        lib.plot_pod_vs_overlap_scatter(summary_df, out_root / "summary_pod_vs_overlap.png", title=title)
        # 2026-08-31: сводка по всей сетке (eps/size_filter/extr_type) на одном
        # рисунке - идея plot_configuration_summary() из ERA5_plot_test.ipynb,
        # адаптирована под нашу схему summary_all_configurations.csv (см.
        # докстринг lib.plot_configuration_summary).
        lib.plot_configuration_summary(summary_df, out_root / "summary_configuration_grid.png", title=title)


if __name__ == "__main__":
    main()
