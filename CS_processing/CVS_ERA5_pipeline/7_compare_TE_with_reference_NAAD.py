"""
Stage G (NAAD) — валидация треков TempestExtremes (NAAD, LoRes/HiRes) против
глобальных референсных баз (IBTrACS, SyCLoPS), по сетке комбинаций из
grid_run_log_{region}.csv (6_grid_runner_TE_NAAD.py). Аналог
7_compare_TE_with_reference.py (ERA5), адаптирован под то, что у NAAD
DATA_TYPE — параметр запуска (LoRes/HiRes), а не константа файла — см.
tracks_comparison_lib_NAAD.py::configure().

Вся общая логика (matching/pod/статистика/графики) — в
tracks_comparison_lib.py (переиспользуется без изменений через
tracks_comparison_lib_NAAD.py), этот файл — только CLI + загрузка референса,
как и у ERA5-версии.

IBTrACS и North Atlantic
-------------------------
IBTrACS — глобальная база ТОЛЬКО тропических/субтропических циклонов. Как и
ERA5-версия, здесь референс по умолчанию отфильтрован по бассейну NA (North
Atlantic) — `--ibtracs-basin` (по умолчанию "NA"), т.к. основной сценарий
использования (см. postfix региона "NA_for_TC..." у ERA5) — валидация
трекинга ИМЕННО тропических циклонов как методическая проверка перед
применением к полярным мезоциклонам (которые в IBTrACS не входят вовсе).
Если домен NAAD шире North Atlantic (например, захватывает часть Arctic) и
нужно другое подмножество IBTrACS - передайте `--ibtracs-basin` (коды
IBTrACS: NA, EP, WP, NI, SI, SP, SA) или несколько через пробел, либо "ALL"
для всего земного шара без фильтра по бассейну.

Пример запуска
--------------
# IBTrACS, все tracking_ok-комбинации LoRes за 2010 год, без карт
python 7_compare_TE_with_reference_NAAD.py --data-type LoRes --reference ibtracs --years 2010

# то же для HiRes, с картами по каждому треку
python 7_compare_TE_with_reference_NAAD.py --data-type HiRes --reference ibtracs --years 2010 --plot-tracks

# посмотреть, какие комбинации найдены, ничего не считать
python 7_compare_TE_with_reference_NAAD.py --data-type LoRes --reference ibtracs --dry-run

2026-09-05: НЕ прогонялось на реальных данных (нет доступа к серверу из этой
сессии) - логика 1:1 скопирована с уже рабочей ERA5-версии, изменено только
то, что описано в докстринге tracks_comparison_lib_NAAD.py.
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


_ibtracs_cache: dict = {}  # basin_key (tuple отсортированных кодов или None) -> dataset


def _get_ibtracs(basins: list[str] | None):
    """basins=None -> весь земной шар, без фильтра. См. докстринг модуля про
    --ibtracs-basin."""
    key = tuple(sorted(basins)) if basins else None
    if key not in _ibtracs_cache:
        import huracanpy
        print(f"Загрузка IBTrACS (один раз на весь прогон, basins={basins or 'ALL'})...")
        all_tracks = huracanpy.load(source="ibtracs")
        if basins:
            all_tracks = all_tracks.where(all_tracks.basin.isin(basins), drop=True)
        _ibtracs_cache[key] = all_tracks
    return _ibtracs_cache[key]


def load_reference_for_combo(reference: str, combo: dict, ibtracs_basins: list[str] | None,
                              syclops_type: str | None, syclops_region: str, syclops_on_multiple: str,
                              _syclops_cache: dict):
    """Возвращает (dataset, reference_label, reference_id_column). Кэшируются —
    комбинации в сетке обычно делят один и тот же набор лет."""
    if reference == "ibtracs":
        tracks = _get_ibtracs(ibtracs_basins)
        ref = tracks.where(tracks.time.dt.year.isin(combo["years"]), drop=True)
        return ref, "IBTrACS", "id_IBTrACS"
    else:
        cache_key = (syclops_type, syclops_region, tuple(combo["years"]))
        if cache_key not in _syclops_cache:
            _syclops_cache[cache_key] = lib.load_syclops_tracks(
                syclops_type, combo["years"], region=syclops_region, on_multiple=syclops_on_multiple
            )
        return _syclops_cache[cache_key], "SyCLoPS", "id_SyCLoPS"


def output_root(reference: str, syclops_type: str | None, syclops_region: str) -> Path:
    if reference == "ibtracs":
        return lib.SIGMA_DIR / "TC_comparison_huracanpy"
    return lib.SIGMA_DIR / "SyCLoPS_comparison_huracanpy" / f"{syclops_type}_{syclops_region}"


def reference_subpath(reference: str, syclops_type: str | None, syclops_region: str) -> Path:
    if reference == "ibtracs":
        return Path("IBTrACS")
    return Path("SyCLoPS") / f"{syclops_type}_{syclops_region}"


def _plot_combo(ref, te_tracks, matches, stats_df, reference_label, reference_id_column, combo_dir) -> None:
    lib.plot_all_reference_tracks(ref, te_tracks, combo_dir / "tracks", reference_label, reference_id_column, matches)
    lib.plot_best_single_match_tracks(ref, te_tracks, stats_df, reference_label, combo_dir / "tracks_1to1_best")


def plot_combo_tracks_from_cache(combo: dict, reference: str, ibtracs_basins, syclops_type, syclops_region,
                                  syclops_on_multiple, syclops_cache, combo_dir: Path, matches_csv: Path,
                                  stats_df: pd.DataFrame, label: str) -> bool:
    if not matches_csv.exists():
        print(f"  [{label}] {matches_csv} не найден - matching для этой комбинации ещё не считался, "
              f"нечего рисовать (сначала обычный запуск без --force, чтобы посчитать matching).")
        return False

    matches = pd.read_csv(matches_csv)
    ref, reference_label, reference_id_column = load_reference_for_combo(
        reference, combo, ibtracs_basins, syclops_type, syclops_region, syclops_on_multiple, syclops_cache
    )
    te_tracks = lib.load_te_tracks_for_combo(combo)
    _plot_combo(ref, te_tracks, matches, stats_df, reference_label, reference_id_column, combo_dir)
    return True


def run_combo(combo: dict, reference: str, ibtracs_basins, syclops_type, syclops_region, syclops_on_multiple,
              syclops_cache, plot_tracks: bool, force: bool) -> dict:
    label = lib.combo_label(combo)
    combo_dir = lib.combo_output_dir(combo, reference_subpath(reference, syclops_type, syclops_region))
    matches_csv = combo_dir / "matches_raw.csv"
    stats_csv = combo_dir / "matching_statistics.csv"

    if not force and stats_csv.exists():
        print(f"[{label}] {stats_csv} уже есть, пропуск пересчёта статистики (--force для пересчёта)")
        stats_df = pd.read_csv(stats_csv)
        if plot_tracks:
            plot_combo_tracks_from_cache(
                combo, reference, ibtracs_basins, syclops_type, syclops_region, syclops_on_multiple,
                syclops_cache, combo_dir, matches_csv, stats_df, label,
            )
        return _summarize_row(combo, stats_df, n_te_tracks=None, n_reference=None, pod=None)

    combo_dir.mkdir(parents=True, exist_ok=True)

    ref, reference_label, reference_id_column = load_reference_for_combo(
        reference, combo, ibtracs_basins, syclops_type, syclops_region, syclops_on_multiple, syclops_cache
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
    summary = lib.summarize_track_statistics(
        stats_df, pod if pod is not None else float(stats_df.detected.mean()) if len(stats_df) else float("nan"),
        n_reference if n_reference is not None else len(stats_df),
        n_te_tracks if n_te_tracks is not None else float("nan"),
    )
    return {**{k: combo[k] for k in lib.COMBO_DIMS}, "years": ",".join(str(y) for y in combo["years"]), **summary}


def main():
    parser = argparse.ArgumentParser(
        description="Stage G (NAAD): TE (LoRes/HiRes) vs IBTrACS/SyCLoPS, по комбинациям из grid_run_log_{region}.csv"
    )
    parser.add_argument("--data-type", choices=["LoRes", "HiRes"], required=True)
    parser.add_argument("--region", default=cfg.DEFAULT_REGION_NAME,
                         help=f"Как в 6_grid_runner_TE_NAAD.py (по умолчанию '{cfg.DEFAULT_REGION_NAME}')")
    parser.add_argument("--sigma", type=int, default=cfg.DEFAULT_SIGMA)
    parser.add_argument("--reference", required=True, choices=["ibtracs", "syclops"])
    parser.add_argument("--ibtracs-basin", nargs="+", default=["NA"],
                         help="Коды бассейна IBTrACS (NA/EP/WP/NI/SI/SP/SA), несколько через пробел, "
                              "или 'ALL' для отключения фильтра по бассейну. По умолчанию NA - см. "
                              "докстринг модуля про то, почему это методическая TC-валидация, не про "
                              "полярные мезоциклоны.")
    parser.add_argument("--syclops-type", choices=lib.SYCLOPS_TYPES, default=None,
                         help="Обязателен при --reference syclops")
    parser.add_argument("--syclops-region", default="NA")
    parser.add_argument("--syclops-on-multiple", choices=["newest", "all"], default="newest")
    parser.add_argument("--years", type=int, nargs="+", default=None)
    parser.add_argument("--eps", type=int, default=None)
    parser.add_argument("--size-filter", type=int, default=None)
    parser.add_argument("--extr-type", choices=["global", "local"], default=None,
                         help="Без 'geom' - для NAAD такой Stage A не считался")
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

    lib.configure(data_type=args.data_type, region_name=args.region, sigma=args.sigma)

    ibtracs_basins = None if (len(args.ibtracs_basin) == 1 and args.ibtracs_basin[0].upper() == "ALL") else args.ibtracs_basin

    log_df = lib.read_combo_log()
    combos = lib.find_combos(
        log_df, years=args.years, eps=args.eps, size_filter=args.size_filter,
        extr_type=args.extr_type, maxgap=args.maxgap, mintime=args.mintime, prioritize=args.prioritize,
    )
    if not combos:
        print(f"Не найдено ни одной подходящей комбинации в {lib.COMBO_LOG_PATH}.")
        return

    print(f"data_type={args.data_type} регион={args.region}\n"
          f"Найдено {len(combos)} комбинаций (после фильтров) в {lib.COMBO_LOG_PATH}:")
    for combo in combos:
        print(f"  {lib.combo_label(combo)} | years={combo['years']}")

    if args.dry_run:
        print("--dry-run: выполнение пропущено")
        return

    plot_tracks_effective = args.plot_tracks
    if args.plot_tracks:
        print(f"\n--plot-tracks: карты будут построены по всем {len(combos)} найденным комбинациям "
              f"(по одной карте на референсный трек - все совпавшие TE-треки на ней; референсные "
              f"треки с >= 10 совпадениями пропускаются автоматически). Если комбинаций много и это "
              f"даёт слишком много PNG - сузьте фильтры (--eps/--size-filter/--extr-type/--maxgap/"
              f"--mintime/--prioritize).")

    out_root = output_root(args.reference, args.syclops_type, args.syclops_region)

    syclops_cache: dict = {}
    summaries = []
    for combo in combos:
        try:
            summaries.append(run_combo(
                combo, args.reference, ibtracs_basins, args.syclops_type, args.syclops_region,
                args.syclops_on_multiple, syclops_cache, plot_tracks_effective, args.force,
            ))
        except Exception as e:
            print(f"ОШИБКА для {lib.combo_label(combo)}: {e}")
            continue

    summary_df = pd.DataFrame(summaries)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_csv = out_root / "summary_all_configurations.csv"

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

    title = (f"NAAD {args.data_type} vs {args.reference.upper()}"
             + (f" ({args.syclops_type}/{args.syclops_region})" if args.reference == "syclops" else ""))
    lib.plot_pod_bar(summary_df, out_root / "summary_pod_bar.png", title=f"Top POD — {title}")
    if len(summary_df) > 1:
        lib.plot_pod_vs_overlap_scatter(summary_df, out_root / "summary_pod_vs_overlap.png", title=title)
        lib.plot_configuration_summary(summary_df, out_root / "summary_configuration_grid.png", title=title)


if __name__ == "__main__":
    main()
