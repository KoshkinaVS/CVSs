#!/usr/bin/env python3
"""
Сравнение треков КВС этого пайплайна (DBSCAN_tracking_ERA5_latlon.py,
lat/lon-адаптация) с внешними референсными базами треков тропических
циклонов — IBTrACS и SyCLoPS (TC/SS/PL) — через huracanpy.assess.match/pod.

По аналогии с ../CVS_alt_tracking/TempestExtremes/7_compare_TE_with_reference.py
(тот же движок matching/POD/графиков, --reference {ibtracs,syclops}), но:
  - вместо сетки StitchNodes-комбинаций из grid_run_log.csv (там это была
    сетка гиперпараметров TempestExtremes) здесь перебираются все реально
    посчитанные конфигурации этого пайплайна — пары (tracking_type,
    CVS_speed) из step_of_tracking.TRACKING_METHODS x
    plot_tracking_results.SPEED_CHOICES, для которых на диске под
    --data-dir реально есть треки за запрошенный период (см. find_configs
    ниже) — а не только "штатный" набор TRACKING_TYPES/SPEED_OPTIONS из
    DBSCAN_tracking_ERA5_latlon.py (те же 3x2, что реально запускаются по
    умолчанию, но раньше могли гоняться и другие комбинации вручную);
  - вся source-specific/generic логика (huracanpy-мэтчинг, POD,
    статистика по трекам, карты, IBTrACS/SyCLoPS-загрузчики) — в
    compare_reference_lib.py рядом (аналог tracks_comparison_lib.py), этот
    файл — только CLI + перебор конфигураций + сводная таблица.

Пример запуска
--------------
# IBTrACS, все посчитанные конфигурации за август-сентябрь 2010, без карт
python3 compare_with_reference.py --reference ibtracs --sigma 0 \\
    --start 2010-08 --end 2010-09 \\
    --data-dir /storage/thalassa/users/vkoshkina/data

# SyCLoPS TC, North Atlantic, с картами по каждому референсному треку
python3 compare_with_reference.py --reference syclops --syclops-type TC \\
    --sigma 0 --start 2010-08 --end 2010-09 --plot-tracks \\
    --data-dir /storage/thalassa/users/vkoshkina/data

# посмотреть, какие конфигурации реально нашлись на диске, ничего не считать
python3 compare_with_reference.py --reference ibtracs --sigma 0 \\
    --start 2010-08 --end 2010-09 --dry-run \\
    --data-dir /storage/thalassa/users/vkoshkina/data

ВАЖНО (честно, как и в compare_reference_lib.py): в этой песочнице нет
доступа ни к huracanpy/cartopy, ни к серверу, ни к вашим реальным трекам —
поэтому весь код, реально вызывающий huracanpy (загрузка IBTrACS/SyCLoPS,
matching/POD, графики), портирован из уже работавшего
7_compare_TE_with_reference.py практически без изменений по сути (тот же
API: huracanpy.load(source="ibtracs"), .where(...), .track_id.hrcn.nunique()),
но здесь не перезапускался. Прогоните --dry-run (не требует huracanpy —
только сканирует диск), затем один короткий период, перед боевым прогоном
на всю историю. Путь SyCLoPS по умолчанию (--syclops-root) — это ДОГАДКА по
аналогии с расположением TempestExtremes/../SyCLoPS у вас на диске
(<--data-dir>/../SyCLoPS/tracks_types_csv) — если это не так, передайте
свой путь явно.

Что сознательно не перенесено из 7_compare_TE_with_reference.py:
  - lib.plot_configuration_summary() (сводная картинка по сетке eps x
    size_filter x extr_type) — там она осмысленна для 3-мерной сетки
    гиперпараметров StitchNodes; здесь "конфигурация" всего двумерная
    (tracking_type x CVS_speed), и summary_pod_vs_overlap.png (цвет —
    tracking_type, маркер — speed) уже показывает то же самое без
    отдельной функции;
  - --eps/--size-filter/--extr-type/--maxgap/--mintime/--prioritize и
    grid_run_log.csv — это фильтры и источник TE-специфичной сетки
    комбинаций, здесь их нет, см. find_configs().
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from step_of_tracking import TRACKING_METHODS, circ as CIRC_DEFAULT  # noqa: E402
from DBSCAN_tracking_ERA5_latlon import DATA_TYPE, PREF_TRACKING  # noqa: E402
from plot_tracking_results import SPEED_CHOICES  # noqa: E402

import compare_reference_lib as lib  # noqa: E402

OWN_NAME = lib.OWN_NAME

# huracanpy.load(source="ibtracs") — один раз на весь прогон, не на конфигурацию.
_ibtracs_all_cache = None
# (syclops_type, syclops_region, years-tuple, on_multiple) -> huracanpy Dataset.
_syclops_cache: dict = {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Сравнение треков КВС (ERA5, lat/lon) с IBTrACS/SyCLoPS по всем "
                     "найденным на диске конфигурациям (tracking_type x CVS_speed)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--reference', required=True, choices=['ibtracs', 'syclops'],
                         help="с какой референсной базой сравнивать")
    parser.add_argument('--syclops-type', choices=lib.SYCLOPS_TYPES, default=None,
                         help="обязателен при --reference syclops")
    parser.add_argument('--syclops-region', default='NA')
    parser.add_argument('--syclops-root', default=None,
                         help="папка с SyCLoPS_{type}_{region}_*_{year}_tracks.csv; "
                              "по умолчанию — <--data-dir>/../SyCLoPS/tracks_types_csv "
                              "(догадка по аналогии с TempestExtremes, см. докстринг файла)")
    parser.add_argument('--syclops-on-multiple', choices=['newest', 'all'], default='newest',
                         help="если для (type,region,year) нашлось несколько файлов — взять "
                              "самый свежий по mtime (по умолчанию) или объединить все")
    parser.add_argument('--ibtracs-basin', default='NA',
                         help="фильтр IBTrACS по бассейну (колонка 'basin'); пустая строка "
                              "('') — без фильтра, весь глобальный IBTrACS")
    parser.add_argument('--ibtracs-online', action='store_true',
                         help="тянуть IBTrACS по сети с сервера NCEI-NOAA (huracanpy "
                              "ibtracs_online=True) вместо встроенной в huracanpy офлайн-копии "
                              "(по умолчанию, без сети вообще, ibtracs_subset='wmo'). Нужен "
                              "доступ в интернет с той ноды/сессии, где запущен этот скрипт — на "
                              "compute-нодах кластеров его обычно нет, и тогда загрузка IBTrACS "
                              "зависает без ошибки (см. README, раздел про зависание на "
                              "«Загрузка IBTrACS»)")
    parser.add_argument('--ibtracs-subset', default=None,
                         help="huracanpy ibtracs_subset; офлайн по умолчанию 'wmo' (официальные "
                              "WMO-треки), также доступен 'usa'; при --ibtracs-online — свои "
                              "варианты (например 'last3years', 'SI'), см. документацию huracanpy")
    parser.add_argument('--sigma', type=int, required=True,
                         help="параметр сглаживания DBSCAN, как при запуске трекинга")
    parser.add_argument('--start', required=True, metavar='YYYY-MM',
                         help="первый месяц периода, включительно")
    parser.add_argument('--end', required=True, metavar='YYYY-MM',
                         help="последний месяц периода, включительно")
    parser.add_argument('--data-dir', default=os.environ.get('ERA5_TRACKS_DATA_DIR'),
                         help="корень с данными, как при запуске трекинга; можно также "
                              "задать переменной окружения ERA5_TRACKS_DATA_DIR")
    parser.add_argument('--circ', default=CIRC_DEFAULT, choices=['C', 'AC'],
                         help="циклонические (C) или антициклонические (AC) КВС")
    parser.add_argument('--plot-tracks', action='store_true',
                         help="рисовать карту по каждому референсному треку (все "
                              "совпавшие свои треки на ней) для каждой конфигурации")
    parser.add_argument('--force', action='store_true',
                         help="пересчитать конфигурации, для которых уже есть "
                              "matching_statistics.csv (по умолчанию — пропускаются)")
    parser.add_argument('--dry-run', action='store_true',
                         help="только показать найденные на диске конфигурации, ничего "
                              "не считать (huracanpy при этом не импортируется вовсе)")
    args = parser.parse_args()

    if args.reference == 'syclops' and args.syclops_type is None:
        parser.error("--reference syclops требует --syclops-type {TC,SS,PL}")

    if not args.data_dir:
        parser.error("не указан путь к данным: передайте --data-dir или переменную "
                      "окружения ERA5_TRACKS_DATA_DIR")

    try:
        args.start_period = pd.Period(args.start, freq='M')
        args.end_period = pd.Period(args.end, freq='M')
    except ValueError as exc:
        parser.error(f"--start/--end должны быть в формате YYYY-MM: {exc}")

    if args.end_period < args.start_period:
        parser.error("--end раньше --start")

    return args


def _has_tracks_for_months(path_tracks_dir, months) -> bool:
    """Есть ли хоть один CSV-трек (save_track_csv) в любом из месяцев `months`."""
    for period in months:
        month_dir = Path(path_tracks_dir) / f"{period.year}-{period.month:02d}"
        if month_dir.is_dir() and next(month_dir.glob('*_track_*.csv'), None) is not None:
            return True
    return False


def find_configs(args, months, results_root):
    """Все (tracking_type, CVS_speed) из TRACKING_METHODS x SPEED_CHOICES, у
    которых на диске реально есть папка с треками за период `months` — а не
    только "штатный" набор TRACKING_TYPES/SPEED_OPTIONS из
    DBSCAN_tracking_ERA5_latlon.py (см. докстринг файла)."""
    configs = []
    for tracking_type in sorted(TRACKING_METHODS):
        for speed in SPEED_CHOICES:
            results_dir = f"{tracking_type}_{speed}_{PREF_TRACKING}"
            path_tracks_dir = f"{results_root}/{results_dir}/tracks_{args.circ}/"
            if not os.path.isdir(path_tracks_dir):
                continue
            if not _has_tracks_for_months(path_tracks_dir, months):
                continue
            configs.append({'type': tracking_type, 'speed': speed, 'path_tracks_dir': path_tracks_dir})
    return configs


def reference_subpath(args) -> Path:
    if args.reference == 'ibtracs':
        return Path('IBTrACS')
    return Path('SyCLoPS') / f"{args.syclops_type}_{args.syclops_region}"


def _get_ibtracs_all(args):
    """huracanpy.load(source="ibtracs") — один раз на весь прогон, не на конфигурацию.

    По умолчанию (--ibtracs-online не передан) явно пин им ibtracs_online=False:
    huracanpy тогда берёт лёгкую копию IBTrACS, встроенную в сам пакет (WMO-треки
    1980..последний год без provisional, обновлялась в huracanpy периодически) —
    БЕЗ сети вообще. Раньше здесь был голый huracanpy.load(source="ibtracs") без
    ibtracs_online — по документации huracanpy это тоже офлайн по умолчанию, но
    зависело от установленной версии huracanpy, и при --reference ibtracs на
    HPC-кластере (сообщение пользователя: скрипт «висит» сразу после «Загрузка
    IBTrACS...») это главный подозреваемый: если бы version по умолчанию
    оказался online, процесс попытался бы стучаться в www.ncei.noaa.gov, а на
    compute-нодах кластеров обычно нет интернета — TCP просто зависает без
    ошибки (никакого таймаута/сообщения), а не падает сразу. Явный
    ibtracs_online=False убирает эту неопределённость независимо от версии.
    """
    global _ibtracs_all_cache
    if _ibtracs_all_cache is None:
        import huracanpy
        if args.ibtracs_online:
            subset = args.ibtracs_subset
            print(f"Загрузка IBTrACS ПО СЕТИ (ibtracs_online=True, ibtracs_subset={subset!r}) "
                  f"с www.ncei.noaa.gov — нужен интернет с этой ноды; если скрипт зависнет здесь "
                  f"без ошибки, скорее всего с этой ноды/сессии нет доступа в интернет "
                  f"(типично для compute-нод кластеров) — прервите (Ctrl+C) и запустите без "
                  f"--ibtracs-online, либо с ноды, где интернет точно есть.")
            kwargs = {"source": "ibtracs", "ibtracs_online": True}
            if subset is not None:
                kwargs["ibtracs_subset"] = subset
            _ibtracs_all_cache = huracanpy.load(**kwargs)
        else:
            subset = args.ibtracs_subset or "wmo"
            print(f"Загрузка IBTrACS (встроенная в huracanpy офлайн-копия, БЕЗ сети, "
                  f"ibtracs_subset={subset!r}, один раз на весь прогон)...")
            _ibtracs_all_cache = huracanpy.load(source="ibtracs", ibtracs_online=False, ibtracs_subset=subset)
    return _ibtracs_all_cache


def load_reference(args, years):
    """(dataset, reference_label, reference_id_column). reference_label — то же
    имя, что уйдёт в huracanpy.assess.match(names=[reference_label, OWN_NAME]),
    поэтому reference_id_column = f"id_{reference_label}" (так huracanpy называет
    id-колонку в matches, см. compare_reference_lib.OWN_ID_COLUMN)."""
    if args.reference == 'ibtracs':
        ref = _get_ibtracs_all(args)
        if args.ibtracs_basin:
            ref = ref.where(ref.basin == args.ibtracs_basin, drop=True)
        ref = ref.where(ref.time.dt.year.isin(years), drop=True)
        label = "IBTrACS"
    else:
        root = args.syclops_root or (Path(args.data_dir).parent / "SyCLoPS" / "tracks_types_csv")
        cache_key = (args.syclops_type, args.syclops_region, tuple(years), args.syclops_on_multiple)
        if cache_key not in _syclops_cache:
            _syclops_cache[cache_key] = lib.load_syclops_tracks(
                args.syclops_type, years, region=args.syclops_region, root=root,
                on_multiple=args.syclops_on_multiple,
            )
        ref = _syclops_cache[cache_key]
        label = "SyCLoPS"

    return ref, label, f"id_{label}"


def _summarize_row(config, stats_df, n_own, n_reference, pod) -> dict:
    pod_effective = pod if pod is not None else (
        float(stats_df.detected.mean()) if len(stats_df) else float("nan")
    )
    n_reference_effective = n_reference if n_reference is not None else len(stats_df)
    summary = lib.summarize_track_statistics(stats_df, pod_effective, n_reference_effective, n_own)
    return {
        "tracking_type": config['type'], "speed": config['speed'],
        "config_label": f"{config['type']}_{config['speed']}", **summary,
    }


def run_config(config, args, months, ref, reference_label, reference_id_column, out_root):
    label = f"{config['type']}_{config['speed']}"
    config_dir = out_root / label
    matches_csv = config_dir / "matches_raw.csv"
    stats_csv = config_dir / "matching_statistics.csv"

    if not args.force and stats_csv.exists():
        print(f"[{label}] {stats_csv} уже есть, пропуск (--force для пересчёта)")
        stats_df = pd.read_csv(stats_csv)
        return _summarize_row(config, stats_df, n_own=None, n_reference=None, pod=None)

    own_tracks = lib.load_own_tracks(months, config['path_tracks_dir'])
    if own_tracks is None:
        print(f"[{label}] треков за период не найдено в {config['path_tracks_dir']}, пропуск")
        return None

    n_reference = (int(ref.track_id.hrcn.nunique()) if hasattr(ref, "track_id")
                   else len(np.unique(ref.track_id.values)))
    n_own = int(own_tracks.track_id.hrcn.nunique())
    print(f"[{label}] {reference_label}: {n_reference} треков, {OWN_NAME}: {n_own} треков")

    matches = lib.calculate_matches(ref, own_tracks, reference_label)
    config_dir.mkdir(parents=True, exist_ok=True)
    matches.to_csv(matches_csv, index=False)

    pod = lib.calculate_pod(matches, ref, reference_label)
    stats_df = lib.calculate_track_statistics(ref, own_tracks, matches, reference_id_column)
    stats_df.to_csv(stats_csv, index=False)
    print(f"[{label}] POD={pod:.3f}, matched={int(stats_df.detected.sum())}/{len(stats_df)}")

    if args.plot_tracks:
        lib.plot_all_reference_tracks(ref, own_tracks, config_dir / "tracks", reference_label,
                                       reference_id_column, matches, config_label=label)

    return _summarize_row(config, stats_df, n_own=n_own, n_reference=n_reference, pod=pod)


def main():
    args = parse_args()

    months = list(pd.period_range(args.start_period, args.end_period, freq='M'))
    years = sorted({p.year for p in months})

    results_root = f"{args.data_dir}/{DATA_TYPE}/{DATA_TYPE}_tracks_sigma_{args.sigma}/{PREF_TRACKING}"
    if not os.path.isdir(results_root):
        sys.exit(f"Нет папки с треками: {results_root}\n"
                  f"Проверьте --data-dir/--sigma — треки должны быть уже посчитаны "
                  f"DBSCAN_tracking_ERA5_latlon.py.")

    configs = find_configs(args, months, results_root)
    if not configs:
        sys.exit(f"Не найдено ни одной конфигурации (tracking_type x CVS_speed) с треками "
                  f"за {args.start}..{args.end} под {results_root}.")

    print(f"Найдено {len(configs)} конфигураций с треками за {args.start}..{args.end}:")
    for config in configs:
        print(f"  {config['type']}_{config['speed']}  ({config['path_tracks_dir']})")

    if args.dry_run:
        print("--dry-run: выполнение пропущено (huracanpy не импортировался)")
        return

    ref, reference_label, reference_id_column = load_reference(args, years)
    out_root = Path(results_root) / "comparison_reference" / reference_subpath(args) / f"{args.start}_{args.end}"

    summaries = []
    for config in configs:
        try:
            row = run_config(config, args, months, ref, reference_label, reference_id_column, out_root)
        except Exception as e:
            print(f"ОШИБКА для {config['type']}_{config['speed']}: {e}")
            continue
        if row is not None:
            summaries.append(row)

    summary_df = pd.DataFrame(summaries)
    if len(summary_df):
        summary_df = summary_df.sort_values(
            ["POD", "mean_overlap_percent"], ascending=[False, False]
        ).reset_index(drop=True)

    out_root.mkdir(parents=True, exist_ok=True)
    summary_csv = out_root / "summary_all_configurations.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nСводная таблица: {summary_csv}")
    print(summary_df.to_string(index=False))

    if len(summary_df):
        title = f"{OWN_NAME} vs {reference_label}" + (
            f" ({args.syclops_type}/{args.syclops_region})" if args.reference == 'syclops' else ""
        )
        lib.plot_pod_bar(summary_df, out_root / "summary_pod_bar.png", title=f"Top POD — {title}")
        if len(summary_df) > 1:
            lib.plot_pod_vs_overlap_scatter(summary_df, out_root / "summary_pod_vs_overlap.png", title=title)


if __name__ == '__main__':
    main()