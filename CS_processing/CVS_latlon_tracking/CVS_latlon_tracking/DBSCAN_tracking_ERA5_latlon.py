#!/usr/bin/env python3
"""
Запуск трекинга КВС по данным ERA5 с адаптацией под честные широту/долготу
(см. README.md и README_adaptation.md в этой же папке).

Пример запуска (годы — [начало, конец), как np.arange):

    python3 DBSCAN_tracking_ERA5_latlon.py --sigma 0 --years 1979 2019 \\
        --data-dir /storage/thalassa/users/vkoshkina/data

Полный список параметров: python3 DBSCAN_tracking_ERA5_latlon.py --help

Отличия от предыдущей версии — см. CHANGELOG.md. Коротко:
  - трекинг только по методам tracking_local_*/tracking_global_only —
    'tracking_local_2_phase_cone' убран целиком (был не адаптирован под
    lat/lon и не входит в текущий набор методов);
  - sigma/годы/путь к данным теперь параметры командной строки (argparse),
    а не input()/переменная окружения — без этого years, например, был
    мёртвым кодом: значение задавалось, но нигде не читалось;
  - убраны неиспользуемые импорты (pathos, bare `multiprocessing`, Manager,
    functools.partial-в-раннере) и неиспользуемая get_available_files;
  - дневной/месячный варианты чтения файлов сведены в один код (раньше были
    двумя почти идентичными копипастами).
"""
import argparse
import os
import re
import sys
from functools import partial
from multiprocessing import Pool

import numpy as np
import xarray as xr
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from step_of_tracking import (  # noqa: E402
    circ, dt_step, time_name,
    get_tracking_function, initialize_tracking_methods,
    initialize_tracks, save_track_csv, save_track_txt,
)

DATA_TYPE = 'ERA5'  # эта копия пайплайна — только для ERA5, см. README_adaptation.md

# Методы трекинга и варианты учёта скорости, которые реально гоняем.
# tracking_local_2_phase_cone здесь больше нет — см. CHANGELOG.md.
TRACKING_TYPES = [
    'tracking_local_2_phase',
    'tracking_global_only',
    'tracking_local_global',
    # 'tracking_local_only',       # адаптирован и рабочий, но не входит в набор по умолчанию
]

SPEED_OPTIONS = [
    'adv_speed',
    'no_speed',
    # 'bg_speed', 'adv_bg_speed',  # раскомментируйте при необходимости
]

PREF_TRACKING = 'update_2026-08-31_latlon'

# --- Шаблоны имён папок с исходными .nc (после ERA5/compute_DBSCAN_latlon*.py) ---
# ВНИМАНИЕ: это конкретные имена под ваш текущий расчёт (эксперимент
# "NA_for_TC_850hPa"), а не универсальный шаблон — проверьте их перед
# запуском под новый расчёт (или передайте свой через --folder-name).
FOLDER_NAME_SIGMA0 = f'{DATA_TYPE}/DBSCAN_02-04-10_sigma_0'
FOLDER_NAME_TEMPLATE = f'{DATA_TYPE}/DBSCAN_02-04-25_NA_for_TC_850hPa_sigma_{{sigma}}_rad'


def parse_args():
    parser = argparse.ArgumentParser(
        description="Трекинг КВС по ERA5 (lat/lon-адаптация)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--sigma', type=int, required=True,
                         help="параметр сглаживания DBSCAN, как в имени входных файлов")
    parser.add_argument('--years', type=int, nargs=2, metavar=('START', 'END'),
                         default=(1979, 2019),
                         help="диапазон лет [START, END) — по аналогии с np.arange")
    parser.add_argument('--data-dir', default=os.environ.get('ERA5_TRACKS_DATA_DIR'),
                         help="корень с данными (см. README.md); можно также задать "
                              "переменной окружения ERA5_TRACKS_DATA_DIR")
    parser.add_argument('--folder-name', default=None,
                         help="переопределить папку с входными .nc внутри --data-dir "
                              "(по умолчанию берётся по FOLDER_NAME_TEMPLATE/_SIGMA0 в этом файле)")
    parser.add_argument('--workers', type=int, default=None,
                         help="число процессов (по умолчанию — по числу конфигураций и ядер)")
    args = parser.parse_args()

    if not args.data_dir:
        parser.error(
            "не указан путь к данным: передайте --data-dir или задайте "
            "переменную окружения ERA5_TRACKS_DATA_DIR"
        )

    if args.folder_name is None:
        args.folder_name = (
            FOLDER_NAME_SIGMA0 if args.sigma == 0 else FOLDER_NAME_TEMPLATE.format(sigma=args.sigma)
        )

    data_dir = os.path.join(args.data_dir, args.folder_name)
    if not os.path.isdir(data_dir):
        parser.error(
            f"папка с данными не найдена: {data_dir}\n"
            f"Проверьте --data-dir/--folder-name (или переменную ERA5_TRACKS_DATA_DIR)."
        )

    return args


_DATE_RE = re.compile(r'(\d{4})-(\d{2})(?:-(\d{2}))?')


def list_data_files(data_dir, name_pattern, years_range):
    """
    Список .nc-файлов в data_dir, содержащих name_pattern в имени и
    попадающих в year_start <= год < year_end, отсортированный по дате.
    Каждый элемент: {'year', 'month', 'day' (None для месячных файлов), 'file'}.
    """
    year_start, year_end = years_range

    entries = []
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith('.nc') or name_pattern not in filename:
            continue

        match = _DATE_RE.search(filename)
        if not match:
            continue

        year, month = int(match.group(1)), int(match.group(2))
        if not (year_start <= year < year_end):
            continue

        day = int(match.group(3)) if match.group(3) else None
        entries.append({'year': year, 'month': month, 'day': day, 'file': filename})

    entries.sort(key=lambda e: (e['year'], e['month'], e['day'] or 0))
    return entries


def process_tracking_config(config, data_dir, name_pattern, years_range, results_root):
    """Прогоняет один метод трекинга (tracking_type + CVS_speed) по всем файлам."""
    tracking_type, CVS_speed = config['type'], config['CVS_speed']
    tracking_func = get_tracking_function(tracking_type=tracking_type, CVS_speed=CVS_speed)

    time_units = list_data_files(data_dir, name_pattern, years_range)
    if not time_units:
        print(f"No files found in {data_dir} matching '{name_pattern}*' for years {years_range}")
        return {'config': config, 'tracks': []}

    time_format = 'daily' if time_units[0]['day'] is not None else 'monthly'

    results_dir = f"{tracking_type}_{CVS_speed}_{PREF_TRACKING}"
    path_data_dir = f"{results_root}/{results_dir}/tracks_{circ}/"
    os.makedirs(path_data_dir, exist_ok=True)

    pbar = tqdm(time_units, desc=f"{tracking_type} ({CVS_speed}) - {time_format}",
                position=os.getpid() % 10, leave=False)

    very_first = True
    CS_tracks_list, clstr_len, cluster_idx = [], 0, 0

    for tu in pbar:
        ds = xr.open_dataset(os.path.join(data_dir, tu['file']))

        if very_first:
            very_first = False
            t_start = 1
            CS_tracks_list, clstr_len = initialize_tracks(ds, tracking_type, DATA_TYPE, CS_tracks_list,
                                                            our_time=0, circ=circ)
        else:
            t_start = 0

        for t in range(t_start, len(ds[time_name])):
            cluster_idx, CS_tracks_list, clstr_len = tracking_func(
                cluster_idx=cluster_idx,
                CS_tracks_list=CS_tracks_list,
                clstr_len=clstr_len,
                ds=ds,
                data_type=DATA_TYPE,
                path_data_dir=path_data_dir,
                our_time=t,
                circ=circ,
                dt_step=dt_step,
            )

        date_label = (f"{tu['year']}-{tu['month']:02d}-{tu['day']:02d}" if tu['day']
                      else f"{tu['year']}-{tu['month']:02d}")
        pbar.set_postfix_str(f"Треков: {cluster_idx}, дата: {date_label}")

        ds.close()

    pbar.close()

    for TC in CS_tracks_list:
        if np.sum(~np.isnan(TC['t'])) >= 3:
            cluster_idx = save_track_csv(cluster_idx, TC, path_data_dir)
            save_track_txt(cluster_idx, TC, path_data_dir)

    return {'config': config, 'tracks': CS_tracks_list}


def build_configs():
    return [{'type': t, 'CVS_speed': s} for t in TRACKING_TYPES for s in SPEED_OPTIONS]


def main():
    args = parse_args()
    initialize_tracking_methods()

    data_dir = os.path.join(args.data_dir, args.folder_name)
    name_pattern = f'sigma_{args.sigma}_DBSCAN_{DATA_TYPE}'
    results_root = f"{args.data_dir}/{DATA_TYPE}/{DATA_TYPE}_tracks_sigma_{args.sigma}/{PREF_TRACKING}"

    configs = build_configs()
    worker = partial(process_tracking_config, data_dir=data_dir, name_pattern=name_pattern,
                      years_range=tuple(args.years), results_root=results_root)

    n_workers = args.workers or min(len(configs), max(os.cpu_count() - 1, 1))

    print(f"type of params: {PREF_TRACKING}")
    print(f"data dir: {data_dir}")
    print(f"years: {args.years[0]}-{args.years[1] - 1}, sigma: {args.sigma}, workers: {n_workers}")

    main_pbar = tqdm(total=len(configs), desc="Все конфигурации", position=0)
    with Pool(processes=n_workers) as pool:
        results = []
        for result in pool.imap_unordered(worker, configs):
            results.append(result)
            main_pbar.update(1)
    main_pbar.close()

    print("\nAll tracking configurations processed successfully!")
    return {r['config']['type']: r['tracks'] for r in results}


if __name__ == '__main__':
    main()
