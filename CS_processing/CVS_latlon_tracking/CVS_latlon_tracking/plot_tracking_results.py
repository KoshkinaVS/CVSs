#!/usr/bin/env python3
"""
Отрисовка результатов трекинга КВС по ERA5 (lat/lon-адаптация) за выбранный
период — например, август-сентябрь 2010.

Пример запуска:

    python3 plot_tracking_results.py \\
        --sigma 0 --tracking-type tracking_local_2_phase --speed no_speed \\
        --start 2010-08 --end 2010-09 \\
        --data-dir /storage/thalassa/users/vkoshkina/data

На каждый временной шаг периода сохраняется один PNG: поле критерия R2D
(закраска), центры локальных экстремумов (звёзды, цвет — id кластера) и уже
накопленные к этому шагу треки (линии). Полный список параметров:

    python3 plot_tracking_results.py --help

Это ERA5-only адаптация ../CS_tracking/after_70RAE/plot_tracking_results.py
(+ .../plot_tracking_results_func.py). Что изменилось и почему — см.
CHANGELOG.md, раздел "plot_tracking_results.py". Коротко:
  - параметры — argparse (--start/--end YYYY-MM, включительно), а не
    input()/захардкоженный `months = np.arange(1,13,1)` на весь год;
  - x/y — реальные lon/lat (эта копия пайплайна и так lat/lon-адаптирована —
    рисовать по индексам сетки, как в оригинале для WRF, было бы неверно
    именно там, где расхождение и адаптировалось, см. README_adaptation.md);
  - путь к входным .nc (DBSCAN/R2D) и структура папки с результатами не
    задаются заново, а берутся из DBSCAN_tracking_ERA5_latlon.py
    (list_data_files, PREF_TRACKING, FOLDER_NAME_*) — один источник правды
    для раннера трекинга и для этого скрипта;
  - убраны неиспользуемые в оригинале зависимости: cartopy и shapely
    импортировались, но не участвовали в рисовании (никакой ccrs-проекции у
    ax не было, Polygon нигде не создавался) — здесь lon/lat рисуются как
    обычные декартовы оси; cmaps (сторонний пакет ради одной палитры)
    заменён на встроенную палитру matplotlib.
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # без дисплея: сервер сохраняет PNG, а не показывает окно
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

import numpy as np
import pandas as pd
import xarray as xr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from step_of_tracking import (  # noqa: E402
    CRIT_FIELD, LEVEL_INDEX, TRACKING_METHODS,
    circ as CIRC_DEFAULT, time_name, x_unit, y_unit, level_unit,
)
from latlon_utils import grid_step_deg  # noqa: E402
from DBSCAN_tracking_ERA5_latlon import (  # noqa: E402
    DATA_TYPE, PREF_TRACKING, FOLDER_NAME_SIGMA0, FOLDER_NAME_TEMPLATE, list_data_files,
)

# Варианты CVS_speed, которые понимает get_next_loc_cases (step_of_tracking.py) —
# это не JSON-параметр и нигде не вынесено в список там, поэтому здесь он свой,
# просто чтобы --speed проверялся argparse'ом, а не падал на этапе загрузки CSV.
SPEED_CHOICES = ['no_speed', 'adv_speed', 'bg_speed', 'adv_bg_speed']

LOCAL_STAR_SIZE = 18    # local_extr_crit/local_extr_cluster — локальные экстремумы (их в кластере может быть несколько)
GLOBAL_STAR_SIZE = 90   # center/center_cluster — глобальный максимум R2D кластера (get_stat_global_max), один на кластер
TRACK_POINT_SIZE = 14   # текущее положение трека (маркер 'o', красный/синий)

# Подберите под свои данные: это диапазон закраски поля R2D (contourf), не
# порог отбора вихрей — тот уже применён при расчёте DBSCAN (local_extr_crit).
CRIT_VMIN, CRIT_VMAX = -0.0004, 0.0004

# .nc с маской суши по умолчанию (см. ERA5/get_lsm_cropped.py — тот же файл,
# что и в ../CS_tracking/after_70RAE/plot_tracking_results_func.py для этого
# региона). Используется, если --lsm-file не передан; если файла там нет —
# просто рисуем без подложки (см. load_land_mask), а не падаем.
DEFAULT_LSM_RELPATH = os.path.join('ERA5', 'ERA5_lsm_cropped_NA_for_TC.nc')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Отрисовка треков КВС (ERA5, lat/lon) за период",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--sigma', type=int, required=True,
                         help="параметр сглаживания DBSCAN, как при запуске трекинга")
    parser.add_argument('--tracking-type', required=True, choices=sorted(TRACKING_METHODS),
                         help="какой метод трекинга рисовать (папка с результатами)")
    parser.add_argument('--speed', required=True, choices=SPEED_CHOICES,
                         help="вариант CVS_speed, с которым гонялся трекинг")
    parser.add_argument('--start', required=True, metavar='YYYY-MM',
                         help="первый месяц периода, включительно (например, 2010-08)")
    parser.add_argument('--end', required=True, metavar='YYYY-MM',
                         help="последний месяц периода, включительно (например, 2010-09)")
    parser.add_argument('--data-dir', default=os.environ.get('ERA5_TRACKS_DATA_DIR'),
                         help="корень с данными, как при запуске трекинга; можно также "
                              "задать переменной окружения ERA5_TRACKS_DATA_DIR")
    parser.add_argument('--folder-name', default=None,
                         help="переопределить папку с входными .nc внутри --data-dir "
                              "(по умолчанию — как в DBSCAN_tracking_ERA5_latlon.py)")
    parser.add_argument('--circ', default=CIRC_DEFAULT, choices=['C', 'AC'],
                         help="циклонические (C) или антициклонические (AC) КВС")
    parser.add_argument('--lsm-file', default=None,
                         help="переопределить .nc с маской суши (переменная 'lsm' или "
                              "'var172'); по умолчанию берётся {data-dir}/" +
                              DEFAULT_LSM_RELPATH.replace(os.sep, '/') +
                              " (см. ERA5/get_lsm_cropped.py), если он существует")
    parser.add_argument('--out-dir', default=None,
                         help="куда сохранять PNG (по умолчанию — рядом с треками, "
                              "см. --help вывод пути перед запуском)")
    parser.add_argument('--show-radius', action='store_true',
                         help="дополнительно рисовать эллипс rad_eff вокруг текущей точки трека")
    parser.add_argument('--dpi', type=int, default=200)
    args = parser.parse_args()

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

    if args.folder_name is None:
        args.folder_name = (
            FOLDER_NAME_SIGMA0 if args.sigma == 0 else FOLDER_NAME_TEMPLATE.format(sigma=args.sigma)
        )

    data_dir = os.path.join(args.data_dir, args.folder_name)
    if not os.path.isdir(data_dir):
        parser.error(f"папка с входными .nc не найдена: {data_dir}\n"
                      f"Проверьте --data-dir/--folder-name.")

    return args


def load_period_tracks(months, path_tracks_dir):
    """Треки (CSV из save_track_csv), попавшие в любой из месяцев `months`."""
    CS_tracks_list = []
    for period in months:
        month_dir = Path(path_tracks_dir) / f"{period.year}-{period.month:02d}"
        if not month_dir.is_dir():
            continue
        for csv_path in sorted(month_dir.glob('*_track_*.csv')):
            df = pd.read_csv(csv_path, index_col=0, parse_dates=['datetime'])
            CS_tracks_list.append(df)
    return CS_tracks_list


def load_land_mask(lsm_file):
    """(lon, lat, mask) для серой подложки суши, либо None, если файл не задан/не найден.

    Печатает, какой путь пробовался и чем закончилось — специально явно и
    безусловно, чтобы в логе всегда было видно, реально ли эта версия
    скрипта вообще пыталась подключить маску суши (а не молчала).
    """
    print(f"Маска суши: {lsm_file}")

    if not lsm_file:
        print("  не задана (--lsm-file пуст и не удалось определить путь по умолчанию)")
        return None
    if not os.path.isfile(lsm_file):
        print("  файл не найден, рисую без подложки суши")
        return None

    with xr.open_dataset(lsm_file) as ds:
        var_name = next((v for v in ('lsm', 'var172') if v in ds.variables), None)
        if var_name is None:
            print(f"  нет переменной 'lsm'/'var172' (есть: {list(ds.variables)}), рисую без подложки")
            return None

        field = ds[var_name]
        # У некоторых lsm-файлов есть лишняя размерность time (field.ndim==3) —
        # у других (как у вашего ERA5_lsm_cropped_NA_for_TC.nc) её нет, там
        # сразу (lat, lon). Оба случая нормальны, только 3D нужно "схлопнуть".
        if field.ndim == 3:
            field = field[0]

        lon = field['longitude'].values if 'longitude' in field.coords else field['lon'].values
        lat = field['latitude'].values if 'latitude' in field.coords else field['lat'].values
        # 0/1 без NaN — нужен непрерывный массив, чтобы contour() мог провести
        # линию границы суша/море (с NaN он просто не рисует линию на стыке).
        mask = np.where(field.values > 0.5, 1.0, 0.0)

    print(f"  загружена: {var_name} {mask.shape}, lon [{lon.min():.2f}, {lon.max():.2f}], "
          f"lat [{lat.min():.2f}, {lat.max():.2f}], суши: {int((mask > 0).sum())} точек")
    return lon, lat, mask


def plot_ground(ax, land_mask):
    """Заливка суши (полупрозрачная) + чёткая линия границы суша/море."""
    if land_mask is None:
        return
    lon, lat, mask = land_mask
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    ax.contourf(lon_grid, lat_grid, np.where(mask > 0.5, 1.0, np.nan),
                levels=[0.5, 1.5], colors=['0.5'], alpha=0.3, zorder=1)
    ax.contour(lon_grid, lat_grid, mask, levels=[0.5], colors='black',
               linewidths=0.7, zorder=6)  # выше поля R2D (zorder=2), чтобы не потеряться под закраской


def _circ_select(crit_field, circ):
    """Тот же знаковый отбор, что и в get_stat_local_max/get_stat_global_max
    (step_of_tracking.py): положительный R2D — циклоны (C), отрицательный —
    антициклоны (AC)."""
    return crit_field > 0 if circ == 'C' else crit_field < 0


def plot_field(ax, ds, t, circ):
    """Поле критерия R2D (закраска) + звёзды локальных экстремумов и центров
    кластеров на шаге t."""
    lon = ds[x_unit].values
    lat = ds[y_unit].values
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    field = ds[CRIT_FIELD].isel({time_name: t, level_unit: LEVEL_INDEX}).values
    ax.contourf(lon_grid, lat_grid, field, cmap='PiYG', vmin=CRIT_VMIN, vmax=CRIT_VMAX,
                levels=50, zorder=2)

    # local_extr_* — локальные экстремумы (может быть несколько на кластер,
    # get_stat_local_max); center/center_cluster — единственный глобальный
    # максимум R2D по кластеру (get_stat_global_max). Оба набора кодируются
    # одной палитрой по id кластера, чтобы маленькая и большая звезда одного
    # кластера были одного цвета — для этого у обоих scatter один и тот же
    # vmin/vmax по числу кластеров, а не автомасштаб по своим же данным.
    extr_crit = ds['local_extr_crit'].isel({time_name: t, level_unit: LEVEL_INDEX}).values
    extr_cluster = ds['local_extr_cluster'].isel({time_name: t, level_unit: LEVEL_INDEX}).values
    extr_vals = np.where(_circ_select(extr_crit, circ), extr_cluster, np.nan)

    center_crit = ds['center'].isel({time_name: t, level_unit: LEVEL_INDEX}).values
    center_cluster = ds['center_cluster'].isel({time_name: t, level_unit: LEVEL_INDEX}).values
    center_vals = np.where(_circ_select(center_crit, circ), center_cluster, np.nan)

    max_cluster_id = np.nanmax(np.concatenate([
        extr_vals[~np.isnan(extr_vals)], center_vals[~np.isnan(center_vals)], [0.0],
    ]))
    n_clusters = int(max_cluster_id) + 1
    cmap = plt.get_cmap('tab20', n_clusters)

    ax.scatter(lon_grid, lat_grid, c=extr_vals, s=LOCAL_STAR_SIZE, cmap=cmap, marker='*',
               vmin=0, vmax=n_clusters - 1, alpha=1.0, zorder=15)
    ax.scatter(lon_grid, lat_grid, c=center_vals, s=GLOBAL_STAR_SIZE, cmap=cmap, marker='*',
               vmin=0, vmax=n_clusters - 1, edgecolors='black', linewidths=0.5,
               alpha=1.0, zorder=16)

    ax.set_title(str(pd.Timestamp(ds[time_name].values[t]))[:16])
    ax.set_xlim(float(lon.min()), float(lon.max()))
    ax.set_ylim(float(lat.min()), float(lat.max()))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(linestyle=':', linewidth=0.5)
    ax.tick_params(axis='both', which='both', direction='in', labelsize=9)


def plot_tracks(ax, ds, CS_tracks_list, current_time, show_radius):
    """Накопленные к current_time треки: линия — путь, точка — текущее положение."""
    dlon_deg, dlat_deg = grid_step_deg(ds) if show_radius else (None, None)

    for df in CS_tracks_list:
        points = df[df['datetime'] <= current_time]
        if points.empty:
            continue
        points = points.sort_values('datetime')

        is_last_point = df['datetime'].max() == current_time
        if len(points) > 1:
            ax.plot(points['lon'], points['lat'], lw=1.2, alpha=0.9,
                     c='red' if is_last_point else 'black', zorder=10)

        current_point = points[points['datetime'] == current_time]
        if current_point.empty:
            continue
        row = current_point.iloc[-1]

        ax.scatter(row['lon'], row['lat'], s=TRACK_POINT_SIZE,
                    c='red' if is_last_point else 'blue', marker='o',
                    edgecolors='white', linewidths=0.5, zorder=20)

        if show_radius and not np.isnan(row['rad']):
            ellipse = Ellipse(
                xy=(row['lon'], row['lat']),
                width=2 * row['rad'] * dlon_deg, height=2 * row['rad'] * dlat_deg,
                fill=False, edgecolor='k', linewidth=1.0, alpha=0.8, zorder=12,
            )
            ax.add_patch(ellipse)


def main():
    args = parse_args()

    data_dir = os.path.join(args.data_dir, args.folder_name)
    name_pattern = f'sigma_{args.sigma}_DBSCAN_{DATA_TYPE}'

    months = list(pd.period_range(args.start_period, args.end_period, freq='M'))
    years_covered = (months[0].year, months[-1].year + 1)
    file_entries = [
        e for e in list_data_files(data_dir, name_pattern, years_covered)
        if any(e['year'] == p.year and e['month'] == p.month for p in months)
    ]
    if not file_entries:
        sys.exit(f"Нет входных .nc в {data_dir} для периода {args.start}..{args.end} "
                  f"(паттерн имени: '{name_pattern}*')")

    results_root = f"{args.data_dir}/{DATA_TYPE}/{DATA_TYPE}_tracks_sigma_{args.sigma}/{PREF_TRACKING}"
    results_dir = f"{args.tracking_type}_{args.speed}_{PREF_TRACKING}"
    path_tracks_dir = f"{results_root}/{results_dir}/tracks_{args.circ}"
    if not os.path.isdir(path_tracks_dir):
        sys.exit(f"Нет папки с треками: {path_tracks_dir}\n"
                  f"Проверьте --sigma/--tracking-type/--speed/--circ — они должны "
                  f"совпадать с тем, чем реально гонялся DBSCAN_tracking_ERA5_latlon.py.")

    CS_tracks_list = load_period_tracks(months, path_tracks_dir)
    print(f"Треков за {args.start}..{args.end}: {len(CS_tracks_list)}")

    lsm_file = args.lsm_file or os.path.join(args.data_dir, DEFAULT_LSM_RELPATH)
    land_mask = load_land_mask(lsm_file)
    if land_mask is not None and args.lsm_file is None:
        print("  (путь по умолчанию; переопределить — флагом --lsm-file)")

    out_dir = args.out_dir or f"{path_tracks_dir}/pics/{args.start}_{args.end}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"PNG сохраняются в: {out_dir}")

    frame_idx = 0
    for entry in file_entries:
        with xr.open_dataset(os.path.join(data_dir, entry['file'])) as ds:
            for t in range(len(ds[time_name])):
                current_time = pd.Timestamp(ds[time_name].values[t])

                fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
                ax.grid(which='major', linewidth=1.0)

                plot_ground(ax, land_mask)
                plot_field(ax, ds, t, args.circ)
                plot_tracks(ax, ds, CS_tracks_list, current_time, args.show_radius)

                fig.savefig(f"{out_dir}/track_{frame_idx:05d}.png",
                            dpi=args.dpi, bbox_inches="tight")
                plt.close(fig)

                frame_idx += 1
                if frame_idx % 24 == 0:
                    print(f"  {frame_idx} кадров готово, последний: {current_time}")

    print(f"Готово: {frame_idx} кадров в {out_dir}")


if __name__ == '__main__':
    main()