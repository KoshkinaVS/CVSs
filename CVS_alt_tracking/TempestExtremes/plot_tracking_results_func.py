from matplotlib import pyplot as plt
import pandas as pd 
import numpy as np
import math
import xarray as xr

import datetime
from datetime import timedelta

# import scipy as sp
# from scipy.ndimage import label, generate_binary_structure
# from scipy import interpolate

import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from shapely.geometry import Polygon
import matplotlib.patches as mpatches
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tqdm import tqdm

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


import cmaps

from pathlib import Path

import sys
import os

# Инициализация путей и параметров
path_init = f'/storage/thalassa/users/vkoshkina'
folder = 'scripts/CS_processing/CVS_tracking/after_70RAE/'
sys.path.insert(2, f'{path_init}/{folder}')

from step_of_tracking import *




local_extr_name = 'local_extr_crit'

dist_m, our_level, level, x_unit, y_unit, u_unit, v_unit, time_unit, level_unit, crit, th, min_samples, eps, CS_points_th = get_init_params(data_type, json_file, print_info=True)

path_data = f'{path_init}/data'
if data_type == 'LoRes':
    path_data = f'{path_data}/LoRes'
    
DBSCAN_name_tracks = f'DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_level_{level}_smoothing_sigma_{sigma}'


if data_type == 'ERA5':
    
    
#     DBSCAN_name = f'DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_sigma_{sigma}'
    
#     if sigma != 0:
#         name_init = f'sigma_{sigma}_DBSCAN_DBSCAN_{data_type}_level_{level}'
#     else:
#         name_init = f'sigma_{sigma}_DBSCAN_{data_type}_level_{level}'
        
        
        
    #### 2026-07-08
    DBSCAN_name = f'DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_NA_for_TC_500hPa_sigma_{sigma}'
    name_init = f'sigma_{sigma}_DBSCAN_{data_type}'
    

        
elif data_type == 'GPN':
    name_init = f'sigma_{sigma}_DBSCAN_{data_type}_level_{level}'
    DBSCAN_name = f'DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_sigma_{sigma}'
elif data_type == 'SMP':
    name_init = f'sigma_{sigma}_DBSCAN_{data_type}_level_{level}'
    DBSCAN_name = f'DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_with_wspd_smoothing'
else:        
    name_init = f'sigma_{sigma}_DBSCAN_{data_type}_level_{level}'
    DBSCAN_name = f'DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_smoothing_sigma_{sigma}'

if data_type == 'LoRes':
    crit_scale = 0.7
    star_size = 30
    year = 2010
    
elif data_type == 'HiRes':
    crit_scale = 0.1
    star_size = 5
    year = 2010
    
elif data_type == 'ERA5':
    crit_scale = 0.1
    star_size = 5
    year = 1979
    
elif data_type == 'GPN':
    crit_scale = 0.7
    star_size = 5
    year = 2022
elif data_type == 'SMP':
    crit_scale = 0.7
    star_size = 5    
    year = 2019

our_level = 0

# print('year: ')
# year = int(input())


if data_type == 'GPN' or data_type == 'SMP':
    folder_name = f'{data_type}/{DBSCAN_name}/{year}'
else:
    folder_name = f'{data_type}/{DBSCAN_name}'

vmin = -127
vmax = 127


vmin = -0.0004
vmax = 0.0004


if data_type != 'ERA5':
    if data_type == 'GPN':
        path_dir_raw = f'/storage/buffer/{data_type}/OUTPUTS/WRF6km/{year}'
    elif data_type == 'SMP':
        path_dir_raw = f'/storage/thalassa/SMP/WRF_OUTPUT/v2/'
    else:
        path_dir_raw = f'/storage/NAAD/NAAD/{data_type}/{year}'
    ncfile = f'{path_dir_raw}/wrfout_d01_{year}-02-01_00:00:00'
    
    ground_ds = xr.open_dataset(f'{ncfile}')['HGT'][0]
    ground = np.where(ground_ds > 5, 1, np.nan)
    xxx = ground_ds[x_unit]
    yyy = ground_ds[y_unit]
    xx, yy = np.meshgrid(xxx, yyy)
    
    ground_ds.close()
else:
    path_dir_raw = f'/storage/thalassa/users/vkoshkina/data/ERA5'
#     ncfile = f'{path_dir_raw}/ERA5_lsm_cropped.nc' 
#     ground_ds = xr.open_dataset(f'{ncfile}')['var172'][0]
    
    ncfile = f'{path_dir_raw}/ERA5_lsm_cropped_NA_for_TC.nc'
    ground_ds = xr.open_dataset(f'{ncfile}')['lsm']
    
    
    ground = np.where(ground_ds > 0.5, 1, np.nan)
    xxx = np.arange(len(ground_ds['lon']))
    yyy = np.arange(len(ground_ds['lat']))

    xx, yy = np.meshgrid(xxx, yyy)
    
    ground_ds.close()

def plot_ground(ax, ground):
    mc = ax.contourf(xxx, yyy, ground, 
                cmap='grey', 
                alpha=0.07
                 )

# def plot_crit(ax, ds, crit_vals, local_extr, t):

#     mc = ax.contourf(xxx, yyy, crit_vals[t, our_level], 
#                 cmap='PiYG', 
#                   vmin=crit_scale*vmin, vmax=crit_scale*vmax,
#                  )

#     plt.title(str(ds[time_name][t].values)[:16])

#     plt.xlim(0,len(xxx))
#     plt.ylim(0,len(yyy))
    
#     n_clusters = int(np.nanmax(ds['cluster'].values)+1)
#     cmap = plt.cm.get_cmap(cmaps.psgcap, n_clusters)

#     Q_center = ax.scatter(xx, yy,
#                           c=local_extr[t,our_level],
#                               s=star_size, 
#                           # cmap=cmaps.temp_diff_1lev_r,
#                           # vmin=-1,vmax=1,
#                           cmap=cmap,
#                           marker='*', alpha=1., label='center',
#                           zorder=15,
#                          )

#     plt.grid(linestyle = ':', linewidth = 0.5)
#     plt.tick_params(axis='both', which='both', direction='in', labelsize=9)


def plot_bounds(ax, CS, i, color='blue'):

    hw = int(np.round(CS['rad'][i]))*1.5
    
    y = CS['y'][i]
    x = CS['x'][i]

    p = mpatches.Rectangle(xy=(x - hw/2,y - hw/2), width=hw, height=hw, linewidth=0.5, 
                                     edgecolor='blue', facecolor='none',)
    ax.add_patch(p)

# def time_plot_tracks(ax, dt_step, CS_tracks_list, our_xtime):
#     """Улучшенная функция отрисовки треков"""
#     for CS in CS_tracks_list:
#         # Проверяем, есть ли текущее время в треке
#         if np.isin(our_xtime, CS['datetime']).any():
#             # Находим все точки трека до текущего времени
#             mask = CS['datetime'] <= our_xtime
#             track_points = CS[mask]
            
#             # Отображаем трек только если есть достаточное количество точек
#             if len(track_points) > 1:
#                 # Сортируем точки по времени для правильного соединения
#                 track_points = track_points.sort_values('datetime')
                
#                 # Отрисовываем линию трека
#                 ax.plot(track_points['x'], track_points['y'], 
#                         alpha=0.9, lw=1.5, c='k')
                
#                 # Отмечаем текущую позицию
#                 last_point = track_points.iloc[-1]
#                 ax.scatter(last_point['x'], last_point['y'], 
#                           s=star_size, c='red', marker='o', zorder=20)


# def time_plot_tracks(ax, dt_step, CS_tracks_list, our_xtime):
#     """Функция отрисовки треков с выделением завершающих треков"""
#     for CS in CS_tracks_list:
#         # Проверяем, есть ли текущее время в треке
#         if np.isin(our_xtime, CS['datetime']).any():
#             # Находим все точки трека до текущего времени
#             mask = CS['datetime'] <= our_xtime
#             track_points = CS[mask]
            
#             # Проверяем, является ли текущее время последним для этого трека
#             is_last_point = (CS['datetime'].max() == our_xtime)
            
#             # Цвет трека: красный для завершающего, черный для остальных
#             track_color = 'red' if is_last_point else 'k'
            
#             # Отображаем трек только если есть достаточное количество точек
#             if len(track_points) > 1:
#                 # Сортируем точки по времени для правильного соединения
#                 track_points = track_points.sort_values('datetime')
                
#                 # Отрисовываем линию трека
#                 ax.plot(track_points['x'], track_points['y'], 
#                         alpha=0.9, lw=1.5, c=track_color)
                
#                 # Отмечаем текущую позицию
#                 last_point = track_points.iloc[-1]
#                 ax.scatter(last_point['x'], last_point['y'], 
#                           s=star_size, c='red' if is_last_point else 'blue', 
#                           marker='o', zorder=20)
                
#                 # Для завершающего трека добавляем дополнительное выделение
#                 if is_last_point:
#                     ax.scatter(last_point['x'], last_point['y'],
#                               s=star_size*1.5, c='red', 
#                               marker='x', linewidths=1.5, zorder=25)

def time_plot_tracks(ax, dt_step, CS_tracks_list, our_xtime):
    """Функция отрисовки треков с полным скрытием завершенных треков"""
    for CS in CS_tracks_list:
        # Проверяем, закончился ли трек до текущего времени
        if CS['datetime'].max() < our_xtime:
            continue  # Пропускаем полностью завершенные треки
            
        # Все точки трека до текущего времени (включительно)
        track_points = CS[CS['datetime'] <= our_xtime]
        
        if len(track_points) == 0:
            continue
            
        # Сортируем по времени
        track_points = track_points.sort_values('datetime')
        
        # Проверяем есть ли точка точно на этом шаге
        exact_point = our_xtime in CS['datetime'].values
        is_last_point = (CS['datetime'].max() == our_xtime)
        
        # Если нет точки на текущем шаге, берем последнюю известную
        if not exact_point:
            last_known_point = track_points.iloc[-1]
            # Выделяем последнюю известную точку зеленым
            ax.scatter(last_known_point['x'], last_known_point['y'],
                     s=star_size*1.2, c='lime', marker='o', zorder=20,
                     edgecolors='green', linewidths=1)
        
        # Отрисовываем весь трек до последней точки
        if len(track_points) > 1:
            # Цвет трека: красный если это последний шаг трека
            track_color = 'red' if is_last_point else 'k'
            ax.plot(track_points['x'], track_points['y'],
                   alpha=0.9, lw=1.5, c=track_color)
        
        # Отмечаем текущую/последнюю точку
        if exact_point:
            current_point = track_points[track_points['datetime'] == our_xtime].iloc[0]
            ax.scatter(current_point['x'], current_point['y'],
                     s=star_size, c='red' if is_last_point else 'blue',
                     marker='o', zorder=25)
            
def plot_track_with_DBSCAN(ds, ds_speed, dt_step, crit_vals, local_extr, 
                         CS_tracks_list, start_time, stop_time, idx, 
                         name_pics, save=True):

    dt_step_int = int(dt_step/3600)
    """Улучшенная функция визуализации с треками"""
    # Создаем папку для сохранения, если нужно
    if save:
        os.makedirs(f'{name_pics}/{year}', exist_ok=True)
    
    # Основной цикл по временным шагам (с шагом 3 для ускорения)
    for t in tqdm(range(start_time, stop_time, dt_step_int), 
                 desc=f"Отрисовка {name_pics}"):
        fig = plt.figure(figsize=(10, 8), dpi=150)
        ax = fig.add_subplot(111)
        
        # Настройка осей и сетки
        ax.grid(which='major', linewidth=1.2)
        ax.grid(which='minor', linestyle='--', color='gray', linewidth=0.5)
        
        # Отрисовка фона и критерия
        plot_ground(ax, ground)
        plot_crit(ax, ds, crit_vals, local_extr, t)
        
        # Отрисовка треков
        time_plot_tracks(ax, dt_step, CS_tracks_list, ds[time_name].values[t])
        
        # Инвертирование оси Y при необходимости
        # plt.gca().invert_yaxis()
        
        if save:
            plt.savefig(
                f"{name_pics}/{year}/track_{(idx + t):05d}.png", 
                dpi=200, 
                bbox_inches="tight", 
                transparent=False
            )
            plt.close()
        else:
            plt.show()
    
    return idx + (stop_time - start_time)
    

def load_season_tracks(year, months, path_data):
    """
    Загрузка треков из файлов TempestExtremes в формате .txt с фильтрацией по месяцам
    
    Параметры:
        year: int - год данных
        months: list - список месяцев для загрузки (1-12)
        path_data: str - путь к файлу с треками
        
    Возвращает:
        list - список треков в формате DataFrame, относящихся к указанным месяцам
    """
    CS_tracks_list = []
    
    if not os.path.isfile(path_data):
        return CS_tracks_list
    
    # Преобразуем months в set для быстрого поиска
    target_months = set(months)
    
    with open(path_data, 'r') as f:
        lines = f.readlines()
    
    current_track = None
    current_track_months = set()
    
    for line in lines:
        if line.startswith('start'):
            # Сохраняем предыдущий трек, если он содержит нужные месяцы
            if current_track is not None and len(current_track) > 0:
                if current_track_months & target_months:  # Проверяем пересечение месяцев
                    df = pd.DataFrame(current_track)
                    df['datetime'] = pd.to_datetime(
                        df[['year', 'month', 'day', 'hour']].astype(str).agg('-'.join, axis=1), 
                        format='%Y-%m-%d-%H'
                    )
                    CS_tracks_list.append(df)
            
            # Инициализируем новый трек
            parts = line.strip().split()
            if len(parts) >= 6:
                # Извлекаем информацию о начале трека: start 17 2010 1 1 0
                track_year = int(parts[2])
                track_month = int(parts[3])
                current_track_months = {track_month}  # Начинаем с месяца начала трека
                
                current_track = {
                    'x': [], 'y': [], 'lon': [], 'lat': [], 
                    'wspd': [], 'r2d': [], 
                    'year': [], 'month': [], 'day': [], 'hour': []
                }
        
        elif line.startswith('\t') and current_track is not None:
            # Данные точки трека
            parts = line.strip().split()
            if len(parts) >= 10:
                point_month = int(parts[7])
                current_track_months.add(point_month)  # Добавляем месяц точки в множество
                
                current_track['x'].append(float(parts[0]))
                current_track['y'].append(float(parts[1]))
                current_track['lon'].append(float(parts[2]))
                current_track['lat'].append(float(parts[3]))
                current_track['wspd'].append(float(parts[4]))
                current_track['r2d'].append(float(parts[5]))
                current_track['year'].append(int(parts[6]))
                current_track['month'].append(point_month)
                current_track['day'].append(int(parts[8]))
                current_track['hour'].append(int(parts[9]))
    
    # Добавляем последний трек, если он подходит
    if current_track is not None and len(current_track) > 0:
        if current_track_months & target_months:
            df = pd.DataFrame(current_track)
            df['datetime'] = pd.to_datetime(
                df[['year', 'month', 'day', 'hour']].astype(str).agg('-'.join, axis=1), 
                format='%Y-%m-%d-%H'
            )
            CS_tracks_list.append(df)
    
    return CS_tracks_list

# def load_local_extrema_nodes(file_path):
#     """
#     Загрузка локальных экстремумов из файла TempestExtremes Nodes
    
#     Параметры:
#         file_path: str - путь к файлу с экстремумами
        
#     Возвращает:
#         DataFrame - данные экстремумов с колонками:
#             ['x', 'y', 'lon', 'lat', 'wspd', 'r2d', 'year', 'month', 'day', 'hour', 'datetime']
#     """
#     extrema_data = []
    
#     if not os.path.isfile(file_path):
#         return pd.DataFrame()
    
#     current_time = None
    
#     with open(file_path, 'r') as f:
#         for line in f:
#             if line.startswith('\t'):
#                 parts = line.strip().split()
#                 if len(parts) >= 6 and current_time is not None:
#                     try:
#                         point_data = {
#                             'x': int(parts[0]),
#                             'y': int(parts[1]),
#                             'lon': float(parts[2]),
#                             'lat': float(parts[3]),
#                             'wspd': float(parts[4]),
#                             'r2d': float(parts[5]),
#                             'year': current_time.year,
#                             'month': current_time.month,
#                             'day': current_time.day,
#                             'hour': current_time.hour,
#                             'datetime': current_time
#                         }
#                         extrema_data.append(point_data)
#                     except ValueError:
#                         continue
#             else:
#                 parts = line.strip().split()
#                 if len(parts) >= 5:
#                     try:
#                         current_time = datetime.datetime(
#                             year=int(parts[0]),
#                             month=int(parts[1]),
#                             day=int(parts[2]),
#                             hour=int(parts[4])
#                         )
#                     except (ValueError, IndexError):
#                         continue
    
#     return pd.DataFrame(extrema_data) if extrema_data else pd.DataFrame()

def load_local_extrema_nodes(file_path):
    """
    Загрузка локальных экстремумов из файла TempestExtremes Nodes
    
    Параметры:
        file_path: str - путь к файлу с экстремумами
        
    Возвращает:
        dict - словарь с данными, где ключи - временные метки,
               значения - списки кортежей (lon, lat, value, area)
    """
    time_slices = {}
    current_time = None
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('\t'):
                # Строка с данными экстремума
                parts = line.strip().split()
                if len(parts) >= 6:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        lon = float(parts[2])
                        lat = float(parts[3])
                        wspd = float(parts[4])
                        r2d = float(parts[5])
                        if current_time is not None:
                            time_slices[current_time].append((x, y, r2d))
                    except ValueError:
                        continue
            else:
                # Строка с временной меткой (например: "2010 8 1 166 0")
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        year = int(parts[0])
                        month = int(parts[1])
                        day = int(parts[2])
                        hour = int(parts[4])  # Часы в последнем столбце
                        current_time = datetime.datetime(year, month, day, hour)
                        time_slices[current_time] = []
                    except (ValueError, IndexError):
                        continue
    
    return time_slices
    
def plot_crit(ax, ds, crit_vals, local_extr_nodes, t):
    """
    Визуализация полей и локальных экстремумов
    
    Параметры:
        ax: matplotlib.axes - оси для отрисовки
        ds: xarray.Dataset - набор данных
        crit_vals: numpy.ndarray - значения критерия
        local_extr_nodes: dict - словарь с локальными экстремумами
        t: int - индекс временного шага
    """
    # Временная метка текущего шага
    current_time = pd.to_datetime(ds[time_name][t].values)
    
    # Контурный график поля критерия
    mc = ax.contourf(xxx, yyy, crit_vals[t, our_level], 
                     cmap='PiYG', 
                     vmin=crit_scale*vmin, vmax=crit_scale*vmax)
    
    plt.title(str(current_time)[:16])
    plt.xlim(0, len(xxx))
    plt.ylim(0, len(yyy))
    plt.gca().invert_yaxis()  # Инвертируем ось Y

    
    # Отрисовка локальных экстремумов для текущего времени
    if current_time in local_extr_nodes:
        # Преобразуем координаты экстремумов в индексы сетки
        x_indices = [point[0] for point in local_extr_nodes[current_time]]
        y_indices = [point[1] for point in local_extr_nodes[current_time]]
        values = [point[2] for point in local_extr_nodes[current_time]]
        
        Q_center = ax.scatter(x_indices, y_indices,
                              c='b',
                              s=star_size,
                              marker='*',
                              alpha=1.,
                              label='center',
                              zorder=15)
    
    plt.grid(linestyle=':', linewidth=0.5)
    plt.tick_params(axis='both', which='both', direction='in', labelsize=9)