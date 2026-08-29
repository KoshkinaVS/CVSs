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


region = 'NA'


if region == 'Arctic':
    # Арктика
    level = 850
    region_name = f'Arctic_{level}hPa'
elif region == 'BarKara':
    region_name = f'BarKara_level_{level}'
    postfix = '_range_025_144h_72h'
    
else:
    # Атлантика
    level = 500
    level = 850
    
    region_name = f'NA_for_TC_{level}hPa'

local_extr_name = 'local_extr_crit'

dist_m, our_level, level, x_unit, y_unit, u_unit, v_unit, time_unit, level_unit, crit, th, min_samples, eps, CS_points_th = get_init_params(data_type, json_file, print_info=True)

path_data = f'{path_init}/data'
if data_type == 'LoRes':
    path_data = f'{path_data}/LoRes'
    
DBSCAN_name_tracks = f'DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_level_{level}_smoothing_sigma_{sigma}'


if data_type == 'ERA5':
    #### 2026-08-19
    DBSCAN_name = f'DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_{region_name}_sigma_{sigma}'
    name_init = f'sigma_{sigma}_DBSCAN_{data_type}'   
elif data_type == 'GPN':
    name_init = f'sigma_{sigma}_DBSCAN_{data_type}_level_{level}'
    DBSCAN_name = f'DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_sigma_{sigma}'
elif data_type == 'SMP':
    name_init = f'sigma_{sigma}_DBSCAN_{data_type}_level_{level}'
    DBSCAN_name = f'DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_with_wspd_smoothing'

elif data_type == 'GLORYS':
    sigma = 0
    level = 15
    DBSCAN_name = f'DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_sigma_{sigma}'
    
    # ncfile = f'{path_dir_data}/ocean_eddies/DBSCAN_02-04-10_sigma_0/'
    name_init = f'sigma_{sigma}_DBSCAN_{data_type}_level_{level}'
elif data_type == 'ALT':
    sigma = 0
    level = 0
    DBSCAN_name = f'DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_sigma_{sigma}'
    name_init = f'sigma_{sigma}_DBSCAN_{data_type}_level_{level}'
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
    year = 2010
elif data_type == 'GLORYS' or data_type == 'ALT':
    crit_scale = 0.1
    star_size = 5
    year = 2023
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

    vmin = -127
    vmax = 127

elif data_type == 'GLORYS' or data_type == 'ALT':
    folder_name = f'ocean_eddies/{DBSCAN_name}'
    vmin = -0.00005
    vmax = 0.00005
else:
    folder_name = f'{data_type}/{DBSCAN_name}'

    vmin = -0.0004
    vmax = 0.0004







if data_type == 'ERA5':
    path_dir_raw = f'/storage/thalassa/users/vkoshkina/data/ERA5'
    ncfile = f'{path_dir_raw}/ERA5_lsm_cropped_NA_for_TC.nc'
    ground_ds = xr.open_dataset(f'{ncfile}')['lsm']
    
    ground = np.where(ground_ds > 0.5, 1, np.nan)
    
    # Сохраняем реальные координаты
    if 'longitude' in ground_ds.coords:
        xxx = ground_ds['longitude'].values
        yyy = ground_ds['latitude'].values
    elif 'lon' in ground_ds.coords:
        xxx = ground_ds['lon'].values
        yyy = ground_ds['lat'].values
    else:
        xxx = np.arange(len(ground_ds['lon']))
        yyy = np.arange(len(ground_ds['lat']))
    
    xx, yy = np.meshgrid(xxx, yyy)
    ground_ds.close()

elif data_type == 'GLORYS':
    path_dir_raw = f'/storage/thalassa/users/vkoshkina/data/ocean_eddies'
    ncfile = f'{path_dir_raw}/BarKara_glorys_aug2023.nc'
    
    ground_ds = xr.open_dataset(f'{ncfile}')['uo']
    ground = np.where(np.isnan(ground_ds[0,0]), 1, np.nan)
    x_unit = 'longitude'
    y_unit = 'latitude'
    
    xxx = ground_ds[x_unit]
    yyy = ground_ds[y_unit]
    xx, yy = np.meshgrid(xxx, yyy)
    ground_ds.close()
elif data_type == 'ALT':
    path_dir_raw = f'/storage/thalassa/users/vkoshkina/data/ocean_eddies'
    ncfile = f'{path_dir_raw}/BarKara_altimetry_aug2023.nc'
    
    ground_ds = xr.open_dataset(f'{ncfile}')['ugos']
    ground = np.where(np.isnan(ground_ds[0]), 1, np.nan)

    x_unit = 'longitude'
    y_unit = 'latitude'
    xxx = ground_ds[x_unit]
    yyy = ground_ds[y_unit]
    xx, yy = np.meshgrid(xxx, yyy)
    ground_ds.close()
else:

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

def plot_ground(ax, ground):
    mc = ax.contourf(xxx, yyy, ground, 
                cmap='grey', 
                alpha=0.07
                 )

def plot_ground(ax, ground):
    """
    Отрисовка подложки (суша/море)
    """
    # Получаем координаты lon и lat
    if 'longitude' in ground_ds.coords:
        lon_coords = ground_ds['longitude'].values
        lat_coords = ground_ds['latitude'].values
    elif 'lon' in ground_ds.coords:
        lon_coords = ground_ds['lon'].values
        lat_coords = ground_ds['lat'].values
    else:
        lon_coords = xxx
        lat_coords = yyy
    
    lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)
    
    mc = ax.contourf(lon_grid, lat_grid, ground, 
                     cmap='gray', 
                     alpha=0.3,
                     levels=[0.5, 1.5])  # Для бинарной маски суши


def plot_bounds(ax, CS, i, color='blue'):

    hw = int(np.round(CS['rad'][i]))*1.5
    
    y = CS['y'][i]
    x = CS['x'][i]

    p = mpatches.Rectangle(xy=(x - hw/2,y - hw/2), width=hw, height=hw, linewidth=0.5, 
                                     edgecolor='blue', facecolor='none',)
    ax.add_patch(p)

def time_plot_tracks(ax, dt_step, CS_tracks_list, our_xtime):
    """Функция отрисовки треков с реальными координатами lon/lat"""
    for CS in CS_tracks_list:
        # Проверяем, закончился ли трек до текущего времени
        if CS['datetime'].max() < our_xtime:
            continue
            
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
            ax.scatter(last_known_point['lon'], last_known_point['lat'],
                     s=star_size*1.2, c='lime', marker='o', zorder=20,
                     edgecolors='green', linewidths=1)
        
        # Отрисовываем весь трек до последней точки
        if len(track_points) > 1:
            # Цвет трека: красный если это последний шаг трека
            track_color = 'red' if is_last_point else 'black'
            ax.plot(track_points['lon'], track_points['lat'],
                   alpha=0.9, lw=1.5, c=track_color)
        
        # Отмечаем текущую/последнюю точку
        if exact_point:
            current_point = track_points[track_points['datetime'] == our_xtime].iloc[0]
            ax.scatter(current_point['lon'], current_point['lat'],
                     s=star_size, c='red' if is_last_point else 'blue',
                     marker='o', zorder=25,
                     edgecolors='white', linewidths=0.5)

#### 2026-07-16 - plot last 10 time points
def time_plot_tracks(ax, dt_step, CS_tracks_list, our_xtime, n_last_points=10):
    """Функция отрисовки треков с реальными координатами lon/lat"""
    for CS in CS_tracks_list:
        # Проверяем, закончился ли трек до текущего времени
        if CS['datetime'].max() < our_xtime:
            continue
            
        # Все точки трека до текущего времени (включительно)
        track_points = CS[CS['datetime'] <= our_xtime]
        
        if len(track_points) == 0:
            continue
            
        # Сортируем по времени
        track_points = track_points.sort_values('datetime')
        
        # Берем только последние n точек
        last_points = track_points.tail(n_last_points)
        
        # Проверяем есть ли точка точно на этом шаге
        exact_point = our_xtime in CS['datetime'].values
        is_last_point = (CS['datetime'].max() == our_xtime)
        
        # Если нет точки на текущем шаге, берем последнюю известную
        if not exact_point:
            last_known_point = track_points.iloc[-1]
            # Выделяем последнюю известную точку зеленым
            ax.scatter(last_known_point['lon'], last_known_point['lat'],
                     s=star_size*1.2, c='lime', marker='o', zorder=20,
                     edgecolors='green', linewidths=1)
        
        # Отрисовываем последние n точек
        if len(last_points) > 1:
            # Цвет трека: красный если это последний шаг трека
            track_color = 'red' if is_last_point else 'black'
            ax.plot(last_points['lon'], last_points['lat'],
                   alpha=0.9, lw=1.5, c=track_color)
        
        # Отмечаем текущую/последнюю точку
        if exact_point:
            current_point = track_points[track_points['datetime'] == our_xtime].iloc[0]
            ax.scatter(current_point['lon'], current_point['lat'],
                     s=star_size, c='red' if is_last_point else 'blue',
                     marker='o', zorder=25,
                     edgecolors='white', linewidths=0.5)


def time_plot_tracks_AC_C(ax, dt_step, CS_tracks_list_C, CS_tracks_list_AC, our_xtime, n_last_points=10):
    """Функция отрисовки треков с разделением по типам вихрей"""
    
    # Отрисовка треков циклонов (синие)
    for CS in CS_tracks_list_C:
        if CS['datetime'].max() < our_xtime:
            continue
            
        track_points = CS[CS['datetime'] <= our_xtime]
        if len(track_points) == 0:
            continue
            
        track_points = track_points.sort_values('datetime')
        last_points = track_points.tail(n_last_points)
        
        exact_point = our_xtime in CS['datetime'].values
        is_last_point = (CS['datetime'].max() == our_xtime)
        
        if not exact_point and len(track_points) > 0:
            last_known_point = track_points.iloc[-1]
            ax.scatter(last_known_point['lon'], last_known_point['lat'],
                     s=star_size*1.2, c='cyan', marker='o', zorder=20,
                     edgecolors='blue', linewidths=1)
        
        if len(last_points) > 1:
            track_color = 'blue' if is_last_point else 'darkblue'
            ax.plot(last_points['lon'], last_points['lat'],
                   alpha=0.9, lw=1.5, c=track_color)
        
        if exact_point:
            current_point = track_points[track_points['datetime'] == our_xtime].iloc[0]
            ax.scatter(current_point['lon'], current_point['lat'],
                     s=star_size, c='blue', marker='o', zorder=25,
                     edgecolors='white', linewidths=0.5)
    
    # Отрисовка треков антициклонов (красные)
    for CS in CS_tracks_list_AC:
        if CS['datetime'].max() < our_xtime:
            continue
            
        track_points = CS[CS['datetime'] <= our_xtime]
        if len(track_points) == 0:
            continue
            
        track_points = track_points.sort_values('datetime')
        last_points = track_points.tail(n_last_points)
        
        exact_point = our_xtime in CS['datetime'].values
        is_last_point = (CS['datetime'].max() == our_xtime)
        
        if not exact_point and len(track_points) > 0:
            last_known_point = track_points.iloc[-1]
            ax.scatter(last_known_point['lon'], last_known_point['lat'],
                     s=star_size*1.2, c='pink', marker='o', zorder=20,
                     edgecolors='red', linewidths=1)
        
        if len(last_points) > 1:
            track_color = 'red' if is_last_point else 'darkred'
            ax.plot(last_points['lon'], last_points['lat'],
                   alpha=0.9, lw=1.5, c=track_color)
        
        if exact_point:
            current_point = track_points[track_points['datetime'] == our_xtime].iloc[0]
            ax.scatter(current_point['lon'], current_point['lat'],
                     s=star_size, c='red', marker='o', zorder=25,
                     edgecolors='white', linewidths=0.5)

            
def plot_track_with_DBSCAN(ds, ds_speed, dt_step, crit_vals, local_extr, 
                         CS_tracks_list, 
                           # CS_tracks_list_C, CS_tracks_list_AC,
                           start_time, stop_time, idx, 
                         name_pics, save=True):

    dt_step_int = int(dt_step/3600)
    """Улучшенная функция визуализации с треками"""
    # Создаем папку для сохранения, если нужно
    if save:
        os.makedirs(f'{name_pics}', exist_ok=True)
    
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

        ### 2026-08-10 - для только Ц или АЦ
        time_plot_tracks(ax, dt_step, CS_tracks_list, ds[time_name].values[t], n_last_points=500)


        # time_plot_tracks_AC_C(ax, dt_step, CS_tracks_list_C, CS_tracks_list_AC, ds[time_name].values[t], n_last_points=500)
        
        # Инвертирование оси Y при необходимости
        # plt.gca().invert_yaxis()
        
        if save:
            plt.savefig(
                f"{name_pics}/track_{(idx + t):05d}.png", 
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


    
def load_local_extrema_nodes(file_path):
    """
    Загрузка локальных экстремумов из файла TempestExtremes Nodes
    Возвращает словарь с реальными координатами lon, lat
    """
    time_slices = {}
    current_time = None
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('\t'):
                parts = line.strip().split()
                if len(parts) >= 6:
                    try:
                        x = float(parts[0])      # индекс x
                        y = float(parts[1])      # индекс y
                        lon = float(parts[2])    # реальная долгота
                        lat = float(parts[3])    # реальная широта
                        wspd = float(parts[4])
                        r2d = float(parts[5])
                        if current_time is not None:
                            # Сохраняем lon, lat, r2d
                            time_slices[current_time].append((lon, lat, r2d))
                    except ValueError:
                        continue
            else:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        year = int(parts[0])
                        month = int(parts[1])
                        day = int(parts[2])
                        hour = int(parts[4])
                        current_time = datetime.datetime(year, month, day, hour)
                        time_slices[current_time] = []
                    except (ValueError, IndexError):
                        continue
    
    return time_slices

def load_and_merge_extrema(file_path_C, file_path_AC):
    """
    Загрузка и объединение экстремумов из двух файлов (C и AC)
    """
    # Загружаем оба файла
    extrema_C = load_local_extrema_nodes(file_path_C)
    extrema_AC = load_local_extrema_nodes(file_path_AC)
    
    # Объединяем словари
    merged_extrema = {}
    
    # Получаем все уникальные временные метки из обоих словарей
    all_times = set(extrema_C.keys()) | set(extrema_AC.keys())
    
    for time in all_times:
        merged_extrema[time] = []
        
        # Добавляем экстремумы из C
        if time in extrema_C:
            merged_extrema[time].extend(extrema_C[time])
        
        # Добавляем экстремумы из AC
        if time in extrema_AC:
            merged_extrema[time].extend(extrema_AC[time])
    
    return merged_extrema

def load_combined_extrema(file_path_C, file_path_AC=None):
    """
    Загрузка экстремумов из одного или двух файлов
    Если file_path_AC указан - объединяет оба файла
    """
    if file_path_AC is None:
        return load_local_extrema_nodes(file_path_C)
    
    # Загружаем оба файла
    extrema_C = load_local_extrema_nodes(file_path_C)
    extrema_AC = load_local_extrema_nodes(file_path_AC)
    
    # Объединяем
    merged = {}
    all_times = set(extrema_C.keys()) | set(extrema_AC.keys())
    
    for t in all_times:
        merged[t] = []
        if t in extrema_C:
            merged[t].extend(extrema_C[t])
        if t in extrema_AC:
            merged[t].extend(extrema_AC[t])
    
    return merged
    
def plot_crit(ax, ds, crit_vals, local_extr_nodes, t):
    """
    Визуализация полей и локальных экстремумов
    """
    # Временная метка текущего шага
    current_time = pd.to_datetime(ds[time_name][t].values)
    
    # Получаем координаты lon и lat из Dataset
    if 'longitude' in ds.coords:
        lon_coords = ds['longitude'].values
        lat_coords = ds['latitude'].values
    elif 'lon' in ds.coords:
        lon_coords = ds['lon'].values
        lat_coords = ds['lat'].values
    else:
        # Если координат нет, используем индексы
        lon_coords = np.arange(len(xxx))
        lat_coords = np.arange(len(yyy))
    
    # Создаем сетку координат
    lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)
    
    # Контурный график поля критерия
    mc = ax.contourf(lon_grid, lat_grid, crit_vals[t, our_level], 
                     cmap='PiYG', 
                     vmin=crit_scale*vmin, vmax=crit_scale*vmax,
                     levels=50)  # Добавляем уровни для лучшей визуализации
    
    plt.title(str(current_time)[:16])
    
    # Устанавливаем пределы по реальным координатам
    plt.xlim(lon_coords.min(), lon_coords.max())
    plt.ylim(lat_coords.min(), lat_coords.max())
    
#     # Инвертируем ось Y (для широты)
#     plt.gca().invert_yaxis()
    
    # Отрисовка локальных экстремумов для текущего времени
    if current_time in local_extr_nodes:
        # Используем реальные координаты lon, lat из файла
        lon_points = [point[0] for point in local_extr_nodes[current_time]]  # это уже lon
        lat_points = [point[1] for point in local_extr_nodes[current_time]]  # это уже lat
        values = [point[2] for point in local_extr_nodes[current_time]]
        
        Q_center = ax.scatter(lon_points, lat_points,
                              c='blue',
                              s=star_size*2,
                              marker='*',
                              alpha=1.,
                              label='center',
                              zorder=15,
                              edgecolors='white',
                              linewidths=0.5)
    
    plt.grid(linestyle=':', linewidth=0.5)
    plt.tick_params(axis='both', which='both', direction='in', labelsize=9)
    
    # Добавляем подписи осей
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')