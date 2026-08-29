from matplotlib import pyplot as plt
import pandas as pd 
import numpy as np
import math
import xarray as xr

import datetime
from datetime import timedelta

import scipy as sp
from scipy.ndimage import label, generate_binary_structure
from scipy import interpolate

import geopy.distance

import seaborn as sns

from shapely.geometry import Polygon
import matplotlib.patches as mpatches
from matplotlib import gridspec

from tqdm import tqdm

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from mpl_toolkits.axes_grid1 import make_axes_locatable

import cmaps

from pathlib import Path

import sys
import os

from step_of_tracking import *

tracking_type = 'tracking_local_2_phase'
CVS_speed = 'adv_speed'

pref_tracking = 'all_points_bound'

local_extr_name = 'local_extr_crit'

dist_m, our_level, level, x_unit, y_unit, u_unit, v_unit, time_unit, level_unit, crit, th, min_samples, eps, CS_points_th = get_init_params(data_type, json_file)

time_th = 3


level = 10

path_init = f'/storage/thalassa/users/vkoshkina'
path_dir_data_ds = f"{path_init}/data/SMP/DBSCAN_02-04-10_with_wspd_smoothing/2019"

ncfile = f'{path_dir_data_ds}/sigma_2_DBSCAN_SMP_level_{level}_2019-02-05.nc'
ds = xr.open_dataset(ncfile)

def load_tracks_for_EC_track(EC_track, path_tracks_dir, time_buffer_days=7):
    """
    Загружает треки CS_tracks_list_LoRes для конкретного EC трека
    в диапазоне (дата начала EC - time_buffer_days, дата конца EC)
    Использует только имена файлов для фильтрации (самая быстрая версия)
    """
    CS_tracks_for_EC = []
    
    # Получаем временные границы для EC трека
    EC_start = EC_track['datetime'].min()
    EC_end = EC_track['datetime'].max()
    start_date_bound = EC_start - pd.Timedelta(days=time_buffer_days)
    end_date_bound = EC_end
    
    # Определяем месяцы, которые нужно проверить
    months_to_check = set()
    current_date = start_date_bound
    while current_date <= end_date_bound:
        months_to_check.add((current_date.year, current_date.month))
        current_date += pd.Timedelta(days=1)
    
    # Проходим по всем нужным месяцам
    for year, month in months_to_check:
        month_path = f"{path_tracks_dir}/{year}-{month:02d}"
        if not os.path.exists(month_path):
            continue
            
        # Получаем список файлов
        all_files = list(Path(month_path).glob(f'*_track_*.csv'))
        
        for ifile in all_files:
            filename = ifile.name
            try:
                # Извлекаем дату начала трека из имени файла
                date_part = filename.split('_track_')[-1].replace('.csv', '')
                track_start_time = pd.to_datetime(date_part)
                
                # Быстрая проверка: если начало трека в нужном диапазоне
                if start_date_bound <= track_start_time <= end_date_bound:
                    df = pd.read_csv(ifile, parse_dates=['datetime'])
                    if len(df) > time_th:
                        df = df.drop(df.columns[0], axis=1)
                        df = df.dropna(how='any')
                        CS_tracks_for_EC.append(df)
                        
            except (ValueError, IndexError):
                # Если не удалось распарсить дату, пропускаем файл
                continue

    print(f'{len(CS_tracks_for_EC)} loaded for {start_date_bound}-{end_date_bound}')
    
    return CS_tracks_for_EC

def find_matching_tracks_for_EC(EC_track, NAAD_tracks, max_distance_km=5, max_time_diff=3, similar_steps=5):
    """
    Находит совпадающие треки для конкретного EC трека
    """
    # Предварительная обработка EC трека
    ec_data = EC_track.copy()
    ec_data['datetime'] = pd.to_datetime(ec_data['datetime'])
    ec_data_sorted = ec_data.sort_values('datetime')
    
    # Предварительно вычисляем временные границы для фильтрации
    min_time = ec_data_sorted['datetime'].min() - pd.Timedelta(hours=max_time_diff)
    max_time = ec_data_sorted['datetime'].max() + pd.Timedelta(hours=max_time_diff)
    
    matching_tracks = []
    
    def process_track(track_data):
        # Быстрая проверка по времени перед полной обработкой
        track_times = track_data['datetime']
        if not ((track_times >= min_time) & (track_times <= max_time)).any():
            return None, None
            
        track_data = track_data.copy()
        track_data['datetime'] = pd.to_datetime(track_data['datetime'])
        track_data_sorted = track_data.sort_values('datetime')

        # Используем merge_asof с предварительно отсортированными данными
        merged = pd.merge_asof(
            ec_data_sorted,
            track_data_sorted,
            on='datetime',
            direction='nearest',
            tolerance=pd.Timedelta(hours=max_time_diff),
            suffixes=('_ec', '_track')
        ).dropna()

        if len(merged) == 0:
            return None, None

        # Векторизованное вычисление расстояния
        x_diff = merged['x_ec'] - merged['x_track']
        y_diff = merged['y_ec'] - merged['y_track']
        distances = np.sqrt(x_diff**2 + y_diff**2)
        
        valid_count = np.sum(distances <= max_distance_km)
        
        if valid_count >= similar_steps:
            return track_data, np.mean(distances[distances <= max_distance_km]) * 6
        else:
            return None, None

    # Обрабатываем треки с предварительной фильтрацией
    for track in NAAD_tracks:
        matched_track, avg_distance = process_track(track)
        if matched_track is not None:
            matching_tracks.append((matched_track, avg_distance))
    
    return matching_tracks

def get_tracks_dist(track_real, TC, x_name='x', y_name='y'):
    """
    Вычисляет расстояния между двумя треками
    """
    # Фильтруем данные по совпадающим датам
    common_dates = np.intersect1d(track_real['datetime'].values, TC['datetime'].values)
    
    if len(common_dates) == 0:
        return []
    
    TC_for_real = TC[TC['datetime'].isin(common_dates)]
    track_real_for_TC = track_real[track_real['datetime'].isin(common_dates)]
    
    # Вычисляем евклидовы расстояния
    dist = []
    for i in range(len(TC_for_real)):
        x1, y1 = TC_for_real.iloc[i][x_name], TC_for_real.iloc[i][y_name]
        x2, y2 = track_real_for_TC.iloc[i][x_name], track_real_for_TC.iloc[i][y_name]
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        dist.append(distance)
    
    return dist

def save_merged_track(EC_track, matching_tracks, path_save, track_id):
    """
    Сохраняет объединенный трек с данными EC и совпадающих треков
    """
    if not matching_tracks:
        return
    
    # Создаем копию EC трека для объединения
    merged_track = EC_track.copy()
    
    for idx, (naad_track, distance) in enumerate(matching_tracks):
        # Добавляем данные NAAD трека
        naad_track_renamed = naad_track.rename(columns=lambda x: f"{x}_NAAD_{idx}" if x not in ['datetime'] else x)
        merged_track = pd.merge(merged_track, naad_track_renamed, on="datetime", how="outer")
    
    # Сортируем по времени
    merged_track = merged_track.sort_values("datetime").reset_index(drop=True)
    
    # Добавляем ID
    merged_track['Id'] = track_id + 1
    
    # Сохраняем
    CS_start = merged_track['datetime'].dt.strftime('%Y-%m-%dT%H').values[0]
    merged_track.to_csv(f'{path_save}/{(track_id+1):09d}_{CS_start}.csv', index=False)

def plot_EC_track_with_matches(EC_track, matching_tracks, path_save, track_id, data_type='LoRes', sigma=2):
    """
    Рисует и сохраняет график EC трека с совпадающими треками
    """
    if not matching_tracks:
        return
    
    fig = plt.figure(figsize=(5, 5), dpi=150)
    ax = fig.add_subplot(111)

    # Контур фона (если доступен ds)
    try:
        ax.contourf(np.where(ds['HGT'] > 15, 1, np.nan), cmap='Greys', alpha=0.7)
    except:
        pass
    
    # EC трек
    ax.plot(EC_track['x'], EC_track['y'], c='k', linewidth=2, label='manual track')
    
    # Совпадающие треки
    for idx, (naad_track, distance) in enumerate(matching_tracks):
        CS_start = naad_track['datetime'].dt.strftime('%Y-%m-%dT%H').values[0]
        ax.plot(naad_track['x'], naad_track['y'], 
                label=f'{data_type} {idx+1} ({int(distance)} km)')
    
    # Начальная и конечная точки EC трека
    ax.scatter(EC_track['x'].values[0], EC_track['y'].values[0], 
               c='g', s=20, zorder=10, label='manual start')
    ax.scatter(EC_track['x'].values[-1], EC_track['y'].values[-1], 
               c='r', s=20, zorder=10, label='manual end')
    
    # Настройки графика
    TC_start = EC_track['datetime'].dt.strftime('%Y-%m-%dT%H').values[0]
    ax.set_title(f'Track {track_id+1} at {TC_start}')
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    ax.legend(fontsize=7)
    
    # Сохраняем график
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    
    fig.savefig(f'{path_save}/{data_type}_{(track_id+1):09d}_sigma_{sigma}.png', 
                dpi=200, 
                bbox_inches="tight", 
                transparent=False)
    
    plt.close(fig)

def plot_EC_track_with_matches(EC_track, matching_tracks, path_save, track_id, data_type='LoRes', sigma=2):
    """
    Рисует и сохраняет график EC трека с совпадающими треками
    Добавляет длительность треков в часах в легенду
    """
    if not matching_tracks:
        return
    
    fig = plt.figure(figsize=(5, 5), dpi=150)
    ax = fig.add_subplot(111)

    # Контур фона (если доступен ds)
    try:
        ax.contourf(np.where(ds['HGT'] > 15, 1, np.nan), cmap='Greys', alpha=0.7)
    except:
        pass
    
    # Вычисляем длительность EC трека в часах (округляем до целых)
    ec_duration = int((EC_track['datetime'].max() - EC_track['datetime'].min()).total_seconds() / 3600)
    
    # EC трек
    ax.plot(EC_track['x'], EC_track['y'], c='k', linewidth=2, 
            label=f'manual ({ec_duration}h)')
    
    # Совпадающие треки
    for idx, (naad_track, distance) in enumerate(matching_tracks):
        # Вычисляем длительность NAAD трека в часах (округляем до целых)
        naad_duration = int((naad_track['datetime'].max() - naad_track['datetime'].min()).total_seconds() / 3600)
        
        ax.plot(naad_track['x'], naad_track['y'], 
                label=f'{data_type} {idx+1} ({int(distance)}km, {naad_duration}h)')
    
    # Начальная и конечная точки EC трека
    ax.scatter(EC_track['x'].values[0], EC_track['y'].values[0], 
               c='g', s=20, zorder=10, label='start')
    ax.scatter(EC_track['x'].values[-1], EC_track['y'].values[-1], 
               c='r', s=20, zorder=10, label='end')
    
    # Настройки графика
    TC_start = EC_track['datetime'].dt.strftime('%Y-%m-%dT%H').values[0]
    ax.set_title(f'Track {track_id+1} at {TC_start}')
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    ax.legend(fontsize=7)
    
    # Сохраняем график
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    
    fig.savefig(f'{path_save}/{data_type}_{(track_id+1):09d}_sigma_{sigma}.png', 
                dpi=200, 
                bbox_inches="tight", 
                transparent=False)
    
    plt.close(fig)

def load_season_tracks_EddyClicker(path_data, year, time_th=3):
    """
    Загружает все EddyClicker треки
    """
    CS_tracks_list = []
    name_NOAA = f'*.csv'
    ls = list(sorted(Path(f"{path_data}/").glob(name_NOAA)))

    if len(ls) != 0:
        for ifile in ls:
            df = pd.read_csv(ifile, parse_dates=['time'])
            
            # Переименовываем колонки
            df = df.rename(columns={
                'time': 'datetime',
                'pxc_ind': 'x',
                'pyc_ind': 'y'
            })
            
            if len(df) > time_th:
                CS_tracks_list.append(df)
    
    return CS_tracks_list

# Основной код
year = 2019
path_dir_data = f'{path_init}/data/'

DBSCAN_name = f'DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_with_uv_time_smoothing'
DBSCAN_name = f'DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_sigma_{sigma}'

folder = f'{path_dir_data}/{data_type}/{data_type}_tracks_sigma_{sigma}/{pref_tracking}/{tracking_type}_{CVS_speed}_{pref_tracking}'
path_tracks_dir = f'{folder}/tracks_{circ}'

# Загрузка EddyClicker треков
path_data_EC = f'{path_dir_data}/{data_type}/EddyClicker_tracks'
EddyClicker_tracks_all = load_season_tracks_EddyClicker(path_data_EC, year, time_th=3)

# Создаем папки для сохранения
merged_folder = f'{path_dir_data}/{data_type}/EC_tracks_merged_new/{data_type}_sigma_{sigma}/{tracking_type}_{CVS_speed}'
plot_folder = f'{path_dir_data}/{data_type}/pics/EC_tracks_comparison_new/{data_type}_sigma_{sigma}/{tracking_type}_{CVS_speed}'

if not os.path.exists(merged_folder):
    os.makedirs(merged_folder)
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)

# Обработка каждого EC трека
for track_id, EC_track in tqdm(enumerate(EddyClicker_tracks_all), desc="Processing EC tracks"):
    try:
        # Загружаем соответствующие NAAD треки
        NAAD_tracks = load_tracks_for_EC_track(EC_track, path_tracks_dir, time_buffer_days=7)
        
        # Находим совпадающие треки
        matching_tracks = find_matching_tracks_for_EC(EC_track, NAAD_tracks)
        
        if matching_tracks:
            # Сохраняем объединенный трек
            save_merged_track(EC_track, matching_tracks, merged_folder, track_id)
            
            # Рисуем и сохраняем график
            plot_EC_track_with_matches(EC_track, matching_tracks, plot_folder, track_id, data_type, sigma)
            
        else:
            print(f"No matches found for track {track_id+1}")
            
    except Exception as e:
        print(f"Error processing track {track_id+1}: {e}")
        continue

print("Processing completed!")