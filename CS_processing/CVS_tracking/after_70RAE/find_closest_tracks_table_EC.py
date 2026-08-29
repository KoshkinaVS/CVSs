import pandas as pd
import numpy as np
import math
import xarray as xr
from numpy import linalg as LA

import datetime
from datetime import timedelta

import scipy as sp

import json

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import DBSCAN
from skimage.feature import peak_local_max

from scipy.spatial import cKDTree

from pathlib import Path

import sys
import os

from typing import Dict, Callable
from functools import partial
import itertools

# Импорты для работы с координатами и расстояниями
from geopy.distance import great_circle, geodesic

path_init = f'/storage/thalassa/users/vkoshkina'
folder = 'scripts/CS_processing/CVS_tracking/after_70RAE/'

sys.path.insert(2, f'{path_init}/{folder}')

json_file = f'{path_init}/{folder}/tracking_init.json'

# Импорт функций из файла func_for_find_closest_tracks.py
from func_for_find_closest_tracks import (
    load_season_tracks, 
    load_season_tracks_NOAA, 
    find_matching_tracks_fast,
    get_tracks_dist,
    save_TC_merged,
    plot_closest_track_for_NOAA
)

def load_init_data(file_path):
    """
    Loads initialization data from a JSON file.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file {file_path}")
        return None

def get_init_params(data_type, json_file, print_info=False):
    init_data = load_init_data(json_file)

    if init_data is not None:
        min_samples = init_data['min_samples']
        eps = init_data['eps']
        CS_points_th = init_data['CS_points_th']
        crit = init_data['crit']
        th = init_data['th']
        x_unit = init_data['x_unit']
        y_unit = init_data['y_unit']

        if print_info:
            print(f'data type: {data_type}')
            print(f'min_samples={min_samples}, eps={eps}, CS_points_th={CS_points_th}')
    else:
        print('Error while reading init json')
    

    if data_type == 'HiRes':
        dist_m = 13897.18
        our_level = 0
        level = 12
        u_unit = 'ue'
        v_unit = 've'
        time_unit = 'Time'
        level_unit = 'interp_level'
        
    elif data_type == 'ERA5':
        dist_m = 111 * 1000 * 0.25
        our_level = 0
        level = 500
        u_unit = 'u'
        v_unit = 'v'
        time_unit = 'Time'
        level_unit = 'pressure_level'
        x_unit = 'longitude'
        y_unit = 'latitude'
        
    elif data_type == 'GPN':
        dist_m = 6000.0
        our_level = 0
        level = 22
        level = 12
        u_unit = 'ua'
        v_unit = 'va'
        time_unit = 'Time'
        level_unit = 'bottom_top'

    elif data_type == 'SMP':
        dist_m = 6000.0
        our_level = 0
        level = 10
        u_unit = 'ua'
        v_unit = 'va'
        time_unit = 'Time'
        level_unit = 'bottom_top'

    elif data_type == 'LoRes':   
        dist_m = 77824.23
        our_level = 0
        level = 12
        u_unit = 'ue'
        v_unit = 've'
        time_unit = 'Time'
        level_unit = 'interp_level'

    return dist_m, our_level, level, x_unit, y_unit, u_unit, v_unit, time_unit, level_unit, crit, th, min_samples, eps, CS_points_th

print('data type: ')
data_type = input() 

print('sigma: ')
sigma = int(input())

circ = 'C'

if data_type == 'GPN':
    dt_step = 3600 #seconds = 1 hour
else: 
    dt_step = 10800 #seconds = 3 hours

speed_level = 0 # from file in 12 level

dist_m, our_level, level, x_unit, y_unit, u_unit, v_unit, time_unit, level_unit, crit, th, min_samples, eps, CS_points_th = get_init_params(data_type, json_file)

time_name = time_unit

if sigma == 0 and data_type == 'LoRes':
    level_unit = 'bottom_top'

# Configuration parameters
tracking_types = ['tracking_local_2_phase', 
                  'tracking_global_only', 
                  'tracking_local_global',
                  # 'tracking_local_only', 
                 ]
speed_options = ['adv_speed', 
                 'no_speed', 'bg_speed', 'adv_bg_speed'
                ]
pref_tracking = 'all_points_bound'

local_extr_name = 'local_extr_crit'

time_th = 3
similar_steps = 3
max_time_diff = 3
max_distance_km = 3*dist_m/1000

if data_type == 'SMP':
    years = np.arange(2019, 2020)
    months = np.arange(1, 7, 1)
    folder_NOAA = 'SMP/EddyClicker_tracks'
    
elif data_type == 'LoRes':
    years = np.arange(2010, 2011)
    months = np.arange(1, 12, 1)
    folder_NOAA = f'{data_type}/EddyClicker_tracks_Egor_2010_params_r2d'


path_dir_data = f'{path_init}/data'
if data_type == 'LoRes':
    path_dir_data = f'{path_dir_data}/LoRes'

path_data = f'{path_init}/data'

# Глобальный словарь для хранения путей к файлам NAAD
NAAD_TRACK_PATHS = {}

def parse_datetime_safe(dt_series):
    """Безопасное преобразование в datetime с обработкой различных форматов"""
    if pd.api.types.is_datetime64_any_dtype(dt_series):
        return dt_series
    
    # Пробуем разные форматы дат
    date_formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%dT%H:%M:%S.%f',
        '%Y%m%d%H%M%S',
        '%Y-%m-%d'
    ]
    
    for fmt in date_formats:
        try:
            return pd.to_datetime(dt_series, format=fmt, errors='raise')
        except:
            continue
    
    # Если ни один формат не подошел, пробуем автоматическое определение
    try:
        return pd.to_datetime(dt_series, errors='coerce')
    except:
        print(f"Warning: Could not parse datetime series: {dt_series.head()}")
        return pd.Series([pd.NaT] * len(dt_series))

def load_season_tracks_with_paths(CS_tracks_list, year, months, path_data, time_th=3):
    """Загружает треки NAAD и сохраняет пути к файлам"""
    null_files = []
    
    for month in tqdm(months, desc='Loading NAAD data...'):
        month_dir = Path(f"{path_data}/{year}-{month:02d}")
        if not month_dir.exists():
            print(f"Directory does not exist: {month_dir}")
            continue
            
        ls = list(sorted(month_dir.glob(f'*_track_*.csv')))
        
        if len(ls) != 0:
            for ii, ifile in enumerate(ls):
                try:
                    df = pd.read_csv(ifile)
                    
                    # Проверяем наличие колонки datetime
                    if 'datetime' not in df.columns:
                        print(f"Warning: 'datetime' column not found in {ifile}")
                        # Пытаемся найти альтернативные колонки с датой
                        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                        if date_cols:
                            df = df.rename(columns={date_cols[0]: 'datetime'})
                            print(f"Renamed column '{date_cols[0]}' to 'datetime'")
                        else:
                            print(f"Skipping file {ifile} - no datetime column found")
                            null_files.append(ifile)
                            continue
                    
                    # Безопасное преобразование datetime
                    df['datetime'] = parse_datetime_safe(df['datetime'])
                    
                    # Удаляем строки с некорректными датами
                    df = df.dropna(subset=['datetime'])
                    
                    if len(df) > time_th: 
                        # Удаляем ненужные колонки
                        if 'Unnamed: 0' in df.columns:
                            df = df.drop('Unnamed: 0', axis=1)
                        df = df.dropna(how='any')
                        
                        # Сохраняем путь к файлу
                        file_id = f"{year}_{month:02d}_{ii}"
                        NAAD_TRACK_PATHS[file_id] = str(ifile)
                        df.attrs['file_path'] = str(ifile)  # Добавляем путь как атрибут
                        
                        CS_tracks_list.append(df)
                    else:
                        null_files.append(ifile)
                except Exception as e:
                    print(f"Error loading file {ifile}: {e}")
                    null_files.append(ifile)
        else:
            print(f"No track files found in {month_dir}")
    
    print(f'too short NAAD tracks in {year}: {len(null_files)}')
    
    return CS_tracks_list

def get_naad_track_path(naad_track):
    """Получает путь к файлу NAAD из трека"""
    if hasattr(naad_track, 'attrs') and 'file_path' in naad_track.attrs:
        return naad_track.attrs['file_path']
    else:
        # Пытаемся найти путь по содержимому трека
        for file_id, path in NAAD_TRACK_PATHS.items():
            # Проверяем, соответствует ли трек файлу
            try:
                df_check = pd.read_csv(path)
                df_check['datetime'] = parse_datetime_safe(df_check['datetime'])
                
                if len(df_check) == len(naad_track):
                    # Сравниваем первые несколько строк для проверки
                    sample_size = min(3, len(df_check))
                    if all(df_check['datetime'].head(sample_size) == naad_track['datetime'].head(sample_size)):
                        return path
            except Exception as e:
                continue
        return "Unknown"

def calculate_track_length(track):
    """Вычисляет длину трека в км"""
    if len(track) < 2:
        return 0
    
    total_distance = 0
    for i in range(len(track) - 1):
        coord1 = (track.iloc[i]['lat'], track.iloc[i]['lon'])
        coord2 = (track.iloc[i+1]['lat'], track.iloc[i+1]['lon'])
        distance = great_circle(coord1, coord2).km
        total_distance += distance
    
    return total_distance

def calculate_matching_percentage(noaa_track, naad_track):
    """Вычисляет процент совпадения по времени"""
    noaa_times = set(noaa_track['datetime'])
    naad_times = set(naad_track['datetime'])
    common_times = noaa_times.intersection(naad_times)
    
    if len(noaa_times) == 0:
        return 0
    
    return len(common_times) / len(noaa_times) * 100

def extract_naad_wind_info(naad_track):
    """Извлекает информацию о скорости ветра из трека NAAD"""
    wind_info = {
        'max_wind': None,
        'max_wind_time': None,
        'max_wind_lat': None,
        'max_wind_lon': None,
        'wind_column': None
    }
    
    possible_wind_columns = [
        'vort', 'vorticity', 'wind', 'windspeed', 'speed',
        'vort_max', 'max_vort', 'rel_vort', 'abs_vort'
    ]
    
    for col in naad_track.columns:
        col_lower = col.lower()
        for wind_keyword in possible_wind_columns:
            if wind_keyword in col_lower:
                if naad_track[col].notna().any():
                    wind_info['max_wind'] = naad_track[col].max()
                    max_idx = naad_track[col].idxmax()
                    wind_info['max_wind_time'] = naad_track.iloc[max_idx]['datetime']
                    wind_info['max_wind_lat'] = naad_track.iloc[max_idx]['lat']
                    wind_info['max_wind_lon'] = naad_track.iloc[max_idx]['lon']
                    wind_info['wind_column'] = col
                    return wind_info
    
    return wind_info

def create_noaa_naad_matching_table(TC_dict, output_path, tracking_type, CVS_speed, year, sigma):
    """Создает таблицу сопоставления треков NOAA с треками NAAD"""
    matching_data = []
    
    for key in TC_dict.keys():
        noaa_track = TC_dict[key]['NOAA_track']
        naad_tracks = TC_dict[key]['NAAD_tracks']
        distances = TC_dict[key]['dist_btwn_tracks']
        
        # Информация о скорости для NOAA
        noaa_max_wind = noaa_track['Wind'].max() if 'Wind' in noaa_track.columns and noaa_track['Wind'].notna().any() else None
        noaa_max_wind_time = None
        noaa_max_wind_lat = None
        noaa_max_wind_lon = None
        
        if noaa_max_wind is not None and 'Wind' in noaa_track.columns:
            max_wind_idx = noaa_track['Wind'].idxmax()
            if pd.notna(max_wind_idx):
                noaa_max_wind_time = noaa_track.iloc[max_wind_idx]['datetime']
                noaa_max_wind_lat = noaa_track.iloc[max_wind_idx]['lat']
                noaa_max_wind_lon = noaa_track.iloc[max_wind_idx]['lon']
        
        # Основная информация о треке NOAA
        noaa_info = {
            'year': year,
            'tracking_type': tracking_type,
            'CVS_speed': CVS_speed,
            'sigma': sigma,
            'noaa_id': key,
            'noaa_name': noaa_track['Name'].values[0] if 'Name' in noaa_track.columns and len(noaa_track['Name'].values) > 0 else 'Unknown',
            'noaa_start_time': noaa_track['datetime'].min(),
            'noaa_end_time': noaa_track['datetime'].max(),
            'noaa_duration_hours': len(noaa_track),
            'noaa_min_pressure': noaa_track['Pres'].min() if 'Pres' in noaa_track.columns and noaa_track['Pres'].notna().any() else None,
            'noaa_max_wind_kt': noaa_max_wind,
            'noaa_max_wind_time': noaa_max_wind_time,
            'noaa_max_wind_lat': noaa_max_wind_lat,
            'noaa_max_wind_lon': noaa_max_wind_lon,
            'matched_naad_tracks_count': len(naad_tracks)
        }
        
        # Если есть совпавшие треки NAAD
        if naad_tracks:
            for i, (naad_track, distance) in enumerate(zip(naad_tracks, distances)):
                # Информация о скорости для NAAD
                naad_wind_info = extract_naad_wind_info(naad_track)
                
                # Получаем путь к файлу NAAD
                naad_file_path = get_naad_track_path(naad_track)
                
                track_info = noaa_info.copy()
                track_info.update({
                    'naad_track_index': i,
                    'mean_distance_km': distance,
                    'naad_start_time': naad_track['datetime'].min(),
                    'naad_end_time': naad_track['datetime'].max(),
                    'naad_duration_hours': len(naad_track),
                    'naad_min_pressure': naad_track['slp'].min() if 'slp' in naad_track.columns and naad_track['slp'].notna().any() else None,
                    'naad_max_wind': naad_wind_info['max_wind'],
                    'naad_max_wind_time': naad_wind_info['max_wind_time'],
                    'naad_max_wind_lat': naad_wind_info['max_wind_lat'],
                    'naad_max_wind_lon': naad_wind_info['max_wind_lon'],
                    'naad_wind_column': naad_wind_info['wind_column'],
                    'naad_track_length_km': calculate_track_length(naad_track),
                    'matching_percentage': calculate_matching_percentage(noaa_track, naad_track),
                    'wind_difference_kt': noaa_max_wind - naad_wind_info['max_wind'] if noaa_max_wind is not None and naad_wind_info['max_wind'] is not None else None,
                    'naad_file_path': naad_file_path  # Добавляем путь к файлу
                })
                matching_data.append(track_info)
        else:
            # Если нет совпавших треков
            noaa_info.update({
                'naad_track_index': None,
                'mean_distance_km': None,
                'naad_start_time': None,
                'naad_end_time': None,
                'naad_duration_hours': None,
                'naad_min_pressure': None,
                'naad_max_wind': None,
                'naad_max_wind_time': None,
                'naad_max_wind_lat': None,
                'naad_max_wind_lon': None,
                'naad_wind_column': None,
                'naad_track_length_km': None,
                'matching_percentage': None,
                'wind_difference_kt': None,
                'naad_file_path': None  # Путь отсутствует
            })
            matching_data.append(noaa_info)
    
    # Создаем DataFrame
    df_matching = pd.DataFrame(matching_data)
    
    # Сохраняем таблицу
    os.makedirs(output_path, exist_ok=True)
    filename = f"noaa_naad_matching_{tracking_type}_{CVS_speed}_{year}_sigma_{sigma}.csv"
    filepath = os.path.join(output_path, filename)
    df_matching.to_csv(filepath, index=False)
    
    print(f"Таблица сохранена: {filepath}")
    print(f"Всего треков NOAA: {len(TC_dict)}")
    matched_count = len(df_matching[df_matching['matched_naad_tracks_count'] > 0])
    print(f"Треков с совпадениями: {matched_count}")
    
    return df_matching

def process_configuration(config):
    """Обрабатывает одну конфигурацию трекинга"""
    tracking_type, CVS_speed = config
    
    DBSCAN_name = f'DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_level_{level}_smoothing_sigma_{sigma}'
    folder_name = f'{data_type}_sigma_{sigma}/{tracking_type}_{CVS_speed}_{pref_tracking}'

#     path_tracks_dir = f'{path_data}/TC_tracks/{data_type}/{data_type}_sigma_{sigma}/tracks_{circ}_params_new/'
    
    path_tracks_dir = f'{path_data}/{data_type}/{data_type}_tracks_sigma_{sigma}/{pref_tracking}/{tracking_type}_{CVS_speed}_{pref_tracking}/tracks_{circ}'
    
    
    all_matching_dfs = []
    
    for year in years:
        print(f"Обработка {year} для {tracking_type} {CVS_speed}...")
        
        CS_tracks_list_NAAD = []
        CS_tracks_list_NOAA = []
        
        # Загружаем треки NAAD с путями
        CS_tracks_list_NAAD = load_season_tracks_with_paths(CS_tracks_list_NAAD, year, months, path_tracks_dir, time_th=time_th)
        
        # Загружаем треки NOAA с обработкой ошибок
        try:
            CS_tracks_list_NOAA = load_season_tracks_NOAA(CS_tracks_list_NOAA, f'{path_dir_data}/{folder_NOAA}', year, time_th=time_th)
        except Exception as e:
            print(f"Error loading NOAA tracks for {year}: {e}")
            continue
        
        if not CS_tracks_list_NOAA:
            print(f"No NOAA tracks found for {year}")
            continue
            
        # Создаем словарь для хранения результатов
        TC_dict = {}
        for CS_track in CS_tracks_list_NOAA:
            if 'Id' in CS_track.columns and len(CS_track['Id'].values) > 0:
                track_id = CS_track['Id'].values[0]
                TC_dict[track_id] = {
                    'NOAA_track': CS_track,
                    'NAAD_tracks': [],
                    'dist_btwn_tracks': []
                }
        
        if not TC_dict:
            print(f"No valid NOAA tracks with ID for {year}")
            continue
        
        # Находим совпадающие треки
        for key in tqdm(TC_dict.keys(), desc=f'Поиск совпадений {tracking_type} {CVS_speed} {year}'):
            try:
                CS_track_NOAA = TC_dict[key]['NOAA_track']
                NOAA_times = CS_track_NOAA['datetime']
                NAAD_tracks = [CS for CS in CS_tracks_list_NAAD if np.isin(CS['datetime'], NOAA_times).any()]
                TC_dict = find_matching_tracks_fast(TC_dict, key, CS_track_NOAA, NAAD_tracks, max_distance_km, max_time_diff, similar_steps)
            except Exception as e:
                print(f"Error finding matches for track {key}: {e}")
                continue
        
        # Вычисляем расстояния
        for key in tqdm(TC_dict.keys(), desc=f'Вычисление расстояний {tracking_type} {CVS_speed} {year}'):
            try:
                dist_list = []
                CS_track_NOAA = TC_dict[key]['NOAA_track']
                for idx in range(len(TC_dict[key]['NAAD_tracks'])):
                    track_NAAD = TC_dict[key]['NAAD_tracks'][idx]
                    if track_NAAD is not None:
                        # Вычисляем расстояние между треками
                        dist = get_tracks_dist(TC_dict[key]['NOAA_track'], track_NAAD)
                        dist_list.append(np.nanmean(dist))
                
                TC_dict[key]['dist_btwn_tracks'] = dist_list
            except Exception as e:
                print(f"Error calculating distances for track {key}: {e}")
                TC_dict[key]['dist_btwn_tracks'] = []
        
        # # Сохраняем объединенные треки
        # try:
        #     save_TC_merged(TC_dict, f'{path_data}/TC_tracks/NAAD_NOAA_tracks_merged', folder_name, NAAD_name='NOAA_track_NAAD')
        # except Exception as e:
        #     print(f"Error saving merged tracks: {e}")
        
        # Создаем таблицу сопоставления
        try:
            output_table_path = f'{path_data}/TC_tracks/{data_type}_EC_matching_tables'
            df_matching = create_noaa_naad_matching_table(
                TC_dict, output_table_path, tracking_type, CVS_speed, year, sigma
            )
            all_matching_dfs.append(df_matching)
        except Exception as e:
            print(f"Error creating matching table for {year}: {e}")
        
        # # Строим графики (опционально)
        # try:
        #     for idx, key in tqdm(enumerate(TC_dict.keys()), desc=f'Построение графиков {tracking_type} {CVS_speed} {year}'):
        #         plot_closest_track_for_NOAA(TC_dict, key, 'NAAD_tracks', 
        #                                   f'{path_data}/TC_tracks/pics_diff_NAAD_NOAA_{data_type}', 
        #                                   folder_name, data_type, sigma=sigma)
        # except Exception as e:
        #     print(f"Error plotting tracks for {year}: {e}")
    
    return all_matching_dfs

def create_summary_table(all_matching_dfs, output_path):
    """Создает сводную таблицу по всем конфигурациям"""
    if not all_matching_dfs:
        print("Нет данных для создания сводной таблицы")
        return None, None
    
    # Объединяем все таблицы
    summary_df = pd.concat(all_matching_dfs, ignore_index=True)
    
    # Сохраняем сводную таблицу
    summary_filename = "noaa_naad_matching_summary_full.csv"
    summary_filepath = os.path.join(output_path, summary_filename)
    summary_df.to_csv(summary_filepath, index=False)
    
    # Создаем сводную статистику
    stats = summary_df.groupby(['tracking_type', 'CVS_speed', 'sigma']).agg({
        'noaa_id': 'count',
        'matched_naad_tracks_count': 'sum',
        'mean_distance_km': 'mean',
        'matching_percentage': 'mean',
        'noaa_max_wind_kt': 'mean',
        'naad_max_wind': 'mean',
        'wind_difference_kt': 'mean'
    }).round(2)
    
    stats = stats.rename(columns={
        'noaa_id': 'total_noaa_tracks',
        'matched_naad_tracks_count': 'total_matches',
        'noaa_max_wind_kt': 'avg_noaa_max_wind_kt',
        'naad_max_wind': 'avg_naad_max_wind',
        'wind_difference_kt': 'avg_wind_difference_kt'
    })
    
    stats_filename = "noaa_naad_matching_stats_full.csv"
    stats_filepath = os.path.join(output_path, stats_filename)
    stats.to_csv(stats_filepath)
    
    print(f"Сводная таблица сохранена: {summary_filepath}")
    print(f"Статистика сохранена: {stats_filepath}")
    
    return summary_df, stats

def load_eddyclicker_tracks(CS_tracks_list, year, path_data, time_th=3):
    """Загружает треки EddyClicker с индексами пикселей"""
    null_files = []
    
    eddyclicker_dir = Path(path_data)
    print(f"Loading EddyClicker tracks from: {eddyclicker_dir}")
    
    if not eddyclicker_dir.exists():
        print(f"Directory does not exist: {eddyclicker_dir}")
        return CS_tracks_list
    
    # Ищем файлы EddyClicker
    ls = list(sorted(eddyclicker_dir.glob('*.csv')))
    print(f"Found {len(ls)} EddyClicker track files")
    
    for ii, ifile in tqdm(enumerate(ls), desc='Loading EddyClicker data...', total=len(ls)):
        try:
            df = pd.read_csv(ifile)
            
            # Проверяем наличие колонки time
            if 'time' not in df.columns:
                print(f"Warning: No 'time' column in {ifile}")
                null_files.append(ifile)
                continue
            
            # Переименовываем time в datetime
            df = df.rename(columns={'time': 'datetime'})
            
            # Преобразуем время
            df['datetime'] = parse_datetime_safe(df['datetime'])
            df = df.dropna(subset=['datetime'])
            
            # Фильтруем по году если указан
            if year is not None:
                df['year'] = df['datetime'].dt.year
                df = df[df['year'] == year]
                if len(df) == 0:
                    continue
            
            # Проверяем наличие индексов пикселей
            if 'pxc_ind' not in df.columns or 'pyc_ind' not in df.columns:
                print(f"Warning: No pixel indices in {ifile}")
                null_files.append(ifile)
                continue
            
            # Используем pxc_ind и pyc_ind как x и y для сравнения с SMP
            # Переименовываем для единообразия
            df = df.rename(columns={
                'pxc_ind': 'x',
                'pyc_ind': 'y'
            })
            
            # Добавляем идентификатор трека
            track_id = ifile.stem
            df['track_id'] = track_id
            df['eddyclicker_id'] = track_id
            
            # Сохраняем оригинальные индексы на всякий случай
            for i in range(1, 4):
                if f'px{i}_ind' in df.columns:
                    df[f'px{i}_ind_orig'] = df[f'px{i}_ind']
                if f'py{i}_ind' in df.columns:
                    df[f'py{i}_ind_orig'] = df[f'py{i}_ind']
            
            # Проверяем минимальную длину трека
            if len(df) > time_th:
                # Удаляем дубликаты времени
                df = df.drop_duplicates(subset=['datetime'])
                
                # Сохраняем путь к файлу
                df.attrs['file_path'] = str(ifile)
                
                CS_tracks_list.append(df)
            else:
                null_files.append(ifile)
                
        except Exception as e:
            print(f"Error loading file {ifile}: {e}")
            null_files.append(ifile)
    
    print(f'too short EddyClicker tracks: {len(null_files)}')
    
    return CS_tracks_list


def load_season_tracks_SMP(CS_tracks_list, year, months, path_data, time_th=3):
    """Загружает треки SMP (аналогично NAAD)"""
    null_files = []
    
    for month in tqdm(months, desc='Loading SMP data...'):
        month_dir = Path(f"{path_data}/{year}-{month:02d}")
        if not month_dir.exists():
            print(f"Directory does not exist: {month_dir}")
            continue
            
        ls = list(sorted(month_dir.glob(f'*_track_*.csv')))
        
        if len(ls) != 0:
            for ii, ifile in enumerate(ls):
                try:
                    df = pd.read_csv(ifile)
                    
                    # Проверяем наличие datetime и координат x, y
                    if 'datetime' not in df.columns:
                        print(f"Warning: 'datetime' column not found in {ifile}")
                        continue
                    
                    if 'x' not in df.columns or 'y' not in df.columns:
                        print(f"Warning: 'x' or 'y' columns not found in {ifile}")
                        continue
                    
                    # Безопасное преобразование datetime
                    df['datetime'] = parse_datetime_safe(df['datetime'])
                    df = df.dropna(subset=['datetime', 'x', 'y'])
                    
                    if len(df) > time_th: 
                        # Удаляем ненужные колонки
                        if 'Unnamed: 0' in df.columns:
                            df = df.drop('Unnamed: 0', axis=1)
                        
                        # Добавляем идентификатор трека
                        df['smp_track_id'] = f"{year}_{month:02d}_{ii}"
                        df.attrs['file_path'] = str(ifile)
                        
                        CS_tracks_list.append(df)
                    else:
                        null_files.append(ifile)
                except Exception as e:
                    print(f"Error loading file {ifile}: {e}")
                    null_files.append(ifile)
        else:
            print(f"No SMP track files found in {month_dir}")
    
    print(f'too short SMP tracks in {year}: {len(null_files)}')
    
    return CS_tracks_list


def process_configuration(config):
    """Обрабатывает одну конфигурацию трекинга"""
    tracking_type, CVS_speed = config
    
    DBSCAN_name = f'DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_level_{level}_smoothing_sigma_{sigma}'
    
    # Путь к трекам SMP
    path_tracks_dir = f'{path_data}/{data_type}/{data_type}_tracks_sigma_{sigma}/{pref_tracking}/{tracking_type}_{CVS_speed}_{pref_tracking}/tracks_{circ}'
    
    all_matching_dfs = []
    
    for year in years:
        print(f"Обработка {year} для {tracking_type} {CVS_speed}...")
        
        SMP_tracks_list = []
        EddyClicker_tracks_list = []
        
        # Загружаем треки SMP (аналогично NAAD)
        SMP_tracks_list = load_season_tracks_SMP(SMP_tracks_list, year, months, path_tracks_dir, time_th=time_th)
        
        # Загружаем треки EddyClicker
        eddyclicker_path = f'{path_dir_data}/{folder_NOAA}'
        EddyClicker_tracks_list = load_eddyclicker_tracks(EddyClicker_tracks_list, year, eddyclicker_path, time_th=time_th)
        
        if not EddyClicker_tracks_list:
            print(f"No EddyClicker tracks found for {year}")
            continue
            
        # Создаем словарь для хранения результатов
        EC_dict = {}
        for EC_track in EddyClicker_tracks_list:
            if 'eddyclicker_id' in EC_track.columns and len(EC_track['eddyclicker_id'].values) > 0:
                track_id = EC_track['eddyclicker_id'].values[0]
                EC_dict[track_id] = {
                    'EddyClicker_track': EC_track,
                    'SMP_tracks': [],
                    'dist_btwn_tracks': []
                }
        
        if not EC_dict:
            print(f"No valid EddyClicker tracks with ID for {year}")
            continue
        
        # Находим совпадающие треки
        for key in tqdm(EC_dict.keys(), desc=f'Поиск совпадений {tracking_type} {CVS_speed} {year}'):
            try:
                EC_track = EC_dict[key]['EddyClicker_track']
                EC_times = EC_track['datetime']
                
                # Ищем SMP треки с совпадающими временами
                SMP_matching_tracks = []
                for SMP_track in SMP_tracks_list:
                    # Проверяем пересечение по времени
                    common_times = set(EC_times).intersection(set(SMP_track['datetime']))
                    if len(common_times) >= similar_steps:
                        SMP_matching_tracks.append(SMP_track)
                
                # Используем существующую функцию или адаптируем для сравнения индексов
                EC_dict = find_matching_tracks_fast(
                    EC_dict, key, EC_track, SMP_matching_tracks, 
                    max_distance_km, max_time_diff, similar_steps
                )
            except Exception as e:
                print(f"Error finding matches for track {key}: {e}")
                continue
        
        # Вычисляем расстояния между треками (в пикселях/индексах)
        for key in tqdm(EC_dict.keys(), desc=f'Вычисление расстояний {tracking_type} {CVS_speed} {year}'):
            try:
                dist_list = []
                EC_track = EC_dict[key]['EddyClicker_track']
                for idx in range(len(EC_dict[key]['SMP_tracks'])):
                    SMP_track = EC_dict[key]['SMP_tracks'][idx]
                    if SMP_track is not None:
                        # Вычисляем евклидово расстояние между индексами
                        # Находим общие временные точки
                        common_times = set(EC_track['datetime']).intersection(set(SMP_track['datetime']))
                        if len(common_times) > 0:
                            distances = []
                            for time in common_times:
                                ec_point = EC_track[EC_track['datetime'] == time]
                                smp_point = SMP_track[SMP_track['datetime'] == time]
                                
                                if len(ec_point) > 0 and len(smp_point) > 0:
                                    # Евклидово расстояние между индексами
                                    dx = ec_point.iloc[0]['x'] - smp_point.iloc[0]['x']
                                    dy = ec_point.iloc[0]['y'] - smp_point.iloc[0]['y']
                                    dist = np.sqrt(dx**2 + dy**2)
                                    distances.append(dist)
                            
                            if distances:
                                dist_list.append(np.mean(distances))
                
                EC_dict[key]['dist_btwn_tracks'] = dist_list
            except Exception as e:
                print(f"Error calculating distances for track {key}: {e}")
                EC_dict[key]['dist_btwn_tracks'] = []
        
        # Создаем таблицу сопоставления
        try:
            output_table_path = f'{path_data}/TC_tracks/{data_type}_EC_matching_tables'
            df_matching = create_eddyclicker_smp_matching_table(
                EC_dict, output_table_path, tracking_type, CVS_speed, year, sigma
            )
            all_matching_dfs.append(df_matching)
        except Exception as e:
            print(f"Error creating matching table for {year}: {e}")
    
    return all_matching_dfs


def find_matching_tracks_indices(EC_dict, key, EC_track, SMP_tracks_list, max_distance_pixels, max_time_diff, similar_steps):
    """
    Находит треки SMP, близкие к треку EddyClicker по индексам пикселей
    
    Args:
        max_distance_pixels: максимальное расстояние в пикселях
        max_time_diff: максимальная разница во времени в часах
        similar_steps: минимальное количество совпадающих временных шагов
    """
    EC_track_times = EC_track['datetime'].values
    
    for SMP_track in SMP_tracks_list:
        try:
            # Находим пересечение по времени
            SMP_track_times = SMP_track['datetime'].values
            
            # Создаем массивы для сравнения времени
            EC_times_np = np.array([pd.Timestamp(t).to_pydatetime() for t in EC_track_times])
            SMP_times_np = np.array([pd.Timestamp(t).to_pydatetime() for t in SMP_track_times])
            
            # Находим индексы близких по времени точек
            matching_indices = []
            for i, ec_time in enumerate(EC_times_np):
                time_diffs = np.abs((SMP_times_np - ec_time).astype('timedelta64[h]').astype(int))
                close_indices = np.where(time_diffs <= max_time_diff)[0]
                
                if len(close_indices) > 0:
                    # Берем ближайшую по времени точку
                    closest_idx = close_indices[np.argmin(time_diffs[close_indices])]
                    
                    # Проверяем расстояние по индексам
                    ec_x = EC_track.iloc[i]['x']
                    ec_y = EC_track.iloc[i]['y']
                    
                    smp_x = SMP_track.iloc[closest_idx]['x']
                    smp_y = SMP_track.iloc[closest_idx]['y']
                    
                    # Евклидово расстояние в пикселях
                    distance = np.sqrt((ec_x - smp_x)**2 + (ec_y - smp_y)**2)
                    
                    if distance <= max_distance_pixels:
                        matching_indices.append((i, closest_idx, distance))
            
            # Если достаточно совпадающих шагов
            if len(matching_indices) >= similar_steps:
                # Вычисляем среднее расстояние
                distances = [dist for _, _, dist in matching_indices]
                avg_distance = np.mean(distances)
                
                # Добавляем трек в словарь
                EC_dict[key]['SMP_tracks'].append(SMP_track)
                EC_dict[key]['dist_btwn_tracks'].append(avg_distance)
                
        except Exception as e:
            print(f"Error comparing tracks: {e}")
            continue
    
    return EC_dict


def calculate_distance_indices(EC_track, SMP_track, common_times=None):
    """
    Вычисляет расстояние между треками на основе индексов пикселей
    """
    if common_times is None:
        # Находим общие временные точки
        EC_times = set(EC_track['datetime'])
        SMP_times = set(SMP_track['datetime'])
        common_times = EC_times.intersection(SMP_times)
    
    if len(common_times) == 0:
        return None
    
    distances = []
    for time in common_times:
        # Получаем точки для данного времени
        ec_point = EC_track[EC_track['datetime'] == time]
        smp_point = SMP_track[SMP_track['datetime'] == time]
        
        if len(ec_point) > 0 and len(smp_point) > 0:
            ec_x = ec_point.iloc[0]['x']
            ec_y = ec_point.iloc[0]['y']
            
            smp_x = smp_point.iloc[0]['x']
            smp_y = smp_point.iloc[0]['y']
            
            # Евклидово расстояние
            dist = np.sqrt((ec_x - smp_x)**2 + (ec_y - smp_y)**2)
            distances.append(dist)
    
    if distances:
        return np.mean(distances)
    return None


def find_best_matching_track(EC_track, SMP_tracks_list, max_distance_pixels=10, min_common_steps=3):
    """
    Находит лучший совпадающий трек SMP для заданного трека EddyClicker
    """
    best_track = None
    best_distance = float('inf')
    best_common_steps = 0
    
    for SMP_track in SMP_tracks_list:
        try:
            # Находим общие временные точки
            EC_times = set(EC_track['datetime'])
            SMP_times = set(SMP_track['datetime'])
            common_times = EC_times.intersection(SMP_times)
            
            if len(common_times) < min_common_steps:
                continue
            
            # Вычисляем расстояние для общих временных точек
            distance = calculate_distance_indices(EC_track, SMP_track, common_times)
            
            if distance is not None and distance <= max_distance_pixels:
                # Оцениваем качество сопоставления
                quality_score = distance / len(common_times)
                
                if quality_score < best_distance / max(best_common_steps, 1):
                    best_track = SMP_track
                    best_distance = distance
                    best_common_steps = len(common_times)
                    
        except Exception as e:
            continue
    
    return best_track, best_distance, best_common_steps


# Затем в process_configuration замените вызов find_matching_tracks_fast:

def process_configuration(config):
    """Обрабатывает одну конфигурацию трекинга"""
    tracking_type, CVS_speed = config
    
    DBSCAN_name = f'DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_level_{level}_smoothing_sigma_{sigma}'
    
    # Путь к трекам SMP
    path_tracks_dir = f'{path_dir_data}/{data_type}/{data_type}_tracks_sigma_{sigma}/{pref_tracking}/{tracking_type}_{CVS_speed}_{pref_tracking}/tracks_{circ}'

    if data_type == 'LoRes':

        path_tracks_dir = '/storage/thalassa/users/vkoshkina/data/TC_tracks/LoRes/LoRes_sigma_2/tracks_C_params_new'
        # path_tracks_dir = f'{path_dir_data}/{data_type}/{data_type}_tracks_sigma_{sigma}/update_2026_wspd/{tracking_type}_{CVS_speed}_{pref_tracking}/tracks_{circ}'
    
    all_matching_dfs = []
    
    for year in years:
        print(f"Обработка {year} для {tracking_type} {CVS_speed}...")
        
        SMP_tracks_list = []
        EddyClicker_tracks_list = []
        
        # Загружаем треки SMP
        SMP_tracks_list = load_season_tracks_SMP(SMP_tracks_list, year, months, path_tracks_dir, time_th=time_th)
        
        # Загружаем треки EddyClicker
        eddyclicker_path = f'{path_dir_data}/{folder_NOAA}'
        EddyClicker_tracks_list = load_eddyclicker_tracks(EddyClicker_tracks_list, year, eddyclicker_path, time_th=time_th)
        
        if not EddyClicker_tracks_list:
            print(f"No EddyClicker tracks found for {year}")
            continue
        
        print(f"Загружено треков EddyClicker: {len(EddyClicker_tracks_list)}")
        print(f"Загружено треков SMP: {len(SMP_tracks_list)}")
            
        # Создаем словарь для хранения результатов
        EC_dict = {}
        for EC_track in EddyClicker_tracks_list:
            if 'eddyclicker_id' in EC_track.columns and len(EC_track['eddyclicker_id'].values) > 0:
                track_id = EC_track['eddyclicker_id'].values[0]
                EC_dict[track_id] = {
                    'EddyClicker_track': EC_track,
                    'SMP_tracks': [],
                    'dist_btwn_tracks': []
                }
        
        if not EC_dict:
            print(f"No valid EddyClicker tracks with ID for {year}")
            continue
        
        # Параметры для сопоставления (в пикселях, а не км)
        max_distance_pixels = 5  # Максимальное расстояние в пикселях
        max_time_diff_hours = 3   # Максимальная разница во времени
        min_common_steps = 3      # Минимальное количество совпадающих шагов
        
        # Находим совпадающие треки
        for key in tqdm(EC_dict.keys(), desc=f'Поиск совпадений {tracking_type} {CVS_speed} {year}'):
            try:
                EC_track = EC_dict[key]['EddyClicker_track']
                EC_times = EC_track['datetime']
                
                # Ищем SMP треки с совпадающими временами
                SMP_matching_tracks = []
                for SMP_track in SMP_tracks_list:
                    # Проверяем пересечение по времени
                    common_times = set(EC_times).intersection(set(SMP_track['datetime']))
                    if len(common_times) >= min_common_steps:
                        SMP_matching_tracks.append(SMP_track)
                
                if SMP_matching_tracks:
                    # Используем адаптированную функцию для сравнения индексов
                    EC_dict = find_matching_tracks_indices(
                        EC_dict, key, EC_track, SMP_matching_tracks, 
                        max_distance_pixels, max_time_diff_hours, min_common_steps
                    )
                    
                    # Или альтернативный подход:
                    # best_track, best_distance, common_steps = find_best_matching_track(
                    #     EC_track, SMP_matching_tracks, 
                    #     max_distance_pixels, min_common_steps
                    # )
                    # 
                    # if best_track is not None:
                    #     EC_dict[key]['SMP_tracks'].append(best_track)
                    #     EC_dict[key]['dist_btwn_tracks'].append(best_distance)
                    
            except Exception as e:
                print(f"Error finding matches for track {key}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Если использовали find_best_matching_track, то расстояния уже вычислены
        # Если использовали find_matching_tracks_indices, расстояния уже в словаре
        
        # Создаем таблицу сопоставления
        try:
            output_table_path = f'{path_data}/TC_tracks/{data_type}_EC_matching_tables'
            df_matching = create_eddyclicker_smp_matching_table(
                EC_dict, output_table_path, tracking_type, CVS_speed, year, sigma
            )
            all_matching_dfs.append(df_matching)
            
            # Выводим статистику для этого года
            print(f"\nСтатистика для {year} ({tracking_type}, {CVS_speed}):")
            print(f"Всего треков EddyClicker: {len(EC_dict)}")
            matched_count = sum(len(EC_dict[key]['SMP_tracks']) > 0 for key in EC_dict)
            print(f"Треков с совпадениями: {matched_count}")
            print(f"Процент сопоставления: {matched_count/len(EC_dict)*100:.1f}%")
            
        except Exception as e:
            print(f"Error creating matching table for {year}: {e}")
            import traceback
            traceback.print_exc()
    
    return all_matching_dfs


def create_eddyclicker_smp_matching_table(EC_dict, output_path, tracking_type, CVS_speed, year, sigma):
    """Создает таблицу сопоставления треков EddyClicker с треками SMP"""
    matching_data = []
    
    for key in EC_dict.keys():
        ec_track = EC_dict[key]['EddyClicker_track']
        smp_tracks = EC_dict[key]['SMP_tracks']
        distances = EC_dict[key]['dist_btwn_tracks']
        
        # Основная информация о треке EddyClicker
        ec_info = {
            'year': year,
            'tracking_type': tracking_type,
            'CVS_speed': CVS_speed,
            'sigma': sigma,
            'eddyclicker_id': key,
            'ec_start_time': ec_track['datetime'].min(),
            'ec_end_time': ec_track['datetime'].max(),
            'ec_duration_hours': len(ec_track),
            'matched_smp_tracks_count': len(smp_tracks)
        }
        
        # Если есть совпавшие треки SMP
        if smp_tracks:
            for i, (smp_track, distance) in enumerate(zip(smp_tracks, distances)):
                track_info = ec_info.copy()
                track_info.update({
                    'smp_track_index': i,
                    'mean_distance_pixels': distance,  # Расстояние в пикселях/индексах
                    'smp_start_time': smp_track['datetime'].min(),
                    'smp_end_time': smp_track['datetime'].max(),
                    'smp_duration_hours': len(smp_track),
                    'matching_percentage': calculate_matching_percentage(ec_track, smp_track),
                    'ec_file_path': ec_track.attrs.get('file_path', 'Unknown'),
                    'smp_file_path': smp_track.attrs.get('file_path', 'Unknown')
                })
                matching_data.append(track_info)
        else:
            # Если нет совпавших треков
            ec_info.update({
                'smp_track_index': None,
                'mean_distance_pixels': None,
                'smp_start_time': None,
                'smp_end_time': None,
                'smp_duration_hours': None,
                'matching_percentage': None,
                'ec_file_path': ec_track.attrs.get('file_path', 'Unknown'),
                'smp_file_path': None
            })
            matching_data.append(ec_info)
    
    # Создаем DataFrame
    df_matching = pd.DataFrame(matching_data)
    
    # Сохраняем таблицу
    os.makedirs(output_path, exist_ok=True)
    filename = f"eddyclicker_smp_matching_{tracking_type}_{CVS_speed}_{year}_sigma_{sigma}.csv"
    filepath = os.path.join(output_path, filename)
    df_matching.to_csv(filepath, index=False)
    
    print(f"Таблица сохранена: {filepath}")
    print(f"Всего треков EddyClicker: {len(EC_dict)}")
    matched_count = len(df_matching[df_matching['matched_smp_tracks_count'] > 0])
    print(f"Треков с совпадениями: {matched_count}")
    
    return df_matching

def main():
    """Основная функция для запуска обработки"""
    all_matching_dfs = []
    
    # Генерируем все конфигурации
    configurations = list(itertools.product(tracking_types, speed_options))
    print(f"Всего конфигураций для обработки: {len(configurations)}")
    
    # Обрабатываем конфигурации и собираем данные
    for i, config in tqdm(enumerate(configurations), total=len(configurations)):
        tracking_type, CVS_speed = config
        print(f"\n{'='*60}")
        print(f"Обработка конфигурации {i+1}/{len(configurations)}: {tracking_type} - {CVS_speed}")
        print(f"{'='*60}")
        
        try:
            result_dfs = process_configuration(config)
            if result_dfs:
                all_matching_dfs.extend(result_dfs)
        except Exception as e:
            print(f"Ошибка при обработке конфигурации {config}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Создаем сводную таблицу
    if all_matching_dfs:
        output_path = f'{path_data}/TC_tracks/{data_type}_EC_matching_tables'
        summary_df, stats = create_summary_table(all_matching_dfs, output_path)
        
        # Выводим краткую статистику
        print("\n" + "="*60)
        print("СВОДНАЯ СТАТИСТИКА СОПОСТАВЛЕНИЯ ТРЕКОВ EDDYCLICKER И SMP")
        print("="*60)
        
        if stats is not None:
            print(stats)
            
            # Дополнительная статистика
            print("\nДополнительная статистика:")
            print(f"Всего обработано конфигураций: {len(configurations)}")
            print(f"Успешно обработано конфигураций: {len(set(summary_df['tracking_type'] + '_' + summary_df['CVS_speed']))}")
            print(f"Всего треков EddyClicker: {len(summary_df)}")
            
            matched_df = summary_df[summary_df['matched_smp_tracks_count'] > 0]
            print(f"Треков с совпадениями: {len(matched_df)}")
            print(f"Процент сопоставления: {len(matched_df)/len(summary_df)*100:.1f}%")
            
            if len(matched_df) > 0:
                avg_distance = matched_df['mean_distance_pixels'].mean()
                print(f"Среднее расстояние (пиксели): {avg_distance:.2f}")
                print(f"Средний процент совпадения: {matched_df['matching_percentage'].mean():.1f}%")
    else:
        print("Нет данных для создания сводной таблицы")


def create_summary_table(all_matching_dfs, output_path):
    """Создает сводную таблицу по всем конфигурациям"""
    if not all_matching_dfs:
        print("Нет данных для создания сводной таблицы")
        return None, None
    
    # Объединяем все таблицы
    summary_df = pd.concat(all_matching_dfs, ignore_index=True)
    
    # Сохраняем сводную таблицу
    os.makedirs(output_path, exist_ok=True)
    summary_filename = f"eddyclicker_smp_matching_summary_{data_type}_sigma_{sigma}.csv"
    summary_filepath = os.path.join(output_path, summary_filename)
    summary_df.to_csv(summary_filepath, index=False)
    
    # Создаем сводную статистику
    stats = summary_df.groupby(['tracking_type', 'CVS_speed', 'sigma']).agg({
        'eddyclicker_id': 'count',
        'matched_smp_tracks_count': 'sum',
        'mean_distance_pixels': 'mean',
        'matching_percentage': 'mean',
        'ec_duration_hours': 'mean',
        'smp_duration_hours': 'mean'
    }).round(2)
    
    stats = stats.rename(columns={
        'eddyclicker_id': 'total_ec_tracks',
        'matched_smp_tracks_count': 'total_matches',
        'ec_duration_hours': 'avg_ec_duration',
        'smp_duration_hours': 'avg_smp_duration'
    })
    
    stats_filename = f"eddyclicker_smp_matching_stats_{data_type}_sigma_{sigma}.csv"
    stats_filepath = os.path.join(output_path, stats_filename)
    stats.to_csv(stats_filepath)
    
    print(f"Сводная таблица сохранена: {summary_filepath}")
    print(f"Статистика сохранена: {stats_filepath}")
    
    return summary_df, stats


if __name__ == "__main__":

    # Запускаем основной процесс
    print("\nЗапуск обработки...")
    main()