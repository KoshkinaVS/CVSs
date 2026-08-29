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

from pathlib import Path
import sys
import os

# Импорты для работы с координатами и расстояниями
from geopy.distance import great_circle, geodesic

# Пути и инициализация
path_init = f'/storage/thalassa/users/vkoshkina'
folder = 'scripts/CS_processing/CVS_tracking/after_70RAE/'
sys.path.insert(2, f'{path_init}/{folder}')

json_file = f'{path_init}/{folder}/tracking_init.json'

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
    
    # Параметры для разных типов данных
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

# Функции для загрузки треков
def load_season_tracks_EddyClicker(path_data, year, time_th=3):
    """
    Загружает все EddyClicker треки
    """
    CS_tracks_list = []
    name_NOAA = f'*.csv'
    ls = list(sorted(Path(f"{path_data}/").glob(name_NOAA)))

    if len(ls) != 0:
        for ifile in tqdm(ls):
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

def load_tracks_for_EC_track(EC_track, path_tracks_dir, time_buffer_days=7):
    """
    Загружает треки NAAD для конкретного EC трека
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

        print(all_files[0])
        
        for ifile in all_files:
            filename = ifile.name
            try:
                # Извлекаем дату начала трека из имени файла
                date_part = filename.split('_track_')[-1].replace('.csv', '')
                track_start_time = pd.to_datetime(date_part)
                
                # Быстрая проверка: если начало трека в нужном диапазоне
                if start_date_bound <= track_start_time <= end_date_bound:
                    df = pd.read_csv(ifile, parse_dates=['datetime'])
                    if len(df) > 3:
                        df = df.drop(df.columns[0], axis=1)
                        df = df.dropna(how='any')
                        CS_tracks_for_EC.append(df)
                        
            except (ValueError, IndexError):
                continue

    print(f'{len(CS_tracks_for_EC)} loaded for {start_date_bound}-{end_date_bound}')
    
    return CS_tracks_for_EC

# Функции для расчета статистики
def calculate_track_duration(track):
    """Вычисляет длительность трека в часах"""
    if len(track) < 2:
        return 0
    
    duration = (track['datetime'].max() - track['datetime'].min()).total_seconds() / 3600
    return duration

def calculate_track_length_km(track, grid_spacing_km=6):
    """Вычисляет длину трека в километрах на основе сеточных координат"""
    if len(track) < 2:
        return 0
    
    total_distance = 0
    for i in range(len(track) - 1):
        x1, y1 = track.iloc[i]['x'], track.iloc[i]['y']
        x2, y2 = track.iloc[i+1]['x'], track.iloc[i+1]['y']
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) * grid_spacing_km
        total_distance += distance
    
    return total_distance

def calculate_track_length_latlon(track):
    """Вычисляет длину трека в километрах на основе широты/долготы"""
    if len(track) < 2:
        return 0
    
    total_distance = 0
    for i in range(len(track) - 1):
        if 'lat' in track.columns and 'lon' in track.columns:
            coord1 = (track.iloc[i]['lat'], track.iloc[i]['lon'])
            coord2 = (track.iloc[i+1]['lat'], track.iloc[i+1]['lon'])
            distance = great_circle(coord1, coord2).km
            total_distance += distance
    
    return total_distance

def calculate_matching_percentage(ec_track, naad_track):
    """Вычисляет процент совпадения по времени"""
    ec_times = set(ec_track['datetime'])
    naad_times = set(naad_track['datetime'])
    common_times = ec_times.intersection(naad_times)
    
    if len(ec_times) == 0:
        return 0
    
    return len(common_times) / len(ec_times) * 100

def calculate_mean_distance(ec_track, naad_track, grid_spacing_km=6):
    """Вычисляет среднее расстояние между треками в км"""
    common_dates = np.intersect1d(ec_track['datetime'].values, naad_track['datetime'].values)
    
    if len(common_dates) == 0:
        return None
    
    ec_subset = ec_track[ec_track['datetime'].isin(common_dates)]
    naad_subset = naad_track[naad_track['datetime'].isin(common_dates)]
    
    distances = []
    for i in range(len(ec_subset)):
        x1, y1 = ec_subset.iloc[i]['x'], ec_subset.iloc[i]['y']
        x2, y2 = naad_subset.iloc[i]['x'], naad_subset.iloc[i]['y']
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) * grid_spacing_km
        distances.append(distance)
    
    return np.mean(distances) if distances else None

def extract_naad_vorticity_info(naad_track):
    """Извлекает информацию о завихренности из трека NAAD"""
    vort_info = {
        'max_vort': None,
        'max_vort_time': None,
        'min_vort': None,
        'mean_vort': None,
        'vort_column': None
    }
    
    possible_vort_columns = [
        'vort', 'vorticity', 'rel_vort', 'abs_vort',
        'vort_max', 'max_vort', 'zeta', 'vor'
    ]
    
    for col in naad_track.columns:
        col_lower = col.lower()
        for vort_keyword in possible_vort_columns:
            if vort_keyword in col_lower:
                if naad_track[col].notna().any():
                    vort_info['max_vort'] = naad_track[col].max()
                    vort_info['min_vort'] = naad_track[col].min()
                    vort_info['mean_vort'] = naad_track[col].mean()
                    
                    max_idx = naad_track[col].idxmax()
                    vort_info['max_vort_time'] = naad_track.iloc[max_idx]['datetime']
                    vort_info['vort_column'] = col
                    return vort_info
    
    return vort_info

# Основная функция для создания таблицы статистики
def create_ec_naad_matching_table(ec_tracks, naad_tracks_list, matching_results, 
                                  output_path, tracking_type, CVS_speed, year, sigma,
                                  grid_spacing_km=6):
    """
    Создает таблицу статистики сопоставления треков EC с треками NAAD
    """
    matching_data = []
    
    for track_id, ec_track in enumerate(ec_tracks):
        # Получаем совпавшие треки для этого EC трека
        matched_naad_tracks = []
        matched_distances = []
        
        if track_id in matching_results:
            matched_naad_tracks = matching_results[track_id]['tracks']
            matched_distances = matching_results[track_id]['distances']
        
        # Основная информация о треке EC
        ec_info = {
            'year': year,
            'tracking_type': tracking_type,
            'CVS_speed': CVS_speed,
            'sigma': sigma,
            'ec_track_id': track_id + 1,
            'ec_start_time': ec_track['datetime'].min(),
            'ec_end_time': ec_track['datetime'].max(),
            'ec_duration_hours': calculate_track_duration(ec_track),
            'ec_length_km': calculate_track_length_km(ec_track, grid_spacing_km),
            'ec_points_count': len(ec_track),
            'ec_start_x': ec_track['x'].values[0] if len(ec_track) > 0 else None,
            'ec_start_y': ec_track['y'].values[0] if len(ec_track) > 0 else None,
            'ec_end_x': ec_track['x'].values[-1] if len(ec_track) > 0 else None,
            'ec_end_y': ec_track['y'].values[-1] if len(ec_track) > 0 else None,
            'matched_naad_tracks_count': len(matched_naad_tracks)
        }
        
        # Если есть совпавшие треки NAAD
        if matched_naad_tracks:
            for i, (naad_track, distance) in enumerate(zip(matched_naad_tracks, matched_distances)):
                # Информация о завихренности для NAAD
                naad_vort_info = extract_naad_vorticity_info(naad_track)
                
                track_info = ec_info.copy()
                track_info.update({
                    'naad_track_index': i,
                    'mean_distance_km': distance,
                    'naad_start_time': naad_track['datetime'].min(),
                    'naad_end_time': naad_track['datetime'].max(),
                    'naad_duration_hours': calculate_track_duration(naad_track),
                    'naad_length_km': calculate_track_length_km(naad_track, grid_spacing_km),
                    'naad_points_count': len(naad_track),
                    'naad_start_x': naad_track['x'].values[0] if len(naad_track) > 0 else None,
                    'naad_start_y': naad_track['y'].values[0] if len(naad_track) > 0 else None,
                    'naad_end_x': naad_track['x'].values[-1] if len(naad_track) > 0 else None,
                    'naad_end_y': naad_track['y'].values[-1] if len(naad_track) > 0 else None,
                    'naad_max_vort': naad_vort_info['max_vort'],
                    'naad_min_vort': naad_vort_info['min_vort'],
                    'naad_mean_vort': naad_vort_info['mean_vort'],
                    'naad_max_vort_time': naad_vort_info['max_vort_time'],
                    'naad_vort_column': naad_vort_info['vort_column'],
                    'matching_percentage': calculate_matching_percentage(ec_track, naad_track),
                    'temporal_overlap_hours': len(np.intersect1d(ec_track['datetime'].values, 
                                                                naad_track['datetime'].values)),
                    'is_complete_match': len(ec_track) == len(np.intersect1d(ec_track['datetime'].values, 
                                                                            naad_track['datetime'].values))
                })
                matching_data.append(track_info)
        else:
            # Если нет совпавших треков
            ec_info.update({
                'naad_track_index': None,
                'mean_distance_km': None,
                'naad_start_time': None,
                'naad_end_time': None,
                'naad_duration_hours': None,
                'naad_length_km': None,
                'naad_points_count': None,
                'naad_start_x': None,
                'naad_start_y': None,
                'naad_end_x': None,
                'naad_end_y': None,
                'naad_max_vort': None,
                'naad_min_vort': None,
                'naad_mean_vort': None,
                'naad_max_vort_time': None,
                'naad_vort_column': None,
                'matching_percentage': None,
                'temporal_overlap_hours': None,
                'is_complete_match': None
            })
            matching_data.append(ec_info)
    
    # Создаем DataFrame
    df_matching = pd.DataFrame(matching_data)
    
    # Сохраняем таблицу
    os.makedirs(output_path, exist_ok=True)
    filename = f"ec_naad_matching_{tracking_type}_{CVS_speed}_{year}_sigma_{sigma}.csv"
    filepath = os.path.join(output_path, filename)
    df_matching.to_csv(filepath, index=False)
    
    print(f"Таблица сохранена: {filepath}")
    print(f"Всего треков EC: {len(ec_tracks)}")
    matched_count = len(df_matching[df_matching['matched_naad_tracks_count'] > 0])
    print(f"Треков с совпадениями: {matched_count}")
    
    return df_matching

# Функция для сопоставления треков (аналогичная find_matching_tracks_for_EC)
def find_matching_tracks_for_EC_stats(EC_track, NAAD_tracks, max_distance_km=5, 
                                      max_time_diff=3, similar_steps=5, grid_spacing_km=6):
    """
    Находит совпадающие треки для конкретного EC трека и возвращает статистику
    """
    # Предварительная обработка EC трека
    ec_data = EC_track.copy()
    ec_data['datetime'] = pd.to_datetime(ec_data['datetime'])
    ec_data_sorted = ec_data.sort_values('datetime')
    
    # Предварительно вычисляем временные границы для фильтрации
    min_time = ec_data_sorted['datetime'].min() - pd.Timedelta(hours=max_time_diff)
    max_time = ec_data_sorted['datetime'].max() + pd.Timedelta(hours=max_time_diff)
    
    matching_tracks = []
    matching_distances = []
    
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
        distances = np.sqrt(x_diff**2 + y_diff**2) * grid_spacing_km
        
        valid_count = np.sum(distances <= max_distance_km)
        
        if valid_count >= similar_steps:
            # Возвращаем среднее расстояние по всем валидным точкам
            valid_distances = distances[distances <= max_distance_km]
            return track_data, np.mean(valid_distances) if len(valid_distances) > 0 else None
        else:
            return None, None

    # Обрабатываем треки с предварительной фильтрацией
    for track in NAAD_tracks:
        matched_track, avg_distance = process_track(track)
        if matched_track is not None:
            matching_tracks.append(matched_track)
            matching_distances.append(avg_distance)
    
    return matching_tracks, matching_distances

# Основная функция обработки
def process_ec_tracks_for_statistics(year, data_type, sigma, tracking_type, CVS_speed, 
                                     path_dir_data, ec_tracks, path_tracks_dir):
    """
    Обрабатывает EC треки для заданной конфигурации и создает таблицу статистики
    """
    print(f"Обработка EC треков для {year}, {tracking_type}, {CVS_speed}, sigma={sigma}")
    

    
    # Словарь для хранения результатов сопоставления
    matching_results = {}
    
    # Обрабатываем каждый EC трек
    for track_id, ec_track in tqdm(enumerate(ec_tracks), desc="Обработка EC треков", total=len(ec_tracks)):
        try:
            # Загружаем соответствующие NAAD треки
            naad_tracks = load_tracks_for_EC_track(ec_track, path_tracks_dir, time_buffer_days=7)
            
            # Находим совпадающие треки
            matched_tracks, distances = find_matching_tracks_for_EC_stats(
                ec_track, naad_tracks, 
                max_distance_km=3*77,  # 3 шага по сетке в км
                max_time_diff=3,    # 3 часа
                similar_steps=3,    # минимум 3 совпадающих шагов
                grid_spacing_km=77   # шаг сетки 77 км
            )
            
            # Сохраняем результаты
            matching_results[track_id] = {
                'tracks': matched_tracks,
                'distances': distances
            }
            
        except Exception as e:
            print(f"Ошибка при обработке трека {track_id+1}: {e}")
            matching_results[track_id] = {'tracks': [], 'distances': []}
    
    # Создаем таблицу статистики
    output_table_path = f'{path_init}/data/TC_tracks/EC_{data_type}_matching_tables'
    
    df_stats = create_ec_naad_matching_table(
        ec_tracks, [], matching_results,  # naad_tracks_list не используется напрямую
        output_table_path, tracking_type, CVS_speed, year, sigma,
        grid_spacing_km=6
    )
    
    return df_stats

# Функция для создания сводной статистики
def create_summary_statistics(all_stats_dfs, output_path):
    """Создает сводную статистику по всем конфигурациям"""
    if not all_stats_dfs:
        print("Нет данных для создания сводной статистики")
        return None, None
    
    # Объединяем все таблицы
    summary_df = pd.concat(all_stats_dfs, ignore_index=True)
    
    # Сохраняем сводную таблицу
    summary_filename = "ec_naad_matching_summary_full.csv"
    summary_filepath = os.path.join(output_path, summary_filename)
    summary_df.to_csv(summary_filepath, index=False)
    
    # Создаем сводную статистику
    stats = summary_df.groupby(['tracking_type', 'CVS_speed', 'sigma']).agg({
        'ec_track_id': 'count',
        'matched_naad_tracks_count': 'sum',
        'ec_duration_hours': 'mean',
        'ec_length_km': 'mean',
        'mean_distance_km': 'mean',
        'matching_percentage': 'mean',
        'naad_duration_hours': 'mean',
        'naad_length_km': 'mean',
        'naad_max_vort': 'mean',
        'naad_mean_vort': 'mean'
    }).round(2)
    
    stats = stats.rename(columns={
        'ec_track_id': 'total_ec_tracks',
        'matched_naad_tracks_count': 'total_matches',
        'ec_duration_hours': 'avg_ec_duration_h',
        'ec_length_km': 'avg_ec_length_km',
        'mean_distance_km': 'avg_distance_km',
        'matching_percentage': 'avg_matching_percent',
        'naad_duration_hours': 'avg_naad_duration_h',
        'naad_length_km': 'avg_naad_length_km',
        'naad_max_vort': 'avg_naad_max_vort',
        'naad_mean_vort': 'avg_naad_mean_vort'
    })
    
    stats_filename = "ec_naad_matching_stats_full.csv"
    stats_filepath = os.path.join(output_path, stats_filename)
    stats.to_csv(stats_filepath)
    
    print(f"Сводная таблица сохранена: {summary_filepath}")
    print(f"Статистика сохранена: {stats_filepath}")
    
    return summary_df, stats

# Главная функция
def main():
    """Основная функция для создания статистики EC-NAAD сопоставления"""
    print('data_type: ')
    data_type = input()
    
    print('sigma: ')
    sigma = int(input().strip())

    # Специальная конфигурация для cone теста
    cone_config = {
        'tracking_type': 'tracking_local_2_phase_cone',
        'speed_option': 'avd_cone',  # специальный speed option для cone
        'pref_tracking': 'update_2026_cone_test'
    }
    
    # Конфигурационные параметры
    tracking_types = [
        
                        'tracking_local_2_phase_cone',
                        'tracking_local_2_phase', 
                      'tracking_global_only', 
                      'tracking_local_global',
                      # 'tracking_local_only'
                     ]
    
    speed_options = ['adv_speed', 'no_speed', 'bg_speed', 'adv_bg_speed']

    if data_type == 'SMP':
        years = [2019]  # Можно изменить на нужные годы
    else:
        years = [2010]  # Можно изменить на нужные годы
    
    # Получаем параметры инициализации
    dist_m, our_level, level, x_unit, y_unit, u_unit, v_unit, time_unit, level_unit, crit, th, min_samples, eps, CS_points_th = get_init_params(data_type, json_file)
    
    # Пути к данным
    path_dir_data = f'{path_init}/data'
    if data_type == 'LoRes':
        path_dir_data = f'{path_dir_data}/LoRes'

    if data_type == 'LoRes':
        path_data_EC = f'{path_dir_data}/{data_type}/EddyClicker_Egor_2010'
    elif data_type == 'SMP':
        path_data_EC = f'{path_dir_data}/{data_type}/EddyClicker_tracks'


    # Загружаем треки EddyClicker
    print("Загрузка EC треков...")
    print(path_data_EC)
    ec_tracks = load_season_tracks_EddyClicker(path_data_EC, years[0], time_th=3)
    print(f"Загружено EC треков: {len(ec_tracks)}")
    
    if not ec_tracks:
        print(f"Нет EC треков для {years[0]}")
        return None
        
    
    pref_tracking = 'update_2026_wspd_hw_3'
    
    # Список для хранения всех статистик
    all_stats_dfs = []
    
    # Обрабатываем все конфигурации
    for tracking_type in tracking_types:
        for CVS_speed in speed_options:
            for year in years:
                try:
                    # Формируем путь к трекам NAAD
                    DBSCAN_name = f'DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_sigma_{sigma}'

                    if tracking_type == 'tracking_local_2_phase_cone':
                        # Формируем путь для cone теста
                        results_dir = f"{cone_config['tracking_type']}_{cone_config['speed_option']}_{cone_config['pref_tracking']}"
                        path_tracks_dir = f"{path_dir_data}/{data_type}/{data_type}_tracks_sigma_2/{cone_config['pref_tracking']}/{results_dir}/tracks_C/"
                    else:
                        # Формируем путь к директории с треками
                        results_dir = f"{tracking_type}_{CVS_speed}_{pref_tracking}"
                        path_tracks_dir = f"{path_dir_data}/{data_type}/{data_type}_tracks_sigma_2/{pref_tracking}/{results_dir}/tracks_C/"
                        
                    # Обрабатываем треки и получаем статистику
                    df_stats = process_ec_tracks_for_statistics(
                        year, data_type, sigma, tracking_type, CVS_speed,
                        path_dir_data, ec_tracks, path_tracks_dir
                    )
                    
                    if df_stats is not None:
                        all_stats_dfs.append(df_stats)
                        
                except Exception as e:
                    print(f"Ошибка при обработке конфигурации {tracking_type} {CVS_speed} {year}: {e}")
                    continue
    
    # Создаем сводную статистику
    if all_stats_dfs:
        output_path = f'{path_init}/data/TC_tracks/EC_{data_type}_matching_tables_2026-05-13'
        summary_df, stats = create_summary_statistics(all_stats_dfs, output_path)
        
        # Выводим краткую статистику
        print("\n" + "="*60)
        print("СВОДНАЯ СТАТИСТИКА СОПОСТАВЛЕНИЯ EC И NAAD ТРЕКОВ")
        print("="*60)
        print(stats)
        
        # Дополнительная аналитика
        print("\n" + "="*60)
        print("ДОПОЛНИТЕЛЬНАЯ АНАЛИТИКА:")
        print("="*60)
        
        total_ec_tracks = len(pd.concat([df[['ec_track_id']].drop_duplicates() for df in all_stats_dfs]))
        total_matches = sum([df['matched_naad_tracks_count'].sum() for df in all_stats_dfs])
        
        print(f"Всего уникальных EC треков: {total_ec_tracks}")
        print(f"Всего совпадений с NAAD треками: {total_matches}")
        print(f"Среднее количество совпадений на трек: {total_matches/total_ec_tracks:.2f}")
        
        # Анализ по лучшей конфигурации
        if not stats.empty:
            best_config = stats.loc[stats['total_matches'].idxmax()]
            print(f"\nЛучшая конфигурация: {stats['total_matches'].idxmax()}")
            print(f"Количество совпадений: {best_config['total_matches']:.0f}")
            print(f"Среднее расстояние: {best_config['avg_distance_km']:.1f} км")
            print(f"Процент совпадения: {best_config['avg_matching_percent']:.1f}%")
    
    else:
        print("Нет данных для создания статистики")

if __name__ == "__main__":
    main()