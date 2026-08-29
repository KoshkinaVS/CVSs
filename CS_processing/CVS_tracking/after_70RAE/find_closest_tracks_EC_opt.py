from pathlib import Path
import sys
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
import re

# Инициализация путей и параметров
path_init = f'/storage/thalassa/users/vkoshkina'
folder = 'scripts/CS_processing/CVS_tracking/after_70RAE/'
sys.path.insert(2, f'{path_init}/{folder}')

from step_of_tracking import *
from func_for_find_closest_tracks_EC import *

# Configuration parameters
tracking_types = ['tracking_local_2_phase']  # Начнем с одного типа для теста
speed_options = ['adv_speed']  # Начнем с одной опции скорости
pref_tracking = 'all_points_bound'

time_th = 3
similar_steps = 3
max_time_diff = 3
max_distance_km = 5

years = [2019]  # Только один год для теста
path_dir_data = f'{path_init}/data'
path_data = f'{path_init}/data'

def parse_track_date_from_filename(filename):
    """Извлекает дату начала трека из имени файла"""
    match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2})\.csv$', str(filename))
    if match:
        return pd.to_datetime(match.group(1))
    return None

def load_relevant_naad_tracks(path_tracks_dir, ec_track_start, ec_track_end, year, months_to_load):
    """Загружает только релевантные треки NAAD на основе временного окна"""
    relevant_tracks = []
    
    # Определяем временное окно: неделя до начала и до конца трека EddyClicker
    time_window_start = ec_track_start - pd.Timedelta(days=7)
    time_window_end = ec_track_end
    
    for month in months_to_load:
        month_dir = Path(f"{path_tracks_dir}/{year}-{month:02d}")
        if not month_dir.exists():
            continue
            
        # Получаем все файлы треков в месяце
        track_files = list(month_dir.glob('*_track_*.csv'))
        
        for track_file in track_files:
            # Извлекаем дату начала трека из имени файла
            track_start_date = parse_track_date_from_filename(track_file)
            
            if track_start_date is None:
                continue
                
            # Проверяем, попадает ли начало трека в наше временное окно
            if time_window_start <= track_start_date <= time_window_end:
                try:
                    df = pd.read_csv(track_file, parse_dates=['datetime'])
                    if len(df) > time_th:
                        df = df.drop(df.columns[0], axis=1)
                        df = df.dropna(how='any')
                        relevant_tracks.append(df)
                except Exception as e:
                    print(f"Error loading {track_file}: {e}")
    print(f'Loaded {len(relevant_tracks)} tracks for {months_to_load}')
    
    return relevant_tracks

def find_matching_tracks_for_single_track_fast(al_data, NAAD_tracks, max_distance_km, max_time_diff, similar_steps):
    """Быстрая версия поиска совпадающих треков"""
    al_data_sorted = al_data.sort_values('datetime')
    
    # Предварительно вычисляем временные границы для фильтрации
    min_time = al_data_sorted['datetime'].min() - pd.Timedelta(hours=max_time_diff)
    max_time = al_data_sorted['datetime'].max() + pd.Timedelta(hours=max_time_diff)
    
    matching_tracks = []
    
    for track in NAAD_tracks:
        # Быстрая проверка по времени перед полной обработкой
        track_times = track['datetime']
        if not ((track_times >= min_time) & (track_times <= max_time)).any():
            continue
            
        track_sorted = track.sort_values('datetime')
        
        # Используем merge_asof с предварительно отсортированными данными
        try:
            merged = pd.merge_asof(
                al_data_sorted,
                track_sorted,
                on='datetime',
                direction='nearest',
                tolerance=pd.Timedelta(hours=max_time_diff),
                suffixes=('_al', '_track')
            ).dropna()
            
            if len(merged) == 0:
                continue

            # Векторизованное вычисление расстояния
            x_diff = merged['x_al'] - merged['x_track']
            y_diff = merged['y_al'] - merged['y_track']
            distances = np.sqrt(x_diff**2 + y_diff**2)
            
            valid_count = np.sum(distances <= max_distance_km)
            
            if valid_count >= similar_steps:
                matching_tracks.append(track)
                
        except Exception as e:
            print(f"Error processing track: {e}")
            continue
    
    return matching_tracks

def calculate_distances_for_track(ec_track, naad_tracks):
    """Вычисляет расстояния между треком EddyClicker и найденными NAAD треками"""
    dist_list = []
    merged_track = ec_track.copy()
    
    for idx, track_NAAD in enumerate(naad_tracks):
        # Переименовываем колонки для мерджа
        track_NAAD_renamed = track_NAAD.rename(columns=lambda x: f"{x}_NAAD_{idx}" if x not in ['datetime'] else x)
        
        # Мерджим данные
        merged_track = pd.merge(merged_track, track_NAAD_renamed, on="datetime", how="outer")
        merged_track = merged_track.sort_values("datetime").reset_index(drop=True)
        
        # Вычисляем расстояние
        dist = get_tracks_dist(ec_track, track_NAAD, x_name='x', y_name='y', manual_data='EC')
        dist_list.append(np.nanmean(dist) * 6)  # умножаем на 6 км
    
    return merged_track, dist_list

def save_single_track_result(track_data, key, tracking_type, CVS_speed, year, path_data):
    """Сохраняет результаты для одного трека"""
    try:
        # Создаем папки для сохранения
        folder_name_csv = f'EC_tracks_merged_opt/SMP_sigma_{sigma}/{tracking_type}_{CVS_speed}'
        folder_name_pics = f'pics/EC_tracks_comparison_opt/SMP_sigma_{sigma}/{tracking_type}_{CVS_speed}'
        
        os.makedirs(f"{path_data}/SMP/{folder_name_csv}", exist_ok=True)
        os.makedirs(f"{path_data}/SMP/{folder_name_pics}", exist_ok=True)
        
        # Сохраняем CSV
        ec_track = track_data['NOAA_track']
        if track_data['NAAD_tracks']:
            merged_track, dist_list = calculate_distances_for_track(ec_track, track_data['NAAD_tracks'])
            
            # Сохраняем объединенные данные
            CS_start = ec_track['datetime'].iloc[0].strftime('%Y-%m-%dT%H')
            merged_track.to_csv(f'{path_data}/SMP/{folder_name_csv}/{(key+1):09d}_{CS_start}.csv', index=False)
            
            # Сохраняем картинку
            plot_closest_track_for_EC_single(
                ec_track, track_data['NAAD_tracks'], dist_list, key,
                f'{path_data}/SMP', folder_name_pics, data_type='SMP'
            )
            
        print(f"Saved results for track {key}")
        
    except Exception as e:
        print(f"Error saving track {key}: {e}")

def plot_closest_track_for_EC_single(ec_track, naad_tracks, dist_list, key, path_data, folder, data_type='SMP', sigma=2):
    """Отрисовка для одного трека EddyClicker"""
    try:
        if not naad_tracks:
            return
            
        fig = plt.figure(figsize=(5, 5), dpi=150)
        ax = fig.add_subplot(111)

        ax.contourf(np.where(ds['HGT'] > 15, 1, np.nan), cmap='Greys', alpha=0.7)
        
        ax.plot(ec_track['x'], ec_track['y'], c='k')
        
        TC_name = key + 1
        TC_start = ec_track['datetime'].iloc[0].strftime('%Y-%m-%dT%H')
        
        ax.set_title(f'{TC_name} at {TC_start}')
        ax.set_xlim(0, 500)
        
        for idx, CS_NAAD in enumerate(naad_tracks):
            CS_start = CS_NAAD['datetime'].iloc[0].strftime('%Y-%m-%dT%H')
            dist = dist_list[idx] if idx < len(dist_list) else 0
            ax.plot(CS_NAAD['x'], CS_NAAD['y'], label=f'{CS_start} ({int(dist)} km)')

        ax.scatter(ec_track['x'].iloc[0], ec_track['y'].iloc[0], 
                  c='g', s=7, zorder=10, label='manual start')
        ax.scatter(ec_track['x'].iloc[-1], ec_track['y'].iloc[-1], 
                  c='r', s=7, zorder=10, label='manual stop')

        ax.legend(fontsize=7)
        
        # Сохраняем картинку
        fig.savefig(f'{path_data}/{folder}/{data_type}_{(TC_name):09d}_sigma_{sigma}.png', 
                   dpi=200, bbox_inches="tight", transparent=False)
        plt.close(fig)
        
    except Exception as e:
        print(f"Error plotting track {key}: {e}")

def process_single_ec_track_complete(args):
    """Полная обработка одного трека EddyClicker с сохранением результатов"""
    key, ec_track, tracking_type, CVS_speed, year, path_tracks_dir, path_data = args
    
    try:
        ec_track['datetime'] = pd.to_datetime(ec_track['datetime'])
        
        # Определяем временные границы трека EddyClicker
        ec_start_time = ec_track['datetime'].min()
        ec_end_time = ec_track['datetime'].max()
        
        # Определяем месяцы для загрузки
        start_month = ec_start_time.month
        months_to_load = []
        for month_offset in [-1, 0, 1]:
            month_to_load = (start_month + month_offset - 1) % 12 + 1
            months_to_load.append(month_to_load)
        months_to_load = list(set(months_to_load))
        
        # Загружаем только релевантные треки NAAD
        relevant_naad_tracks = load_relevant_naad_tracks(
            path_tracks_dir, ec_start_time, ec_end_time, year, months_to_load
        )
        
        # Предварительная обработка NAAD треков
        naad_tracks_preprocessed = []
        for track in relevant_naad_tracks:
            track_copy = track.copy()
            track_copy['datetime'] = pd.to_datetime(track_copy['datetime'])
            naad_tracks_preprocessed.append(track_copy)
        
        # Находим ближайшие треки
        matching_tracks = find_matching_tracks_for_single_track_fast(
            ec_track, naad_tracks_preprocessed, 
            max_distance_km, max_time_diff, similar_steps
        )
        
        # Сразу сохраняем результаты
        track_data = {
            'NOAA_track': ec_track,
            'NAAD_tracks': matching_tracks,
            'dist_btwn_tracks': []
        }
        
        save_single_track_result(track_data, key, tracking_type, CVS_speed, year, path_data)
        
        return key, len(matching_tracks)
        
    except Exception as e:
        print(f"Error processing track {key}: {e}")
        return key, 0

def process_configuration(config):
    tracking_type, CVS_speed = config
    
    print(f"Starting processing for {tracking_type} {CVS_speed}")
    
    folder_NAAD = f'{path_data}/SMP/SMP_tracks_sigma_{sigma}/{pref_tracking}/{tracking_type}_{CVS_speed}_{pref_tracking}'
    path_tracks_dir = f'{folder_NAAD}/tracks_{circ}'
    
    for year in years:
        print(f"Processing year {year}")
        
        # Загрузка треков EddyClicker
        CS_tracks_list_NOAA = []
        path_data_EC = f'{path_data}/SMP/EddyClicker_tracks'
        CS_tracks_list_NOAA = load_season_tracks_EddyClicker(CS_tracks_list_NOAA, path_data_EC, year, time_th=time_th)

        # Подготовка аргументов для параллельной обработки
        args_list = []
        for idx, ec_track in enumerate(CS_tracks_list_NOAA):
            args_list.append((
                idx, 
                ec_track.copy(),
                tracking_type,
                CVS_speed,
                year,
                path_tracks_dir,
                path_data
            ))
        
        # Параллельная обработка треков с немедленным сохранением
        print(f"Processing {len(args_list)} tracks in parallel...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(tqdm(
                executor.map(process_single_ec_track_complete, args_list),
                total=len(args_list),
                desc=f'Processing {tracking_type} {CVS_speed} {year}'
            ))
        
        # Статистика
        tracks_with_matches = sum(1 for _, match_count in results if match_count > 0)
        print(f"Found matches for {tracks_with_matches} out of {len(results)} tracks")
        
        print(f"Finished processing {tracking_type} {CVS_speed} {year}")

# Запускаем по одному конфигу за раз
for config in itertools.product(tracking_types, speed_options):
    process_configuration(config)