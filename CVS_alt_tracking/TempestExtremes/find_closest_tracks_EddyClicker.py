from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Initialize paths and parameters
path_init = f'/storage/thalassa/users/vkoshkina'
folder = 'scripts/CS_processing/CVS_tracking/after_70RAE/'
sys.path.insert(2, f'{path_init}/{folder}')

from step_of_tracking import *
from func_for_find_closest_tracks import *

from geopy.distance import great_circle
import pandas as pd

def find_matching_tracks_fast(TC_dict, key, CS_track_NOAA, NAAD_tracks, max_distance_km=500, max_time_diff=3, similar_steps=5):
    al_data = CS_track_NOAA.copy()
    al_data['datetime'] = pd.to_datetime(al_data['datetime'])
    
    # Словарь для хранения совпадений (ключ - индекс трека NAAD, значение - (количество совпадений, медианное расстояние))
    track_matches = {}
    
    def process_track(track_data):
        track_data = track_data.copy()
        track_data['datetime'] = pd.to_datetime(track_data['datetime'])

        # Объединение по ближайшему времени с допуском max_time_diff
        merged = pd.merge_asof(
            al_data.sort_values('datetime'),
            track_data.sort_values('datetime'),
            on='datetime',
            direction='nearest',
            tolerance=pd.Timedelta(hours=max_time_diff),
            suffixes=('_al', '_track')
        ).dropna()

        if len(merged) == 0:
            return None

        # Векторизованное вычисление расстояния
        x_diff = merged['x_al'] - merged['x_track']
        y_diff = merged['y_al'] - merged['y_track']
        distances = np.sqrt(x_diff**2 + y_diff**2)
        merged['distance'] = distances

        # Фильтрация точек, где расстояние <= max_distance_km
        valid = merged[merged['distance'] <= max_distance_km]
        
        if len(valid) >= similar_steps:
            return len(valid), np.nanmedian(merged['distance'])
        return None

    # Обработка всех треков NAAD
    for idx, track in enumerate(NAAD_tracks):
        result = process_track(track)
        if result:
            matches_count, median_dist = result
            # Фильтрация по медианному расстоянию
            if median_dist <= max_distance_km:
                track_matches[idx] = (matches_count, median_dist)
    
    # Отбираем только треки, удовлетворяющие условиям
    TC_dict[key]['NAAD_tracks'] = [
        NAAD_tracks[i] 
        for i in track_matches 
        if track_matches[i][1] <= max_distance_km  # Проверка медианного расстояния
    ]
    
    return TC_dict

def calculate_detection_metrics(TC_dict, all_naad_tracks):
    hits = 0
    misses = 0
    false_alarms = 0
    false_alarm_tracks = []
    
    # 1. Подсчет hits и misses
    for noaa_id, track_data in TC_dict.items():
        if track_data['NAAD_tracks']:
            hits += 1
        else:
            misses += 1

    # 2. Определение false alarms
    # Собираем все NAAD треки, которые были сопоставлены
    matched_naad_tracks = []
    for track_data in TC_dict.values():
        matched_naad_tracks.extend(track_data['NAAD_tracks'])
    
    def is_track_in_list(track, track_list):
        """Custom function to check if a track exists in a list of tracks"""
        for t in track_list:
            if (track[['x', 'y', 'datetime']].equals(t[['x', 'y', 'datetime']])):  # adjust columns as needed
                return True
        return False
    
    false_alarm_tracks = [t for t in all_naad_tracks 
                         if not is_track_in_list(t, matched_naad_tracks)]
    false_alarms = len(false_alarm_tracks)
    
    return hits, misses, false_alarms, false_alarm_tracks
    
def process_sigma(years, output_dir, path_data_EC, NOAA_matching_dir, data_type, sigma):
    # Список для сбора метрик по всем годам
    all_metrics = []
    
    total_hits = 0
    total_misses = 0
    total_false_alarms = 0
    all_false_alarm_tracks = []
    
    
    for year in years:
        CS_tracks_list_NAAD = []
        filename = f"{output_dir}/{data_type}_TC_tracks_{year}.txt"
        CS_tracks_list_NAAD = load_TE_tracks(CS_tracks_list_NAAD, filename)

        # Загрузка EddyClicker треков
        CS_tracks_list_NOAA = load_season_tracks_EddyClicker(path_data_EC, year, time_th=3)

        
        TC_dict = {
                idx: {
                    'NOAA_track': CS_track,
                    'NAAD_tracks': [],
                    'dist_btwn_tracks': []
                }
                for idx, CS_track in enumerate(CS_tracks_list_NOAA)
            }

        # Find matching tracks
        for key in TC_dict.keys():
            CS_track_NOAA = TC_dict[key]['NOAA_track']
            NOAA_times = CS_track_NOAA['datetime']
            NAAD_tracks = [CS for CS in CS_tracks_list_NAAD if np.isin(CS['datetime'], NOAA_times).any()]
            TC_dict = find_matching_tracks_fast(TC_dict, key, CS_track_NOAA, NAAD_tracks, max_distance_km, max_time_diff, similar_steps)


        for key in tqdm(TC_dict.keys(), desc=f'Calculating distances {year}'):
            dist_list = []
            CS_track_NOAA = TC_dict[key]['NOAA_track']
            for idx in range(len(TC_dict[key]['NAAD_tracks'])):
                track_NAAD = TC_dict[key]['NAAD_tracks'][idx]
                if track_NAAD is not None:
                    CS_track_NAAD = track_NAAD.rename(columns=lambda x: f"{x}_NAAD_{idx}" if x not in ['datetime'] else x)   
                    CS_track_NOAA = pd.merge(CS_track_NOAA, CS_track_NAAD, on="datetime", how="outer")
                    CS_track_NOAA = CS_track_NOAA.sort_values("datetime").reset_index(drop=True)
                    dist = get_tracks_dist(TC_dict[key]['NOAA_track'], track_NAAD)
                    dist_list.append(np.nanmean(dist))
            
            TC_dict[key]['NOAA_track_NAAD'] = CS_track_NOAA
            TC_dict[key]['dist_btwn_tracks'] = dist_list
        
        for idx, key in tqdm(enumerate(TC_dict.keys()), 
                    desc=f'Plotting tracks for {year}',
                    total=len(TC_dict)):
            plot_EC_track_with_matches(TC_dict[key], TC_dict[key]['NAAD_tracks'], NOAA_matching_dir, key, data_type, sigma)


        # Calculate metrics for current year
        year_hits, year_misses, year_false_alarms, false_alarm_tracks = calculate_detection_metrics(TC_dict, CS_tracks_list_NAAD)
        total_hits += year_hits
        total_misses += year_misses
        total_false_alarms += year_false_alarms
        all_false_alarm_tracks.extend(false_alarm_tracks)
        
        # Save individual year results
        metrics = {
            'year': year,
            'POD': year_hits / (year_hits + year_misses) if (year_hits + year_misses) > 0 else 0,
            'FAR': year_false_alarms / (year_hits + year_false_alarms) if (year_hits + year_false_alarms) > 0 else 0,
            'Hits': year_hits,
            'Misses': year_misses,
            'False_Alarms': year_false_alarms
        }
        
        all_metrics.append(metrics)
    
    # Calculate overall metrics for cluster
    overall_pod = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0
    overall_far = total_false_alarms / (total_hits + total_false_alarms) if (total_hits + total_false_alarms) > 0 else 0
    
    overall_metrics = {
        'year': 'overall',
        'POD': overall_pod,
        'FAR': overall_far,
        'Hits': total_hits,
        'Misses': total_misses,
        'False_Alarms': total_false_alarms
    }
    
    all_metrics.append(overall_metrics)
    
    # Save all metrics (by year + overall) to a single CSV
    metrics_summary_file = os.path.join(f'{NOAA_matching_dir}', 'detection_metrics_summary.csv')
    pd.DataFrame(all_metrics).to_csv(metrics_summary_file, index=False)
    
    return overall_metrics

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

dist_m, our_level, level, x_unit, y_unit, u_unit, v_unit, time_unit, level_unit, crit, th, min_samples, eps, CS_points_th = get_init_params(data_type, json_file, print_info=True)

time_th = 3
similar_steps = 3
max_time_diff = 3
# max_distance_km = 3*dist_m/1000

similar_steps = 5
max_distance_km = 250


years = np.arange(1979, 2019)
years = np.arange(2019, 2020)

months = np.arange(1, 13, 1)

sigmas = [
    4,
    2,
    # 0, 
]

path_data = f'{path_init}/data'

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





def load_TE_tracks(tracks_list, file_path, time_th=3):
    """
    Загружает треки из файла ERA5 с прогресс-баром
    
    Args:
        file_path (str): Путь к файлу с треками
        time_th (int): Минимальное количество точек для сохранения трека
        
    Returns:
        list: Список DataFrame с треками
    """
    current_track = []
    track_count = 0
    
    # Сначала посчитаем общее количество строк для прогресс-бара
    with open(file_path, 'r') as f:
        total_lines = sum(1 for _ in f)
    
    # Теперь обрабатываем файл с прогресс-баром
    with open(file_path, 'r') as f:
        pbar = tqdm(total=total_lines, desc="Processing tracks", unit="lines")
        
        for line in f:
            pbar.update(1)
            
            if line.startswith('start'):
                # Если начинается новый трек, сохраняем предыдущий (если он есть)
                if current_track:
                    df = pd.DataFrame(current_track, 
                                    columns=['x', 'y', 'lon', 'lat', 'wind', 'r2d', 
                                             'year', 'month', 'day', 'hour'])
                    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
                    if len(df) > time_th:
                        tracks_list.append(df)
                        track_count += 1
                    current_track = []
            else:
                # Парсим строку с данными
                parts = line.strip().split()
                if len(parts) == 10:  # Проверяем, что строка содержит все 10 значений
                    current_track.append([float(x) for x in parts])
        
        # Добавляем последний трек (если он есть)
        if current_track:
            df = pd.DataFrame(current_track, 
                            columns=['x', 'y', 'lon', 'lat', 'wind', 'r2d', 
                                     'year', 'month', 'day', 'hour'])
            df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
            
            if len(df) > time_th:
                tracks_list.append(df)
                track_count += 1
        
        pbar.close()
    
    print(f"\nLoaded {track_count} tracks with more than {time_th} points")
    return tracks_list


path_data_EC = f'{path_data}/{data_type}/EddyClicker_tracks'



all_metrics = []

if data_type == 'LoRes':
    level = 12
elif data_type == 'ERA5':
    level = 500
elif data_type == 'SMP':
    level = 10

    
for sigma in sigmas:
    NOAA_matching_dir = f'/storage/thalassa/users/vkoshkina/data/TempestExtremes/{data_type}/EC_matching_results_timefilter_1h/{data_type}_sigma_{sigma}'
    
    path_init_TE = f'/storage/thalassa/users/vkoshkina/data/TempestExtremes'
    sigma_dir = f"{path_init_TE}/{data_type}/R2D_{data_type}_level_{level}_sigma_{sigma}"
    # path_data_tracks = f"{sigma_dir}/Tracks_3"
    path_data_tracks = f"{sigma_dir}/Tracks_timefilter_1h"
    
    print(f"\nProcessing sigma={sigma}...")
    metrics = process_sigma(years, path_data_tracks, path_data_EC, NOAA_matching_dir, data_type, sigma)
    all_metrics.append(metrics)

# Save combined metrics for all clusters
combined_metrics_file = os.path.join(NOAA_matching_dir, 'combined_detection_metrics.csv')
pd.DataFrame(all_metrics).to_csv(combined_metrics_file, index=False)