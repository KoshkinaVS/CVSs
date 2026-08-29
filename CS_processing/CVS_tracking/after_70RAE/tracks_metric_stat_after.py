#!/usr/bin/env python3
"""
Скрипт для сравнения треков NOAA и NAAD с различными параметрами трекинга
(модифицирован для работы с предварительно объединенными треками)
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geopy.distance import great_circle

# Инициализация путей и параметров
path_init = f'/storage/thalassa/users/vkoshkina'
folder = 'scripts/CS_processing/CVS_tracking/after_70RAE/'
sys.path.insert(2, f'{path_init}/{folder}')

from step_of_tracking import *
from func_for_find_closest_tracks import *

from datetime import datetime


dist_m, our_level, x_unit, y_unit, u_unit, v_unit, time_unit, level_unit, crit, th, min_samples, eps, CS_points_th = get_init_params(data_type, json_file, print_info=True)

# # Конфигурационные параметры
# TRACKING_CONFIGS = [
#     {
#         'tracking_type': 'tracking_local_2_phase',
#         'CVS_speed': 'adv_speed', 
#         'pref_tracking': 'all_points_bound',
#         'description': f'Локальный трекинг (2 фазы), adv speed, {data_type}, sigma=2',
#         'sigma': 2,
#         'data_type': data_type
#     },
#     {
#         'tracking_type': 'tracking_local_2_phase',
#         'CVS_speed': 'adv_speed', 
#         'pref_tracking': 'all_points_bound',
#         'description': f'Локальный трекинг (2 фазы), adv speed, {data_type}, sigma=4',
#         'sigma': 4,
#         'data_type': data_type
#     },
#     {
#         'tracking_type': 'tracking_local_2_phase',
#         'CVS_speed': 'adv_bg_speed', 
#         'pref_tracking': 'all_points_bound',
#         'description': f'Локальный трекинг (2 фазы), adv_bg_speed, {data_type}, sigma=4',
#         'sigma': 4,
#         'data_type': data_type
#     },
#     {
#         'tracking_type': 'tracking_local_2_phase',
#         'CVS_speed': 'adv_bg_speed', 
#         'pref_tracking': 'all_points_bound',
#         'description': f'Локальный трекинг (2 фазы), adv_bg_speed, {data_type}, sigma=2',
#         'sigma': 2,
#         'data_type': data_type
#     },
#     # Добавьте другие конфигурации по аналогии
# ]

# Базовые параметры конфигурации
tracking_types = ['tracking_local_2_phase', 
                 'tracking_global_only', 
                 'tracking_local_global']
speed_options = ['adv_speed', 'no_speed', 'bg_speed', 'adv_bg_speed']
pref_tracking = 'all_points_bound'
sigma_values = [sigma]  # Добавляем возможные значения sigma
data_types = [data_type]  # Добавляем возможные типы данных

# Генерация всех возможных комбинаций конфигураций
TRACKING_CONFIGS = []

for tracking_type in tracking_types:
    for speed in speed_options:
        for sigma in sigma_values:
            for data_type in data_types:
                # Для глобального трекинга используем только no_speed
                if tracking_type == 'tracking_global_only' and speed != 'no_speed':
                    continue
                    
                config = {
                    'tracking_type': tracking_type,
                    'CVS_speed': speed,
                    'pref_tracking': pref_tracking,
                    'description': f"{tracking_type}, {speed}, {pref_tracking}, {data_type}, sigma={sigma}",
                    'sigma': sigma,
                    'data_type': data_type
                }
                TRACKING_CONFIGS.append(config)

# Пример вывода первых 5 конфигураций
print(f"Сгенерировано {len(TRACKING_CONFIGS)} конфигураций")
# for config in TRACKING_CONFIGS[:5]:
#     print(config)

# Параметры анализа
MAX_DISTANCE_KM = 3*dist_m/1000  # Максимальное расстояние для покрытия (км)
YEARS = range(1979, 2019)

def setup_paths(config):
    """Настройка путей к данным для конфигурации"""
    path_init = '/storage/thalassa/users/vkoshkina'
    
    # Путь к объединенным трекам
    folder_name = f"{config['tracking_type']}_{config['CVS_speed']}_{config['pref_tracking']}"
    merged_path = f"{path_init}/data/TC_tracks/NAAD_NOAA_tracks_merged/{config['data_type']}_sigma_{config['sigma']}/{folder_name}"
    
    # Путь к исходным трекам NOAA
    noaa_path = f"{path_init}/data/TC_tracks/splitted"
    
    return {
        'merged_path': merged_path,
        'noaa_path': noaa_path,
        'output_folder': f"{path_init}/data/TC_tracks/tracking_comparison"
    }

def get_total_noaa_tracks(year, paths):
    """Получение общего количества треков NOAA для заданного года"""
    pattern = f"AL*{year}.csv"
    track_files = list(Path(paths['noaa_path']).glob(pattern))
    return len(track_files)

def load_merged_tracks(year, paths):
    """Загрузка объединенных треков NOAA и NAAD для заданного года"""
    track_files = list(Path(paths['merged_path']).glob(f'*{year}_*.csv'))
    
    tracks = []
    for file in track_files:
        try:
            df = pd.read_csv(file, parse_dates=['datetime'])
            tracks.append(df)
        except Exception as e:
            print(f"Ошибка при загрузке файла {file}: {e}")
    
    return tracks

def calculate_coverage(main_track, max_distance_km):
    """Расчет пространственного покрытия для объединенного трека"""
    if len(main_track) == 0:
        return 0.0
    
    # Определяем колонки с точками NAAD
    naad_cols = {
        'lat': [c for c in main_track.columns if 'lat_NAAD_' in c],
        'lon': [c for c in main_track.columns if 'lon_NAAD_' in c]
    }
    
    if not naad_cols['lat'] or not naad_cols['lon']:
        return 0.0
    
    # Собираем все точки NAAD
    naad_points = []
    for lat_col, lon_col in zip(naad_cols['lat'], naad_cols['lon']):
        valid_points = main_track[[lat_col, lon_col]].dropna()
        naad_points.extend(valid_points.values.tolist())
    
    if not naad_points:
        return 0.0
    
    # Преобразуем в DataFrame для удобства
    naad_df = pd.DataFrame(naad_points, columns=['lat', 'lon'])
    
    # Рассчитываем покрытие
    covered = 0
    for _, row in main_track[['lat', 'lon']].dropna().iterrows():
        distances = [
            great_circle((row['lat'], row['lon']), (naad_lat, naad_lon)).km
            for naad_lat, naad_lon in zip(naad_df['lat'], naad_df['lon'])
        ]
        if distances and np.min(distances) <= max_distance_km:
            covered += 1
            
    return (covered / len(main_track)) * 100

def analyze_configuration(config):
    """Анализ одной конфигурации трекинга с объединенными треками"""
    paths = setup_paths(config)
    
    results = []
    
    for year in tqdm(YEARS, desc=f"Анализ {config['description']}"):
        # Получаем общее количество треков NOAA для этого года
        total_noaa_tracks = get_total_noaa_tracks(year, paths)
        
        merged_tracks = load_merged_tracks(year, paths)
        
        if not merged_tracks:
            # Если нет объединенных треков, но есть треки NOAA
            if total_noaa_tracks > 0:
                results.append({
                    'year': year,
                    'space_coverage_median': 0.0,
                    'median_distance_km': np.nan,
                    'tracks_covered': 0,
                    'tracks_total': total_noaa_tracks,
                    'config': config['description']
                })
            continue
        
        # Анализ покрытия и расстояний для каждого трека
        space_cov = []
        all_distances = []  # Для сбора всех расстояний между точками
        
        for track in merged_tracks:
            # coverage, distances = calculate_coverage_and_distance(track, MAX_DISTANCE_KM)
            coverage, distances = calculate_coverage_same_time(track, MAX_DISTANCE_KM)
            
            space_cov.append(coverage)
            all_distances.extend(distances)
        
        # Рассчитываем среднюю ошибку (игнорируя NaN)
        valid_distances = [d for d in all_distances if not np.isnan(d)]
        mean_distance = np.median(valid_distances) if valid_distances else np.nan
        
        results.append({
            'year': year,
            'space_coverage_median': np.median(space_cov) if space_cov else 0.0,
            'median_distance_km': mean_distance,
            'tracks_covered': len([c for c in space_cov if c > 0]),
            'tracks_total': total_noaa_tracks,
            'config': config['description']
        })
    
    return pd.DataFrame(results)

# def calculate_coverage_and_distance(main_track, max_distance_km):
#     """Расчет покрытия и расстояний между треками NOAA и NAAD"""
#     if len(main_track) == 0:
#         return 0.0, []
    
#     # Определяем колонки с точками NAAD
#     naad_cols = {
#         'lat': [c for c in main_track.columns if 'lat_NAAD_' in c],
#         'lon': [c for c in main_track.columns if 'lon_NAAD_' in c]
#     }
    
#     if not naad_cols['lat'] or not naad_cols['lon']:
#         return 0.0, []
    
#     # Собираем все точки NAAD
#     naad_points = []
#     for lat_col, lon_col in zip(naad_cols['lat'], naad_cols['lon']):
#         valid_points = main_track[[lat_col, lon_col]].dropna()
#         naad_points.extend(valid_points.values.tolist())
    
#     if not naad_points:
#         return 0.0, []
    
#     # Преобразуем в DataFrame для удобства
#     naad_df = pd.DataFrame(naad_points, columns=['lat', 'lon'])
    
#     # Рассчитываем покрытие и расстояния
#     covered = 0
#     distances = []
    
#     for _, row in main_track[['lat', 'lon']].dropna().iterrows():
#         point_distances = [
#             great_circle((row['lat'], row['lon']), (naad_lat, naad_lon)).km
#             for naad_lat, naad_lon in zip(naad_df['lat'], naad_df['lon'])
#         ]
        
#         if point_distances:
#             min_dist = np.min(point_distances)
#             distances.append(min_dist)  # Добавляем минимальное расстояние для каждой точки
            
#             if min_dist <= max_distance_km:
#                 covered += 1
                
#     coverage = (covered / len(main_track)) * 100 if len(main_track) > 0 else 0.0
#     return coverage, distances

def calculate_coverage_same_time(main_track, max_distance_km):
    """Расчет покрытия, учитывая, что время одинаково для всех точек в строке."""
    if len(main_track) == 0:
        return 0.0, []

    # Определяем колонки NAAD
    naad_cols = {
        'lat': [c for c in main_track.columns if 'lat_NAAD_' in c],
        'lon': [c for c in main_track.columns if 'lon_NAAD_' in c]
    }

    if not naad_cols['lat'] or not naad_cols['lon']:
        return 0.0, []

    covered = 0
    distances = []

    for _, row in main_track.dropna(subset=['lat', 'lon']).iterrows():
        # Собираем все NAAD-точки из этой строки (с одинаковым временем)
        naad_points = []
        for lat_col, lon_col in zip(naad_cols['lat'], naad_cols['lon']):
            if not pd.isna(row[lat_col]) and not pd.isna(row[lon_col]):
                naad_points.append((row[lat_col], row[lon_col]))

        if not naad_points:
            continue  # Нет NAAD-точек в этой строке

        # Вычисляем расстояния до всех NAAD-точек в этой строке
        point_distances = [
            great_circle((row['lat'], row['lon']), (naad_lat, naad_lon)).km
            for naad_lat, naad_lon in naad_points
        ]

        min_dist = min(point_distances) if point_distances else float('inf')
        distances.append(min_dist)

        if min_dist <= max_distance_km:
            covered += 1

    coverage = (covered / len(main_track)) * 100 if len(main_track) > 0 else 0.0
    return coverage, distances

# def generate_summary_plot(results_df, output_path):
#     """Генерация графика сравнения конфигураций"""
#     plt.figure(figsize=(12, 6))
    
#     # Группируем по конфигурациям
#     grouped = results_df.groupby('config')
    
#     # Среднее покрытие по годам
#     for name, group in grouped:
#         plt.plot(group['year'], group['space_coverage_median'], label=name)
    
#     plt.title('Сравнение покрытия треков NOAA треками NAAD')
#     plt.xlabel('Год')
#     plt.ylabel('Среднее покрытие (%)')
#     plt.legend()
#     plt.grid(True)
    
#     # Сохраняем график
#     os.makedirs(output_path, exist_ok=True)
#     plot_file = f"{output_path}/{data_type}_sigma_{sigma}_coverage_comparison.png"
#     plt.savefig(plot_file, bbox_inches='tight', dpi=300)
#     plt.close()
#     print(f"График сохранен: {plot_file}")

def generate_summary_plot(results_df, output_path, data_type, sigma):
    """Генерация графиков сравнения конфигураций:
    1. Среднее покрытие (space_coverage_median)
    2. Медианное расстояние (median_distance_km)
    3. Отношение покрытия (coverage_ratio)
    """
    os.makedirs(output_path, exist_ok=True)
    grouped = results_df.groupby('config')
    
    # --- 1. График для space_coverage_median ---
    plt.figure(figsize=(12, 6))
    for name, group in grouped:
        plt.plot(group['year'], group['space_coverage_median'], label=name)
    
    plt.title('Сравнение покрытия треков NOAA треками NAAD')
    plt.xlabel('Год')
    plt.ylabel('Среднее покрытие (%)')
    plt.legend()
    plt.grid(True)
    
    plot_file = f"{output_path}/{data_type}_sigma_{sigma}_coverage_comparison.png"
    plt.savefig(plot_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"График сохранен: {plot_file}")

    # --- 2. График для median_distance_km ---
    plt.figure(figsize=(12, 6))
    for name, group in grouped:
        plt.plot(group['year'], group['median_distance_km'], label=name)
    
    plt.title('Медианное расстояние между треками NOAA и NAAD')
    plt.xlabel('Год')
    plt.ylabel('Расстояние (км)')
    plt.legend()
    plt.grid(True)
    
    plot_file = f"{output_path}/{data_type}_sigma_{sigma}_distance_comparison.png"
    plt.savefig(plot_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"График сохранен: {plot_file}")

    # --- 3. График для coverage_ratio ---
    plt.figure(figsize=(12, 6))
    for name, group in grouped:
        plt.plot(group['year'], group['coverage_ratio'], label=name)
    
    plt.title('Отношение покрытия треков NOAA треками NAAD')
    plt.xlabel('Год')
    plt.ylabel('Отношение покрытия')
    plt.legend()
    plt.grid(True)
    
    plot_file = f"{output_path}/{data_type}_sigma_{sigma}_ratio_comparison.png"
    plt.savefig(plot_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"График сохранен: {plot_file}")

def main():
    """Основная функция"""
    all_results = []
    
    for config in TRACKING_CONFIGS:
        df = analyze_configuration(config)
        all_results.append(df)
    
    if not all_results:
        print("Нет данных для анализа!")
        return
    
    # Объединение результатов
    comparison_df = pd.concat(all_results)
    comparison_df['coverage_ratio'] = comparison_df['tracks_covered'] / comparison_df['tracks_total']
    
    
    # Проверяем наличие необходимых столбцов
    required_columns = ['year', 'space_coverage_median', 'median_distance_km', 'tracks_covered', 'tracks_total', 'config']
    if not all(col in comparison_df.columns for col in required_columns):
        missing = set(required_columns) - set(comparison_df.columns)
        print(f"Ошибка: в данных отсутствуют столбцы: {missing}")
        print("Доступные столбцы:", comparison_df.columns.tolist())
        return
    
    # Сохранение результатов
    paths = setup_paths(TRACKING_CONFIGS[0])
    output_path = paths['output_folder']
    os.makedirs(output_path, exist_ok=True)
    
    csv_file = f"{output_path}/{data_type}_sigma_{sigma}_tracking_comparison_{datetime.now().strftime('%Y%m%d')}.csv"
    comparison_df.to_csv(csv_file, index=False)
    print(f"Результаты сохранены: {csv_file}")
    
    # Вывод сводной таблицы
    try:
        summary = comparison_df.groupby('config').agg({
            'space_coverage_median': 'median',
            'median_distance_km': 'median',
            'tracks_covered': 'sum',
            'tracks_total': 'sum'
        })
        summary['coverage_ratio'] = summary['tracks_covered'] / summary['tracks_total']
        
        print("\nСравнительные результаты:")
        print(summary.round(2))
        
        # Генерация графика
        # generate_summary_plot(comparison_df, output_path)
        generate_summary_plot(
                            results_df=comparison_df, 
                            output_path=output_path, 
                            data_type=data_type, 
                            sigma=sigma
                        )
    except Exception as e:
        print(f"Ошибка при анализе результатов: {e}")
        print("Данные:", comparison_df.head())

if __name__ == "__main__":
    main()