#!/usr/bin/env python3
"""
Скрипт для сравнения треков NOAA и NAAD с различными параметрами трекинга
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

# Конфигурационные параметры
TRACKING_CONFIGS = [
    {
        'tracking_type': 'tracking_local_2_phase',
        'CVS_speed': 'adv_speed', 
        'pref_tracking': 'all_points_bound',
        'description': 'Локальный трекинг (2 фазы), adv speed, все точки'
    },
    # Добавьте другие конфигурации по аналогии
]

# Параметры анализа
MAX_DISTANCE_KM = 233.5  # Максимальное расстояние для покрытия (км)
DATA_TYPE = 'LoRes'
SIGMA = 2
YEARS = range(1979, 2019)
MONTHS = range(1, 13)

def setup_paths(data_type, tracking_type, CVS_speed, pref_tracking, sigma):
    """Настройка путей к данным"""
    path_init = '/storage/thalassa/users/vkoshkina'
    folder = 'scripts/CS_processing/CVS_tracking/after_70RAE/'
    sys.path.insert(2, f'{path_init}/{folder}')

    path_dir_data = f'{path_init}/data'
    if data_type == 'LoRes':
        path_dir_data = f'{path_dir_data}/LoRes'

    folder_NOAA = 'TC_tracks/splitted'
    folder_name = f'{tracking_type}_{CVS_speed}_{pref_tracking}'
    
    return {
        'path_data': path_init,
        'folder_NOAA': folder_NOAA,
        'folder_NAAD': f'{path_dir_data}/{data_type}_tracks/{pref_tracking}/{folder_name}',
        'output_folder': f'{path_init}/results/tracking_comparison'
    }

def load_tracks(year, months, paths):
    """Загрузка треков NOAA и NAAD для заданного года"""
    from func_for_find_closest_tracks import load_season_tracks, load_season_tracks_NOAA
    
    CS_tracks_list_NAAD = []
    CS_tracks_list_NOAA = []
    
    CS_tracks_list_NAAD = load_season_tracks(
        CS_tracks_list_NAAD, year, months, paths['path_tracks_dir'], time_th=3
    )
    CS_tracks_list_NOAA = load_season_tracks_NOAA(
        CS_tracks_list_NOAA, f'{paths["path_data"]}/{paths["folder_NOAA"]}', year, time_th=3
    )
    
    return CS_tracks_list_NAAD, CS_tracks_list_NOAA

def calculate_coverage(main_track, naad_track, max_distance_km):
    """Расчет пространственного покрытия"""
    if len(main_track) == 0 or len(naad_track) == 0:
        return 0.0
    
    covered = 0
    for _, row in main_track.iterrows():
        distances = [
            great_circle((row['lat'], row['lon']), (naad_lat, naad_lon)).km
            for naad_lat, naad_lon in zip(naad_track['lat'], naad_track['lon'])
        ]
        if distances and np.min(distances) <= max_distance_km:
            covered += 1
            
    return (covered / len(main_track)) * 100

def analyze_configuration(config):
    """Анализ одной конфигурации трекинга"""
    paths = setup_paths(
        DATA_TYPE, config['tracking_type'], 
        config['CVS_speed'], config['pref_tracking'], SIGMA
    )
    
    results = []
    
    for year in tqdm(YEARS, desc=f"Анализ {config['description']}"):
        CS_tracks_list_NAAD, CS_tracks_list_NOAA = load_tracks(year, MONTHS, paths)
        
        TC_dict = {
            track['Id'].values[0]: {
                'NOAA_track': track,
                'NAAD_tracks': [],
                'dist_btwn_tracks': []
            }
            for track in CS_tracks_list_NOAA
        }
        
        # Анализ покрытия
        space_cov = []
        for key in TC_dict.keys():
            data = TC_dict[key]['NOAA_track']
            
            # Сбор точек NAAD
            naad_indices = sorted(set(
                int(col.split('_')[-1]) 
                for col in data.columns 
                if 'NAAD' in col and ('lat' in col or 'lon' in col)
            ))
            
            naad_points = []
            for idx in naad_indices:
                lat_col = f'lat_NAAD_{idx}'
                lon_col = f'lon_NAAD_{idx}'
                if lat_col in data.columns and lon_col in data.columns:
                    valid_points = data[[lat_col, lon_col]].dropna()
                    naad_points.extend(valid_points.values.tolist())
            
            if naad_points:
                coverage = calculate_coverage(
                    data[['lat', 'lon']].dropna(),
                    pd.DataFrame(naad_points, columns=['lat', 'lon']),
                    MAX_DISTANCE_KM
                )
                space_cov.append(coverage)
        
        if space_cov:
            results.append({
                'year': year,
                'coverage_median': np.median(space_cov),
                'coverage_mean': np.mean(space_cov),
                'tracks_covered': len(space_cov),
                'tracks_total': len(TC_dict)
            })
    
    return pd.DataFrame(results)

def main():
    """Основная функция"""
    all_results = []
    
    for config in TRACKING_CONFIGS:
        df = analyze_configuration(config)
        df['config'] = config['description']
        all_results.append(df)
    
    # Объединение результатов
    comparison_df = pd.concat(all_results)
    
    # Сохранение результатов
    output_path = f"{setup_paths(DATA_TYPE, '', '', '', SIGMA)['output_folder']}/tracking_comparison.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    comparison_df.to_csv(output_path, index=False)
    
    # Вывод сводной таблицы
    summary = comparison_df.groupby('config').agg({
        'coverage_median': 'median',
        'coverage_mean': 'mean',
        'tracks_covered': 'sum',
        'tracks_total': 'sum'
    })
    summary['coverage_ratio'] = summary['tracks_covered'] / summary['tracks_total']
    
    print("\nСравнительные результаты:")
    print(summary.round(2))

if __name__ == "__main__":
    main()