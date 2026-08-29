import huracanpy

from matplotlib import pyplot as plt
import pandas as pd 
import numpy as np
import math
import xarray as xr
from numpy import linalg as LA

# from matplotlib import animation
import datetime
from datetime import timedelta

import scipy as sp
from scipy.ndimage import label, generate_binary_structure
from scipy import interpolate

import geopy.distance

import seaborn as sns

import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature

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

print('data type: ')
data_type = input() 

print('sigma: ')
sigma = int(input())


dist_m = 6

path_init = f'/storage/thalassa/users/vkoshkina'
path_dir_data = f'{path_init}/data/'
folder = f'{path_dir_data}/{data_type}/EddyClicker_txt'
filename = f'{folder}/tracks_2019.txt'

# Load the tracks with metadata
tracks_EC = huracanpy.load(
    filename,
    source="tempestextremes",
    variable_names=[
        'r2d',
        'rad',
    ],
)

# Отдельные списки параметров
tracking_types = [
    'tracking_local_2_phase', 
    'tracking_global_only', 
    'tracking_local_global',
    # 'tracking_local_only',
    'tracking_local_2_phase_cone',  # Раскомментировали
]

speed_options = [
    'adv_speed', 
    'no_speed', 
    'bg_speed', 
    'adv_bg_speed'
]

# Специальная конфигурация для cone теста
cone_config = {
    'tracking_type': 'tracking_local_2_phase_cone',
    'speed_option': 'avd_cone',  # специальный speed option для cone
    'pref_tracking': 'update_2026_cone_test'
}

pref_tracking_default = 'update_2026_wspd_hw_3'
path_init = f'/storage/thalassa/users/vkoshkina'
path_dir_data = f'{path_init}/data'



# Результаты будем сохранять в список
results = []

# Функция для загрузки треков
def load_tracks(path_data_dir):
    """Загружает годовые треки из указанной директории"""
    filename = f'{path_data_dir}/tracks_2019-01.txt'
    
    tracks_year = huracanpy.load(
        filename,
        source="tempestextremes",
        variable_names=[
            'r2d',
            'rad',
            'track_len',
        ],
    )
    
    # Загружаем остальные месяцы
    months = np.arange(2, 7)
    for month in months:
        filename = f'{path_data_dir}/tracks_2019-{month:02d}.txt'
        try:
            tracks1 = huracanpy.load(
                filename,
                source="tempestextremes",
                variable_names=[
                    'r2d',
                    'rad',
                    'track_len',
                ],
            )
            tracks_year = huracanpy.concat_tracks([tracks_year, tracks1])
        except FileNotFoundError:
            print(f"  Warning: File not found {filename}")
            continue
    
    return tracks_year
    
# # Функция для загрузки треков
# def load_tracks(path_data_dir):
#     # Словарь для хранения треков по годам
#     tracks_by_year = {}
#     all_tracks = []
    
#     years = np.arange(1979, 2019)
    
#     for year in tqdm(years, desc="Loading years"):
#         try:
#             # Загружаем январь
#             filename = f'{path_data_dir}/tracks_{year}-01.txt'
#             tracks_year = huracanpy.load(
#                 filename,
#                 source="tempestextremes",
#                 variable_names=['r2d', 'rad', 'track_len'],
#             )
            
#             # Загружаем остальные месяцы
#             for month in range(2, 13):
#                 if year == 2018 and month == 12:
#                     continue
#                 filename = f'{base_folder}/tracks_{year}-{month:02d}.txt'
#                 try:
#                     tracks_month = huracanpy.load(
#                         filename,
#                         source="tempestextremes",
#                         variable_names=['r2d', 'rad', 'track_len'],
#                     )
#                     tracks_year = huracanpy.concat_tracks([tracks_year, tracks_month])
#                 except FileNotFoundError:
#                     continue
            
#             # Сохраняем
#             tracks_by_year[year] = tracks_year
#             all_tracks.append(tracks_year)
            
#         except FileNotFoundError:
#             print(f"Year {year} not found, skipping...")
#             continue

#     tracks_all_years = huracanpy.concat_tracks(all_tracks)

#     return tracks_all_years

# Цикл по всем комбинациям для default pref_tracking
for tracking_type in tracking_types:
    for CVS_speed in speed_options:
        
        # Формируем путь к директории с треками
        results_dir = f"{tracking_type}_{CVS_speed}_{pref_tracking_default}"
        path_data_dir = f"{path_dir_data}/{data_type}/{data_type}_tracks_sigma_2/{pref_tracking_default}/{results_dir}/tracks_C/txt_yearly/"

        print(f"\nProcessing: {tracking_type} - {CVS_speed}")
        print(f"Path: {path_data_dir}")
        
        try:
            # Загружаем треки
            tracks_year = load_tracks(path_data_dir)

            tracks_year = huracanpy.trackswhere(
                tracks_year, tracks_year.track_id, lambda x: (x.lon < 100).all()   
            )
            
            # tracks_year = huracanpy.trackswhere(
            #     tracks_subset, tracks_subset.track_id, lambda x: (x.lat > 55).all()   
            # )
            
            # Сравниваем с tracks_EC
            matches = huracanpy.assess.match(
                [tracks_EC, tracks_year], 
                names=['manual', f'{tracking_type}_{CVS_speed}'], 
                max_dist=dist_m*5,                 
                # mean_dist=dist_m*3, 
                min_overlap=3, 
                tracks1_is_ref=True
            )
            
            # Вычисляем метрики
            pod = huracanpy.assess.pod(matches, ref=tracks_EC, ref_name='manual')
            # far = huracanpy.assess.far(matches, detected=tracks_year, detected_name=f'{tracking_type}_{CVS_speed}')
            
            # # Фильтруем matches для FAR_filtered
            # filtered_matches = matches.loc[matches.groupby('id_manual')['temp'].idxmax()]
            # filtered_far = huracanpy.assess.far(filtered_matches, detected=tracks_year, detected_name=f'{tracking_type}_{CVS_speed}')
            
            # Сохраняем результаты
            results.append({
                'tracking_type': tracking_type,
                'speed_option': CVS_speed,
                'pref_tracking': pref_tracking_default,
                'POD': pod,
                # 'FAR': far,
                # 'FAR_filtered': filtered_far,
                'total_matches': len(matches),
                'n_ref_tracks': len(tracks_EC),
                'n_auto_tracks': len(tracks_year)
            })
            
            print(f"  POD: {pod:.3f}")
            
        except Exception as e:
            print(f"  Error processing {tracking_type}_{CVS_speed}: {e}")
            results.append({
                'tracking_type': tracking_type,
                'speed_option': CVS_speed,
                'pref_tracking': pref_tracking_default,
                'POD': np.nan,
                # 'FAR': np.nan,
                # 'FAR_filtered': np.nan,
                'error': str(e)
            })

# Добавляем специальную конфигурацию из cone теста
print(f"\nProcessing SPECIAL CONFIG: {cone_config['tracking_type']} - {cone_config['speed_option']}")
print(f"Pref_tracking: {cone_config['pref_tracking']}")

try:
    # Формируем путь для cone теста
    results_dir = f"{cone_config['tracking_type']}_{cone_config['speed_option']}_{cone_config['pref_tracking']}"
    path_data_dir = f"{path_dir_data}/{data_type}/{data_type}_tracks_sigma_2/{cone_config['pref_tracking']}/{results_dir}/tracks_C/txt_yearly/"
    
    print(f"Path: {path_data_dir}")
    
    # Загружаем треки
    tracks_year = load_tracks(path_data_dir)
    
    # Сравниваем с tracks_EC
    matches = huracanpy.assess.match(
        [tracks_EC, tracks_year], 
        names=['manual', f"{cone_config['tracking_type']}_{cone_config['speed_option']}"], 
        max_dist=dist_m*5, 
        # mean_dist=dist_m*3, 
        min_overlap=3, 
        tracks1_is_ref=True
    )
    
    # Вычисляем метрики
    pod = huracanpy.assess.pod(matches, ref=tracks_EC, ref_name='manual')
    # far = huracanpy.assess.far(matches, detected=tracks_year, detected_name=f"{cone_config['tracking_type']}_{cone_config['speed_option']}")
    
    # # Фильтруем matches для FAR_filtered
    # filtered_matches = matches.loc[matches.groupby('id_manual')['temp'].idxmax()]
    # filtered_far = huracanpy.assess.far(filtered_matches, detected=tracks_year, detected_name=f"{cone_config['tracking_type']}_{cone_config['speed_option']}")
    
    # Сохраняем результаты
    results.append({
        'tracking_type': cone_config['tracking_type'],
        'speed_option': cone_config['speed_option'],
        'pref_tracking': cone_config['pref_tracking'],
        'POD': pod,
        # 'FAR': far,
        # 'FAR_filtered': filtered_far,
        'total_matches': len(matches),
        'n_ref_tracks': len(tracks_EC),
        'n_auto_tracks': len(tracks_year)
    })
    
    print(f"  POD: {pod:.3f}")
    
except Exception as e:
    print(f"  Error processing cone config: {e}")
    results.append({
        'tracking_type': cone_config['tracking_type'],
        'speed_option': cone_config['speed_option'],
        'pref_tracking': cone_config['pref_tracking'],
        'POD': np.nan,
        # 'FAR': np.nan,
        # 'FAR_filtered': np.nan,
        'error': str(e)
    })

# Создаем итоговую таблицу
results_df = pd.DataFrame(results)
print("\n" + "="*70)
print("FINAL RESULTS:")
print("="*70)
print(results_df[['tracking_type', 'speed_option', 'pref_tracking', 'POD']].to_string(index=False))

# Сохраняем таблицу в файл
output_dir = f'{path_dir_data}/tracking_comparison_results/{data_type}'
output_file = f'{output_dir}/{data_type}_tracking_comparison_results.csv'
Path(output_dir).mkdir(parents=True, exist_ok=True)

results_df.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")