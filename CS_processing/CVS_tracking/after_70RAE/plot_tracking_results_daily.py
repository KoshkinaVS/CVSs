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

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from collections import defaultdict
from itertools import product

import calendar

# Инициализация путей и параметров
path_init = f'/storage/thalassa/users/vkoshkina'
folder = 'scripts/CS_processing/CVS_tracking/after_70RAE/'
sys.path.insert(2, f'{path_init}/{folder}')

from step_of_tracking import *
from plot_tracking_results_func import *

local_extr_name = 'local_extr_crit'

months = np.arange(1,13,1)

if data_type == 'HiRes' or data_type == 'LoRes':
    level = 12
elif data_type == 'ERA5':
    level = 500
elif data_type == 'GPN':
    level = 12
    months = np.arange(2,3,1)

# Отдельные списки параметров
tracking_types = [
    'tracking_local_global', 
    'tracking_local_2_phase', 
    'tracking_global_only',
    'tracking_local_only',
]
speed_options = ['adv_speed', 'no_speed', 'bg_speed', 'adv_bg_speed']

### pref of tracking ###
pref_tracking = 'all_points_bound'

path_dir_data = f'{path_init}/data'
if data_type == 'LoRes':
    path_dir_data = f'{path_dir_data}/LoRes'

def calculate_total_frames_per_config(year, months, data_type):
    """Вычисляет общее количество кадров для каждого конфига"""
    total_frames = {}
    configs = list(product(tracking_types, speed_options))
    
    for tracking_type, CVS_speed in configs:
        config_key = f"{tracking_type}_{CVS_speed}"
        total_frames[config_key] = 0
        
        for month in months:
            if data_type in ['SMP', 'GPN']:
                num_days = calendar.monthrange(year, month)[1]
                for day in range(1, num_days + 1):
                    try:
                        ds = xr.open_dataset(f"{path_dir_data}/{folder_name}/{name_init}_{year}-{month:02d}-{day:02d}.nc")
                        total_frames[config_key] += len(ds[time_unit])
                        ds.close()
                    except:
                        continue
            else:
                try:
                    ds = xr.open_dataset(f"{path_dir_data}/{folder_name}/{name_init}_{year}-{month:02d}.nc")
                    total_frames[config_key] += len(ds[time_unit])
                    ds.close()
                except:
                    continue
    
    return total_frames

def process_config_month(args, frame_offsets):
    """Обрабатывает все месяцы для одного конфига последовательно"""
    tracking_type, CVS_speed, month = args
    config_key = f"{tracking_type}_{CVS_speed}"
    
    results_dir = f"{tracking_type}_{CVS_speed}_{pref_tracking}"

    if data_type == 'HiRes':
        folder_NAAD = f'{path_data}/TC_tracks/my_tracking_results/{pref_tracking}/{results_dir}/{data_type}_tracks_sigma_{sigma}/'
    elif data_type == 'ERA5':
        folder_NAAD = f'{path_data}/TC_tracks/{data_type}/my_tracking_results_2010/{pref_tracking}/{results_dir}/{data_type}_tracks_sigma_{sigma}/'
    elif data_type == 'GPN':
        folder_NAAD = f"{path_dir_data}/{data_type}/{data_type}_tracks_sigma_{sigma}/{pref_tracking}/{results_dir}/"
    else:
        folder_NAAD = f'{path_dir_data}/{data_type}/{data_type}_tracks/{pref_tracking}/{results_dir}/{DBSCAN_name_tracks}'
    
    path_tracks_dir = f'{folder_NAAD}/tracks_{circ}'
    name_pics = f'tracks_anim_{pref_tracking}/{config_key}_{pref_tracking}/{DBSCAN_name_tracks}'

    # Загрузка треков
    CS_tracks_list = load_season_tracks(year, [month], path_tracks_dir)
    
    # Определяем начальный индекс для этого месяца
    start_idx = frame_offsets[config_key][month]
    
    if data_type in ['SMP', 'GPN']:
        num_days = calendar.monthrange(year, month)[1]
        current_idx = start_idx
        
        for day in range(1, num_days + 1):
            try:
                ds = xr.open_dataset(f"{path_dir_data}/{folder_name}/{name_init}_{year}-{month:02d}-{day:02d}.nc")
                local_extr = np.where(ds['local_extr_crit'] > 0, ds['local_extr_cluster'], np.nan)
                crit_vals = np.where(ds[crit] > 0, ds[crit], np.nan)
                
                # Обрабатываем день
                new_idx = plot_track_with_DBSCAN(
                    ds, ds, dt_step, crit_vals, local_extr, 
                    CS_tracks_list, 0, len(ds[time_unit]), 
                    current_idx, name_pics
                )
                
                current_idx = new_idx
                ds.close()
                
            except Exception as e:
                print(f"Ошибка при обработке {year}-{month:02d}-{day:02d}: {e}")
                continue
    else:
        try:
            ds = xr.open_dataset(f"{path_dir_data}/{folder_name}/{name_init}_{year}-{month:02d}.nc")
            local_extr = np.where(ds['local_extr_crit'] > 0, ds['local_extr_cluster'], np.nan)
            crit_vals = np.where(ds[crit] > 0, ds[crit], np.nan)
            
            # Обрабатываем месяц
            current_idx = plot_track_with_DBSCAN(
                ds, ds, dt_step, crit_vals, local_extr, 
                CS_tracks_list, 0, len(ds[time_unit]), 
                start_idx, name_pics
            )
            
            ds.close()
            
        except Exception as e:
            print(f"Ошибка при обработке {year}-{month:02d}: {e}")
            current_idx = start_idx
    
    return (config_key, month, start_idx, current_idx)

def parallel_process_configs(year, months, tracking_types, speed_options):
    """Параллельная обработка конфигов с правильной индексацией"""
    # Сначала вычисляем смещения для каждого конфига и месяца
    frame_offsets = defaultdict(dict)
    configs = list(product(tracking_types, speed_options))
    
    # Вычисляем общее количество кадров для планирования индексации
    total_frames_per_config = calculate_total_frames_per_config(year, months, data_type)
    
    # Создаем задачи для каждого конфига и месяца
    tasks = []
    for tracking_type, CVS_speed in configs:
        config_key = f"{tracking_type}_{CVS_speed}"
        current_offset = 0
        
        for month in months:
            frame_offsets[config_key][month] = current_offset
            
            # Вычисляем количество кадров в этом месяце
            if data_type in ['SMP', 'GPN']:
                month_frames = 0
                num_days = calendar.monthrange(year, month)[1]
                for day in range(1, num_days + 1):
                    try:
                        ds = xr.open_dataset(f"{path_dir_data}/{folder_name}/{name_init}_{year}-{month:02d}-{day:02d}.nc")
                        month_frames += len(ds[time_unit])
                        ds.close()
                    except:
                        continue
            else:
                try:
                    ds = xr.open_dataset(f"{path_dir_data}/{folder_name}/{name_init}_{year}-{month:02d}.nc")
                    month_frames = len(ds[time_unit])
                    ds.close()
                except:
                    month_frames = 0
            
            current_offset += month_frames
            tasks.append((tracking_type, CVS_speed, month))
    
    # Главный прогресс-бар
    main_pbar = tqdm(total=len(tasks), desc="Общий прогресс", position=0)
    
    results = []
    with ProcessPoolExecutor(max_workers=min(len(tasks), os.cpu_count()-1)) as executor:
        # Создаем partial функцию с frame_offsets
        process_func = partial(process_config_month, frame_offsets=frame_offsets)
        
        # Запускаем задачи
        futures = {executor.submit(process_func, task): task for task in tasks}
        
        # Собираем результаты
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                task = futures[future]
                print(f"Ошибка при обработке {task}: {e}")
            finally:
                main_pbar.update(1)
    
    main_pbar.close()
    
    # Анализ результатов
    for config_key in set([f"{t}_{s}" for t, s in configs]):
        config_results = [r for r in results if r[0] == config_key]
        if config_results:
            start_idx = min(r[2] for r in config_results)
            end_idx = max(r[3] for r in config_results)
            print(f"{config_key}: индексы {start_idx}-{end_idx}")
    
    return results

# Запуск обработки
results = parallel_process_configs(year, months, tracking_types, speed_options)