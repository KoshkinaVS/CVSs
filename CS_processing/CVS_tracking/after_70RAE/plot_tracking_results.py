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
    # level = 9
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
# pref_tracking = 'rad_and_bound'
pref_tracking = 'all_points_bound'

path_dir_data = f'{path_init}/data'
if data_type == 'LoRes':
    path_dir_data = f'{path_dir_data}/LoRes'

# path_data = f'{path_init}/data'


# Глобальный счетчик индексов для каждого конфига
config_counters = defaultdict(lambda: 1)

def process_config_month(args):
    """Обрабатывает все месяцы для одного конфига последовательно"""
    tracking_type, CVS_speed = args
    config_key = f"{tracking_type}_{CVS_speed}"

    
    results = []
    
    # Прогресс-бар для этого конфига
    pbar = tqdm(months, 
                # desc=f"Конфиг {config_key}", 
                leave=False)

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

    for month in pbar:

        CS_tracks_list = load_season_tracks(year, [month, month+1], path_tracks_dir)
        
        # Получаем текущий индекс для этого конфига
        current_idx = config_counters[config_key]

        if data_type == 'SMP' or 'GPN':
            num_days = calendar.monthrange(year, month)[1]
            
            for day in range(1, num_days + 1):
    
                ds = xr.open_dataset(f"{path_dir_data}/{folder_name}/{name_init}_{year}-{month:02d}-{day:02d}.nc")
            
                local_extr = np.where(ds['local_extr_crit'] > 0, ds['local_extr_cluster'], np.nan)
                crit_vals = np.where(ds[crit] > 0, ds[crit], np.nan)
    
                # Обрабатываем месяц с текущим индексом
                new_idx = plot_track_with_DBSCAN(
                    ds, ds, dt_step, crit_vals, local_extr, 
                    CS_tracks_list, 0, len(ds[time_unit]), 
                    current_idx, name_pics
                )

                # Обновляем счетчик для этого конфига
                config_counters[config_key] = new_idx
                pbar.set_postfix_str(f"Индекс: {current_idx}→{new_idx}")
                
                results.append((config_key, month, current_idx, new_idx))
                
        else:   
            ds = xr.open_dataset(f"{path_dir_data}/{folder_name}/{name_init}_{year}-{month:02d}.nc")
            
            local_extr = np.where(ds['local_extr_crit'] > 0, ds['local_extr_cluster'], np.nan)
            crit_vals = np.where(ds[crit] > 0, ds[crit], np.nan)
            
            # Обрабатываем месяц с текущим индексом
            new_idx = plot_track_with_DBSCAN(
                ds, ds, dt_step, crit_vals, local_extr, 
                CS_tracks_list, 0, len(ds[time_unit]), 
                current_idx, name_pics
            )

            # Обновляем счетчик для этого конфига
            config_counters[config_key] = new_idx
            pbar.set_postfix_str(f"Индекс: {current_idx}→{new_idx}")
            
            results.append((config_key, month, current_idx, new_idx))
    
    pbar.close()
    return results

def parallel_process_configs(year, months, tracking_types, speed_options):
    """Параллельная обработка конфигов с сохранением порядка месяцев"""
    configs = list(product(tracking_types, speed_options))
    
    # Главный прогресс-бар
    main_pbar = tqdm(total=len(configs), desc="Общий прогресс", position=0)
    
    def update_progress(future):
        main_pbar.update(1)
        main_pbar.refresh()
    
    results = []
    with ProcessPoolExecutor(max_workers=min(len(configs), os.cpu_count()-1)) as executor:
        # Запускаем каждый конфиг в отдельном процессе
        futures = [executor.submit(process_config_month, config) for config in configs]
        
        # Добавляем callback для обновления прогресса
        for future in futures:
            future.add_done_callback(update_progress)
        
        # Собираем результаты по мере готовности
        for future in as_completed(futures):
            results.extend(future.result())
    
    main_pbar.close()
    
    # Проверка уникальности индексов
    index_ranges = {}
    for config in configs:
        config_key = f"{config[0]}_{config[1]}"
        index_ranges[config_key] = (
            min(idx for (key, _, idx, _) in results if key == config_key),
            max(idx for (key, _, _, idx) in results if key == config_key)
        )
        print(f"{config_key}: индексы {index_ranges[config_key][0]}-{index_ranges[config_key][1]}")
    
    return results


results = parallel_process_configs(year, months, tracking_types, speed_options)

    
# for month in months:

#     ds = xr.open_dataset(f"{path_data}/{folder_name}/{name_init}_{year}-{month:02d}.nc")

#     local_extr_name = 'local_extr_cluster'
    
#     local_extr = np.where(ds['local_extr_crit'] > 0, ds[local_extr_name], np.nan)
#     crit_vals = np.where(ds[crit] > 0, ds[crit], np.nan)

#     for tracking_type, CVS_speed in product(tracking_types, speed_options):
    
#         results_dir = f"{tracking_type}_{CVS_speed}_crit_sorted"
#         path_tracks_dir = f'{path_data}/{data_type}/{data_type}_tracks/{results_dir}/{DBSCAN_name}/tracks_{circ}/'
#         name_pics = f'tracks_anim_{tracking_type}_{CVS_speed}'
            
#         CS_tracks_list = load_season_tracks(year, months, path_tracks_dir)
        
#         idx = plot_track_with_DBSCAN(ds, ds, dt_step, crit_vals, local_extr, CS_tracks_list, 0, len(ds[time_unit]), idx, name_pics)