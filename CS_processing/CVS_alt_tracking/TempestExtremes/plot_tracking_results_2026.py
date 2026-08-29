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

from plot_tracking_results_func_2026_upd import *

months = np.arange(1,13,1)

if data_type == 'HiRes' or data_type == 'LoRes':
    level = 12
    dt_step = 3*3600
    
elif data_type == 'ERA5':
    level = 500
    dt_step = 1*3600
elif data_type == 'GLORYS':
    level = 8
    dt_step = 24*3600
elif data_type == 'ALT':
    level = 0
    dt_step = 24*3600
elif data_type == 'GPN':
    level = 12
    months = np.arange(2,3,1)
    dt_step = 1*3600
elif data_type == 'SMP':
    level = 10
    dt_step = 1*3600
    

path_dir_data = f'{path_init}/data'
if data_type == 'LoRes':
    path_dir_data = f'{path_dir_data}/LoRes'

# path_data = f'{path_init}/data'

path_init_TE = '/storage/thalassa/users/vkoshkina/data/TempestExtremes'

current_idx = 0


postfix = ''
# postfix = '_range_1'
# postfix = '_range_2'
postfix = '_range_1_5_18h_12h'

size_filter = 25
extr_type = '_global'
postfix = f'_range_1_5_18h_12h_2010_{size_filter}points{extr_type}'


# postfix = '_range_1_prioritize'

prefix = ''
# prefix = 'last_10_'

variables = ['rad', 'r2d', 'wspd']

        
sigma_dir = f"{path_init_TE}/{data_type}/R2D_{data_type}_{region_name}_sigma_{sigma}"



nodes_dir = f"{sigma_dir}/R2D_txt_files"
nodes_dir = f"{sigma_dir}/R2D_txt_files_2010_{size_filter}points{extr_type}"

path_tracks_dir = f"{sigma_dir}/Tracks_R2D_txt_files{postfix}"
name_pics = f'{sigma_dir}/{prefix}Tracks_pics_R2D_txt_files{postfix}'


# nodes_dir = f"{sigma_dir}/Nodes"
# name_pics = f'{sigma_dir}/{prefix}Tracks_pics{postfix}'
# path_tracks_dir = f'{sigma_dir}/Tracks{postfix}'

for month in months:
    CS_tracks_list = load_season_tracks(year, [month, month+1], f'{path_tracks_dir}/{data_type}_TC_tracks_{year}.txt', variables=variables)
    
    #### 2026-07-08 test
#     CS_tracks_list = load_season_tracks(year, [month, month+1], f'{sigma_dir}/{data_type}_TC_tracks.txt')
    
    
    if data_type == 'SMP' or data_type == 'GPN':
        num_days = calendar.monthrange(year, month)[1]
        
        for day in range(1, num_days + 1):

            ds = xr.open_dataset(f"{path_dir_data}/{folder_name}/{name_init}_{year}-{month:02d}-{day:02d}.nc")
        
            # local_extr = np.where(ds['local_extr_crit'] > 0, ds['local_extr_cluster'], np.nan)
            local_extr = load_local_extrema_nodes(f'{nodes_dir}/{data_type}_R2D_extr_{year}-{month:02d}-{day:02d}.txt',
                                                 variables=variables)
            
            crit_vals = np.where(ds[crit] > 0, ds[crit], np.nan)

            # Обрабатываем месяц с текущим индексом
            current_idx = plot_track_with_DBSCAN(
                ds, ds, dt_step, crit_vals, local_extr, 
                CS_tracks_list, 0, len(ds[time_unit]), 
                current_idx, name_pics
            )
            
            

    else:   
        
        if data_type == 'ERA5':

            # Загружаем данные по месяцам или дням
            datasets = []
            
            # Пробуем загрузить месячный файл
            monthly_file = f"{path_dir_data}/{folder_name}/{name_init}_{year}-{month:02d}.nc"
            try:
                ds = xr.open_dataset(monthly_file)
                datasets.append(ds)
            except FileNotFoundError:
                # Если нет месячного, ищем ежедневные
                import glob
                daily_files = sorted(glob.glob(f"{path_dir_data}/{folder_name}/{name_init}_{year}-{month:02d}-*.nc"))
                for daily_file in tqdm(daily_files):
                    try:
                        cluster_ds = xr.open_dataset(daily_file)
                        # Пересоздаем R2D с размерностью Time
                        if 'R2D' in cluster_ds.data_vars and 'time' in cluster_ds['R2D'].dims:
                            # Создаем новую переменную с размерностью Time
                            cluster_ds = cluster_ds.assign(
                                R2D_fixed=(['Time', 'level', 'latitude', 'longitude'], 
                                           cluster_ds['R2D'].values)
                            )
                            # Удаляем старую R2D
                            cluster_ds = cluster_ds.drop_vars('R2D')
                            # Переименовываем исправленную переменную
                            cluster_ds = cluster_ds.rename({'R2D_fixed': 'R2D'})

                        # Удаляем пустую размерность time
                        if 'time' in cluster_ds.dims:
                            cluster_ds = cluster_ds.drop_dims('time')
    
                        datasets.append(cluster_ds)
                    except:
                        continue

            if datasets:
                ds = xr.concat(datasets, dim='Time')
                print(ds)
            else:
                print("Не найдено ни одного файла данных!")
                sys.exit(1)
        
        else:
            
            ds = xr.open_dataset(f"{path_dir_data}/{folder_name}/{name_init}_{year}-{month:02d}.nc")
        
        # local_extr = np.where(ds['local_extr_crit'] > 0, ds['local_extr_cluster'], np.nan)
        local_extr = load_local_extrema_nodes(f'{nodes_dir}/{data_type}_R2D_extr_{year}-{month:02d}.txt', variables=variables)
        
        crit_vals = np.where(ds[crit] > 0, ds[crit], np.nan)
        
        # Обрабатываем месяц с текущим индексом
        current_idx = plot_track_with_DBSCAN(
            ds, ds, dt_step, crit_vals, local_extr, 
            CS_tracks_list, 0, len(ds[time_unit]), 
            current_idx, name_pics, show_radius=True
        )