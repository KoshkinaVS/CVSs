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

from plot_tracking_results_func_2026 import *

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


region = 'LV'

circ = 'AC'
circ = 'C_AC'
circ = 'C'


    
name_crit = 'R2D'
name_crit = 'Q'
# name_crit = 'lambda2'


postfix = '_range_025_144h_72h'


prefix = ''
# prefix = 'last_10_'


        

in_file_name_list = ['LV_alt_0125deg_2008-2009', 'LV_alt_025deg_2008-2009', 
                     'LV_alt_Novoselova_0125', 'LV_alt_Novoselova_025', 
                     'LV_alt_0125deg_2014', 'LV_alt_025deg_2014']



# nodes_dir = f"{sigma_dir}/Nodes"
# name_pics = f'{sigma_dir}/{prefix}Tracks_pics{postfix}'
# path_tracks_dir = f'{sigma_dir}/Tracks{postfix}'

### 2026-08-04
dt_step = 3600


size_filter = 10

year = 2008

for in_file_name in tqdm(in_file_name_list):
    
    ground_ds = xr.open_dataset(f'{path_dir_data}/ocean_eddies/LV/{in_file_name}.nc')['ugos']
    ground = np.where(np.isnan(ground_ds[0]), 1, np.nan)

    x_unit = 'longitude'
    y_unit = 'latitude'
    xxx = ground_ds[x_unit]
    yyy = ground_ds[y_unit]
    xx, yy = np.meshgrid(xxx, yyy)
    ground_ds.close()

    
    sigma_dir = f"{path_init_TE}/{data_type}/LV/{in_file_name}"
    
    nodes_dir = f"{sigma_dir}/{name_crit}_txt_files"
    path_tracks_dir = f"{sigma_dir}/Tracks_{name_crit}_txt_files{postfix}"
    name_pics = f'{sigma_dir}/{prefix}{circ}_Tracks_pics_{name_crit}_txt_files{postfix}'

    # filename = f"{path_tracks_dir}/{circ}_{data_type}_TC_tracks.txt"  # ваш файл с данными
    # CS_tracks_list = load_season_tracks(year, months, filename)

    
    path_DB = f'{path_dir_data}/ocean_eddies/LV_DBSCAN/DBSCAN_{eps:02d}-{min_samples:02d}-{size_filter:02d}_sigma_{sigma}/'
    ds = xr.open_dataset(f"{path_DB}/sigma_{sigma}_DBSCAN_{in_file_name}.nc")

    # local_extr = np.where(ds['local_extr_crit'] > 0, ds['local_extr_cluster'], np.nan)

    
    # local_extr = load_local_extrema_nodes(f'{nodes_dir}/{circ}_{in_file_name}_R2D_extr.txt')


    if circ == 'C':
        crit_vals = np.where(ds[crit] > 0, ds[crit], np.nan)
        local_extr = load_local_extrema_nodes(f'{nodes_dir}/{circ}_{in_file_name}_{name_crit}_extr.txt')
        filename = f"{path_tracks_dir}/{circ}_{data_type}_TC_tracks.txt"
        CS_tracks_list = load_season_tracks(year, months, filename)

        # Обрабатываем месяц с текущим индексом
        current_idx = plot_track_with_DBSCAN(
        ds, ds, dt_step, crit_vals, local_extr, 
        CS_tracks_list, 0, len(ds[time_unit]), 
        current_idx, name_pics
    )
        
    elif circ == 'AC':
        crit_vals = np.where(ds[crit] < 0, ds[crit], np.nan)
        local_extr = load_local_extrema_nodes(f'{nodes_dir}/{circ}_{in_file_name}_{name_crit}_extr.txt')
        filename = f"{path_tracks_dir}/{circ}_{data_type}_TC_tracks.txt"
        CS_tracks_list = load_season_tracks(year, months, filename)

        # Обрабатываем месяц с текущим индексом
        current_idx = plot_track_with_DBSCAN(
        ds, ds, dt_step, crit_vals, local_extr, 
        CS_tracks_list, 0, len(ds[time_unit]), 
        current_idx, name_pics
    )
        
    elif circ == 'C_AC':
        # Загружаем и объединяем экстремумы из двух файлов
        local_extr = load_combined_extrema(
            f'{nodes_dir}/C_{in_file_name}_{name_crit}_extr.txt',
            f'{nodes_dir}/AC_{in_file_name}_{name_crit}_extr.txt'
        )
        
        crit_vals = np.where(ds[crit] != 0, ds[crit], np.nan)
        
        # Загружаем треки отдельно для C и AC
        filename_C = f"{path_tracks_dir}/C_{data_type}_TC_tracks.txt"
        CS_tracks_list_C = load_season_tracks(year, months, filename_C)
        
        filename_AC = f"{path_tracks_dir}/AC_{data_type}_TC_tracks.txt"
        CS_tracks_list_AC = load_season_tracks(year, months, filename_AC)
        
        # Передаем оба списка в функцию
        current_idx = plot_track_with_DBSCAN(
            ds, ds, dt_step, crit_vals, local_extr, 
            CS_tracks_list_C, CS_tracks_list_AC,  # ← теперь два списка
            0, len(ds[time_unit]), 
            current_idx, name_pics
        )

        
    
