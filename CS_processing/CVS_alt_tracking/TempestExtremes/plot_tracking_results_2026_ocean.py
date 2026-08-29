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

region = 'Arctic'

region = 'BarKara'

postfix = ''
# postfix = '_range_1'
# postfix = '_range_2'
postfix = '_range_1_5_18h_12h'

# postfix = '_range_1_prioritize'

prefix = ''
# prefix = 'last_10_'


if region == 'Arctic':
    # Арктика
    level = 850
    region_name = f'Arctic_{level}hPa'
elif region == 'BarKara':
    region_name = f'BarKara_level_{level}'
    postfix = '_range_025_144h_72h'
    postfix = '_range_015_144h_72h'
    
    
else:
    # Атлантика
    level = 500
    level = 850
    
    region_name = f'NA_for_TC_{level}hPa'
        
sigma_dir = f"{path_init_TE}/{data_type}/R2D_{data_type}_{region_name}_sigma_{sigma}"



nodes_dir = f"{sigma_dir}/R2D_txt_files"
path_tracks_dir = f"{sigma_dir}/Tracks_R2D_txt_files{postfix}"
name_pics = f'{sigma_dir}/{prefix}Tracks_pics_R2D_txt_files{postfix}'


# nodes_dir = f"{sigma_dir}/Nodes"
# name_pics = f'{sigma_dir}/{prefix}Tracks_pics{postfix}'
# path_tracks_dir = f'{sigma_dir}/Tracks{postfix}'

### 2026-08-04
dt_step = 3600

filename = f"{path_tracks_dir}/{data_type}_TC_tracks_{year}.txt"  # ваш файл с данными

CS_tracks_list = load_season_tracks(year, [8,9], filename)

if data_type == 'GLORYS' or data_type == 'ALT':
    ds = xr.open_dataset(f"{path_dir_data}/{folder_name}/sigma_{sigma}_DBSCAN_{data_type}_level_{level}_{year}.nc")

    # local_extr = np.where(ds['local_extr_crit'] > 0, ds['local_extr_cluster'], np.nan)
    local_extr = load_local_extrema_nodes(f'{nodes_dir}/{data_type}_R2D_extr_{year}.txt')
    
    crit_vals = np.where(ds[crit] > 0, ds[crit], np.nan)
    # Обрабатываем месяц с текущим индексом
    current_idx = plot_track_with_DBSCAN(
        ds, ds, dt_step, crit_vals, local_extr, 
        CS_tracks_list, 0, len(ds[time_unit]), 
        current_idx, name_pics
    )