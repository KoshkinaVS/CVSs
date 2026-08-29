from matplotlib import pyplot as plt
import pandas as pd 
import numpy as np
import math
import xarray as xr

import datetime
from datetime import timedelta

# import scipy as sp
# from scipy.ndimage import label, generate_binary_structure
# from scipy import interpolate

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

# Инициализация путей и параметров
path_init = f'/storage/thalassa/users/vkoshkina'
folder = 'scripts/CS_processing/CVS_tracking/after_70RAE/'
sys.path.insert(2, f'{path_init}/{folder}')

from step_of_tracking import *




local_extr_name = 'local_extr_crit'

dist_m, our_level, level, x_unit, y_unit, u_unit, v_unit, time_unit, level_unit, crit, th, min_samples, eps, CS_points_th = get_init_params(data_type, json_file, print_info=True)

path_data = f'{path_init}/data'
if data_type == 'LoRes':
    path_data = f'{path_data}/LoRes'
    
DBSCAN_name_tracks = f'DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_level_{level}_smoothing_sigma_{sigma}'


if data_type == 'ERA5':
    DBSCAN_name = f'DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_sigma_{sigma}'
    
    if sigma != 0:
        name_init = f'sigma_{sigma}_DBSCAN_DBSCAN_{data_type}_level_{level}'
    else:
        name_init = f'sigma_{sigma}_DBSCAN_{data_type}_level_{level}'
elif data_type == 'GPN':
    name_init = f'sigma_{sigma}_DBSCAN_{data_type}_level_{level}'
    DBSCAN_name = f'DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_sigma_{sigma}'
else:        
    name_init = f'sigma_{sigma}_DBSCAN_{data_type}_level_{level}'
    DBSCAN_name = f'DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_smoothing_sigma_{sigma}'

folder_name = f'{data_type}/{DBSCAN_name}'


our_level = 0

print('year: ')
year = int(input())


if data_type == 'GPN':
    folder_name = f'{data_type}/{DBSCAN_name}/{year}'
else:
    folder_name = f'{data_type}/{DBSCAN_name}'

vmin = -127
vmax = 127
if data_type == 'LoRes':
    crit_scale = 0.7
    star_size = 30
elif data_type == 'HiRes':
    crit_scale = 0.1
    star_size = 5
elif data_type == 'ERA5':
    crit_scale = 0.1
    star_size = 5
elif data_type == 'GPN':
    crit_scale = 0.7
    star_size = 5
    

if data_type != 'ERA5':
    if data_type == 'GPN':
        path_dir_raw = f'/storage/buffer/GPN/OUTPUTS/WRF6km/{year}'
        ncfile = f'{path_dir_raw}/wrfout_d01_{year}-02-01_00:00:00'
        
    elif data_type == 'HiRes':
        path_dir_raw = f'/storage/NAAD/NAAD/{data_type}/1979'
        ncfile = f'{path_dir_raw}/wrfout_d01_1979-02-01_00:00:00'
        
    else:
        path_dir_raw = f'/storage/NAAD/NAAD/{data_type}/{year}'
        ncfile = f'{path_dir_raw}/wrfout_d01_{year}-02-01_00:00:00'
    
    ground_ds = xr.open_dataset(f'{ncfile}')['HGT'][0]
    ground = np.where(ground_ds > 5, 1, np.nan)
    xxx = ground_ds[x_unit]
    yyy = ground_ds[y_unit]
    xx, yy = np.meshgrid(xxx, yyy)
    
    ground_ds.close()
else:
    path_dir_raw = f'/storage/thalassa/users/vkoshkina/data/ERA5'
    ncfile = f'{path_dir_raw}/ERA5_lsm_cropped.nc'
    
    ground_ds = xr.open_dataset(f'{ncfile}')['var172'][0]
    ground = np.where(ground_ds > 0.5, 1, np.nan)
    xxx = np.arange(len(ground_ds['lon']))
    yyy = np.arange(len(ground_ds['lat']))

    xx, yy = np.meshgrid(xxx, yyy)
    
    ground_ds.close()

def plot_ground(ax, ground):
    mc = ax.contourf(xxx, yyy, ground, 
                cmap='grey', 
                alpha=0.07
                 )

def plot_crit(ax, ds, crit_vals, local_extr, t):

    mc = ax.contourf(xxx, yyy, crit_vals[t, our_level], 
                cmap='PiYG', 
                  vmin=crit_scale*vmin, vmax=crit_scale*vmax,
                 )

    plt.title(str(ds[time_name][t].values)[:16])

    plt.xlim(0,len(xxx))
    plt.ylim(0,len(yyy))
    
    n_clusters = int(np.nanmax(ds['cluster'].values)+1)
    cmap = plt.cm.get_cmap(cmaps.psgcap, n_clusters)

    Q_center = ax.scatter(xx, yy,
                          c=local_extr[t,our_level],
                              s=star_size, 
                          # cmap=cmaps.temp_diff_1lev_r,
                          # vmin=-1,vmax=1,
                          cmap=cmap,
                          marker='*', alpha=1., label='center',
                          zorder=15,
                         )

    plt.grid(linestyle = ':', linewidth = 0.5)
    plt.tick_params(axis='both', which='both', direction='in', labelsize=9)

    
# перемещение в новую точку исходя из скорости КС между предыдущими шагами
def get_loc_CS_speed(CS, dt, i):

    hw = (CS['rad'][i])
    
    y = (CS['y'][i])
    x = (CS['x'][i])
    
    y_pr = CS['y'][i-1]
    x_pr = CS['x'][i-1]
    
    u = (x - x_pr)/dt
    v = (y - y_pr)/dt

    return x, y, u, v, hw

def compute_mean_uv_for_track(ds, t, y_in, y_out, x_in, x_out, our_level):
        
    u = np.nanmean(ds[u_unit][t, our_level, 
                          y_in:y_out, x_in:x_out])
    v = np.nanmean(ds[v_unit][t, our_level, 
                          y_in:y_out, x_in:x_out])
    
    return u, v

def get_mean_wspd(ds, CS, i, level):
    
    hw = int(np.round(CS['rad'][i]))
    
    y = CS['y'][i]
    x = CS['x'][i]

    y_int = int(y)
    x_int = int(x)

    if ds == None:
        u = 0
        v = 0
    else:
        
        y_in = y_int - hw
        x_in = x_int - hw
        
        y_out = y_int + hw
        x_out = x_int + hw
        
        if y_in < 0:
            y_in = 0
        if x_in < 0:
            x_in = 0
                
        if y_out >= len(ds[y_unit]):
            y_out = -1
        if x_out >= len(ds[x_unit]):
            x_out = -1

        u, v = compute_mean_uv_for_track(ds, CS['t'][i], y_in, y_out, x_in, x_out, our_level=level)
    
    return x_int, y_int, u, v, hw

def plot_mean_wspd(ax, x, y, u, v, hw, dt, dist_m, color='blue'):
    
    ax.plot([x, x+u/dist_m*dt], [y, y+v/dist_m*dt], color)
    
    p = mpatches.Rectangle(xy=(x - hw/2,y - hw/2), width=hw, height=hw, linewidth=0.5, 
                                     edgecolor='blue', facecolor='none',)
    ax.add_patch(p)

def plot_bounds(ax, CS, i, color='blue'):

    hw = int(np.round(CS['rad'][i]))*1.5
    
    y = CS['y'][i]
    x = CS['x'][i]

    p = mpatches.Rectangle(xy=(x - hw/2,y - hw/2), width=hw, height=hw, linewidth=0.5, 
                                     edgecolor='blue', facecolor='none',)
    ax.add_patch(p)

def time_plot_tracks(ax, ds_speed, dt_step, CS_tracks_list_100, our_xtime):

    # print(our_xtime)
    for CS in CS_tracks_list_100:
       
        if np.isin(our_xtime, CS['datetime']).any():
            
            stop_time = np.argwhere(np.array(CS['datetime']) == our_xtime)[0][0]
            
            if np.sum(~np.isnan(CS['x'])) > 3:
                ax.plot(CS['x'][:stop_time+1], CS['y'][:stop_time+1], alpha=0.9, lw=0.5, c='k')

                plot_bounds(ax, CS, stop_time, color='blue')
                
                # if stop_time == 0:
                #     x, y, u, v, hw = get_mean_wspd(ds_speed, CS, stop_time, our_level)
                #     plot_mean_wspd(ax, x, y, u, v, hw, dt_step, dist_m=dist_m, color='blue')
                # else:
                #     x, y, u, v, hw = get_loc_CS_speed(CS, dt_step, stop_time)
                #     plot_mean_wspd(ax, x, y, u, v, hw, dt_step, dist_m=1, color='blue')

# def plot_track_with_DBSCAN(ds, ds_speed, dt_step, crit_vals, local_extr, CS_tracks_list, start_time, stop_time, idx, name_pics, save=True):
    
#     for t in (range(start_time, stop_time)):
        
#         fig = plt.figure(figsize=(5, 5), dpi=150)
#         ax = fig.add_subplot(111)
        
#         ax.grid(which='major', linewidth=1.2)
#         ax.grid(which='minor', linestyle='--', color='gray', linewidth=0.5)

#         plot_ground(ax, ground)
#         plot_crit(ax, ds, crit_vals, local_extr, t)
        
#         time_plot_tracks(ax, ds_speed, dt_step, CS_tracks_list, ds[time_name].values[t])    
        
#         if save:
#             folder = f'{path_data}/pics/{data_type}/{name_pics}/{year}'
            
#             if not os.path.exists(f"{folder}"):
#                 os.makedirs(f"{folder}")
#             plt.savefig(f"{folder}/track_{(idx + t):05d}.png", 
#                         dpi=200, bbox_inches="tight", transparent=False)
#             plt.close()
        
#     return idx + t + 1 


def plot_track_with_DBSCAN(ds, ds_speed, dt_step, crit_vals, local_extr, 
                          CS_tracks_list, start_time, stop_time, idx, 
                          name_pics, save=True):
    """
    Визуализация треков с прогресс-баром
    
    Параметры:
        ... (ваши существующие параметры)
        name_pics: str - имя для подписи прогресс-бара
    """
    # Настройка прогресс-бара
    pbar = tqdm(
        range(start_time, stop_time),
        desc=f"Отрисовка {name_pics}",
        leave=False,  # Не оставлять бар после завершения
        position=getattr(plot_track_with_DBSCAN, 'position', 0)  # Для параллелизма
    )
    
    last_t = start_time
    for t in pbar:
        fig = plt.figure(figsize=(5, 5), dpi=150)
        ax = fig.add_subplot(111)
        
        ax.grid(which='major', linewidth=1.2)
        ax.grid(which='minor', linestyle='--', color='gray', linewidth=0.5)

        # if data_type != 'ERA5':
        #     plot_ground(ax, ground)
        plot_ground(ax, ground)
            
        plot_crit(ax, ds, crit_vals, local_extr, t)
        time_plot_tracks(ax, ds_speed, dt_step, CS_tracks_list, ds[time_name].values[t])    
        
        if save:
            folder = f'{path_init}/data/pics/{data_type}_pics/{name_pics}/{year}'
            os.makedirs(folder, exist_ok=True)
            plt.savefig(
                f"{folder}/track_{(idx + t):05d}.png", 
                dpi=200, 
                bbox_inches="tight", 
                transparent=False
            )
            plt.close()
        
        last_t = t
        pbar.set_postfix_str(f"Индекс: {idx + t}")
    
    return idx + (stop_time - start_time)

def load_season_tracks(year, months, path_data): 
    CS_tracks_list = []
    # print(path_data)
    for month in months:
        if os.path.exists(path_data):
            ls = list(sorted(Path(f"{path_data}/{year}-{month:02d}").glob(f'*_track_*.csv')))
            
            if len(ls) != 0:
                for ii,ifile in enumerate(ls):
                    df = pd.read_csv(ifile, parse_dates=['datetime'])
                    df = df.drop(df.columns[0], axis=1)
#                     df = df.drop(df.columns[0], axis=1)
                    CS_tracks_list.append(df)
        # print(f'Tracks for {year}-{month:02d} loaded ({len(CS_tracks_list)} tracks now)')
    return CS_tracks_list