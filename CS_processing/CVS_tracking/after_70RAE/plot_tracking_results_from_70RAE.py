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

# path_init = f'/Volumes/KINGSTON/рейс/'
path_init = f'/Volumes/Backup Plus/70RAE/'

# caution: path[0] is reserved for script path (or '' in REPL)
# sys.path.insert(1, f'{path_init}/scripts/scripts_from_kubrik/vortex_identification')

# from vortex_dir import vortex_processing as vortex
# from vortex_dir import show_vortex as show_vx
# from vortex_dir import compute_criteria as compute

sys.path.insert(2, f'{path_init}/scripts/scripts_from_kubrik/DBSCAN_tracking/tracking_lib')

# from step_of_tracking_local_extrema import *
# tracking_type = 'tracking_local_extrema'

# from step_of_tracking_IoU import *
# tracking_type = 'tracking_local_extrema_IoU'

# from step_of_tracking_local_2_phase import *
# tracking_type = 'step_of_tracking_local_2_phase'

# from step_of_tracking_local_only import *
# tracking_type = 'step_of_tracking_local_only'

from step_of_tracking import *
tracking_type = 'tracking_local_2_phase'


# data_type = 'LoRes'

local_extr_name = 'local_extr_crit'

dist_m, our_level, x_unit, y_unit, u_unit, v_unit, time_unit, level_unit, crit, th, min_samples, eps, CS_points_th = get_init_params(data_type, json_file, print_info=True)

if data_type == 'HiRes' or data_type == 'LoRes':
    level = 12
elif data_type == 'ERA5':
    level = 9

path_init = f'/Volumes/Backup Plus/70RAE/'
path_data = f'{path_init}/data'  

# DBSCAN_name = f'DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_smoothing'
DBSCAN_name = f'DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_level_{level}_smoothing'


folder_name = f'{data_type}/{DBSCAN_name}'
name_init = f'sigma_2_DBSCAN_{data_type}'

our_level = 0

print('year: ')
year = int(input())
    
# path_tracks_dir = f'{path_data}/{data_type}_tracks/{tracking_type}/{DBSCAN_name}/tracks_{circ}/'

path_tracks_dir = f'{path_data}/{data_type}/{data_type}_tracks/{tracking_type}/DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_smoothing/tracks_{circ}/'

vmin = -127
vmax = 127

# folder = f'{path_raw_data}/{crit}_{data_type}'
# name_ds = f'rortex_2d_criteria_{data_type}_level_{level}'

name_pics = f'tracks_anim_{tracking_type}'


months = np.arange(1,13,1)

# plotting ground
# if data_type == 'HiRes':
#     ncfile = f'/Volumes/KINGSTON/рейс/data/raw_data/{data_type}/wrfout_d01_2010-01-01_00:00:00'
# elif data_type == 'LoRes':
#     ncfile = f'/Volumes/KINGSTON/рейс/data/raw_data/{data_type}/2010/wrfout_d01_2010-01-01_00:00:00'

# ground_ds = xr.open_dataset(f'{ncfile}')['HGT'][0]
# ground = np.where(ground_ds > 5, 1, np.nan)

def plot_ground(ax, ds, ground):
    mc = ax.contourf(ds[x_unit], ds[y_unit], ground, 
                cmap='grey', 
                alpha=0.07
                 )


def plot_crit(ax, ds, crit_vals, local_extr, t):

    mc = ax.contourf(ds[x_unit], ds[y_unit], crit_vals[t, our_level], 
                cmap='PiYG', 
                  vmin=vmin, vmax=vmax,
                 )

    plt.title(str(ds.XTIME[t].values)[:16])

    plt.xlim(0,len(ds[x_unit]))
    plt.ylim(0,len(ds[y_unit]))
    
    xx, yy = np.meshgrid(ds[x_unit], ds[y_unit])

    n_clusters = int(np.nanmax(ds['cluster'].values)+1)
    cmap = plt.cm.get_cmap(cmaps.psgcap, n_clusters)

    Q_center = ax.scatter(xx, yy,
                          c=local_extr[t,our_level],
                              s=30, 
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
            
            if np.sum(~np.isnan(CS['x'])) > 6:
                ax.plot(CS['x'][:stop_time+1], CS['y'][:stop_time+1], alpha=0.9, lw=0.5, c='k')

                plot_bounds(ax, CS, stop_time, color='blue')
                
                # if stop_time == 0:
                #     x, y, u, v, hw = get_mean_wspd(ds_speed, CS, stop_time, our_level)
                #     plot_mean_wspd(ax, x, y, u, v, hw, dt_step, dist_m=dist_m, color='blue')
                # else:
                #     x, y, u, v, hw = get_loc_CS_speed(CS, dt_step, stop_time)
                #     plot_mean_wspd(ax, x, y, u, v, hw, dt_step, dist_m=1, color='blue')

def plot_track_with_DBSCAN(ds, ds_speed, dt_step, crit_vals, local_extr, CS_tracks_list, start_time, stop_time, idx, name_pics, save=True):
    
    for t in tqdm(range(start_time, stop_time)):
        
        fig = plt.figure(figsize=(5, 5), dpi=150)
        ax = fig.add_subplot(111)
        
        ax.grid(which='major', linewidth=1.2)
        ax.grid(which='minor', linestyle='--', color='gray', linewidth=0.5)

        # plot_ground(ax, ds, ground)
        plot_crit(ax, ds, crit_vals, local_extr, t)
        
        time_plot_tracks(ax, ds_speed, dt_step, CS_tracks_list, ds[time_name].values[t])    

        if save:
            folder = f'{path_data}/pics/{data_type}/{name_pics}/{year}'
            
            if not os.path.exists(f"{folder}"):
                os.makedirs(f"{folder}")
            plt.savefig(f"{folder}/track_{(idx + t):05d}.png", 
                        dpi=200, bbox_inches="tight", transparent=False)
            plt.close()
        
    return idx + t + 1 

def load_season_tracks(year, months, path_data): 
    CS_tracks_list = []
    for month in months:
        if os.path.exists(path_data):
            ls = list(sorted(Path(f"{path_data}/{year}-{month:02d}").glob(f'*_track_*.csv')))
            # print(f'tracks: {ls}')
            
            if len(ls) != 0:
                for ii,ifile in enumerate(ls):
                    df = pd.read_csv(ifile, parse_dates=['datetime'])
                    df = df.drop(df.columns[0], axis=1)
#                     df = df.drop(df.columns[0], axis=1)
                    CS_tracks_list.append(df)
    return CS_tracks_list


idx = 0 

for month in months:

    r2d_folder_name = f'{data_type}/{crit}_{data_type}_level_{level}_smoothing'
    name_ds = f'sigma_2_R2D_{data_type}_level_12'
    uv_path = f'{path_data}/{r2d_folder_name}'

    if year == 2010:
        ds_speed = None
    else:
        ds_speed = xr.open_dataset(f"{uv_path}/{name_ds}_{year}-{month:02d}.nc")

    ds = xr.open_dataset(f"{path_data}/{folder_name}/{name_init}_{year}-{month:02d}.nc")

    local_extr_name = 'local_extr_cluster'
    
    local_extr = np.where(ds['local_extr_crit'] > 0, ds[local_extr_name], np.nan)
    crit_vals = np.where(ds[crit] > 0, ds[crit], np.nan)
    
    dt_step = np.timedelta64(ds[time_unit].values[1] - ds[time_unit].values[0], 's').astype(int)
    
    CS_tracks_list = load_season_tracks(year, months, path_tracks_dir)
    
    idx = plot_track_with_DBSCAN(ds, ds_speed, dt_step, crit_vals, local_extr, CS_tracks_list, 0, len(ds[time_unit]), idx, name_pics)