import pandas as pd 
import numpy as np
import math
import xarray as xr
from numpy import linalg as LA

import datetime
from datetime import timedelta

import scipy as sp

import json

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import DBSCAN
from skimage.feature import peak_local_max

from scipy.spatial import cKDTree

from pathlib import Path

import sys
import os

# path_init = f'/storage/kubrick/vkoshkina/scripts/DBSCAN_tracking'
# path_init = f'/Volumes/KINGSTON/рейс/scripts/2025_kubrik/DBSCAN_tracking'
path_init = f'/Volumes/Backup Plus/70RAE/scripts/2025_kubrik/DBSCAN_tracking'

folder = 'after_70RAE/tracking_2025'
sys.path.insert(2, f'{path_init}/{folder}')

json_file = f'{path_init}/after_70RAE/tracking_init.json'




def save_track_csv(cluster_idx, TC, path_data_dir):
    df_TC_Danielle = pd.DataFrame({
                             "t": TC['t'][:-1],
                             "datetime": TC['time'][:-1], 
                             "x": TC['x'][:-1],
                             "y": TC['y'][:-1],
                             "lat": TC['lat'][:-1], 
                             "lon": TC['lon'][:-1], 
                             "rad": TC['rad'][:-1], 
                             "crit": TC['crit'][:-1], 
                             "track_len": TC['track_len'][:-1],
#                                "wspd": TC_wspd,
        })

    
    start_date = TC['time'][0]
    # cluster = TC['cluster']
    dt = str(start_date)[:-16]

    df_TC_Danielle.to_csv(f'{path_data_dir}/{cluster_idx:06d}_track_{dt}.csv')

    return cluster_idx

def load_init_data(file_path):
    """
    Loads initialization data from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Initialization data.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file {file_path}")
        return None


def get_init_params(data_type, json_file, print_info=False):

    init_data = load_init_data(json_file)

    if init_data is not None:
        # Use the loaded data
        # data_type = init_data['data_type']
        min_samples = init_data['min_samples']
        eps = init_data['eps']
        CS_points_th = init_data['CS_points_th']

        crit = init_data['crit']
        th = init_data['th']

        x_unit = init_data['x_unit']
        y_unit = init_data['y_unit']

        if print_info:
            print(f'data type: {data_type}')
            print(f'min_samples={min_samples}, eps={eps}, CS_points_th={CS_points_th}')
    else:
        print('Error while reading init json')
    

    if data_type == 'HiRes':
        dist_m = 13897.18
        our_level = 0
        level = 12
        u_unit = 'ue'
        v_unit = 've'
        time_unit = 'Time'
        level_unit = 'interp_level'
        

    elif data_type == 'ERA5':
        dist_m = 13897.18
        our_level = 0
        level = 9

        u_unit = 'U'
        v_unit = 'V'
        time_unit = 'time'
        level_unit = 'interp_level'
        

    else:    
        dist_m = 77824.23
        our_level = 0
        u_unit = 'ue'
        v_unit = 've'
        time_unit = 'Time'
        # time_unit = 'XTIME'
        
        level_unit = 'interp_level'

    return dist_m, our_level, x_unit, y_unit, u_unit, v_unit, time_unit, level_unit, crit, th, min_samples, eps, CS_points_th

print('data type: ')
data_type = input() 

circ = 'C'


speed_level = 0 # from file in 12 level
# speed_level = 12 # from raw file in 12 level


dist_m, our_level, x_unit, y_unit, u_unit, v_unit, time_unit, level_unit, crit, th, min_samples, eps, CS_points_th = get_init_params(data_type, json_file)

time_name = 'XTIME'

def get_all_coords(ds, t, circ):    
    ds_mini = ds.loc[{time_unit: ds[time_unit].values[t],
                          level_unit: 500.0
                     }][['cluster', 'R2D']]

    X_arr = ds_mini.to_dataframe().dropna(how='any')

    coords = X_arr['cluster'].index.to_frame(name=['y', 'x'], index=False)
    coords['cluster'] = X_arr['cluster'].values
    coords['crit'] = X_arr['R2D'].values

    if circ == 'C':
        coords = coords[coords['cluster'] > 0]
    else:
        coords = coords[coords['cluster'] < 0]

    ## sorted stat
    coords = coords.reset_index(drop=True)

    return coords
    
def get_stat_local_max(ds, t, circ):    
    ds_mini = ds.loc[{time_unit: ds[time_unit].values[t], 
                          level_unit: 500.0
                     }][['local_extr_crit', 'local_extr_cluster', 'local_extr_rad_eff']]

    X_arr = ds_mini.to_dataframe().dropna(how='any')

    coords = X_arr['local_extr_crit'].index.to_frame(name=['y', 'x'], index=False)
    coords['crit'] = X_arr['local_extr_crit'].values  
    coords['cluster'] = X_arr['local_extr_cluster'].values  
    coords['rad_eff'] = X_arr['local_extr_rad_eff'].values  
    coords = coords.dropna(how='any')

    if circ == 'C':
        coords = coords[coords['crit'] > 0]
    else:
        coords = coords[coords['crit'] < 0]

    ## sorted stat
    coords = coords.sort_values(by='crit', ascending=False).reset_index(drop=True)
    
    return coords

def get_stat_global_max(ds, t, circ):    
    ds_mini = ds.loc[{time_unit: ds[time_unit].values[t], 
                          level_unit: 500.0
                     }]

    X_arr = ds_mini.to_dataframe().dropna(how='any')

    coords = X_arr['center'].index.to_frame(name=['y', 'x'], index=False)
    coords['crit'] = X_arr['center'].values
    coords['cluster'] = X_arr['center_cluster'].values  
    coords['rad_eff'] = X_arr['rad_eff'].values 
    coords = coords.dropna(how='any')

    if circ == 'C':
        coords = coords[coords['crit'] > 0]
    else:
        coords = coords[coords['crit'] < 0]

    ## sorted stat
    coords = coords.sort_values(by='crit', ascending=False).reset_index(drop=True)

    return coords

def find_local_minimum_near(bp, crit_field, blue_points):
    """Ищет ближайший локальный минимум по пути от blue к red."""

    dists = np.linalg.norm(blue_points - bp, axis=1)
    dists = np.where(dists == 0, 999, dists)
    min_idx = np.argmin(dists)
    min_d = np.min(dists)
    
    point_near = blue_points[min_idx]

    path = np.linspace(bp, point_near, num=100)  # Генерируем путь из 100 точек

    # Создаем KD-дерево для поиска ближайших точек
    tree = cKDTree(crit_field[:, :2]) 

    # Находим ближайшие точки из crit_field для каждой точки пути
    distances, indices = tree.query(path)  # Получаем индексы ближайших точек

    values = crit_field[indices, 2]  # Берем значения crit для найденных точек

    # Ищем локальный минимум на пути
    min_idx = np.argmin(np.abs(values))
    return path[min_idx]  # Возвращаем координаты локального минимума


def update_rad(local_max, r2d_coords):
    clusters = np.unique(local_max['cluster'])

    local_rad_df = pd.DataFrame(data=None)
    
    for cluster in clusters:
    
        CVS = r2d_coords[r2d_coords['cluster'] == cluster]
        CVS_extr = local_max[local_max['cluster'] == cluster]
        
        blue_points = CVS_extr[['x', 'y']].values 
        crit_field = CVS[['x', 'y', 'crit']].values  
    
        if len(blue_points) > 1:

            green_points = [find_local_minimum_near(bp, crit_field, blue_points) for bp in blue_points]
            radii = np.linalg.norm(blue_points - green_points, axis=1)
        else:
            radii = CVS_extr['rad_eff'].values
        cluster_idx = local_max[local_max['cluster'] == cluster].index
        
        rad_df = pd.DataFrame(radii, columns=['rad_eff'], index=cluster_idx)
        local_rad_df = pd.concat([local_rad_df, rad_df])

    local_max['rad_eff'] = local_rad_df['rad_eff'].sort_index().values
    return local_max

# расчет расстояния между точками по декарту для поиска следующей координаты трека
def dist(p1_x, p1_y, p2_x, p2_y):
    return np.sqrt((p1_x - p2_x)*(p1_x - p2_x) + (p1_y - p2_y)*(p1_y - p2_y))

# перемещение в новую точку исходя из средней скорости потока
def get_new_loc_mean_speed(ds, data_type, dt, x, y, hw, t, speed_level):
    
    dt_time = ds[time_unit][t]

    y_int = int(np.round(y))
    x_int = int(np.round(x))
    
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

#     u, v = compute_mean_uv_for_track(data_type, dt_time, y_in, y_out, x_in, x_out, level=speed_level)

    ### для DBSCAN_with_uv_time ###
    u, v = compute_mean_uv_for_track(ds, t, y_in, y_out, x_in, x_out)
    
    x_new = x+u/dist_m*dt
    y_new = y+v/dist_m*dt
    
    if y_new < 0:
        y_new = 0
    if x_new < 0:
        x_new = 0
            
    if y_new >= len(ds[y_unit]):
        y_new = len(ds[y_unit])-1
    if x_new >= len(ds[x_unit]):
        x_new = len(ds[x_unit])-1
    
    return x_new, y_new

# # расчет средней скорости потока в промежутке hw=rad
# def compute_mean_uv_for_track(data_type, dt, y_in, y_out, x_in, x_out, level=12):
#     path_dir = f'/storage/NAADSERVER/NAAD/{data_type}/PressureLevels/'

#     year = int(dt.dt.year)
    
# #     path_dir = f'/Volumes/KINGSTON/рейс/data/raw_data/R2D_{data_type}'
    
#     params = ['ue', 've']
    
#     hour = int(dt.dt.hour)
#     month = int(dt.dt.month)
#     idx = int(hour/3)
#     date = str(dt.dt.date.values)
    
#     u_path = f"{path_dir}/{params[0]}/{year}/{params[0]}_{date}.nc"
#     v_path = f"{path_dir}/{params[1]}/{year}/{params[1]}_{date}.nc"
    
# #     uv_path = f"{path_dir}/rortex_2d_criteria_{data_type}_level_12_{year}-{month:02d}.nc"
    
#     u = xr.open_dataset(u_path)[params[0]][idx, level, y_in:y_out, x_in:x_out]
#     v = xr.open_dataset(v_path)[params[1]][idx, level, y_in:y_out, x_in:x_out]
#     u_mean = np.nanmean(u)
#     v_mean = np.nanmean(v)
        
#     return u_mean, v_mean

# расчет средней скорости потока в промежутке hw=rad для данных DBSCAN_with_uv_time, где скорость внутри
def compute_mean_uv_for_track(ds, idx, y_in, y_out, x_in, x_out, level=0):
    
    params = ['ue', 've']
    u = ds[params[0]][idx, level, y_in:y_out, x_in:x_out]
    v = ds[params[1]][idx, level, y_in:y_out, x_in:x_out]
    u_mean = np.nanmean(u)
    v_mean = np.nanmean(v)
        
    return u_mean, v_mean



# перемещение в новую точку исходя из скорости КС между предыдущими шагами
def get_new_loc_CS_speed(ds, dt, x_pr, y_pr, x_prpr, y_prpr):
    
    u = (x_pr - x_prpr)/dt
    v = (y_pr - y_prpr)/dt
    
    x_new = x_pr+u*dt
    y_new = y_pr+v*dt
    
    if y_new < 0:
        y_new = 0
    if x_new < 0:
        x_new = 0
            
    if y_new >= len(ds[y_unit]):
        y_new = len(ds[y_unit])-1
    if x_new >= len(ds[x_unit]):
        x_new = len(ds[x_unit])-1

    return x_new, y_new


################################################
################   сам трекинг    ###############
################################################

# инициализация точек-начал треков по ДБСКАНу
def track_init(ds, clstr_len, data_type, stat_Q, CS_tracks_list, our_time, time_name='XTIME'):
    
    for i in range(len(stat_Q)):
        
        ts = []
        times = []
        x_coords = []
        y_coords = []
        
        lon_coords = []
        lat_coords = []
        
        rads = []
        crits = []
        
        track_l = []
        
        ts.append(our_time)
        times.append(ds[time_name][our_time].values)
    
        x = int(stat_Q.x.values[i])
        y = int(stat_Q.y.values[i])
    
        x_coords.append(x)
        y_coords.append(y)
        
        if data_type == 'ERA5':
            lon_coords.append(float(ds.XLONG[y,x].values))
            lat_coords.append(float(ds.XLAT[y,x].values))
        else:
            lon_coords.append(float(ds.XLONG[our_time,y,x].values))
            lat_coords.append(float(ds.XLAT[our_time,y,x].values))
        
        
        rads.append(stat_Q.rad_eff.values[i])
        crits.append(stat_Q.crit.values[i])
        
        track_l.append(0)
        
        CS_coords = {'cluster': clstr_len+i,
                     't': ts,
                     'time': times,
                     'x': x_coords,
                     'y': y_coords,
                     'lon': lon_coords,
                     'lat': lat_coords,
                     'rad': rads,
                     'crit': crits,
                     'track_len': track_l,
                    }
        
        CS_tracks_list.append(CS_coords)
        
    clstr_len = CS_tracks_list[-1]['cluster']
                   
    return CS_tracks_list, clstr_len