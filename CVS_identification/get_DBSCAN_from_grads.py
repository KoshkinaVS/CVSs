import pandas as pd 
import numpy as np
import math
import xarray as xr
from numpy import linalg as LA

from tqdm import tqdm
import time
import datetime
from datetime import timedelta

from scipy.ndimage import gaussian_filter

from netCDF4 import Dataset
import wrf as wrf

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path
import sys
import os

import calendar

# path_init = f'/storage/thalassa/users/vkoshkina'
# sys.path.insert(3, f'{path_init}/scripts')

from compute_DBSCAN_func_new import *
from compute_rortex_func import *



years = np.arange(2010,2011)
months = np.arange(1,13)


print('data type: ')
data_type = input() 



# path = f'/storage/OPENDATA/NAAD/{data_type}/'
# path_dir = f'{path}/ModelLevels/tensor/'



dim_type = '2D'
name_crit = "R2D"
# dim_type = '3D'


our_level = 22 # 5 km

# path_init = f'/storage/thalassa/users/vkoshkina'



with open(f'config_NAAD.json', 'r') as file:
    config = json.load(file)
    
data = config


params = ['ue', 've', 'HGT']


    

# Распаковка данных из JSON файла и присвоение переменным
path_dir_data = data["path_dir"]
# data_type = data["data_type"]
dist_m = data["dist_m"]
eps = data["eps"]
min_samples = data["min_samples"]
size_filter = data["size_filter"]
min_dist = data["min_dist"]

name_crit = data["name_crit"]
level_name = data["level_name"]
x_name = data["x_name"]
y_name = data["y_name"]
time_name = data["time_name"]
time_unit = data["time_unit"]



# levels = ds.bottom_top.values
levels = [22]

# def change_time(ds, ds_raw):
#     new_xtime = xr.DataArray(data=ds_raw['Time'], dims='Time', name='Time')

#     # Добавляем новое измерение XTIME
#     ds = ds.assign_coords(Time=new_xtime)

#     if 'XTIME' in ds.coords:
#         del ds.coords['XTIME']
        
#     return ds

def convert_time_units(ds, reference_time):
    time_values = ds['Time'].values

    ds['Time'].encoding['units'] = f"hours since {reference_time}"
    ds['Time'].attrs['description'] = f"hours since {reference_time}"
    
    return ds

def add_latlon(ds, ds_raw):
#     lat = xr.DataArray(data=ds_raw['XLAT'], dims={'west_east', 'south_north'}, name='XLAT')
#     lon = xr.DataArray(data=ds_raw['XLONG'], dims={'west_east', 'south_north'}, name='XLONG')
    lat = xr.DataArray(data=ds_raw['XLAT'], dims={'south_north', 'west_east'}, name='XLAT')
    lon = xr.DataArray(data=ds_raw['XLONG'], dims={'south_north', 'west_east'}, name='XLONG')
    
    # Добавляем новое измерение XTIME
    ds = ds.assign_coords({'XLAT': lat, 'XLONG': lon})
        
    return ds

# def add_r2d(ds, ds_raw):
        
#     r2d = ds_raw['R2D']
#     r2d_max = r2d.max()
#     r2d_normalized = r2d / r2d_max
#     r2d_scaled = r2d_normalized * 127
#     r2d_int8 = r2d_scaled.astype(np.int8)

#     ds['R2D'] = (('Time', 'bottom_top', 'south_north', 'west_east'), r2d_int8.values) 
#     ds['R2D'].attrs['description'] = 'Rortex criterion 2D discrete (neg - AC, pos - C)'
#     ds['R2D'].attrs['long_name'] = 'Rortex 2D discrete'
    
#     return ds

def add_r2d(ds, ds_raw, our_level=None):
    """
    Добавляет R2D данные с правильным порядком размерностей
    """
    if our_level is not None:
        # Берем только нужный уровень
        r2d = ds_raw['R2D'].isel({level_name: our_level})
    else:
        # Берем все уровни
        r2d = ds_raw['R2D']
    
    r2d_max = r2d.max()
    r2d_normalized = r2d / r2d_max
    r2d_scaled = r2d_normalized * 127
    r2d_int8 = r2d_scaled.astype(np.int8)

    # Убеждаемся в правильном порядке размерностей
    if our_level is not None:
        # Для отдельных уровней: добавляем измерение уровня
        r2d_int8 = r2d_int8.expand_dims({level_name: [our_level]})
        # Правильный порядок: Time, bottom_top, south_north, west_east
        r2d_int8 = r2d_int8.transpose('Time', level_name, 'south_north', 'west_east')
    else:
        # Для всех уровней: уже правильный порядок
        r2d_int8 = r2d_int8.transpose('Time', level_name, 'south_north', 'west_east')
    
    ds['R2D'] = r2d_int8
    ds['R2D'].attrs['description'] = 'Rortex criterion 2D discrete (neg - AC, pos - C)'
    ds['R2D'].attrs['long_name'] = 'Rortex 2D discrete'
    
    return ds


def add_param(ds, ds_raw, param):

    ds[param] = (('Time', 'bottom_top', 'south_north', 'west_east'), ds_raw[param].values) 
    ds[param].attrs['description'] = u.attrs['description']
    ds[param].attrs['units'] = u.attrs['units']
    
    return ds


path = f'/storage/OPENDATA/NAAD/{data_type}/'
if data_type == 'LoRes':
    hgt_file_path = f'{path}/Invariants/NAAD77km_hgt.nc'
else:
    hgt_file_path = f'{path}/Invariants/NAAD14km_hgt.nc'
    

if not os.path.exists(hgt_file_path):
    raise FileNotFoundError(f"HGT file not found: {hgt_file_path}")

hgt = xr.open_dataset(hgt_file_path)


for year in tqdm(years):
    rortex_path = f'{path_dir_data}/{data_type}/rortex_from_grads/R2D_R3D_dummy/{year}'
    
    dbscan_path = f'{path_dir_data}/{data_type}/rortex_from_grads/DBSCAN_{eps:02d}-{min_samples:02d}-{size_filter:02d}'

    # Создаем общий прогресс-бар для месяцев
    monthly_progress = tqdm(range(1, 13), desc="Processing months", position=0)
    
    for month in monthly_progress:
        monthly_progress.set_description(f"Processing {dim_type} {year}-{month:02d}")
        start_month = time.time()
        
        num_days = calendar.monthrange(year, month)[1]
        
        # Прогресс-бар для дней с обновлением информации
        daily_progress = tqdm(range(1, num_days + 1), 
                             desc=f"Days in {year}-{month:02d}", 
                             leave=False, 
                             position=1)
        
        for day in daily_progress:
            try:
                ds = xr.open_dataset(f"{rortex_path}/{name_crit}_{data_type}_{year}-{month:02d}-{day:02d}.nc")

                levels = ds.bottom_top.values
                
                
                # Список для хранения результатов по всем уровням
                all_levels_cluster_ds = []
                
                for our_level in levels:
        
                    print(f'\n compute DBSCAN for level {our_level}')
        
                    # if len(levels) == 1:
                    #     our_level = 0
                    # else: 
                    #     our_level = level
        
                    cluster_ds = get_DBSCAN_ds(ds, config, our_level=our_level)
        
                    cluster_ds = add_latlon(cluster_ds, hgt)
                    # cluster_ds = change_time(cluster_ds, ds)
                    cluster_ds = convert_time_units(cluster_ds, "1970-01-01 00:00:00")
                    cluster_ds = add_r2d(cluster_ds, ds, our_level=our_level)
        
                    dbscan_path_level = f'{dbscan_path}/{year}/level_{our_level}'
                    if not os.path.exists(dbscan_path_level):
                        os.makedirs(f'{dbscan_path_level}')
                    cluster_ds.to_netcdf(f'{dbscan_path_level}/DBSCAN_{data_type}_level_{our_level}_{year}-{month:02d}-{day:02d}.nc', mode='w')

                    all_levels_cluster_ds.append(cluster_ds)
                    
                # Объединяем все уровни в один dataset
                if all_levels_cluster_ds:
                    combined_ds = xr.concat(all_levels_cluster_ds, dim=level_name)

                    # Убеждаемся в правильном порядке размерностей для объединенного dataset
                    for var_name in combined_ds.data_vars:
                        combined_ds[var_name] = combined_ds[var_name].transpose('Time', level_name, 'south_north', 'west_east')
                    
                    # Сохраняем объединенный файл со всеми уровнями
                    combined_ds.to_netcdf(
                        f'{dbscan_path}/{year}/DBSCAN_{data_type}_{year}-{month:02d}-{day:02d}.nc', 
                        mode='w'
                    )
                    
                    print(f"Saved combined file for {year}-{month:02d}-{day:02d} with {len(levels)} levels")


            except Exception as e:
                tqdm.write(f"Error processing {year}-{month:02d}-{day:02d}: {str(e)}")
                continue
        
        # Вычисляем время выполнения месяца
        month_time = (time.time() - start_month)/60
        monthly_progress.set_postfix({
            'Month time (min)': f"{month_time:.2f}",
            'Last month': f"{year}-{month:02d}"
        })
        
        # Закрываем daily_progress чтобы избежать наложения
        daily_progress.close()