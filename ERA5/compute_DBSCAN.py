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

# Инициализация путей и параметров
path_init = f'/storage/thalassa/users/vkoshkina'
folder = 'scripts/CS_processing/CVS_identification/'
sys.path.insert(2, f'{path_init}/{folder}')

from compute_DBSCAN_func import *


years = np.arange(1984,2025)

months = np.arange(1,13)


data_type = 'ERA5'

smooth = True
smooth = False

sigma = 4
# sigma = 2

level = 500
name_crit = 'R2D'

if smooth:
    print(f'{data_type} at {level} hPa with sigma={sigma}')
else:
    print(f'{data_type} at {level} hPa without smoothing')


path_dir_data = f'/storage/thalassa/users/vkoshkina/data'
rortex_path = f'{path_dir_data}/{data_type}/{name_crit}_{data_type}_level_{level}/'
if smooth:
    rortex_path = f'{path_dir_data}/{data_type}/{name_crit}_{data_type}_level_{level}_sigma_{sigma}/'
    
params = ['u', 'v']



    


with open(f'config_{data_type}.json', 'r') as file:
    config = json.load(file)
    
data = config

# Распаковка данных из JSON файла и присвоение переменным
# path_dir = data["path_dir"]
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

folder = f'{path_dir_data}/{data_type}/DBSCAN_{eps:02d}-{min_samples:02d}-{size_filter:02d}/'
if smooth:
    folder = f'{path_dir_data}/{data_type}/DBSCAN_{eps:02d}-{min_samples:02d}-{size_filter:02d}_sigma_{sigma}/'

if not os.path.exists(folder):
    os.makedirs(f'{folder}') 

# levels = ds.bottom_top.values
levels = [level]

def change_time(ds, ds_raw):
    # Проверяем, что в ds_raw есть корректное время
    if 'time' not in ds_raw.coords:
        raise ValueError("В ds_raw отсутствует координата 'time'")
    
    # Создаем новый DataArray с datetime64 и именем измерения 'Time'
    new_time = xr.DataArray(
        data=ds_raw['time'].values,
        dims=['Time'],  # Важно: используем новое имя измерения
        name='Time',
        attrs={'description': 'Time coordinate'}
    )
    
    # 1. Переносим все данные на новую временную координату
    # Для этого нужно переименовать измерение time -> Time во всех переменных
    rename_dict = {'time': 'Time'}
    ds = ds.rename_dims(rename_dict)
    ds = ds.rename_vars(rename_dict)
    
    # 2. Удаляем старую координату time (целочисленную)
    if 'time' in ds.coords:
        ds = ds.drop('time')
    
    # 3. Добавляем новую координату Time
    ds = ds.assign_coords(Time=new_time)
    
    return ds

def convert_time_units(ds, reference_time):
    # time_values = ds['Time'].values

    ds['Time'].encoding['units'] = f"hours since {reference_time}"
    ds['Time'].attrs['description'] = f"hours since {reference_time}"
    
    return ds

def add_latlon(ds, ds_raw):
    lat = xr.DataArray(data=ds_raw['latitude'], dims={'latitude'}, name='latitude')
    lon = xr.DataArray(data=ds_raw['longitude'], dims={'longitude'}, name='longitude')
    
    # Добавляем новое измерение XTIME
    ds = ds.assign_coords({'latitude': lat, 'longitude': lon})
        
    return ds

def add_r2d(ds, ds_raw):
        
    r2d = ds_raw['R2D']
    r2d_max = r2d.max()
    r2d_normalized = r2d / r2d_max
    r2d_scaled = r2d_normalized * 127
    r2d_int8 = r2d_scaled.astype(np.int8)

    ds['R2D'] = (('Time', 'pressure_level', 'latitude', 'longitude'), r2d_int8.values) 
    ds['R2D'].attrs['description'] = 'Rortex criterion 2D discrete (neg - AC, pos - C)'
    ds['R2D'].attrs['long_name'] = 'Rortex 2D discrete'
    
    return ds




def add_param(ds, ds_raw, param):

    ds[param] = (('Time', 'bottom_top', 'south_north', 'west_east'), ds_raw[param].values) 
    ds[param].attrs['description'] = u.attrs['description']
    ds[param].attrs['units'] = u.attrs['units']
    
    return ds

for year in tqdm(years):
    for month in tqdm(months):
        for level in levels:
    
            r2d_name = f'{name_crit}_{data_type}_level_{level}_{year}-{month:02d}'
            if smooth:
                r2d_name = f'sigma_{sigma}_{r2d_name}'
                
            ls = list(sorted(Path(f"{rortex_path}").glob(f'{r2d_name}*')))
    
            for ii,ifile in tqdm(enumerate(ls)):
                ds = xr.open_dataset(ifile)
    
                print(f'\n compute DBSCAN for {year}-{month:02d}')
    
                if len(levels) == 1:
                    our_level = 0
                else: 
                    our_level = level
    
                cluster_ds = get_DBSCAN_ds(ds, config, our_level=our_level)
    
                cluster_ds = add_latlon(cluster_ds, ds)
                cluster_ds = add_r2d(cluster_ds, ds)
                
                cluster_ds = change_time(cluster_ds, ds)
                cluster_ds = convert_time_units(cluster_ds, "1970-01-01 00:00:00")

                dbscan_name = f'DBSCAN_{data_type}_level_{level}_{year}-{month:02d}'
                if smooth:
                    dbscan_name = f'sigma_{sigma}_{dbscan_name}'
    
                cluster_ds.to_netcdf(f'{folder}/{dbscan_name}.nc', mode='w')
