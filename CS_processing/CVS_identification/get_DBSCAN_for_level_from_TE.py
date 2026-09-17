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

import re

path_init = f'/storage/thalassa/users/vkoshkina'
sys.path.insert(3, f'{path_init}/scripts/CS_processing/CVS_identification')

from compute_DBSCAN_func_new import *
from compute_rortex_func import *

months = np.arange(1, 13)

years = np.arange(1979, 2019)
years = np.arange(2010, 2011)



data_type = 'HiRes'
# data_type = 'LoRes'

level = 12




params = ['ue', 've', 'HGT']

with open(f'config_NAAD_01-04-10.json', 'r') as file:
    config = json.load(file)
    
data = config

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

time_name = 'Time'
time_unit = 'Time'


path_dir_data = f'/storage/thalassa/users/vkoshkina/data'

if data_type == 'LoRes':
    path_dir_data = f'{path_dir_data}/LoRes'


def change_time(ds, ds_raw):
    new_xtime = xr.DataArray(data=ds_raw['Time'], dims='Time', name='Time')

    # Добавляем новое измерение
    ds = ds.assign_coords(Time=new_xtime)

    if 'XTIME' in ds.coords:
        del ds.coords['XTIME']
        
    return ds

def convert_time_units(ds, reference_time):
    time_values = ds['Time'].values

    ds['Time'].encoding['units'] = f"hours since {reference_time}"
    ds['Time'].attrs['description'] = f"hours since {reference_time}"
    
    return ds

def add_latlon(ds, ds_raw):
    lat = xr.DataArray(data=ds_raw['XLAT'], dims={'south_north', 'west_east'}, name='XLAT')
    lon = xr.DataArray(data=ds_raw['XLONG'], dims={'south_north', 'west_east'}, name='XLONG')
    
    ds = ds.assign_coords({'XLAT': lat, 'XLONG': lon})
        
    return ds

def add_r2d(ds, ds_raw):
        
    r2d = ds_raw['R2D']
    r2d_max = r2d.max()
    r2d_normalized = r2d / r2d_max
    r2d_scaled = r2d_normalized * 127
    r2d_int8 = r2d_scaled.astype(np.int8)

    ds['R2D'] = (('Time', 'bottom_top', 'south_north', 'west_east'), r2d_int8.values) 
    ds['R2D'].attrs['description'] = 'Rortex criterion 2D discrete (neg - AC, pos - C)'
    ds['R2D'].attrs['long_name'] = 'Rortex 2D discrete'
    
    return ds

def add_param(ds, ds_raw, param):
    ds[param] = (('Time', 'bottom_top', 'south_north', 'west_east'), ds_raw[param].values) 
    ds[param].attrs['description'] = ds_raw[param].attrs.get('description', '')
    ds[param].attrs['units'] = ds_raw[param].attrs.get('units', '')
    
    return ds

def add_init_params(r2d, u, v):
    # del v['Time']
    # del u['Time']

    r2d = r2d.rename({'time': 'Time'})
    
    ds = xr.merge([r2d, u, v])

    # Удаляем проблемные атрибуты из всех переменных
    for var in ds.data_vars:
        if 'projection' in ds[var].attrs:
            del ds[var].attrs['projection']
    
    # Удаляем проблемные атрибуты из глобальных атрибутов
    if 'projection' in ds.attrs:
        del ds.attrs['projection']
        
    return ds

def extract_date_from_filename(filename):
    """
    Извлекает дату из имени файла рортекса.
    Ожидаемый формат: sigma_0_R2D_GPN_level_12_YYYY-MM.nc
    """
    patterns = [
        r'sigma_\d+_R2D_'+data_type+r'_level_\d+_(\d{4})-(\d{2})\.nc',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, str(filename))
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            return year, month
    
    # Если не нашли совпадение, возвращаем None
    return None, None

level = 12
sigmas = [
    # 4,
    2,
#           0
         ]

start = time.time()

# Создаем общий прогресс-бар для сигм
sigma_progress = tqdm(sigmas, desc="Processing sigmas", position=0)

for sigma in sigma_progress:
    sigma_progress.set_description(f"Processing sigma {sigma}")

    rortex_path = f'{path_init}/data/TempestExtremes/{data_type}/R2D_{data_type}_level_{level}_sigma_{sigma}/Input'  # Измененный путь к рортексу

    
    folder = f'{path_dir_data}/{data_type}/DBSCAN_{eps:02d}-{min_samples:02d}-{size_filter:02d}_sigma_{sigma}'
    if not os.path.exists(folder):
        os.makedirs(f'{folder}') 
            
    # Прогресс-бар для месяцев
    yearly_progress = tqdm(years, desc="Processing years", leave=False, position=1)
    
    for year in yearly_progress:
        yearly_progress.set_description(f"Processing {year}")
        start_month = time.time()

        
        # Ищем файлы рортекса в новой папке
        ls = list(sorted(Path(rortex_path).glob(f'sigma_{sigma}_R2D_{data_type}_level_{level}_{year}-*.nc')))
        
        # Прогресс-бар для файлов
        file_progress = tqdm(enumerate(ls), total=len(ls), desc="Processing files", leave=False, position=2)
        
        for ii, ifile in file_progress:
            try:
                file_progress.set_description(f"Processing {ifile.name}")

                # Извлекаем дату из имени файла рортекса
                file_year, file_month = extract_date_from_filename(ifile.name)
                
                # Загружаем данные рортекса
                ds = xr.open_dataset(ifile)
                ds = ds.expand_dims(level_name, axis=1)  # Добавляем измерение уровня
                ds = ds.rename({'time': 'Time', 'lat': 'XLAT', 'lon': 'XLONG'})
                
                if file_month is None:
                    file_month = 1  # fallback


                our_level = 0

                cluster_ds = get_DBSCAN_ds(ds, config, our_level=our_level)

                # cluster_ds = add_latlon(cluster_ds, ds)
                cluster_ds = change_time(cluster_ds, ds)
                cluster_ds = convert_time_units(cluster_ds, "1970-01-01 00:00:00")
                # cluster_ds = add_r2d(cluster_ds, ds)
                # cluster_ds = add_param(cluster_ds, ds, params[0])
                # cluster_ds = add_param(cluster_ds, ds, params[1])

                ### new
                cluster_ds = xr.merge([ds, cluster_ds])

                del cluster_ds['R2D']
                cluster_ds = add_r2d(cluster_ds, ds)
                
                
                cluster_ds.to_netcdf(f'{folder}/sigma_{sigma}_DBSCAN_{data_type}_level_{level}_{year}-{file_month:02d}.nc', mode='w')
                
                # Обновляем информацию о прогрессе
                file_progress.set_postfix({
                    'Last file': f'{ifile.name}',
                    'Month progress': f'{ii+1}/{len(ls)}'
                })
                
                # Закрываем файлы для освобождения памяти
                ds.close()
                
            except Exception as e:
                tqdm.write(f"Error processing {ifile.name}: {str(e)}")
                continue
        
        # Вычисляем время выполнения месяца
        month_time = (time.time() - start_month) / 60
        yearly_progress.set_postfix({
            'Year time (min)': f"{month_time:.2f}",
            'Last year': f"{year}"
        })
        
        file_progress.close()
    
    yearly_progress.close()

# Общее время выполнения
total_time = (time.time() - start) / 60
sigma_progress.set_postfix({'Total time (min)': f"{total_time:.2f}"})
sigma_progress.close()