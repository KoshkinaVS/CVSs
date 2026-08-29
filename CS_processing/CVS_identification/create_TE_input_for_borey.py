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

from compute_rortex_func import *

months = np.arange(2, 3)

data_type = 'GPN'
path_dir = f'/storage/thalassa/users/gavr/{data_type}/Coherents/rortex/'
path_dir_raw = f'/storage/buffer/{data_type}/OUTPUTS/WRF6km/'

year = 2022
    
params = ['ua', 'va']

with open(f'config_{data_type}.json', 'r') as file:
    config = json.load(file)
    
data = config

# Распаковка данных из JSON файла
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



def add_init_params(r2d, u, v):
    u = u.rename({
                    'XLAT': 'lat',
                    'XLONG': 'lon',
                    'XTIME': 'time',
                })
    v = v.rename({
                    'XLAT': 'lat',
                    'XLONG': 'lon',
                    'XTIME': 'time',
                })
    
    del v['Time']
    del u['Time']

    # Переименовываем измерение Time в time для переменных
    u = u.rename({'Time': 'time'})
    v = v.rename({'Time': 'time'})
    
    ds = xr.merge([r2d, u, v])

    ds.coords['lat'] = u['lat']
    ds.coords['lon'] = u['lon']

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
    Извлекает дату из имени файла WRF.
    Ожидаемый формат: wrfout_d01_YYYY-MM-DD_HH:MM:SS или wrfout_d01_YYYY-MM-DD
    """
    # Паттерны для разных форматов имен файлов
    patterns = [
        r'rortex_d01_(\d{4})-(\d{2})-(\d{2})_(\d{2})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, str(filename))
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))
            return year, month, day
    
    # Если не нашли совпадение, возвращаем None
    return None, None, None

level = 22
sigmas = [
    # 4, 2, 
    0]

start = time.time()

# Создаем общий прогресс-бар для сигм
sigma_progress = tqdm(sigmas, desc="Processing sigmas", position=0)

for sigma in sigma_progress:
    sigma_progress.set_description(f"Processing sigma {sigma}")
    
    # Прогресс-бар для месяцев
    monthly_progress = tqdm(months, desc="Processing months", leave=False, position=1)
    
    for month in monthly_progress:
        monthly_progress.set_description(f"Processing {year}-{month:02d}")
        start_month = time.time()
        
        ls = list(sorted(Path(f"{path_dir}/{year}").glob(f'rortex_d01_{year}-{month:02d}*')))
        
        # Прогресс-бар для файлов
        file_progress = tqdm(enumerate(ls), total=len(ls), desc="Processing files", leave=False, position=2)
        
        for ii, ifile in file_progress:
            try:
                file_progress.set_description(f"Processing {ifile.name}")

                # Извлекаем дату из имени файла
                file_year, file_month, file_day = extract_date_from_filename(ifile.name)
                r2d = xr.open_dataset(ifile)['R2D'][:,level]
        
                if file_day is None:
                    file_day = 1  # fallback

                raw_file = f'{path_dir_raw}/{year}/wrfout_d01_{year}-{month:02d}-{file_day:02d}_00:00:00'
                
                wrfnc = Dataset(raw_file)
                u = wrf.getvar(wrfnc, params[0], timeidx=wrf.ALL_TIMES)[:,level]
                v = wrf.getvar(wrfnc, params[1], timeidx=wrf.ALL_TIMES)[:,level]

                ds = add_init_params(r2d, u, v)

                # Новая директория для обработанных данных
                path_dir_new = f"{path_init}/data/TempestExtremes/{data_type}/R2D_{data_type}_level_{level}_sigma_{sigma}/Input"
                
                # Создаём директорию, если её нет
                os.makedirs(path_dir_new, exist_ok=True)
        
                # Сохраняем в новый файл с тем же именем в новой директории
                new_ncfile = f'{path_dir_new}/sigma_{sigma}_R2D_{data_type}_level_{level}_{year}-{month:02d}-{file_day:02d}.nc'
                ds.to_netcdf(new_ncfile, format='NETCDF4')

                # Обновляем информацию о прогрессе
                file_progress.set_postfix({
                    'Last file': f'{ifile.name}',
                    'Month progress': f'{ii+1}/{len(ls)}'
                })
                
            except Exception as e:
                tqdm.write(f"Error processing {ifile.name}: {str(e)}")
                continue
        
        # Вычисляем время выполнения месяца
        month_time = (time.time() - start_month) / 60
        monthly_progress.set_postfix({
            'Month time (min)': f"{month_time:.2f}",
            'Last month': f"{year}-{month:02d}"
        })
        
        file_progress.close()
    
    monthly_progress.close()

# Общее время выполнения
total_time = (time.time() - start) / 60
sigma_progress.set_postfix({'Total time (min)': f"{total_time:.2f}"})
sigma_progress.close()