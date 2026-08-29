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


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path
import sys
import os

import calendar


# path_init = f'/storage/thalassa/users/vkoshkina/scripts/CS_processing/CVS_identification'

# sys.path.insert(3, f'{path_init}')

# from compute_DBSCAN_func import *
from compute_rortex_func import *


years = np.arange(2010,2011)
month = 8


print('data type: ')
data_type = input() 



path = f'/storage/OPENDATA/NAAD/{data_type}/'
path_dir = f'{path}/ModelLevels/tensor/'



dim_type = '2D'
# dim_type = '3D'




our_level = 22 # 5 km

path_init = f'/storage/thalassa/users/vkoshkina'
path_dir_data = f'{path_init}/data/'



with open(f'config_NAAD.json', 'r') as file:
    config = json.load(file)
    
data = config

# Распаковка данных из JSON файла и присвоение переменным
# path_dir = data["path_dir"]
# data_type = data["data_type"]
# dist_m = data["dist_m"]
eps = data["eps"]
min_samples = data["min_samples"]
size_filter = data["size_filter"]
min_dist = data["min_dist"]

name_crit = data["name_crit"]
level_name = data["level_name"]

level_name = 'bottom_top'


x_name = data["x_name"]
y_name = data["y_name"]
time_name = data["time_name"]
time_unit = data["time_unit"]


dudx_name, dudy_name, dudz_name = 'dudx', 'dudy', 'dudz'
dvdx_name, dvdy_name, dvdz_name = 'dvdx', 'dvdy', 'dvdz'
dwdx_name, dwdy_name, dwdz_name = 'dwdx', 'dwdy', 'dwdz'


def get_rortex_nc(R_2d, u, name_crit, dim_type):
    """
    Создает xarray Dataset с критерием Rortex.
    
    Параметры:
        R_2d: np.ndarray или xarray.DataArray
            Массив с критерием Rortex
        u: xarray.DataArray
            Массив с одной из компонент скорости (для получения координат)
        name_crit: str
            Имя переменной для сохранения критерия
    """
    # Создаем Dataset из компоненты скорости
    ds = u.to_dataset(name='u')
    
    # Добавляем критерий Rortex
    ds[name_crit] = (
        (time_name, level_name, y_name, x_name),
        R_2d.astype(np.float32)
    )


    
    # # Удаляем временные переменные
    del ds['u']
    
    # Добавляем атрибуты
    ds[name_crit].attrs = {
        'description': f'Rortex criterion {dim_type}',
        'long_name': f'Rortex {dim_type}'
    }
    
    return ds


if data_type == 'LoRes':
    hgt_file_path = f'{path}/Invariants/NAAD77km_hgt.nc'  # предполагаем, что файл лежит здесь
else:
    hgt_file_path = f'{path}/Invariants/NAAD14km_hgt.nc'  # предполагаем, что файл лежит здесь
    

# Шаг 2: Загружаем HGT из внешнего файла
if not os.path.exists(hgt_file_path):
    raise FileNotFoundError(f"HGT file not found: {hgt_file_path}")

hgt = xr.open_dataset(hgt_file_path)


year = 2010

start = time.time() ## точка отсчета времени

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
            name = f'tensor_d01_{year}-{month:02d}-{day:02d}_00.nc'
            ds = xr.open_dataset(f"{path_dir}/{year}/{name}")

            u = xr.open_dataset(f"{path}/ModelLevels/ue/{year}/ue_{year}-{month:02d}-{day:02d}.nc")
            v = xr.open_dataset(f"{path}/ModelLevels/ve/{year}/ve_{year}-{month:02d}-{day:02d}.nc")
            
            
            
            # Ваши вычисления
            du_dx, du_dy, du_dz = ds[dudx_name], ds[dudy_name], ds[dudz_name]
            dv_dx, dv_dy, dv_dz = ds[dvdx_name], ds[dvdy_name], ds[dvdz_name]
            dw_dx, dw_dy, dw_dz = ds[dwdx_name], ds[dwdy_name], ds[dwdz_name]

            if dim_type == '2D':
                name_crit = "R2D"
                r2d = get_R2D_ds_from_grads(du_dx, du_dy, dv_dx, dv_dy)
                
            elif dim_type == '3D':
                name_crit = "R3D"
                r2d = get_R3D_ds_from_grads(du_dx, du_dy, du_dz,
                                          dv_dx, dv_dy, dv_dz,
                                          dw_dx, dw_dy, dw_dz)
            
            rortex_ds = get_rortex_nc(r2d, ds[dudx_name], name_crit, dim_type)
            rortex_ds.attrs = ds.attrs.copy()
            
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            rortex_ds.attrs["CREATED"] = current_time
        
            # for param in params:
            #     ds = add_param(ds, wrfnc, param)

            # Добавляем критерий Rortex 3D (заглушка!!!!!!!!!) 
            rortex_ds['R3D'] = (
                (time_name, level_name, y_name, x_name),
                r2d.astype(np.float32)
            )
            
            # Добавляем атрибуты
            rortex_ds['R3D'].attrs = {
                'description': f'Rortex criterion 3D (dummy)',
                'long_name': f'Rortex 3D (dummy)'
            }


            rortex_ds['ue'] = ((time_name, level_name, y_name, x_name), u['ue'].values)
            rortex_ds['ue'].attrs['description'] = u['ue'].attrs['description']
            rortex_ds['ue'].attrs['units'] = u['ue'].attrs['units']

            rortex_ds['ve'] = ((time_name, level_name, y_name, x_name), v['ve'].values)
            rortex_ds['ve'].attrs['description'] = v['ve'].attrs['description']
            rortex_ds['ve'].attrs['units'] = v['ve'].attrs['units']
        
            rortex_ds['HGT'] = ((y_name, x_name), hgt['hgt'].values)
            rortex_ds['HGT'].attrs['description'] = hgt['hgt'].attrs['description']
            rortex_ds['HGT'].attrs['units'] = hgt['hgt'].attrs['units']
        

            folder = f'{path_dir_data}/{data_type}/rortex_from_grads/R2D_R3D_dummy/{year}'
            os.makedirs(folder, exist_ok=True)
            
            rortex_ds.to_netcdf(f'{folder}/{name_crit}_{data_type}_{year}-{month:02d}-{day:02d}.nc', mode='w')
            
            # Обновляем описание прогресс-бара
            daily_progress.set_postfix({
                'Last processed': f'{year}-{month:02d}-{day:02d}',
                'File': name
            })
            
            del rortex_ds, ds
            
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