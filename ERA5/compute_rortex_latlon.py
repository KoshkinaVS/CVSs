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

# Инициализация путей и параметров
path_init = f'/storage/thalassa/users/vkoshkina'
folder = 'scripts/CS_processing/CVS_identification/'
sys.path.insert(2, f'{path_init}/{folder}')

from compute_DBSCAN_func import *
from compute_rortex_func import *


years = np.arange(1979,2025)
years = np.arange(2020,2025)

# year = 2010


def process_file(ncfile):
    ds = xr.open_dataset(ncfile)

    # Проверяем, нужна ли конвертация
    if ds.longitude.max() > 180:
        # Создаем новую долготу
        lon_new = xr.where(ds.longitude > 180, ds.longitude - 360, ds.longitude)
        ds = ds.assign_coords(longitude=lon_new)
        
        # Важно: сортируем по новой долготе, чтобы данные шли от -180 до 180
        ds = ds.sortby('longitude')

    ds = ds.sel(latitude=slice(71, 0), longitude=slice(-110, 17))
    print(f'cutting done: {ds['u'].shape}')
    
    
    # Проверяем, что pressure_level существует
    if 'pressure_level' in ds.dims:
        # Если level уже существует, удаляем его (или переименовываем по-другому)
        if 'level' in ds.dims:
            # Удаляем существующий level, чтобы избежать конфликта
            ds = ds.drop_dims('level')
        ds = ds.rename({'pressure_level': 'level'})
    
    # Добавляем уровень, если его нет (для 500 гПа)
    if 'level' not in ds.dims:
        ds = ds.expand_dims({'level': [500]})
    
    # Переименовываем valid_time в time, если нужно
    if 'valid_time' in ds.dims:
        ds = ds.rename({'valid_time': 'time'})
    
    # Проверяем структуру
    required_dims = {'time', 'longitude', 'latitude', 'level'}
    if not set(ds.dims).issuperset(required_dims):
        raise ValueError(f"Dataset missing required dimensions. Found: {set(ds.dims)}")
    
    return ds

def regular_grid_gradient(field, dx, dy):
    """
    Расчет градиента для 4D массива [time, level, lat, lon]
    
    Параметры:
        field: 4D массив компоненты ветра (u или v)
        dx: 2D массив расстояний по долготе [lat, lon] в метрах
        dy: скалярное расстояние по широте в метрах
    
    Возвращает:
        df_dx, df_dy: градиенты по долготе и широте
    """
    # Градиент по долготе (x-направление)
    df_dx = np.gradient(field, 1., axis=-1)/ dx[None,None,:,None]
    
    # Градиент по широте (y-направление)
    df_dy = np.gradient(field, dy, axis=-2) 
    
    return df_dx, df_dy

def compute_omega(du_dx, du_dy, dv_dx, dv_dy):

    omega_z = dv_dx - du_dy

    return omega_z

def swirling_strength(du_dlon, du_dlat, dv_dlon, dv_dlat):
    """Векторизованный расчет λ_ci"""
    a = du_dlon  # ∂u/∂x
    b = du_dlat  # ∂u/∂y
    c = dv_dlon  # ∂v/∂x
    d = dv_dlat  # ∂v/∂y
    
    # Характеристическое уравнение: λ² - (a+d)λ + (ad-bc) = 0
    trace = a + d
    det = a*d - b*c
    
    # Дискриминант (комплексный, если trace² < 4det)
    discriminant = trace**2 - 4*det
    
    # Swirling strength - мнимая часть собственных значений
    lambda_ci = np.sqrt(np.maximum(-discriminant, 0)) / 2
    # Замена невихревых точек на NaN
    lambda_ci[lambda_ci <= 0] = np.nan
    
    return lambda_ci

# считаем 2d Rortex-критерий, Rortex > 0 - циклон, < 0 - АЦ
def compute_rortex_2d(sw_str_2d, omega_2d):
    R_2d = (1-np.sqrt(1-4*sw_str_2d*sw_str_2d/(omega_2d*omega_2d)))*omega_2d
    return R_2d

data_type = 'ERA5'
# level = 500

sigma = 2
smooth = False
smooth = True


if smooth:
    print(f'{data_type} with sigma={sigma}')
else:
    print(f'{data_type} without smoothing')



path_init = f'/storage/thalassa/users/vkoshkina/data'
path_dir_raw = f'{path_init}/ERA5/ERA5_raw/ERA5_NA_for_TC'

path_dir_raw = "/storage/thalassa/DATA/ERA5/PL/NC/uvwth_500hPa"

path_dir_data = f'{path_init}'



params = ['u', 'v']

name_crit = 'R2D'
level_name = 'level'
time_name = 'time'
time_unit = 'time'
x_name = 'longitude'
y_name = 'latitude'


# Константы
DEG_TO_M = 111 * 1000  # 1 градус ~ 111 км
dlon = 0.25  # шаг по долготе
dlat = 0.25  # шаг по широте

# Расстояния между точками сетки (в метрах)
dy = dlat * DEG_TO_M  # постоянное по широте


def get_R2D_nc(u_smooth, v_smooth):
    
    R_2d = get_R2D_ds(u_smooth, v_smooth, dist_m)
    
    #### создание датасета со скоростями ветра на одном уровне высоты ####
    ds = u.to_dataset(name = params[0])
    
    ds[params[0]] = ({time_name: len(ds[time_name]), 
                      level_name: len(ds[level_name]), 
                      y_name: len(ds[y_name]), x_name: len(ds[x_name])}, u_smooth)
    ds[params[1]] = ({time_name: len(ds[time_name]), 
                      level_name: len(ds[level_name]), 
                      y_name: len(ds[y_name]), x_name: len(ds[x_name])}, v_smooth)
    ds[name_crit] = ({time_name: len(ds[time_name]), 
                  level_name: len(ds[level_name]), 
                  y_name: len(ds[y_name]), x_name: len(ds[x_name])}, R_2d.astype(np.float32))
    del ds[params[0]]
 
    return ds

def unification(ds):

    ds[crit_name].attrs['description'] = f'Rortex criterion 2D'
    ds[crit_name].attrs['long_name'] = 'Rortex 2D'

    ds.attrs = {}
        
    return ds


start = time.time() ## точка отсчета времени

years = range(1979, 2025)  # 2000-2023

years = range(2020, 2025)  # 2000-2023


for year in tqdm(years):
    for month in tqdm(np.arange(1,13,1)):
        folder = f'{path_dir_data}/{data_type}/{name_crit}_{data_type}_NA_for_TC_500hPa/'
        r2d_name = f'{name_crit}_{data_type}_{year}-{month:02d}'
        if smooth:
            folder = f'{path_dir_data}/{data_type}/{name_crit}_{data_type}_NA_for_TC_500hPa_sigma_{sigma}/'
            r2d_name = f'sigma_{sigma}_{r2d_name}'
            
        if not os.path.exists(f'{folder}'):
            os.makedirs(f'{folder}')
        
        # Полный путь к файлу
        output_file = f'{folder}/{r2d_name}.nc'
        
        # Проверка на существование файла
        if os.path.exists(output_file):
            print(f'Файл {output_file} уже существует, пропускаем создание')
            continue  # Переходим к следующей итерации
   
        
        ncfile = f'{path_dir_raw}/era5_{year}-{month:02d}_cropped.nc'
        
        ncfile = f'{path_dir_raw}/era5_uvwth_500hPa_{year}-{month:02d}.nc'

        
        ds = process_file(ncfile)
        
        print(f'dims: {ds.dims}')
        
        dx = dlon * DEG_TO_M * np.cos(np.radians(ds.latitude.values))  # зависит от широты
        

        
        ##### определение компонент скорости (на уровне высоты) ######
        #### ВАЖНО: код кушает скорости 4д (т.е. уровень высоты - отдельный dim, не редуцирован) ####
        u = ds[params[0]]
        v = ds[params[1]]
        
        # Применение гауссового сглаживания с сигмой
        sigma_2d = (0, 0, sigma, sigma)

        if smooth:
            u_smooth = gaussian_filter(u.values, sigma=sigma_2d)
            v_smooth = gaussian_filter(v.values, sigma=sigma_2d)
        else:
            u_smooth = u
            v_smooth = v

        print('smoothing done')
        # Вычисляем градиенты ветра
        du_dx, du_dy = regular_grid_gradient(u_smooth, dx, dy)
        dv_dx, dv_dy = regular_grid_gradient(v_smooth, dx, dy)

        omega_2d = compute_omega(du_dx, -du_dy, dv_dx, -dv_dy)
        print('omega done')
        
        
        sw_str_2d = swirling_strength(du_dx, -du_dy, dv_dx, -dv_dy)
        r2d = compute_rortex_2d(sw_str_2d, omega_2d)

        ds_smooth = ds[[params[0], params[1]]].copy()

        ds_smooth[name_crit] = ({time_name: len(ds[time_name]), 
                      level_name: len(ds[level_name]), 
                      y_name: len(ds[y_name]), x_name: len(ds[x_name])}, r2d.astype(np.float32))

        try:
            ds_smooth.to_netcdf(output_file, mode='w')
            print(f'Файл {output_file} успешно сохранен')
        except Exception as e:
            print(f'Ошибка при сохранении файла {output_file}: {str(e)}')
            

        # print(f'\n compute DBSCAN')
        # cluster_ds = get_DBSCAN_ds(ds_smooth, config)


        # folder = f'{path_dir_data}/{data_type}/{name_crit}_{data_type}_level_{level}_smoothing/'
        # if not os.path.exists(f'{folder}'):
        #     os.makedirs(f'{folder}')
      
        # # собираем в файлик
        # ds_smooth.to_netcdf(f'{folder}/sigma_{sigma}_{name_crit}_{data_type}_level_{level}_{year}-{month:02d}.nc', mode='w')
        # del ds_smooth
        
        # folder = f'{path_dir_data}/{data_type}/DBSCAN_{eps:02d}-{min_samples:02d}-{size_filter:02d}_level_{level}_smoothing'
        # if not os.path.exists(folder):
        #     os.makedirs(f'{folder}')  
            
        # cluster_ds.to_netcdf(f'{folder}/sigma_{sigma}_DBSCAN_{data_type}_level_{level}_{year}-{month:02d}.nc', mode='w')
        # del cluster_ds 
        

        end = time.time() - start ## собственно время работы программы
        print(f'{end/60} min for year {year}') ## вывод времени
