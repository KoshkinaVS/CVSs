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

import gc

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




def process_file(ncfile):
    ds = xr.open_dataset(ncfile)
    
    ds = ds[['u', 'v']]

    # Проверяем, нужна ли конвертация
    if ds.longitude.max() > 180:
        # Создаем новую долготу
        lon_new = xr.where(ds.longitude > 180, ds.longitude - 360, ds.longitude)
        ds = ds.assign_coords(longitude=lon_new)
        
        # Важно: сортируем по новой долготе, чтобы данные шли от -180 до 180
        ds = ds.sortby('longitude')


    if region == 'Arctic':
        ds = ds.sel(latitude=slice(85, 65), longitude=slice(-30, 80))
    else:
        ds = ds.sel(latitude=slice(71, 0), longitude=slice(-110, 15))
    
    
    print(f"cutting done: {ds['u'].shape}")

    # Проверяем, что pressure_level существует
    if 'pressure_level' in ds.dims:
        # Если level уже существует, удаляем его (или переименовываем по-другому)
        if 'level' in ds.dims:
            # Удаляем существующий level, чтобы избежать конфликта
            ds = ds.drop_dims('level')
        ds = ds.rename({'pressure_level': 'level'})
    
    # Добавляем уровень, если его нет (для 500 гПа)
    if 'level' not in ds.dims:
        ds = ds.expand_dims({'level': [level_hPa]})
    
    # Переименовываем valid_time в time, если нужно
    if 'valid_time' in ds.dims:
        ds = ds.rename({'valid_time': 'time'})
        
    # ✅ УДАЛЯЕМ ЛИШНИЕ КООРДИНАТЫ
    # Удаляем координату 'number', если она существует
    if 'number' in ds.coords:
        ds = ds.drop_vars('number')
    
    # Удаляем координату 'expver', если она существует
    if 'expver' in ds.coords:
        ds = ds.drop_vars('expver')
    
    # Удаляем другие возможные лишние координаты
    useless_coords = ['number', 'expver', 'step', 'surface', 'isobaricInhPa']
    for coord in useless_coords:
        if coord in ds.coords:
            ds = ds.drop_vars(coord)
            
    expected_dims = ['time', 'level', 'latitude', 'longitude']
    ds = ds.transpose(*expected_dims)
    
    return ds

    
def process_file(grib_file, level_hPa=850, region='Arctic'):
    """
    Обрабатывает GRIB файл и возвращает Dataset с указанным уровнем давления
    
    Parameters:
    -----------
    grib_file : str
        Путь к GRIB файлу
    level_hPa : int
        Уровень давления в гПа (по умолчанию 850)
    region : str
        Регион для обрезки ('Arctic' или другой)
    
    Returns:
    --------
    xarray.Dataset
        Dataset с размерами (time, level, latitude, longitude)
    """
    # Открываем GRIB файл через cfgrib
    ds = xr.open_dataset(grib_file, engine='cfgrib')
    
    # Выбираем только переменные u и v
    ds = ds[['u', 'v']]
    
    # Проверяем и конвертируем долготу из 0-360 в -180-180
    if 'longitude' in ds.coords and ds.longitude.max() > 180:
        lon_new = xr.where(ds.longitude > 180, ds.longitude - 360, ds.longitude)
        ds = ds.assign_coords(longitude=lon_new)
        ds = ds.sortby('longitude')
    
    # Обрезаем по региону
    if region == 'Arctic':
        ds = ds.sel(latitude=slice(85, 65), longitude=slice(-30, 80))
    else:
        ds = ds.sel(latitude=slice(71, 0), longitude=slice(-110, 15))
    
    
    # Выбираем нужный уровень давления
    # В исходных данных уровень называется 'isobaricInhPa'
    if 'isobaricInhPa' in ds.dims:
        # Выбираем ближайший уровень к level_hPa
        ds = ds.sel(isobaricInhPa=level_hPa, method='nearest')
        # Переименовываем измерение в 'level'
        ds = ds.expand_dims('level', axis=1)
        ds = ds.assign_coords(level=[level_hPa])
    else:
        # Если уровень не найден, создаем его
        ds = ds.expand_dims({'level': [level_hPa]})
    
    # Переименовываем valid_time в time, если нужно
    if 'valid_time' in ds.dims:
        ds = ds.rename({'valid_time': 'time'})
    elif 'time' not in ds.dims and 'valid_time' in ds.coords:
        # Иногда valid_time может быть координатой, а не измерением
        ds = ds.rename({'valid_time': 'time'})
    
    # Удаляем лишние координаты
    useless_coords = ['number', 'expver', 'step', 'surface', 'isobaricInhPa']
    for coord in useless_coords:
        if coord in ds.coords:
            ds = ds.drop_vars(coord, errors='ignore')
    
    # Убеждаемся, что координаты идут в правильном порядке
    expected_dims = ['time', 'level', 'latitude', 'longitude']
    # Проверяем, какие измерения присутствуют
    existing_dims = [dim for dim in expected_dims if dim in ds.dims]
    if len(existing_dims) == len(expected_dims):
        ds = ds.transpose(*expected_dims)
    else:
        print(f"Внимание: не все ожидаемые измерения присутствуют. Доступны: {list(ds.dims)}")
        # Транспонируем с доступными измерениями
        ds = ds.transpose(*[dim for dim in ds.dims if dim in expected_dims])
    
    print(f"level: {ds.level.values if 'level' in ds.coords else 'not found'}, ds.dims: {ds.dims}")
    
    return ds

def vectorized_2d_smooth(u_data, v_data, sigma):

    # Применяем фильтр ко всем срезам сразу
    u_smooth = np.zeros_like(u_data)
    v_smooth = np.zeros_like(v_data)
    
    
    for i in range(u_data.shape[0]):
        u_smooth[i,0] = gaussian_filter(u_data[i,0], sigma=sigma, mode='reflect')
        v_smooth[i,0] = gaussian_filter(v_data[i,0], sigma=sigma, mode='reflect')

    
    return u_smooth, v_smooth


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

region = 'Arctic'
region = 'NA'




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
    ds = u.to_dataset(name = 'u')
    
    ds[params[0]] = ({time_name: len(ds[time_name]), 
                      level_name: len(ds[level_name]), 
                      y_name: len(ds[y_name]), x_name: len(ds[x_name])}, u_smooth)
    ds[params[1]] = ({time_name: len(ds[time_name]), 
                      level_name: len(ds[level_name]), 
                      y_name: len(ds[y_name]), x_name: len(ds[x_name])}, v_smooth)
    ds[name_crit] = ({time_name: len(ds[time_name]), 
                  level_name: len(ds[level_name]), 
                  y_name: len(ds[y_name]), x_name: len(ds[x_name])}, R_2d.astype(np.float32))
    del ds['u']
 
    return ds

def unification(ds):

    ds[crit_name].attrs['description'] = f'Rortex criterion 2D'
    ds[crit_name].attrs['long_name'] = 'Rortex 2D'

    ds.attrs = {}
        
    return ds


start = time.time() ## точка отсчета времени

years = range(2010, 2027)  
# years = range(1990, 2010)  
# years = range(2024, 2027) 
# years = range(1979, 1990)  




# Основной цикл
for year in tqdm(years, desc="Years"):
    for month in tqdm(np.arange(1, 13, 1), desc="Months", leave=False):
        start_time = time.time()
       
        if region == 'Arctic':
            # Арктика
            level_hPa = 850
            path_dir_raw = f"/storage/thalassa/DATA/ERA5/PL/NC/{level_hPa}hPa"
            ncfile = f'{path_dir_raw}/era5_uvwth_{level_hPa}hPa_{year}-{month:02d}.nc'
            folder_name = f'{name_crit}_{data_type}_Arctic_{level_hPa}hPa'
        else:
            # Атлантика
            level_hPa = 500
            path_dir_raw = f"/storage/thalassa/DATA/ERA5/PL/NC/uvwth_{level_hPa}hPa"
            
            level_hPa = 850
            path_dir_raw = f"/storage/thalassa/DATA/ERA5/PL/NC/{level_hPa}hPa"

            level_hPa = 700
            path_dir_raw = f"/storage/thalassa/DATA/ERA5/PL/grib/{year}"
            
            
            ncfile = f'{path_dir_raw}/era5_uvwth_{level_hPa}hPa_{year}-{month:02d}.nc'
            folder_name = f'{name_crit}_{data_type}_NA_for_TC_{level_hPa}hPa'

        
        
        if smooth:
            output_folder = f'{path_dir_data}/{data_type}/{folder_name}_sigma_{sigma}/daily/{year}'
        else:
            output_folder = f'{path_dir_data}/{data_type}/{folder_name}/daily/{year}'
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Загружаем датасет
        ds = process_file(ncfile, level_hPa=level_hPa, region=region)

        
        # Сеточные параметры
        dx = dlon * DEG_TO_M * np.cos(np.radians(ds.latitude.values))
        
        # Получаем дни в этом месяце
        days_in_month = np.unique(ds.time.dt.day.values)
        
        
        for day in days_in_month:
            ds_day = ds.sel(time=ds.time.dt.day == day)
            
            # ✅ Проверяем и исправляем порядок размерностей
            if ds_day['u'].dims != ('time', 'level', 'latitude', 'longitude'):
                print(f"Исправляем порядок: {ds_day['u'].dims} -> ('time', 'level', 'latitude', 'longitude')")
                ds_day = ds_day.transpose('time', 'level', 'latitude', 'longitude')
    
        
#         for t in range(len(ds.time)):
#             ds_day = ds.isel(time=slice(t, t+1))

#             day = ds_day.time.dt.day.values[0]
#             hour = ds_day.time.dt.hour.values[0]
            
            print(f"ds day shape: {ds_day['u'].shape}")

            
            output_file = f"{output_folder}/sigma_{sigma}_{name_crit}_{year}-{month:02d}-{day:02}.nc"
            
            if os.path.exists(output_file):
                print(f"{year}-{month:02d}-{day:02d} уже обработан, пропускаем")
                continue
            
            
            print(f"Обработка {year}-{month:02d}-{day:02d}")
            
            u = ds_day[params[0]] #.values[0,0]
            v = ds_day[params[1]] #.values[0,0]
            
            if smooth:
#                 u_smooth_2d = gaussian_filter(u, sigma=sigma, mode='reflect')
#                 v_smooth_2d = gaussian_filter(v, sigma=sigma, mode='reflect')
                u_smooth, v_smooth = vectorized_2d_smooth(u.values, v.values, sigma)
            else:
                u_smooth = u
                v_smooth = v
                
#             u_smooth = u_smooth_2d[np.newaxis, np.newaxis, :, :]  # (1, 1, 285, 509)
#             v_smooth = v_smooth_2d[np.newaxis, np.newaxis, :, :]  # (1, 1, 285, 509)

            print(f"smoothing done, shape: {u_smooth.shape}")
            # Вычисляем градиенты ветра
            du_dx, du_dy = regular_grid_gradient(u_smooth, dx, dy)
            dv_dx, dv_dy = regular_grid_gradient(v_smooth, dx, dy)

            omega_2d = compute_omega(du_dx, -du_dy, dv_dx, -dv_dy)
            print('omega done')


            sw_str_2d = swirling_strength(du_dx, -du_dy, dv_dx, -dv_dy)
            r2d = compute_rortex_2d(sw_str_2d, omega_2d)

            ds_smooth = ds_day[['u', 'v']].copy()
            # Добавляем R2D с правильными размерностями
            ds_smooth[name_crit] = (('time', 'level', 'latitude', 'longitude'), 
                                    r2d.astype(np.float32))


#             ds_smooth[name_crit] = ({time_name: len(ds_day[time_name]), 
#                           level_name: len(ds_day[level_name]), 
#                           y_name: len(ds_day[y_name]), x_name: len(ds_day[x_name])}, r2d.astype(np.float32))

            try:
                ds_smooth.to_netcdf(output_file, mode='w')
                print(f"Файл {output_file} успешно сохранен")
            except Exception as e:
                print(f"Ошибка при сохранении файла {output_file}: {str(e)}")

            
            
            # Очищаем
            del ds_smooth, omega_2d, sw_str_2d, r2d
            gc.collect()
        
        ds.close()
        del ds
        gc.collect()
        
        elapsed = (time.time() - start_time) / 60
        print(f"Месяц {year}-{month:02d} обработан за {elapsed:.2f} минут")