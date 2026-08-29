import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import glob
from pathlib import Path
import os
import sys
from geopy.distance import great_circle
from tqdm import tqdm
import xarray as xr

import netCDF4 as nc
import wrf

# ✅ ИСПОЛЬЗУЕМ ТОЛЬКО ОДИН КЭШ С LRU (Least Recently Used)
from collections import OrderedDict
import functools


# Инициализация путей и параметров
path_init = f'/storage/thalassa/users/vkoshkina'
folder = 'scripts/CS_processing/CVS_tracking/after_70RAE/'
sys.path.insert(2, f'{path_init}/{folder}')

from step_of_tracking import *

if data_type == 'LoRes':
    path_dir_raw = f'/storage/OPENDATA/NAAD/{data_type}/Surface/msl'
    ncfile = f'{path_dir_raw}/NAAD77km_msl_2010.nc'
            
    ds0 = xr.open_dataset(f'{ncfile}')
elif data_type == 'HiRes':
    path_dir_raw = f'/storage/OPENDATA/NAAD/HiRes/PressureLevels/geopotential/2010/'
    ncfile = f'{path_dir_raw}/geopotential_2010-08-28.nc'
    
    ds0 = xr.open_dataset(f'{ncfile}')
elif data_type == 'ERA5':
    path_init = f'/storage/thalassa/users/vkoshkina/data'
    path_dir_raw = f'{path_init}/ERA5/ERA5_raw/ERA5_fluxes'

    # path_dir_raw = f'/storage/thalassa/DATA/ERA5/w10'
    ncfile = f'{path_dir_raw}/era5_fluxes_1979-01_cropped.nc'
    
    ds0 = xr.open_dataset(f'{ncfile}')
elif data_type == 'SMP':
    path_dir_raw = f'/storage/buffer/SMP/MODELS/WRF/OUTPUT/2019/'
    ncfile = f'{path_dir_raw}/wrfout_d01_2019-08-28_00.nc'
    
    ds0 = xr.open_dataset(f'{ncfile}')
    
agg_funcs = {
    'min': lambda x: np.nanpercentile(x, 5),
    'max': lambda x: np.nanpercentile(x, 95),
    'median': np.nanmedian,
    'mean': np.nanmean,
    'std': np.nanstd,
    'delta': lambda x: np.nanpercentile(x, 95) - np.nanpercentile(x, 5)
}




@functools.lru_cache(maxsize=16)  # Самый простой!
def get_dataset(path_or_key):
    """Универсальный кэш: path(str) или (y,m,d,type)(tuple)"""
    if isinstance(path_or_key, str):  # FILE_CACHE логика
        return xr.open_dataset(path_or_key)
    else:  # (year,month,day,data_type) → WRF_CACHE логика
        year, month, day, dtype = path_or_key
        if dtype == 'SMP':
            path = f'/storage/buffer/SMP/MODELS/WRF/OUTPUT/{year}/wrfout_d01_{year}-{month:02d}-{day:02d}_00.nc'
        else:
            name = f'wrfout_d01_{year}-{month:02d}-{day:02d}_00:00:00'
            path = f'/storage/NAAD/NAAD/{dtype}/{year}/{name}'
        return xr.open_dataset(path, chunks={})

def calculate_effective_radius_vectorized(df):
    p1x, p1y = df['px1_ind'], df['py1_ind']
    p2x, p2y = df['px2_ind'], df['py2_ind']
    p3x, p3y = df['px3_ind'], df['py3_ind']

    semi_major = np.sqrt((p1x - p2x)**2 + (p1y - p2y)**2) / 2
    center_x = (p1x + p2x) / 2
    center_y = (p1y + p2y) / 2
    semi_minor = np.sqrt((center_x - p3x)**2 + (center_y - p3y)**2)
    return (semi_major + semi_minor) / 2, semi_major, semi_minor

def preprocessing_EC_tracks(df):
    # Векторизованное удаление первой строки
    if len(df) > 1:
        df = df.iloc[1:].reset_index(drop=True)
    else:
        return df.iloc[0:0]
    
    # Переименование
    df = df.rename(columns={'time': 'datetime', 'pxc_ind': 'x', 'pyc_ind': 'y'})
    
    rad, semi_major, semi_minor = calculate_effective_radius_vectorized(df)

    df['rad'] = rad
    df['semi_minor'] = semi_minor
    df['semi_major'] = semi_major
    
    # Векторизованный elongation (избегаем if, используем np.maximum/abs)
    a, b = df['semi_major'].values, df['semi_minor'].values
    df['elongation'] = np.minimum(a/b, b/a)
    
    return df


def find_hires_indices(lon, lat, hires_ds):
    """Находит индексы (y, x) в HiRes данных, ближайшие к заданным координатам."""
    hires_lon = hires_ds.XLONG.values[0]
    hires_lat = hires_ds.XLAT.values[0]
    
    # Вычисляем расстояния до всех точек HiRes сетки
    distances = np.sqrt((hires_lon - lon)**2 + (hires_lat - lat)**2)
    y_idx, x_idx = np.unravel_index(np.argmin(distances), distances.shape)
    return y_idx, x_idx

def add_parameters_batch(df, params_config, data_type='LoRes', km=77):
    """
    Добавляет параметры с:
    1. Добавлением lat/lon если отсутствуют (один раз из ds0)
    2. Группировкой по дням для кэша ds
    3. Векторизацией где возможно
    """
    # 1. Добавляем lat/lon если нет (используем существующий ds0)
    if 'lat' not in df.columns or 'lon' not in df.columns:
        if data_type == 'LoRes' and ds0 is not None:
            y_idx = df['y'].astype(int).values  # shape (N,)
            x_idx = df['x'].astype(int).values  # shape (N,)
            df['lat'] = ds0['XLAT'].values[y_idx, x_idx] 
            df['lon'] = ds0['XLONG'].values[y_idx, x_idx] 
        
        else:
            print("Lat/lon не добавлены: ds0 отсутствует или не LoRes")
    
    # 2. Создаем date колонку для группировки (если нет)
    df['date'] = pd.to_datetime(df['datetime']).dt.date
    
    # 3. Инициализация результатов (dict списков → потом assign)
    result_cols = {f"{cfg['param']}_{cfg['agg']}_{cfg['level']}": [] for cfg in params_config}
    datetimes = df['datetime'].values
    rads = df['rad'].values
    xs = df['x'].astype(int).values
    ys = df['y'].astype(int).values
    lats = df['lat'].values
    lons = df['lon'].values
    
    # 4. Группировка по дням + обработка батчами
    for date, group in tqdm(df.groupby('date'), desc="Processing days"):
        year, month, day = pd.to_datetime(date).year, pd.to_datetime(date).month, pd.to_datetime(date).day
        ds_daily = get_dataset((year, month, day, data_type))  # Кэшированный ds на день
        
        group_idx = group.index.values  # Глобальные индексы строк
        group_rad = rads[group_idx]
        group_xs, group_ys = xs[group_idx], ys[group_idx]
        # group_lats, group_lons = lats[group_idx], lons[group_idx]
        group_times = datetimes[group_idx]
        
        # Батч по конфигам параметров
        for cfg in params_config:
            param, agg_type, level = cfg['param'], cfg['agg'], cfg['level']
            batch_vals = []
            
            for j in range(len(group_idx)):
                i_global = group_idx[j]
                # t, rad, x, y, lat, lon = group_times[j], group_rad[j], group_xs[j], group_ys[j], group_lats[j], group_lons[j]
                t, rad, x, y = group_times[j], group_rad[j], group_xs[j], group_ys[j]

                val = get_mean_value(df, i_global, level, param, agg_type, data_type, km)  # Использует индексы
                batch_vals.append(val)
                print(f"Day {date} → {t} | {param}_{agg_type}_{level}: {val}")
            
            result_cols[f"{param}_{agg_type}_{level}"].extend(batch_vals)
    
    # 5. Векторизованное присвоение в конец
    for col, values in result_cols.items():
        df[col] = values
    
    return df


def get_center_value_wrf(param, t, x_idx, y_idx, data_type='LoRes'):
    try:
        ds = get_dataset((t.year, t.month, t.day, data_type))
        hour_idx = t.hour
        if data_type != 'SMP':
            hour_idx = t.hour // 3

        # Отладка:
        # print(f"Reading {param} at {t} (hour_idx={hour_idx}), x={x_idx}, y={y_idx}, data_type={data_type}")

        if param == 'slp':
            slp = wrf.getvar(ds, "slp", timeidx=hour_idx)
            # print(f"SLP shape: {slp.shape}, value at center: {slp[y_idx, x_idx].item()}")
            return float(slp[y_idx, x_idx].item())
        elif param == 'wspd':
            u10 = wrf.getvar(ds, "U10", timeidx=hour_idx)
            v10 = wrf.getvar(ds, "V10", timeidx=hour_idx)
            wspd = np.sqrt(u10**2 + v10**2)
            # print(f"WSPD shape: {wspd.shape}, value: {wspd[y_idx, x_idx].item()}")
            return float(wspd[y_idx, x_idx].item())
        else:
            raise ValueError(f"Центральный параметр {param} не поддерживается")
    except Exception as e:
        print(f"❌ Ошибка центра WRF {param} в {t}: {e}")
        return np.nan

def get_center_value_ERA5(param, t, lat, lon):
    """
    Извлекает значение параметра в точке (lat, lon) из ERA5.
    """
    try:
        # Округляем до ближайшей точки сетки (0.25°)
        lat_era = np.round(lat / 0.25) * 0.25
        lon_era = np.round(lon / 0.25) * 0.25
        lon_era = (lon_era + 360) % 360  # в [0, 360]

        if param == 'msl':
            ncfile = f"{path_init}/ERA5/ERA5_raw/ERA5_slp/era5_slp_{t.year}-{t.month:02d}_cropped.nc"
        else:
            raise NotImplementedError(f"Центральный параметр {param} не реализован для ERA5")

        ds = get_dataset(ncfile)
        value = ds[param].sel(time=t, latitude=lat_era, longitude=lon_era, method='nearest').item()
        return float(value)
    except Exception as e:
        # print(f"Ошибка центра ERA5 {param} в {t}: {e}")
        return np.nan


def get_center_value_ERA5_wspd(t, lat, lon):
    """
    Извлекает wspd в центре из ERA5.
    """
    try:
        lat_era = np.round(lat / 0.25) * 0.25
        lon_era = np.round(lon / 0.25) * 0.25
        lon_era = (lon_era + 360) % 360

        ncfile = f"{path_init}/ERA5/ERA5_raw/ERA5_w10/era5_w10_{t.year}-{t.month:02d}_cropped.nc"
        ds = get_dataset(ncfile)
        
        u10 = ds['u10'].sel(time=t, latitude=lat_era, longitude=lon_era, method='nearest').item()
        v10 = ds['v10'].sel(time=t, latitude=lat_era, longitude=lon_era, method='nearest').item()
        wspd = np.sqrt(u10**2 + v10**2)
        return float(wspd)
    except Exception as e:
        # print(f"Ошибка wspd центра ERA5 в {t}: {e}")
        return np.nan

def get_mean_value(CS, i, level, param, agg='median', data_type='LoRes', km=77):
    hw = int(CS['rad'].iloc[i])
    y = int(CS['y'].iloc[i])
    x = int(CS['x'].iloc[i])
    
    # Просто задаём окно, без проверки через ds0
    y_in = max(y - hw, 0)
    x_in = max(x - hw, 0)
    y_out = y + hw
    x_out = x + hw

    if param == 'wspd':
        func = compute_aggregation_wspd if data_type != 'SMP' else compute_aggregation_value_wrfout
    else:
        func = compute_aggregation_value_wrfout  # для SMP и WRF-подобных данных

    return func(param, CS['datetime'].iloc[i], y_in, y_out, x_in, x_out,
                our_level=level, data_type=data_type, km=km, agg=agg)

def compute_aggregation_value(param, t, y_in, y_out, x_in, x_out, our_level, data_type='LoRes', km=77, agg='min'):
    if our_level == 0:
        path_dir_raw = f'/storage/OPENDATA/NAAD/{data_type}/Surface/{param}'
        ncfile = f'{path_dir_raw}/NAAD{km}km_{param}_{t.year}.nc'
        ds = get_dataset(ncfile)
        data = ds[param].sel(time=t, south_north=slice(y_in, y_out), west_east=slice(x_in, x_out))
    else:
        path_dir_raw = f'/storage/OPENDATA/NAAD/{data_type}/PressureLevels/{param}/{t.year}'
        ncfile = f'{path_dir_raw}/{param}_{t.year}-{t.month:02d}-{t.day:02d}.nc'
        ds = get_dataset(ncfile)
        hour_idx = int((t.hour/3))
        data = ds[param].isel(Time=hour_idx, interp_level=our_level, south_north=slice(y_in, y_out), west_east=slice(x_in, x_out))
    
    agg_func = {
        'min': np.nanmin,
        'max': np.nanmax,
        'median': np.nanmedian
    }.get(agg.lower(), np.nanmedian)
    
    return agg_func(data)

def compute_aggregation_wspd(param, t, y_in, y_out, x_in, x_out, our_level, data_type='LoRes', km=77, agg='min'):

    t_idx = int(t.hour/3)
    
    if our_level == 0:
        u_path = f'/storage/OPENDATA/NAAD/{data_type}/Surface/u10e/NAAD{km}km_u10e_{t.year}.nc'
        v_path = f'/storage/OPENDATA/NAAD/{data_type}/Surface/v10e/NAAD{km}km_v10e_{t.year}.nc'
        u_ds = get_dataset(u_path)
        v_ds = get_dataset(v_path)
        u = u_ds['u10e'].sel(time=t, south_north=slice(y_in, y_out), west_east=slice(x_in, x_out))
        v = v_ds['v10e'].sel(time=t, south_north=slice(y_in, y_out), west_east=slice(x_in, x_out))
    else:
        ds = get_dataset((t.year, t.month, t.day, data_type))
        u = ds['U'].isel(Time=t_idx, bottom_top=our_level, south_north=slice(y_in, y_out), west_east_stag=slice(x_in, x_out))
        v = ds['V'].isel(Time=t_idx, bottom_top=our_level, south_north_stag=slice(y_in, y_out), west_east=slice(x_in, x_out))
        
    wspd = np.sqrt(u**2 + v**2)

    agg_func = {
        'min': np.nanmin,
        'max': np.nanmax,
        'median': np.nanmedian
    }.get(agg.lower(), np.nanmedian)
    
    return agg_func(wspd)

def compute_aggregation_value_wrfout(param, t, y_in, y_out, x_in, x_out, our_level, data_type='SMP', km=6, agg='median'):
    ds = get_dataset((t.year, t.month, t.day, data_type))
    hour_idx = t.hour  # SMP hourly
    if data_type != 'SMP':
        hour_idx = t.hour // 3

    try:
        if param == 'wspd':
            if our_level == 0:
                u10 = wrf.getvar(ds, 'U10', timeidx=hour_idx)
                v10 = wrf.getvar(ds, 'V10', timeidx=hour_idx)
                data = np.sqrt(u10**2 + v10**2)
            else:
                u = wrf.getvar(ds, 'ua', timeidx=hour_idx)[our_level]
                v = wrf.getvar(ds, 'va', timeidx=hour_idx)[our_level]
                data = np.sqrt(u**2 + v**2)
                
        # elif param == 'theta_e':
        #     theta = wrf.getvar(ds, 'theta', timeidx=hour_idx)
        #     qv = wrf.getvar(ds, 'QVAPOR', timeidx=hour_idx)
        #     # Bolton (1980) approximation
        #     T = wrf.getvar(ds, 'tk', timeidx=hour_idx)  # temperature in K
        #     P = wrf.getvar(ds, 'pressure', timeidx=hour_idx) * 100  # hPa → Pa
        #     e = qv * P / (0.622 + 0.378 * qv)
        #     L = 2.501e6  # latent heat
        #     Cp = 1005.0
        #     theta_e_data = theta * np.exp((L * qv) / (Cp * T))
        #     if our_level > 0:
        #         data = theta_e_data[our_level]
        #     else:
        #         data = theta_e_data


        elif param == ['T2', 'temp', 'theta', 'theta_e']:
            tk = wrf.getvar(ds, param, timeidx=hour_idx)  # K
            if our_level > 0:
                data = tk[our_level]
            else:
                data = tk

        elif param in ['LH', 'HFX', 'SST']:
            if param in ds.variables:
                var = ds.variables[param][hour_idx]
                if len(var.shape) == 2:  # 3D surface field
                    data = var
                elif len(var.shape) == 3:  # 4D — берем нижний уровень
                    data = var[0]
            else:
                return np.nan
        

        elif param in ['slp', 'cape_2d', 'pw', 'helicity', 'updraft_helicity']:
            data = wrf.getvar(ds, param, timeidx=hour_idx)

        elif param == 'pblh':
            data = wrf.getvar(ds, 'PBLH', timeidx=hour_idx)

        else:
            # Стандартный случай через wrf.getvar
            if our_level == 0:
                data = wrf.getvar(ds, param, timeidx=hour_idx)
            else:
                data = wrf.getvar(ds, param, timeidx=hour_idx)[our_level]

        # Обрезка окна
        data_cropped = data[slice(y_in, y_out), slice(x_in, x_out)]

        # Обработка NaN
        if param in ['cape_2d', 'cape_3d']:
            data_cropped = np.nan_to_num(data_cropped, nan=0.0)

        # Агрегация
        if agg == 'delta':
            result = np.nanpercentile(data_cropped, 95) - np.nanpercentile(data_cropped, 5)
        else:
            result = agg_funcs.get(agg.lower(), np.nanmedian)(data_cropped)

        return result

    except Exception as e:
        print(f"⚠️ Ошибка при вычислении {param} в {t}: {e}")
        return np.nan

def compute_aggregation_value_ERA5(param, t, rounded_lat, rounded_lon, rad, our_level, data_type='ERA5', km=25, agg='min'):
    """Вычисляет агрегированное значение для заданных параметров и времени."""
    # 1. Подготовка координат области
    hw = rad * 0.25
    x_in = max(rounded_lon - hw * np.cos(np.radians(rounded_lat)), -180)
    x_out = min(rounded_lon + hw * np.cos(np.radians(rounded_lat)), 180)
    y_in = rounded_lat + hw
    y_out = rounded_lat - hw

    # 2. Конфигурация параметров в зависимости от параметра и уровня
    param_config = {
        'msl': {
            'dir': 'ERA5_slp',
            'file_prefix': 'era5_slp',
            'lat_dim': 'latitude',
            'lon_dim': 'longitude'
        },
        'mlhf': {
            'dir': 'ERA5_fluxes',
            'file_prefix': 'era5_fluxes',
            'lat_dim': 'lat',
            'lon_dim': 'lon'
        },
        'mshf': {
            'dir': 'ERA5_fluxes',
            'file_prefix': 'era5_fluxes',
            'lat_dim': 'lat',
            'lon_dim': 'lon'
        },
        'tp': {
            'dir': 'ERA5_precip',
            'file_prefix': 'era5_precip',
            'lat_dim': 'latitude',
            'lon_dim': 'longitude'
        },
        't2m': {
            'dir': 'ERA5_t2',
            'file_prefix': 'era5_t2',
            'lat_dim': 'latitude',
            'lon_dim': 'longitude'
        },
        'default': {
            'dir': f'ERA5_{param}',
            'file_prefix': f'era5_{param}',
            'lat_dim': 'latitude',
            'lon_dim': 'longitude'
        }
    }

    level_config = {
        0: {
            'get_config': lambda: param_config.get(param, param_config['default'])
        },
        500: {
            'dir': 'ERA5_500',
            'file_prefix': 'era5_uvwth_500hPa',
            'lat_dim': 'latitude',
            'lon_dim': 'longitude',
            'extra_sel': {'pressure_level': 500}
        }
    }

    # 3. Получение окончательной конфигурации
    config = level_config.get(our_level, level_config[0])
    if our_level == 0:
        config.update(config['get_config']())
    elif our_level != 500:
        print(f"Уровень {our_level} не поддерживается")
        return np.nan


    t_datetime = t.astype('datetime64[s]').item()  # Convert to Python datetime
    ncfile = f"{path_init}/ERA5/ERA5_raw/{config['dir']}/{config['file_prefix']}_{t_datetime.year}-{t_datetime.month:02d}_cropped.nc"

    # 4. Формирование пути к файлу
    # ncfile = f"{path_init}/ERA5/ERA5_raw/{config['dir']}/{config['file_prefix']}_{t.year}-{t.month:02d}_cropped.nc"
    
    # 5. Загрузка и обработка данных
    try:
        ds = get_dataset(ncfile)
        
        # Определение правильного измерения времени
        time_dim = None
        for possible_time_dim in ['time', 'valid_time']:
            if possible_time_dim in ds.dims:
                time_dim = possible_time_dim
                break
        
        if time_dim is None:
            print(f"Не найдено подходящего измерения времени в файле {ncfile}")
            return np.nan
            
        if t not in ds[time_dim]:
            print(f"Время {t} отсутствует в файле {ncfile}")
            return np.nan

        # Параметры для выборки данных
        sel_params = {
            time_dim: t,
            config['lat_dim']: slice(y_in, y_out),
            config['lon_dim']: slice(x_in, x_out)
        }

        if 'pressure_level' in ds.dims:
            sel_params.update(config['extra_sel'])
            
        # # Дополнительные параметры для уровня 500
        # if 'extra_sel' in config:
        #     sel_params.update(config['extra_sel'])

        data = ds[param].sel(**sel_params)

        if data.isnull().all():
            print(f"Нет данных для {t} и {param} в указанном диапазоне")
            return np.nan

    except (FileNotFoundError, KeyError) as e:
        print(f"Ошибка загрузки данных: {str(e)}")
        return np.nan


    return agg_funcs.get(agg.lower(), np.nanmedian)(data)


def compute_aggregation_wspd_ERA5(param, t, rounded_lat, rounded_lon, rad, our_level, data_type='LoRes', km=77, agg='min'):
    
    hw = rad * 0.25
    # rounded_lat = np.round(lat / 0.25) * 0.25
    # rounded_lon = np.round(lon / 0.25) * 0.25

    x_in = max(rounded_lon - hw * np.cos(np.radians(rounded_lat)), -180)
    x_out = min(rounded_lon + hw * np.cos(np.radians(rounded_lat)), 180)
    y_in = rounded_lat + hw
    y_out = rounded_lat - hw

    path_dir_raw = f'{path_init}/ERA5/ERA5_raw/ERA5_w10'

    t_datetime = t.astype('datetime64[s]').item()  # Convert to Python datetime

    ncfile = f'{path_dir_raw}/era5_w10_{t_datetime.year}-{t_datetime.month:02d}_cropped.nc'
    ds = get_dataset(ncfile)
    if t not in ds['time']:
        print(f"Время {t} отсутствует в файле {ncfile}")
        return np.nan
        
    data = ds.sel(time=t, latitude=slice(y_in, y_out), longitude=slice(x_in, x_out))
    
    wspd = np.sqrt(data['u10']**2 + data['v10']**2)

    result = agg_funcs.get(agg.lower(), np.nanmedian)(wspd)
    
    return result
    

def compute_aggregation_value_ERA5_raw(param, t, lat, lon, rad, our_level, data_type='LoRes', km=77, agg='min'):
    if param == 'msl':
        path_dir_raw = f'/storage/thalassa/DATA/ERA5/slp'
        ncfile = f'{path_dir_raw}/era5_mslp_{t.year}-{t.month:02d}.nc'
    elif param in ['mlhf', 'mshf']:
        path_dir_raw = f'/storage/thalassa/DATA/ERA5/fluxes'
        ncfile = f'{path_dir_raw}/era5_fluxes_{t.year}-{t.month:02d}.nc'
    else:
        path_dir_raw = f'/storage/thalassa/DATA/ERA5/{param}'
        ncfile = f'{path_dir_raw}/era5_{param}_{t.year}-{t.month:02d}.nc'

    ###### проверить корректность размера ######
    hw = rad * 0.25  # Convert radius from grid points to degrees
    
    # Округляем до 0.25
    rounded_lat = np.round(lat / 0.25) * 0.25
    rounded_lon = np.round(lon / 0.25) * 0.25
    
    # Преобразуем долготу в диапазон 0–360
    lon_era5 = (rounded_lon + 360) % 360

    # Проверяем, не выходит ли окно за пределы 0–360
    lon_min = max(lon_era5 - hw * np.cos(np.radians(lat)), 0)
    lon_max = min(lon_era5 + hw * np.cos(np.radians(lat)), 360)
    
    try:
        ds = get_dataset(ncfile)
        if param != 'msl':
            ds = ds.rename({'lat': 'latitude', 'lon': 'longitude'})

        data = ds[param].sel(time=t, 
                            latitude=slice(lat + hw, lat - hw), 
                            longitude=slice(lon_min, lon_max))
        

        return agg_funcs.get(agg.lower(), np.nanmedian)(data)
    except Exception as e:
        print(f"Error processing ERA5 data for {param} at {t}: {str(e)}")
        return np.nan

def compute_aggregation_wspd_ERA5_raw(param, t, lat, lon, rad, our_level, data_type='LoRes', km=77, agg='min'):
    path_dir_raw = f'/storage/thalassa/DATA/ERA5/w10'
    ncfile = f'{path_dir_raw}/era5_uv10m_{t.year}-{t.month:02d}.nc'
        
    hw = rad * 0.25
    
    # Округляем до 0.25
    rounded_lat = np.round(lat / 0.25) * 0.25
    rounded_lon = np.round(lon / 0.25) * 0.25
    
    # Преобразуем долготу в диапазон 0–360
    lon_era5 = (rounded_lon + 360) % 360

    # Проверяем, не выходит ли окно за пределы 0–360
    lon_min = max(lon_era5 - hw * np.cos(np.radians(lat)), 0)
    lon_max = min(lon_era5 + hw * np.cos(np.radians(lat)), 360)
    
    try:
        ds = get_dataset(ncfile)
        ds['wspd'] = np.sqrt(ds['u10']**2 + ds['v10']**2)
        
        data = ds['wspd'].sel(time=t, 
                             latitude=slice(lat + hw, lat - hw), 
                             longitude=slice(lon_min, lon_max))
        
        return agg_funcs.get(agg.lower(), np.nanmedian)(data)
    except Exception as e:
        print(f"Error processing ERA5 wind speed at {t}: {str(e)}")
        return np.nan