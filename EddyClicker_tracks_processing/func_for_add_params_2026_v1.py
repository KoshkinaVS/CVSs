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

# Modify your load_wrf_file function to include closing capability
FILE_CACHE = {}
MAX_CACHE_SIZE = 10  # Adjust based on your system's limits

def load_wrf_file(year, month, day, data_type='LoRes'):
    global FILE_CACHE
    
    if data_type == 'SMP':
        file_path = f'/storage/buffer/SMP/MODELS/WRF/OUTPUT/{year}/wrfout_d01_{year}-{month:02d}-{day:02d}_00.nc'
    else:
        # старый путь для LoRes
        name = f'wrfout_d01_{year}-{month:02d}-{day:02d}_00:00:00'
        file_path = f'/storage/NAAD/NAAD/LoRes/{year}/{name}'
    
    if file_path not in FILE_CACHE:
        if len(FILE_CACHE) >= MAX_CACHE_SIZE:
            oldest_key = next(iter(FILE_CACHE))
            FILE_CACHE[oldest_key].close()
            del FILE_CACHE[oldest_key]
        FILE_CACHE[file_path] = nc.Dataset(file_path)
    
    return FILE_CACHE[file_path]


def calculate_effective_radius(row):
    """
    Вычисляет эффективный радиус для одной строки данных
    """
    # Центр вихря
    x_center, y_center = row['x'], row['y']
    
    # Точки границ
    p1 = (row['px1_ind'], row['py1_ind'])
    p2 = (row['px2_ind'], row['py2_ind'])
    p3 = (row['px3_ind'], row['py3_ind'])
    
    # Вычисляем большую полуось как полусумму расстояний между p1 и p2
    distance_p1_p2 = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    semi_major = distance_p1_p2 / 2
    
    # Вычисляем центр между p1 и p2
    center_p1_p2 = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    
    # Вычисляем малую полуось как расстояние между центром p1-p2 и p3
    semi_minor = np.sqrt((center_p1_p2[0] - p3[0])**2 + (center_p1_p2[1] - p3[1])**2)
    
    # Эффективный радиус как сумма полуосей
    effective_radius = (semi_major + semi_minor)/2
    
    return effective_radius

def preprocessing_EC_tracks(df):

    # Выкидываем первую строку, если в DataFrame больше 1 строки
    if len(df) > 1:
        df = df.iloc[1:].reset_index(drop=True)
    else:
        # Если только одна строка, то делаем пустой DataFrame
        df = df.iloc[0:0]
                
    df = df.rename(columns={
                    'time': 'datetime',
                    'pxc_ind': 'x',
                    'pyc_ind': 'y'
                })
    df['rad'] = df.apply(calculate_effective_radius, axis=1)

    # Дополнительно можно добавить отдельно полуоси
    df['semi_major'] = df.apply(lambda row: np.sqrt((row['px1_ind'] - row['px2_ind'])**2 + 
                                                   (row['py1_ind'] - row['py2_ind'])**2) / 2, axis=1)
    
    df['semi_minor'] = df.apply(lambda row: np.sqrt(((row['px1_ind'] + row['px2_ind'])/2 - row['px3_ind'])**2 + 
                                                   ((row['py1_ind'] + row['py2_ind'])/2 - row['py3_ind'])**2), axis=1)

    df['elongation'] = [a/b if a<b else b/a for a, b in  zip(df['semi_major'], df['semi_minor'])]

    return df

def get_cached_dataset(path):
    if path not in FILE_CACHE:
        FILE_CACHE[path] = xr.open_dataset(path)
    return FILE_CACHE[path]

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
    Добавляет:
    - агрегированные параметры по окну (как раньше)
    - + slp_center и wspd_center (в центре трека)
    """
    # 1. Агрегированные параметры (как раньше)
    results = {f"{cfg['param']}_{cfg['agg']}_{cfg['level']}": [] for cfg in params_config}
    
    datetimes = df['datetime'].values
    lats = df['lat'].values if 'lat' in df.columns else None
    lons = df['lon'].values if 'lon' in df.columns else None
    rads = df['rad'].values

    # # 2. Новые: центральные значения
    # slp_center_vals = []
    # wspd_center_vals = []

    for i in tqdm(range(len(df))):
        t = df['datetime'].iloc[i]
        rad = rads[i]

        
        # --- Агрегированные параметры ---
        for cfg in params_config:

            param = cfg['param']
            agg_type = cfg['agg']
            level = cfg['level']
            
            
            
            if data_type == 'ERA5':
                if param == 'wspd':
                    val = compute_aggregation_wspd_ERA5(param, t, lats[i], lons[i], rad, level, data_type, km, agg_type)
                else:
                    val = compute_aggregation_value_ERA5(param, t, lats[i], lons[i], rad, level, data_type, km, agg_type)
            else:
                # Для LoRes/HiRes используем x, y
                val = get_mean_value(df, i, level, param, agg_type, data_type, km)
            results[f"{param}_{agg_type}_{level}"].append(val)

            print(f"  → Processed {t} | rad={rad:.1f} | {param} ({agg_type}) at {level}: {val}")
            

        # # --- Центральные значения ---
        # if data_type == 'ERA5':
        #     # В ERA5 работаем с lat/lon
        #     slp_c = get_center_value_ERA5('msl', t, lats[i], lons[i])
        #     wspd_c = get_center_value_ERA5_wspd(t, lats[i], lons[i])
        # else:
        #     # В WRF/NAAD работаем с x, y
        #     slp_c = get_center_value_wrf('slp', t, int(xs[i]), int(ys[i]), data_type)
        #     wspd_c = get_center_value_wrf('wspd', t, int(xs[i]), int(ys[i]), data_type)

        # slp_center_vals.append(slp_c)
        # wspd_center_vals.append(wspd_c)

    # # 3. Добавляем всё в DataFrame
    for col, values in results.items():
        df[col] = values
    # df['slp_center'] = slp_center_vals
    # df['wspd_center'] = wspd_center_vals
    
    return df

def get_center_value_wrf(param, t, x_idx, y_idx, data_type='LoRes'):
    try:
        ds = load_wrf_file(t.year, t.month, t.day, data_type=data_type)
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


def get_mean_value(CS, i, level, param, agg='median', data_type='LoRes', km=77):
    hw = max(1, int(CS['rad'].iloc[i]))
    y = int(CS['y'].iloc[i])
    x = int(CS['x'].iloc[i])
    
    # Просто задаём окно, без проверки через ds0
    y_in = max(y - hw, 0)
    x_in = max(x - hw, 0)
    y_out = y + hw
    x_out = x + hw

    # Убираем проверку ds0 — она не нужна для SMP и опасна для LoRes/HiRes при больших радиусах
    if param == 'wspd' and level == 0:
        func = compute_aggregation_wspd if data_type != 'SMP' else compute_aggregation_value_wrfout
    else:
        func = compute_aggregation_value_wrfout  # для SMP и WRF-подобных данных

    return func(param, CS['datetime'].iloc[i], y_in, y_out, x_in, x_out,
                our_level=level, data_type=data_type, km=km, agg=agg)

def compute_aggregation_value(param, t, y_in, y_out, x_in, x_out, our_level, data_type='LoRes', km=77, agg='min'):
    if our_level == 0:
        path_dir_raw = f'/storage/OPENDATA/NAAD/{data_type}/Surface/{param}'
        ncfile = f'{path_dir_raw}/NAAD{km}km_{param}_{t.year}.nc'
        ds = get_cached_dataset(ncfile)
        data = ds[param].sel(time=t, south_north=slice(y_in, y_out), west_east=slice(x_in, x_out))
    else:
        path_dir_raw = f'/storage/OPENDATA/NAAD/{data_type}/PressureLevels/{param}/{t.year}'
        ncfile = f'{path_dir_raw}/{param}_{t.year}-{t.month:02d}-{t.day:02d}.nc'
        ds = get_cached_dataset(ncfile)
        hour_idx = int((t.hour/3))
        level = ds['interp_level'][our_level].values
        data = ds[param].sel(Time=hour_idx, interp_level=level, south_north=slice(y_in, y_out), west_east=slice(x_in, x_out))
    
    agg_func = {
        'min': np.nanmin,
        'max': np.nanmax,
        'median': np.nanmedian
    }.get(agg.lower(), np.nanmedian)
    
    return agg_func(data)

def compute_aggregation_wspd(param, t, y_in, y_out, x_in, x_out, our_level, data_type='LoRes', km=77, agg='min'):
    u_path = f'/storage/OPENDATA/NAAD/{data_type}/Surface/u10e/NAAD{km}km_u10e_{t.year}.nc'
    v_path = f'/storage/OPENDATA/NAAD/{data_type}/Surface/v10e/NAAD{km}km_v10e_{t.year}.nc'
    
    u_ds = get_cached_dataset(u_path)
    v_ds = get_cached_dataset(v_path)
    
    wspd = np.sqrt(u_ds['u10e']**2 + v_ds['v10e']**2)
    data = wspd.sel(time=t, south_north=slice(y_in, y_out), west_east=slice(x_in, x_out))
    
    agg_func = {
        'min': np.nanmin,
        'max': np.nanmax,
        'median': np.nanmedian
    }.get(agg.lower(), np.nanmedian)
    
    return agg_func(data)

def compute_aggregation_value_wrfout(param, t, y_in, y_out, x_in, x_out, our_level, data_type='SMP', km=6, agg='median'):
    ds = load_wrf_file(t.year, t.month, t.day, data_type=data_type)
    hour_idx = t.hour  # SMP hourly
    if data_type != 'SMP':
        hour_idx = int((t.hour/3))

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
