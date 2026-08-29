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

# Инициализация путей и параметров
path_init = f'/storage/thalassa/users/vkoshkina'
folder = 'scripts/CS_processing/CVS_tracking/after_70RAE/'
sys.path.insert(2, f'{path_init}/{folder}')

from step_of_tracking import *

path_init = f'/storage/thalassa/users/vkoshkina'
tracks_dir = f'TC_tracks/NAAD_NOAA_tracks_merged/{data_type}_sigma_{sigma}/tracking_local_2_phase_adv_speed_all_points_bound'
files = glob.glob(f"{path_init}/data/{tracks_dir}/AL*.csv")
TCs = []
TC_names = []

for file in files:
    df = pd.read_csv(file, na_values=-999, parse_dates=['datetime'])

    if len(df.dropna(subset=['lat_NAAD_0'])) > 24:
        if df['status'].str.contains('TS|HU').any():
            if all("_NAAD_1" not in col for col in df.columns):
                valid_rows = df.dropna(subset=['lat', 'lon', 'lat_NAAD_0', 'lon_NAAD_0'])
    
                if len(valid_rows) > 5:
                    # Вычисление расстояний между точками
                    distances = []
                    for index, row in valid_rows.iterrows():
                        point1 = (row['lat'], row['lon'])
                        point2 = (row['lat_NAAD_0'], row['lon_NAAD_0'])
                        distance = great_circle(point1, point2).kilometers
                        distances.append(distance)
                    valid_dist = [d <= 3*dist_m/1000 for d in distances]
                    # Среднее расстояние
                    average_distance = np.nanmedian(distances) if distances else 0
        
                    if average_distance <= 3*dist_m/1000:
                        if sum(valid_dist) > 8:
                            TCs.append(df)
                            file_name = Path(file).stem
                            TC_names.append(file_name)


path_init = f'/storage/thalassa/users/vkoshkina'
track_folder = f'data/TC_tracks/NAAD_NOAA_with_params_{data_type}_sigma_{sigma}'
files = glob.glob(f'{path_init}/{track_folder}/*.csv')
track_folder = f'data/TC_tracks/NAAD_NOAA_with_params/{data_type}_sigma_{sigma}'
files_NOAA = glob.glob(f'{path_init}/{track_folder}/*_NOAA.csv')

TCs = []
TCs_NOAA = []

TC_names = []

for file, file_NOAA in zip(files, files_NOAA):
    df = pd.read_csv(file, parse_dates=['datetime'])
    df_NOAA = pd.read_csv(file_NOAA, parse_dates=['datetime'])
    
    TCs.append(df)
    TCs_NOAA.append(df_NOAA)
    file_name = Path(file).stem        
    TC_names.append(file_name)


path_dir_raw = f'/storage/OPENDATA/NAAD/HiRes/PressureLevels/geopotential/2010/'
ncfile = f'{path_dir_raw}/geopotential_2010-08-28.nc'

hires_data = xr.open_dataset(f'{ncfile}')

path_dir_raw = f'/storage/tartar/DATA/ERA5/w10'
ncfile = f'{path_dir_raw}/era5_uv10m_2010-09.nc'

era5_data = xr.open_dataset(f'{ncfile}')

path_dir_raw = f'/storage/OPENDATA/NAAD/{data_type}/Surface/msl'
ncfile = f'{path_dir_raw}/NAAD77km_msl_2010.nc'

ds = xr.open_dataset(f'{ncfile}')

# Кэширование открытых файлов
FILE_CACHE = {}

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
    Добавляет несколько параметров за один проход по данным
    params_config: список словарей с настройками параметров
    Пример: [{'param': 'msl', 'level': 12, 'agg': 'min'}, ...]
    """
    results = {f"{cfg['param']}_{data_type}": [] for cfg in params_config}
    
    for i in tqdm(range(len(df)), desc=f"Adding {data_type} parameters"):
        for cfg in params_config:
            param = cfg['param']
            if data_type == 'ERA5':
                if param == 'wspd':
                    value = compute_aggregation_wspd_ERA5(param, df['datetime'][i], df['lat'][i], 
                                                         df['lon'][i], df['rad'][i], cfg['level'], 
                                                         data_type=data_type, km=km, agg=cfg.get('agg', 'median'))
                else:
                    value = compute_aggregation_value_ERA5(param, df['datetime'][i], df['lat'][i], 
                                                          df['lon'][i], df['rad'][i], cfg['level'], 
                                                          data_type=data_type, km=km, agg=cfg.get('agg', 'median'))
            else:
                value = get_mean_value(df, i, cfg['level'], param, 
                                     agg=cfg.get('agg', 'median'), 
                                     data_type=data_type, km=km)
            results[f"{param}_{data_type}"].append(value)
    
    for col, values in results.items():
        df[col] = values
    return df

def get_mean_value(CS, i, level, param, agg='median', post_name='', data_type='LoRes', km=77):
    rad = CS[f'rad{post_name}'][i]
    hw = int((rad) * 77 / km)
    
    y = CS[f'y{post_name}'][i]
    x = CS[f'x{post_name}'][i]
    lat = CS['lat'][i]
    lon = CS['lon'][i]

    if data_type == 'HiRes':
        y, x = find_hires_indices(float(lon), float(lat), hires_data)
    
    y_int, x_int = int(y), int(x)
    y_in, x_in = max(y_int - hw, 0), max(x_int - hw, 0)
    
    if data_type == 'HiRes':
        y_out = -1 if y_int + hw >= len(hires_data.south_north) else y_int + hw
        x_out = -1 if x_int + hw >= len(hires_data.west_east) else x_int + hw
    else:
        y_out = -1 if y_int + hw >= len(ds.south_north) else y_int + hw
        x_out = -1 if x_int + hw >= len(ds.west_east) else x_int + hw

    if data_type in ['HiRes', 'LoRes']:
        if param == 'wspd':
            func = compute_aggregation_wspd
        else:
            func = compute_aggregation_value
        return func(param, CS['datetime'][i], y_in, y_out, x_in, x_out, 
                   our_level=level, data_type=data_type, km=km, agg=agg)

def compute_aggregation_value(param, t, y_in, y_out, x_in, x_out, our_level, data_type='LoRes', km=77, agg='min'):
    path_dir_raw = f'/storage/OPENDATA/NAAD/{data_type}/Surface/{param}'
    ncfile = f'{path_dir_raw}/NAAD{km}km_{param}_{t.year}.nc'
    
    ds = get_cached_dataset(ncfile)
    data = ds[param].sel(time=t, south_north=slice(y_in, y_out), west_east=slice(x_in, x_out))
    
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

def compute_aggregation_value_ERA5(param, t, lat, lon, rad, our_level, data_type='LoRes', km=77, agg='min'):
    if param == 'msl':
        path_dir_raw = f'/storage/tartar/DATA/ERA5/slp'
        ncfile = f'{path_dir_raw}/era5_mslp_{t.year}-{t.month:02d}.nc'
    elif param in ['mlhf', 'mshf']:
        path_dir_raw = f'/storage/tartar/DATA/ERA5/fluxes'
        ncfile = f'{path_dir_raw}/era5_fluxes_{t.year}-{t.month:02d}.nc'
        
    hw = rad * 77 / 111  # Convert radius from grid points to degrees
    
    # Округляем до 0.25
    rounded_lat = np.round(lat / 0.25) * 0.25
    rounded_lon = np.round(lon / 0.25) * 0.25
    
    # Преобразуем долготу в диапазон 0–360
    lon_era5 = (rounded_lon + 360) % 360

    # Проверяем, не выходит ли окно за пределы 0–360
    lon_min = max(lon_era5 - hw * np.cos(np.radians(lat)), 0)
    lon_max = min(lon_era5 + hw * np.cos(np.radians(lat)), 360)
    
    try:
        ds = get_cached_dataset(ncfile)
        if param != 'msl':
            ds = ds.rename({'lat': 'latitude', 'lon': 'longitude'})

        data = ds[param].sel(time=t, 
                            latitude=slice(lat + hw, lat - hw), 
                            longitude=slice(lon_min, lon_max))
        
        agg_func = {
            'min': np.nanmin,
            'max': np.nanmax,
            'median': np.nanmedian
        }.get(agg.lower(), np.nanmedian)

        # print(agg_func(data))
        return agg_func(data)
    except Exception as e:
        print(f"Error processing ERA5 data for {param} at {t}: {str(e)}")
        return np.nan

def compute_aggregation_wspd_ERA5(param, t, lat, lon, rad, our_level, data_type='LoRes', km=77, agg='min'):
    path_dir_raw = f'/storage/tartar/DATA/ERA5/w10'
    ncfile = f'{path_dir_raw}/era5_uv10m_{t.year}-{t.month:02d}.nc'
        
    hw = rad * 77 / 111
    
    # Округляем до 0.25
    rounded_lat = np.round(lat / 0.25) * 0.25
    rounded_lon = np.round(lon / 0.25) * 0.25
    
    # Преобразуем долготу в диапазон 0–360
    lon_era5 = (rounded_lon + 360) % 360

    # Проверяем, не выходит ли окно за пределы 0–360
    lon_min = max(lon_era5 - hw * np.cos(np.radians(lat)), 0)
    lon_max = min(lon_era5 + hw * np.cos(np.radians(lat)), 360)
    
    try:
        ds = get_cached_dataset(ncfile)
        ds['wspd'] = np.sqrt(ds['u10']**2 + ds['v10']**2)
        
        data = ds['wspd'].sel(time=t, 
                             latitude=slice(lat + hw, lat - hw), 
                             longitude=slice(lon_min, lon_max))
        
        agg_func = {
            'min': np.nanmin,
            'max': np.nanmax,
            'median': np.nanmedian
        }.get(agg.lower(), np.nanmedian)
        
        return agg_func(data)
    except Exception as e:
        print(f"Error processing ERA5 wind speed at {t}: {str(e)}")
        return np.nan

# Основной цикл обработки
params_config = [
    {'param': 'msl', 'level': 12, 'agg': 'min'},
    {'param': 'mslhf', 'level': 12, 'agg': 'max'},
    {'param': 'msshf', 'level': 12, 'agg': 'max'},
    {'param': 'wspd', 'level': 12, 'agg': 'max'}
]

era5_params_config = [
    {'param': 'msl', 'level': 12, 'agg': 'min'},
    {'param': 'mlhf', 'level': 12, 'agg': 'max'},
    {'param': 'mshf', 'level': 12, 'agg': 'max'},
    {'param': 'wspd', 'level': 12, 'agg': 'max'}
]

for idx, TC in tqdm(enumerate(TCs), total=len(TCs), desc="Processing TCs"):
    df_NAAD = TC.dropna(subset=['lon_NAAD_0']).copy().reset_index(drop=True)
    
    rename_map = {
        'lat': 'lat_NOAA',
        'lon': 'lon_NOAA',
        'lat_NAAD_0': 'lat',
        'lon_NAAD_0': 'lon',
        'x_NAAD_0': 'x',
        'y_NAAD_0': 'y',
        't_NAAD_0': 't',
        'crit_NAAD_0': 'crit',
        'rad_NAAD_0': 'rad'
    }
    
    df_NAAD = df_NAAD.rename(columns=rename_map)
    
    df_NAAD = df_NAAD[['x', 'y', 'lat', 'lon', 'rad', 'crit', 'track_len_NAAD_0', 'datetime']]

    # Добавляем параметры из разных источников
    df = add_parameters_batch(df_NAAD, params_config, data_type='LoRes', km=77)
    df = add_parameters_batch(df, params_config, data_type='HiRes', km=14)
    df = add_parameters_batch(df, era5_params_config, data_type='ERA5', km=25)

    # Сохраняем результаты
    folder = f'{path_init}/data/TC_tracks/NAAD_NOAA_with_params/{data_type}_sigma_{sigma}'
    if not os.path.exists(f"{folder}"):
        os.makedirs(f"{folder}")
            
    df.to_csv(f'{folder}/{TC_names[idx]}_with_params.csv', index=False)