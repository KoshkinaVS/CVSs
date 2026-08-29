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

agg_funcs = {
        'min': lambda x: np.nanpercentile(x, 5),
        'max': lambda x: np.nanpercentile(x, 95),
        'median': np.nanmedian,
        'mean': np.nanmean,
        'std': np.nanstd
    }

# Modify your load_wrf_file function to include closing capability
FILE_CACHE = {}
MAX_CACHE_SIZE = 10  # Adjust based on your system's limits

def load_wrf_file(year, month, day):
    global FILE_CACHE
    
    name = f'wrfout_d01_{year}-{month:02d}-{day:02d}_00:00:00'
    path = f'/storage/NAAD/NAAD/LoRes/{year}'
    file_path = os.path.join(path, name)

    if file_path not in FILE_CACHE:
        # If cache is full, close the oldest file
        if len(FILE_CACHE) >= MAX_CACHE_SIZE:
            oldest_key = next(iter(FILE_CACHE))
            FILE_CACHE[oldest_key].close()
            del FILE_CACHE[oldest_key]
        
        FILE_CACHE[file_path] = nc.Dataset(file_path)
    
    return FILE_CACHE[file_path]

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
    
def add_new_parameters_to_tracks(data_type, path_tracks_dir, params_config_new, km, output_suffix='_updated'):
    """Добавляет новые параметры к существующим трекам"""
    
    if data_type == 'LoRes':
        years = np.arange(1979, 2019)
    elif data_type == 'ERA5':
        years = np.arange(1979, 2025)
    else:
        years = np.arange(1979, 2019)
    
    months = np.arange(1, 13)
    
    for year in years:
        for month in tqdm(months, desc=f"Processing year {year}"):
            # Путь к исходным файлам
            original_path = f'{path_tracks_dir}/{year}-{month:02d}'
            
            if not os.path.exists(original_path):
                continue
            
            # Создаем новую папку для обновленных файлов
            updated_path = f'{path_tracks_dir}{output_suffix}/{year}-{month:02d}'
            os.makedirs(updated_path, exist_ok=True)
            
            # Получаем все CSV файлы
            files = sorted(glob.glob(f'{original_path}/*.csv'))
            
            for file in tqdm(files, desc=f"Processing {year}-{month:02d}"):
                # Читаем существующий трек
                df = pd.read_csv(file, parse_dates=['datetime'])
                
                # Проверяем, какие параметры уже есть
                params_to_add = []
                for cfg in params_config_new:
                    if cfg['level'] == 0:
                        col_name = f"{cfg['param']}_{cfg['agg']}"
                    else:
                        col_name = f"{cfg['param']}_{cfg['level']}_{cfg['agg']}"
                    
                    if col_name not in df.columns:
                        params_to_add.append(cfg)
                
                if not params_to_add:
                    print(f"All parameters already present in {os.path.basename(file)}")
                    # Все равно копируем файл в новую папку
                    df.to_csv(f'{updated_path}/{os.path.basename(file)}', index=False)
                    continue
                
                # print(f"Adding {len(params_to_add)} new parameters to {os.path.basename(file)}")
                for cfg in params_to_add:
                    if cfg['level'] == 0:
                        print(f"  - {cfg['param']}_{cfg['agg']}")
                    else:
                        print(f"  - {cfg['param']}_{cfg['level']}_{cfg['agg']}")
                
                # Добавляем только новые параметры
                df = add_parameters_batch_with_level(df, params_to_add, data_type=data_type, km=km)
                
                # Сохраняем обновленный файл
                output_file = f'{updated_path}/{os.path.basename(file)}'
                df.to_csv(output_file, index=False)
                # print(f"✅ Saved to {output_file}")


def add_parameters_batch_with_level(df, params_config, data_type='LoRes', km=77):
    """Добавляет параметры с указанием уровня в имени колонки"""
    results = {}
    
    datetimes = df['datetime'].values
    lats = df['lat'].values
    lons = df['lon'].values
    rads = df['rad'].values
    
    for cfg in params_config:
        param = cfg['param']
        agg_type = cfg['agg']
        level = cfg['level']
        
        # Создаем имя колонки с уровнем
        if level == 0:
            col_name = f"{param}_{agg_type}"  # для поверхностных параметров
        else:
            col_name = f"{param}_{level}_{agg_type}"  # для уровней выше поверхности
        
        if data_type == 'ERA5':
            if param == 'wspd':
                values = [compute_aggregation_wspd_ERA5(param, dt, lat, lon, rad, level, 
                         data_type, km, agg_type) 
                        for dt, lat, lon, rad in zip(datetimes, lats, lons, rads)]
            else:
                values = [compute_aggregation_value_ERA5(param, dt, lat, lon, rad, level, 
                         data_type, km, agg_type) 
                        for dt, lat, lon, rad in zip(datetimes, lats, lons, rads)]
        else:
            values = [get_mean_value(df, i, level, param, agg_type, data_type, km) 
                     for i in range(len(df))]
        
        results[col_name] = values
    
    for col, values in results.items():
        df[col] = values
    return df

def add_parameters_batch(df, params_config, data_type='LoRes', km=77):
    results = {f"{cfg['param']}_{cfg['agg']}": [] for cfg in params_config}
    
    datetimes = df['datetime'].values
    lats = df['lat'].values
    lons = df['lon'].values
    rads = df['rad'].values
    
    for cfg in params_config:
        param = cfg['param']
        agg_type = cfg['agg']
        level = cfg['level']
        
        if data_type == 'ERA5':
            if param == 'wspd':
                values = [compute_aggregation_wspd_ERA5(param, dt, lat, lon, rad, level, 
                         data_type, km, agg_type) 
                        for dt, lat, lon, rad in zip(datetimes, lats, lons, rads)]
            else:
                values = [compute_aggregation_value_ERA5(param, dt, lat, lon, rad, level, 
                         data_type, km, agg_type) 
                        for dt, lat, lon, rad in zip(datetimes, lats, lons, rads)]
        else:
            values = [get_mean_value(df, i, level, param, agg_type, data_type, km) 
                     for i in range(len(df))]
        
        results[f"{param}_{agg_type}"] = values
    
    for col, values in results.items():
        df[col] = values
    return df

def get_mean_value(CS, i, level, param, agg='median', data_type='LoRes', km=77):
    hw = int(CS[f'rad'][i])
    
    y = CS[f'y'][i]
    x = CS[f'x'][i]

    y_int, x_int = int(y), int(x)
    y_in, x_in = max(y_int - hw, 0), max(x_int - hw, 0)
    
    y_out = -1 if y_int + hw >= len(ds0.south_north) else y_int + hw
    x_out = -1 if x_int + hw >= len(ds0.west_east) else x_int + hw

    if data_type in ['HiRes', 'LoRes']:
        if param == 'wspd':
            func = compute_aggregation_wspd
        else:
            # func = compute_aggregation_value
            func = compute_aggregation_value_wrfout
        return func(param, CS['datetime'][i], y_in, y_out, x_in, x_out, 
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

def compute_aggregation_value_wrfout(param, t, y_in, y_out, x_in, x_out, our_level, data_type='LoRes', km=77, agg='min'):
    
    ds = load_wrf_file(t.year, t.month, t.day)
    hour_idx = int((t.hour/3))
    ##### [0] check

    try:
        # Особые случаи обработки
        if param == 'wspd':
            # Вычисляем скорость ветра из компонент U и V
            u = wrf.getvar(ds, 'U10', timeidx=hour_idx)  # U at 10m
            v = wrf.getvar(ds, 'V10', timeidx=hour_idx)  # V at 10m
            data = np.sqrt(u**2 + v**2)
            
        elif param in ['LH', 'HFX', 'SST']:
            # Прямой доступ к переменным через NetCDF
            if param in ds.variables:
                data = ds.variables[param][hour_idx]
                # Если данные 4D (Time, bottom_top, south_north, west_east), берем первый уровень
                if len(data.shape) == 4:
                    data = data[0]  # Берем первый вертикальный уровень
            else:
                print(f"Variable '{param}' not found in WRF file")
                return np.nan
                
        elif param == 'cape_2d':
            data = wrf.getvar(ds, param, timeidx=hour_idx)[0]
        elif param == 'cape_3d':
            data = wrf.getvar(ds, param, timeidx=hour_idx)[0, our_level]
        else:
            # Стандартная обработка через wrf.getvar
            if our_level == 0:
                data = wrf.getvar(ds, param, timeidx=hour_idx)
            else:
                data = wrf.getvar(ds, param, timeidx=hour_idx)[our_level]

    
        data_slice = slice(y_in, y_out), slice(x_in, x_out)
        data_cropped = data[data_slice]
    
        # Замена NaN на 0 только для cape_2d и cape_3d
        if param in ['cape_2d', 'cape_3d']:
            data_cropped = np.nan_to_num(data_cropped, nan=0.0)
    
        if agg == 'delta':
            result = np.nanpercentile(data_cropped, 95) - np.nanpercentile(data_cropped, 5)
        else:
            result = agg_funcs.get(agg.lower(), np.nanmedian)(data_cropped)

    except (ValueError, KeyError, AttributeError) as e:
        print(f"Error extracting {param}: {str(e)}")
        return np.nan
        
    return result


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
        ds = get_cached_dataset(ncfile)
        
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
    ds = get_cached_dataset(ncfile)
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
        ds = get_cached_dataset(ncfile)
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
        ds = get_cached_dataset(ncfile)
        ds['wspd'] = np.sqrt(ds['u10']**2 + ds['v10']**2)
        
        data = ds['wspd'].sel(time=t, 
                             latitude=slice(lat + hw, lat - hw), 
                             longitude=slice(lon_min, lon_max))
        
        return agg_funcs.get(agg.lower(), np.nanmedian)(data)
    except Exception as e:
        print(f"Error processing ERA5 wind speed at {t}: {str(e)}")
        return np.nan