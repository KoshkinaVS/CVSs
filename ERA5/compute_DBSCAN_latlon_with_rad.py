import numpy as np
import xarray as xr
from tqdm import tqdm
import time
import os
import sys
from pathlib import Path
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import glob

# Инициализация путей и параметров
path_init = f'/storage/thalassa/users/vkoshkina'
# folder = 'scripts/CS_processing/CVS_identification/'
# sys.path.insert(2, f'{path_init}/{folder}')

# from compute_DBSCAN_func_new import *

from func_for_DBSCAN_opt import *

# Ignore warnings
warnings.filterwarnings("ignore")

# Configuration
with open('config_ERA5_02-04-10.json', 'r') as file:
    config = json.load(file)

# Parameters
data_type = 'ERA5'
years = np.arange(1979, 2027)
years = np.arange(1979, 2010)
# years = np.arange(2010, 2027)


params = ['u', 'v']
name_crit = 'R2D'
smooth = True
sigma = 2

region = 'Arctic'
# region = 'NA'


if smooth:
    print(f'{data_type} for {region} with sigma={sigma}')
else:
    print(f'{data_type} for {region} without smoothing')

path_dir_data = f'/storage/thalassa/users/vkoshkina/data'


if region == 'Arctic':
    # Арктика
    level_hPa = 850
    region_name = f'Arctic_{level_hPa}hPa'
else:
    # Атлантика
    level_hPa = 500
    level_hPa = 850
    
    region_name = f'NA_for_TC_{level_hPa}hPa'


if smooth:
    rortex_path = f'{path_dir_data}/{data_type}/{name_crit}_{data_type}_{region_name}_sigma_{sigma}/daily'
    output_folder = f'{path_dir_data}/{data_type}/DBSCAN_{config["eps"]:02d}-{config["min_samples"]:02d}-{config["size_filter"]:02d}_{region_name}_sigma_{sigma}_rad/'
else:
    rortex_path = f'{path_dir_data}/{data_type}/{name_crit}_{data_type}_{region_name}/daily'
    output_folder = f'{path_dir_data}/{data_type}/DBSCAN_{config["eps"]:02d}-{config["min_samples"]:02d}-{config["size_filter"]:02d}_{region_name}_rad/'



os.makedirs(output_folder, exist_ok=True)

# Config variables
dist_m = config["dist_m"]
time_name = 'time'
level_name = 'level'
x_name = 'longitude'
y_name = 'latitude'

def change_time(ds, ds_raw):
    """Replace time coordinate with datetime64 values"""
    new_time = xr.DataArray(
        data=ds_raw[time_name].values,
        dims=['Time'],
        name='Time',
        attrs={'description': 'Time coordinate'}
    )
    
    ds = ds.rename_dims({'time': 'Time'})
    ds = ds.rename_vars({'time': 'Time'})
    
    if 'time' in ds.coords:
        ds = ds.drop('time')
    
    ds = ds.assign_coords(Time=new_time)
    return ds

def convert_time_units(ds, reference_time):
    """Convert time units to hours since reference"""
    ds['Time'].encoding['units'] = f"hours since {reference_time}"
    ds['Time'].attrs['description'] = f"hours since {reference_time}"
    return ds

def add_latlon(ds, ds_raw):
    """Add latitude and longitude coordinates"""
    ds = ds.assign_coords({
        level_name: ds_raw[level_name],
        x_name: ds_raw[x_name],
        y_name: ds_raw[y_name]
    })
    return ds


def add_r2d(ds, ds_raw):
    """Add normalized R2D field by level"""
    # r2d = ds_raw[name_crit]

    r2d = np.where(ds['cluster'] != 0, ds_raw[name_crit], np.nan)

    ds[name_crit] = ((time_name, level_name, y_name, x_name), r2d)
    ds[name_crit].attrs.update({
        'description': 'Rortex criterion 2D (neg - AC, pos - C)',
        'long_name': 'Rortex 2D without noise'
    })

    wspd = np.sqrt(ds_raw['u']*ds_raw['u']+ds_raw['v']*ds_raw['v'])
    ds['wspd'] = ((time_name, level_name, y_name, x_name), wspd.values.astype(np.float32))
    ds['wspd'].attrs.update({
        'description': f'Wind speed at {level_hPa} hPa, m/s',
        'long_name': 'Wind speed'
    })

    return ds


def calculate_radius_for_clusters(cluster_da, center_cluster_da, dlon_deg=0.25, dlat_deg=0.125):
    """
    Рассчитывает эффективный радиус для каждого кластера вихря
    Использует xarray groupby для обработки временных шагов
    """
    R_earth = 6371.0
    
    # Площадь ячейки с учетом широты
    lats = cluster_da.latitude.values
    lons = cluster_da.longitude.values
    
    dlon_rad = np.radians(dlon_deg)
    dlat_rad = np.radians(dlat_deg)
    
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    lat_rad = np.radians(lat_grid)
    areas_km2 = R_earth**2 * dlon_rad * dlat_rad * np.cos(lat_rad)
    
    # Создаем DataArray с площадями
    area_da = xr.DataArray(
        areas_km2,
        coords={'latitude': cluster_da.latitude, 'longitude': cluster_da.longitude},
        dims=['latitude', 'longitude']
    )
    
    # Проверяем размерности
    if time_name in cluster_da.dims:
        # Обрабатываем каждый временной шаг через groupby
        rad_list = []
        time_coords = []
        
        for t_idx in range(len(cluster_da[time_name])):
            cluster_t = cluster_da.isel({time_name: t_idx})
            center_t = center_cluster_da.isel({time_name: t_idx})
            
            # Создаем массив радиусов для этого временного шага
            rad_t = xr.full_like(center_t, np.nan, dtype=np.float32)
            
            # Получаем уникальные кластеры (без фона)
            cluster_ids = np.unique(cluster_t.values)
            cluster_ids = cluster_ids[cluster_ids != 0]
            
            for cid in cluster_ids:
                # Маска кластера
                mask = cluster_t == cid
                if mask.sum() > 0:
                    # Площадь кластера
                    area = (area_da * mask).sum().item()
                    radius = np.sqrt(area / np.pi) if area > 0 else np.nan
                    
                    # Заполняем центры
                    rad_t = rad_t.where(center_t != cid, radius)
            
            rad_list.append(rad_t)
        
        # Объединяем по времени
        rad_da = xr.concat(rad_list, dim=time_name)
        
    else:
        # 2D случай
        rad_da = xr.full_like(center_cluster_da, np.nan, dtype=np.float32)
        cluster_ids = np.unique(cluster_da.values)
        cluster_ids = cluster_ids[cluster_ids != 0]
        
        for cid in cluster_ids:
            mask = cluster_da == cid
            if mask.sum() > 0:
                area = (area_da * mask).sum().item()
                radius = np.sqrt(area / np.pi) if area > 0 else np.nan
                rad_da = rad_da.where(center_cluster_da != cid, radius)
    
    return rad_da
    
# Основной цикл
for year in tqdm(years, desc="Years"):
    for month in tqdm(np.arange(1, 13, 1), desc="Months", leave=False):
        start_time = time.time()
        
        pattern = f'{rortex_path}/{year}/{name_crit}_{year}-{month:02d}-*.nc'
        if smooth:
            pattern = f'{rortex_path}/{year}/sigma_{sigma}_{name_crit}_{year}-{month:02d}-*.nc'
           
        
        all_input_files = glob.glob(pattern)
    
        all_input_files = sorted(list(set(all_input_files)))  # Remove duplicates, sort
        print(f"Found {len(all_input_files)} daily files to process")
        
        for ncfile in all_input_files:
            ds_smooth = xr.open_dataset(ncfile)
            day = ds_smooth.time.dt.day.values[0]
            
            output_file = f'{output_folder}/sigma_{sigma}_DBSCAN_{data_type}_{year}-{month:02d}-{day:02d}.nc'
            if os.path.exists(output_file):
                print(f'{year}-{month:02d}-{day:02d} уже обработан, пропускаем')
                continue
            
            
            cluster_ds = get_DBSCAN_ds(ds_smooth, config, our_level=0, data_type=data_type)

            
            
            # ✅ Use external functions
            cluster_ds = add_latlon(cluster_ds, ds_smooth)          # adds x_name, y_name


            # #### 2026-08-12 add radius in km correct in lat
            # rad_km = calculate_radius_for_clusters(cluster_ds['cluster'][:,0], cluster_ds['center_cluster'][:,0], dlon_deg=0.25, dlat_deg=0.25)
            # #### 2026-08-17 add LOCAL radius in km correct in lat
            # # rad_km = calculate_radius_for_clusters(cluster_ds['cluster'][:,0], cluster_ds['local_extr_cluster'][:,0], dlon_deg=0.25, dlat_deg=0.25)
            
            # rad_km = rad_km.expand_dims(level_name=[level_hPa], axis=1)
            
            # cluster_ds['local_rad_km'] = ({
            #                 time_name: len(cluster_ds[time_name]), 
            #               level_name: len(cluster_ds[level_name]), 
            #               y_name: len(cluster_ds[y_name]), x_name: len(cluster_ds[x_name])}, rad_km.values.astype(np.float32))

            # cluster_ds['local_rad_km'].attrs.update({
            #         'units': 'km',
            #         'long_name': 'Effective local radius of vortex',
            #         'description': f'R_eff = sqrt(Area/π), area calculated with cos(lat) correction'
            #     })
            
            cluster_ds = add_r2d(cluster_ds, ds_smooth)          # adds r2d

            cluster_ds = change_time(cluster_ds, ds_smooth)         # replaces time → Time
            cluster_ds = convert_time_units(cluster_ds, "1970-01-01 00:00:00")
        
        
            try:
                cluster_ds.to_netcdf(output_file, mode='w')
                print(f'Файл {output_file} успешно сохранен')
            except Exception as e:
                print(f'Ошибка при сохранении файла {output_file}: {str(e)}')

            
            
            # Очищаем
            del ds_smooth, cluster_ds
        
        
        elapsed = (time.time() - start_time) / 60
        print(f'Месяц {year}-{month:02d} обработан за {elapsed:.2f} минут')