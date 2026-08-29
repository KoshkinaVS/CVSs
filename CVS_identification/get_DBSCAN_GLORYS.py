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


from func_for_DBSCAN_opt import *

# Ignore warnings
warnings.filterwarnings("ignore")



# Parameters
data_type = 'GLORYS'
# data_type = 'ALT'

years = [2023]


# Configuration
with open(f'config_{data_type}.json', 'r') as file:
    config = json.load(file)


params = ['uo', 'vo']
params = ['ugos', 'vgos']

name_crit = 'R2D'
smooth = False
sigma = 0

region = 'Arctic'
region = 'NA'
region = 'BarKara'



if smooth:
    print(f'{data_type} for {region} with sigma={sigma}')
else:
    print(f'{data_type} for {region} without smoothing')

path_dir_data = f'/storage/thalassa/users/vkoshkina/data'


if region == 'Arctic':
    # Арктика
    level_hPa = 850
    region_name = f'Arctic_{level_hPa}hPa'
if region == 'BarKara':
    level = 8
    # level = 15
    # level = 0
    
    region_name = f'BarKara_level_{level}'
else:
    # Атлантика
    level_hPa = 500
    level_hPa = 850
    
    region_name = f'NA_for_TC_{level_hPa}hPa'


rortex_path = f'{path_dir_data}/ocean_eddies/'
output_folder = f'{path_dir_data}/ocean_eddies/DBSCAN_{config["eps"]:02d}-{config["min_samples"]:02d}-{config["size_filter"]:02d}_sigma_{sigma}/'




os.makedirs(output_folder, exist_ok=True)

# Config variables
dist_m = config["dist_m"]
time_name = config["time_name"]
level_name = config["level_name"]
x_name = config["x_name"]
y_name = config["y_name"]

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
    r2d = ds_raw[name_crit]

    ds[name_crit] = ((time_name, level_name, y_name, x_name), r2d.values)
    ds[name_crit].attrs.update({
        'description': 'Rortex criterion 2D (neg - AC, pos - C)',
        'long_name': 'Rortex 2D'
    })
    return ds



# Основной цикл
for year in tqdm(years, desc="Years"):
    start_time = time.time()
    
    pattern = f'{rortex_path}/sigma_{sigma}_{name_crit}_{data_type}.nc'

    all_input_files = glob.glob(pattern)

    all_input_files = sorted(list(set(all_input_files)))  # Remove duplicates, sort
    print(f"Found {len(all_input_files)} files to process")
    
    for ncfile in all_input_files:
        ds_smooth = xr.open_dataset(ncfile)
        if data_type != 'ALT':
            ds_smooth = ds_smooth.isel(depth=slice(level, level+1))
        print(ds_smooth.dims)


        output_file = f'{output_folder}/sigma_{sigma}_DBSCAN_{data_type}_level_{level}_{year}.nc'
        if os.path.exists(output_file):
            print(f'{year} уже обработан, пропускаем')
            continue
        
        
        cluster_ds = get_DBSCAN_ds(ds_smooth, config, our_level=0, data_type=data_type)
        
        
        # ✅ Use external functions
        cluster_ds = add_latlon(cluster_ds, ds_smooth)          # adds x_name, y_name
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
        

        # for month in np.unique(ds_smooth.time.dt.month.values):
        #     ds_month = ds_smooth.sel(time=ds_smooth.time.dt.month == month)
        #     # Получаем дни в этом месяце
        #     days_in_month = np.unique(ds_month.time.dt.day.values)
        
        
        #     for day in tqdm(days_in_month, desc=f"Days in Month {month}", leave=False):
        #         ds_day = ds_month.sel(time=ds_month.time.dt.day == day)
            
        #         day = ds_day.time.dt.day.values[0]
                
        #         output_file = f'{output_folder}/sigma_{sigma}_DBSCAN_{data_type}_{year}-{month:02d}-{day:02d}.nc'
        #         if os.path.exists(output_file):
        #             print(f'{year}-{month:02d}-{day:02d} уже обработан, пропускаем')
        #             continue
                
                
        #         cluster_ds = get_DBSCAN_ds(ds_day, config, our_level=0, data_type=data_type)
                
                
        #         # ✅ Use external functions
        #         cluster_ds = add_latlon(cluster_ds, ds_smooth)          # adds x_name, y_name
        #         cluster_ds = change_time(cluster_ds, ds_smooth)         # replaces time → Time
        #         cluster_ds = convert_time_units(cluster_ds, "1970-01-01 00:00:00")
        #         cluster_ds = add_r2d(cluster_ds, ds_smooth)          # adds x_name, y_name
            
            
        #         try:
        #             cluster_ds.to_netcdf(output_file, mode='w')
        #             print(f'Файл {output_file} успешно сохранен')
        #         except Exception as e:
        #             print(f'Ошибка при сохранении файла {output_file}: {str(e)}')
                
                
        #         # Очищаем
        #         del ds_day, cluster_ds
    
    
    elapsed = (time.time() - start_time) / 60
    print(f'Год {year} обработан за {elapsed:.2f} минут')