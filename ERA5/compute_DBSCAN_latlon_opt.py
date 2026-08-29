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
with open('config_ERA5.json', 'r') as file:
    config = json.load(file)

# Parameters
data_type = 'ERA5'
years = np.arange(1979, 2025)
# years = np.arange(2010, 2011)

params = ['u', 'v']
name_crit = 'R2D'
smooth = True
sigma = 2

if smooth:
    print(f'{data_type} all levels with sigma={sigma}')
else:
    print(f'{data_type} all levels without smoothing')

path_dir_data = f'/storage/thalassa/users/vkoshkina/data'


if smooth:
    rortex_path = f'{path_dir_data}/{data_type}/{name_crit}_{data_type}_NA_for_TC_500hPa_sigma_{sigma}/daily'
    output_folder = f'{path_dir_data}/{data_type}/DBSCAN_{config["eps"]:02d}-{config["min_samples"]:02d}-{config["size_filter"]:02d}_NA_for_TC_500hPa_sigma_{sigma}/'
else:
    rortex_path = f'{path_dir_data}/{data_type}/{name_crit}_{data_type}_NA_for_TC_500hPa/daily'
    output_folder = f'{path_dir_data}/{data_type}/DBSCAN_{config["eps"]:02d}-{config["min_samples"]:02d}-{config["size_filter"]:02d}_NA_for_TC_500hPa/'


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

def add_r2d_level(ds, ds_raw, level_idx):
    """Add normalized R2D field by level"""
    r2d = ds_raw[name_crit][:,level_idx:level_idx+1]
    r2d_max = r2d.max()
    r2d_normalized = r2d / r2d_max
    r2d_scaled = r2d_normalized * 127
    r2d_int8 = r2d_scaled.astype(np.int8)

    ds[name_crit] = ((time_name, level_name, y_name, x_name), r2d_int8.values)
    ds[name_crit].attrs.update({
        'description': 'Rortex criterion 2D discrete (neg - AC, pos - C)',
        'long_name': 'Rortex 2D discrete'
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

def process_daily_file(input_file):
    """Process a single daily file for ALL levels"""
    try:
        print(f'Processing: {os.path.basename(input_file)}')
        
        input_basename = os.path.basename(input_file)
        
        # Строка, которую нужно заменить
        old_part = f"{name_crit}_{data_type}_"
        new_part = f"DBSCAN_{data_type}_"
        
        # Заменяем первое вхождение (даже если перед ним есть sigma_...)
        if old_part in input_basename:
            output_basename = input_basename.replace(old_part, new_part, 1)  # 1 = только первое вхождение
        else:
            # На всякий случай: если не найдено — оставляем как есть или логируем
            output_basename = f"DBSCAN_{data_type}_" + input_basename
        
        output_path = f'{output_folder}/{output_basename}'
        
        if os.path.exists(output_path):
            return f"Output exists: {output_path} - skipping"
        
        ds = xr.open_dataset(input_file)
        # ds = ds.rename_dims({'time': 'Time'})
        
        
        required_dims = {time_name, level_name, y_name, x_name}
        if not required_dims.issubset(ds.dims):
            ds.close()
            return f"Invalid dims in {input_file}: {set(ds.dims)}"
        
        cluster_datasets = []
        levels = ds[level_name].values
        
        for level_idx, level_val in enumerate(tqdm(levels, total=len(levels))):
            
            # Run DBSCAN with correct level index
            cluster_ds = get_DBSCAN_ds(ds, config, our_level=level_idx, data_type=data_type)
            
            # --- Add R2D field (level-wise normalization) ---
            # cluster_ds = add_r2d_level(cluster_ds, ds, level_idx)

            cluster_datasets.append(cluster_ds)
        
        if not cluster_datasets:
            ds.close()
            return f"No valid levels in {input_basename}"
        
        # Concatenate all levels
        cluster_ds_4d = xr.concat(cluster_datasets, dim=level_name)
        
        # ✅ Use external functions
        cluster_ds_4d = add_latlon(cluster_ds_4d, ds)          # adds x_name, y_name
        cluster_ds_4d = change_time(cluster_ds_4d, ds)         # replaces time → Time
        cluster_ds_4d = convert_time_units(cluster_ds_4d, "1970-01-01 00:00:00")

        cluster_ds_4d = add_r2d(cluster_ds_4d, ds)          # adds x_name, y_name
        
        
        # Save
        encoding = {var: {'zlib': True, 'complevel': 4} for var in cluster_ds_4d.data_vars}
        cluster_ds_4d.to_netcdf(output_path, encoding=encoding)
        
        cluster_ds_4d.close()
        ds.close()
        
        return f"Success: {input_basename} -> {output_basename} ({len(levels)} levels)"
        
    except Exception as e:
        if 'ds' in locals():
            ds.close()
        return f"Error processing {os.path.basename(input_file)}: {str(e)}"

def main():
    start_time = time.time()
    
    # Process ALL daily files in rortex_path
    all_input_files = []
    for year in tqdm(years, desc="Scanning years"):
        for month in range(1, 13):
            pattern = f'{rortex_path}/{year}/{name_crit}_{data_type}_{year}-{month:02d}-*.nc'
            if smooth:
                pattern = f'{rortex_path}/{year}/sigma_{sigma}_{name_crit}_{data_type}_{year}-{month:02d}-*.nc'
            files = glob.glob(pattern)
            all_input_files.extend(files)
    
    all_input_files = sorted(list(set(all_input_files)))  # Remove duplicates, sort
    print(f"Found {len(all_input_files)} daily files to process")
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_daily_file, file): file 
                  for file in all_input_files}
        
        for future in tqdm(as_completed(futures), total=len(all_input_files), desc="Processing files"):
            result = future.result()
            if result.startswith("Error"):
                print(result)
            else:
                print(result)
    
    elapsed = (time.time() - start_time) / 60
    print(f'Total processing time: {elapsed:.1f} minutes')

if __name__ == '__main__':
    main()
