import numpy as np
import xarray as xr
from tqdm import tqdm
import time
from scipy.ndimage import gaussian_filter
import os
import sys
from pathlib import Path
import json
import warnings

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
from functools import partial

# Инициализация путей и параметров
path_init = f'/storage/thalassa/users/vkoshkina'
folder = 'scripts/CS_processing/CVS_identification/'
sys.path.insert(2, f'{path_init}/{folder}')

from compute_DBSCAN_func import *

# from func_for_DBSCAN_opt import *


# Ignore warnings
warnings.filterwarnings("ignore")

# Configuration
with open('config_ERA5.json', 'r') as file:
    config = json.load(file)

# Parameters
data_type = 'ERA5'
years = np.arange(1984, 2025)
years = np.arange(1979, 2025)

level = 500
params = ['u', 'v']
name_crit = 'R2D'
smooth = False
# smooth = True

sigma = 2
# sigma = 4


if smooth:
    print(f'{data_type} at {level} hPa with sigma={sigma}')
else:
    print(f'{data_type} at {level} hPa without smoothing')


path_dir_data = f'/storage/thalassa/users/vkoshkina/data'
rortex_path = f'{path_dir_data}/{data_type}/{name_crit}_{data_type}_level_{level}/'
output_folder = f'{path_dir_data}/{data_type}/DBSCAN_{config["eps"]:02d}-{config["min_samples"]:02d}-{config["size_filter"]:02d}/'

if smooth:
    rortex_path = f'{path_dir_data}/{data_type}/{name_crit}_{data_type}_level_{level}_sigma_{sigma}/'
    output_folder = f'{path_dir_data}/{data_type}/DBSCAN_{config["eps"]:02d}-{config["min_samples"]:02d}-{config["size_filter"]:02d}_sigma_{sigma}/'
    

os.makedirs(rortex_path, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# Config variables
dist_m = config["dist_m"]
time_name = config["time_name"]
level_name = config["level_name"]
x_name = config["x_name"]
y_name = config["y_name"]
time_unit = config["time_unit"]

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
        x_name: ds_raw[x_name],
        y_name: ds_raw[y_name]
    })
    return ds

def add_r2d(ds, ds_raw):
    """Add normalized R2D field"""
    r2d = ds_raw[name_crit]
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

def process_month(year_month):
    """Process a single month of data"""
    year, month = year_month
    try:
        print(f'\nProcessing {year}-{month:02d}')

        
        r2d_name = f'{name_crit}_{data_type}_level_{level}_{year}-{month:02d}.nc'
        output_name = f'DBSCAN_{data_type}_level_{level}_{year}-{month:02d}.nc'
        
        # Input filename
        if smooth:
            r2d_name = f'sigma_{sigma}_{r2d_name}'
            output_name = f'sigma_{sigma}_DBSCAN_{output_name}'
            
        input_file = f'{rortex_path}/{r2d_name}'
        output_path = f'{output_folder}/{output_name}'
        
        # Check if output file already exists
        if os.path.exists(output_path):
            return f"Output already exists: {output_path} - skipping"
        
        if not os.path.exists(input_file):
            return f"File not found: {input_file}"

        # Load dataset
        ds = xr.open_dataset(input_file)
        
        # Process DBSCAN clustering
        cluster_ds = get_DBSCAN_ds(ds, config, our_level=0)
        
        # Add metadata and coordinates
        cluster_ds = add_latlon(cluster_ds, ds)
        cluster_ds = add_r2d(cluster_ds, ds)
        
        cluster_ds = change_time(cluster_ds, ds)
        cluster_ds = convert_time_units(cluster_ds, "1970-01-01 00:00:00")

        # Save results

        
        encoding = {var: {'zlib': True, 'complevel': 4} for var in cluster_ds.data_vars}
        cluster_ds.to_netcdf(output_path, encoding=encoding)
        
        return f"Success: {year}-{month:02d}"
    except Exception as e:
        return f"Error processing {year}-{month:02d}: {str(e)}"

def main():
    start_time = time.time()
    year_months = [(year, month) for year in years for month in range(1, 13)]
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=max(1, cpu_count() - 1)) as executor:
        futures = {executor.submit(process_month, ym): ym for ym in year_months}
        
        for future in tqdm(as_completed(futures), total=len(year_months), desc="Processing months"):
            result = future.result()
            if result.startswith("Error"):
                print(result)
    
    elapsed = (time.time() - start_time) / 60
    print(f'Total processing time: {elapsed:.1f} minutes')

if __name__ == '__main__':
    main()