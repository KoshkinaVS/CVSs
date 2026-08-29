# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import json
import re

import sys
import os
import glob


from netCDF4 import Dataset
import wrf as wrf


from compute_DBSCAN_func_new import *
from compute_rortex_func import *

data_type = 'SMP'
# data_type = 'HiRes'


# path_dir = f'/storage/kubrick/SMP6km/' 
# year = 2021
# month = 1

path_dir = f'/storage/buffer/SMP/MODELS/WRF/OUTPUT'
year = 2019
month = 1

months = np.arange(3,13)

level = 10

sigma = 2

with open(f'config_{data_type}.json', 'r') as file:
    config = json.load(file)
    
data = config

# Распаковка данных из JSON файла и присвоение переменным
# path_dir = data["path_dir"]
# data_type = data["data_type"]
dist_m = data["dist_m"]
eps = data["eps"]
min_samples = data["min_samples"]
size_filter = data["size_filter"]
min_dist = data["min_dist"]

name_crit = data["name_crit"]
level_name = data["level_name"]
x_name = data["x_name"]
y_name = data["y_name"]
time_name = data["time_name"]
time_unit = data["time_unit"]


path_dir_data = '/storage/kubrick/vkoshkina/data'
folder = f'{path_dir_data}/{data_type}/DBSCAN_{eps:02d}-{min_samples:02d}-{size_filter:02d}_smoothing/{year}'


params = ['ua', 'va', 'geopotential', 'HGT']



def add_param(ds, wrfnc, param):
        
    u = wrf.getvar(wrfnc, param, timeidx=wrf.ALL_TIMES)[:,level:level+1]

    ds[param] = (('Time', 'bottom_top', 'south_north', 'west_east'), u.values) 
    ds[param].attrs['description'] = u.attrs['description']
    ds[param].attrs['units'] = u.attrs['units']
    
    return ds

# Функция для извлечения номера level из имени файла
def extract_level(filename):
    match = re.search(r'level_(\d+)', filename)  # Ищем "level_" и число после него
    return int(match.group(1)) if match else -1  # Возвращаем число, если найдено

for month in tqdm(months):
    ls = list(sorted(Path(f"{folder}").glob(f'sigma_{sigma}_DBSCAN_{data_type}_level_{level}_{year}-{month:02d}-*')))

    for ii,ifile in tqdm(enumerate(ls)):
    
#     file_list = list(sorted(Path(f'{folder}/').glob(f'sigma_{sigma}_DBSCAN_{data_type}_level_*_{year}-{month:02d}-{day:02d}.nc')))

#     # Сортируем список файлов по номеру level
#     sorted_files = sorted(file_list, key=extract_level)
#     ds_list = [xr.open_dataset(f) for f in sorted_files]
#     ds = xr.concat(ds_list, dim="bottom_top")

        file_name = f'sigma_{sigma}_DBSCAN_{data_type}_level_{level}_{year}-{month:02d}-{ii+1:02d}'
    
        ds = xr.open_dataset(f'{folder}/{file_name}.nc') 
    
        wrfnc = Dataset(f'{path_dir}/{year}/wrfout_d01_{year}-{month:02d}-{ii+1:02d}_00.nc')
    
        hgt = wrf.extract_vars(wrfnc, timeidx=wrf.ALL_TIMES, varnames=params[3])
    
        for param in params[:-1]:
            ds = add_param(ds, wrfnc, param)
    
        ds['HGT'] = (('south_north', 'west_east'), hgt['HGT'][0].values)
        ds['HGT'].attrs['description'] = hgt['HGT'].attrs['description']
        ds['HGT'].attrs['units'] = hgt['HGT'].attrs['units']

    
        ds.to_netcdf(f'{folder}/{file_name}_with_params.nc', mode='w') 
