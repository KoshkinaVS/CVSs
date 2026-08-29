from matplotlib import pyplot as plt
import pandas as pd 
import numpy as np
import math
import xarray as xr
from numpy import linalg as LA

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import sys
import os
import json

import cmaps


path_init = f'/storage/thalassa/users/vkoshkina/data'
path_dir_data = f'{path_init}/ERA5/'

path = '/storage/thalassa/DATA/ERA5/PL/grib/invariants'

ncfile = f'{path}/_e5.oper.invariant.128_172_lsm.ll025sc.1979010100_1979010100.grb'
ds = xr.open_dataset(ncfile)

print(ds)

# Проверяем, нужна ли конвертация
if ds.longitude.max() > 180:
    # Создаем новую долготу
    lon_new = xr.where(ds.longitude > 180, ds.longitude - 360, ds.longitude)
    ds = ds.assign_coords(longitude=lon_new)
    
    # Важно: сортируем по новой долготе, чтобы данные шли от -180 до 180
    ds = ds.sortby('longitude')

ds = ds.sel(latitude=slice(71, 0), longitude=slice(-110, 15))
    
    
# Удаляем ненужные координаты
ds_clean = ds.drop_vars(['number', 'time', 'step', 'surface', 'valid_time'])

ds_renamed = ds_clean.rename({
    'latitude': 'lat', 
    'longitude': 'lon',
    # можно добавить другие переименования если нужно
})

# Сохраните результат (опционально)
ds_renamed.to_netcdf(f'{path_dir_data}/ERA5_lsm_cropped_NA_for_TC.nc')