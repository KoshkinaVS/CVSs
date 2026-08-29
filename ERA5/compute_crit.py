import pandas as pd 
import numpy as np
import math
import xarray as xr
from numpy import linalg as LA

import time

from tqdm import tqdm

import datetime
from datetime import timedelta

from tqdm import tqdm

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/storage/kubrick/vkoshkina/scripts/vortex_identification')

from vortex_dir.load_data import *
from vortex_dir.compute_criteria import *

# param = 5 # indent from the horizontal boundaries of the domain

path_dir = "/storage/kubrick/gavr/Koshkina/data/era5"

season = 'SUMMER'
# name = 'ERA5_criteria'
init_name = f'ERA5_2010_{season}toNAAD'


# year = int(input('Enter year: '))

# month = int(input('Enter month number: '))

# day = int(input('Enter start day: '))

# period = int(input('Enter period (in days): '))

step = int(input('Enter time step (in days) (step=-1 for for continuous sequence): '))

name = input('Alias for final file: ')

# season = get_season(year, month, day)

start = time.time() ## точка отсчета времени

if step == -1:
    ds, dist_m = open_step_dataset_nc(path_dir, init_name, step=1) # подряд
else:
    ds, dist_m = open_step_dataset_NAAD(path_dir, year, month, day, step=step, param=param, period=period, crit='HiRes') # период с шагом

print(f'ds.dims: {ds.dims}')

g = 9.80665
# шаг по вертикали в Pa
# dz = np.abs(np.gradient(ds.z, 1., axis = [1]))/g 

# grad_tensor = compute_grad_tensor(ds['u'], ds['v'], ds['w'], dist_m, dz)

dist_m = 13897.18

grad_tensor = compute_grad_tensor_2d(ds['U'], ds['V'], dist_m)


############# расчет 3 базовых критериев ###################

S, A = compute_S_A_2d(grad_tensor)

Q = compute_Q(S, A)

ds['Q'] = ({'time': len(ds.time), 
                  'level': len(ds.plev), 
                  'south_north': len(ds.south_north), 'west_east': len(ds.west_east)}, Q)

delta = compute_delta(grad_tensor, S, A, case='2d')

ds['delta'] = ({'time': len(ds.time), 
                  'level': len(ds.plev), 
                  'south_north': len(ds.south_north), 'west_east': len(ds.west_east)}, delta)

# lambda2 = compute_lambda2(S, A)

# ds['lambda2'] = ({'time': len(ds.time), 
#                   'level': len(ds.plev), 
#                   'south_north': len(ds.south_north), 'west_east': len(ds.west_east)}, lambda2)

##################################



############# расчет swirling_strength и rortex ###################

omega_z = compute_omega_2d(ds['U'], ds['V'], dist_m)

# sw_str, sw_vec_reoredered = compute_swirling_strength(grad_tensor)

# ds['sw_str'] = ({'time': len(ds.time), 
#                   'level': len(ds.level), 
#                   'south_north': len(ds.south_north), 'west_east': len(ds.west_east)}, sw_str)

# R = compute_rortex(sw_str, sw_vec_reoredered, omega)

# ds['R'] = ({'time': len(ds.time), 
#                   'level': len(ds.level), 
#                   'south_north': len(ds.south_north), 'west_east': len(ds.west_east)}, R)

################ 2d расчет swirling_strength и rortex ###################
sw_str_2d = compute_swirling_strength_2d(grad_tensor)

ds['sw_str_2d'] = ({'time': len(ds.time), 
                  'level': len(ds.plev), 
                  'south_north': len(ds.south_north), 'west_east': len(ds.west_east)}, sw_str_2d)

R_2d = compute_rortex_2d(sw_str_2d, omega_z)

ds['R_2d'] = ({'time': len(ds.time), 
                  'level': len(ds.plev), 
                  'south_north': len(ds.south_north), 'west_east': len(ds.west_east)}, R_2d)

path_dir = '/storage/kubrick/vkoshkina/data'
# собираем в файлик
ds.to_netcdf(f'{path_dir}/ERA5/{name}_criteria.nc', mode='w')

end = time.time() - start ## собственно время работы программы

print(f'{end/60} min') ## вывод времени
