import pandas as pd 
import numpy as np
import math
import xarray as xr

import datetime
from datetime import timedelta

import scipy as sp

from tqdm import tqdm

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import DBSCAN


from pathlib import Path

import sys
import os

path_init = f'/storage/thalassa/users/vkoshkina/'

# folder = 'after_70RAE/tracking_2025'
folder = 'scripts/ERA5'

sys.path.insert(2, f'{path_init}/{folder}')

from step_of_tracking import *

# from func_for_local_extrema import *
# from func_for_local_only import *
# from func_for_global_only import *
from func_for_local_2_phase import *


tracking_type = 'tracking_local_2_phase'

print(tracking_type)

json_file = f'{path_init}/{folder}/tracking_init.json'

dist_m, our_level, level, x_unit, y_unit, u_unit, v_unit, time_unit, level_unit, crit, th, min_samples, eps, CS_points_th = get_init_params(data_type, json_file, print_info=True)

print('level: ')
level = int(input())


path_data = f'{path_init}/data'  

# DBSCAN_name = f'DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_level_{level}_smoothing'

# folder_name = f'{data_type}/{DBSCAN_name}'
# name_init = f'sigma_2_DBSCAN_{data_type}'

# print('circ: ')
# circ = input()

circ = 'C'


cols = ['t', 'datetime' ,'x','y',
        'lat','lon',
        # 'rad', # if tracking_local_extrema_only -- without rad
        'crit',
        'track_len']

if tracking_type != 'tracking_local_extrema_only':
    cols.append('rad')


pref_tracking = 'all_points_bound'
CVS_speed = 'adv_speed'


results_dir = f"{tracking_type}_{CVS_speed}_{pref_tracking}"

path_data_tracks = f"{path_data}/TC_tracks/{data_type}/my_tracking_results/{pref_tracking}/{results_dir}/{data_type}_tracks_sigma_{sigma}/"


months = np.arange(1,13,1)
days = np.arange(1,32,1)

def compute_param_for_track(t, y_in, y_out, x_in, x_out, our_level, param, value_type='mean'):
        
    if value_type == 'max':
        value = np.nanmax(ds[param][t, our_level, 
                              y_in:y_out, x_in:x_out])
    else:
        value = np.nanmedian(ds[param][t, our_level, 
                              y_in:y_out, x_in:x_out])
    
    return value

def open_ds(ls):
    if len(ls) != 0:
        for ii,ifile in enumerate(ls):

            ds = xr.open_dataset(ifile)
        return ds
    
    
def compute_mean_value(ds, t, y_in, y_out, x_in, x_out, our_level):
    value = np.nanmedian(ds[t, our_level,
                          y_in:y_out, x_in:x_out])
    return value


def compute_max_value(ds, t, y_in, y_out, x_in, x_out, our_level):
    if our_level == 0:
        value = np.nanmax(ds[t,
                          y_in:y_out, x_in:x_out])
    else:
        value = np.nanmax(ds[t, our_level,
                          y_in:y_out, x_in:x_out])
    return value

def compute_max_R_value(ds, t, y_in, y_out, x_in, x_out, our_level):
    value = np.nanmax(np.abs(ds[t, our_level, y_in:y_out, x_in:x_out]))
    
    if circ == 'AC':
        value = -value
 
    return value


def get_max_wspd(ds, t, y_in, y_out, x_in, x_out, our_level=0):

    u = ds['u'][t, our_level,
                          y_in:y_out, x_in:x_out]
    
    v = ds['v'][t, our_level,
                          y_in:y_out, x_in:x_out]
    
    wspd = np.sqrt(u*u + v*v)
    wspd = np.nanmax(wspd)
    
    return wspd


# def get_mean_delta_theta(ds, start_h, y_in, y_out, x_in, x_out):

#     fin_value_up = compute_mean_value(ds['T'], start_h, y_in, y_out, x_in, x_out, our_level=12)+290
#     fin_value_dwn = compute_mean_value(ds['TH2'], start_h, y_in, y_out, x_in, x_out, our_level=0)
#     fin_delta = fin_value_up - fin_value_dwn
    
#     return fin_delta


def get_mean_delta_theta(ds, t, y_in, y_out, x_in, x_out, our_level):

    T_up = ds['T'][t, our_level,
                          y_in:y_out, x_in:x_out] + 290
    
    T_dowm = ds['TH2'][t,
                          y_in:y_out, x_in:x_out]
    
    delta = T_dowm - T_up
    fin_delta = np.nanmedian(delta)
    
    return fin_delta


def get_bounds(ds, x_int, y_int, hw):

#     y_int = int(y+5)
#     x_int = int(x+5) # поправка на обрезание границ
    
#     y_int = int(y)
#     x_int = int(x) # поправка на обрезание границ
    
    y_in = y_int - hw
    x_in = x_int - hw
    
    y_out = y_int + hw + 1
    x_out = x_int + hw + 1
    
    if y_in < 0:
        y_in = 0
    if x_in < 0:
        x_in = 0
            
    if y_out >= len(ds[y_unit]):
        y_out = -1
    if x_out >= len(ds[x_unit]):
        x_out = -1
    return y_in, y_out, x_in, x_out


def add_params(df, path_dir):
    grads = []
    TC_wspd = []
    TC_t = []
    TC_geopot = []
    TC_w = []
    
    # TC_sst = []
    # TC_hf = []
    # TC_lf = []
    
    # TC_th2 = []
    # TC_W = []
    
    lat = []
    lon = []
    
    
    
    max_R = []

    # params = ['mlhf', 'mshf', 'slp', 't2', 'w10', 'sst', 'precip']

    
    # params = ['TH2', 'SST', 'HFX', 'LH', 'W']

    params = ['t', 'w', 'z']
    
    
    
    for i in range(len(df)):
        date = df['datetime'].dt.date[i]
        
        year = df['datetime'].dt.year[i]
        month = df['datetime'].dt.month[i]
        
        
        start_h = int(df.hour_idx.values[i])

        x = int(df.x.values[i])
        y = int(df.y.values[i])

        rad = df.rad.values[i]
        

        if rad == 0:
            hw = 3 
        else:
            hw = int(np.round(df.rad.values[i]))
            

        # name = f'wrfout_d01_{date}*'

        name = f'era5_uvwth_500hPa_{year}-{month:02d}*'

        ls = list(sorted(Path(f"{path_dir}").glob(f'{name}')))
        # print(ls)
        # print(path_dir)
        # print(name)
        
        ds = xr.open_dataset(ls[0])
        
        y_in, y_out, x_in, x_out = get_bounds(ds, x, y, hw)
        
        # grad_T = get_mean_delta_theta(ds, start_h, y_in, y_out, x_in, x_out, level)
        wspd = get_max_wspd(ds, start_h, y_in, y_out, x_in, x_out, our_level=0)
        t = compute_mean_value(ds['t'], start_h, y_in, y_out, x_in, x_out, our_level=0)
        geopot = compute_mean_value(ds['z'], start_h, y_in, y_out, x_in, x_out, our_level=0)
        w = compute_mean_value(ds['w'], start_h, y_in, y_out, x_in, x_out, our_level=0)
        
        
        
        
            
        # th2 = compute_mean_value(ds['TH2'], start_h, y_in, y_out, x_in, x_out, our_level=0)
#         t2 = compute_mean_value(ds['T2'], start_h, y_in, y_out, x_in, x_out, our_level=0)
        # sst = compute_mean_value(ds['SST'], start_h, y_in, y_out, x_in, x_out, our_level=0)
        # heat = compute_mean_value(ds['HFX'], start_h, y_in, y_out, x_in, x_out, our_level=0)
        # latent = compute_mean_value(ds['LH'], start_h, y_in, y_out, x_in, x_out, our_level=0)
        
        # W = compute_mean_value(ds['W'], start_h, y_in, y_out, x_in, x_out, our_level=level)
        
        # lat.append(float(ds.XLAT[start_h,y,x].values))
        # lon.append(float(ds.XLONG[start_h,y,x].values))
        
        
        # grads.append(grad_T)
        TC_wspd.append(wspd)
        TC_geopot.append(geopot)
        TC_t.append(t)
        TC_w.append(w)
        
        
#         TC_t2.append(t2)
        # TC_th2.append(th2)
        
        # TC_sst.append(sst)
        # TC_hf.append(heat)
        # TC_lf.append(latent) 
        
        # TC_W.append(W)  
        
        
        
    # df['dT_med'] = grads
    df['wspd_max'] = TC_wspd
    # df['SST_med'] = TC_sst
    # df['HFX_med'] = TC_hf
    # df['LH_med'] = TC_lf
#     df['T2'] = TC_t2
    # df['TH2_med'] = TC_th2
    df['w_med'] = TC_w
    df['t_med'] = TC_t
    df['geopot_med'] = TC_geopot
    
    
    
    # df['lat'] = lat
    # df['lon'] = lon
    
    
    return df



years = np.arange(1979,2025)

# years = np.arange(2001,2019)

# years = np.arange(2010,2011)

i = 0

# i = 989+1 # idx for 03

# i = 13201+1 # idx for 2001

for year in years:

    print(f'year: {year}')
    
    for month in tqdm(months):
        path_data_tracks_month = f'{path_data_tracks}/tracks_{circ}/{year}-{month:02d}'
        path_init = f'/storage/thalassa/users/vkoshkina/data'
        # path_dir_raw = f'{path_data}/ERA5/ERA5_raw')
        path_dir_raw = f'{path_data}/ERA5/ERA5_raw/ERA5_500'
        
       
        if os.path.exists(path_data_tracks_month):
            ls = list(sorted(Path(f"{path_data_tracks_month}/").glob(f'*_track_*.csv')))
    #        print(ls)
            if len(ls) != 0:
                for ii,ifile in enumerate(ls):
                    df = pd.read_csv(ifile, usecols=cols)
                    df['datetime'] = pd.to_datetime(df['datetime'].values)
                    df['hour_idx'] = (df['datetime'].dt.hour/3).values.astype(int)
                    df = add_params(df, path_dir_raw)
                    if len(df['datetime']) > 3:
                        path_data_new = f'{path_data_tracks}/tracks_{circ}_params/{year}-{month:02d}'
                        
                        if not os.path.exists(f'{path_data_new}'):
                            os.makedirs(f'{path_data_new}')
                        date_start = str(df['datetime'].values[0])[:-16]
                        df.to_csv(f'{path_data_new}/{i:08d}_track_{date_start}.csv')
                        i+=1