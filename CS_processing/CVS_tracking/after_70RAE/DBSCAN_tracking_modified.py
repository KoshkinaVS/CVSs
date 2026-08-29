import sys
import os
import numpy as np
import xarray as xr
from tqdm import tqdm



# Инициализация путей и параметров
path_init = f'/storage/thalassa/users/vkoshkina'
folder = 'scripts/CS_processing/CVS_tracking/after_70RAE/'
sys.path.insert(2, f'{path_init}/{folder}')

from step_of_tracking import *

path_dir_data = f'{path_init}/data/LoRes'
folder_name = f'{data_type}/DBSCAN_02-04-10_with_uv_time_smoothing'
name_init = f'sigma_2_DBSCAN_{data_type}_level_{level}'

years = np.arange(1979, 2019)
months = np.arange(1, 13)

    
# Параметры для разных вариантов трекинга
tracking_configs = [
    {'type': 'tracking_local_global', 'CVS_speed': 'adv_speed'},
    {'type': 'tracking_local_2_phase', 'CVS_speed': 'no_speed'},
    {'type': 'tracking_global_only', 'CVS_speed': 'bg_speed'}
]

initialize_tracking_methods()


# Результаты для каждого метода
results = {}

for config in tracking_configs:
    tracking_type = config['type']
    CVS_speed = config['CVS_speed']
    
    print(f"\nRunning {tracking_type} with speed {CVS_speed}")
    
    # Получаем функцию трекинга с настройками
    tracking_func = get_tracking_function(tracking_type=tracking_type, CVS_speed=CVS_speed)

    
    very_first = True
    CS_tracks_list = []
    cluster_idx = 0
    results_dir = f'{tracking_type}_{CVS_speed}_crit_sorted'
    
    # Основной цикл обработки
    for year in years:
        print(f'year: {year}')
        
        for month in months:
            path_data_dir = f'{path_dir_data}/{data_type}/{data_type}_tracks/{results_dir}/DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_level_{level}_smoothing/tracks_{circ}/{year}-{month:02d}/'
            
            if not os.path.exists(f'{path_data_dir}'):
                os.makedirs(f'{path_data_dir}')
            
            ds = xr.open_dataset(f"{path_dir_data}/{folder_name}/{name_init}_{year}-{month:02d}.nc")
            dt_step = np.timedelta64(ds[time_name].values[1] - ds[time_name].values[0], 's').astype(int)
                  
            if very_first:
                t_start = 1
                very_first = False
                CS_tracks_list, clstr_len = initialize_tracks(ds, tracking_type, data_type, CS_tracks_list, time_name, circ)
            else:
                t_start = 0
                
            # Основной цикл по временным шагам с использованием выбранной функции трекинга
            for t in tqdm(range(t_start, len(ds[time_unit]))):
                cluster_idx, CS_tracks_list, clstr_len = tracking_func(
                    cluster_idx=cluster_idx,
                    CS_tracks_list=CS_tracks_list,
                    clstr_len=clstr_len,
                    ds=ds,
                    data_type=data_type,
                    path_data_dir=path_data_dir,
                    our_time=t,
                    speed_level=speed_level, 
                    CVS_speed=CVS_speed,
                    circ=circ,
                    dt_step=dt_step
                )
    
        # Сохранение завершенных треков
        for TC in tqdm(CS_tracks_list):
            if np.sum(~np.isnan(TC['t'])) >= 3:
                cluster_idx = save_track_csv(cluster_idx, TC, path_data_dir)