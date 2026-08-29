import sys
import os
import numpy as np
import xarray as xr
from tqdm import tqdm
import multiprocessing
from pathos.multiprocessing import ProcessingPool  # Альтернатива для лучшей сериализации
from itertools import product

from multiprocessing import Pool, Manager
from functools import partial

import shutil

# # Инициализация путей и параметров
# path_init = f'/storage/thalassa/users/vkoshkina'
# folder = 'scripts/CS_processing/CVS_tracking/after_70RAE/'
# sys.path.insert(2, f'{path_init}/{folder}')

from step_of_tracking import *

path_dir_data = f'{path_init}/data'
if data_type == 'LoRes':
    path_dir_data = f'{path_dir_data}/LoRes'

# if data_type == 'ERA5':
#     level = 500
    
folder_name = f'{data_type}/DBSCAN_02-04-10_sigma_{sigma}'
name_init = f'sigma_{sigma}_DBSCAN_DBSCAN_{data_type}_level_{level}'

if sigma == 0:
    folder_name = f'{data_type}/DBSCAN_02-04-10'
    name_init = f'DBSCAN_{data_type}_level_{level}'

years = np.arange(1979, 2025)
# years = np.arange(2010, 2011)
# years = np.arange(1979, 1980)
# years = np.arange(1984, 2015)


months = np.arange(1, 13)
# months = np.arange(8, 10)



# Отдельные списки параметров
tracking_types = ['tracking_local_2_phase', 
                  # 'tracking_global_only', 
                  # 'tracking_local_global'
                 ]
speed_options = ['adv_speed', 'no_speed', 
                 # 'bg_speed', 'adv_bg_speed'
                ]

pref_tracking = 'all_points_bound'

# Генерация всех комбинаций
tracking_configs = [
    {'type': t, 'CVS_speed': s} 
    for t, s in product(tracking_types, speed_options)
]

initialize_tracking_methods()


from tqdm import tqdm
from multiprocessing import Pool, Manager
from functools import partial
import os
import numpy as np

def process_tracking_config(config):
    """Основная функция обработки с прогресс-баром для каждого конфига"""
    tracking_type = config['type']
    CVS_speed = config['CVS_speed']

    print(f"\nRunning {tracking_type} with speed {CVS_speed}")

    # Получаем функцию трекинга с настройками
    tracking_func = get_tracking_function(tracking_type=tracking_type, CVS_speed=CVS_speed)
    
    # Инициализация прогресс-бара для этого конфига
    pbar_desc = f"{tracking_type} ({CVS_speed})"
    year_months = [(y, m) for y in years for m in months]
    pbar = tqdm(year_months, desc=pbar_desc, position=os.getpid() % 10, leave=False)
    
    very_first = True
    CS_tracks_list = []
    cluster_idx = 0

    #### FOLDER NAME ####
    results_dir = f"{tracking_type}_{CVS_speed}_{pref_tracking}"

    path_data_dir_main = f"{path_dir_data}/TC_tracks/{data_type}/my_tracking_results/{pref_tracking}/{results_dir}/{data_type}_tracks_sigma_{sigma}/tracks_{circ}"

    if os.path.exists(path_data_dir_main):
        try:
            print(f'Delete {path_data_dir_main}')
            shutil.rmtree(path_data_dir_main)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))        
    
    for year, month in pbar:
        # path_data_dir = f"{path_dir_data}/{data_type}/{data_type}_tracks/{pref_tracking}/{results_dir}/DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_level_{level}_smoothing_sigma_{sigma}/tracks_{circ}/{year}-{month:02d}/"
        
        path_data_dir = f"{path_data_dir_main}/{year}-{month:02d}/"
        
        # Создаём папку заново
        try:
            os.makedirs(path_data_dir, exist_ok=True)  # exist_ok=True для избежания ошибки если папка уже существует
        except Exception as e:
            print(f'Не удалось создать папку {path_data_dir}. Причина: {e}')
            raise
        
        ds = xr.open_dataset(f"{path_dir_data}/{folder_name}/{name_init}_{year}-{month:02d}.nc")
              
        if very_first:
            t_start = 1
            very_first = False
            CS_tracks_list, clstr_len = initialize_tracks(ds, tracking_type, data_type, CS_tracks_list, time_name, circ)
        else:
            t_start = 0
            
        for t in range(t_start, len(ds[time_unit])):
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
        pbar.set_postfix_str(f"Треков: {cluster_idx}, год: {year}")
            
    pbar.close()
    
    for TC in CS_tracks_list:
        if np.sum(~np.isnan(TC['t'])) >= 3:
            cluster_idx = save_track_csv(cluster_idx, TC, path_data_dir)
    
    return {'config': config, 'tracks': CS_tracks_list}

def parallel_process_tracking():
    """Параллельная обработка с многоуровневым прогресс-баром"""
    # Главный прогресс-бар
    main_pbar = tqdm(total=len(tracking_configs), desc="Все конфигурации", position=0)
    
    def update_progress(_):
        main_pbar.update(1)
    
    with Pool(processes=min(len(tracking_configs), os.cpu_count()-1)) as pool:
        results = []
        # Используем imap_unordered для более плавного прогресса
        for result in pool.imap_unordered(process_tracking_config, tracking_configs):
            results.append(result)
            update_progress(result)
    
    main_pbar.close()
    return {r['config']['type']: r['tracks'] for r in results}


if __name__ == '__main__':
    
    print(f"type of params: {pref_tracking}")
    
    # Запускаем параллельную обработку
    final_results = parallel_process_tracking()
    print("\nAll tracking configurations processed successfully!")