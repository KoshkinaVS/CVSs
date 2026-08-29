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


# Инициализация путей и параметров
path_init = f'/storage/thalassa/users/vkoshkina'
folder = 'scripts/CS_processing/CVS_tracking/after_70RAE/'
sys.path.insert(2, f'{path_init}/{folder}')

from step_of_tracking import *

path_dir_data = f'{path_init}/data'
if data_type == 'LoRes':
    path_dir_data = f'{path_dir_data}/LoRes'
    
name_init = f'sigma_{sigma}_DBSCAN_{data_type}_level_{level}'



if data_type != 'SMP':
    level = 12
    folder_name = f'{data_type}/DBSCAN_02-04-10_smoothing_sigma_{sigma}'
    if sigma == 0:
        folder_name = f'{data_type}/DBSCAN_02-04-10_sigma_{sigma}'
    years = np.arange(1979, 2019)
        
else:
    level = 10
    folder_name = f'{data_type}/DBSCAN_02-04-10_with_wspd_smoothing/2019'
    years = np.arange(2019, 2020)



months = np.arange(1, 13)


# Отдельные списки параметров
tracking_types = [
                  'tracking_local_2_phase', 
                  'tracking_global_only', 
                  'tracking_local_global',
                  # 'tracking_local_only',
                  'tracking_local_2_phase_cone',
                 ]

speed_options = [
                'avd_cone',
    
                'adv_speed', 
                'no_speed', 
                 'bg_speed', 'adv_bg_speed'
]

pref_tracking = 'update_2026-05-13'
# pref_tracking = 'update_2026_cone_test'


# Генерация всех комбинаций
tracking_configs = [
    {'type': t, 'CVS_speed': s} 
    for t, s in product(tracking_types, speed_options)
]

initialize_tracking_methods()




def get_available_files(data_dir, file_pattern):
    """Получает список всех доступных файлов в директории"""
    import glob
    import re
    
    # Ищем все файлы, соответствующие паттерну
    files = glob.glob(os.path.join(data_dir, f"{file_pattern}*.nc"))
    
    # Извлекаем даты из имен файлов
    file_dates = []
    for file in files:
        # Пример для monthly: name_2024-01.nc
        # Пример для daily: name_2024-01-15.nc
        filename = os.path.basename(file)
        
        # Ищем дату в формате YYYY-MM или YYYY-MM-DD
        match = re.search(r'(\d{4})-(\d{2})(?:-(\d{2}))?', filename)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3)) if match.group(3) else None
            
            if day:
                file_dates.append((year, month, day, file))
            else:
                file_dates.append((year, month, file))
    
    return sorted(file_dates)

def process_tracking_config_improved(config):
    """Улучшенная функция с автоматическим определением файлов"""
    tracking_type = config['type']
    CVS_speed = config['CVS_speed']
    
    print(f"\nRunning {tracking_type} with speed {CVS_speed}")
    
    # Получаем функцию трекинга
    tracking_func = get_tracking_function(tracking_type=tracking_type, CVS_speed=CVS_speed)
    
    # Автоматически находим все доступные файлы
    data_dir = f"{path_dir_data}/{folder_name}"
    file_pattern = f"{name_init}_"  # Базовый паттерн
    
    # Определяем тип файлов
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.nc') and name_init in f])
    
    if not all_files:
        print(f"No files found in {data_dir}")
        return {'config': config, 'tracks': []}
    
    # Определяем периодичность по первому файлу
    sample_file = all_files[0]
    if len(sample_file.split('-')) >= 3:  # Есть день
        time_format = 'daily'
        time_units = []
        for file in all_files:
            # Извлекаем дату
            parts = file.replace('.nc', '').split('_')
            date_str = parts[-1]  # Предполагаем, что дата в конце
            year, month, day = map(int, date_str.split('-'))
            time_units.append((year, month, day, file))
    else:  # Только месяц
        time_format = 'monthly'
        time_units = []
        for file in all_files:
            parts = file.replace('.nc', '').split('_')
            date_str = parts[-1]
            year, month = map(int, date_str.split('-'))
            time_units.append((year, month, file))
    
    # Инициализация прогресс-бара
    pbar_desc = f"{tracking_type} ({CVS_speed}) - {time_format}"
    pbar = tqdm(time_units, desc=pbar_desc, position=os.getpid() % 10, leave=False)
    
    very_first = True
    CS_tracks_list = []
    cluster_idx = 0
    
    # Создаем директорию для результатов
    results_dir = f"{tracking_type}_{CVS_speed}_{pref_tracking}"
    path_data_dir = f"{path_dir_data}/{data_type}/{data_type}_tracks_sigma_{sigma}/{pref_tracking}/{results_dir}/tracks_{circ}/"
    
    if not os.path.exists(path_data_dir):
        os.makedirs(path_data_dir)
    
    for time_unit_info in pbar:
        if time_format == 'daily':
            year, month, day, filename = time_unit_info
            # Открываем файл
            file_path = os.path.join(data_dir, filename)
            ds = xr.open_dataset(file_path)
            
            # Для ежедневных данных, t_start зависит от первого файла
            if very_first:
                t_start = 1
                very_first = False
                CS_tracks_list, clstr_len = initialize_tracks(ds, tracking_type, data_type, 
                                                              CS_tracks_list, time_name, circ)
            else:
                t_start = 0
            
            # Обрабатываем временные шаги в файле
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
            
            pbar.set_postfix_str(f"Треков: {cluster_idx}, дата: {year}-{month:02d}-{day:02d}")
            
        else:  # monthly
            year, month, filename = time_unit_info
            file_path = os.path.join(data_dir, filename)
            ds = xr.open_dataset(file_path)
            
            if very_first:
                t_start = 1
                very_first = False
                CS_tracks_list, clstr_len = initialize_tracks(ds, tracking_type, data_type,
                                                              CS_tracks_list, time_name, circ)
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
            
            pbar.set_postfix_str(f"Треков: {cluster_idx}, месяц: {year}-{month:02d}")
        
        ds.close()
    
    pbar.close()
    
    # Сохраняем треки
    for TC in CS_tracks_list:
        if np.sum(~np.isnan(TC['t'])) >= 3:
            cluster_idx = save_track_csv(cluster_idx, TC, path_data_dir)
            save_track_txt(cluster_idx, TC, path_data_dir)
    
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
        for result in pool.imap_unordered(process_tracking_config_improved, tracking_configs):
            results.append(result)
            update_progress(result)
    
    main_pbar.close()
    return {r['config']['type']: r['tracks'] for r in results}


if __name__ == '__main__':
    
    print(f"type of params: {pref_tracking}")
    
    # Запускаем параллельную обработку
    final_results = parallel_process_tracking()
    print("\nAll tracking configurations processed successfully!")