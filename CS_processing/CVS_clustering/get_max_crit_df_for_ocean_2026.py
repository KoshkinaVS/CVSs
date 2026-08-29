import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm
from func_for_CVS_clusters import *

# ==================== НАСТРОЙКИ ====================
tracking_type = 'tracking_local_2_phase'
pref_tracking = 'all_points_bound'
CVS_speed = 'adv_speed'
circ = 'C'
sigma = 2

data_type = 'LoRes'  # или 'ERA5', 'SMP'

# Пути
path_init = '/storage/thalassa/users/vkoshkina'
path_data = f'{path_init}/data'

if data_type == 'LoRes':
    path_data_dir = f'{path_data}/LoRes'
    # tracks_params_folder = 'LoRes_tracks_params_mattiew_2026-05-13'
    # tracks_params_folder = 'LoRes_tracks_2010_15params_2026-08-11'
    tracks_params_folder = 'LoRes_tracks_1979-2018_15params_2026-08-11'
    
    time_th = 8
    ocean_threshold = 0.8
    
    # Диапазон лет и месяцев
    years = np.arange(1979, 2019)
    # years = np.arange(2010, 2011)
    
    months = np.arange(1, 13, 1)
    
    # Загрузка ground для фильтрации над океаном
    path_dir_raw = f'/storage/NAAD/NAAD/LoRes/2010'
    ncfile = f'{path_dir_raw}/wrfout_d01_2010-01-01_00:00:00'
    ground_ds = xr.open_dataset(f'{ncfile}')['HGT'][0]
    ground = np.where(ground_ds > 5, 1, np.nan)
    
elif data_type == 'ERA5':
    path_data_dir = f'{path_data}'
    tracks_params_folder = 'my_tracking_results'
    time_th = 24
    ocean_threshold = 0.5
    
    years = np.arange(1979, 2025)
    months = np.arange(1, 13, 1)
    
    path_dir_raw = f'/storage/thalassa/users/vkoshkina/data/ERA5'
    ncfile = f'{path_dir_raw}/ERA5_lsm_cropped.nc'
    ground_ds_ERA5 = xr.open_dataset(f'{ncfile}')['var172'][0]
    ground = np.where(ground_ds_ERA5 > 0.5, 1, np.nan)
    
elif data_type == 'SMP':
    path_data_dir = f'{path_data}'
    tracks_params_folder = 'EddyClicker_tracks_nikitenko_2019_r2d'
    time_th = 8
    ocean_threshold = 0.8
    
    years = np.arange(2019, 2020)  # нужный год
    months = np.arange(1, 13, 1)
    ground = None

# Путь к данным
path_data_tracks = f'{path_data_dir}/{data_type}/{data_type}_tracks/{tracks_params_folder}/'

above_ocean = '_ocean'
# above_ocean = ''


# Папка для сохранения результатов
track_folder_new = f'{path_data_dir}/{data_type}/{data_type}_tracks/{tracks_params_folder}/max_wspd_day{above_ocean}'
os.makedirs(track_folder_new, exist_ok=True)

# ==================== ФУНКЦИИ ====================

def calculate_track_len(df, data_type='LoRes'):
    """Вычисление длины трека"""
    df = df.copy()
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    
    dx = df['x'].diff()
    dy = df['y'].diff()
    step_lengths = np.sqrt(dx**2 + dy**2)
    df['track_len'] = step_lengths.cumsum().fillna(0)
    return df

def calculate_velocity(track, max_idx, data_type='LoRes'):
    """Расчет скорости в момент максимума"""
    start_stop = False
    length = len(track)
    
    if length == 1:
        return np.nan, False
    
    try:
        if hasattr(max_idx, 'item'):
            max_idx = max_idx.item()
        max_idx = int(max_idx)
    except (ValueError, TypeError):
        return np.nan, False
    
    track_len = track['track_len'].values
    
    start = max(max_idx - 2, 0)
    end = min(max_idx + 2, length - 1)
    start, end = int(start), int(end)
    
    if max_idx == length - 1 or max_idx == 0:
        start_stop = True
    
    if end - start < 1:
        return np.nan, start_stop
    
    diffs = np.diff(track_len[start:end+1])
    
    if data_type == 'ERA5':
        diffs = diffs * np.cos(np.radians(track['lat'][start:end]))
        avg_velocity = (111*1000*0.25 * np.mean(diffs) / 3600)
    elif data_type == 'SMP':    
        avg_velocity = (6*1000 * np.mean(diffs) / 3600) * 3.6  # km/h
    else:    
        avg_velocity = (77*1000 * np.mean(diffs) / (3*3600)) * 3.6  # km/h
    
    return avg_velocity, start_stop

def is_track_over_ocean(track_df, ground, ocean_threshold=0.8):
    """Проверка, проходит ли трек преимущественно над океаном"""
    if ground is None:
        return True
    
    if 'x' in track_df.columns and 'y' in track_df.columns:
        xs = track_df['x'].values.astype(int)
        ys = track_df['y'].values.astype(int)
        
        xs = np.clip(xs, 0, ground.shape[1] - 1)
        ys = np.clip(ys, 0, ground.shape[0] - 1)
        
        ocean_mask = np.isnan(ground[ys, xs])
        ocean_fraction = np.mean(ocean_mask)
        
        return ocean_fraction >= ocean_threshold
    return True

def remove_unnecessary_columns(df):
    """Удаление лишних колонок"""
    cols_to_drop = ['time_ind', 'px1_ind', 'py1_ind', 'px2_ind', 'py2_ind',
                    'px3_ind', 'py3_ind', 'semi_major', 'semi_minor', 
                    'bergeron_2h', 'bergeron_4h', 'bergeron_6h', 'bergeron_12h', 'bergeron_24h']
    
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    if existing_cols_to_drop:
        df = df.drop(columns=existing_cols_to_drop)
    return df

def process_track_file(file, ground, ocean_threshold, data_type):
    """Обработка одного файла трека"""
    try:
        df = pd.read_csv(file)
        if df.empty:
            return None
    except Exception as e:
        print(f"Error reading {file}: {e}")
        return None
    
    # # Переименование колонок
    # df = df.rename(columns={
    #     'pxc_ind': 'x',
    #     'pyc_ind': 'y',
    #     'time': 'datetime'
    # })

    # # Переименование колонок
    df = df.rename(columns={
        'crit': 'R2D_max',
    })
    
    if 'datetime' not in df.columns:
        return None
    
    df['datetime'] = pd.to_datetime(df['datetime'])

    ### 2026-08-11 turned off
    # # Фильтрация над океаном
    if above_ocean == '_ocean':
        if not is_track_over_ocean(df, ground, ocean_threshold):
            return None
    
    # Вычисление длины трека
    df = calculate_track_len(df, data_type)
    
    # Определение параметра для максимума/минимума
    if 'crit' in df.columns:
        max_param = 'crit'
        use_min = True
    elif 'R2D_max' in df.columns:
        max_param = 'R2D_max'
        use_min = False
    elif 'wspd_max' in df.columns:
        max_param = 'wspd_max'
        use_min = False
    elif 'msl_min' in df.columns:
        max_param = 'msl_min'
        use_min = True
    else:
        return None
    
    # Поиск экстремума
    max_candidates = df[max_param].iloc[1:]
    if len(max_candidates) == 0:
        return None
    
    if use_min:
        max_idx = max_candidates.idxmin()
    else:
        max_idx = max_candidates.idxmax()
    
    max_row = df.loc[max_idx]
    max_row_df = pd.DataFrame([max_row]).reset_index(drop=True)
    
    # Добавление метаданных
    max_row_df['track_len'] = df['track_len'].iloc[-1]
    max_row_df['duration'] = (df['datetime'].iloc[-1] - df['datetime'].iloc[0]).total_seconds() / 3600
    
    vel, start_stop = calculate_velocity(df, max_idx, data_type)
    max_row_df['velocity'] = vel
    max_row_df['start_stop'] = start_stop
    
    max_row_df['path'] = file
    max_row_df['name'] = os.path.basename(file)
    
    return max_row_df

# ==================== ОСНОВНАЯ ОБРАБОТКА ПО МЕСЯЦАМ ====================

all_tracks_list = []  # для сбора всех треков (опционально)

for year in tqdm(years, desc="Processing years"):
    for month in months:
        
        # Путь к папке с данными за конкретный месяц
        month_path = f'{path_data_tracks}/{year}-{month:02d}/hourly_data/'
        
        if not os.path.exists(month_path):
            print(f"Path not found: {month_path}")
            continue
        
        # Поиск CSV файлов за этот месяц
        files = sorted(glob.glob(f'{month_path}/*.csv'))
        
        if not files:
            print(f"No files found for {year}-{month:02d}")
            continue
        
        print(f"\nProcessing {year}-{month:02d}: {len(files)} tracks")
        
        # Обработка треков за текущий месяц
        month_tracks = []
        
        for file in tqdm(files, desc=f"Processing {year}-{month:02d}", leave=False):
            result = process_track_file(file, ground, ocean_threshold, data_type)
            if result is not None:
                month_tracks.append(result)
        
        # Сохранение результатов за месяц
        if month_tracks:
            month_df = pd.concat(month_tracks, ignore_index=True)
            
            # Удаление лишних колонок
            month_df = remove_unnecessary_columns(month_df)
            
            # Упорядочивание колонок
            cols = ['datetime'] + [col for col in month_df.columns if col != 'datetime']
            month_df = month_df[cols]
            
            # Сохранение файла за месяц
            output_file = f'{track_folder_new}/max_crit_day_CVS_{year}-{month:02d}.csv'
            month_df.to_csv(output_file, index=False)
            print(f"Saved {len(month_df)} records to {output_file}")
            
            # Добавляем в общий список (опционально)
            all_tracks_list.append(month_df)
        else:
            print(f"No ocean tracks found for {year}-{month:02d}")

# ==================== СОХРАНЕНИЕ ОБЩЕГО ФАЙЛА (опционально) ====================

if all_tracks_list:
    final_df = pd.concat(all_tracks_list, ignore_index=True)
    
    # Сохранение общего файла
    # output_path = f'{track_folder_new}/all_tracks_max_intensity_ocean_only.csv'
    output_path = f'{track_folder_new}/all_tracks_max_intensity{above_ocean}.csv'
    
    final_df.to_csv(output_path, index=False)
    print(f"\n{'='*50}")
    print(f"Total processed: {len(final_df)} records")
    print(f"Saved to {output_path}")
    print(f"{'='*50}")
    
    # Вывод статистики
    print(f"\nStatistics:")
    print(f"Total tracks over ocean: {len(final_df)}")
    print(f"Average velocity: {final_df['velocity'].mean():.2f} km/h")
    print(f"Average duration: {final_df['duration'].mean():.1f} hours")
    print(f"Years range: {final_df['datetime'].min().year} - {final_df['datetime'].max().year}")
    
else:
    print("\nNo tracks were processed")