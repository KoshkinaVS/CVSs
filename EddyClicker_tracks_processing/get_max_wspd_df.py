import os
import glob
import pandas as pd
from tqdm import tqdm
import numpy as np
import xarray as xr


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
    
def calculate_track_len(df, data_type='LoRes'):
    df = df.copy()  # чтобы не трогать оригинал, если не нужно
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    
    # Вычисляем смещения между соседними точками
    dx = df['x'].diff()
    dy = df['y'].diff()
    
    # Евклидово расстояние между последовательными точками
    step_lengths = np.sqrt(dx**2 + dy**2)
    
    # Накопленная сумма (первая строка будет NaN → заменяем на 0)
    df['track_len'] = step_lengths.cumsum().fillna(0)
    return df
    
def calculate_velocity(track, max_idx, data_type='LoRes'):

    # для проверки, что максимум не на границе трека
    start_stop = False
    length = len(track)
    
    if length == 1:
        return np.nan, False
    
    # Преобразуем max_idx в целое число, если это Index объект
    try:
        if hasattr(max_idx, 'item'):  # Для pandas Index и подобных
            max_idx = max_idx.item()
        max_idx = int(max_idx)
    except (ValueError, TypeError):
        return np.nan, False
    
    track_len = track['track_len'].values
    
    # Границы окна (гарантируем целые числа)
    start = max(max_idx - 2, 0)
    end = min(max_idx + 2, length - 1)
    start, end = int(start), int(end)  # Явное преобразование

    
    # Проверяем граничные условия
    if max_idx == length - 1 or max_idx == 0:
        start_stop = True
    
    # Если окно слишком маленькое (меньше 2 точек), возвращаем NaN
    if end - start < 1:
        return np.nan, start_stop
    
    # Вычисляем все разности в окрестности
    diffs = np.diff(track_len[start:end+1])

    if data_type == 'ERA5':
        diffs = diffs*np.cos(np.radians(track['lat'][start:end]))

    if data_type == 'ERA5':
        avg_velocity = (111*1000*0.25 * np.mean(diffs) / (3600))
    elif data_type == 'SMP':    
        # Вычисляем среднюю скорость (в м/с)
        avg_velocity = (6*1000 * np.mean(diffs) / (3600)) * 3.6 # km/h
    else:    
        # Вычисляем среднюю скорость (в м/с)
        avg_velocity = (77*1000 * np.mean(diffs) / (3*3600)) * 3.6 # km/h
    
    return avg_velocity, start_stop


path_data_dir = '/storage/thalassa/users/vkoshkina/data'
data_type = 'LoRes'  # 
data_type = 'SMP'





if data_type == 'LoRes':
    path_data_dir = f'{path_data_dir}/LoRes'
    # tracks_params_folder = 'LoRes_tracks_params_mattiew_2026-05-13'
    tracks_params_folder = 'EddyClicker_tracks_Egor_2010_15params_2028-08-10_r2d'
    tracks_params_folder = 'EddyClicker_tracks_Egor_2010_15params_2026-08-11_r2d'
    
    time_th = 8
    ocean_threshold = 0.5
    
    # Диапазон лет и месяцев
    years = np.arange(1979, 2019)
    months = np.arange(1, 13, 1)
    
    # Загрузка ground для фильтрации над океаном
    path_dir_raw = f'/storage/NAAD/NAAD/LoRes/2010'
    ncfile = f'{path_dir_raw}/wrfout_d01_2010-01-01_00:00:00'
    ground_ds = xr.open_dataset(f'{ncfile}')['HGT'][0]
    ground = np.where(ground_ds > 5, 1, np.nan)
    
elif data_type == 'ERA5':
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
    tracks_params_folder = 'EddyClicker_tracks_nikitenko_2019_r2d'
    tracks_params_folder = 'EddyClicker_tracks_2019_15params_2026-08-12_r2d'
    time_th = 8
    ocean_threshold = 0.8
    
    years = np.arange(2019, 2020)  # нужный год
    months = np.arange(1, 13, 1)
    ground = None
    

above_ocean = '_ocean'
above_ocean = ''

path_data_tracks = f'{path_data_dir}/{data_type}/{tracks_params_folder}/hourly_data'


# track_folder_new = f'{path_data_dir}/{data_type}/{tracks_params_folder}/max_wspd_day_ocean'
track_folder_new = f'{path_data_dir}/{data_type}/{tracks_params_folder}/max_wspd_day'

# Создаем конечную папку, если ее нет
if not os.path.exists(track_folder_new):
    os.makedirs(track_folder_new)

# Собираем все CSV-файлы
files = sorted(glob.glob(f'{path_data_tracks}/*.csv'))

all_max_rows = []

for file in tqdm(files, desc="Extracting max-intensity rows"):
    # 1. Безопасное чтение с проверкой на пустоту
    try:
        df = pd.read_csv(file)
        if df.empty:
            print(f"⚠️ Warning: {os.path.basename(file)} is empty. Skipping.")
            continue
    except pd.errors.EmptyDataError:
        print(f"⚠️ Warning: {os.path.basename(file)} has no columns to parse. Skipping.")
        continue
    except Exception as e:
        print(f"❌ Error reading {os.path.basename(file)}: {e}. Skipping.")
        continue

    df = df.rename(columns={
        'pxc_ind': 'x',
        'pyc_ind': 'y',
    })

    if above_ocean == '_ocean':
        # Фильтрация над океаном 2026-08-11
        if not is_track_over_ocean(df, ground, ocean_threshold):
            continue

    df = calculate_track_len(df)
    
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    elif 'time' in df.columns:
        df = df.rename(columns={'time': 'datetime'})
        df['datetime'] = pd.to_datetime(df['datetime'])
    else:
        print(f"⚠️ Warning: no datetime/time column in {os.path.basename(file)}. Skipping.")
        continue

    max_param_name = 'R2D_max'        
    if max_param_name not in df.columns:
        print(f"⚠️ Warning: {max_param_name} not found in {os.path.basename(file)}. Skipping.")
        continue

    # 2. Ищем максимум, исключая первую строку
    max_candidates = df[max_param_name].iloc[1:]
    if len(max_candidates) == 0:
        print(f"⚠️ Skipping {os.path.basename(file)}: only one timestep")
        continue
    
    max_idx = max_candidates.idxmax()  # Индекс максимума в оригинальном df
    max_row = df.loc[max_idx]
    max_row_df = pd.DataFrame([max_row]).reset_index(drop=True)

    max_row_df = remove_unnecessary_columns(max_row_df)
    
    max_row_df['path'] = file
    max_row_df['name'] = os.path.basename(file)
    
    all_max_rows.append(max_row_df)

# Объединяем всё в один DataFrame
if all_max_rows:
    final_df = pd.concat(all_max_rows, ignore_index=True)
    
    # Убедимся, что datetime — первая колонка
    cols = ['datetime'] + [col for col in final_df.columns if col != 'datetime']
    final_df = final_df[cols]
    
    # Сохраняем результат
    output_path = f'{track_folder_new}/all_tracks_with_params_max_wspd_day{above_ocean}.csv'
    final_df.to_csv(output_path, index=False)
    print(f"Сохранено {len(final_df)} записей в {output_path}")
else:
    print("Ни один файл не был обработан.")