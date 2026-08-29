import os
import glob
import pandas as pd
from tqdm import tqdm
import numpy as np

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


# Укажите путь к папке с обработанными треками
path_data_dir = '/storage/thalassa/users/vkoshkina/data'  # как в вашем коде
data_type = 'LoRes'  # или 'LoRes', в зависимости от того, что вы сейчас обрабатываете

if data_type == 'LoRes':
    path_data_dir = f'{path_data_dir}/{data_type}'
tracks_params_folder = 'EddyClicker_tracks_Egor_2010_params_r2d'
# tracks_params_folder = 'EddyClicker_tracks_Egor_params_nv'

if data_type == 'SMP':
    tracks_params_folder = 'EddyClicker_tracks_nikitenko_2019_r2d'

path_data_tracks = f'{path_data_dir}/{data_type}/{tracks_params_folder}/hourly_data'

track_folder_new = f'{path_data_dir}/{data_type}/{tracks_params_folder}/start_stop_max'
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

    # Проверяем, что трек имеет хотя бы 3 точки (начало, максимум, конец)
    if len(df) < 3:
        print(f"⚠️ Skipping {os.path.basename(file)}: less than 3 timesteps")
        continue
    
    # 2. Ищем максимум, исключая первую строку
    max_candidates = df[max_param_name].iloc[1:]
    if len(max_candidates) == 0:
        print(f"⚠️ Skipping {os.path.basename(file)}: only one timestep")
        continue
    
    max_idx = max_candidates.idxmax()  # Индекс максимума в оригинальном df
    
    # Создаем словарь для хранения данных текущего трека
    track_data = {}
    
    # Добавляем общие параметры трека (не зависящие от конкретного таймстепа)
    track_data['track_len_total'] = df['track_len'].iloc[-1]
    track_data['duration_hours'] = (df['datetime'].iloc[-1] - df['datetime'].iloc[0]).total_seconds() / 3600
    track_data['n_timesteps'] = len(df)
    track_data['path'] = file
    track_data['name'] = os.path.basename(file)
    
    # Вычисляем скорость в момент максимума
    vel, start_stop = calculate_velocity(df, max_idx, data_type)
    track_data['vel_at_max_kmh'] = vel
    track_data['max_at_boundary'] = start_stop
    
    # Список колонок для удаления
    cols_to_drop = ['time_ind', 'px1_ind', 'py1_ind', 'px2_ind', 'py2_ind',
                    'px3_ind', 'py3_ind', 'semi_major', 'semi_minor']
    
    # Обрабатываем строку начала трека (первый индекс, 0)
    start_row = df.iloc[0]
    for col in start_row.index:
        if col not in cols_to_drop and col != 'datetime':
            track_data[f'{col}_start'] = start_row[col]
    track_data['datetime_start'] = start_row['datetime']
    
    # Обрабатываем строку максимума
    max_row = df.loc[max_idx]
    for col in max_row.index:
        if col not in cols_to_drop and col != 'datetime':
            track_data[f'{col}_max'] = max_row[col]
    track_data['datetime_max'] = max_row['datetime']
    
    # Обрабатываем строку конца трека (последний индекс)
    stop_row = df.iloc[-1]
    for col in stop_row.index:
        if col not in cols_to_drop and col != 'datetime':
            track_data[f'{col}_stop'] = stop_row[col]
    track_data['datetime_stop'] = stop_row['datetime']
    
    # Добавляем в список
    all_max_rows.append(track_data)

# Объединяем всё в один DataFrame
if all_max_rows:
    final_df = pd.DataFrame(all_max_rows)
    
    # Переупорядочиваем колонки: datetime_* сначала
    datetime_cols = ['datetime_start', 'datetime_max', 'datetime_stop']
    other_cols = [col for col in final_df.columns if col not in datetime_cols]
    final_df = final_df[datetime_cols + other_cols]
    
    # Сохраняем результат
    output_path = f'{track_folder_new}/all_tracks_with_params_start_stop_max.csv'
    final_df.to_csv(output_path, index=False)
    print(f"✅ Сохранено {len(final_df)} записей в {output_path}")
    
    # Выводим информацию о колонках
    print(f"\n📊 Всего колонок: {len(final_df.columns)}")
    print("\nКолонки с начальными параметрами (_start):", 
          [col for col in final_df.columns if col.endswith('_start')][:5], "...")
    print("Колонки с параметрами максимума (_max):", 
          [col for col in final_df.columns if col.endswith('_max')][:5], "...")
    print("Колонки с конечными параметрами (_stop):", 
          [col for col in final_df.columns if col.endswith('_stop')][:5], "...")
    
    # Показываем первые несколько строк для проверки
    print("\n📋 Первые 3 строки результата:")
    print(final_df.head(3))
else:
    print("❌ Ни один файл не был обработан.")