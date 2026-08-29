import sys
from glob import glob
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import os
from multiprocessing import Pool, cpu_count

path_init = f'/storage/thalassa/users/vkoshkina'
folder = 'scripts/CS_processing/CVS_clustering/'
sys.path.insert(2, f'{path_init}/{folder}')

from func_for_add_params_nv import *

def build_r2d_file_map(r2d_dir, r2d_name):
    pattern = os.path.join(r2d_dir, r2d_name)
    file_map = {}
    for f in glob(pattern):
        base = os.path.basename(f)
        ym = base.split("_")[-1].split(".")[0]
        year, month = ym.split("-")
        file_map[(int(year), int(month))] = f
    return file_map

def build_r2d_file_map_universal(r2d_dir, r2d_name):
    """
    Автоматически определяет тип файлов (месячные или посуточные)
    и строит соответствующий маппинг.
    """
    import re
    pattern = os.path.join(r2d_dir, r2d_name)
    file_map = {}
    
    for f in glob(pattern):
        base = os.path.basename(f)
        name_without_ext = os.path.splitext(base)[0]
        
        # Сначала пробуем найти полную дату (YYYY-MM-DD)
        match = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', name_without_ext)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))
            if 1 <= month <= 12 and 1 <= day <= 31:
                file_map[(year, month, day)] = f
                continue
        
        # Если нет полной даты - ищем месяц (YYYY-MM)
        match = re.search(r'(\d{4})[-_]?(\d{2})(?!\d)', name_without_ext)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            if 1 <= month <= 12:
                file_map[(year, month)] = f
                continue
        
        print(f"Предупреждение: не удалось распознать дату в {base}")
    
    return file_map

def add_r2d_max_to_track_df(df, r2d_file_map):
    """
    df: уже посчитанный трек со всеми параметрами (есть time, pxc_ind, pyc_ind, mean_radius)
    r2d_file_map: {(year, month): path_to_nc}
    """
    df = df.copy()
    df['R2D_max'] = np.nan

    for i, row in df.iterrows():
        current_time = row['time']
        year, month = current_time.year, current_time.month
        r2d_path = r2d_file_map.get((year, month))
        if r2d_path is None or pd.isna(row['pxc_ind']) or pd.isna(row['pyc_ind']):
            continue

        try:
            center_lon_idx = int(row['pxc_ind'])
            center_lat_idx = int(row['pyc_ind'])
            radius_pixels = float(row['mean_radius'])

            current_time = row['time'] # на всякий случай

            ds = xr.open_dataset(r2d_path)
            
            # Извлечь срез
            # Шаг внутри суток (00,03,...,21 → 0-7)
            hour_idx = int(current_time.hour // 3)  # или current_time.hour // 3
            
            # Дни с начала месяца (1-го числа 00Z)
            month_start = current_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0).to_numpy()
            seconds_since_start = int((current_time - month_start).total_seconds())
            days_since_month_start = seconds_since_start // (24 * 3600)
            
            # Полный индекс Time
            time_idx = days_since_month_start * 8 + hour_idx
            
            # Извлечение
            r2d_2d = ds['R2D'].isel(Time=time_idx, interp_level=0).values

            r2d_max = get_single_level_values(r2d_2d, center_lat_idx, center_lon_idx, radius_pixels, method='max')
            r2d_95 = get_single_level_values(r2d_2d, center_lat_idx, center_lon_idx, radius_pixels, method=95)
            
            df.at[i, 'R2D_max'] = r2d_max
            df.at[i, 'R2D_95'] = r2d_95
            

        except Exception as e:
            print(f'R2D_max: ошибка в строке {i}: {e}')
            continue

    return df

def add_r2d_max_to_track_df_daily(df, r2d_file_map):
    """
    Для посуточных треков - каждый файл R2D соответствует одному дню.
    
    df: DataFrame с колонками time, pxc_ind, pyc_ind, mean_radius
    r2d_file_map: {(year, month, day): path_to_nc}
    """
    df = df.copy()
    df['R2D_max'] = np.nan
    df['R2D_95'] = np.nan

    for i, row in df.iterrows():
        current_time = row['time']
        
        # Извлекаем год, месяц, день
        year, month, day = current_time.year, current_time.month, current_time.day
        
        # Ищем файл для конкретного дня
        r2d_path = r2d_file_map.get((year, month, day))
        
        if r2d_path is None:
            print(f"Нет файла R2D для {year}-{month:02d}-{day:02d}")
            continue
            
        if pd.isna(row['pxc_ind']) or pd.isna(row['pyc_ind']):
            continue

        try:
            center_lon_idx = int(row['pxc_ind'])
            center_lat_idx = int(row['pyc_ind'])
            radius_pixels = float(row['mean_radius'])

            ds = xr.open_dataset(r2d_path)
            
            # Для посуточных данных - берем первый (или единственный) временной слой
            # Если в файле несколько снимков за день, берем ближайший к времени трека
            if 'Time' in ds.dims:
                time_size = ds.dims['Time']
                
                if time_size == 1:
                    # Единственный снимок за день
                    time_idx = 0
                else:
                    # Несколько снимков - ищем ближайший по времени
                    times = pd.to_datetime(ds.Time.values)
                    time_idx = np.argmin(np.abs(times - current_time))
                    
                    # Проверяем, что разница не слишком большая
                    time_diff = abs((times[time_idx] - current_time).total_seconds() / 3600)
                    if time_diff > 12:  # больше 12 часов - возможно, не тот день
                        print(f"Предупреждение: большая временная разница {time_diff:.1f}ч для {current_time}")
            else:
                # Если нет размерности Time - берем как есть
                time_idx = 0
            
            # Извлекаем R2D
            if 'interp_level' in ds.dims:
                r2d_2d = ds['R2D'].isel(Time=time_idx, interp_level=0).values
            if 'bottom_top' in ds.dims:
                r2d_2d = ds['R2D'].isel(Time=time_idx, bottom_top=0).values
                r2d_center = ds['local_extr_crit'].isel(Time=time_idx, bottom_top=0).values
            else:
                r2d_2d = ds['R2D'].isel(Time=time_idx).values
            
            # r2d_max = get_single_level_values(r2d_2d, center_lat_idx, center_lon_idx, radius_pixels, method='max')

            r2d_max = get_single_level_values(r2d_center, center_lat_idx, center_lon_idx, radius_pixels, method='max')
            
            r2d_95 = get_single_level_values(r2d_2d, center_lat_idx, center_lon_idx, radius_pixels, method=95)
            
            df.at[i, 'R2D_max'] = r2d_max
            df.at[i, 'R2D_95'] = r2d_95
            
            ds.close()  # закрываем Dataset

        except Exception as e:
            print(f'Ошибка в строке {i} для даты {current_time}: {e}')
            continue

    return df
    
def calculate_velocity_rolling(track, data_type='LoRes'):
    """
    Использование pandas rolling для максимальной производительности.
    """
    length = len(track)
    
    if length == 0:
        return np.array([]), np.array([])
    
    if length == 1:
        return np.array([0.0]), np.array([0.0])
    
    # Координаты
    x = track['pxc_ind'].values.astype(float)
    y = track['pyc_ind'].values.astype(float)
    
    # Расстояния между точками
    dx = np.diff(x)
    dy = np.diff(y)
    distances = np.sqrt(dx**2 + dy**2)
    distances = np.insert(distances, 0, 0)
    
    # Скользящее среднее с окном 5 (центрированное)
    # min_periods=2 для учета границ
    rolling_mean = pd.Series(distances).rolling(
        window=5, 
        center=True, 
        min_periods=2
    ).mean()
    
    velocities = rolling_mean.values
    
    # Коэффициент перевода
    if data_type == 'ERA5':
        pixel_to_km = 111 * 0.25
        time_step = 1
    elif data_type == 'SMP':
        pixel_to_km = 6
        time_step = 1
    else:  # LoRes
        pixel_to_km = 77
        time_step = 3
    
    velocities = (velocities * pixel_to_km) / time_step
#     velocities[0] = 0.0
    velocities = np.nan_to_num(velocities, nan=0.0)
    
    # ========== УГЛЫ ==========
    dx_vec = np.diff(x, prepend=x[0])
    dy_vec = np.diff(y, prepend=y[0])
    
    dx_next = np.diff(x, append=x[-1])
    dy_next = np.diff(y, append=y[-1])
    dx_next[-1] = dx_vec[-1]
    dy_next[-1] = dy_vec[-1]
    
    cross = dx_vec * dy_next - dy_vec * dx_next
    dot = dx_vec * dx_next + dy_vec * dy_next
    angles = np.arctan2(cross, dot)
    angles[0] = 0.0
    angles[-1] = 0.0
    angles = np.nan_to_num(angles, nan=0.0)
    
    return velocities, angles

def process_single_csv(args):
    in_path, out_path, r2d_file_map = args
    try:
        # Если в OUTPUT уже есть файл — обновляем его
        if out_path.exists():
            df = pd.read_csv(out_path)
        else:
            df = pd.read_csv(in_path)

        if df.empty:
            return None

        # Обновляем/добавляем R2D-метрики
        df = add_r2d_max_to_track_df(df, r2d_file_map)

        velocities, angles = calculate_velocity_rolling(df, data_type)

        # Добавляем в DataFrame
        df['velocity'] = velocities
        df['angle_radians'] = angles

        # Например, убираем старый p95
        if "R2D_p95" in df.columns:
            df = df.drop(columns=["R2D_p95"])

        # Перезаписываем файл в OUTPUT_PATH
        df.to_csv(out_path, index=False)
        return out_path
    except Exception as e:
        print(f"Ошибка при обработке {in_path.name}: {e}")
        return None

def update_all_tracks_parallel(input_hourly_dir, output_hourly_dir, r2d_file_map, n_processes=None):
    input_hourly_dir = Path(input_hourly_dir)
    output_hourly_dir = Path(output_hourly_dir)

    # создаём OUT только один раз
    output_hourly_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_hourly_dir.glob("*.csv"))
    
    print(f"Найдено треков (CSV): {len(csv_files)}")

    if not csv_files:
        return

    if n_processes is None:
        n_processes = min(cpu_count(), len(csv_files))

    args_list = []
    for in_path in csv_files:
        out_path = output_hourly_dir / in_path.name
        args_list.append((in_path, out_path, r2d_file_map))

    print(f"\nПараллельная обработка:")
    print(f"  • Файлов: {len(args_list)}")
    print(f"  • Процессов: {n_processes}")

    with Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_csv, args_list),
            total=len(args_list),
            desc=" Обновление CSV"
        ))

    done = sum(1 for r in results if r is not None)
    print(f"\nГотово файлов: {done} / {len(args_list)}")



def process_single_csv(args):
    in_path, out_path, r2d_file_map = args
    try:
        # Делаем директорию только по родителю и с exist_ok=True
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Если в OUTPUT уже есть файл — обновляем его
        if out_path.exists():
            df = pd.read_csv(out_path)
        else:
            df = pd.read_csv(in_path)

        if df.empty:
            return None

        df['time'] = pd.to_datetime(df['time'])

        if data_type == 'LoRes':
            df = add_r2d_max_to_track_df(df, r2d_file_map)
        elif data_type == 'SMP':
            df = add_r2d_max_to_track_df_daily(df, r2d_file_map)
        

        velocities, angles = calculate_velocity_rolling(df, data_type)

        # Добавляем в DataFrame
        df['velocity'] = velocities
        df['angle_radians'] = angles

        if "R2D_p95" in df.columns:
            df = df.drop(columns=["R2D_p95"])

        df.to_csv(out_path, index=False)
        return out_path
    except Exception as e:
        print(f"Ошибка при обработке {in_path.name}: {e}")
        return None


def process_all_tracks_csv(input_hourly_dir, output_hourly_dir, r2d_file_map, n_processes=None):
    os.makedirs(output_hourly_dir, exist_ok=True)

    csv_files = sorted(glob(os.path.join(input_hourly_dir, "*.csv")))
    print(f"Найдено треков (CSV): {len(csv_files)}")

    if n_processes is None:
        n_processes = min(cpu_count(), len(csv_files))

    args_list = [(csv_path, output_hourly_dir, r2d_file_map) for csv_path in csv_files]

    print(f"\nПараллельная обработка CSV:")
    print(f"  • Файлов: {len(csv_files)}")
    print(f"  • Процессов: {n_processes}")

    with Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_csv, args_list),
            total=len(args_list),
            desc=" Обработка треков (R2D)"
        ))

    done = sum(1 for r in results if r is not None)
    print(f"\nГотово файлов: {done} / {len(csv_files)}")


data_type = 'LoRes'
data_type = 'SMP'


if data_type == 'LoRes':
    r2d_dir = "/storage/thalassa/users/vkoshkina/data/LoRes/LoRes/R2D_LoRes_level_12_smoothing_sigma_2"
    r2d_name = "sigma_2_R2D_LoRes_level_12_*.nc"
elif data_type == 'SMP':
    r2d_dir = "/storage/thalassa/users/vkoshkina/data/SMP/DBSCAN_02-04-10_with_wspd_smoothing/2019"
    r2d_name = "sigma_2_DBSCAN_SMP_level_10_*.nc"

R2D_FILE_MAP = build_r2d_file_map_universal(r2d_dir, r2d_name)

if data_type == 'LoRes':
    # INPUT_PATH = Path("/storage/thalassa/users/vkoshkina/data/LoRes/LoRes/EddyClicker_tracks_Egor_2010_params")
    # OUTPUT_PATH = Path("/storage/thalassa/users/vkoshkina/data/LoRes/LoRes/EddyClicker_tracks_Egor_2010_params_r2d")
    
    INPUT_PATH = Path("/storage/thalassa/users/vkoshkina/data/LoRes/LoRes/EddyClicker_tracks_Egor_2010_15params_2026-08-11")
    OUTPUT_PATH = Path("/storage/thalassa/users/vkoshkina/data/LoRes/LoRes/EddyClicker_tracks_Egor_2010_15params_2026-08-11_r2d")
    
    INPUT_PATH = Path("/storage/thalassa/users/vkoshkina/data/LoRes/LoRes/EddyClicker_tracks_Egor_2010_15params_2026-08-11")
    OUTPUT_PATH = Path("/storage/thalassa/users/vkoshkina/data/LoRes/LoRes/EddyClicker_tracks_Egor_2010_15params_2026-08-11_r2d")
elif data_type == 'SMP':
    INPUT_PATH = Path("/storage/thalassa/users/vkoshkina/data/SMP/EddyClicker_tracks_2019_15params_2028-08-10/")
    OUTPUT_PATH = Path("/storage/thalassa/users/vkoshkina/data/SMP/EddyClicker_tracks_2019_15params_2026-08-12_r2d/")

in_hourly = INPUT_PATH / "hourly_data"
out_hourly = OUTPUT_PATH / "hourly_data"

update_all_tracks_parallel(in_hourly, out_hourly, R2D_FILE_MAP, n_processes=30)


# INPUT_PATH = Path("/storage/thalassa/users/vkoshkina/data/LoRes/LoRes/EddyClicker_tracks_Egor_params_nv")
# OUTPUT_PATH = Path("/storage/thalassa/users/vkoshkina/data/LoRes/LoRes/EddyClicker_tracks_Egor_params_r2d")

# path_hourly_in = INPUT_PATH / "hourly_data"
# path_hourly_out = OUTPUT_PATH / "hourly_data"

# # Создаем выходные папки (если нет)
# path_hourly_out.mkdir(parents=True, exist_ok=True)

# # Все csv во входной папке
# csv_files = sorted(path_hourly_in.glob("*.csv"))

# print(f"Найдено файлов: {len(csv_files)}")

# for csv_path in csv_files:
#     try:
#         df_track = pd.read_csv(csv_path)
#         df_track['time'] = pd.to_datetime(df_track['time'])
        
#         if df_track.empty:
#             continue

#         # Обработка
#         df_track = add_r2d_max_to_track_df(df_track, R2D_FILE_MAP)

#         # Имя файла без изменений
#         out_path = path_hourly_out / csv_path.name
#         df_track.to_csv(out_path, index=False)
#         print(f"Готово: {csv_path.name} → {out_path}")
#     except Exception as e:
#         print(f"Ошибка при обработке {csv_path.name}: {e}")