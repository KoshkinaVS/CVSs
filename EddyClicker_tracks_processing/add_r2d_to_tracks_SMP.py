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


def build_r2d_file_map(r2d_dir, data_type, level):
    """
    Строит словарь для доступа к файлам R2D по дате (год, месяц, день)
    """
    pattern = os.path.join(r2d_dir, f"sigma_2_R2D_{data_type}_level_{level}_*.nc")
    file_map = {}
    for f in glob(pattern):
        base = os.path.basename(f)
        # Ожидаемый формат: sigma_2_R2D_SMP_level_10_2019-01-01.nc
        # или sigma_2_R2D_SMP_level_10_2019-01.nc (если помесячно)
        date_part = base.split("_")[-1].split(".")[0]
        
        parts = date_part.split("-")
        if len(parts) == 3:  # Есть день: 2019-01-01
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            file_map[(year, month, day)] = f
        elif len(parts) == 2:  # Только месяц: 2019-01
            year, month = int(parts[0]), int(parts[1])
            file_map[(year, month)] = f
        else:
            print(f"Предупреждение: неожиданный формат имени файла: {base}")
            continue
            
    return file_map


def add_r2d_max_to_track_df(df, r2d_file_map):
    """
    df: уже посчитанный трек со всеми параметрами (есть time, pxc_ind, pyc_ind, mean_radius)
    r2d_file_map: {(year, month, day): path_to_nc} или {(year, month): path_to_nc}
    """
    df = df.copy()
    df['R2D_max'] = np.nan
    df['R2D_95'] = np.nan

    for i, row in df.iterrows():
        current_time = row['time']
        year, month, day = current_time.year, current_time.month, current_time.day
        
        # Пытаемся найти файл сначала с днем, потом без дня
        r2d_path = r2d_file_map.get((year, month, day))
        if r2d_path is None:
            r2d_path = r2d_file_map.get((year, month))
            
        if r2d_path is None or pd.isna(row['pxc_ind']) or pd.isna(row['pyc_ind']):
            continue

        try:
            center_lon_idx = int(row['pxc_ind'])
            center_lat_idx = int(row['pyc_ind'])
            radius_pixels = float(row['mean_radius'])

            ds = xr.open_dataset(r2d_path)
            
            
            # Извлечение
            r2d_2d = ds['R2D'].isel(time=int(current_time.hour)).values

            r2d_max = get_single_level_values(r2d_2d, center_lat_idx, center_lon_idx, radius_pixels, method='max')
            r2d_95 = get_single_level_values(r2d_2d, center_lat_idx, center_lon_idx, radius_pixels, method=95)
            
            df.at[i, 'R2D_max'] = r2d_max
            df.at[i, 'R2D_95'] = r2d_95
            
            ds.close()

        except Exception as e:
            print(f'R2D_max: ошибка в строке {i} (time={current_time}): {e}')
            continue

    return df


def process_single_csv(args):
    in_path, out_path, r2d_file_map = args
    try:
        # Создаём директорию выхода (безопасно)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Проверка: файл существует и не пустой
        if not in_path.exists() or in_path.stat().st_size == 0:
            print(f"⚠️ Пропущен пустой/отсутствующий файл: {in_path.name}")
            return None

        # 2. Безопасное чтение CSV
        try:
            # Если выходной файл есть — читаем его (чтобы не терять уже добавленные поля)
            if out_path.exists() and out_path.stat().st_size > 0:
                df = pd.read_csv(out_path)
            else:
                df = pd.read_csv(in_path)
        except pd.errors.EmptyDataError:
            print(f"⚠️ EmptyDataError: {in_path.name} — файл пуст или не имеет колонок")
            return None
        except pd.errors.ParserError as e:
            print(f"⚠️ ParserError в {in_path.name}: {e}")
            return None
        except Exception as e:
            print(f"⚠️ Ошибка чтения CSV {in_path.name}: {type(e).__name__}: {e}")
            return None

        if df.empty or 'time' not in df.columns:
            print(f"⚠️ Пропущен: {in_path.name} — пустой DataFrame или нет колонки 'time'")
            return None

        # Приводим time к datetime
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        if df['time'].isna().all():
            print(f"⚠️ Пропущен: {in_path.name} — не удалось распарсить даты")
            return None

        # Добавляем R2D-метрики
        df = add_r2d_max_to_track_df(df, r2d_file_map)

        # Убираем дубликаты/старые колонки при необходимости
        if "R2D_p95" in df.columns:
            df = df.drop(columns=["R2D_p95"])

        # Сохраняем
        df.to_csv(out_path, index=False)
        return out_path

    except Exception as e:
        print(f"❌ Критическая ошибка при обработке {in_path.name}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
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
        
        df = add_r2d_max_to_track_df(df, r2d_file_map)

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


# # один раз заранее
# r2d_dir = "/storage/thalassa/users/vkoshkina/data/LoRes/LoRes/R2D_LoRes_level_12_smoothing_sigma_2"
# R2D_FILE_MAP = build_r2d_file_map(r2d_dir)


# INPUT_PATH = Path("/storage/thalassa/users/vkoshkina/data/LoRes/LoRes/EddyClicker_tracks_Egor_2010_params")
# OUTPUT_PATH = Path("/storage/thalassa/users/vkoshkina/data/LoRes/LoRes/EddyClicker_tracks_Egor_2010_params_r2d")

data_type = 'SMP'
level = 10
sigma = 2

path_init = f'/storage/thalassa/users/vkoshkina/data'
r2d_dir = f"{path_init}/TempestExtremes/{data_type}/R2D_{data_type}_level_{level}_sigma_{sigma}/Input"

R2D_FILE_MAP = build_r2d_file_map(r2d_dir, data_type, level)


INPUT_PATH = Path(f"/storage/thalassa/users/vkoshkina/data/{data_type}/EddyClicker_tracks_nikitenko_2019")
OUTPUT_PATH = Path(f"/storage/thalassa/users/vkoshkina/data/{data_type}/EddyClicker_tracks_nikitenko_2019_r2d")

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