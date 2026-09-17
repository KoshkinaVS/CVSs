"""
Отбор моментов максимальной интенсивности (max R2D) для треков из НОВОГО
альтернативного пайплайна трекинга (TempestExtremes/StitchNodes по критерию
R2D на уровне 850 гПа), адаптация get_max_crit_df_for_ocean_2026.py.

Источник данных
----------------
/storage/thalassa/users/vkoshkina/data/TempestExtremes/ERA5/
    R2D_ERA5_NA_for_TC_850hPa_sigma_2/
    csv_Tracks_range_1_5_18h_12h_2010_25points_global_params/hourly_data/

Это результат работы:
    1) CVS_alt_tracking/TempestExtremes/create_Nodes_from_DBSCAN.py  -> узлы для TE
    2) CVS_alt_tracking/TempestExtremes/run_StitchNodes_by_year.py   -> треки (StitchNodes,
       --in_fmt "lon,lat,rad,r2d,wind")
    3) (конвертация текстового вывода StitchNodes в отдельные CSV на трек
       вида NNNNNN_track_YYYY-MM-DDTHH.csv, с колонками как минимум
       i, j, lon, lat, rad, r2d, wind, time)
    4) ERA5/add_params_v2_fast_2026-08-24.py -> добавляет 12 колонок ERA5-параметров
       в момент каждой точки трека (RESULT_PARAMS), пишет их в .../hourly_data/*.csv

Что изменено относительно оригинального get_max_crit_df_for_ocean_2026.py
---------------------------------------------------------------------------
1. Папка с треками ПЛОСКАЯ и на ОДИН ГОД (2010) — не по месяцам
   ({year}-{month:02d}/hourly_data/), поэтому вместо двойного цикла
   год x месяц делается один glob по hourly_data и группировка результатов
   по месяцу уже ПОСЛЕ вычислений (для месячных выходных файлов, см. ниже).
2. Критерий интенсивности называется в этих файлах 'r2d' (а не 'crit', как
   в старом DBSCAN/LoRes-пайплайне) — переименовывается в 'R2D_max', дальше
   код работает как раньше (max_param='R2D_max', use_min=False).
3. Индексы сетки в этих файлах называются 'i'/'j' (а не 'x'/'y' или
   'pxc_ind'/'pyc_ind') — переименовываются в 'x'/'y', это нужно для
   track_len/скорости и для фильтра "над океаном" (см. ниже).
4. Добавлена колонка 'mean_radius' = средний 'rad' по всему треку (в
   ADDED_PARAMS/numeric_cols второго скрипта используется именно она, а не
   'rad' в конкретной точке).
5. Маска суша/океан взята из ERA5_lsm_cropped_NA_for_TC.nc (переменная
   'lsm'), а НЕ из ERA5_lsm_cropped.nc (var172), который использовался для
   старого (DBSCAN) ERA5-пайплайна и обрезан под другой регион (Arctic/NA
   для DBSCAN, а не NA_for_TC для TE). Именно этот файл уже используется для
   этого же TE/NA_for_TC пайплайна в
   CVS_alt_tracking/TempestExtremes/plot_tracking_results_func_2026_upd.py,
   поэтому его пиксельная сетка (и, следовательно, индексы i/j) должна
   совпадать с сеткой, на которой считался DetectNodes/StitchNodes.
   ВАЖНО: это предположение стоит визуально проверить (см. README ниже) —
   если после первого запуска фильтр "над океаном" отсекает подозрительно
   много/мало треков, или карты в get_clusters_with_metrics_TE_ERA5.py
   выглядят смещёнными относительно берега, значит сетка масок не совпадает
   с сеткой индексов i/j, и фильтр нужно будет переписать через lat/lon
   (ближайший узел по .sel(...), а не через целочисленные индексы).
6. Итоговые файлы сохраняются в новую папку max_r2d_point{above_ocean}
   рядом с hourly_data (т.е. внутри .../csv_Tracks..._params/), в ТОМ ЖЕ
   помесячном формате имён 'max_crit_day_CVS_{year}-{month:02d}.csv', что и
   раньше — это специально сделано, чтобы load_season_df() из
   func_for_CVS_clusters.py (используется в get_clusters_with_metrics)
   продолжал работать без изменений, хотя исходные данные больше не лежат
   по папкам месяцев.

Параметр pbl_trop_frac (тропопауза/погранслой) сейчас не считается
(пустой) — он просто пропускается через remove_unnecessary_columns не
будет, а в numeric_cols второго скрипта исключён явно.
"""

import os
import glob

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


# ============================================================
# НАСТРОЙКИ
# ============================================================

path_init = '/storage/thalassa/users/vkoshkina'
path_data = f'{path_init}/data'

# Папка с часовыми точками треков (после add_params_v2_fast_2026-08-24.py)
HOURLY_DATA_DIR = (
    f'{path_data}/TempestExtremes/ERA5/R2D_ERA5_NA_for_TC_850hPa_sigma_2/'
    'csv_Tracks_range_1_5_18h_12h_2010_25points_global_params/hourly_data'
)

# Родительская папка (…_params) — рядом с ней складываем результат отбора максимумов
PARAMS_DIR = os.path.dirname(HOURLY_DATA_DIR)

# Эта папка построена только для 2010 года (год "зашит" в имени постфикса)
YEAR = 2010

above_ocean = '_ocean'
# above_ocean = ''   # раскомментировать эту строку и закомментировать above_ocean='_ocean' выше, чтобы отключить фильтр по океану

ocean_threshold = 0.5  # как для ветки data_type == 'ERA5' в оригинальном скрипте

# Куда сохраняем результат
track_folder_new = f'{PARAMS_DIR}/max_r2d_point{above_ocean}'
os.makedirs(track_folder_new, exist_ok=True)

# Маска суша/океан именно для региона NA_for_TC (см. п.5 в шапке файла)
LSM_FILE = f'{path_data}/ERA5/ERA5_lsm_cropped_NA_for_TC.nc'

if above_ocean == '_ocean':
    _lsm_da = xr.open_dataset(LSM_FILE)['lsm']
    if 'time' in _lsm_da.dims:
        _lsm_da = _lsm_da.isel(time=0)
    ground = np.where(_lsm_da.values > 0.5, 1, np.nan)  # 1 = суша, nan = океан
else:
    ground = None


# ============================================================
# ФУНКЦИИ (в основном без изменений от get_max_crit_df_for_ocean_2026.py,
# кроме process_track_file — см. комментарии внутри)
# ============================================================

def calculate_track_len(df):
    """Длина трека в шагах сетки (в единицах индексов x=i, y=j)."""
    df = df.copy()
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')

    dx = df['x'].diff()
    dy = df['y'].diff()
    step_lengths = np.sqrt(dx**2 + dy**2)
    df['track_len'] = step_lengths.cumsum().fillna(0)
    return df


def calculate_velocity(track, max_idx):
    """
    Скорость смещения центра трека вблизи момента максимума.
    Формула идентична ветке data_type == 'ERA5' в оригинальном скрипте:
    сетка ERA5 0.25°, данные часовые (шаг = 1 час), поэтому она подходит и
    для новых TE-треков без изменений.
    """
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

    diffs = np.diff(track_len[start:end + 1])
    diffs = diffs * np.cos(np.radians(track['lat'].values[start:end]))

    # NB: как и в оригинальном скрипте (ветка ERA5) — результат в м/с
    # (шаг данных = 1 час, поэтому деление на 3600 даёт м/с, а не км/ч,
    # в отличие от LoRes/SMP-веток, где явно умножается на 3.6).
    avg_velocity = (111 * 1000 * 0.25 * np.mean(diffs) / 3600)

    return avg_velocity, start_stop


def is_track_over_ocean(track_df, land_mask, ocean_threshold=0.8):
    """Проверка, проходит ли трек преимущественно над океаном (по индексам x,y=i,j)."""
    if land_mask is None:
        return True

    if 'x' in track_df.columns and 'y' in track_df.columns:
        xs = track_df['x'].values.astype(int)
        ys = track_df['y'].values.astype(int)

        xs = np.clip(xs, 0, land_mask.shape[1] - 1)
        ys = np.clip(ys, 0, land_mask.shape[0] - 1)

        ocean_mask = np.isnan(land_mask[ys, xs])
        ocean_fraction = np.mean(ocean_mask)

        return ocean_fraction >= ocean_threshold
    return True


def remove_unnecessary_columns(df):
    """Удаление служебных колонок, если вдруг где-то сохранились (безвредно, если их нет)."""
    cols_to_drop = [
        'time_ind', 'px1_ind', 'py1_ind', 'px2_ind', 'py2_ind',
        'px3_ind', 'py3_ind', 'semi_major', 'semi_minor',
        'bergeron_2h', 'bergeron_4h', 'bergeron_6h', 'bergeron_12h', 'bergeron_24h',
    ]
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    if existing_cols_to_drop:
        df = df.drop(columns=existing_cols_to_drop)
    return df


def process_track_file(file, land_mask, ocean_threshold):
    """Обработка одного файла трека (адаптировано под схему TE/StitchNodes+add_params)."""
    try:
        df = pd.read_csv(file)
        if df.empty:
            return None
    except Exception as e:
        print(f"Error reading {file}: {e}")
        return None

    # --- унификация имён колонок под общий пайплайн (см. п.2-3 в шапке файла) ---
    rename_map = {}
    if 'r2d' in df.columns:
        rename_map['r2d'] = 'R2D_max'
    elif 'crit' in df.columns:  # на случай, если где-то попадётся старое имя
        rename_map['crit'] = 'R2D_max'

    if 'i' in df.columns and 'j' in df.columns:
        rename_map['i'] = 'x'
        rename_map['j'] = 'y'

    if rename_map:
        df = df.rename(columns=rename_map)

    if 'datetime' not in df.columns:
        if 'time' in df.columns:
            df = df.rename(columns={'time': 'datetime'})
        else:
            return None

    df['datetime'] = pd.to_datetime(df['datetime'])

    if not {'x', 'y'}.issubset(df.columns):
        print(f"В {file} нет колонок индексов сетки (i/j) — пропуск фильтра по океану/длины трека в пикселях")
        return None

    # Фильтрация над океаном
    if land_mask is not None:
        if not is_track_over_ocean(df, land_mask, ocean_threshold):
            return None

    if 'R2D_max' not in df.columns:
        print(f"В {file} нет колонки критерия (r2d/crit) — пропуск")
        return None

    # Вычисление длины трека
    df = calculate_track_len(df)

    # Поиск момента максимальной интенсивности (max r2d)
    max_candidates = df['R2D_max'].iloc[1:]
    if len(max_candidates) == 0:
        return None

    max_idx = max_candidates.idxmax()

    max_row = df.loc[max_idx]
    max_row_df = pd.DataFrame([max_row]).reset_index(drop=True)

    # Добавление метаданных трека
    max_row_df['track_len'] = df['track_len'].iloc[-1]
    max_row_df['duration'] = (df['datetime'].iloc[-1] - df['datetime'].iloc[0]).total_seconds() / 3600

    vel, start_stop = calculate_velocity(df, max_idx)
    max_row_df['velocity'] = vel
    max_row_df['start_stop'] = start_stop

    if 'rad' in df.columns:
        max_row_df['mean_radius'] = df['rad'].mean()

    max_row_df['path'] = file
    max_row_df['name'] = os.path.basename(file)

    return max_row_df


# ============================================================
# ОСНОВНАЯ ОБРАБОТКА (один год, плоская папка — без цикла по месяцам на входе)
# ============================================================

def main():
    files = sorted(glob.glob(f'{HOURLY_DATA_DIR}/*_track_{YEAR}*.csv'))

    if not files:
        print(f"Файлы треков не найдены в {HOURLY_DATA_DIR}")
        return

    print(f"Найдено {len(files)} треков за {YEAR} год")

    results = []
    for file in tqdm(files, desc=f"Processing TE ERA5 {YEAR}"):
        result = process_track_file(file, ground, ocean_threshold)
        if result is not None:
            results.append(result)

    if not results:
        print("Ни один трек не прошёл фильтрацию")
        return

    all_df = pd.concat(results, ignore_index=True)
    all_df = remove_unnecessary_columns(all_df)

    cols = ['datetime'] + [c for c in all_df.columns if c != 'datetime']
    all_df = all_df[cols]

    # Сохраняем ПОМЕСЯЧНО в старом формате имён — чтобы load_season_df()
    # из func_for_CVS_clusters.py читал эти файлы без изменений (см. п.6 в шапке).
    all_df['_month'] = all_df['datetime'].dt.month
    for month, month_df in all_df.groupby('_month'):
        month_df = month_df.drop(columns='_month')
        output_file = f'{track_folder_new}/max_crit_day_CVS_{YEAR}-{month:02d}.csv'
        month_df.to_csv(output_file, index=False)
        print(f"Saved {len(month_df)} records to {output_file}")

    output_path_all = f'{track_folder_new}/all_tracks_max_intensity{above_ocean}.csv'
    all_df.drop(columns='_month').to_csv(output_path_all, index=False)

    print(f"\n{'=' * 50}")
    print(f"Total processed: {len(all_df)} records")
    print(f"Saved to {output_path_all}")
    print(f"{'=' * 50}")
    print(f"\nStatistics:")
    print(f"Average velocity: {all_df['velocity'].mean():.4f} m/s")
    print(f"Average duration: {all_df['duration'].mean():.1f} hours")
    if 'mean_radius' in all_df.columns:
        print(f"Average mean_radius: {all_df['mean_radius'].mean():.3f}")


if __name__ == "__main__":
    main()
