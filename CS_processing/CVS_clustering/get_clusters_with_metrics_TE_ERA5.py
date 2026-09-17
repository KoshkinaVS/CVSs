"""
Кластеризация треков из НОВОГО альтернативного пайплайна трекинга
(TempestExtremes/StitchNodes по критерию R2D на 850 гПа), адаптация
get_clusters_with_metrics_2026-08-11.py.

Ожидается, что перед запуском этого скрипта уже отработал
get_max_crit_df_for_ocean_TE_ERA5.py — он кладёт помесячные файлы
max_crit_day_CVS_{year}-{month:02d}.csv в PATH_TRACKS_DIR ниже
(см. переменную track_folder_new там же).

Что изменено относительно оригинального get_clusters_with_metrics_2026-08-11.py
----------------------------------------------------------------------------
1. Новая ветка данных вместо data_type in {'LoRes', 'ERA5', 'SMP'} — путь
   ведёт прямо в папку, которую создаёт get_max_crit_df_for_ocean_TE_ERA5.py
   (см. п.6 в шапке того файла), без месячных подпапок на входе.
2. numeric_cols — это ровно 12 параметров из RESULT_PARAMS
   ERA5/add_params_v2_fast_2026-08-24.py (за вычетом pbl_trop_frac —
   пока пустой, исключён по договорённости) + R2D_max/mean_radius/velocity,
   которые считает get_max_crit_df_for_ocean_TE_ERA5.py.
3. Один год (2010, зашит в имени папки-источника).
4. Маска суша/океан для карт — ERA5_lsm_cropped_NA_for_TC.nc ('lsm'), тот
   же файл, что и в get_max_crit_df_for_ocean_TE_ERA5.py, а не
   ERA5_lsm_cropped.nc (var172) из старого DBSCAN-пайплайна.
5. Для отрисовки полных треков (adding_cluster_tracks) добавлена локальная
   версия adding_cluster_tracks_TE(), которая переименовывает 'i'/'j' в
   'x'/'y' (и 'r2d'->'R2D_max') — она читает СЫРЫЕ файлы из hourly_data по
   колонке 'path', в них ещё нет переименований, которые делает
   get_max_crit_df_for_ocean_TE_ERA5.py для итоговой таблицы максимумов.
   Остальные функции отрисовки (plot_tracks_by_clusters_all_with_TC и т.д.)
   импортируются из func_for_CVS_clusters.py без изменений — им передаётся
   data_type='ERA5' (буквально), чтобы сохранить те же условности, что и в
   старом ERA5-пайплайне: инверсия оси Y (т.к. x/y — это пиксельные
   индексы i/j на растре ERA5, где широта убывает вниз по строкам).
"""

import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings('ignore')

from func_for_CVS_clusters import *          # get_clusters_universal, plot_*, group_cyclones, load_season_df, ...
from func_for_metrics_upd import *           # compute_metrics, save_metrics, plot_clustering_metrics


# ============================================================
# НАСТРОЙКИ
# ============================================================

path_init = '/storage/thalassa/users/vkoshkina'
path_data = f'{path_init}/data'

# Должно ТОЧНО совпадать с YEAR и above_ocean в get_max_crit_df_for_ocean_TE_ERA5.py
YEAR = 2010
years = np.arange(YEAR, YEAR + 1)
months = np.arange(1, 13, 1)

above_ocean = '_ocean'
# above_ocean = ''

# Папка, которую создаёт get_max_crit_df_for_ocean_TE_ERA5.py
PARAMS_DIR = (
    f'{path_data}/TempestExtremes/ERA5/R2D_ERA5_NA_for_TC_850hPa_sigma_2/'
    'csv_Tracks_range_1_5_18h_12h_2010_25points_global_params'
)
path_tracks_dir = f'{PARAMS_DIR}/max_r2d_point{above_ocean}'

input_season = 'year'  # используем весь год (короткий ряд — сезонное деление не показательно)

print('Выберите метод кластеризации:')
print('1. KMeans')
print('2. GMM')
print('(по умолчанию KMeans): ', end='')
choice1 = input().strip()
clustering_type = 'kmeans' if choice1 in ['1', ''] else 'gmm'

print('Выберите метод нормализации:')
print('1. StandardScaler')
print('2. RobustScaler')
print('(по умолчанию StandardScaler): ', end='')
choice2 = input().strip()
scaler_type = 'standard' if choice2 in ['1', ''] else 'robust'

# 12 параметров из add_params_v2_fast_2026-08-24.py::RESULT_PARAMS минус
# pbl_trop_frac (пока пустой) + величины, которые считает
# get_max_crit_df_for_ocean_TE_ERA5.py (R2D_max, mean_radius, velocity).
numeric_cols = [
    'R2D_max',
    'mean_radius',
    'velocity',
    'SLP_diff_cent_95',
    'U10_mean',
    'U850_mean',
    'U500_U850_frac',
    'U500_minus_U850',
    'PV_850_mean',
    'T2_minus_T850_mean',
    'T850_disp',
    # 'pbl_trop_frac',  # исключено: параметр пока не считается (пустой)
    'w_850',
    'RH_850',
    'RAIN_HOURLY_sum',
]

params_type = 'max_r2d_point'

output_dir = f"{path_tracks_dir}/{clustering_type}_results_with_pics/{scaler_type}_scaler"
os.makedirs(f"{output_dir}/cluster_tables", exist_ok=True)
os.makedirs(f"{output_dir}/cluster_pics", exist_ok=True)

# Маска суша/океан для карт (тот же файл, что и в get_max_crit_df_for_ocean_TE_ERA5.py)
LSM_FILE = f'{path_data}/ERA5/ERA5_lsm_cropped_NA_for_TC.nc'
_lsm_da = xr.open_dataset(LSM_FILE)['lsm']
if 'time' in _lsm_da.dims:
    _lsm_da = _lsm_da.isel(time=0)
ground = np.where(_lsm_da.values > 0.5, 1, np.nan)


# ============================================================
# Локальная версия adding_cluster_tracks — читает СЫРЫЕ файлы из
# hourly_data (колонка 'path'), где ещё нет переименований i/j->x/y,
# r2d->R2D_max (см. п.5 в шапке файла).
# ============================================================

import re


def adding_cluster_tracks_TE(df, cluster_groups):
    clusters_full_tracks = {cluster: [] for cluster in cluster_groups.keys()}

    for cluster in cluster_groups.keys():
        print(f"Загрузка треков для кластера {cluster}...")
        for idx in tqdm(cluster_groups[cluster]):
            path = df.iloc[idx]['path']
            name = df.iloc[idx]['name']

            try:
                full_track = pd.read_csv(path)
            except Exception:
                print(f"Не удалось загрузить файл: {path}")
                continue

            numbers = re.findall(r'\d+', name.replace('.csv', ''))
            if numbers:
                track_id = int(numbers[-1])
            else:
                print(f"⚠️ Warning: No digits found in filename: {name}")
                continue

            rename_map = {}
            if 'r2d' in full_track.columns:
                rename_map['r2d'] = 'R2D_max'
            elif 'crit' in full_track.columns:
                rename_map['crit'] = 'R2D_max'
            if 'i' in full_track.columns and 'j' in full_track.columns:
                rename_map['i'] = 'x'
                rename_map['j'] = 'y'
            if rename_map:
                full_track = full_track.rename(columns=rename_map)

            if 'datetime' not in full_track.columns:
                if 'time' in full_track.columns:
                    full_track = full_track.rename(columns={'time': 'datetime'})

            full_track['track_id'] = track_id
            clusters_full_tracks[cluster].append(full_track)

    return clusters_full_tracks


# ============================================================
# ОБРАБОТКА
# ============================================================

def process_season(season_name, months, n_clusters_range, clustering_type='gmm', scaler_type='standard', plot_tracks_boxplots=True):
    metrics_list_season = []

    df_season = load_season_df(path_tracks_dir, years, months, season_name)

    n_initial = len(df_season)

    # ---------- 1. Фильтрация start_stop ----------
    df_season = df_season[df_season['start_stop'] == False]
    n_after_start_stop = len(df_season)
    print(
        f'После start_stop: {n_after_start_stop} строк '
        f'(удалено {n_initial - n_after_start_stop}, '
        f'{(n_initial - n_after_start_stop) / n_initial * 100:.2f}%)'
    )

    # ---------- 2. Удаление NaN по параметрам кластеризации ----------
    df_season = df_season.dropna(subset=numeric_cols)
    n_after_nan = len(df_season)
    print(
        f'После dropna: {n_after_nan} строк '
        f'(удалено {n_after_start_stop - n_after_nan}, '
        f'{(n_after_start_stop - n_after_nan) / n_after_start_stop * 100:.2f}%)'
    )
    print(f'Итого осталось: {n_after_nan} из {n_initial} ({n_after_nan / n_initial * 100:.2f}%)')

    numeric_cols_filt = [c for c in numeric_cols if df_season[c].notna().all()]
    print(f'всего {len(numeric_cols_filt)} параметров для кластеризации (не None)')
    print('Параметры:', numeric_cols_filt)

    for n_clusters in tqdm(n_clusters_range, desc=f"{season_name} ({clustering_type})"):
        colors = plt.cm.get_cmap('tab20', n_clusters)

        df_season_clustered, cluster_groups, gmm_metrics = get_clusters_universal(
            df_season, numeric_cols, n_clusters, clustering_type=clustering_type, scaler_type=scaler_type
        )

        df_season_clustered.to_csv(
            f"{output_dir}/cluster_tables/{season_name}_n{n_clusters}_{clustering_type}_{scaler_type}.csv",
            index=False,
        )

        if plot_tracks_boxplots:
            clusters_full_tracks = adding_cluster_tracks_TE(df_season_clustered, cluster_groups)

            plot_tracks_by_clusters_all_with_TC(
                clusters_full_tracks,
                n_clusters,
                colors,
                ground,
                output_dir,
                season=season_name,
                data_type='ERA5',  # буквально 'ERA5' — сохраняет инверсию оси Y для пиксельных x/y=i/j
                clustering_type=clustering_type,
                params_type=params_type,
                plot_TC=False,
                plot_monthly_dist=True,
                monthly_mode='absolute',
                df_season=df_season_clustered,
            )

            plot_cluster_boxplots_all(df_season_clustered, numeric_cols, n_clusters, colors, output_dir, season=season_name)

        print('Computing metrics>>>')

        X_for_metrics = df_season_clustered[numeric_cols].dropna().values
        labels_for_metrics = df_season_clustered['cluster'].loc[df_season_clustered[numeric_cols].dropna().index].values

        metrics = compute_metrics(
            X_for_metrics, labels_for_metrics, gmm_metrics, clustering_type,
            compute_classifiability_flag=False,
            compute_reproducibility_flag=False,
        )

        metrics.update({
            'n_clusters': n_clusters,
            'season': season_name,
            'clustering_type': clustering_type,
        })
        metrics_list_season.append(metrics)

    return metrics_list_season


n_clusters_range = range(2, 16)
metrics_file = f"{output_dir}/clustering_metrics_internal_{scaler_type}.csv"

months_full = list(range(1, 13))
print(f"🔍 Обработка: {input_season}")
metrics_list = process_season(input_season, months_full, n_clusters_range, clustering_type=clustering_type, scaler_type=scaler_type, plot_tracks_boxplots=True)
metrics_df_new = pd.DataFrame(metrics_list)

save_metrics(metrics_df_new, input_season, metrics_file)
plot_clustering_metrics(metrics_df_new, input_season, output_dir, clustering_type, scaler_type)

print(f"✅ Анализ для {scaler_type} {clustering_type} завершён!")
