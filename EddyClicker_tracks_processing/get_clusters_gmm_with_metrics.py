import pandas as pd
import xarray as xr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import os
import sys
warnings.filterwarnings('ignore')

path_init = f'/storage/thalassa/users/vkoshkina'
folder = 'scripts/CS_processing/CVS_clustering/'
sys.path.insert(2, f'{path_init}/{folder}')

from func_for_CVS_clusters import *

# Добавляем выбор сезона
print('Выберите сезон (winter/summer/year/all): ')
input_season = input().strip().lower()

# Добавьте выбор метода
print('Выберите метод (gmm/kmeans): ')
clustering_type = input().strip().lower()

# Проверяем валидность ввода
valid_seasons = ['winter', 'summer', 'year', 'all']
if input_season not in valid_seasons:
    raise ValueError(f"Неправильный сезон. Допустимые значения: {valid_seasons}")

path_data_dir = f'{path_init}/data'  

data_type = 'LoRes'  # или 'LoRes', в зависимости от того, что вы сейчас обрабатываете
if data_type == 'LoRes':
    path_data_dir = f'{path_data_dir}/{data_type}'

path_tracks_dir = f'{path_data_dir}/{data_type}/EddyClicker_with_params_list'

params_type = 'max_wspd_day'

    
output_dir = f"{path_data_dir}/{data_type}/EddyClicker_{clustering_type}_results/{params_type}"

os.makedirs(f"{output_dir}/cluster_tables", exist_ok=True)
os.makedirs(f"{output_dir}/cluster_pics", exist_ok=True)

if data_type == 'LoRes':
    path_dir_raw = f'/storage/NAAD/NAAD/LoRes/2010'
    ncfile = f'{path_dir_raw}/wrfout_d01_2010-01-01_00:00:00'
    
    ground_ds = xr.open_dataset(f'{ncfile}')['HGT'][0]
    ground = np.where(ground_ds > 5, 1, np.nan)
elif data_type == 'ERA5':
    path_dir_raw = f'/storage/thalassa/users/vkoshkina/data/ERA5'
    ncfile = f'{path_dir_raw}/ERA5_lsm_cropped.nc'
    
    ground_ds_ERA5 = xr.open_dataset(f'{ncfile}')['var172'][0]
    ground = np.where(ground_ds_ERA5 > 0.5, 1, np.nan)

# ✅ Обновленный process_season
def process_season(season_name, months, n_clusters_range, clustering_type='gmm'):
    """Обработка сезона с универсальной кластеризацией"""
    metrics_list_season = []
    
    for n_clusters in tqdm(n_clusters_range, desc=f"{season_name} ({clustering_type})"):
        colors = plt.cm.get_cmap('tab20', n_clusters)
        
        # Загрузка данных
        df_season = load_season_df_EC(path_tracks_dir, months, season_name, params_type)
        numeric_cols = [col for col in df_season.select_dtypes(include=[np.number]).columns
                       if col not in {'datetime', 'x', 'y'}]
        
        # Кластеризация
        df_season, cluster_groups, gmm_metrics = get_clusters_universal(
            df_season, numeric_cols, n_clusters, clustering_type
        )
        clusters_full_tracks = adding_cluster_tracks(df_season, cluster_groups)
        
        # Визуализация
        plot_tracks_by_clusters_all(clusters_full_tracks, n_clusters, colors, ground, 
                                  output_dir, season=season_name, data_type=data_type)
        plot_cluster_boxplots_all(df_season, numeric_cols, n_clusters, colors, output_dir, 
                                season=season_name)

        X_for_metrics = df_season[numeric_cols].dropna().values  # ← numpy array
        labels_for_metrics = df_season['cluster'].loc[df_season[numeric_cols].dropna().index].values
        
        metrics = compute_metrics(X_for_metrics, labels_for_metrics, gmm_metrics, clustering_type)
        
        metrics.update({
            'n_clusters': n_clusters,
            'season': season_name,
            'clustering_type': clustering_type
        })
        metrics_list_season.append(metrics)
        
        # Сохранение
        df_season.to_csv(f"{output_dir}/cluster_tables/{season_name}_n{n_clusters}_{clustering_type}.csv", 
                        index=False)
    
    return metrics_list_season

# Основной блок выполнения (исправленный)
n_clusters_range = range(2, 15)
metrics_file = f"{output_dir}/clustering_metrics_internal.csv"

if input_season == 'all':
    print("🚀 Запуск последовательной обработки всех сезонов...")
    seasons_config = {
        'winter': {'months': [1, 2, 3]},
        'summer': {'months': [7, 8, 9]},
        'year': {'months': list(range(1, 13))}
    }
    
    all_metrics = []
    for season_name, config in seasons_config.items():
        print(f"🔍 Обработка сезона: {season_name}")
        try:
            season_metrics = process_season(season_name, config['months'], n_clusters_range, clustering_type)
            all_metrics.extend(season_metrics)
            print(f"✅ Сезон {season_name} завершен")
        except Exception as e:
            print(f"❌ Ошибка в сезоне {season_name}: {e}")
            import traceback
            traceback.print_exc()
    
    metrics_df_new = pd.DataFrame(all_metrics)
    
else:
    # Обработка одного сезона
    if input_season == 'winter':
        months = [1, 2, 3]
    elif input_season == 'summer':
        months = [7, 8, 9]
    else:  # year
        months = list(range(1, 13))
    
    print(f"🔍 Обработка сезона: {input_season}")
    metrics_list = process_season(input_season, months, n_clusters_range, clustering_type)
    metrics_df_new = pd.DataFrame(metrics_list)

# ✅ Сохранение метрик
save_metrics(metrics_df_new, input_season, metrics_file)

# ✅ Визуализация (отдельная функция)
plot_clustering_metrics(metrics_df_new, input_season, output_dir, clustering_type)

print(f"✅ Анализ завершен!")