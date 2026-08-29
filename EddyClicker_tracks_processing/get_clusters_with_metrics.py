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

    
output_dir = f"{path_data_dir}/{data_type}/EddyClicker_KMeansConstrained_results/{params_type}"

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

def process_season(season_name, months, n_clusters_range):
    """Функция для обработки одного сезона"""
    metrics_list_season = []
    
    for n_clusters in tqdm(n_clusters_range, desc=f"{season_name.capitalize()} clusters"):
        colors = plt.cm.get_cmap('tab20', n_clusters)
        
        # Загружаем данные для сезона
        df_season = load_season_df_EC(path_tracks_dir, months, season_name, params_type)

        
        # df_season = df_season[df_season['start_stop'] == False]
        # df_season['dT'] = df_season[t_up] - df_season[t_surf]


        # Берём только числовые колонки, кроме x, y
        numeric_cols = [
            col for col in df_season.select_dtypes(include=[np.number]).columns
            if col not in {'datetime', 'x', 'y'}
        ]
        
        # Кластеризация
        df_season, cluster_groups = get_clusters_KMeansConstrained(df_season, numeric_cols, n_clusters)
        clusters_full_tracks = adding_cluster_tracks(df_season, cluster_groups)
        
        # Визуализация
        plot_tracks_by_clusters_all(clusters_full_tracks, n_clusters, colors, ground, output_dir, 
                                  season=season_name, data_type=data_type, track_type='track')
        plot_cluster_boxplots_all(df_season, numeric_cols, n_clusters, colors, output_dir, season=season_name)
        
        # Расчет метрик
        X_season = df_season[numeric_cols].dropna().values
        labels_season = df_season['cluster'].dropna().values
        
        valid_idx = ~np.isnan(labels_season)
        X_season = X_season[valid_idx]
        labels_season = labels_season[valid_idx].astype(int)
        
        sil_season = compute_silhouette(X_season, labels_season)
        conn_season = compute_connectivity(X_season, labels_season, L=20)
        hubert_season = compute_huberts_g_statistic(X_season, labels_season)
        stab_season = compute_stability(X_season, labels_season, n_subsamples=20, n_clusters=n_clusters)
        
        # Сохраняем метрики
        metrics_list_season.append({
            'n_clusters': n_clusters,
            'season': season_name,
            'silhouette': sil_season,
            'connectivity': conn_season,
            'hubert_g': hubert_season,
            'stability': stab_season
        })
        
        # Сохранение результатов
        df_season.to_csv(f"{output_dir}/cluster_tables/{season_name}_nclusters_{n_clusters}_tracks.csv", index=False)
    
    return metrics_list_season

# Основной блок выполнения
n_clusters_range = range(2, 15)
metrics_file = f"{output_dir}/clustering_metrics_internal.csv"

if input_season == 'all':
    print("🚀 Запуск последовательной обработки всех сезонов...")
    
    # Определяем параметры для каждого сезона
    seasons_config = {
        'winter': {'months': [1, 2, 3]},
        'summer': {'months': [7, 8, 9]},
        'year': {'months': list(range(1, 13))}
    }
    
    all_metrics = []
    
    # Последовательная обработка вместо параллельной
    for season_name, config in seasons_config.items():
        print(f"🔍 Обработка сезона: {season_name}")
        try:
            season_metrics = process_season(season_name, config['months'], n_clusters_range)
            all_metrics.extend(season_metrics)
            print(f"✅ Сезон {season_name} завершен")
        except Exception as e:
            print(f"❌ Ошибка в сезоне {season_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Создаем DataFrame со всеми метриками
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
    metrics_list = process_season(input_season, months, n_clusters_range)
    metrics_df_new = pd.DataFrame(metrics_list)

# Сохранение/обновление файла метрик
if os.path.exists(metrics_file):
    # Загружаем существующие данные
    metrics_df_existing = pd.read_csv(metrics_file)
    
    if input_season == 'all':
        # Удаляем все существующие данные, так как мы пересчитали все сезоны
        metrics_df_combined = metrics_df_new
        print("✅ Все сезоны пересчитаны, файл метрик обновлен")
    else:
        # Удаляем строки для текущего сезона (если они уже есть)
        metrics_df_existing = metrics_df_existing[metrics_df_existing['season'] != input_season]
        # Объединяем старые данные с новыми
        metrics_df_combined = pd.concat([metrics_df_existing, metrics_df_new], ignore_index=True)
        print(f"✅ Метрики дозаписаны в существующий файл для сезона: {input_season}")
    
    # Сохраняем объединенный файл
    metrics_df_combined.to_csv(metrics_file, index=False)
else:
    # Создаем новый файл
    metrics_df_new.to_csv(metrics_file, index=False)
    if input_season == 'all':
        print("✅ Создан новый файл метрик для всех сезонов")
    else:
        print(f"✅ Создан новый файл метрик для сезона: {input_season}")

# Визуализация метрик
sns.set_style("whitegrid")

if input_season == 'all':
    # Визуализация для всех сезонов
    seasons_to_plot = ['winter', 'summer', 'year']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Синий, оранжевый, зеленый
    
    # Также создадим сводный график со всеми сезонами
    plt.figure(figsize=(16, 10))
    metrics_names = ['silhouette', 'connectivity', 'hubert_g', 'stability']
    titles = [
        'Silhouette Width',
        'Connectivity (L=20)',
        "Hubert's G Statistic",
        'Stability (ARI)'
    ]
    ylabs = [
        'Silhouette Score',
        'Connectivity',
        "Hubert's G",
        'Adjusted Rand Index (Stability)'
    ]

    for i, (metric, title, ylabel) in enumerate(zip(metrics_names, titles, ylabs)):
        plt.subplot(2, 2, i + 1)
        for season_name, color in zip(seasons_to_plot, colors):
            data_season = metrics_df_new[metrics_df_new['season'] == season_name]
            plt.plot(data_season['n_clusters'], data_season[metric], 
                     marker='o', label=season_name.capitalize(), 
                     color=color, linewidth=2)

        plt.xlabel('Number of Clusters')
        plt.ylabel(ylabel)
        plt.title(f'{title} - All Seasons')
        plt.xticks(n_clusters_range)
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/clustering_metrics_plot_all_seasons.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Отдельные графики для каждого сезона
    for season_name in seasons_to_plot:
        plt.figure(figsize=(16, 10))
        
        for i, (metric, title, ylabel) in enumerate(zip(metrics_names, titles, ylabs)):
            plt.subplot(2, 2, i + 1)
            data_season = metrics_df_new[metrics_df_new['season'] == season_name]
            plt.plot(data_season['n_clusters'], data_season[metric], 
                     marker='o', label=season_name.capitalize(), 
                     color=colors[seasons_to_plot.index(season_name)], linewidth=2)

            plt.xlabel('Number of Clusters')
            plt.ylabel(ylabel)
            plt.title(f'{title} - {season_name.capitalize()}')
            plt.xticks(data_season['n_clusters'].unique())
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/clustering_metrics_plot_{season_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print("✅ Графики метрик кластеризации сохранены для всех сезонов")
    
else:
    # Визуализация для одного сезона
    plt.figure(figsize=(16, 10))
    metrics_names = ['silhouette', 'connectivity', 'hubert_g', 'stability']
    titles = [
        'Silhouette Width',
        'Connectivity (L=20)',
        "Hubert's G Statistic",
        'Stability (ARI)'
    ]
    ylabs = [
        'Silhouette Score',
        'Connectivity',
        "Hubert's G",
        'Adjusted Rand Index (Stability)'
    ]

    data = metrics_df_new[metrics_df_new['season'] == input_season]
    
    for i, (metric, title, ylabel) in enumerate(zip(metrics_names, titles, ylabs)):
        plt.subplot(2, 2, i + 1)
        plt.plot(data['n_clusters'], data[metric], 
                 marker='o', label=input_season.capitalize(), 
                 color='#1f77b4', linewidth=2)

        plt.xlabel('Number of Clusters')
        plt.ylabel(ylabel)
        plt.title(f'{title} - {input_season.capitalize()}')
        plt.xticks(data['n_clusters'].unique())
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/clustering_metrics_plot_{input_season}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Графики метрик кластеризации сохранены для сезона: {input_season}")

print(f"✅ Анализ завершен!")