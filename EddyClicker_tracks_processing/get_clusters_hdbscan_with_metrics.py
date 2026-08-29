import pandas as pd
import xarray as xr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import os
import sys
import hdbscan  # Добавлен импорт HDBSCAN
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

output_dir = f"{path_data_dir}/{data_type}/EddyClicker_HDBSCAN_results/{params_type}"  # Изменено название для HDBSCAN
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


def hdbscan_clustering(df, numeric_cols, min_cluster_size=5, min_samples=3):
    """HDBSCAN аналогично get_clusters: preprocess + кластеризация + group_cyclones"""
    
    # Предобработка (обязательна для HDBSCAN!)
    X_reduced, scaler, pca, valid_indices = preprocess_data(df, numeric_cols)
    
    # HDBSCAN кластеризатор
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean'
    )
    
    # Предсказание кластеров (-1 = шум)
    clusters = clusterer.fit_predict(X_reduced)
    
    # Маппинг кластеров (исключаем шум -1, начинаем с 0)
    unique_labels = sorted(set(clusters) - {-1})
    cluster_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    n_clusters = len(unique_labels)
    
    mapped_clusters = np.array([cluster_mapping.get(label, -1) for label in clusters])
    
    # Создаем df с кластерами (аналогично get_clusters)
    df_result = df.copy()
    df_result['cluster'] = np.nan  # Инициализируем NaN
    df_result.loc[df_result.index[valid_indices], 'cluster'] = mapped_clusters
    
    print(f"HDBSCAN: {n_clusters} кластеров (mcs={min_cluster_size}, ms={min_samples})")
    print(df_result['cluster'].value_counts().sort_index().dropna())
    
    # Группируем циклоны (аналогично get_clusters)
    cluster_groups = group_cyclones(df_result, df_result['cluster'].values)
    
    return df_result, cluster_groups

def process_season(season_name, months, hdbscan_params_list):
    """Функция для обработки одного сезона с HDBSCAN"""
    metrics_list_season = []
    
    for params in tqdm(hdbscan_params_list, desc=f"{season_name.capitalize()} HDBSCAN"):
        min_cluster_size, min_samples = params
        
        # Загружаем данные для сезона
        df_season = load_season_df_EC(path_tracks_dir, months, season_name, params_type)
        
        # Берём только числовые колонки, кроме x, y
        numeric_cols = [
            col for col in df_season.select_dtypes(include=[np.number]).columns
            if col not in {'datetime', 'x', 'y'}
        ]
        
        # HDBSCAN кластеризация
        df_season, cluster_groups = hdbscan_clustering(
            df_season, numeric_cols, min_cluster_size, min_samples
        )
        

        n_clusters = len(set(df_season['cluster'].dropna().unique()) - {-1})
        
        clusters_full_tracks = adding_cluster_tracks(df_season, cluster_groups)

        if n_clusters > 1:
            # Визуализация (используем n_clusters для совместимости)
            colors = plt.cm.get_cmap('tab20', n_clusters)
            plot_tracks_by_clusters_all(clusters_full_tracks, n_clusters, colors, ground, output_dir, 
                                      season=season_name, data_type=data_type, track_type='track', 
                                        clustering_type='hdbscan', clustering_params=[min_cluster_size, min_samples])
            plot_cluster_boxplots_all(df_season, numeric_cols, n_clusters, colors, output_dir, season=season_name, 
                                      clustering_type='hdbscan', clustering_params=[min_cluster_size, min_samples])
            
            # Расчет метрик (только для точек, принадлежащих кластерам)
            valid_mask = df_season['cluster'].notna() & (df_season['cluster'] >= 0)
            if valid_mask.sum() < 10:  # Минимум точек для метрик
                print(f"⚠️ Недостаточно точек для метрик в {season_name}")
                continue
                
            X_season = df_season.loc[valid_mask, numeric_cols].values
            labels_season = df_season.loc[valid_mask, 'cluster'].astype(int).values
            
            sil_season = compute_silhouette(X_season, labels_season)
            conn_season = compute_connectivity(X_season, labels_season, L=20)
            hubert_season = compute_huberts_g_statistic(X_season, labels_season)
            stab_season = compute_stability(X_season, labels_season, n_subsamples=20, n_clusters=n_clusters)
            
            # Сохраняем метрики
            metrics_list_season.append({
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'season': season_name,
                'silhouette': sil_season,
                'connectivity': conn_season,
                'hubert_g': hubert_season,
                'stability': stab_season
            })
            
            # Сохранение результатов
            df_season.to_csv(f"{output_dir}/cluster_tables/{season_name}_mcs{min_cluster_size}_ms{min_samples}_tracks.csv", index=False)
    
    return metrics_list_season

def smart_hdbscan_params(n_points):
    log_n = int(np.log(n_points)) + 1  # DBSCAN правило
    
    # 3 уровня детализации
    return [
        (log_n*2, log_n),      # Базовый (много кластеров)
        (log_n*3, log_n*2),  # Средний
        (log_n*4, log_n*2)     # Консервативный (мало кластеров)
    ]
    
# Основной блок выполнения
# Параметры для HDBSCAN вместо range n_clusters: (min_cluster_size, min_samples)
# hdbscan_params_list = [
#     (5, 3), (5, 5), (5, 7), 
#     (7, 3), (7, 5), (7, 7), (7, 10), (7, 12),
#     (10, 3), (10, 5), (10, 7), (10, 10), (10, 15),
#     (15, 3), (15, 5), (15, 7), (15, 10), (15, 15), (15, 20),
#     (20, 3), (20, 5), (20, 7), (20, 10), (20, 15), (20, 20),   
# ]

n_points = 1161
hdbscan_params_list = smart_hdbscan_params(n_points)

metrics_file = f"{output_dir}/hdbscan_clustering_metrics.csv"  # Изменено название файла

if input_season == 'all':
    print("🚀 Запуск последовательной обработки всех сезонов с HDBSCAN...")
    
    # Определяем параметры для каждого сезона
    seasons_config = {
        'winter': {'months': [1, 2, 3]},
        'summer': {'months': [7, 8, 9]},
        'year': {'months': list(range(1, 13))}
    }
    
    all_metrics = []
    
    # Последовательная обработка
    for season_name, config in seasons_config.items():
        print(f"🔍 Обработка сезона: {season_name}")
        try:
            season_metrics = process_season(season_name, config['months'], hdbscan_params_list)
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
    
    print(f"🔍 Обработка сезона: {input_season} с HDBSCAN")
    metrics_list = process_season(input_season, months, hdbscan_params_list)
    metrics_df_new = pd.DataFrame(metrics_list)

# Сохранение/обновление файла метрик
if os.path.exists(metrics_file):
    metrics_df_existing = pd.read_csv(metrics_file)
    
    if input_season == 'all':
        metrics_df_combined = metrics_df_new
        print("✅ Все сезоны пересчитаны HDBSCAN, файл метрик обновлен")
    else:
        # Фильтруем существующие данные по параметрам
        mask_existing = (
            (metrics_df_existing['min_cluster_size'].isin([p[0] for p in hdbscan_params_list])) &
            (metrics_df_existing['min_samples'].isin([p[1] for p in hdbscan_params_list])) &
            (metrics_df_existing['season'] == input_season)
        )
        metrics_df_existing = metrics_df_existing[~mask_existing]
        metrics_df_combined = pd.concat([metrics_df_existing, metrics_df_new], ignore_index=True)
        print(f"✅ HDBSCAN метрики дозаписаны для сезона: {input_season}")
    
    metrics_df_combined.to_csv(metrics_file, index=False)
else:
    metrics_df_new.to_csv(metrics_file, index=False)
    print(f"✅ Создан новый файл HDBSCAN метрик для сезона: {input_season}")

# Визуализация метрик (адаптирована для HDBSCAN)
sns.set_style("whitegrid")

if input_season == 'all':
    seasons_to_plot = ['winter', 'summer', 'year']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    plt.figure(figsize=(16, 12))
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
            # Группируем по min_cluster_size для гладкой линии
            pivot_data = data_season.pivot_table(
                values=metric, index='min_cluster_size', columns='min_samples'
            )
            for col in pivot_data.columns:
                plt.scatter(pivot_data.index, pivot_data[col], 
                           label=f'{season_name} (ms={col})', color=color, alpha=0.7, s=100)
        
        plt.xlabel('Min Cluster Size')
        plt.ylabel(ylabel)
        plt.title(f'{title} - HDBSCAN All Seasons')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/hdbscan_metrics_plot_all_seasons.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ HDBSCAN графики метрик сохранены для всех сезонов")
    
else:
    plt.figure(figsize=(16, 12))
    metrics_names = ['silhouette', 'connectivity', 'hubert_g', 'stability']
    titles = ['Silhouette Width', 'Connectivity (L=20)', "Hubert's G Statistic", 'Stability (ARI)']
    ylabs = ['Silhouette Score', 'Connectivity', "Hubert's G", 'Adjusted Rand Index (Stability)']

    data = metrics_df_new
    
    for i, (metric, title, ylabel) in enumerate(zip(metrics_names, titles, ylabs)):
        plt.subplot(2, 2, i + 1)
        
        # Scatter plot по параметрам HDBSCAN
        scatter = plt.scatter(data['min_cluster_size'], data[metric], 
                             c=data['min_samples'], cmap='viridis', s=100, alpha=0.8)
        plt.colorbar(scatter, label='Min Samples')
        
        plt.xlabel('Min Cluster Size')
        plt.ylabel(ylabel)
        plt.title(f'{title} - {input_season.capitalize()} HDBSCAN')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/hdbscan_metrics_plot_{input_season}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ HDBSCAN графики метрик сохранены для сезона: {input_season}")

print(f"✅ HDBSCAN анализ завершен!")
