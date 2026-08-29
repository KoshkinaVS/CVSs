import pandas as pd
import xarray as xr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from func_for_CVS_clusters import *

path_init = f'/storage/thalassa/users/vkoshkina'

# Добавляем выбор сезона
print('Выберите сезон (winter/summer/year): ')
input_season = input().strip().lower()

# Проверяем валидность ввода
valid_seasons = ['winter', 'summer', 'year']
if input_season not in valid_seasons:
    raise ValueError(f"Неправильный сезон. Допустимые значения: {valid_seasons}")

path_init = f'/storage/thalassa/users/vkoshkina'


tracking_type = 'tracking_local_2_phase'
pref_tracking = 'all_points_bound'
CVS_speed = 'adv_speed'
circ = 'C'


years = np.arange(1979, 2019)
path_data = f'{path_init}/data'  
results_dir = f"{tracking_type}_{CVS_speed}_{pref_tracking}"


if data_type == 'LoRes':
    time_th = 8
    t_up = 'theta_median'
    t_surf = 't2_median'
    lh = 'mslhf_max'
    sh = 'msshf_max'

    # t_surf = 'T2_median'
    # lh = 'LH_max'
    # sh = 'HXF_max'

elif data_type == 'ERA5':
    time_th = 24
    t_up = 't_median'
    t_surf = 't2m_median'
    lh = 'mlhf_max'
    sh = 'mshf_max'
    
    

numeric_cols = [
                    'rad', 'crit', 
                    'track_len',
                    'duration',
                    'vel',
                    'msl_min', 'wspd_max', 
                    lh, sh, 
                    t_surf, 'dT',
                    'w_median'
                    ]


    
if data_type == 'LoRes':
    path_data_tracks = f'{path_data}/TC_tracks/{data_type}/{data_type}_sigma_{sigma}'
    path_tracks_dir = f'{path_data_tracks}/max_crit_day_data_ocean'

    numeric_cols = [
                    'rad', 'crit', 
                    'track_len',
                    'duration',
                    'vel',
                    'msl_min', 'wspd_max', 
                    lh, sh, 
                    t_surf, 'dT',
                    'w_median',

                    'cape_2d_max', 
                    # 'cape_3d_max', 
                    'pvo_max', 'helicity_max', 
                    'updraft_helicity_max', 'pw_max', 
                    'slp_delta'
                ]
    
elif data_type == 'ERA5':
    # path_data_tracks = f"{path_data}/TC_tracks/{data_type}/my_tracking_results/{pref_tracking}/{results_dir}/{data_type}_tracks_sigma_{sigma}"
    # path_tracks_dir = f'{path_data_tracks}/max_crit_day_data'

    path_data_tracks = f'{path_data}/TC_tracks/{data_type}/{data_type}_sigma_{sigma}'
    path_tracks_dir = f'{path_data_tracks}/max_crit_day_data_ocean'

    numeric_cols.append('tp_median')

output_dir = f"{path_data_tracks}/cluster_results_august"

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


# Список для сбора метрик
metrics_list = []

for n_clusters in tqdm(range(4, 15)):
    colors = plt.cm.get_cmap('tab20', n_clusters)
    
    if input_season == 'year':
        # Загружаем данные за весь год
        months = list(range(1, 13))
        df_year = load_season_df(path_tracks_dir, years, months, 'year')
        df_year = df_year[df_year['start_stop'] == False]
        df_year['dT'] = df_year[t_up] - df_year[t_surf]
        
        # Кластеризация для всего года
        df_year, cluster_groups = get_clusters(df_year, numeric_cols, n_clusters)
        clusters_full_tracks = adding_cluster_tracks(df_year, cluster_groups)
        
        # Визуализация
        plot_tracks_by_clusters_all(clusters_full_tracks, n_clusters, colors, ground, output_dir, 
                                  season='year', data_type=data_type, track_type='track')
        plot_cluster_boxplots_all(df_year, numeric_cols, n_clusters, colors, output_dir, season='year')
        
        # Расчет метрик
        X_year = df_year[numeric_cols].dropna().values
        labels_year = df_year['cluster'].dropna().values
        
        valid_idx = ~np.isnan(labels_year)
        X_year = X_year[valid_idx]
        labels_year = labels_year[valid_idx].astype(int)
        
        sil_year = compute_silhouette(X_year, labels_year)
        conn_year = compute_connectivity(X_year, labels_year, L=20)
        hubert_year = compute_huberts_g_statistic(X_year, labels_year)
        stab_year = compute_stability(X_year, labels_year, n_subsamples=20, n_clusters=n_clusters)
        
        # Сохраняем метрики
        metrics_list.append({
            'n_clusters': n_clusters,
            'season': 'year',
            'silhouette': sil_year,
            'connectivity': conn_year,
            'hubert_g': hubert_year,
            'stability': stab_year
        })
        
        # Сохранение результатов
        df_year.to_csv(f"{output_dir}/cluster_tables/year_nclusters_{n_clusters}_tracks.csv", index=False)
        
    else:
        # Оригинальный код для зимы/лета
        if input_season == 'winter':
            months = [1, 2, 3]
            df_season = load_season_df(path_tracks_dir, years, months, 'winter')
        else:  # summer
            months = [7, 8, 9]
            df_season = load_season_df(path_tracks_dir, years, months, 'summer')
        
        df_season = df_season[df_season['start_stop'] == False]
        df_season['dT'] = df_season[t_up] - df_season[t_surf]
        
        # Кластеризация
        df_season, cluster_groups = get_clusters(df_season, numeric_cols, n_clusters)
        clusters_full_tracks = adding_cluster_tracks(df_season, cluster_groups)
        
        # Визуализация
        plot_tracks_by_clusters_all(clusters_full_tracks, n_clusters, colors, ground, output_dir, 
                                  season=input_season, data_type=data_type, track_type='track')
        plot_cluster_boxplots_all(df_season, numeric_cols, n_clusters, colors, output_dir, season=input_season)
        
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
        metrics_list.append({
            'n_clusters': n_clusters,
            'season': input_season,
            'silhouette': sil_season,
            'connectivity': conn_season,
            'hubert_g': hubert_season,
            'stability': stab_season
        })
        
        # Сохранение результатов
        df_season.to_csv(f"{output_dir}/cluster_tables/{input_season}_nclusters_{n_clusters}_tracks.csv", index=False)

# Преобразуем в DataFrame
metrics_df_new = pd.DataFrame(metrics_list)
metrics_file = f"{output_dir}/clustering_metrics_internal.csv"

# Проверяем, существует ли файл метрик
if os.path.exists(metrics_file):
    # Загружаем существующие данные
    metrics_df_existing = pd.read_csv(metrics_file)
    
    # Удаляем строки для текущего сезона (если они уже есть)
    metrics_df_existing = metrics_df_existing[metrics_df_existing['season'] != input_season]
    
    # Объединяем старые данные с новыми
    metrics_df_combined = pd.concat([metrics_df_existing, metrics_df_new], ignore_index=True)
    
    # Сохраняем объединенный файл
    metrics_df_combined.to_csv(metrics_file, index=False)
    print(f"✅ Метрики дозаписаны в существующий файл для сезона: {input_season}")
else:
    # Создаем новый файл
    metrics_df_new.to_csv(metrics_file, index=False)
    print(f"✅ Создан новый файл метрик для сезона: {input_season}")

# Визуализация метрик (адаптированная для одного сезона)
sns.set_style("whitegrid")
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
    
    # Если анализируем один сезон
    data = metrics_df_new[metrics_df_new['season'] == input_season]
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

print(f"✅ Анализ завершен для сезона: {input_season}")
print(f"✅ Графики метрик кластеризации сохранены.")