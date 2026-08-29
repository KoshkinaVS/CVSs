import pandas as pd
import xarray as xr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# from scipy.stats import gaussian_kde



from tqdm import tqdm

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from func_for_CVS_clusters import *


path_init = f'/storage/thalassa/users/vkoshkina'

# data_type = 'LoRes'
# sigma = 2

# print('n_clusters: ')
# n_clusters = int(input())

tracking_type = 'tracking_local_2_phase'
pref_tracking = 'all_points_bound'
CVS_speed = 'adv_speed'
circ = 'C'


# n_clusters = 10
# n_clusters = 7



years = np.arange(1979, 2019)
path_data = f'{path_init}/data'  
results_dir = f"{tracking_type}_{CVS_speed}_{pref_tracking}"


if data_type == 'LoRes':
    time_th = 8
    t_up = 'theta_median'
    t_surf = 't2_median'
    lh = 'mslhf_max'
    sh = 'msshf_max'

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
    
    months = [1, 2, 3]
    df_winter = load_season_df(path_tracks_dir, years, months, 'winter')
    
    months = [7, 8, 9]
    df_summer = load_season_df(path_tracks_dir, years, months, 'summer')
    
    df_winter = df_winter[df_winter['start_stop'] == False]
    df_summer = df_summer[df_summer['start_stop'] == False]

    df_winter['dT'] = df_winter[t_up] - df_winter[t_surf]
    df_summer['dT'] = df_summer[t_up] - df_summer[t_surf]
    
    # === Зима ===
    df_winter, cluster_groups = get_clusters(df_winter, numeric_cols, n_clusters)
    clusters_full_tracks = adding_cluster_tracks(df_winter, cluster_groups)
    
    plot_tracks_by_clusters_all(clusters_full_tracks, n_clusters, colors, ground, output_dir, season='winter', data_type=data_type, track_type='track')

    plot_cluster_boxplots_all(df_winter, numeric_cols, n_clusters, colors, output_dir, season='winter')

    
    # Подготовка данных для метрик
    X_winter = df_winter[numeric_cols].dropna().values
    labels_winter = df_winter['cluster'].dropna().values

    # Убедимся, что нет NaN в метках
    valid_idx = ~np.isnan(labels_winter)
    X_winter = X_winter[valid_idx]
    labels_winter = labels_winter[valid_idx].astype(int)

    # Расчёт метрик для зимы
    sil_winter = compute_silhouette(X_winter, labels_winter)
    conn_winter = compute_connectivity(X_winter, labels_winter, L=20)
    hubert_winter = compute_huberts_g_statistic(X_winter, labels_winter)
    stab_winter = compute_stability(X_winter, labels_winter, n_subsamples=20, n_clusters=n_clusters)

    # === Лето ===
    df_summer, cluster_groups = get_clusters(df_summer, numeric_cols, n_clusters)
    clusters_full_tracks = adding_cluster_tracks(df_summer, cluster_groups)

    # Для лета
    plot_tracks_by_clusters_all(clusters_full_tracks, n_clusters, colors, ground, output_dir, season='summer', data_type=data_type, track_type='track')
    plot_cluster_boxplots_all(df_summer, numeric_cols, n_clusters, colors, output_dir, season='summer')

    
    X_summer = df_summer[numeric_cols].dropna().values
    labels_summer = df_summer['cluster'].dropna().values

    valid_idx = ~np.isnan(labels_summer)
    X_summer = X_summer[valid_idx]
    labels_summer = labels_summer[valid_idx].astype(int)

    sil_summer = compute_silhouette(X_summer, labels_summer)
    conn_summer = compute_connectivity(X_summer, labels_summer, L=20)
    hubert_summer = compute_huberts_g_statistic(X_summer, labels_summer)
    stab_summer = compute_stability(X_summer, labels_summer, n_subsamples=20, n_clusters=n_clusters)

    # Сохраняем метрики
    metrics_list.append({
        'n_clusters': n_clusters,
        'season': 'winter',
        'silhouette': sil_winter,
        'connectivity': conn_winter,
        'hubert_g': hubert_winter,
        'stability': stab_winter
    })

    metrics_list.append({
        'n_clusters': n_clusters,
        'season': 'summer',
        'silhouette': sil_summer,
        'connectivity': conn_summer,
        'hubert_g': hubert_summer,
        'stability': stab_summer
    })

    # === Сохранение результатов кластеризации ===
    df_winter.to_csv(f"{output_dir}/cluster_tables/winter_nclusters_{n_clusters}_tracks.csv", index=False)
    df_summer.to_csv(f"{output_dir}/cluster_tables/summer_nclusters_{n_clusters}_tracks.csv", index=False)




# Преобразуем в DataFrame и сохраняем
metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv(f"{output_dir}/clustering_metrics_internal.csv", index=False)


# # === Визуализация метрик кластеризации ===
# metrics_df = pd.read_csv(f"{output_dir}/clustering_metrics_internal.csv")

# Устанавливаем стиль
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

# Цвета для сезонов
color_map = {'winter': '#1f77b4', 'summer': '#ff7f0e'}

for i, (metric, title, ylabel) in enumerate(zip(metrics_names, titles, ylabs)):
    plt.subplot(2, 2, i + 1)
    
    for season in ['winter', 'summer']:
        data = metrics_df[metrics_df['season'] == season]
        plt.plot(data['n_clusters'], data[metric], 
                 marker='o', label=season.capitalize(), 
                 color=color_map[season], linewidth=2)

    plt.xlabel('Number of Clusters')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(data['n_clusters'].unique())
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_dir}/clustering_metrics_plot.png", dpi=300, bbox_inches='tight')
# plt.close()

print("✅ Графики метрик кластеризации сохранены.")