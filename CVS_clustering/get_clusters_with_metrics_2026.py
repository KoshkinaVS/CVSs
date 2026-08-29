import pandas as pd
import xarray as xr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings('ignore')

from func_for_CVS_clusters import *
from func_for_metrics import *

path_init = f'/storage/thalassa/users/vkoshkina'

# # Добавляем выбор сезона
# print('Выберите сезон (winter/summer/year/all): ')
# input_season = input().strip().lower()

# # Проверяем валидность ввода
# valid_seasons = ['winter', 'summer', 'year', 'all']
# if input_season not in valid_seasons:
#     raise ValueError(f"Неправильный сезон. Допустимые значения: {valid_seasons}")

input_season = 'year'

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
    
    path_data_tracks = f'{path_data}/LoRes/LoRes/LoRes_tracks/LoRes_tracks_params_mattiew_2026-05-13'
    path_tracks_dir = f'{path_data_tracks}/max_wspd_day_ocean'

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
        'pvo_max', 'helicity_max', 
        'updraft_helicity_max', 
        'pw_max', 
        'slp_delta'
    ]


    numeric_cols = [ 
        'mean_radius',
         'crit',
         'track_len',
         # 'SLP_delta',
         'SLP_diff(cent-95)',
         'U10_mean',
         # 'U500_mean',
         # 'U500_poleward',
         # 'PV_850_mean',
         'PV_500_mean',
         # 'wspd_850_mean',
         # 'T2_mean',
         # 'T500_mean',
         # 'T700_mean',
         # 'T850_mean',
         # 'T2_minus_T500_mean',
         # 'T2_minus_T700_mean',
         # 'TH2_mean',
         # 'TH2_minus_theta_500_mean',
         # 'TH2_minus_theta_700_mean',
         # 'TH2_minus_theta_850_mean',
         # 'SST_mean',
         # 'theta_e_700_mean',
         # 'theta_e_850_mean',
         # 'TH850',
         'SST_minus_T500_mean',
         # 'SST_minus_T700_mean',
         # 'theta_SST_minus_theta_500_mean',
         # 'theta_SST_minus_theta_700_mean',
         # 'theta_SST_minus_theta_850_mean',
         'theta_e_SST_minus_theta_e_500_mean',
         # 'theta_e_SST_minus_theta_e_700_mean',
         # 'theta_e_SST_minus_theta_e_850_mean',
         'MCAO1_500_mean',
         # 'MCAO1_700_mean',
         # 'MCAO2_mean',
         'grad_theta_e_850_mean',
         # 'T2_delta',
         # 'T850_delta',
         # 'T700_delta',
         # 'T500_delta',
         # 'TH500_delta',
         # 'TH700_delta',
         # 'TH850_delta',
         'PBL_med',
         # 'rel_vor_850_med',
         # 'trop_height',
         # 'theta_trop_med',
         # 'delta_theta_trop-theta_sst',
         # 'pressure_trop',
         # 'wspd_trop',
         'HFX_rad',
         'LH_rad',
         'mucape_95',
         'mcin_95',
         'helicity_95',
         'pw_95',
         # 'pw_sum',
         # 'w_925',
         'w_850',
         # 'w_500',
         'N_500',
         # 'differential_wind_vector',
         # 'vertical_shear_strength',
         # 'alpha_d',
         # 'alpha_p',
         # 'vertical_shear_angle',
         # 'vertical_shear_vector_u',
         # 'vertical_shear_vector_v',
         'wind_shear_10m_500',
         'T2_disp',
         # 'SST_disp',
         # 'TH850_disp',
         # 'RAIN_HOURLY_sum',
         'RAIN_HOURLY_95',
         'george_index',
         # 'LI_rad',
         # 'z500_rad',
         # 'U200_95',
         # 'Q850_95',
         # 'LCL_95',
         # 'LFC_95',
         # 'LFC-LCL',
         'DBZ_sfc_500_mean',
         # 'rh_95',
         # 'updraft_helicity',
         'duration',
         'vel',
                   ]
    
elif data_type == 'ERA5':
    path_data_tracks = f'{path_data}/TC_tracks/{data_type}/{data_type}_sigma_{sigma}'
    path_tracks_dir = f'{path_data_tracks}/max_crit_day_data_ocean'
    numeric_cols.append('tp_median')

output_dir = f"{path_data_tracks}/cluster_results_KMEANS"

params_type = 'max_crit_day'
output_dir = f"{path_data}/TC_tracks/{data_type}/{data_type}_sigma_{sigma}/LoRes_tracks_params_mattiew_2026-05-25/{clustering_type}_results/{scaler_type}_scaler/{params_type}"


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

def process_season(season_name, months, n_clusters_range, clustering_type='gmm', scaler_type='standart'):
    """Обработка сезона с универсальной кластеризацией"""
    metrics_list_season = []
    
    for n_clusters in tqdm(n_clusters_range, desc=f"{season_name} ({clustering_type})"):
        colors = plt.cm.get_cmap('tab20', n_clusters)
        
        # Загружаем данные для сезона
        df_season = load_season_df(path_tracks_dir, years, months, season_name)
        df_season = df_season[df_season['start_stop'] == False]
        # df_season['dT'] = df_season[t_up] - df_season[t_surf]
        
        # Кластеризация
        # df_season, cluster_groups = get_clusters(df_season, numeric_cols, n_clusters)

        # Кластеризация
        df_season, cluster_groups, gmm_metrics = get_clusters_universal(
            df_season, numeric_cols, n_clusters, clustering_type=clustering_type, scaler_type=scaler_type
        )
        
        clusters_full_tracks = adding_cluster_tracks(df_season, cluster_groups)
        
        # # Визуализация
        # plot_tracks_by_clusters_all(clusters_full_tracks, n_clusters, colors, ground, output_dir, 
        #                           season=season_name, data_type=data_type, track_type='track')
        
        plot_tracks_by_clusters_all_with_TC(
                                        clusters_full_tracks,
                                        n_clusters,
                                        colors,
                                        ground,
                                        output_dir,
                                        season=season_name,
                                        data_type=data_type,
                                        clustering_type=clustering_type,
                                        params_type=params_type,
                                        plot_TC=False,
                                        plot_monthly_dist=True, # new
                                        monthly_mode='absolute',
                                        df_season=df_season, # new
                                        # track_to_cyclone=track_to_pmc
                                    )
        
        plot_cluster_boxplots_all(df_season, numeric_cols, n_clusters, colors, output_dir, season=season_name)
        
        # Расчет метрик
        X_for_metrics = df_season[numeric_cols].dropna().values  # ← numpy array
        labels_for_metrics = df_season['cluster'].loc[df_season[numeric_cols].dropna().index].values
        
        metrics = compute_metrics(X_for_metrics, labels_for_metrics, gmm_metrics, clustering_type,
                                 compute_classifiability_flag=False,
                                 compute_reproducibility_flag=False)
        
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
n_clusters_range = range(2, 16)
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
    metrics_list = process_season(input_season, months, n_clusters_range, clustering_type=clustering_type, scaler_type=scaler_type)
    metrics_df_new = pd.DataFrame(metrics_list)

# ✅ Сохранение метрик
save_metrics(metrics_df_new, input_season, metrics_file)

# ✅ Визуализация (отдельная функция)
plot_clustering_metrics(metrics_df_new, input_season, output_dir, clustering_type)

print(f"✅ Анализ для {scaler_type} {clustering_type} завершен!")