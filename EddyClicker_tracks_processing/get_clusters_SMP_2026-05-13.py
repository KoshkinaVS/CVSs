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
from func_for_metrics import *

# # Добавляем выбор сезона
# print('Выберите сезон (winter/summer/year/all): ')
# input_season = input().strip().lower()
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

# Проверяем валидность ввода
valid_seasons = ['winter', 'summer', 'year', 'all']
if input_season not in valid_seasons:
    raise ValueError(f"Неправильный сезон. Допустимые значения: {valid_seasons}")

path_data_dir = f'{path_init}/data'  

data_type = 'LoRes'  # 
data_type = 'SMP'  # 


if data_type == 'LoRes':
    path_data_dir = f'{path_data_dir}/{data_type}'


params_type = 'max_wspd_day'
if data_type == 'LoRes':
    tracks_params_folder = 'EddyClicker_tracks_Egor_2010_params_r2d'
elif data_type == 'SMP':
    tracks_params_folder = 'EddyClicker_tracks_Egor_2010_params_r2d'
    /storage/kubrick/nikitenko/recalc_testing_3012/hourly_data
    
# tracks_params_folder = 'EddyClicker_tracks_Egor_params_r2d'
# tracks_params_folder = 'EddyClicker_tracks_params_mattiew'

path_tracks_dir = f'{path_data_dir}/{data_type}/{tracks_params_folder}/{params_type}'

    
output_dir = f"{path_data_dir}/{data_type}/EddyClicker_clustering_r2d/{scaler_type}_scaler/EddyClicker_{clustering_type}_results_2010_small_v2/{params_type}"

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

track_to_cyclone = {
    12: {'cyclone': 'DANIELLE', 'type': 'Hurricane'},
    14: {'cyclone': 'EARL', 'type': 'Hurricane'},
    66: {'cyclone': 'IGOR', 'type': 'Hurricane'},
    383: {'cyclone': 'FIONA', 'type': 'Tropical Storm'},
    1090: {'cyclone': 'BONNIE', 'type': 'Part 1', 'note': 'Часть циклона'},
    1102: {'cyclone': 'BONNIE', 'type': 'Part 2', 'note': 'Часть циклона'},
    723: {'cyclone': 'HERMINE', 'type': 'Part 1', 'note': 'Часть циклона'},
    732: {'cyclone': 'HERMINE', 'type': 'Part 2', 'note': 'Часть циклона'},
    710: {'cyclone': 'HERMINE', 'type': 'Part 3', 'note': 'Часть циклона'},
    707: {'cyclone': 'HERMINE', 'type': 'Part 4', 'note': 'Часть циклона'},
    757: {'cyclone': 'FIVE', 'type': 'Tropical Depression'},
    38: {'cyclone': 'UNKNOWN', 'type': 'Tropical'},
    752: {'cyclone': 'UNKNOWN', 'type': 'Tropical'},
    362: {'cyclone': 'UNKNOWN', 'type': 'Tropical'}
}

track_to_pmc = {
    1102: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL032010'},  # перекрытие 30.0 ч, dist=94.1 км
    1157: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL122010'},  # перекрытие 72.0 ч, dist=98.2 км
    1168: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL092010'},  # перекрытие 93.0 ч, dist=190.0 км
    1172: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL122010'},  # перекрытие 24.0 ч, dist=153.0 км
    1178: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL142010'},  # перекрытие 135.0 ч, dist=117.3 км
    1183: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL152010'},  # перекрытие 27.0 ч, dist=91.2 км
    12: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL062010'},  # перекрытие 267.0 ч, dist=188.1 км
    1294: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL172010'},  # перекрытие 96.0 ч, dist=101.4 км
    14: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL072010'},  # перекрытие 291.0 ч, dist=104.3 км
    1497: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL192010'},  # перекрытие 84.0 ч, dist=119.4 км
    1623: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL212010'},  # перекрытие 114.0 ч, dist=189.5 км
    1624: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL212010'},  # перекрытие 126.0 ч, dist=137.4 км
    1631: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL202010'},  # перекрытие 18.0 ч, dist=178.1 км
    1727: {'база': 'ERA5 tracks (zenodo)', 'тип': 'ПМЦ', 'id_в_базе': '458'},  # перекрытие 18.0 ч, dist=151.2 км
    1729: {'база': 'Cyclone infos (zenodo)', 'тип': 'ПМЦ', 'id_в_базе': '457'},  # перекрытие 22.8 ч, dist=160.9 км
    1877: {'база': 'Climatology NH (Stoll et al.)', 'тип': 'ПМЦ', 'id_в_базе': '120101106120'},  # перекрытие 47.0 ч, dist=134.9 км
    2130: {'база': 'Rojo et al. 2019 (PANGAEA)', 'тип': 'ПМЦ', 'id_в_базе': '143.g'},  # перекрытие 24.1 ч, dist=120.0 км
    2314: {'база': 'ERA5 tracks (zenodo)', 'тип': 'ПМЦ', 'id_в_базе': '479'},  # перекрытие 39.0 ч, dist=162.7 км
    2358: {'база': 'Cyclone infos (zenodo)', 'тип': 'ПМЦ', 'id_в_базе': '481'},  # перекрытие 31.9 ч, dist=145.8 км
    276: {'база': 'Climatology NH (Stoll et al.)', 'тип': 'ПМЦ', 'id_в_базе': '120100112610'},  # перекрытие 22.0 ч, dist=139.3 км
    38: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL062010'},  # перекрытие 24.0 ч, dist=61.1 км
    383: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL082010'},  # перекрытие 81.0 ч, dist=139.3 км
    491: {'база': 'Climatology NH (Stoll et al.)', 'тип': 'ПМЦ', 'id_в_базе': '120100200570'},  # перекрытие 29.0 ч, dist=174.4 км
    500: {'база': 'Climatology NH (Stoll et al.)', 'тип': 'ПМЦ', 'id_в_базе': '120100200210'},  # перекрытие 25.0 ч, dist=180.1 км
    537: {'база': 'ERA5 tracks (zenodo)', 'тип': 'ПМЦ', 'id_в_базе': '419'},  # перекрытие 47.0 ч, dist=164.4 км
    568: {'база': 'Cyclone infos (zenodo)', 'тип': 'ПМЦ', 'id_в_базе': '424'},  # перекрытие 18.1 ч, dist=73.7 км
    644: {'база': 'Climatology NH (Stoll et al.)', 'тип': 'ПМЦ', 'id_в_базе': '120100213970'},  # перекрытие 42.0 ч, dist=165.9 км
    66: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL112010'},  # перекрытие 351.0 ч, dist=92.7 км
    710: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL042010'},  # перекрытие 39.0 ч, dist=189.4 км
    732: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL042010'},  # перекрытие 33.0 ч, dist=168.2 км
    757: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL052010'},  # перекрытие 105.0 ч, dist=134.0 км
    77: {'база': 'Climatology NH (Stoll et al.)', 'тип': 'ПМЦ', 'id_в_базе': '120100903050'},  # перекрытие 21.0 ч, dist=177.0 км
}

# Определяем агрегации для каждого параметра
param_aggregations = {
    # '.............stolls...........'
    #pressures
    # 'SLP_center': ['mean', 'min', 'max'],
    'R2d_max': ['mean', 'min', 'max'],
    
    'SLP_median': ['mean', 'min', 'max'],
    # 'SLP_95': ['mean', 'min', 'max'],
    # 'SLP_diff(cent-med)': ['mean', 'min', 'max'],
    'SLP_diff(cent-95)': ['mean', 'min', 'max'],
    'PV_850_mean_cyclone': ['mean', 'min', 'max'],
    #temps
    'T500_mean': ['mean', 'min', 'max'],
    'T700_mean': ['mean', 'min', 'max'],
    'SST_mean': ['mean', 'min', 'max'],
    'theta_e_700_mean': ['mean', 'min', 'max'],
    'theta_e_850_mean': ['mean', 'min', 'max'],
    'SST_minus_T500_mean': ['mean', 'min', 'max'],
    'SST_minus_T700_mean': ['mean', 'min', 'max'],
    'theta_SST_minus_theta_500_mean': ['mean', 'min', 'max'],
    'theta_SST_minus_theta_700_mean': ['mean', 'min', 'max'], 
    'theta_SST_minus_theta_850_mean': ['mean', 'min', 'max'],
    'theta_e_SST_minus_theta_e_500_mean': ['mean', 'min', 'max'],
    'theta_e_SST_minus_theta_e_700_mean': ['mean', 'min', 'max'],
    'theta_e_SST_minus_theta_e_850_mean': ['mean', 'min', 'max'],
    'MCAO1_500_mean': ['mean', 'min', 'max'],
    'MCAO1_700_mean': ['mean', 'min', 'max'],
    'MCAO2_mean': ['mean', 'min', 'max'],
    'grad_theta_e_850_mean': ['mean', 'min', 'max'],
    'rel_vor_850_med': ['mean', 'min', 'max'],
    #wspeeds
    'U10_mean_cyclone': ['mean', 'min', 'max'],
    'U500_mean': ['mean', 'min', 'max'],
    # 'U500_poleward_cyclone': ['mean', 'min', 'max'],
    #trop
    'trop_height': ['mean', 'min', 'max'],
    'theta_trop_med': ['mean', 'min', 'max'],
    'delta_theta_trop-theta_sst': ['mean', 'min', 'max'],   
    'pressure_trop': ['mean', 'min', 'max'],
    'wspd_trop': ['mean', 'min', 'max'],
    
    # '..............custom params.............'
    'HFX_rad': ['mean', 'min', 'max'], 
    'LH_rad': ['mean', 'min', 'max'], 
    'mucape_95': ['mean', 'min', 'max'],
    'cape_sur': ['mean', 'min', 'max'],
    'helicity_rad': ['mean', 'min', 'max'], 
    'pw_rad': ['mean', 'min', 'max'],
    'w_10m_mean': ['mean', 'min', 'max'], 
    # 'SLP_diff_cyclone': ['mean', 'min', 'max'],
    'N_mean': ['mean', 'min', 'max'],
    'propagation_speed': ['mean', 'min', 'max'],
    'differential_wind_vector': ['mean', 'min', 'max'],
    'vertical_shear_strength': ['mean', 'min', 'max'],
    'alpha_d': ['mean', 'min', 'max'],
    'alpha_p': ['mean', 'min', 'max'],
    'vertical_shear_angle': ['mean', 'min', 'max'],
    'vertical_shear_vector_u': ['mean', 'min', 'max'],
    'vertical_shear_vector_v': ['mean', 'min', 'max'],
    'RAIN_HOURLY': ['mean', 'min', 'max'],
    'PBL_med': ['mean', 'min', 'max'],
    #dispersions
    'T850': ['mean', 'min', 'max'],
    'T850_disp': ['mean', 'min', 'max'],
    'T2': ['mean', 'min', 'max'], 
    'T2_disp': ['mean', 'min', 'max'],
    'SST_disp': ['mean', 'min', 'max'],
    #Bergeron indexes
    # 'bergeron_2h': ['mean', 'min', 'max'],
    # 'bergeron_4h': ['mean', 'min', 'max'],
    # 'bergeron_6h': ['mean', 'min', 'max'],
    # 'bergeron_12h': ['mean', 'min', 'max'],
    # 'bergeron_24h': ['mean', 'min', 'max'],
    #others from list
    'george_index': ['mean', 'min', 'max'],
    'LI_rad': ['mean', 'min', 'max'],
    'z500_rad': ['mean', 'min', 'max'],
    'U850_95': ['mean', 'min', 'max'],
    'U200_95': ['mean', 'min', 'max'],
    'Q850_95': ['mean', 'min', 'max'],
    
    # 'LCL_95':['mean', 'min', 'max'],
    # 'LFC_95':['mean', 'min', 'max'],
    # 'LFC-LCL':['mean', 'min', 'max'],
    
    'mcin_95':['mean', 'min', 'max'],
    'cin_sur': ['mean', 'min', 'max'],
    
    'DBZ_sfc_500_mean':['mean', 'min', 'max'],
    'rh_95':['mean', 'min', 'max'],
    'updraft_helicity':['mean', 'min', 'max'],
    'INTEGR_VAPOR_TRANSP':['mean', 'min', 'max'],
    'U_VAPOR_TRANS':['mean', 'min', 'max'],
    'V_VAPOR_TRANS':['mean', 'min', 'max'],
     
    }


# Определяем агрегации для каждого параметра
param_aggregations = {
    'R2D_max': ['mean', 'min', 'max'],
    'SLP_diff(cent-95)': ['mean', 'min', 'max'],
    # 'PV_850_mean_cyclone': ['mean', 'min', 'max'],
    'theta_e_SST_minus_theta_e_500_mean': ['mean', 'min', 'max'],
    'MCAO1_500_mean': ['mean', 'min', 'max'],
    'grad_theta_e_850_mean': ['mean', 'min', 'max'],
    'U10_mean_cyclone': ['mean', 'min', 'max'],
    # 'U500_mean': ['mean', 'min', 'max'],
    'HFX_rad': ['mean', 'min', 'max'], 
    'LH_rad': ['mean', 'min', 'max'], 
    'mucape_95': ['mean', 'min', 'max'],
    'helicity_rad': ['mean', 'min', 'max'], 
    'pw_rad': ['mean', 'min', 'max'],
    'vel': ['mean', 'min', 'max'],
    'differential_wind_vector': ['mean', 'min', 'max'],
    'PBL_med': ['mean', 'min', 'max'],
    'mean_radius': ['mean', 'min', 'max'],
    # 'track_len': ['mean', 'min', 'max'],
    'T2': ['mean', 'min', 'max'], 
    'T2_disp': ['mean', 'min', 'max'], 
    'george_index': ['mean', 'min', 'max'],
    'LI_rad': ['mean', 'min', 'max'],
    }




param_cols = list(param_aggregations.keys())

param_cols = ['LH_rad',
              'helicity_95',
              # 'propagation_speed',
              'PBL_med',
              'pw_sum',
              'w_500',
              # 'alpha_p',
              'RAIN_HOURLY_sum',
              # 'RAIN_HOURLY_95',
              'N_500',
              'TH2_minus_theta_700_mean',
              'SST_mean',
              'HFX_rad',
              'V_VAPOR_TRANS',
              'T2_mean',
              'R2D_max',
              'U500_mean',
              'DBZ_sfc_500_mean',
              # 'LFC_95',
              # 'T2_minus_T700_mean',
              'T2_delta',
              'PV_500_mean',
              'track_len',
              # 'U_VAPOR_TRANS',
              # 'LCL_95',
              'updraft_helicity',
              # 'SLP_diff(cent-med)',
              'SLP_diff(cent-95)',
              # 'duration',
              'mean_radius',
              'vel',
              # 'INTEGR_VAPOR_TRANSP',
              # 'SLP_center',
              # 'SLP_delta',
              # 'T700_mean',
              # 'TH700_delta',
              # 'T500_delta',
              # 'U10_mean',
              # 'theta_e_SST_minus_theta_e_700_mean',
              # 'eccentricity',
              # 'U200_95',
              # 'SST_minus_T500_mean'
             ]


param_cols = [
                'track_len', 'mean_radius', 'R2D_max', 'vel',
                'helicity_95', 'PV_500_mean',
                'U10_mean', 'w_500', # досчитать максимальные
                'LH_rad', 'HFX_rad', # максимальные?
                'T2_mean', 'T2_delta', # + disp - почему не досчиталась?
                'SLP_diff(cent-95)',
                'pw_95', 
                'mucape_95',
                'wind_shear_10m_500',
    
             ]

print(f'всего {len(param_cols)} параметров для кластеризации')


def process_season(season_name, months, n_clusters_range, clustering_type='gmm', scaler_type='standart'):
    metrics_list_season = []
    
    for n_clusters in tqdm(n_clusters_range, desc=f"{season_name} ({clustering_type})"):
        colors = plt.cm.get_cmap('tab20', n_clusters)
        
        # Загрузка данных
        df_season = load_season_df_EC(path_tracks_dir, months, season_name, params_type)

        # ---------- 1. выкидываем строки без vertical_shear_strength ----------
        n_before = len(df_season)
        mask_bad = df_season['vertical_shear_strength'].isna()
        n_bad = mask_bad.sum()

        if n_bad > 0:
            print(f'⚠️ строк без vertical_shear_strength: {n_bad} (из {n_before})')
            df_season = df_season[~mask_bad]
        else:
            print('✅ нет строк с отсутствующим vertical_shear_strength')

        # ---------- 2. NaN → 0 для выбранных параметров ----------
        # какие поля хотим заполнять нулями (пример)
        fill0_cols = [
            'mucape_95', 'mcin_95',
            # добавь сюда остальные нужные параметры
        ]
        # оставляем только реально существующие в df_season
        fill0_cols = [c for c in fill0_cols if c in df_season.columns]
        if fill0_cols:
            df_season[fill0_cols] = df_season[fill0_cols].fillna(0)
            print(f'NaN → 0 для колонок: {fill0_cols}')

        # ---------- 3. дальше как у тебя ----------
        numeric_cols = [c for c in param_cols if df_season[c].notna().all()]
        print(f'всего {len(numeric_cols)} параметров для кластеризации (не None)')  
        print('Параметры:', numeric_cols)

        df_season, cluster_groups, gmm_metrics = get_clusters_universal(
            df_season, numeric_cols, n_clusters,
            clustering_type=clustering_type, scaler_type=scaler_type
        )
        
        # df_season['is_tropical'] = df_season['track_id'].isin(tropical_ids)

        clusters_full_tracks = adding_cluster_tracks(df_season, cluster_groups)

        # треки, для которых есть метки ТЦ/ПМЦ
        all_marked_ids = set(track_to_pmc.keys())
        
        # треки, реально присутствующие в clusters_full_tracks
        present_ids = set()
        for clust, tracks in clusters_full_tracks.items():
            for tr in tracks:
                # tr — это DataFrame; нужно взять уникальные значения track_id
                present_ids.update(tr['track_id'].unique())
        
        # кто из словаря есть в данных сезона
        present = all_marked_ids & present_ids
        missing = all_marked_ids - present_ids
        
        print('Всего в словаре:', len(all_marked_ids))
        print('Есть в clusters_full_tracks:', len(present), sorted(present))
        print('Нет в clusters_full_tracks:', len(missing), sorted(missing))

        # Визуализация
        # plot_tracks_by_clusters_all(clusters_full_tracks, n_clusters, colors, ground, 
        #                           output_dir, season=season_name, data_type=data_type)

        plot_tracks_by_clusters_all_with_TC(
                                        clusters_full_tracks,
                                        n_clusters,
                                        colors,
                                        ground,
                                        output_dir,
                                        season=season_name,
                                        data_type=data_type,
                                        clustering_type=clustering_type,
                                        plot_TC=True,
                                        plot_monthly_dist=True, # new
                                        monthly_mode='absolute',
                                        df_season=df_season, # new
                                        track_to_cyclone=track_to_pmc
                                    )
        
        plot_cluster_boxplots_all(df_season, numeric_cols, n_clusters, colors, output_dir, 
                                season=season_name)

        X_for_metrics = df_season[numeric_cols].dropna().values  # ← numpy array
        labels_for_metrics = df_season['cluster'].loc[df_season[numeric_cols].dropna().index].values
        
        metrics = compute_metrics(X_for_metrics, labels_for_metrics, gmm_metrics, clustering_type, 
                                    compute_classifiability_flag=True,
                                    compute_reproducibility_flag=True)
        
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
    metrics_list = process_season(input_season, months, n_clusters_range, clustering_type=clustering_type, scaler_type=scaler_type)
    metrics_df_new = pd.DataFrame(metrics_list)

# ✅ Сохранение метрик
save_metrics(metrics_df_new, input_season, metrics_file)

# ✅ Визуализация (отдельная функция)
plot_clustering_metrics(metrics_df_new, input_season, output_dir, clustering_type)

print(f"✅ Анализ для {scaler_type} {clustering_type} завершен!")