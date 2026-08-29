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

output_dir = f"{path_data_tracks}/cluster_results_upd"

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

for n_clusters in tqdm(range(4,15)):

    colors = plt.cm.get_cmap('tab20', n_clusters)
    
    months = [1, 2, 3]
    df_winter = load_season_df(path_tracks_dir, years, months, 'winter')
    
    months = [7, 8, 9]
    df_summer = load_season_df(path_tracks_dir, years, months, 'summer')
    
    df_winter = df_winter[df_winter['start_stop'] == False]
    df_summer = df_summer[df_summer['start_stop'] == False]

    df_winter['dT'] = df_winter[t_up] - df_winter[t_surf]
    df_summer['dT'] = df_summer[t_up] - df_summer[t_surf]
    
    df_winter, cluster_groups = get_clusters(df_winter, numeric_cols, n_clusters)
    clusters_full_tracks = adding_cluster_tracks(df_winter, cluster_groups)
    
    df_winter.to_csv(f"{output_dir}/cluster_tables/winter_nclusters_{n_clusters}_tracks.csv", index=False)
    
    # plot_tracks_by_clusters(clusters_full_tracks, n_clusters, colors, ground, path_data_tracks, season='winter', data_type=data_type,)
    # plot_tracks_by_clusters(clusters_full_tracks, n_clusters, colors, ground, path_data_tracks, season='winter', data_type=data_type, track_type='start')
    
    plot_tracks_by_clusters_all(clusters_full_tracks, n_clusters, colors, ground, output_dir, season='winter', data_type=data_type, track_type='track')
    # plot_tracks_by_clusters_all(clusters_full_tracks, n_clusters, colors, ground, path_data_tracks, season='winter', data_type=data_type, track_type='start')
    
    # plot_cluster_boxplots(df_winter, numeric_cols, n_clusters, path_data_tracks, season='winter')
    
    plot_cluster_boxplots_all(df_winter, numeric_cols, n_clusters, colors, output_dir, season='winter')
    
    df_summer, cluster_groups = get_clusters(df_summer, numeric_cols, n_clusters)
    clusters_full_tracks = adding_cluster_tracks(df_summer, cluster_groups)
    
    df_summer.to_csv(f"{output_dir}/cluster_tables/summer_nclusters_{n_clusters}_tracks.csv", index=False)
    
    # plot_tracks_by_clusters(clusters_full_tracks, n_clusters, colors, ground, path_data_tracks, season='summer', data_type=data_type,)
    # plot_tracks_by_clusters(clusters_full_tracks, n_clusters, colors, ground, path_data_tracks, season='summer', data_type=data_type,, track_type='start')
    
    plot_tracks_by_clusters_all(clusters_full_tracks, n_clusters, colors, ground, output_dir, season='summer', data_type=data_type, track_type='track')
    # plot_tracks_by_clusters_all(clusters_full_tracks, n_clusters, colors, ground, path_data_tracks, season='summer', data_type=data_type, track_type='start')
    
    # plot_cluster_boxplots(df_summer, numeric_cols, n_clusters, path_data_tracks, season='summer')
    
    plot_cluster_boxplots_all(df_summer, numeric_cols, n_clusters, colors, output_dir, season='summer')