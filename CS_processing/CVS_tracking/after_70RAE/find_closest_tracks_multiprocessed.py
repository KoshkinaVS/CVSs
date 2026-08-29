from pathlib import Path
import sys
import os
from concurrent.futures import ProcessPoolExecutor
import itertools

# Инициализация путей и параметров
path_init = f'/storage/thalassa/users/vkoshkina'
folder = 'scripts/CS_processing/CVS_tracking/after_70RAE/'
sys.path.insert(2, f'{path_init}/{folder}')

from step_of_tracking import *
from func_for_find_closest_tracks import *

# Configuration parameters
tracking_types = ['tracking_local_2_phase', 
                  'tracking_global_only', 
                  'tracking_local_global',
                  'tracking_local_only', 
                 ]
speed_options = ['adv_speed', 'no_speed', 'bg_speed', 'adv_bg_speed']
pref_tracking = 'all_points_bound'

local_extr_name = 'local_extr_crit'

dist_m, our_level, level, x_unit, y_unit, u_unit, v_unit, time_unit, level_unit, crit, th, min_samples, eps, CS_points_th = get_init_params(data_type, json_file, print_info=True)

time_th = 3
similar_steps = 3
max_time_diff = 3  # Максимальная разница по времени
max_distance_km = 5*dist_m/1000  # Максимальная допустимая дистанция (км)
max_distance_km = 3*dist_m/1000  # Максимальная допустимая дистанция (км)

# max_distance_km = 300  # Максимальная допустимая дистанция (км)


years = np.arange(1979,2019)
# years = np.arange(2010,2011)
months = np.arange(1,13,1)
# months = np.arange(1,2,1)

path_dir_data = f'{path_init}/data'
if data_type == 'LoRes':
    path_dir_data = f'{path_dir_data}/LoRes'

path_data = f'{path_init}/data'
folder_NOAA = 'TC_tracks/splitted'

def process_configuration(config):
    tracking_type, CVS_speed = config
    
    DBSCAN_name = f'DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_level_{level}_smoothing_sigma_{sigma}'
    folder_name = f'{data_type}_sigma_{sigma}/{tracking_type}_{CVS_speed}_{pref_tracking}'

    if data_type == 'HiRes':
        folder_NAAD = f'{path_data}/TC_tracks/my_tracking_results/{pref_tracking}/{tracking_type}_{CVS_speed}_{pref_tracking}/{data_type}_tracks_sigma_{sigma}/'
    elif data_type == 'ERA5':
        folder_NAAD = f'{path_data}/TC_tracks/{data_type}/my_tracking_results_mini/{pref_tracking}/{tracking_type}_{CVS_speed}_{pref_tracking}/{data_type}_tracks_sigma_{sigma}/'
    else:
        if sigma == 0:
            folder_NAAD = f"{path_dir_data}/{data_type}/{data_type}_tracks_sigma_{sigma}/{pref_tracking}/{tracking_type}_{CVS_speed}_{pref_tracking}"
        else:
            folder_NAAD = f'{path_dir_data}/{data_type}/{data_type}_tracks/{pref_tracking}/{tracking_type}_{CVS_speed}_{pref_tracking}/{DBSCAN_name}'
    path_tracks_dir = f'{folder_NAAD}/tracks_{circ}'
    
    for year in years:
        CS_tracks_list_NAAD = []
        CS_tracks_list_NOAA = []
        CS_tracks_list_NAAD = load_season_tracks(CS_tracks_list_NAAD, year, months, path_tracks_dir, time_th=time_th)
        CS_tracks_list_NOAA = load_season_tracks_NOAA(CS_tracks_list_NOAA, f'{path_data}/{folder_NOAA}', year, time_th=time_th)
        
        TC_dict = {
            CS_track['Id'].values[0]: {
                'NOAA_track': CS_track,
                'NAAD_tracks': [],
                'dist_btwn_tracks': []
            }
            for CS_track in CS_tracks_list_NOAA
        }
        
        for key in tqdm(TC_dict.keys(), desc=f'Processing {tracking_type} {CVS_speed} {year}'):
            CS_track_NOAA = TC_dict[key]['NOAA_track']
            NOAA_times = CS_track_NOAA['datetime']
            NAAD_tracks = [CS for CS in CS_tracks_list_NAAD if np.isin(CS['datetime'], NOAA_times).any()]
            TC_dict = find_matching_tracks_fast(TC_dict, key, CS_track_NOAA, NAAD_tracks, max_distance_km, max_time_diff, similar_steps)
        
        for key in tqdm(TC_dict.keys(), desc=f'Calculating distances {tracking_type} {CVS_speed} {year}'):
            dist_list = []
            CS_track_NOAA = TC_dict[key]['NOAA_track']
            for idx in range(len(TC_dict[key]['NAAD_tracks'])):
                track_NAAD = TC_dict[key]['NAAD_tracks'][idx]
                if track_NAAD is not None:
                    CS_track_NAAD = track_NAAD.rename(columns=lambda x: f"{x}_NAAD_{idx}" if x not in ['datetime'] else x)   
                    CS_track_NOAA = pd.merge(CS_track_NOAA, CS_track_NAAD, on="datetime", how="outer")
                    CS_track_NOAA = CS_track_NOAA.sort_values("datetime").reset_index(drop=True)
                    dist = get_tracks_dist(TC_dict[key]['NOAA_track'], track_NAAD)
                    dist_list.append(np.nanmean(dist))
            
            TC_dict[key]['NOAA_track_NAAD'] = CS_track_NOAA
            TC_dict[key]['dist_btwn_tracks'] = dist_list
        
        save_TC_merged(TC_dict, f'{path_data}/TC_tracks/NAAD_NOAA_tracks_merged', folder_name, NAAD_name='NOAA_track_NAAD')

        for idx, key in tqdm(enumerate(TC_dict.keys()), desc=f'Plotting {tracking_type} {CVS_speed} {year}'):
            plot_closest_track_for_NOAA(TC_dict, key, 'NAAD_tracks', f'{path_data}/TC_tracks/pics_diff_NAAD_NOAA_{data_type}', folder_name, data_type, sigma=sigma)

# Generate all configuration combinations
configurations = list(itertools.product(tracking_types, speed_options))

# Process configurations in parallel
with ProcessPoolExecutor() as executor:
    executor.map(process_configuration, configurations)