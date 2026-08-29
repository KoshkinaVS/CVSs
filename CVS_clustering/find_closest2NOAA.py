from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Initialize paths and parameters
path_init = f'/storage/thalassa/users/vkoshkina'
folder = 'scripts/CS_processing/CVS_tracking/after_70RAE/'
sys.path.insert(2, f'{path_init}/{folder}')

from step_of_tracking import *
from func_for_find_closest_tracks import *

def calculate_detection_metrics(TC_dict, all_naad_tracks):
    hits = 0
    misses = 0
    false_alarms = 0
    false_alarm_tracks = []  # Сохраняем треки для визуализации
    
    # 1. Hits и Misses
    for noaa_id, track_data in TC_dict.items():
        if track_data['NAAD_tracks']:
            hits += 1
        else:
            misses += 1

    matched_track_ids = {t['track_id'].values[0] for v in TC_dict.values() 
                    for t in v['NAAD_tracks']}
    
    unmatched_tracks = [t for t in all_naad_tracks 
                   if t['track_id'].values[0] not in matched_track_ids]
    false_alarms = len(unmatched_tracks)

    return hits, misses, false_alarms, unmatched_tracks
    
def process_cluster(cluster_num, cluster_all, years, path_data, data_type, sigma):
    # Initialize cumulative counters
    total_hits = 0
    total_misses = 0
    total_false_alarms = 0
    all_false_alarm_tracks = []
    
    # Load cluster tracks data
    cluster_tracks_file = f"{path_init}/data/TC_tracks/{data_type}/{data_type}_sigma_2/cluster_results_upd/cluster_tables/summer_nclusters_{cluster_all}_tracks.csv"
    cluster_tracks = pd.read_csv(cluster_tracks_file)
    cluster_tracks = cluster_tracks[cluster_tracks['cluster'] == cluster_num]
    cluster_tracks_paths = cluster_tracks['path'].unique().tolist()
    
    folder_name = f'{output_dir}/NOAA_matching_cluster_{cluster_num}_in_{cluster_all}'
    os.makedirs(folder_name, exist_ok=True)
    
    for year in years:
        # Load NAAD tracks for current year and cluster
        CS_tracks_list_NAAD = []
        for path in cluster_tracks_paths:
            track = pd.read_csv(path, parse_dates=['datetime'])
            if track['datetime'].dt.year[0] == year:
                track['track_id'] = Path(path).stem
                track['file_path'] = path
                CS_tracks_list_NAAD.append(track)
        
        # Load NOAA tracks for current year
        CS_tracks_list_NOAA = []
        CS_tracks_list_NOAA = load_season_tracks_NOAA(CS_tracks_list_NOAA, f'{path_data}/{folder_NOAA}', year, time_th=time_th)
        
        # Initialize dictionary for current year
        TC_dict = {
            CS_track['Id'].values[0]: {
                'NOAA_track': CS_track,
                'NAAD_tracks': [],
                'dist_btwn_tracks': []
            }
            for CS_track in CS_tracks_list_NOAA
        }

        # Find matching tracks
        for key in TC_dict.keys():
            CS_track_NOAA = TC_dict[key]['NOAA_track']
            NOAA_times = CS_track_NOAA['datetime']
            NAAD_tracks = [CS for CS in CS_tracks_list_NAAD if np.isin(CS['datetime'], NOAA_times).any()]
            TC_dict = find_matching_tracks_fast(TC_dict, key, CS_track_NOAA, NAAD_tracks, max_distance_km, max_time_diff, similar_steps)


        for key in tqdm(TC_dict.keys(), desc=f'Calculating distances {year}'):
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
        
        for idx, key in tqdm(enumerate(TC_dict.keys()), 
                    desc=f'Plotting cluster {cluster_num} tracks for {year}',
                    total=len(TC_dict)):
            plot_closest_track_for_NOAA(TC_dict, key, 'NAAD_tracks', path_data, folder_name, data_type)

    
        # Calculate metrics for current year
        year_hits, year_misses, year_false_alarms, false_alarm_tracks = calculate_detection_metrics(TC_dict, CS_tracks_list_NAAD)
        total_hits += year_hits
        total_misses += year_misses
        total_false_alarms += year_false_alarms
        all_false_alarm_tracks.extend(false_alarm_tracks)
        
        # Save individual year results
        metrics = {
            'cluster': cluster_num,
            'year': year,
            'POD': year_hits / (year_hits + year_misses) if (year_hits + year_misses) > 0 else 0,
            'FAR': year_false_alarms / (year_hits + year_false_alarms) if (year_hits + year_false_alarms) > 0 else 0,
            'Hits': year_hits,
            'Misses': year_misses,
            'False_Alarms': year_false_alarms
        }
        
        metrics_file = os.path.join(folder_name, f'detection_metrics_{year}.csv')
        pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
    
    # Calculate overall metrics for cluster
    overall_pod = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0
    overall_far = total_false_alarms / (total_hits + total_false_alarms) if (total_hits + total_false_alarms) > 0 else 0
    
    # Save overall metrics for cluster
    overall_metrics = {
        'cluster': cluster_num,
        'POD': overall_pod,
        'FAR': overall_far,
        'Hits': total_hits,
        'Misses': total_misses,
        'False_Alarms': total_false_alarms,
        'total_years': len(years)
    }
    
    overall_metrics_file = os.path.join(folder_name, 'overall_detection_metrics.csv')
    pd.DataFrame([overall_metrics]).to_csv(overall_metrics_file, index=False)
    
    return overall_metrics

pref_tracking = 'all_points_bound'
local_extr_name = 'local_extr_crit'

dist_m, our_level, level, x_unit, y_unit, u_unit, v_unit, time_unit, level_unit, crit, th, min_samples, eps, CS_points_th = get_init_params(data_type, json_file, print_info=True)

time_th = 3
similar_steps = 3
max_time_diff = 3
max_distance_km = 3*dist_m/1000

years = np.arange(1979, 2019)
months = np.arange(1, 13, 1)

path_dir_data = f'{path_init}/data'
if data_type == 'LoRes':
    path_dir_data = f'{path_dir_data}/LoRes'

path_data = f'{path_init}/data'
folder_NOAA = 'TC_tracks/splitted'

cluster_num = 6
cluster_all = 7


# Load cluster tracks data
cluster_tracks_file = f"{path_init}/data/TC_tracks/{data_type}/{data_type}_sigma_2/cluster_results_upd/cluster_tables/summer_nclusters_{cluster_all}_tracks.csv"
cluster_tracks = pd.read_csv(cluster_tracks_file)
cluster7_tracks = cluster_tracks[cluster_tracks['cluster'] == cluster_num]
cluster7_tracks_paths = cluster7_tracks['path'].unique().tolist()

def load_cluster_tracks(CS_tracks_list_NAAD, year, tracks_paths):
    for path in tracks_paths:
        track = pd.read_csv(path, parse_dates=['datetime'])
        if track['datetime'].dt.year[0] == year:
            track['track_id'] = Path(path).stem  # например, 'track_2018_001'
            track['file_path'] = path
            CS_tracks_list_NAAD.append(track)
    return CS_tracks_list_NAAD

if data_type == 'LoRes':
    path_data_tracks = f'{path_data}/TC_tracks/{data_type}/{data_type}_sigma_{sigma}'
    path_tracks_dir = f'{path_data_tracks}/max_crit_day_data_ocean'
elif data_type == 'ERA5':
    path_data_tracks = f'{path_data}/TC_tracks/{data_type}/{data_type}_sigma_{sigma}'
    path_tracks_dir = f'{path_data_tracks}/max_crit_day_data_ocean'

output_dir = f"{path_data_tracks}/cluster_results_upd"
folder_name = f'{output_dir}/NOAA_matching_cluster_{cluster_num}_in_{cluster_all}'


# Process all clusters and collect results
all_clusters_metrics = []

for cluster_num in range(cluster_all):
    print(f"\nProcessing cluster {cluster_num}...")
    cluster_metrics = process_cluster(cluster_num, cluster_all, years, path_data, data_type, sigma)
    all_clusters_metrics.append(cluster_metrics)

# Save combined metrics for all clusters
combined_metrics_file = os.path.join(output_dir, 'combined_detection_metrics_all_clusters.csv')
pd.DataFrame(all_clusters_metrics).to_csv(combined_metrics_file, index=False)