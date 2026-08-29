from pathlib import Path
import sys
import os

# Инициализация путей и параметров
path_init = f'/storage/thalassa/users/vkoshkina'
folder = 'scripts/CS_processing/CVS_tracking/after_70RAE/'
sys.path.insert(2, f'{path_init}/{folder}')

from step_of_tracking import *
from func_for_find_closest_tracks import *

tracking_type = 'tracking_local_2_phase'
# tracking_type = 'tracking_local_global'

CVS_speed = 'adv_speed'
CVS_speed = 'adv_bg_speed'

# CVS_speed = 'no_speed'


# pref_tracking = 'rad_and_bound'
pref_tracking = 'all_points_bound'

folder_name = f'{tracking_type}_{CVS_speed}_{pref_tracking}_{data_type}_sigma_{sigma}'

local_extr_name = 'local_extr_crit'

dist_m, our_level, x_unit, y_unit, u_unit, v_unit, time_unit, level_unit, crit, th, min_samples, eps, CS_points_th = get_init_params(data_type, json_file, print_info=True)


time_th = 3

similar_steps = 3
max_time_diff = 3  # Максимальная разница по времени
max_distance_km = 3*dist_m/1000  # Максимальная допустимая дистанция (км)


years = np.arange(1979,2019)
# years = np.arange(2010,2011)

months = np.arange(1,13,1)
# months = np.arange(1,2,1)


path_dir_data = f'{path_init}/data'
if data_type == 'LoRes':
    path_dir_data = f'{path_dir_data}/LoRes'

path_data = f'{path_init}/data'

folder_NOAA = 'TC_tracks/splitted'

DBSCAN_name = f'DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_level_{level}_smoothing_sigma_{sigma}'


folder_NAAD = f'{path_dir_data}/{data_type}/{data_type}_tracks/{pref_tracking}/{tracking_type}_{CVS_speed}_{pref_tracking}/{DBSCAN_name}'

path_tracks_dir = f'{folder_NAAD}/tracks_{circ}'
# path_tracks_dir = f'{folder}/tracks_{circ}_params/'




for year in years:
    CS_tracks_list_NAAD = []
    CS_tracks_list_NOAA = []
    CS_tracks_list_NAAD = load_season_tracks(CS_tracks_list_NAAD, year, months, path_tracks_dir, time_th=time_th)
    CS_tracks_list_NOAA = load_season_tracks_NOAA(CS_tracks_list_NOAA, f'{path_data}/{folder_NOAA}', year, time_th=time_th)
    
    # Создаем словарь
    TC_dict = {
        CS_track['Id'].values[0]: {
            'NOAA_track': CS_track,  # Полный путь к файлу NOAA
            'NAAD_tracks': [],    # Пустой список для файлов NAAD, который будет заполняться позже
            'dist_btwn_tracks': []    # Пустой список для расстояний, который будет заполняться позже
            
        }
        for CS_track in CS_tracks_list_NOAA
    }
    
    
    for key in tqdm(TC_dict.keys()):
        
        CS_track_NOAA = TC_dict[key]['NOAA_track']
        NOAA_times = CS_track_NOAA['datetime']
        NAAD_tracks = [CS for CS in CS_tracks_list_NAAD if np.isin(CS['datetime'], NOAA_times).any()]
        # TC_dict = find_closest_track_many(TC_dict, key, CS_track_NOAA, NAAD_tracks, max_distance_km)
        TC_dict = find_matching_tracks_fast(TC_dict, key, CS_track_NOAA, NAAD_tracks, max_distance_km, max_time_diff, similar_steps)
    
    
    for key in tqdm(TC_dict.keys()):
        dist_list = []
        CS_track_NOAA = TC_dict[key]['NOAA_track']
        for idx in range(len(TC_dict[key]['NAAD_tracks'])):
            track_NAAD = TC_dict[key]['NAAD_tracks'][idx]
            if track_NAAD is not None:
                CS_track_NAAD = track_NAAD.rename(columns=lambda x: f"{x}_NAAD_{idx}" if x not in ['datetime'] else x)   
                
                # Объединяем таблицы на основе поля datetime (outer join для сохранения всех значений)
                CS_track_NOAA = pd.merge(CS_track_NOAA, CS_track_NAAD, on="datetime", how="outer")
                # Сортируем по дате для удобства
                CS_track_NOAA = CS_track_NOAA.sort_values("datetime").reset_index(drop=True)
        
                dist = get_tracks_dist(TC_dict[key]['NOAA_track'], track_NAAD)
                dist_list.append(np.nanmean(dist))
                
        TC_dict[key]['NOAA_track_NAAD'] = CS_track_NOAA
        TC_dict[key]['dist_btwn_tracks'] = dist_list
    
    
    save_TC_merged(TC_dict, path_data, folder_name, NAAD_name='NOAA_track_NAAD',)

    for idx, key in tqdm(enumerate(TC_dict.keys())):
        plot_closest_track_for_NOAA(TC_dict, key, 'NAAD_tracks', path_data, folder_name, data_type)