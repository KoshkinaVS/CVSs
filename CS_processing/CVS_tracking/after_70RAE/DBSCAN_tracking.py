import sys
import pickle

path_init = f'/storage/thalassa/users/vkoshkina'
folder = 'scripts/CS_processing/CVS_tracking/after_70RAE/'

sys.path.insert(2, f'{path_init}/{folder}')

from step_of_tracking import *

# from func_for_local_only import *

from func_for_local_extrema import *
from func_for_global_only import *
from func_for_local_2_phase import *


print(tracking_type)
print(CVS_speed)



print('level: ')
level = int(input())

path_dir_data = f'{path_init}/data/LoRes'

folder_name = f'{data_type}/DBSCAN_02-04-10_smoothing'
name_init = f'sigma_2_DBSCAN_{data_type}'

folder_name = f'{data_type}/DBSCAN_02-04-10_with_uv_time_smoothing'
name_init = f'sigma_2_DBSCAN_{data_type}_level_{level}'

years = np.arange(1979,2019)
# years = np.arange(2010,2011)

months = np.arange(1,13,1)
# months = np.arange(8,11)

very_first = True

CS_tracks_list = []

cluster_idx = 0

results_dir = f'{tracking_type}_{CVS_speed}_crit_sorted'

for year in years:
     
    print(f'year: {year}')
    
    for month in months:

        path_data_dir = f'{path_dir_data}/{data_type}/{data_type}_tracks/{results_dir}/DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_level_{level}_smoothing/tracks_{circ}/{year}-{month:02d}/'

        if not os.path.exists(f'{path_data_dir}'):
            os.makedirs(f'{path_data_dir}')
        
        ds = xr.open_dataset(f"{path_dir_data}/{folder_name}/{name_init}_{year}-{month:02d}.nc")
        dt_step = np.timedelta64(ds[time_name].values[1] - ds[time_name].values[0], 's').astype(int)
              
        if very_first == True:
            t_start = 1
            very_first = False
            
            if tracking_type == 'tracking_local_only':
                local_max = get_stat_local_max(ds, 0, circ)
                CS_tracks_list, clstr_len = track_init(ds, 0, data_type, local_max, CS_tracks_list, 0, time_name)
            elif tracking_type == 'tracking_local_2_phase':
                r2d_coords = get_all_coords(ds, 0, circ)
                local_max = get_stat_local_max(ds, 0, circ)
                local_max = update_rad(local_max, r2d_coords)
                CS_tracks_list, clstr_len = track_init(ds, 0, data_type, local_max, CS_tracks_list, 0, time_name)
            else:
                stat_Q = get_stat_global_max(ds, 0, circ)
                CS_tracks_list, clstr_len = track_init(ds, 0, data_type, stat_Q, CS_tracks_list, 0, time_name)
        else:
            t_start = 0
            
            
        for t in tqdm(range(t_start,len(ds[time_unit]))):
            cluster_idx, CS_tracks_list, clstr_len = step_of_tracking(cluster_idx, CS_tracks_list, clstr_len, ds, data_type, path_data_dir, t,                      speed_level, CVS_speed, circ=circ, dt_step=dt_step)

    for TC in tqdm(CS_tracks_list):
        if np.sum(~np.isnan(TC['t'])) >= 3:
            cluster_idx = save_track_csv(cluster_idx, TC, path_data_dir)
            
    
            