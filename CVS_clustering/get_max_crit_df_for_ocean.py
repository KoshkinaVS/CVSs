import seaborn as sns
import matplotlib.pyplot as plt

from func_for_CVS_clusters import *


tracking_type = 'tracking_local_2_phase'
pref_tracking = 'all_points_bound'
CVS_speed = 'adv_speed'
circ = 'C'
n_clusters = 5

months = np.arange(1, 13, 1)
path_data = f'{path_init}/data'  
DBSCAN_name = f'DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_level_{level}_smoothing'
results_dir = f"{tracking_type}_{CVS_speed}_{pref_tracking}"

if data_type == 'LoRes':
    path_data_tracks = f'{path_data}/TC_tracks/{data_type}/{data_type}_sigma_{sigma}/'
    years = np.arange(1979, 2019)
    path_tracks_dir = f'{path_data_tracks}/tracks_{circ}_params_new'    
    
elif data_type == 'ERA5':
    path_data_tracks = f"{path_data}/TC_tracks/{data_type}/my_tracking_results/{pref_tracking}/{results_dir}/{data_type}_tracks_sigma_{sigma}"
    years = np.arange(1979, 2025)
    path_tracks_dir = f'{path_data_tracks}/tracks_{circ}_params'
    
    

if data_type == 'LoRes':
    time_th = 8
    # param_cols = [
    #     'datetime', 'x', 'y',
    #     'lat', 'lon', 
    #     'rad', 'crit', 
    #     # 'track_len',
    #     'msl_min', 'mslhf_max', 'msshf_max', 'wspd_max', 
    #     't2_median', 'theta_median',
    #     'w_median']

    param_cols = [
                    'datetime', 'x', 'y',
                    'lat', 'lon', 
                    'rad', 'crit', 
                    'msl_min', 'wspd_max', 
                    'SST_median',
                    'HFX_max', 'LH_max',
                    't2_median', 'theta_median',
                    'w_median',
                    'cape_2d_max', 'cape_3d_max', 
                    'pvo_max', 'helicity_max', 
                    'updraft_helicity_max', 'pw_max', 
                    'slp_delta', 'T2_delta'
                ]
elif data_type == 'ERA5':
    time_th = 24
    param_cols = [
    'datetime', 'x', 'y',
    'lat', 'lon', 
    'rad', 'crit', 
        # 'track_len',
    'msl_min', 'mlhf_max', 'mshf_max', 'wspd_max', 
    't2m_median', 't_median',
    'w_median', 'tp_median']

    
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


def calculate_velocity(track, max_idx, data_type='LoRes'):

    # для проверки, что максимум не на границе трека
    start_stop = False
    length = len(track)
    
    if length == 1:
        return np.nan, False
    
    # Преобразуем max_idx в целое число, если это Index объект
    try:
        if hasattr(max_idx, 'item'):  # Для pandas Index и подобных
            max_idx = max_idx.item()
        max_idx = int(max_idx)
    except (ValueError, TypeError):
        return np.nan, False
    
    track_len = track['track_len'].values
    
    # Границы окна (гарантируем целые числа)
    start = max(max_idx - 2, 0)
    end = min(max_idx + 2, length - 1)
    start, end = int(start), int(end)  # Явное преобразование

    
    # Проверяем граничные условия
    if max_idx == length - 1 or max_idx == 0:
        start_stop = True
    
    # Если окно слишком маленькое (меньше 2 точек), возвращаем NaN
    if end - start < 1:
        return np.nan, start_stop
    
    # Вычисляем все разности в окрестности
    diffs = np.diff(track_len[start:end+1])

    if data_type == 'ERA5':
        diffs = diffs*np.cos(np.radians(track['lat'][start:end]))

    if data_type == 'ERA5':
        avg_velocity = (111*1000*0.25 * np.mean(diffs) / (3600))
    else:    
        # Вычисляем среднюю скорость (в м/с)
        avg_velocity = (77*1000 * np.mean(diffs) / (3*3600))
    
    return avg_velocity, start_stop



for year in tqdm(years):
    for month in months:
        path_list = []
        basenames_list = []
        CS_tracks_list = []
        
        CS_tracks_list, path_list, basenames_list = load_season_tracks(
            CS_tracks_list, path_list, basenames_list, year, [month], path_tracks_dir, time_th=time_th)

        print(path_list[0])
    
        # Фильтруем треки по положению над океаном
        ocean_tracks, filtered_paths, filtered_names = filter_ocean_tracks(CS_tracks_list, path_list, basenames_list, ground, ocean_threshold=0.8)
            
        df_max = pd.DataFrame()
        max_idx_list = []
        
        for idx, CS in enumerate(tqdm(ocean_tracks)):
            df_max, max_idx = get_max_crit_day_values(CS, df_max, idx, param_cols)
            max_idx_list.append(max_idx)

        df_max['track_len'] = [track['track_len'].values[-1] for track in ocean_tracks]
        df_max['duration'] = [(track['datetime'].iloc[-1] - track['datetime'].iloc[0]).total_seconds() / 3600 for track in ocean_tracks]
        
        # df_max['vel'] = [calculate_velocity(track, max_idx)[0] 
        #         for max_idx, track in zip(max_idx_list, ocean_tracks)]

        results = [calculate_velocity(track, max_idx, data_type) for max_idx, track in zip(max_idx_list, ocean_tracks)]
        df_max['vel'], df_max['start_stop'] = zip(*results)
        
        # df_max['x_start'] = [track['x'][0] for track in ocean_tracks]
        
        df_max['path'] = filtered_paths
        df_max['name'] = filtered_names

        if data_type == 'ERA5':
            path_data_tracks_new = f'{path_data}/TC_tracks/{data_type}/{data_type}_sigma_{sigma}/'
        else:
            path_data_tracks_new = path_data_tracks
            
        os.makedirs(f'{path_data_tracks_new}/max_crit_day_data_ocean', exist_ok=True)
        df_max.to_csv(f'{path_data_tracks_new}/max_crit_day_data_ocean/max_crit_day_CVS_{year}-{month:02d}.csv', index=False)
