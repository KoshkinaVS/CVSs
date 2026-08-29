from func_for_add_params import *
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import itertools

# Инициализация путей и параметров
path_init = f'/storage/thalassa/users/vkoshkina'
folder = 'scripts/CS_processing/CVS_tracking/after_70RAE/'
sys.path.insert(2, f'{path_init}/{folder}')

tracking_type = 'tracking_local_2_phase'
json_file = f'{path_init}/{folder}/tracking_init.json'

dist_m, our_level, level, x_unit, y_unit, u_unit, v_unit, time_unit, level_unit, crit, th, min_samples, eps, CS_points_th = get_init_params(data_type, json_file, print_info=True)

path_data = f'{path_init}/data'  
circ = 'C'
pref_tracking = 'all_points_bound'
CVS_speed = 'adv_speed'

DBSCAN_name = f'DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_level_{level}_smoothing'
results_dir = f"{tracking_type}_{CVS_speed}_{pref_tracking}"

if data_type == 'HiRes':
    path_data_tracks = f'{path_data}/TC_tracks/my_tracking_results/{pref_tracking}/{results_dir}/{data_type}_tracks_sigma_{sigma}/'
elif data_type == 'ERA5':
    path_data_tracks = f"{path_data}/TC_tracks/{data_type}/my_tracking_results/{pref_tracking}/{results_dir}/{data_type}_tracks_sigma_{sigma}/"
else:
    path_data_tracks = f'{path_data}/{data_type}/{data_type}/{data_type}_tracks/{pref_tracking}/{results_dir}/{DBSCAN_name}'

path_tracks_dir = f'{path_data_tracks}/tracks_{circ}'

if data_type == 'LoRes':
    path_data_tracks = f'{path_data}/TC_tracks/{data_type}/{data_type}_sigma_{sigma}/'
    years = np.arange(1979, 2019)
elif data_type == 'ERA5':
    path_data_tracks = f"{path_data}/TC_tracks/{data_type}/my_tracking_results/{pref_tracking}/{results_dir}/{data_type}_tracks_sigma_{sigma}"
    years = np.arange(1979, 2025)

path_tracks_dir = f'{path_data_tracks}/tracks_{circ}_params'
months = np.arange(1, 13)

if data_type != 'ERA5':
    params_config = [
        {'param': 'cape_2d', 'level': 0, 'agg': 'max'},
        {'param': 'cape_3d', 'level': 23, 'agg': 'max'},
        {'param': 'pvo', 'level': 23, 'agg': 'max'},
        {'param': 'helicity', 'level': 0, 'agg': 'max'},
        {'param': 'updraft_helicity', 'level': 0, 'agg': 'max'},
        {'param': 'pw', 'level': 0, 'agg': 'max'},
        {'param': 'slp', 'level': 0, 'agg': 'delta'},
    ]
else:
    params_config = []

def add_parameters_batch(df, params_config, data_type='LoRes', km=77):
    results = {f"{cfg['param']}_{cfg['agg']}": [] for cfg in params_config}
    
    datetimes = df['datetime'].values
    lats = df['lat'].values
    lons = df['lon'].values
    rads = df['rad'].values
    
    for cfg in params_config:
        param = cfg['param']
        agg_type = cfg['agg']
        level = cfg['level']
        
        if data_type == 'ERA5':
            if param == 'wspd':
                values = [compute_aggregation_wspd_ERA5(param, dt, lat, lon, rad, level, 
                         data_type, km, agg_type) 
                        for dt, lat, lon, rad in zip(datetimes, lats, lons, rads)]
            else:
                values = [compute_aggregation_value_ERA5(param, dt, lat, lon, rad, level, 
                         data_type, km, agg_type) 
                        for dt, lat, lon, rad in zip(datetimes, lats, lons, rads)]
        else:
            values = [get_mean_value(df, i, level, param, agg_type, data_type, km) 
                     for i in range(len(df))]
        
        results[f"{param}_{agg_type}"] = values
    
    for col, values in results.items():
        df[col] = values
    return df

def process_year_month(args):
    year, month, start_idx = args
    i = start_idx
    
    files = sorted(glob.glob(f'{path_tracks_dir}/{year}-{month:02d}/*.csv'))
    track_folder_new = f'{path_data_tracks}/tracks_{circ}_params_new/{year}-{month:02d}'
    
    if not os.path.exists(track_folder_new):
        os.makedirs(track_folder_new)
    
    for idx, file in enumerate(files):
        try:
            df = pd.read_csv(file, parse_dates=['datetime'])
            df = df.drop(df.columns[0], axis=1)
            df = df.dropna(how='any')
            
            date_start = str(df['datetime'].values[0])[:-16]
            output_file = f'{track_folder_new}/{i:08d}_track_{date_start}.csv'
            
            if os.path.exists(output_file):
                continue
            
            df = add_parameters_batch(df, params_config, data_type=data_type, km=77)
            df.to_csv(output_file, index=False)
            i += 1
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue
    
    return i

def main():
    # Создаем список задач (год, месяц) с начальными индексами
    tasks = []
    current_idx = 0
    
    for year in years:
        # Сначала подсчитываем общее количество файлов для года
        year_files_count = 0
        for month in months:
            files = glob.glob(f'{path_tracks_dir}/{year}-{month:02d}/*.csv')
            year_files_count += len(files)
        
        # Затем создаем задачи для каждого месяца с правильными индексами
        month_start_idx = current_idx
        for month in months:
            files = glob.glob(f'{path_tracks_dir}/{year}-{month:02d}/*.csv')
            if files:
                tasks.append((year, month, month_start_idx))
                month_start_idx += len(files)
        
        current_idx += year_files_count
    
    # Параллельная обработка
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(process_year_month, task) for task in tasks]
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing task: {e}")

if __name__ == '__main__':
    main()