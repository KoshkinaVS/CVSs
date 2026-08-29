from func_for_add_params import *


# Инициализация путей и параметров
path_init = f'/storage/thalassa/users/vkoshkina'
folder = 'scripts/CS_processing/CVS_tracking/after_70RAE/'
sys.path.insert(2, f'{path_init}/{folder}')


tracking_type = 'tracking_local_2_phase'



json_file = f'{path_init}/{folder}/tracking_init.json'

dist_m, our_level, level, x_unit, y_unit, u_unit, v_unit, time_unit, level_unit, crit, th, min_samples, eps, CS_points_th = get_init_params(data_type, json_file, print_info=True)


path_data = f'{path_init}/data'  

circ = 'C'


cols = ['t', 'datetime' ,'x','y',
        'lat','lon',
        # 'rad', # if tracking_local_extrema_only -- without rad
        'crit',
        'track_len']

if tracking_type != 'tracking_local_extrema_only':
    cols.append('rad')


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

years = np.arange(1979,2025)
# years = np.arange(1984,2025)
# years = np.arange(2005,2025)
# years = np.arange(2003,2005)



months = np.arange(1,13,1)
days = np.arange(1,32,1)


if data_type != 'ERA5':
    params_config = [
        {'param': 'msl', 'level': 0, 'agg': 'min'},
        {'param': 'mslhf', 'level': 0, 'agg': 'max'},
        {'param': 'msshf', 'level': 0, 'agg': 'max'},
        {'param': 'wspd', 'level': 0, 'agg': 'max'},
        {'param': 't2', 'level': 0, 'agg': 'median'},
        {'param': 'theta', 'level': 12, 'agg': 'median'},
        {'param': 'w', 'level': 12, 'agg': 'median'},
        # {'param': 'w', 'level': 12, 'agg': 'median'},
    ]
else:
    params_config = [
        {'param': 'msl', 'level': 0, 'agg': 'min'},
        {'param': 'mlhf', 'level': 0, 'agg': 'max'},
        {'param': 'mshf', 'level': 0, 'agg': 'max'},
        
        # {'param': 'sst', 'level': 0, 'agg': 'median'},
        {'param': 'wspd', 'level': 0, 'agg': 'max'},
        {'param': 't2m', 'level': 0, 'agg': 'median'},
        {'param': 't', 'level': 500, 'agg': 'median'},
        {'param': 'w', 'level': 500, 'agg': 'median'},
        {'param': 'tp', 'level': 0, 'agg': 'median'},
        
    ]


i = 0
for year in years:

    for month in tqdm(months):

        files = sorted(glob.glob(f'{path_tracks_dir}/{year}-{month:02d}/*.csv'))
        
        track_folder_new = f'{path_data_tracks}/tracks_{circ}_params/{year}-{month:02d}'
        
        # Создаем конечную папку, если ее нет
        if not os.path.exists(track_folder_new):
            os.makedirs(track_folder_new)
        
        TCs = []
        
        for idx, file in tqdm(enumerate(files), total=len(files), desc=f"Loading tracks for {year}-{month:02d}"):
            df = pd.read_csv(file, parse_dates=['datetime'])
            df = df.drop(df.columns[0], axis=1)
            df = df.dropna(how='any')
            TCs.append(df)
        

        
        for idx, TC in tqdm(enumerate(TCs), total=len(TCs), desc="Processing TCs"):
            date_start = str(TC['datetime'].values[0])[:-16]
            output_file = f'{track_folder_new}/{i:08d}_track_{date_start}.csv'
            
            # Проверяем, существует ли уже файл
            if os.path.exists(output_file):
                print(f"File {output_file} already exists. Skipping...")
                continue
            
            df = add_parameters_batch(TC, params_config, data_type=data_type, km=25)
            
            # Сохраняем результаты
            df.to_csv(output_file, index=False)
            i+=1