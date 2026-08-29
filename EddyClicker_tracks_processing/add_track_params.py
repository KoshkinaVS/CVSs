from func_for_add_params import *


# Инициализация путей и параметров
path_init = f'/storage/thalassa/users/vkoshkina'
folder = 'scripts/CS_processing/CVS_tracking/after_70RAE/'
sys.path.insert(2, f'{path_init}/{folder}')



json_file = f'{path_init}/{folder}/tracking_init.json'

dist_m, our_level, level, x_unit, y_unit, u_unit, v_unit, time_unit, level_unit, crit, th, min_samples, eps, CS_points_th = get_init_params(data_type, json_file, print_info=True)


path_data = f'{path_init}/data'  

if data_type == 'SMP':
    years = np.arange(2019, 2020)
    
    path_init = f'/storage/thalassa/users/vkoshkina'
    path_dir_data = f'{path_init}/data'
    path_tracks_dir = f'{path_dir_data}/{data_type}/EddyClicker_tracks_with_params'
    
    # Добавьте эти строки:
    new_circ_folder = f'EddyClicker_tracks_with_params_fin'  # или как угодно
    path_data_tracks = f'{path_dir_data}/{data_type}'  # куда сохранять результаты
    
    



months = np.arange(1,7,1)
days = np.arange(1,32,1)


if data_type == 'ERA5':
    params_config = [
        {'param': 'msl', 'level': 0, 'agg': 'min'},
        {'param': 'mlhf', 'level': 0, 'agg': 'max'},
        {'param': 'mshf', 'level': 0, 'agg': 'max'},
        {'param': 'sst', 'level': 0, 'agg': 'median'},
        {'param': 'wspd', 'level': 0, 'agg': 'max'},
        {'param': 't2m', 'level': 0, 'agg': 'median'},
        {'param': 't', 'level': 500, 'agg': 'median'},
        {'param': 'w', 'level': 500, 'agg': 'median'},
        {'param': 'tp', 'level': 0, 'agg': 'median'},
        
    ]

    km = 25

elif data_type == 'SMP':
    params_config = [
        {'param': 'SST', 'level': 0, 'agg': 'median'},
        {'param': 'LH', 'level': 0, 'agg': 'max'},
        {'param': 'HFX', 'level': 0, 'agg': 'max'},
        {'param': 'slp', 'level': 0, 'agg': 'min'},
        {'param': 'wspd', 'level': 0, 'agg': 'max'},
        {'param': 'T2', 'level': 0, 'agg': 'median'},
        {'param': 'theta', 'level': 12, 'agg': 'median'},
        {'param': 'wa', 'level': 12, 'agg': 'median'},
        {'param': 'cape_2d', 'level': 0, 'agg': 'max'},
        {'param': 'cape_3d', 'level': 23, 'agg': 'max'},
        {'param': 'pvo', 'level': 23, 'agg': 'max'},
        {'param': 'helicity', 'level': 0, 'agg': 'max'},
        {'param': 'updraft_helicity', 'level': 0, 'agg': 'max'},
        {'param': 'pw', 'level': 0, 'agg': 'max'},
        {'param': 'slp', 'level': 0, 'agg': 'delta'},
        {'param': 'T2', 'level': 0, 'agg': 'delta'},
        
    ]

    km = 6 
else:
    params_config = [
        {'param': 'SST', 'level': 0, 'agg': 'median'},
        {'param': 'LH', 'level': 0, 'agg': 'max'},
        {'param': 'HFX', 'level': 0, 'agg': 'max'},
        {'param': 'slp', 'level': 0, 'agg': 'min'},
        {'param': 'wspd', 'level': 0, 'agg': 'max'},
        {'param': 'T2', 'level': 0, 'agg': 'median'},
        {'param': 'theta', 'level': 12, 'agg': 'median'},
        {'param': 'wa', 'level': 12, 'agg': 'median'},
        {'param': 'cape_2d', 'level': 0, 'agg': 'max'},
        {'param': 'cape_3d', 'level': 23, 'agg': 'max'},
        {'param': 'pvo', 'level': 23, 'agg': 'max'},
        {'param': 'helicity', 'level': 0, 'agg': 'max'},
        {'param': 'updraft_helicity', 'level': 0, 'agg': 'max'},
        {'param': 'pw', 'level': 0, 'agg': 'max'},
        {'param': 'slp', 'level': 0, 'agg': 'delta'},
        {'param': 'T2', 'level': 0, 'agg': 'delta'},
        
    ]

    km = 77 #### change for HiRes



# i = 0
for year in years:
    for month in tqdm(months):

        if data_type == 'SMP':
            files = sorted(glob.glob(f'{path_tracks_dir}/*.csv'))
            track_folder_new = f'{path_data_tracks}/{new_circ_folder}'
        else: 
            files = sorted(glob.glob(f'{path_tracks_dir}/{year}-{month:02d}/*.csv'))
            track_folder_new = f'{path_data_tracks}/{new_circ_folder}/{year}-{month:02d}'
        
        # Создаем конечную папку, если ее нет
        if not os.path.exists(track_folder_new):
            os.makedirs(track_folder_new)
        
        TCs = []
        
        for idx, file in tqdm(enumerate(files), total=len(files), desc=f"Loading tracks for {year}-{month:02d}"):
            if data_type == 'SMP':
                df = pd.read_csv(file, parse_dates=['time'])
                df = preprocessing_EC_tracks(df)
            else:
                df = pd.read_csv(file, parse_dates=['datetime'])
                df = df.drop(df.columns[0], axis=1)
            df = df.dropna(how='any')
            TCs.append((df, file))  # Store both the dataframe and the original file path
        
        for idx, (TC, original_file) in tqdm(enumerate(TCs), total=len(TCs), desc="Processing TCs"):
            # Extract the original filename without path
            original_filename = os.path.basename(original_file)
            output_file = f'{track_folder_new}/{original_filename}'
            
            # Проверяем, существует ли уже файл
            if os.path.exists(output_file):
                print(f"File {output_file} already exists. Skipping...")
                continue
            
            df = add_parameters_batch(TC, params_config, data_type=data_type, km=km)
            
            # Сохраняем результаты
            df.to_csv(output_file, index=False)