from func_for_add_params_2026_v1 import *


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
    # path_tracks_dir = f'{path_dir_data}/{data_type}/EddyClicker_tracks_with_params'
    path_tracks_dir = f'{path_dir_data}/{data_type}'
    
    
    new_circ_folder = f'EddyClicker_tracks_with_params_max_day_2025'  # или как угодно
    path_data_tracks = f'{path_dir_data}/{data_type}'  # куда сохранять результаты

    km = 6  # разрешение SMP
    
    
if data_type == 'LoRes':
    years = np.arange(2010, 2011)

    # Устанавливаем стартовую дату
    start_date = pd.Timestamp('2010-01-01 00:00:00')
    dt = 3
    
    path_init = f'/storage/thalassa/users/vkoshkina'
    path_dir_data = f'{path_init}/data'
    path_tracks_dir = f'{path_dir_data}/{data_type}/{data_type}/EddyClicker_tracks'
    
    
    new_circ_folder = f'EddyClicker_tracks_with_params_max_day_2025'  # или как угодно
    path_data_tracks = f'{path_dir_data}/{data_type}/{data_type}'  # куда сохранять результаты
    
    km = 77


months = np.arange(1,13,1)
days = np.arange(1,32,1)

# level = 10 # ~ 850 hPa
# level = 13 # ~ 700 hPa

# level = 18 # ~ 500 hPa


if data_type == 'SMP':
    params_config = [
        # --- Критерии интенсивности ---
        {'param': 'pvo', 'level': 10, 'agg': 'max'},              # proxy for ζf,850 (потенциальная завихренность)
        {'param': 'wspd', 'level': 0, 'agg': 'max'},              # U10m max
        {'param': 'wspd', 'level': 18, 'agg': 'max'},              # U10m max
        {'param': 'slp', 'level': 0, 'agg': 'min'},               # SLP в центре
        {'param': 'slp', 'level': 0, 'agg': 'mean'},               # SLP в центре
        {'param': 'slp', 'level': 0, 'agg': 'delta'},             # ΔSLP = mean(SLP in r) - center(SLP)

        # --- MCAO: температура и потенциальная температура ---
        {'param': 'T2', 'level': 0, 'agg': 'median'},             # SST ≈ T2 над океаном
        {'param': 'SST', 'level': 0, 'agg': 'median'},             # SST
        
        {'param': 'theta', 'level': 10, 'agg': 'median'},         # θ ~850 hPa (уровень 12 ≈ 850 гПа в вашей модели)
        {'param': 'theta', 'level': 13, 'agg': 'median'},     # θ на ~700 гПа (уровень 18 — уточните по wrf.getvar("pressure"))
        {'param': 'theta', 'level': 18, 'agg': 'median'},     # θ на ~700 гПа (уровень 18 — уточните по wrf.getvar("pressure"))
        
        {'param': 'temp', 'level': 10, 'agg': 'median'},             # T на ~850 гПа
        {'param': 'temp', 'level': 13, 'agg': 'median'},             # T на ~700 гПа
        {'param': 'temp', 'level': 18, 'agg': 'median'},             # T на ~500 гПа
        

        # --- Эквивалентная потенциальная температура (θe) ---
        {'param': 'theta_e', 'level': 10, 'agg': 'median'},       # θe,850
        {'param': 'theta_e', 'level': 13, 'agg': 'median'},       # θe,700
        {'param': 'theta_e', 'level': 18, 'agg': 'median'},       # θe,500
    

        # --- Дополнительные параметры ---
        # {'param': 'cape_2d', 'level': 0, 'agg': 'max'},           # proxy for moisture/instability
        {'param': 'pw', 'level': 0, 'agg': 'max'},                # precipitable water ≈ total column water
        {'param': 'helicity', 'level': 0, 'agg': 'max'},          # связано с вращением
        {'param': 'updraft_helicity', 'level': 0, 'agg': 'max'},
        {'param': 'pblh', 'level': 0, 'agg': 'median'},           # высота ППС (planetary boundary layer height)

        # --- Градиенты и тропопауза (приближённо) ---
        # Временно опускаем P_tr, θ_tr — если нужны, реализуем отдельно.
    ]
    
elif data_type == 'LoRes':
    params_config = [
        # --- Критерии интенсивности ---
        {'param': 'pvo', 'level': 10, 'agg': 'max'},              # proxy for ζf,850 (потенциальная завихренность)
        {'param': 'wspd', 'level': 0, 'agg': 'max'},              # U10m max
        {'param': 'wspd', 'level': 18, 'agg': 'max'},              # U10m max
        {'param': 'slp', 'level': 0, 'agg': 'min'},               # SLP в центре
        {'param': 'slp', 'level': 0, 'agg': 'mean'},               # SLP в центре
        {'param': 'slp', 'level': 0, 'agg': 'delta'},             # ΔSLP = mean(SLP in r) - center(SLP)

        # --- MCAO: температура и потенциальная температура ---
        {'param': 'T2', 'level': 0, 'agg': 'median'},             # SST ≈ T2 над океаном
        {'param': 'SST', 'level': 0, 'agg': 'median'},             # SST
        
        {'param': 'theta', 'level': 10, 'agg': 'median'},         # θ ~850 hPa (уровень 12 ≈ 850 гПа в вашей модели)
        {'param': 'theta', 'level': 13, 'agg': 'median'},     # θ на ~700 гПа (уровень 18 — уточните по wrf.getvar("pressure"))
        {'param': 'theta', 'level': 18, 'agg': 'median'},     # θ на ~700 гПа (уровень 18 — уточните по wrf.getvar("pressure"))
        
        {'param': 'temp', 'level': 10, 'agg': 'median'},             # T на ~850 гПа
        {'param': 'temp', 'level': 13, 'agg': 'median'},             # T на ~700 гПа
        {'param': 'temp', 'level': 18, 'agg': 'median'},             # T на ~500 гПа
        

        # --- Эквивалентная потенциальная температура (θe) ---
        {'param': 'theta_e', 'level': 10, 'agg': 'median'},       # θe,850
        {'param': 'theta_e', 'level': 13, 'agg': 'median'},       # θe,700
        {'param': 'theta_e', 'level': 18, 'agg': 'median'},       # θe,500
    

        # --- Дополнительные параметры ---
        # {'param': 'cape_2d', 'level': 0, 'agg': 'max'},           # proxy for moisture/instability
        {'param': 'pw', 'level': 0, 'agg': 'max'},                # precipitable water ≈ total column water
        {'param': 'helicity', 'level': 0, 'agg': 'max'},          # связано с вращением
        {'param': 'updraft_helicity', 'level': 0, 'agg': 'max'},
        {'param': 'pblh', 'level': 0, 'agg': 'median'},           # высота ППС (planetary boundary layer height)

        # --- Градиенты и тропопауза (приближённо) ---
        # Временно опускаем P_tr, θ_tr — если нужны, реализуем отдельно.
    ]



# i = 0
for year in years:
    for month in tqdm(months):

        files = sorted(glob.glob(f'{path_tracks_dir}/*.csv'))
        track_folder_new = f'{path_data_tracks}/{new_circ_folder}'
        
        # Создаем конечную папку, если ее нет
        if not os.path.exists(track_folder_new):
            os.makedirs(track_folder_new)
        
        TCs = []
        
        for idx, file in tqdm(enumerate(files), total=len(files), desc=f"Loading tracks for {year}-{month:02d}"):
            if data_type == 'SMP':
                df = pd.read_csv(file, parse_dates=['datetime'])
                # df = preprocessing_EC_tracks(df)
            else:
                df = pd.read_csv(file)
                df = df.drop(df.columns[0], axis=1)
                
                # Создаем список дат с шагом 3 часа
                dates = [start_date + pd.Timedelta(hours=dt*i) for i in df['time_ind'].values]
                
                # Добавляем в DataFrame
                df['time'] = dates
                df = preprocessing_EC_tracks(df)
                
                
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