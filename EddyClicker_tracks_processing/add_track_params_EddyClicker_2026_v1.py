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
        {'param': 'cape_2d', 'level': 0, 'agg': 'max'},           # proxy for moisture/instability
        {'param': 'pw', 'level': 0, 'agg': 'max'},                # precipitable water ≈ total column water
        {'param': 'helicity', 'level': 0, 'agg': 'max'},          # Storm Relative Helicity
        {'param': 'updraft_helicity', 'level': 0, 'agg': 'max'},
        {'param': 'pblh', 'level': 0, 'agg': 'median'},           # высота ППС (planetary boundary layer height)
        {'param': 'rh2', 'level': 0, 'agg': 'median'},           # 2m Relative Humidity
        

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
        
        # Внутри цикла по TCs:
        for idx, (TC, original_file) in tqdm(enumerate(TCs), total=len(TCs), desc="Processing TCs"):
            original_filename = os.path.basename(original_file)
            output_file = f'{track_folder_new}/{original_filename}'
        
            # Загружаем существующий файл, если он есть
            if os.path.exists(output_file):
                existing_df = pd.read_csv(output_file, parse_dates=['datetime'])
                existing_columns = set(existing_df.columns)
            else:
                existing_df = None
                existing_columns = set()
        
            # Определяем, какие параметры ещё не рассчитаны
            missing_configs = []
            for cfg in params_config:
                col_name = f"{cfg['param']}_{cfg['agg']}_{cfg['level']}"
                if col_name not in existing_columns:
                    missing_configs.append(cfg)
        
            # Если всё уже есть — пропускаем
            if not missing_configs:
                print(f"✅ Все параметры уже есть в {output_file}. Пропускаем.")
                continue
        
            print(f" ➕ Добавляем {len(missing_configs)} новых параметров: {[f'{c['param']}_{c['agg']}_{c['level']}' for c in missing_configs]}")
        
            # Вычисляем ТОЛЬКО недостающие параметры
            new_results = {}
            for cfg in missing_configs:
                col_name = f"{cfg['param']}_{cfg['agg']}_{cfg['level']}"
                new_results[col_name] = []
        
            datetimes = TC['datetime'].values
            rads = TC['rad'].values
            lats = TC['lat'].values if 'lat' in TC.columns else None
            lons = TC['lon']..values if 'lon' in TC.columns else None
        
            for i in range(len(TC)):
                t = TC['datetime'].iloc[i]
                rad = rads[i]
        
                for cfg in missing_configs:
                    param = cfg['param']
                    agg_type = cfg['agg']
                    level = cfg['level']
        
                    if data_type == 'ERA5':
                        if param == 'wspd':
                            val = compute_aggregation_wspd_ERA5(param, t, lats[i], lons[i], rad, level, data_type, km, agg_type)
                        else:
                            val = compute_aggregation_value_ERA5(param, t, lats[i], lons[i], rad, level, data_type, km, agg_type)
                    else:
                        # Для LoRes/SMP используем x, y
                        val = get_mean_value_for_missing(TC, i, level, param, agg_type, data_type, km)
                    
                    col_name = f"{param}_{agg_type}_{level}"
                    new_results[col_name].append(val)
        
            # Объединяем старые и новые данные
            if existing_df is not None:
                df_to_save = existing_df.copy()
                for col, values in new_results.items():
                    df_to_save[col] = values
            else:
                # Если файла не было — создаём новый на основе TC + новых колонок
                df_to_save = TC.copy()
                for col, values in new_results.items():
                    df_to_save[col] = values
        
            # Сохраняем
            df_to_save.to_csv(output_file, index=False)