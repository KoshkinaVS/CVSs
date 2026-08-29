import pandas as pd
import os
import glob
from tqdm import tqdm

mini_folder = 'EddyClicker_tracks_Egor_2010_params_r2d'
mini_folder = 'EddyClicker_tracks_Egor_2010_15params_2028-08-10_r2d'


OUTPUT_PATH = f"/storage/thalassa/users/vkoshkina/data/LoRes/LoRes/{mini_folder}"

TRACKS_PATH = f"/storage/thalassa/users/vkoshkina/data/LoRes/LoRes/{mini_folder}/hourly_data/*.csv"


# Список нужных колонок
columns_to_keep = [
    'time', 
    'pxc_ind', 
    'pyc_ind', 
    'latitude', 
    'longitude', 
    'R2D_max', 
    'mean_radius', 
    # 'eccentricity', 
    # 'propagation_speed',
    'velocity',
    # 'angle_radians',
    
    # 'helicity_95', 
    # 'PV_500_mean',
    # 'U10_mean', 
    # 'w_500',
    # 'LH_rad', 
    # 'HFX_rad',
    # 'T2_mean', 
    # 'T2_delta',
    # 'SLP_diff(cent-95)',
    # 'pw_95', 
    # 'mucape_95',
    # 'wind_shear_10m_500'

    'SLP_diff_cent_95',
    'U10_mean', 'U500_mean', 'U850_mean', 'U500_U850_frac', 'U500_minus_U850',
    'PV_850_mean','PV_500_mean',
    'T2_minus_T500_mean', 'T2_minus_T850_mean', 'T850_disp',
    # 'trop_height', 'pbl_height', 
    'pbl_trop_frac',
    'w_850', 'w_500', 
    'RH_850', 'RAIN_HOURLY_sum', 
    'RAIN_HOURLY_95', 
    # 'RAIN_HOURLY_med',
    
]



# Список для хранения DataFrame
dfs = []

# Обрабатываем каждый файл
for file_path in tqdm(glob.glob(TRACKS_PATH)):
    try:
        # Читаем файл, используем только нужные колонки
        df = pd.read_csv(file_path, usecols=columns_to_keep)
        
        # Добавляем путь и имя файла
        df['path'] = file_path
        df['name'] = os.path.basename(file_path)
        
        track_id_str = os.path.basename(file_path).replace('.csv', '').lstrip('0')
        
        if track_id_str == '':
            track_id = -1
        else:
            track_id = int(track_id_str)
    
        df['track_id'] = track_id
        
        
        dfs.append(df)
        print(f"Обработан: {os.path.basename(file_path)} - {len(df)} строк")
        
    except Exception as e:
        print(f"Ошибка при обработке {file_path}: {e}")

# Объединяем все DataFrame
if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Сохраняем результат
    output_file = os.path.join(OUTPUT_PATH, "EddyClicker_tracks_Egor_2010_params_all_in_one.csv")
    combined_df.to_csv(output_file, index=False)
    
    print(f"\nГотово! Объединено {len(dfs)} файлов")
    print(f"Всего строк: {len(combined_df)}")
    print(f"Результат сохранен в: {output_file}")
    print(f"Колонки: {list(combined_df.columns)}")
else:
    print("Не найдено ни одного CSV файла для обработки")