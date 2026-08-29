import pandas as pd
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

def convert_csv_to_txt(csv_file_path, output_txt_file):
    """
    Конвертирует один CSV файл трека в формат TempestExtremes и добавляет в общий txt файл
    """
    # Читаем CSV файл
    df = pd.read_csv(csv_file_path)

    # Переименовываем колонки
    df = df.rename(columns={
        'pxc_ind': 'x',
        'pyc_ind': 'y',
        # 'R2D_max': 'crit',
        'U10_mean_cyclone': 'crit',
        
        'mean_radius': 'rad'
    })
    
    # Проверяем наличие необходимых колонок
    required_cols = ['time', 'x', 'y', 'longitude', 'latitude']
    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: {csv_file_path} missing column {col}")
            return
    
    # Первая точка трека
    first_time = pd.to_datetime(df['time'].iloc[0])
    npoints = len(df)
    
    with open(output_txt_file, 'a') as f:
        # Записываем строку start
        f.write(f"start\t{npoints}\t{first_time.year}\t{first_time.month}\t{first_time.day}\t{first_time.hour}\n")
        
        # Записываем все точки трека
        for idx, row in df.iterrows():
            time_point = pd.to_datetime(row['time'])
            
            x_val = int(row['x']) if not pd.isna(row['x']) else 0
            y_val = int(row['y']) if not pd.isna(row['y']) else 0
            lon_val = row['longitude'] if not pd.isna(row['longitude']) else 0.0
            lat_val = row['latitude'] if not pd.isna(row['latitude']) else 0.0
            rad_val = row['rad'] if 'rad' in row and not pd.isna(row['rad']) else 0.0
            crit_val = row['crit'] if 'crit' in row and not pd.isna(row['crit']) else 0.0
            
            f.write(f"\t{x_val}\t{y_val}\t"
                   f"{lon_val:.6f}\t{lat_val:.6f}\t"
                   f"{crit_val:.6e}\t{rad_val:.6e}\t"
                   f"{time_point.year}\t{time_point.month}\t{time_point.day}\t{time_point.hour}\n")

def batch_convert_csv_to_txt(input_dir, output_dir=None, year=None):
    """
    Конвертирует все CSV файлы в директории в единый годовой txt файл
    """
    # Получаем все CSV файлы
    csv_files = list(Path(input_dir).glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Определяем выходную директорию
    if output_dir:
        txt_dir = Path(output_dir)
    else:
        txt_dir = Path(input_dir) / "txt_yearly"
    
    txt_dir.mkdir(parents=True, exist_ok=True)
    
    # Создаем словарь для output файлов по годам
    output_files = {}
    
    # Один цикл - сразу конвертируем
    for csv_file in tqdm(csv_files, desc="Converting tracks"):
        try:
            # Определяем год из файла
            df_first = pd.read_csv(csv_file, nrows=1)
            df_first = df_first.rename(columns={
                'pxc_ind': 'x',
                'pyc_ind': 'y',
                'R2D_max': 'crit',
                'mean_radius': 'rad'
            })
            first_time = pd.to_datetime(df_first['time'].iloc[0])
            file_year = first_time.year
        except:
            file_year = year if year else 1979
        
        # Получаем или создаем output файл для этого года
        if file_year not in output_files:
            output_file = txt_dir / f"tracks_{file_year}.txt"
            # Если файл существует, удаляем его (чтобы начать с чистого)
            if output_file.exists():
                output_file.unlink()
            output_files[file_year] = output_file
        else:
            output_file = output_files[file_year]
        
        # Конвертируем и записываем
        try:
            convert_csv_to_txt(csv_file, output_file)
        except Exception as e:
            print(f"Error converting {csv_file.name}: {e}")
    
    # Выводим информацию о созданных файлах
    print("\nConversion completed!")
    for yr, output_file in output_files.items():
        print(f"  Year {yr}: {output_file}")

# Использование
path_init = f'/storage/thalassa/users/vkoshkina'
path_dir_data = f'{path_init}/data/'
input_dir = f'{path_dir_data}/LoRes/LoRes/EddyClicker_tracks_Egor_2010_params_r2d/hourly_data'
output_dir = f'{path_dir_data}/LoRes/LoRes/EddyClicker_tracks_Egor_2010_params_r2d/txt_yearly'

# batch_convert_csv_to_txt(input_dir, output_dir=output_dir)

# # 4. Конвертировать и сохранить в другую директорию
batch_convert_csv_to_txt('/storage/kubrick/nikitenko/recalc_testing_3012/hourly_data', output_dir='/storage/thalassa/users/vkoshkina/data/SMP/EddyClicker_txt')