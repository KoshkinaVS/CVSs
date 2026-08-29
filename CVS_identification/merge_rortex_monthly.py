import os
import subprocess
from pathlib import Path
import calendar
from tqdm import tqdm

# Конфигурационные параметры
data_type = 'HiRes'  # или 'HiRes'
our_level = 22
path_dir_data = '/storage/thalassa/users/vkoshkina/data/'

# Пути
daily_dir = f'{path_dir_data}/{data_type}/rortex_from_grads/R2D_level_{our_level}'
monthly_dir = f'{path_dir_data}/{data_type}/rortex_from_grads/R2D_level_{our_level}_monthly'

# Создаем директорию для месячных файлов
os.makedirs(monthly_dir, exist_ok=True)

# Годы для обработки
years = range(1979, 2019)

def merge_daily_to_monthly(year, month):
    """Объединяет ежедневные файлы в месячный с помощью CDO"""
    # Получаем количество дней в месяце
    num_days = calendar.monthrange(year, month)[1]
    
    # Формируем список файлов для этого месяца
    daily_files = []
    for day in range(1, num_days + 1):
        file_pattern = f'R2D_{data_type}_level_{our_level}_{year}-{month:02d}-{day:02d}.nc'
        file_path = os.path.join(daily_dir, str(year), file_pattern)
        if os.path.exists(file_path):
            daily_files.append(file_path)
    
    if not daily_files:
        print(f"No files found for {year}-{month:02d}")
        return
    
    # Имя выходного файла
    output_file = os.path.join(monthly_dir, f'R2D_{data_type}_level_{our_level}_{year}-{month:02d}.nc')
    
    # Команда CDO для объединения по времени
    cmd = ['cdo', 'mergetime'] + daily_files + [output_file]
    
    try:
        # Выполняем команду
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Successfully created: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error processing {year}-{month:02d}: {e.stderr}")
        return False

# Обрабатываем все годы и месяцы
for year in tqdm(years, desc="Processing years"):
    for month in range(1, 13):
        merge_daily_to_monthly(year, month)

print("Monthly files creation completed!")