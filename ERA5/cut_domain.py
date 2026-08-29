import os
import subprocess
from pathlib import Path
from tqdm import tqdm

# Пути к данным
input_dir = Path("/storage/thalassa/DATA/ERA5/PL/NC/uvwth_500hPa")
# input_dir = Path("/storage/thalassa/users/gavr/Coherents/ERA5/DATA/ERA5NC")

path_init = f'/storage/thalassa/users/vkoshkina/data'
output_dir = Path(f'{path_init}/ERA5/ERA5_raw/ERA5_NA_for_TC')
output_dir.mkdir(parents=True, exist_ok=True)  # Создаем папку, если её нет

# Годы и месяцы для обработки
years = range(1979, 2025)  # 2000-2023
# years = range(2010, 2011)  # 2000-2023

year = 2010

months = range(1, 13)      # 1-12
# months = range(8, 10)      # 1-12

days = range(1, 31)      # 1-12

for year in tqdm(years):
    # Цикл по файлам
    for month in tqdm(months):
        # Имя входного файла
        input_file = input_dir / f"era5_uvwth_500hPa_{year}-{month:02d}.nc"
        
        # Имя выходного файла
        output_file = output_dir / f"era5_{year}-{month:02d}_cropped.nc"
        
        # Проверяем, существует ли входной файл
        if not input_file.exists():
            print(f"Файл {input_file} не найден, пропускаем...")
            continue
            
        # Проверяем, существует ли уже выходной файл
        if output_file.exists():
            print(f"Файл {output_file} уже существует, пропускаем...")
            continue
        
        # Запускаем команду
        print(f"Обрабатываем {input_file.name} -> {output_file.name}...")
        # Шаг 1: CDO - обрезка и фильтрация
        subprocess.run([
            "cdo",
            "-setgrid,grid.txt",
            "-selname,u,v",
            "-delete,name=number,step,valid_time",
            
            str(input_file),
            f"temp_{year}-{month:02d}.nc"
        ], check=True)

        subprocess.run([
            "cdo",
            # "-selname,u,v,w,t,z",
            "-selname,u,v",
            "-sellonlatbox,-110,17,0,71",
            # "-delete,name=number,step,valid_time",
            f"temp_{year}-{month:02d}.nc",
            str(output_file),
        ], check=True)
        
        # # Шаг 2: Переименование (CDO или NCO)
        # try:
        #     subprocess.run(["cdo", "chname,valid_time,time", f"temp_{year}-{month:02d}-{day:02d}.nc", str(output_file)], check=True)
        # except subprocess.CalledProcessError:
        #     subprocess.run(["ncrename", "-v", "valid_time,time", f"temp_{year}-{month:02d}-{day:02d}.nc", str(output_file)], check=True)
        
        # Удаляем временный файл
        Path(f"temp_{year}-{month:02d}.nc").unlink()
    
    print("Обработка завершена!")