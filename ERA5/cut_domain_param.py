import os
import subprocess
from pathlib import Path
from tqdm import tqdm


params = ['fluxes', 'slp', 't2', 'w10', 'sst', 'precip']
file_names = ['era5_fluxes', 'era5_mslp', 'era5_t2', 'era5_uv10m', 'era5_sst', 'ERA5_precip']


for param, file_name in zip(params, file_names):
    # Пути к данным
    input_dir = Path(f"/storage/thalassa/DATA/ERA5/{param}")
    path_init = f'/storage/thalassa/users/vkoshkina/data'
    output_dir = Path(f'{path_init}/ERA5/ERA5_raw/ERA5_{param}')
    output_dir.mkdir(parents=True, exist_ok=True)  # Создаем папку, если её нет
    
    # Годы и месяцы для обработки
    years = range(1979, 2025)  # 2000-2023
    # years = range(2003, 2004)  # 2000-2023
    
    months = range(1, 13)      # 1-12
    # months = range(12, 13)      # 1-12
    
    
    # Цикл по файлам
    for year in tqdm(years):
        for month in months:
            # Имя входного файла
            input_file = input_dir / f"{file_name}_{year}-{month:02d}.nc"
            
            # Имя выходного файла
            output_file = output_dir / f"era5_{param}_{year}-{month:02d}_cropped.nc"
            
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
                # "-selname,u,v,w,t,z",
                "-sellonlatbox,-100,17,4,80",
                # "-delete,name=number,expver",
                str(input_file),
                f"temp_{year}-{month:02d}.nc"
            ], check=True)
            
            # Шаг 2: Переименование (CDO или NCO)
            try:
                subprocess.run(["cdo", "chname,valid_time,time", f"temp_{year}-{month:02d}.nc", str(output_file)], check=True)
            except subprocess.CalledProcessError:
                subprocess.run(["ncrename", "-v", "valid_time,time", f"temp_{year}-{month:02d}.nc", str(output_file)], check=True)
            
            # Удаляем временный файл
            Path(f"temp_{year}-{month:02d}.nc").unlink()
    
    print("Обработка завершена!")