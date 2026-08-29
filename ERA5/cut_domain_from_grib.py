import os
import subprocess
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Основной код
if __name__ == "__main__":
    # Пути к данным
    path_init = '/storage/thalassa/users/vkoshkina'
    path_dir_data = f'{path_init}/data'
    
    # Параметры обработки
    region = [-110, 17, 0, 71]  # [lon_min, lon_max, lat_min, lat_max]
    variables = ["u", "v"]  # Переменные для извлечения: u, v, w, t, z
    
    # Годы для обработки
    years = np.arange(1979, 2026, 1)
    # years = np.arange(1979, 1980, 1)  # Для теста
    
    # Общий цикл по годам
    for year in tqdm(years, desc="Years"):
        input_dir = Path(f"/storage/thalassa/DATA/ERA5/PL/grib/{year}")
        output_dir = Path(f'{path_dir_data}/ERA5/ERA5_raw/ERA5_NA_for_TC/{year}')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Папка для временных файлов
        temp_dir = Path(f"/tmp/cdo_temp_{year}")
        temp_dir.mkdir(exist_ok=True)
        
        # Проверяем существование входной директории
        if not input_dir.exists():
            print(f"Директория {input_dir} не найдена, пропускаем...")
            continue
        
        # Находим все GRIB файлы за год
        grib_files = sorted(input_dir.glob("*.grib"))
        if not grib_files:
            grib_files = sorted(input_dir.glob("*.grb"))
        if not grib_files:
            print(f"GRIB файлы не найдены в {input_dir}, пропускаем...")
            continue
        
        # Цикл по GRIB файлам
        for grib_file in tqdm(grib_files, desc=f"Processing {year}", leave=False):
            # Извлекаем дату из имени файла
            try:
                # Пробуем разные форматы имен
                filename = grib_file.stem
                
                # Формат: era5_pl_2010-01-01
                if 'era5_pl_' in filename:
                    date_part = filename.replace('era5_pl_', '')
                    year_file, month, day = date_part.split('-')
                # Формат: ERA5_20100101
                elif 'ERA5_' in filename:
                    date_part = filename.replace('ERA5_', '')
                    if len(date_part) == 8:
                        year_file = date_part[:4]
                        month = date_part[4:6]
                        day = date_part[6:8]
                else:
                    # Используем имя файла целиком
                    output_file = output_dir / f"{filename}.nc"
                    
                    if output_file.exists():
                        print(f"Файл {output_file} уже существует, пропускаем...")
                        continue
                    
                    # Прямая конвертация
                    subprocess.run([
                        "cdo",
                        "-f", "nc",
                        
                        "-sellonlatbox", f"{region[0]},{region[1]},{region[2]},{region[3]}",
                        "-selname", ",".join(variables),
                        str(grib_file),
                        str(output_file)
                    ], check=True)
                    continue
                    
            except (IndexError, ValueError):
                print(f"Не удалось извлечь дату из {grib_file.name}, пропускаем...")
                continue
            
            # Имя выходного файла
            output_file = output_dir / f"era5_{year_file}-{month}-{day}.nc"
            
            # Проверяем существование выходного файла
            if output_file.exists():
                print(f"Файл {output_file} уже существует, пропускаем...")
                continue
            
            # Временные файлы
            temp_selected = temp_dir / f"selected_{year_file}_{month}_{day}.nc"
            temp_cropped = temp_dir / f"cropped_{year_file}_{month}_{day}.nc"
            
            print(f"Обрабатываем {grib_file.name} -> {output_file.name}...")
            
            try:
                # Шаг 1: Конвертация GRIB в NetCDF и выбор переменных
                subprocess.run([
                    "cdo",
                    "-f", "nc",
                    
                    "-selname", ",".join(variables),
                    str(grib_file),
                    str(temp_selected)
                ], check=True)
                
                # Шаг 2: Обрезка по региону
                subprocess.run([
                    "cdo",
                    "-sellonlatbox", f"{region[0]},{region[1]},{region[2]},{region[3]}",
                    str(temp_selected),
                    str(output_file)
                ], check=True)
                
                # Удаляем временные файлы
                if temp_selected.exists():
                    temp_selected.unlink()
                    
            except subprocess.CalledProcessError as e:
                print(f"Ошибка при обработке {grib_file.name}: {e}")
                # Очистка временных файлов в случае ошибки
                if temp_selected.exists():
                    temp_selected.unlink()
                if temp_cropped.exists():
                    temp_cropped.unlink()
                continue
        
        # Очищаем временную директорию для года
        try:
            temp_dir.rmdir()
        except:
            pass
            
        print(f"Год {year} обработан!")

print("Обработка всех лет завершена!")