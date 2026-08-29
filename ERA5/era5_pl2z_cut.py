import subprocess
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Константы
G = 9.80665  # Ускорение свободного падения (м/с²)

def process_era5_file(input_file, output_file, region=[-100, 17, 4, 80], 
                      variables=['131', '132'], interp_levels=None):
    """
    Полная обработка ERA5 файла - раздельные команды
    
    Параметры:
    - variables: список кодов переменных (paramId)
        131 = u (U-component of wind)
        132 = v (V-component of wind)  
        129 = z (Geopotential)
    """
    
    temp_files = []
    
    try:
        # Шаг 1: Конвертация GRIB -> NetCDF и выбор переменных по кодам
        temp1 = Path(f"temp1_{output_file.stem}.nc")
        temp_files.append(temp1)
        
        var_codes = ",".join(variables)
        cmd1 = ["cdo", "-f", "nc", f"-selcode,{var_codes}", str(input_file), str(temp1)]
        print(f"Выполняется: {' '.join(cmd1)}")
        subprocess.run(cmd1, check=True, capture_output=True)
        
        # Шаг 2: Обрезка региона
        temp2 = Path(f"temp2_{output_file.stem}.nc")
        temp_files.append(temp2)
        
        lon_min, lon_max, lat_min, lat_max = region
        cmd2 = ["cdo", "-sellonlatbox", f"{lon_min},{lon_max},{lat_min},{lat_max}", 
                str(temp1), str(temp2)]
        print(f"Выполняется: {' '.join(cmd2)}")
        subprocess.run(cmd2, check=True, capture_output=True)
        
        # Шаг 3: Переименование переменных из varXXX в понятные имена
        temp2_renamed = Path(f"temp2_renamed_{output_file.stem}.nc")
        temp_files.append(temp2_renamed)
        
        # Создаем команду для переименования
        rename_cmds = []
        if '131' in variables:
            rename_cmds.extend(["-chname", "var131", "u"])
        if '132' in variables:
            rename_cmds.extend(["-chname", "var132", "v"])
        if '129' in variables:
            rename_cmds.extend(["-chname", "var129", "z"])
        
        if rename_cmds:
            cmd_rename = ["cdo"] + rename_cmds + [str(temp2), str(temp2_renamed)]
            print(f"Выполняется: {' '.join(cmd_rename)}")
            subprocess.run(cmd_rename, check=True, capture_output=True)
        else:
            # Если переименовывать нечего, просто копируем
            temp2_renamed = temp2
        
        # Шаг 4: Интерполяция на нужные уровни (только если есть геопотенциал)
        if interp_levels and '129' in variables:
            temp3 = Path(f"temp3_{output_file.stem}.nc")
            temp_files.append(temp3)
            
            # Сначала пересчитываем геопотенциал в метры
            temp_z = Path(f"temp_z_{output_file.stem}.nc")
            temp_files.append(temp_z)
            
            cmd_z = ["cdo", "-expr", f"height=z/{G}", str(temp2_renamed), str(temp_z)]
            print(f"Выполняется: {' '.join(cmd_z)}")
            subprocess.run(cmd_z, check=True, capture_output=True)
            
            # Интерполяция на целевые высоты
            levels_str = ",".join(map(str, interp_levels))
            # Примечание: intlevel работает только с давлением (hPa)
            # Для высот нужен другой подход
            print(f"Предупреждение: intlevel работает с давлением (hPa), не с высотой (м)")
            print(f"Рекомендуется использовать данные на изобарических уровнях")
            
            # Временный вариант - просто копируем без интерполяции
            temp_z.rename(output_file)
        else:
            # Если интерполяция не нужна, просто переименовываем
            temp2_renamed.rename(output_file)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Ошибка в {input_file.name}: {e}")
        if e.stderr:
            print(f"   {e.stderr.decode()}")
        return False
        
    finally:
        # Очистка временных файлов
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()
                print(f"   Удален временный файл: {temp_file}")

# Основной код
if __name__ == "__main__":
    # Пути к данным
    path_init = '/storage/thalassa/users/vkoshkina'
    path_dir_data = f'{path_init}/data'
    
    # Параметры обработки
    region = [-100, 17, 4, 80]
    
    # ВАЖНО: Используем коды переменных (paramId) вместо имен
    # Из вашего файла: var129, var131, var132
    variables = ['131', '132']  # Только u и v (ветер)
    # variables = ['129', '131', '132']  # Если нужен еще и геопотенциал
    
    levels = [1500, 3000, 5000]  # Высоты для интерполяции (пока не используется)
    
    # Годы для обработки
    years = np.arange(1979, 2026, 1)
    # years = np.arange(1979, 1980, 1)  # Для теста
    
    
    # Общий цикл по годам
    for year in tqdm(years, desc="Years"):
        input_dir = Path(f"/storage/thalassa/DATA/ERA5/PL/grib/{year}")
        output_dir = Path(f'{path_dir_data}/ERA5/ERA5_raw/ERA5_heights_NA/{year}')
        
        if not input_dir.exists():
            print(f"Input directory {input_dir} does not exist! Skipping year {year}")
            continue
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Находим все GRIB файлы
        grib_files = list(input_dir.glob("*.grib")) + list(input_dir.glob("*.grb"))
        
        if not grib_files:
            print(f"No GRIB files found in {input_dir}! Skipping year {year}")
            continue
        
        print(f"\nProcessing year {year}: found {len(grib_files)} files")
        
        # Обрабатываем каждый файл
        for file_in in tqdm(grib_files, desc=f"Files {year}", leave=False):
            output_file = output_dir / f"{file_in.stem}_NA.nc"
            
            if output_file.exists():
                print(f"Skip (exists): {output_file.name}")
                continue
            
            process_era5_file(
                input_file=file_in,
                output_file=output_file,
                region=region,
                variables=variables,
                interp_levels=None  # Пока отключаем интерполяцию
            )
    
    print("\n✅ Обработка всех лет завершена!")