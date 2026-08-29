import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import re

def convert_nc_to_txt_format(nc_file_path, output_txt_path, date_str=None, append_mode=False, extr_type='_local'):
    """
    Конвертирует NetCDF файл с данными о локальных экстремумах в текстовый формат.
    
    Parameters:
    -----------
    nc_file_path : str
        Путь к NetCDF файлу
    output_txt_path : str
        Путь для сохранения текстового файла
    date_str : str, optional
        Дата в формате 'YYYY-MM-DD'
    append_mode : bool
        Если True, добавляет данные в конец файла (для группировки по месяцам)
    """
    
    # Открываем NetCDF файл
    ds = xr.open_dataset(nc_file_path)
    
    # Определяем дату
    if date_str is None:
        match = re.search(r'(\d{4})-(\d{2})-(\d{2})', nc_file_path)
        if match:
            year, month, day = match.groups()
            date_str = f"{year}-{month}-{day}"
        else:
            if 'Time' in ds.coords or 'time' in ds.coords:
                time_coord = ds['Time'] if 'Time' in ds.coords else ds['time']
                dt = pd.to_datetime(time_coord.values[0])
                date_str = dt.strftime('%Y-%m-%d')
            else:
                raise ValueError("Не удалось определить дату. Укажите date_str.")
    
    dt = pd.to_datetime(date_str)
    year = dt.year
    month = dt.month
    day = dt.day
    
    # Получаем координаты
    if 'longitude' in ds.coords:
        lon_coord = ds['longitude']
    elif 'lon' in ds.coords:
        lon_coord = ds['lon']
    else:
        raise ValueError("Координата долготы не найдена")
    
    if 'latitude' in ds.coords:
        lat_coord = ds['latitude']
    elif 'lat' in ds.coords:
        lat_coord = ds['lat']
    else:
        raise ValueError("Координата широты не найдена")


    if extr_type == '_local':
        local_extr_data = ds['local_extr_crit'].values  # (time, level, lat, lon)
        local_extr_rad = ds['local_extr_rad_eff'].values  # (time, level, lat, lon)
    else:
        local_extr_data = ds['center'].values  # (time, level, lat, lon)
        local_extr_rad = ds['rad_eff'].values  # (time, level, lat, lon)

    wspd_data = ds['wspd'].values  # (time, level, lat, lon)
    
    
    # Получаем временные шаги
    if 'Time' in ds.coords:
        time_coord = ds['Time']
    elif 'time' in ds.coords:
        time_coord = ds['time']
    else:
        time_coord = None
    
    if time_coord is not None:
        n_times = len(time_coord)
        times = pd.to_datetime(time_coord.values)
    else:
        n_times = 1
        times = [dt]
    
    # Определяем режим записи
    mode = 'a' if append_mode else 'w'
    
    # Открываем файл для записи
    with open(output_txt_path, mode) as f:
        # Проходим по временным шагам (часам)
        for t_idx in range(n_times):
            current_time = times[t_idx] if time_coord is not None else dt
            
            # Извлекаем данные для текущего времени (level=0)
            extr_crit = local_extr_data[t_idx, 0, :, :]
            extr_rad = local_extr_rad[t_idx, 0, :, :]
            extr_wspd = wspd_data[t_idx, 0, :, :]
            
            
            ###### тут берутся только циклонические!!!!
            # Находим индексы ячеек с экстремумами
            valid_mask = (extr_crit > 0) & ~np.isnan(extr_crit)
            valid_indices = np.where(valid_mask)
            
            # Собираем все экстремумы
            extrema_list = []
            for i, j in zip(valid_indices[0], valid_indices[1]):
                lon_idx = j
                lat_idx = i
                
                lon_val = float(lon_coord[lon_idx].values)
                lat_val = float(lat_coord[lat_idx].values)
                rad_val = float(extr_rad[i, j])
                crit_val = float(extr_crit[i, j])
                wspd_val = float(extr_wspd[i, j])
                
                
                extrema_list.append({
                    'lon_idx': lon_idx,
                    'lat_idx': lat_idx,
                    'lon': lon_val,
                    'lat': lat_val,
                    'rad': rad_val,
                    'crit': crit_val,
                    'wspd': wspd_val,
                })
            
            n_extrema = len(extrema_list)
            
            # Записываем строку с метаданными: год, месяц, день, кол-во экстремумов, час
            f.write(f"{year}\t{month}\t{day}\t{n_extrema}\t{current_time.hour}\n")
            
            # Записываем каждый экстремум
            for ext in extrema_list:
                f.write(f"\t{ext['lon_idx']}\t{ext['lat_idx']}\t{ext['lon']:.6f}\t{ext['lat']:.6f}\t{ext['rad']:.6e}\t{ext['crit']:.6e}\t{ext['wspd']:.6e}\n")
    
    ds.close()
    print(f"Добавлено {n_times} временных шагов из {nc_file_path}")

def convert_entire_dataset(nc_dir, output_dir, years_range, eps, min_samples, size_filter, sigma, data_type='ERA5', extr_type='_local'):
    """
    Конвертирует все NetCDF файлы в текстовый формат, группируя по месяцам.
    Каждый временной шаг (час) сохраняется отдельно.
    """
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    pattern = f"sigma_{sigma}_DBSCAN_{data_type}_*.nc"
    nc_files = sorted(Path(nc_dir).glob(pattern))
    
    if not nc_files:
        print(f"Файлы по шаблону {pattern} не найдены в {nc_dir}")
        return
    
    print(f"Найдено {len(nc_files)} файлов")
    
    if years_range is not None:
        start_year, end_year = years_range
        filtered_files = []
        for nc_file in nc_files:
            match = re.search(r'(\d{4})-(\d{2})-(\d{2})', str(nc_file))
            if match:
                year = int(match.group(1))
                if start_year <= year <= end_year:
                    filtered_files.append(nc_file)
        nc_files = filtered_files
        print(f"После фильтрации по годам {start_year}-{end_year}: {len(nc_files)} файлов")
        
        if not nc_files:
            print("Нет файлов для обработки после фильтрации по годам")
            return
    
    files_by_month = {}
    for nc_file in nc_files:
        match = re.search(r'(\d{4})-(\d{2})-(\d{2})', str(nc_file))
        if match:
            year, month, day = match.groups()
            month_key = f"{year}-{month}"
            if month_key not in files_by_month:
                files_by_month[month_key] = []
            files_by_month[month_key].append(nc_file)
    
    for month_key, files in files_by_month.items():
        year, month = month_key.split('-')
        output_filename = f"{data_type}_R2D_extr_{year}-{month}.txt"
        output_path = Path(output_dir) / output_filename
        
        # ПРОВЕРКА: существует ли уже файл?
        if output_path.exists():
            print(f"Файл {output_filename} уже существует. Пропускаем обработку месяца {month_key}")
            continue  # Пропускаем этот месяц
        
        print(f"\nОбработка месяца {month_key}. Файлов: {len(files)}")
        
        for idx, nc_file in enumerate(sorted(files)):
            append_mode = idx > 0
            
            try:
                convert_nc_to_txt_format(
                    str(nc_file),
                    str(output_path),
                    append_mode=append_mode,
                    extr_type=extr_type
                )
            except Exception as e:
                print(f"Ошибка при обработке {nc_file}: {e}")
                # Если произошла ошибка, удаляем частично созданный файл?
                if output_path.exists() and idx == 0:
                    output_path.unlink()  # Удаляем неполный файл
                continue
        
        print(f"Файл сохранен: {output_path}")

def convert_single_file(nc_file_path, output_txt_path):
    """
    Конвертирует один NetCDF файл в текстовый формат.
    """
    convert_nc_to_txt_format(nc_file_path, output_txt_path, append_mode=False)

# Пример использования
if __name__ == "__main__":
    # Параметры
    eps = 2
    min_samples = 4
    size_filter = 10
    size_filter = 25
    
    sigma = 2
    data_type = 'ERA5'

    years = (1979, 2027)
    
    years = (2010, 2010) 
    

    region = 'Arctic'
    region = 'NA'
    
    if region == 'Arctic':
        # Арктика
        level_hPa = 850
        region_name = f'Arctic_{level_hPa}hPa'
    else:
        # Атлантика
        level_hPa = 500
        level_hPa = 850
        region_name = f'NA_for_TC_{level_hPa}hPa'

    extr_type = '_global'
    extr_type = '_local'
    
    
    # Пути
    path_init = '/storage/thalassa/users/vkoshkina/data'
    path_dir_raw = f'{path_init}/{data_type}/DBSCAN_{eps:02d}-{min_samples:02d}-{size_filter:02d}_{region_name}_sigma_{sigma}_rad'
    
    # Выходная директория для текстовых файлов
    output_dir = f'{path_init}/TempestExtremes/{data_type}/R2D_{data_type}_{region_name}_sigma_{sigma}/R2D_txt_files_2010_{size_filter}points{extr_type}'
    
    # Конвертируем все данные
    convert_entire_dataset(
        nc_dir=path_dir_raw,
        output_dir=output_dir,
        years_range=years,
        eps=eps,
        min_samples=min_samples,
        size_filter=size_filter,
        sigma=sigma,
        data_type=data_type,
        extr_type=extr_type
    )
    
    # Или для одного файла:
    # convert_single_file(
    #     '/path/to/file.nc',
    #     '/path/to/output.txt'
    # )