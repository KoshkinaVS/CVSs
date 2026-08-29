import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import re
import os

def convert_nc_to_txt_format(nc_file_path, output_txt_path, circ='C', date_str=None, append_mode=False):
    """
    Конвертирует NetCDF файл с данными о локальных экстремумах в текстовый формат.
    
    Parameters:
    -----------
    nc_file_path : str
        Путь к NetCDF файлу
    output_txt_path : str
        Путь для сохранения текстового файла
    date_str : str, optional
        Дата в формате 'YYYY-MM-DD' (игнорируется, т.к. берется из данных)
    append_mode : bool
        Если True, добавляет данные в конец файла (для группировки по месяцам)
    """
    
    # Открываем NetCDF файл
    ds = xr.open_dataset(nc_file_path)
    
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
    
    # Получаем данные
    local_extr_data = ds['local_extr_crit'].values  # (time, level, lat, lon)
    local_extr_rad = ds['local_extr_rad_eff'].values  # (time, level, lat, lon)
    
    # Получаем временную координату
    if 'Time' in ds.coords:
        time_coord = ds['Time']
    elif 'time' in ds.coords:
        time_coord = ds['time']
    else:
        raise ValueError("Временная координата не найдена")
    
    # Конвертируем времена в pandas datetime
    times = pd.to_datetime(time_coord.values)
    n_times = len(times)
    
    # Определяем режим записи
    mode = 'a' if append_mode else 'w'
    
    # Открываем файл для записи
    with open(output_txt_path, mode) as f:
        # Проходим по каждому временному шагу
        for t_idx in range(n_times):
            current_time = times[t_idx]
            
            # Извлекаем реальную дату и время из временной координаты
            current_year = current_time.year
            current_month = current_time.month
            current_day = current_time.day
            current_hour = current_time.hour
            
            # Извлекаем данные для текущего времени (level=0)
            extr_crit = local_extr_data[t_idx, 0, :, :]
            extr_rad = local_extr_rad[t_idx, 0, :, :]

            if circ == 'C':
                # Находим индексы ячеек с экстремумами (только циклонические!)
                valid_mask = (extr_crit > 0) & ~np.isnan(extr_crit)
            else:
                valid_mask = (extr_crit < 0) & ~np.isnan(extr_crit)
            
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
                
                extrema_list.append({
                    'lon_idx': lon_idx,
                    'lat_idx': lat_idx,
                    'lon': lon_val,
                    'lat': lat_val,
                    'rad': rad_val,
                    'crit': crit_val
                })
            
            n_extrema = len(extrema_list)
            
            # Записываем строку с метаданными: год, месяц, день, кол-во экстремумов, час
            f.write(f"{current_year}\t{current_month}\t{current_day}\t{n_extrema}\t{current_hour}\n")
            
            # Записываем каждый экстремум
            for ext in extrema_list:
                f.write(f"\t{ext['lon_idx']}\t{ext['lat_idx']}\t{ext['lon']:.6f}\t{ext['lat']:.6f}\t{ext['rad']:.6e}\t{ext['crit']:.6e}\n")
    
    ds.close()
    print(f"Обработано {n_times} временных шагов из {nc_file_path}")

def convert_entire_dataset(nc_dir, output_dir, years_range, eps, min_samples, size_filter, sigma, data_type='GLORYS', level=None):
    """
    Конвертирует все NetCDF файлы в текстовый формат, группируя по годам.
    Каждый временной шаг сохраняется с правильной датой из данных.
    """
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Адаптируем шаблон под новый формат
    if level is not None:
        pattern = f"sigma_{sigma}_DBSCAN_{data_type}_level_{level}_*.nc"
    else:
        pattern = f"sigma_{sigma}_DBSCAN_{data_type}_*.nc"
    
    nc_files = sorted(Path(nc_dir).glob(pattern))
    
    if not nc_files:
        print(f"Файлы по шаблону {pattern} не найдены в {nc_dir}")
        return
    
    print(f"Найдено {len(nc_files)} файлов")
    
    # Фильтрация по годам (извлекаем год из имени файла)
    if years_range is not None:
        start_year, end_year = years_range
        filtered_files = []
        for nc_file in nc_files:
            match = re.search(r'(\d{4})\.nc$', str(nc_file))
            if match:
                year = int(match.group(1))
                if start_year <= year <= end_year:
                    filtered_files.append(nc_file)
        nc_files = filtered_files
        print(f"После фильтрации по годам {start_year}-{end_year}: {len(nc_files)} файлов")
        
        if not nc_files:
            print("Нет файлов для обработки после фильтрации по годам")
            return
    
    # Группируем по годам (из имени файла)
    files_by_year = {}
    for nc_file in nc_files:
        match = re.search(r'(\d{4})\.nc$', str(nc_file))
        if match:
            year = match.group(1)
            if year not in files_by_year:
                files_by_year[year] = []
            files_by_year[year].append(nc_file)
    
    for year, files in files_by_year.items():
        # Формируем имя выходного файла
        output_filename = f"{data_type}_R2D_extr_{year}.txt"
        output_path = Path(output_dir) / output_filename
        
        # Проверка: существует ли уже файл?
        if output_path.exists():
            print(f"Файл {output_filename} уже существует. Пропускаем обработку года {year}")
            continue
        
        print(f"\nОбработка года {year}. Файлов: {len(files)}")
        
        for idx, nc_file in enumerate(sorted(files)):
            append_mode = idx > 0
            
            try:
                # Не передаем date_str - берем из данных
                convert_nc_to_txt_format(
                    str(nc_file),
                    str(output_path),
                    append_mode=append_mode
                )
            except Exception as e:
                print(f"Ошибка при обработке {nc_file}: {e}")
                if output_path.exists() and idx == 0:
                    output_path.unlink()
                continue
        
        print(f"Файл сохранен: {output_path}")
        
def convert_single_file(nc_file_path, output_txt_path, circ='C'):
    """
    Конвертирует один NetCDF файл в текстовый формат.
    """
    convert_nc_to_txt_format(nc_file_path, output_txt_path, circ=circ, append_mode=False)

# Пример использования
if __name__ == "__main__":
    # Параметры
    eps = 2
    min_samples = 4
    size_filter = 10
    sigma = 0
    data_type = 'GLORYS'
    data_type = 'ALT'
    

    years = np.arange(2023,2025)

    region = 'BarKara'


    region = 'LV'

    circ = 'AC'
    circ = 'C'
    

    name_crit = 'R2D'
    name_crit = 'Q'
    name_crit = 'lambda2'
    
    threshholded = True

    if threshholded:
        name_crit = f'{name_crit}_th'
    
    if region == 'BarKara':
        level = 8
        level = 15
        level = 0
        
        region_name = f'BarKara_level_{level}'


    in_file_name_list = ['LV_alt_0125deg_2008-2009', 'LV_alt_Novoselova_0125', 'LV_alt_0125deg_2014',
                         # 'LV_alt_025deg_2008-2009', 'LV_alt_Novoselova_025', 'LV_alt_025deg_2014',
                      ]

    
    # Пути
    path_init = '/storage/thalassa/users/vkoshkina/data'
    # path_dir_raw = f'{path_init}/{data_type}/DBSCAN_{eps:02d}-{min_samples:02d}-{size_filter:02d}_{region_name}_sigma_{sigma}'
    # path_dir_raw = f'{path_init}/ocean_eddies/DBSCAN_{eps:02d}-{min_samples:02d}-{size_filter:02d}_sigma_{sigma}/'

    path_dir_raw = f'{path_init}/ocean_eddies/LV_DBSCAN/DBSCAN_{eps:02d}-{min_samples:02d}-{size_filter:02d}_sigma_{sigma}/'
    
    
    # # Конвертируем все данные
    # convert_entire_dataset(
    #     nc_dir=path_dir_raw,
    #     output_dir=output_dir,
    #     years_range=years,
    #     eps=eps,
    #     min_samples=min_samples,
    #     size_filter=size_filter,
    #     sigma=sigma,
    #     data_type=data_type,
    #     level=level
    # )
    
    for in_file_name in in_file_name_list:
        # Выходная директория для текстовых файлов
        output_dir = f'{path_init}/TempestExtremes/{data_type}/LV/{in_file_name}/{name_crit}_txt_files'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        output_file = f'{output_dir}/{circ}_{in_file_name}_{name_crit}_extr.txt'
        # if os.path.exists(output_file):
        #     print(f'{in_file_name} уже обработан, пропускаем')
        #     continue
        
        convert_single_file(
            f'{path_dir_raw}/sigma_{sigma}_{name_crit}_DBSCAN_{in_file_name}.nc',
            output_file,
            circ=circ,
        )