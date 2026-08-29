import os
import xarray as xr
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import calendar

# Путь к данным
path_init = '/storage/thalassa/users/vkoshkina/data'

# Диапазоны переменных
years = range(1979, 2026)
months = range(1, 13)
sigmas = [2]  # или [0, 2, 4]


def get_days_in_month(year, month):
    """Возвращает количество дней в месяце"""
    _, num_days = calendar.monthrange(year, month)
    return range(1, num_days + 1)


def process_year(args):
    """Обрабатывает все дни одного года последовательно"""
    sigma, year = args
    
    print(f"Обработка {year} для sigma={sigma}")
    
    # Создаем список для хранения данных всех дней года
    yearly_data = []
    days_processed = 0
    
    for month in months:
        days = get_days_in_month(year, month)
        
        for day in days:
            # Формируем путь к исходному файлу (ежедневный)
            if sigma == 0:
                path_dir_raw = f'{path_init}/ERA5/R2D_ERA5_level_500'
                ncfile = f'{path_dir_raw}/R2D_ERA5_level_500_{year}-{month:02d}.nc'
            else:
                # Ежедневные файлы
                path_dir_raw = f'{path_init}/ERA5/R2D_ERA5_NA_for_TC_500hPa_sigma_{sigma}/daily/{year}'
                ncfile = f'{path_dir_raw}/sigma_{sigma}_R2D_{year}-{month:02d}-{day:02d}.nc'
            
            # Проверка существования
            if not os.path.exists(ncfile):
                return f'Файл не найден: {ncfile}'
            
            try:
                # Открываем ежедневный файл
                with xr.open_dataset(ncfile) as ds:
                    # Переименовываем координаты
                    ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
                    ds = ds.set_coords(['lat', 'lon'])
                    
                    # Убираем размерность pressure_level, если она есть
                    if 'pressure_level' in ds.dims:
                        ds = ds.isel(pressure_level=0, drop=True)
                    elif 'level' in ds.dims:
                        ds = ds.isel(level=0, drop=True)
                    
                    # Убеждаемся, что координаты lat и lon корректны
                    if 'lat' in ds.coords and 'lon' in ds.coords:
                        if 'lat' not in ds.dims:
                            ds = ds.expand_dims('lat')
                        if 'lon' not in ds.dims:
                            ds = ds.expand_dims('lon')
                    
                    yearly_data.append(ds)
                    days_processed += 1
                    
            except Exception as e:
                return f'Ошибка при обработке {ncfile}: {e}'
    
    # Если есть данные за год, объединяем их
    if yearly_data:
        try:
            print(f"  Обработано {days_processed} дней для {year}")
            
            # Объединяем все дни года по времени
            combined_ds = xr.concat(yearly_data, dim='time')
            combined_ds = combined_ds.sortby('time')
            
            # Проверяем финальную структуру данных
            print(f"  {year}: форма данных {combined_ds.dims}, время: {len(combined_ds.time)}")
            
            # Создаём путь для сохранения (погодовой)
            path_dir_new = f'{path_init}/TempestExtremes/ERA5/R2D_ERA5_NA_for_TC_500hPa_sigma_{sigma}/Input_yearly'
            os.makedirs(path_dir_new, exist_ok=True)
            
            # Имя файла для погодового сохранения
            new_ncfile = f'{path_dir_new}/sigma_{sigma}_R2D_ERA5_level_500_{year}.nc'
            
            # Оптимизация записи
            encoding = {
                var: {"zlib": True, "complevel": 1}
                for var in combined_ds.data_vars
            }
            
            # Сохраняем объединенный файл
            combined_ds.to_netcdf(new_ncfile, format='NETCDF4', encoding=encoding)
            
            return None  # успех
            
        except Exception as e:
            return f'Ошибка при сохранении {year}: {e}'
    
    return None


# Собираем задачи: для каждого года
tasks = []
for sigma in sigmas:
    for year in years:
        tasks.append((sigma, year))

print(f"Всего лет для обработки: {len(tasks)}")

# Параллельный запуск
if __name__ == '__main__':
    # Используем меньше процессов для избежания проблем с памятью
    num_workers = min(8, max(1, mp.cpu_count() // 2))
    print(f"Используется {num_workers} процессов")
    errors = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {executor.submit(process_year, task): task for task in tasks}

        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Обработка годов ERA5"):
            result = future.result()
            if result:
                errors.append(result)

    # Вывод ошибок
    if errors:
        print(f"\n❌ Ошибок: {len(errors)}")
        for err in errors[:10]:  # первые 10
            print(err)
        if len(errors) > 10:
            print(f"... и ещё {len(errors)-10} ошибок")
    else:
        print("\n✅ Все файлы успешно обработаны!")