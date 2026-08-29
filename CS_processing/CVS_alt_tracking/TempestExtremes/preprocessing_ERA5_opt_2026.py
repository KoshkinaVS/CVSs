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


region = 'Arctic'
# region = 'NA'

if region == 'Arctic':
    # Арктика
    level_hPa = 850
    region_name = f'Arctic_{level_hPa}hPa'
else:
    # Атлантика
    level_hPa = 500
    region_name = f'NA_for_TC_{level_hPa}hPa'

    

def get_days_in_month(year, month):
    """Возвращает количество дней в месяце"""
    _, num_days = calendar.monthrange(year, month)
    return range(1, num_days + 1)


def process_month(args):
    """Обрабатывает все дни одного месяца последовательно"""
    sigma, year, month = args
    
    # Формируем путь к конечному файлу
    path_dir_new = f'{path_init}/TempestExtremes/ERA5/R2D_ERA5_{region_name}_sigma_{sigma}/Input'
    new_ncfile = f'{path_dir_new}/sigma_{sigma}_R2D_ERA5_level_{level_hPa}_{year}-{month:02d}.nc'
    
    # Проверяем, существует ли уже конечный файл
    if os.path.exists(new_ncfile):
        # Проверяем, что файл не поврежден (валидный NetCDF)
        try:
            with xr.open_dataset(new_ncfile) as ds:
                # Проверяем, что есть данные
                if len(ds.time) > 0:
                    print(f"✅ {year}-{month:02d} для sigma={sigma} уже обработан. Пропускаем.")
                    return None
                else:
                    print(f"⚠️ {year}-{month:02d} для sigma={sigma} файл существует, но поврежден. Пересоздаем.")
        except Exception as e:
            print(f"⚠️ {year}-{month:02d} для sigma={sigma} файл поврежден ({e}). Пересоздаем.")
            # Удаляем поврежденный файл
            try:
                os.remove(new_ncfile)
            except:
                pass
    
    print(f"🔄 Обработка {year}-{month:02d} для sigma={sigma}")
    
    # Получаем все дни месяца
    days = get_days_in_month(year, month)
    
    # Создаем список для хранения данных всех дней
    monthly_data = []
    
    for day in days:
        # Формируем путь к исходному файлу (ежедневный)
        if sigma == 0:
            path_dir_raw = f'{path_init}/ERA5/R2D_ERA5_level_{level_hPa}'
            ncfile = f'{path_dir_raw}/R2D_ERA5_level_{level_hPa}_{year}-{month:02d}.nc'
        else:
            # Ежедневные файлы
            path_dir_raw = f'{path_init}/ERA5/R2D_ERA5_{region_name}_sigma_{sigma}/daily/{year}'
            ncfile = f'{path_dir_raw}/sigma_{sigma}_R2D_{year}-{month:02d}-{day:02d}.nc'
        
        # Проверка существования исходного файла
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
                    # Проверяем, что lat и lon - координаты, а не размерности
                    if 'lat' not in ds.dims:
                        ds = ds.expand_dims('lat')
                    if 'lon' not in ds.dims:
                        ds = ds.expand_dims('lon')
                
                monthly_data.append(ds)
                
        except Exception as e:
            return f'Ошибка при обработке {ncfile}: {e}'
    
    # Если есть данные за месяц, объединяем их
    if monthly_data:
        try:
            # Объединяем все дни месяца по времени
            combined_ds = xr.concat(monthly_data, dim='time')
            combined_ds = combined_ds.sortby('time')
            
            # Проверяем финальную структуру данных
            print(f"  {year}-{month:02d}: форма данных {combined_ds.dims}")
            
            # Создаём директорию для сохранения
            os.makedirs(path_dir_new, exist_ok=True)
            
            # Оптимизация записи
            encoding = {
                var: {"zlib": True, "complevel": 1}
                for var in combined_ds.data_vars
            }
            
            # Сохраняем объединенный файл
            combined_ds.to_netcdf(new_ncfile, format='NETCDF4', encoding=encoding)
            
            return None  # успех
            
        except Exception as e:
            return f'Ошибка при сохранении {year}-{month:02d}: {e}'
    
    return None


# Собираем задачи: для каждого месяца каждого года
tasks = []
for sigma in sigmas:
    for year in years:
        for month in months:
            tasks.append((sigma, year, month))

print(f"Всего месяцев для обработки: {len(tasks)}")

# Параллельный запуск
if __name__ == '__main__':
    # Используем меньше процессов для избежания проблем с памятью
    num_workers = min(8, max(1, mp.cpu_count() // 2))
    print(f"Используется {num_workers} процессов")
    errors = []
    skipped = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {executor.submit(process_month, task): task for task in tasks}

        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Обработка месяцев ERA5"):
            result = future.result()
            if result:
                if "уже обработан" in str(result) or "Пропускаем" in str(result):
                    skipped += 1
                else:
                    errors.append(result)

    # Вывод ошибок
    if errors:
        print(f"\n❌ Ошибок: {len(errors)}")
        for err in errors[:10]:  # первые 10
            print(err)
        if len(errors) > 10:
            print(f"... и ещё {len(errors)-10} ошибок")
    else:
        print(f"\n✅ Все файлы успешно обработаны! Пропущено: {skipped} существующих файлов")