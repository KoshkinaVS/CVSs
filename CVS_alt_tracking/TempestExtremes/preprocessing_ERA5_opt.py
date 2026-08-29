import os
import xarray as xr
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Путь к данным
path_init = '/storage/thalassa/users/vkoshkina/data'

# Диапазоны переменных
years = range(1979, 2025)
months = range(1, 13)
sigmas = [0, 2, 4]

def process_file(args):
    sigma, year, month = args

    # Формируем путь
    if sigma == 0:
        path_dir_raw = f'{path_init}/ERA5/R2D_ERA5_level_500'
        ncfile = f'{path_dir_raw}/R2D_ERA5_level_500_{year}-{month:02d}.nc'
    else:
        path_dir_raw = f'{path_init}/ERA5/R2D_ERA5_level_500_sigma_{sigma}'
        ncfile = f'{path_dir_raw}/sigma_{sigma}_R2D_ERA5_level_500_{year}-{month:02d}.nc'

    # Проверка существования
    if not os.path.exists(ncfile):
        return f'Файл не найден: {ncfile}'

    try:
        # Открываем только нужные переменные (ленивая загрузка)
        with xr.open_dataset(ncfile) as ds:
            ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
            ds = ds.set_coords(['lat', 'lon'])

            if 'pressure_level' in ds.dims:
                ds_squeezed = ds.squeeze(dim="pressure_level", drop=True)
            else:
                ds_squeezed = ds

            # Создаём путь для сохранения
            path_dir_new = f'{path_init}/TempestExtremes/ERA5/R2D_ERA5_level_500_sigma_{sigma}/Input'
            os.makedirs(path_dir_new, exist_ok=True)

            new_ncfile = f'{path_dir_new}/sigma_{sigma}_R2D_ERA5_level_500_{year}-{month:02d}.nc'

            # Оптимизация записи
            encoding = {
                var: {"zlib": True, "complevel": 1}
                for var in ds_squeezed.data_vars
            }
            ds_squeezed.to_netcdf(new_ncfile, format='NETCDF4', encoding=encoding)

        return None  # успех
    except Exception as e:
        return f'Ошибка при обработке {ncfile}: {e}'

# Собираем задачи
tasks = [(sigma, year, month) for sigma in sigmas for year in years for month in months]

# Параллельный запуск
if __name__ == '__main__':
    num_workers = max(1, mp.cpu_count() // 2)  # например, 4–8 процессов
    errors = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {executor.submit(process_file, task): task for task in tasks}

        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Обработка файлов ERA5"):
            result = future.result()
            if result:
                errors.append(result)

    # Вывод ошибок
    if errors:
        print("\nОшибки:")
        for err in errors[:10]:  # первые 10
            print(err)
        if len(errors) > 10:
            print(f"... и ещё {len(errors)-10} ошибок")
    else:
        print("\n Все файлы успешно обработаны!")