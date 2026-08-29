import os
import xarray as xr
from tqdm import tqdm

# Путь к данным
path_init = '/storage/thalassa/users/vkoshkina/data'

# Диапазоны переменных
years = range(1979, 2025)   # включительно до 2024
months = range(1, 13)        # от 1 до 12
sigmas = [
    0, 
    2, 4]

# Создаём список всех задач (файлов), которые нужно обработать
tasks = []
for sigma in sigmas:
    for year in years:
        for month in months:
            tasks.append((sigma, year, month))

# Обработка с прогресс-баром
for task in tqdm(tasks, desc="Обработка файлов ERA5", unit="файл"):
    sigma, year, month = task

    # Формируем пути и имена файлов по-разному для sigma == 0 и других случаев
    if sigma == 0:
        path_dir_raw = f'{path_init}/ERA5/R2D_ERA5_level_500'
        ncfile = f'{path_dir_raw}/R2D_ERA5_level_500_{year}-{month:02d}.nc'
    else:
        path_dir_raw = f'{path_init}/ERA5/R2D_ERA5_level_500_sigma_{sigma}'
        ncfile = f'{path_dir_raw}/sigma_{sigma}_R2D_ERA5_level_500_{year}-{month:02d}.nc'

    # Проверяем существование файла
    if not os.path.exists(ncfile):
        tqdm.write(f'Файл не найден: {ncfile}')
        continue

    try:
        # Открываем датасет
        ds = xr.open_dataset(ncfile)

        # Переименовываем координаты
        ds = ds.rename({
            'latitude': 'lat',
            'longitude': 'lon'
        })

        # Делаем lat и lon координатами
        ds = ds.set_coords(['lat', 'lon'])

        # Убираем размерность pressure_level, если она есть
        if 'pressure_level' in ds.dims:
            ds_squeezed = ds.squeeze(dim="pressure_level", drop=True)
        else:
            ds_squeezed = ds  # если нет такой размерности, просто используем исходный датасет

        # Новая директория для обработанных данных
        path_dir_new = f'{path_init}/TempestExtremes/ERA5/R2D_ERA5_level_500_sigma_{sigma}/Input'

        # Создаём директорию, если её нет
        os.makedirs(path_dir_new, exist_ok=True)

        # Сохраняем в новый файл с тем же именем в новой директории
        new_ncfile = f'{path_dir_new}/sigma_{sigma}_R2D_ERA5_level_500_{year}-{month:02d}.nc'
        ds_squeezed.to_netcdf(new_ncfile, format='NETCDF4')

    except Exception as e:
        tqdm.write(f'Ошибка при обработке {ncfile}: {e}')