import os
import xarray as xr
from tqdm import tqdm

# Путь к данным
path_init = '/storage/thalassa/users/vkoshkina/data'

# Диапазоны переменных
years = range(1979, 2025)   # включительно до 2024
months = range(1, 13)        # от 1 до 12
sigmas = [0, 
          # 2, 4
         ]

# Список задач
tasks = [(sigma, year, month) for sigma in sigmas for year in years for month in months]

# Обработка с прогресс-баром
for task in tqdm(tasks, desc="Фильтрация R2D > 0", unit="файл"):
    sigma, year, month = task

    # === ЧИТАЕМ ИЗ УЖЕ ОБРАБОТАННОЙ ПАПКИ (Input) ===
    input_dir = f'{path_init}/TempestExtremes/ERA5/R2D_ERA5_level_500_sigma_{sigma}/'
    input_file = f'{input_dir}/sigma_{sigma}_R2D_ERA5_level_500_{year}-{month:02d}.nc'

    # Проверяем, существует ли входной файл
    if not os.path.exists(input_file):
        tqdm.write(f"Файл не найден (пропуск): {input_file}")
        continue

    # === ПУТЬ ДЛЯ ВЫХОДНОГО ФАЙЛА (новая папка) ===
    output_dir = f'{path_init}/TempestExtremes/ERA5/test/Input_C'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/sigma_{sigma}_R2D_ERA5_level_500_{year}-{month:02d}.nc'

    try:
        # Открываем датасет
        ds = xr.open_dataset(input_file)

        # Проверяем наличие переменной R2D
        if 'R2D' not in ds.data_vars:
            tqdm.write(f"Переменная 'R2D' не найдена в: {input_file} → пропуск")
            # Просто копируем файл как есть (или можно пропустить)
            ds.to_netcdf(output_file, format='NETCDF4')
            ds.close()
            continue

        # Создаём модифицированную копию
        ds_mod = ds.copy()

        # Заменяем R2D <= 0 на NaN
        ds_mod['R2D'] = ds_mod['R2D'].where(ds_mod['R2D'] > 0)

        # Сохраняем в новую папку
        ds_mod.to_netcdf(output_file, format='NETCDF4')

        ds.close()  # явное закрытие
        ds_mod.close()

    except Exception as e:
        tqdm.write(f"Ошибка при обработке {input_file}: {e}")

print("✅ Все файлы обработаны и сохранены в Input_C/")