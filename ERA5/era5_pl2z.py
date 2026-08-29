from pathlib import Path
import xarray as xr
import numpy as np
import sys
from scipy.interpolate import interp1d

from tqdm import tqdm

import warnings
# Ignore warnings
warnings.filterwarnings("ignore")

path_init = f'/storage/thalassa/users/vkoshkina'
path_dir_data = f'{path_init}/data'

years = np.arange(1979,2026,1)

# Константа для перевода геопотенциала в высоту (м)
G0 = 9.80665

path_dir_raw = '/storage/thalassa/DATA/ERA5/'
ncfile_surface = f'{path_dir_raw}/era5_surface_geopotential.nc'

ds = xr.open_dataset(ncfile_surface)  # геопотенциал поверхности

# Преобразуем долготу из [0,360) в [-180,180)
lon_values = ds.longitude.values
lon_values = ((lon_values + 180) % 360) - 180
ds = ds.assign_coords(longitude=('longitude', lon_values))
ds = ds.sortby('longitude')

# Выбираем регион
ds_subset = ds.sel(
    latitude=slice(80, 4), 
    longitude=slice(-100, 17)
)

# Геопотенциальная высота поверхности
surface_geopotential_height = ds_subset['z'][0] / G0  # в метрах

def interp_column(var_col, height_col, target_levels):
    """
    Интерполяция переменной с уровней давления на целевые высоты
    
    Parameters:
    -----------
    var_col : array shape (n_levels,)
        Значения переменной на исходных уровнях
    height_col : array shape (n_levels,)
        Высоты (в метрах) для каждого уровня давления
    target_levels : array shape (n_targets,)
        Целевые высоты для интерполяции
    
    Returns:
    --------
    array shape (n_targets,)
        Проинтерполированные значения
    """
    # Удаляем NaN значения перед интерполяцией
    valid_mask = ~(np.isnan(var_col) | np.isnan(height_col))
    
    if np.sum(valid_mask) < 2:
        return np.full(len(target_levels), np.nan)
    
    var_valid = var_col[valid_mask]
    height_valid = height_col[valid_mask]
    
    # Сортируем по возрастанию высоты (важно для интерполяции)
    sort_idx = np.argsort(height_valid)
    height_sorted = height_valid[sort_idx]
    var_sorted = var_valid[sort_idx]
    
    # Создаем интерполяционную функцию
    f = interp1d(
        height_sorted, 
        var_sorted, 
        kind='linear',
        bounds_error=False, 
        fill_value=np.nan
    )
    
    return f(target_levels)

def process_var(ds, var_name, target_levels):
    """
    Обработка одной переменной: интерполяция на целевые уровни высоты
    """
    print(f"    Processing {var_name}..............................")
    
    # Рассчитываем геопотенциальную высоту в метрах из переменной 'z'
    # ds['z'] - это геопотенциал (м²/с²)
    geopot_height = ds['z'] / G0  # теперь в метрах
    
    # Применяем интерполяцию
    ds_out = xr.apply_ufunc(
        interp_column,
        ds[var_name],           # переменная для интерполяции
        geopot_height,          # высоты исходных уровней
        input_core_dims=[['plev'], ['plev']],   # используем plev как имя координаты
        kwargs={'target_levels': target_levels},
        output_core_dims=[['level']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[ds[var_name].dtype],
    )
    
    ds_out.name = var_name
    ds_out = ds_out.assign_coords(level=('level', target_levels))
    
    return ds_out

for year in years:
    # Папка с входными GRIB файлами
    input_dir = Path(f"/storage/thalassa/DATA/ERA5/PL/grib/{year}")
    output_dir = Path(f'{path_dir_data}/ERA5/ERA5_raw/ERA5_heights_NA/{year}')
    
    if not input_dir.exists():
        print(f"Warning: Input directory {input_dir} does not exist! Skipping year {year}")
        continue
    
    # Создаем выходную папку, если её нет
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Находим все GRIB файлы
    grib_files = list(input_dir.glob("*.grib")) + list(input_dir.glob("*.grb"))
    
    if not grib_files:
        print(f"Warning: No GRIB files found in {input_dir}! Skipping year {year}")
        continue
    
    # Обрабатываем каждый файл
    for file_in in tqdm(grib_files, desc=f"Processing {year}"):
        file_out = output_dir / f"{file_in.stem}_interp_to_height.nc"
        print(f"Processing: {file_out}")
        
        try:
            # Открываем GRIB файл
            ds = xr.open_dataset(file_in, engine='cfgrib')
            
            # Преобразуем долготу из [0,360) в [-180,180)
            lon_values = ds.longitude.values
            lon_values = ((lon_values + 180) % 360) - 180
            ds = ds.assign_coords(longitude=('longitude', lon_values))
            ds = ds.sortby('longitude')
            
            # Выбираем регион
            ds_subset = ds.sel(
                latitude=slice(80, 4), 
                longitude=slice(-100, 17)
            )
            
            # Переименовываем координаты для удобства
            # ВАЖНО: не переименовываем isobaricInhPa в 'z', т.к. 'z' уже существует как переменная
            ds_new = ds_subset.rename({
                # 'longitude': 'x',
                # 'latitude': 'y',
                'isobaricInhPa': 'plev',  # теперь координата уровней давления называется plev
            })
            
            # Проверяем, что переменная 'z' (геопотенциал) существует
            if 'z' not in ds_new.data_vars:
                print(f"    Warning: No geopotential (z) found in {file_in.name}, skipping...")
                continue
            
            # Проверяем, что нужные переменные существуют
            required_vars = ['u', 'v']
            missing_vars = [var for var in required_vars if var not in ds_new.data_vars]
            if missing_vars:
                print(f"    Warning: Missing variables {missing_vars} in {file_in.name}, skipping...")
                continue
            
            # Целевые высоты (в метрах)
            target_levels = np.array([1500, 3000, 5000])
            
            # Обрабатываем компоненты ветра
            u_new = process_var(ds_new, "u", target_levels)
            v_new = process_var(ds_new, "v", target_levels)
            
            # Если есть температура, тоже интерполируем
            if 't' in ds_new.data_vars:
                t_new = process_var(ds_new, "t", target_levels)
                vars_to_merge = [u_new, v_new, t_new]
            else:
                vars_to_merge = [u_new, v_new]
            
            # Собираем результат
            print("    Merging down to one xarray.................")
            ds_out = xr.merge(vars_to_merge)
            
            # Добавляем атрибуты для документирования
            ds_out.attrs['description'] = 'ERA5 data interpolated from pressure levels to geopotential height levels'
            ds_out.attrs['interpolation_method'] = 'linear'
            ds_out.attrs['target_levels_meters'] = str(target_levels.tolist())
            ds_out.attrs['geopotential_to_height_factor'] = f'1/{G0}'
            
            # Добавляем координаты времени, широты, долготы из исходных данных
            ds_out = ds_out.assign_coords({
                'time': ds_new.time,
                'latitude': ds_new.latitude,
                'longitude': ds_new.longitude,
            })
            
            # Переставляем размерности для удобства
            ds_out = ds_out.transpose('time', 'level', 'latitude', 'longitude')
            
            # Записываем результат
            encoding = {var: {'zlib': True, 'complevel': 4} for var in ds_out.data_vars}
            ds_out.to_netcdf(file_out, encoding=encoding)
            
            print(f"    Successfully saved to {file_out}")
            
            # Закрываем dataset для освобождения памяти
            ds.close()
            ds_out.close()
            
        except Exception as e:
            print(f"    Error processing {file_in.name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue