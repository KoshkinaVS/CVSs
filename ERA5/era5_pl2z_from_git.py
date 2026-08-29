from pathlib import Path
import xarray as xr
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Constants
Re = 6371000  # Earth's radius in meters
g = 9.80665   # Gravity constant in m/s^2

path_dir_raw = '/storage/thalassa/DATA/ERA5/'
ncfile_surface = f'{path_dir_raw}/era5_surface_geopotential.nc'

ds = xr.open_dataset(ncfile_surface)  # геопотенциал поверхности

# Преобразуем долготу из [0,360) в [-180,180)
if 'longitude' in ds.coords:
    lon_values = ds.longitude.values
    lon_values = ((lon_values + 180) % 360) - 180
    ds = ds.assign_coords(longitude=('longitude', lon_values))
    ds = ds.sortby('longitude')

# Выбираем регион
if 'latitude' in ds.dims and 'longitude' in ds.dims:
    ds_subset = ds.sel(
        latitude=slice(80, 4),
        longitude=slice(-100, 17)
    )
else:
    ds_subset = ds

# Геопотенциальная высота поверхности
surface_geopotential_height = ds_subset['z'][0] / g  # в геопотенциальных метрах

# Геометрическая высота поверхности (истинная высота над уровнем моря)
surface_geometric_height = Re * surface_geopotential_height / (Re - surface_geopotential_height)

def pressure_levels_to_geometric_height_custom(ds, target_heights, var_list=None):
    """
    Transform ERA5 pressure level data to geometric height at custom levels.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Pressure Level Dataset containing variables like 'z' (geopotential)
    target_heights : list or array
        Target geometric heights in meters
    var_list : list, optional
        List of variables to interpolate (default: all except 'z')
    
    Returns
    -------
    xarray.Dataset
        Interpolated dataset on target height levels
    """

    
    # Calculate geometric height from geopotential
    geopot_height = ds["z"] / g
    geometric_height = Re * geopot_height / (Re - geopot_height)
    
    # Define variables to interpolate
    if var_list is None:
        var_list = [var for var in ds.data_vars if var != "z"]
    
    # Interpolate each variable
    from scipy.interpolate import interp1d
    
    interpolated_vars = {}
    for var in var_list:
        if var not in ds.data_vars:
            print(f"    Warning: {var} not found in dataset, skipping...")
            continue
        
        print(f"    Interpolating {var}...")
        
        interp_data = xr.apply_ufunc(
            lambda x, y: interp1d(y, x, bounds_error=False, fill_value=np.nan)(target_heights),
            ds[var],
            geometric_height,
            input_core_dims=[["plev"], ["plev"]],
            output_core_dims=[["alt"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[ds[var].dtype],
        )
        
        interp_data.attrs = ds[var].attrs
        interpolated_vars[var] = interp_data
    
    return interpolated_vars, target_heights


# Основной код обработки
path_init = '/storage/thalassa/users/vkoshkina'
path_dir_data = f'{path_init}/data'
years = np.arange(1979, 2026, 1)

# Целевые высоты (геометрические, метры)
TARGET_HEIGHTS = np.array([1500, 3000, 5000])

for year in years:
    input_dir = Path(f"/storage/thalassa/DATA/ERA5/PL/grib/{year}")
    output_dir = Path(f'{path_dir_data}/ERA5/ERA5_raw/ERA5_heights_NA/{year}')
    
    if not input_dir.exists():
        print(f"Warning: {input_dir} does not exist! Skipping {year}")
        continue
    
    output_dir.mkdir(parents=True, exist_ok=True)
    grib_files = list(input_dir.glob("*.grib")) + list(input_dir.glob("*.grb"))
    
    if not grib_files:
        print(f"Warning: No files in {input_dir}! Skipping {year}")
        continue
    
    for file_in in tqdm(grib_files, desc=f"Processing {year}"):
        file_out = output_dir / f"{file_in.stem}_geometric_height.nc"
        
        # Проверяем, не обработан ли уже файл
        if file_out.exists():
            print(f"Skipping {file_in.name} (already processed)")
            continue
            
        print(f"\nProcessing: {file_in.name}")
        
        try:
            # Открываем GRIB файл с обработкой ошибок индекса
            try:
                ds = xr.open_dataset(file_in, engine='cfgrib')
            except EOFError:
                print(f"    Index file corrupted, trying without index...")
                # Пробуем открыть без использования сохраненного индекса
                ds = xr.open_dataset(file_in, engine='cfgrib', backend_kwargs={'indexpath': ''})
            
            # Преобразуем долготу из [0,360) в [-180,180)
            if 'longitude' in ds.coords:
                lon_values = ds.longitude.values
                lon_values = ((lon_values + 180) % 360) - 180
                ds = ds.assign_coords(longitude=('longitude', lon_values))
                ds = ds.sortby('longitude')
            
            # Выбираем регион
            if 'latitude' in ds.dims and 'longitude' in ds.dims:
                ds_subset = ds.sel(
                    latitude=slice(80, 4),
                    longitude=slice(-100, 17)
                )
            else:
                ds_subset = ds
            
            # Переименовываем размерности БЕЗ конфликтов
            rename_dict = {}
            
            # Проверяем существование каждой координаты перед переименованием
            if 'valid_time' in ds_subset.coords and 'time' not in ds_subset.coords:
                rename_dict['valid_time'] = 'time'
            
            if 'latitude' in ds_subset.dims and 'lat' not in ds_subset.dims:
                rename_dict['latitude'] = 'lat'
            
            if 'longitude' in ds_subset.dims and 'lon' not in ds_subset.dims:
                rename_dict['longitude'] = 'lon'
            
            if 'isobaricInhPa' in ds_subset.dims:
                rename_dict['isobaricInhPa'] = 'plev'
            
            # Применяем переименование только если есть что переименовывать
            if rename_dict:
                ds_renamed = ds_subset.rename(rename_dict)
            else:
                ds_renamed = ds_subset
            
            # Удаляем проблемные переменные
            ds_renamed = ds_renamed.drop_vars(['number', 'expver'], errors='ignore')
            
            # Проверяем наличие геопотенциала
            if 'z' not in ds_renamed.data_vars:
                print(f"    Warning: No geopotential (z) in {file_in.name}, skipping...")
                continue
            
            # Убеждаемся, что вертикальная координата называется 'plev'
            if 'plev' not in ds_renamed.dims:
                # Если называется иначе, находим её
                for dim in ds_renamed.dims:
                    if 'isobaric' in dim or 'level' in dim or 'pressure' in dim:
                        ds_renamed = ds_renamed.rename({dim: 'plev'})
                        break
            
            # Определяем переменные для интерполяции (u, v, опционально t)
            vars_to_interp = []
            for var in ['u', 'v', 't', 'q', 'w']:  # добавьте нужные переменные
                if var in ds_renamed.data_vars:
                    vars_to_interp.append(var)
            
            if not vars_to_interp:
                print(f"    Warning: No variables to interpolate in {file_in.name}")
                continue
            
            # Выполняем интерполяцию
            interpolated_vars, heights = pressure_levels_to_geometric_height_custom(
                ds_renamed, TARGET_HEIGHTS, vars_to_interp
            )
            
            if not interpolated_vars:
                print(f"    Warning: No variables interpolated for {file_in.name}")
                continue
            
            # Создаем выходной датасет
            # Определяем координаты
            coords = {}
            
            if 'time' in ds_renamed.coords:
                coords['time'] = ds_renamed.time
            elif 'valid_time' in ds_renamed.coords:
                coords['time'] = ds_renamed.valid_time
            
            coords['alt'] = heights
            
            if 'lat' in ds_renamed.coords:
                coords['lat'] = ds_renamed.lat
            elif 'latitude' in ds_renamed.coords:
                coords['lat'] = ds_renamed.latitude
            
            if 'lon' in ds_renamed.coords:
                coords['lon'] = ds_renamed.lon
            elif 'longitude' in ds_renamed.coords:
                coords['lon'] = ds_renamed.longitude
            
            # Создаем датасет
            ds_out = xr.Dataset(interpolated_vars, coords=coords)
            
            # Переставляем размерности
            available_dims = ['time', 'alt', 'lat', 'lon']
            existing_dims = [dim for dim in available_dims if dim in ds_out.dims]
            ds_out = ds_out.transpose(*existing_dims)
            
            # Добавляем метаданные
            ds_out.attrs['description'] = 'ERA5 data interpolated from pressure levels to geometric height'
            ds_out.attrs['method'] = 'Geometric height using Earth radius correction (Re=6371000m, g=9.80665)'
            ds_out.attrs['target_heights_m'] = str(TARGET_HEIGHTS.tolist())
            ds_out.attrs['source'] = str(file_in.name)
            
            ds_out['alt'].attrs = {
                'standard_name': 'altitude',
                'units': 'm',
                'long_name': 'Geometric Height',
                'positive': 'up',
            }
            
            # Сохраняем с компрессией
            encoding = {var: {'zlib': True, 'complevel': 4} for var in ds_out.data_vars}
            ds_out.to_netcdf(file_out, encoding=encoding)
            
            print(f"    ✓ Saved to {file_out}")
            print(f"      Variables: {list(ds_out.data_vars.keys())}")
            print(f"      Shape: {ds_out.dims}")
            
            # Закрываем датасеты
            ds.close()
            ds_out.close()
            
        except Exception as e:
            print(f"    ✗ Error processing {file_in.name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue