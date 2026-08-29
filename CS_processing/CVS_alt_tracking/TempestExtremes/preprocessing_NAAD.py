import os
import xarray as xr
from tqdm import tqdm

# Путь к данным
path_init = '/storage/thalassa/users/vkoshkina/data'

# data_type = 'LoRes'

months = range(1, 13)        # от 1 до 12
sigmas = [
    4,
    2,
    0,
]

print('data type: ')
data_type = input() 


path = f'/storage/OPENDATA/NAAD/{data_type}/'
if data_type == 'LoRes':
    hgt_file_path = f'{path}/Invariants/NAAD77km_hgt.nc'
else:
    hgt_file_path = f'{path}/Invariants/NAAD14km_hgt.nc'
    

if not os.path.exists(hgt_file_path):
    raise FileNotFoundError(f"HGT file not found: {hgt_file_path}")

ds_raw = xr.open_dataset(hgt_file_path)


def add_latlon(ds, ds_raw):

    ds = ds.assign_coords({"XLAT": ds_raw["XLAT"], "XLONG": ds_raw["XLONG"]})
        
    return ds


if data_type == 'LoRes' or data_type == 'HiRes':
    level = 12
    years = range(1979, 2019)   
    u_name = 'ue'
    v_name = 've'
elif data_type == 'SMP':
    level = 10
    years = range(2019, 2020)   
    u_name = 'ua'
    v_name = 'va'
elif data_type == 'ERA5':
    level = 500
    years = range(1979, 2025)   
    u_name = 'u'
    v_name = 'v'
    


# Создаём список всех задач (файлов), которые нужно обработать
tasks = []
for sigma in sigmas:
    for year in years:
        for month in months:
            tasks.append((sigma, year, month))

# Обработка с прогресс-баром
for task in tqdm(tasks, desc=f"Обработка файлов {data_type}", unit="файл"):
    sigma, year, month = task

    if data_type == 'LoRes':
        # Формируем пути и имена файлов по-разному для sigma == 0 и других случаев
        if sigma == 0:
            path_dir_raw = f"{path_init}/{data_type}/rortex_from_grads/R2D_level_22_monthly"
            ncfile = f'{path_dir_raw}/R2D_{data_type}_level_22_{year}-{month:02d}.nc'
        else:
            path_dir_raw = f"{path_init}/{data_type}/{data_type}/R2D_{data_type}_level_12_smoothing_sigma_{sigma}"
            ncfile = f'{path_dir_raw}/sigma_{sigma}_R2D_{data_type}_level_12_{year}-{month:02d}.nc'
    elif data_type == 'HiRes':
        if sigma == 0:
            path_dir_raw = f"{path_init}/{data_type}/rortex_from_grads/R2D_level_22_monthly"
            ncfile = f'{path_dir_raw}/R2D_{data_type}_level_22_{year}-{month:02d}.nc'
        else:
            path_dir_raw = f"{path_init}/{data_type}/R2D_{data_type}_level_12_smoothing_sigma_{sigma}"
            ncfile = f'{path_dir_raw}/sigma_{sigma}_R2D_{data_type}_level_12_{year}-{month:02d}.nc'
    elif data_type == 'SMP':
        path_dir_raw = f"{path_init}/{data_type}/R2D_{data_type}_level_10_smoothing_sigma_{sigma}"
        ncfile = f'{path_dir_raw}/sigma_{sigma}_R2D_{data_type}_level_10_{year}-{month:02d}.nc'
    else:
        path_dir_raw = f"{path_init}/{data_type}/R2D_{data_type}_level_500_smoothing_sigma_{sigma}"
        ncfile = f'{path_dir_raw}/sigma_{sigma}_R2D_{data_type}_level_500_{year}-{month:02d}.nc'
        
    # Проверяем существование файла
    if not os.path.exists(ncfile):
        tqdm.write(f'Файл не найден: {ncfile}')
        continue

    try:
        # Открываем датасет
        ds = xr.open_dataset(ncfile)
        ds = add_latlon(ds, ds_raw)
        

        ds = ds.rename({
                    'XTIME': 'time',
                })
        ds = ds.rename({
                    'XLAT': 'lat',
                    'XLONG': 'lon',
            
                })


        # # Берем значения lat и lon для первого временного шага
        # lat_const = ds.lat.isel(time=0)
        # lon_const = ds.lon.isel(time=0)
        
        # # Удаляем старые координаты
        # ds = ds.drop_vars(['lat', 'lon'])
        
        # # Добавляем новые постоянные координаты
        # ds = ds.assign_coords(lat=(['south_north', 'west_east'], lat_const.data),
        #                       lon=(['south_north', 'west_east'], lon_const.data))
        

        # Убираем размерность pressure_level, если она есть
        if 'interp_level' in ds.dims:
            ds_squeezed = ds.squeeze(dim="interp_level", drop=True)
        elif 'bottom_top' in ds.dims:
            ds_squeezed = ds.squeeze(dim="bottom_top", drop=True)
        else:
            ds_squeezed = ds  # если нет такой размерности, просто используем исходный датасет

        # Новая директория для обработанных данных
        path_dir_new = f"{path_init}/TempestExtremes/{data_type}/R2D_{data_type}_level_12_sigma_{sigma}/Input"
        
        # Создаём директорию, если её нет
        os.makedirs(path_dir_new, exist_ok=True)

        # Сохраняем в новый файл с тем же именем в новой директории
        new_ncfile = f'{path_dir_new}/sigma_{sigma}_R2D_{data_type}_level_12_{year}-{month:02d}.nc'
        ds_squeezed.to_netcdf(new_ncfile, format='NETCDF4')

    except Exception as e:
        tqdm.write(f'Ошибка при обработке {ncfile}: {e}')