import xarray as xr
import os
from datetime import datetime
from tqdm import tqdm

# Пути к данным
path_kub1 = '/storage/kubrick/vkoshkina/data/HiRes/R2D_HiRes_level_12'
path_kub2 = '/storage/kubrick/vkoshkina/data/HiRes/DBSCAN_with_uv_time'

# Путь к данным
path_init = '/storage/thalassa/users/vkoshkina/data'

# data_type = 'LoRes'

months = range(1, 13)        # от 1 до 12
sigmas = [
    0, 
    # 2, 4
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
            # Проверяем, существует ли уже обработанный файл
            path_dir_new = f"{path_init}/TempestExtremes/{data_type}/R2D_{data_type}_level_12_sigma_{sigma}/Input"
            new_ncfile = f'{path_dir_new}/sigma_{sigma}_R2D_{data_type}_level_12_{year}-{month:02d}.nc'
            
            # Добавляем задачу только если файл еще не существует
            if not os.path.exists(new_ncfile):
                tasks.append((sigma, year, month))
            else:
                print(f"Файл уже существует, пропускаем: {new_ncfile}")

print(f"Найдено {len(tasks)} файлов для обработки")

# Обработка с прогресс-баром
for task in tqdm(tasks, desc=f"Обработка файлов {data_type}", unit="файл"):
    sigma, year, month = task
    
    # Проверяем еще раз на случай параллельного выполнения
    path_dir_new = f"{path_init}/TempestExtremes/{data_type}/R2D_{data_type}_level_12_sigma_{sigma}/Input"
    new_ncfile = f'{path_dir_new}/sigma_{sigma}_R2D_{data_type}_level_12_{year}-{month:02d}.nc'
    
    if os.path.exists(new_ncfile):
        print(f"Файл уже существует, пропускаем: {new_ncfile}")
        continue
        
    try:
        # Открываем первый датасет
        ds1 = xr.open_dataset(f'{path_kub1}/R2D_HiRes_level_12_{year}-{month:02d}.nc')
        
        # Открываем второй датасет
        ds2 = xr.open_dataset(f'{path_kub2}/DBSCAN_HiRes_level_12_{year}-{month:02d}.nc')
        
        # Берем нужные переменные из второго датасета
        ds2_subset = ds2[['ue', 've']]
        
        # Объединяем R2D из первого датасета с ue, ve из второго
        combined_ds = xr.merge([
            ds1[['R2D']],
            ds2_subset,
            ds_raw[['hgt']],
        ])
        
        # Переименовываем координаты
        combined_ds = combined_ds.rename({
            'Time': 'time',
            'XLAT': 'lat', 
            'XLONG': 'lon'
        })
        # Переименовываем координаты
        combined_ds = combined_ds.rename({
            'hgt': 'HGT',
        })
        
        # Убираем размерность pressure_level, если она есть
        if 'interp_level' in combined_ds.dims:
            combined_ds = combined_ds.squeeze(dim="interp_level", drop=True)
        elif 'bottom_top' in combined_ds.dims:
            combined_ds = combined_ds.squeeze(dim="bottom_top", drop=True)
        else:
            combined_ds = combined_ds  # если нет такой размерности, просто используем исходный датасет
        
        if 'XTIME' in combined_ds.variables:
            combined_ds = combined_ds.drop_vars(['XTIME'])


        # # Берем значения lat и lon для первого временного шага
        # lat_const = combined_ds.lat.isel(time=0)
        # lon_const = combined_ds.lon.isel(time=0)
        
        # # Удаляем старые координаты
        # combined_ds = combined_ds.drop_vars(['lat', 'lon'])
        
        # # Добавляем новые постоянные координаты
        # combined_ds = combined_ds.assign_coords(lat=(['south_north', 'west_east'], lat_const.data),
        #                       lon=(['south_north', 'west_east'], lon_const.data))
        
        
        # Устанавливаем координаты lat и lon
        combined_ds = combined_ds.set_coords(['lat', 'lon'])
        
        # Добавляем атрибуты для соответствия целевому формату
        combined_ds.attrs.update({
            'CREATED': datetime.now().strftime('%Y-%m-%d_%H:%M:%S'),
            'history': f'Processed on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        })


        # Создаём директорию, если её нет
        os.makedirs(path_dir_new, exist_ok=True)

        # Сохраняем в новый файл с тем же именем в новой директории
        combined_ds.to_netcdf(new_ncfile, format='NETCDF4')
        
        # Закрываем датасеты для освобождения памяти
        ds1.close()
        ds2.close()
        del combined_ds
        
    except FileNotFoundError as e:
        print(f"Файл не найден: {e}")
        continue
    except Exception as e:
        print(f"Ошибка при обработке {year}-{month:02d}: {e}")
        continue

