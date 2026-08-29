# import os
# import pandas as pd
# import xarray as xr
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from matplotlib.patches import Ellipse
# import math
# import glob
# from datetime import datetime
# import matplotlib.dates as mdates

from func_for_stat import *
import os
import glob
import xarray as xr
import pandas as pd
import numpy as np

data_type = 'LoRes'
km = 77

year = 2010

# Конфигурация параметров (только нужные статистики)
PARAMETERS = {
    't2': {
        'name': 'Temperature',
        'unit': '°C',
        'convert_kelvin': True,
        'path': f'/storage/OPENDATA/NAAD/{data_type}/Surface/t2/NAAD{km}km_t2_{year}.nc',
        'var_name': 't2',
        'stats': ['median', 'p90', 'p10', 'std']  # Только эти статистики
    },
    'u10e': {
        'name': 'U-wind',
        'unit': 'm/s',
        'convert_kelvin': False,
        'path': f'/storage/OPENDATA/NAAD/{data_type}/Surface/u10e/NAAD{km}km_u10e_{year}.nc',
        'var_name': 'u10e',
        'stats': ['median', 'p90', 'p10', 'std']
    },
    'v10e': {
        'name': 'V-wind',
        'unit': 'm/s',
        'convert_kelvin': False,
        'path': f'/storage/OPENDATA/NAAD/{data_type}/Surface/v10e/NAAD{km}km_v10e_{year}.nc',
        'var_name': 'v10e',
        'stats': ['median', 'p90', 'p10', 'std']
    },
    'hfx': {
        'name': 'Sensible Heat Flux',
        'unit': 'W/m²',
        'convert_kelvin': False,
        'path': f'/storage/OPENDATA/NAAD/{data_type}/Surface/hfx/NAAD{km}km_hfx_{year}.nc',
        'var_name': 'hfx',
        'stats': ['median', 'p90', 'p10', 'std']
    },
    'lh': {
        'name': 'Latent Heat Flux',
        'unit': 'W/m²',
        'convert_kelvin': False,
        'path': f'/storage/OPENDATA/NAAD/{data_type}/Surface/lh/NAAD{km}km_lh_{year}.nc',
        'var_name': 'lh',
        'stats': ['median', 'p90', 'p10', 'std']
    },
    'msl': {
        'name': 'Mean Sea Level',
        'unit': 'hPa',
        'convert_kelvin': False,
        'path': f'/storage/OPENDATA/NAAD/{data_type}/Surface/msl/NAAD{km}km_msl_{year}.nc',
        'var_name': 'msl',
        'stats': ['median', 'p90', 'p10', 'std']
    },       
}

def compute_basic_stats(data_array, mask, convert_kelvin=False):
    """Вычисляет только median и p90"""
    if data_array is None or mask is None:
        return {'median': np.nan, 'p90': np.nan, 'p10': np.nan, 'std': np.nan}
    
    values = data_array.values if hasattr(data_array, 'values') else data_array
    masked_data = values[mask]
    
    if convert_kelvin:
        masked_data = masked_data - 273.15
    
    return {
        'median': np.nanmedian(masked_data),
        'p90': np.nanpercentile(masked_data, 90),
        'p90': np.nanpercentile(masked_data, 10),
        'p90': np.nanstd(masked_data),
    }

def process_track_data(track_file, temp_file, year, output_dir):
    """Основная функция обработки"""
    track_df = pd.read_csv(track_file)
    results = {}

    temp_data = xr.open_dataset(temp_file)
    lores_xlong, lores_xlat = load_lores_coords(temp_file)
    
    # Инициализация результатов только для нужных статистик
    for param in PARAMETERS:
        for stat in PARAMETERS[param]['stats']:
            results[f'{param.lower()}_{stat}'] = []
    
    # Отдельно для wind_speed (производный параметр)
    results['wind_speed_median'] = []
    results['wind_speed_p90'] = []
    
    print("Processing track points...")
    for i, row in tqdm(track_df.iterrows(), total=len(track_df)):

        row = track_df.iloc[i]
        x_idx = int(row['pxc_ind'])
        y_idx = int(row['pyc_ind'])
        
        lon = lores_xlong.isel(west_east=x_idx, south_north=y_idx).values
        lat = lores_xlat.isel(west_east=x_idx, south_north=y_idx).values
        track_df.at[i, 'longitude'] = lon
        track_df.at[i, 'latitude'] = lat

        date_str = str(row['time']) if 'time' in row else "2010-08-15"

        # Загрузка данных
        data = {}
        for param in PARAMETERS:
            file_path = PARAMETERS[param]['path'].format(year=year)
            try:
                ds = xr.open_dataset(file_path)
                ds = ds.assign_coords({"XTIME": ds['time']})
                data[param] = ds[PARAMETERS[param]['var_name']]
            except:
                data[param] = None
        
        track_time = np.datetime64(pd.to_datetime(date_str))
        time_idx = np.argmin(np.abs(data['XTIME'].values - track_time))
        center_y, center_x = find_hires_indices(float(lon), float(lat), data)
        
        a = float(row.semi_major_axis)
        b = float(row.semi_minor_axis)
        theta = float(row.orientation_rad)
        
        hires_scale_factor = 77/14  # LoRes 77km / HiRes 14km
        a_hires = a * hires_scale_factor
        b_hires = b * hires_scale_factor
        
        south_north_dim = 'south_north' if 'south_north' in data.sizes else 'y'
        west_east_dim = 'west_east' if 'west_east' in data.sizes else 'x'
        mask_shape = (data.sizes[south_north_dim], data.sizes[west_east_dim])
        mask = get_ellipse_mask(center_x, center_y, a_hires, b_hires, theta, mask_shape)
        
        # Расчет статистик
        for param in PARAMETERS:
            stats = compute_basic_stats(
                data[param], 
                mask,
                PARAMETERS[param]['convert_kelvin']
            )
            for stat in PARAMETERS[param]['stats']:
                results[f'{param.lower()}_{stat}'].append(stats[stat])
        
        # Расчет wind_speed
        if data['u10e'] is not None and data['v10e'] is not None:
            wind_speed = np.sqrt(data['u10e']**2 + data['v10e']**2)
            wind_stats = compute_basic_stats(wind_speed, mask)
            results['wind_speed_median'].append(wind_stats['median'])
            results['wind_speed_p90'].append(wind_stats['p90'])
            results['wind_speed_p10'].append(wind_stats['p10'])
            results['wind_speed_std'].append(wind_stats['std'])
        else:
            results['wind_speed_median'].append(np.nan)
            results['wind_speed_p90'].append(np.nan)
            results['wind_speed_p10'].append(np.nan)
            results['wind_speed_std'].append(np.nan)
            
    
    # Добавляем результаты в DataFrame
    for col in results:
        track_df[col] = results[col]
    
    # Сохранение
    os.makedirs(output_dir, exist_ok=True)
    track_df.to_csv(os.path.join(output_dir, 'track_stats.csv'), index=False)
    
    return track_df


"""Main function to run the analysis."""

data_dir = "./"
temp_file = f"/storage/OPENDATA/NAAD/{data_type}/Surface/t2/NAAD{km}km_t2_{year}.nc"
track_file = os.path.join(data_dir, "earl_500hpa.csv")
hires_dir = f"/storage/NAAD/NAAD/{data_type}/{year}"
output_dir = f"output/{data_type}_analysis"

result_df = process_track_data(track_file, temp_file, year, output_dir)
print(f"{data_type} analysis completed successfully!")
