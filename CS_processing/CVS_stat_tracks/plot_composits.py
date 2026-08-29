import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import glob

from pathlib import Path

import os
import sys
# import shutil

# from geopy.distance import great_circle

# Инициализация путей и параметров
path_init = f'/storage/thalassa/users/vkoshkina'
folder = 'scripts/CS_processing/CVS_tracking/after_70RAE/'
sys.path.insert(2, f'{path_init}/{folder}')

from step_of_tracking import *


path_init = f'/storage/thalassa/users/vkoshkina'
track_folder = f'data/TC_tracks/NAAD_NOAA_with_params_{data_type}_sigma_{sigma}'
files = glob.glob(f'{path_init}/{track_folder}/*.csv')

track_folder = f'data/TC_tracks/NAAD_NOAA_with_params/{data_type}_sigma_{sigma}'
files = glob.glob(f'{path_init}/{track_folder}/*_with_params.csv')
track_folder = f'data/TC_tracks/NAAD_NOAA_with_params/{data_type}_sigma_{sigma}'
files_NOAA = glob.glob(f'{path_init}/{track_folder}/*_NOAA.csv')

TCs = []
TCs_NOAA = []

TC_names = []

for file, file_NOAA in zip(files, files_NOAA):
    df = pd.read_csv(file, parse_dates=['datetime'])
    # df = df.drop(df.columns[0], axis=1)
    
    df_NOAA = pd.read_csv(file_NOAA, parse_dates=['datetime'])
    
    TCs.append(df)
    TCs_NOAA.append(df_NOAA)
    file_name = Path(file).stem        
    TC_names.append(file_name)




def interpolate_to_max_length(df, l_max):
    # Проверяем наличие столбца datetime и преобразуем в таймстемпы
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        datetime_timestamps = df['datetime'].astype(np.int64)  # Преобразуем в таймстемпы
    else:
        datetime_timestamps = np.arange(len(df))  # Индекс вместо даты, если даты нет

    # Создаём новый индекс с линейным распределением на нужную длину
    new_index = np.linspace(0, len(df) - 1, l_max)

    # Интерполируем временные метки отдельно
    new_datetimes = np.interp(new_index, np.arange(len(df)), datetime_timestamps)
    new_datetimes = pd.to_datetime(new_datetimes)  # Преобразуем обратно в datetime

    # Интерполируем числовые данные
    numeric_cols = df.select_dtypes(include=['float32', 'float64', 'int64', 'int32']).columns
    
    df_interpolated = pd.DataFrame(index=new_index)
    for col in numeric_cols:
        if col != 'datetime':  # Пропускаем datetime при интерполяции числовых данных
            df_interpolated[col] = np.interp(new_index, np.arange(len(df)), df[col])

    # Добавляем интерполированные даты в результирующий DataFrame
    df_interpolated['datetime'] = new_datetimes

    return df_interpolated.reset_index(drop=True)

def get_composit(dfs, columns):
    # Объединяем данные в трехмерный массив (n файлов x m строк x k столбцов)
    data = np.array([df[columns].values for df in dfs])

    # Вычисляем среднее и стандартное отклонение вдоль оси файлов (ось 0)
    mean_values = np.nanmedian(data, axis=0)
    std_values = np.nanstd(data, axis=0)

    # Создаём итоговый DataFrame с метками колонок
    mean_df = pd.DataFrame(mean_values, columns=columns)
    std_df = pd.DataFrame(std_values, columns=[f"{col}_std" for col in columns])

    # Объединяем средние значения и стандартные отклонения в одну таблицу
    composite_df = pd.concat([mean_df, std_df], axis=1)
    return composite_df


def plot_dist(composite_CS, var_name, data_type):

    x_Bar_1 = composite_CS['normalized_time']
    v_dist_Bar_1 = composite_CS[f'{var_name}_{data_type}']
    std_1 = composite_CS[f'{var_name}_{data_type}_std']

    fig = plt.figure(figsize=(10,7), dpi=200)
    plt.plot(x_Bar_1, v_dist_Bar_1, label='mean', lw=5, c='deeppink')
    plt.fill_between(x_Bar_1, v_dist_Bar_1-std_1, v_dist_Bar_1+std_1, color='deeppink', alpha=0.1)
    plt.title(f'{var_name}')
    # plt.ylim(vmin,vmax)
    plt.grid()
    plt.legend()
    path_pics_tracks = f'/storage/thalassa/users/vkoshkina/data/pics/TC_tracks/TC_composits'
    plt.savefig(f'{path_pics_tracks}/{var_name}_composit_{data_type}_sigma_{sigma}.png', dpi=300, bbox_inches='tight')

def plot_dist_dual(composite_CS, var_name):
    x_Bar_1 = composite_CS['normalized_time']
    v_dist_Bar_1 = composite_CS[f'{var_name}_LoRes']
    std_1 = composite_CS[f'{var_name}_LoRes_std']

    v_dist_Bar_2 = composite_CS[f'{var_name}_HiRes']
    std_2 = composite_CS[f'{var_name}_HiRes_std']

    # # Добавьте проверку данных
    # print(f"\nDebug {var_name}:")
    # print(f"LoRes std stats: min={std_1.min()}, max={std_1.max()}, NaN={std_1.isna().sum()}")
    # print(f"HiRes std stats: min={std_2.min()}, max={std_2.max()}, NaN={std_2.isna().sum()}")
    
    
    fig = plt.figure(figsize=(10,7), dpi=200)
    plt.plot(x_Bar_1, v_dist_Bar_1, lw=5, c='deeppink', label='LoRes')
    plt.fill_between(x_Bar_1, v_dist_Bar_1-std_1, v_dist_Bar_1+std_1, color='deeppink', alpha=0.1)

    plt.plot(x_Bar_1, v_dist_Bar_2, lw=5, c='tab:cyan', label='HiRes')
    plt.fill_between(x_Bar_1, v_dist_Bar_2-std_2, v_dist_Bar_2+std_2, color='tab:cyan', alpha=0.1)

    plt.title(f'{var_name}')
    plt.legend()
    plt.grid()
    path_pics_tracks = f'/storage/thalassa/users/vkoshkina/data/pics/TC_tracks/TC_composits'
    plt.savefig(f'{path_pics_tracks}/{var_name}_composit_dual_res_sigma_{sigma}.png', dpi=300, bbox_inches='tight')

def plot_dist_trial(composite_CS, var_name, NOAA=None):
    x_Bar_1 = composite_CS['normalized_time']
    v_dist_Bar_1 = composite_CS[f'{var_name}_LoRes']
    std_1 = composite_CS[f'{var_name}_LoRes_std']

    v_dist_Bar_2 = composite_CS[f'{var_name}_HiRes']
    std_2 = composite_CS[f'{var_name}_HiRes_std']

    if var_name == 'mslhf':
        v_dist_Bar_3 = composite_CS[f'mlhf_ERA5']
        std_3 = composite_CS[f'mlhf_ERA5_std']
    elif var_name == 'msshf':
        v_dist_Bar_3 = composite_CS[f'mshf_ERA5']
        std_3 = composite_CS[f'mshf_ERA5_std']
    else:
        v_dist_Bar_3 = composite_CS[f'{var_name}_ERA5']
        std_3 = composite_CS[f'{var_name}_ERA5_std']
    
    fig = plt.figure(figsize=(10,7), dpi=200)

    if NOAA is not None:
        x_Bar_n = NOAA['normalized_time']
        v_dist_Bar_n = NOAA[f'{var_name}']
        std_n = NOAA[f'{var_name}_std']
        plt.plot(x_Bar_n, v_dist_Bar_n, lw=5, c='k', label='NOAA')
        plt.fill_between(x_Bar_n, v_dist_Bar_n-std_n, v_dist_Bar_n+std_n, color='k', alpha=0.1)
    
    plt.plot(x_Bar_1, v_dist_Bar_1, lw=5, c='deeppink', label='LoRes')
    plt.fill_between(x_Bar_1, v_dist_Bar_1-std_1, v_dist_Bar_1+std_1, color='deeppink', alpha=0.1)

    plt.plot(x_Bar_1, v_dist_Bar_2, lw=5, c='tab:cyan', label='HiRes')
    plt.fill_between(x_Bar_1, v_dist_Bar_2-std_2, v_dist_Bar_2+std_2, color='tab:cyan', alpha=0.1)

    plt.plot(x_Bar_1, v_dist_Bar_3, lw=5, c='tab:blue', label='ERA5')
    plt.fill_between(x_Bar_1, v_dist_Bar_3-std_3, v_dist_Bar_3+std_3, color='tab:blue', alpha=0.1)

    plt.title(f'{var_name}')
    plt.legend()
    plt.grid()
    path_pics_tracks = f'/storage/thalassa/users/vkoshkina/data/pics/TC_tracks/TC_composits'
    plt.savefig(f'{path_pics_tracks}/{var_name}_composit_trial_res_sigma_{sigma}.png', dpi=300, bbox_inches='tight')


max_len = 0.
for TC in TCs:
    CVS_len = len(TC)
    if CVS_len > max_len:
        max_len = CVS_len

TCs_interp = []

for TC in TCs:
    TC = interpolate_to_max_length(TC, max_len)
    TCs_interp.append(TC)


param_cols = ['rad', 'crit', 
        'msl_LoRes', 'mslhf_LoRes', 'msshf_LoRes', 'wspd_LoRes',
        'msl_HiRes', 'mslhf_HiRes', 'msshf_HiRes', 'wspd_HiRes',
        'msl_ERA5', 'mlhf_ERA5', 'mshf_ERA5', 'wspd_ERA5'
             ]



composite_CS = get_composit(TCs_interp, param_cols)
composite_CS['normalized_time'] = (composite_CS.index - composite_CS.index.min()) / (composite_CS.index.max() - composite_CS.index.min())

# composite_CS['rad'] = composite_CS['rad']*dist_m/1000
# composite_CS['rad_std'] = composite_CS['rad_std']*dist_m/1000

for param in ['msl_LoRes', 'msl_HiRes', 'msl_ERA5']:
# for param in ['msl_LoRes', 'msl_HiRes']:
    
    
    composite_CS[f'{param}'] = composite_CS[f'{param}']/100
    composite_CS[f'{param}_std'] = composite_CS[f'{param}_std']/100


TCs_interp_NOAA = []

for TC in TCs_NOAA:
    TC = interpolate_to_max_length(TC, max_len)
    TCs_interp_NOAA.append(TC)

param_cols_NOAA = [
            'wnd', 'pres', 'rad_34', 'rad_50', 'rad_64']

composite_CS_NOAA = get_composit(TCs_interp_NOAA, param_cols_NOAA)
composite_CS_NOAA['normalized_time'] = (composite_CS_NOAA.index - composite_CS_NOAA.index.min()) / (composite_CS_NOAA.index.max() - composite_CS_NOAA.index.min())

composite_CS_NOAA['wspd'] = composite_CS_NOAA['wnd']*0.514444
composite_CS_NOAA['wspd_std'] = composite_CS_NOAA['wnd_std']*0.514444

composite_CS_NOAA['msl'] = composite_CS_NOAA['pres']
composite_CS_NOAA['msl_std'] = composite_CS_NOAA['pres_std']


param_cols = [
        'msl', 'mslhf', 'msshf', 'wspd',
        ]

# for col in param_cols:
#     plot_dist(composite_CS, col, data_type=data_type)
#     plot_dist(composite_CS, col, data_type='HiRes')
    
for col in param_cols:    
    # plot_dist_dual(composite_CS, col)  
    if col == 'wspd' or col == 'msl':
        NOAA = composite_CS_NOAA
    else:
        NOAA = None
    plot_dist_trial(composite_CS, col, NOAA=NOAA)

# param_cols = [
#         'msl', 'mlhf', 'mshf', 'wspd',
#         ]

# for col in param_cols:       
#     plot_dist(composite_CS, col, data_type='ERA5')
    
    