import pandas as pd
import numpy as np
import xarray as xr
import glob

from tqdm import tqdm

TRACKS_PATH = "/storage/thalassa/users/vkoshkina/data/LoRes/LoRes/EddyClicker_Egor_2010/*.csv"
WRF_PATH = "/storage/NAAD/NAAD/LoRes/2010/wrfout_d01_2010*"

# 1. Загружаем один WRF‑файл (для координат достаточно любой даты года)
wrf_files = sorted(glob.glob(WRF_PATH))
ds = xr.open_dataset(wrf_files[0])   # XLAT/XLONG: (Time, south_north, west_east) [web:16]

# Берём координаты без времени (Time=0)
lat2d = ds["XLAT"].isel(Time=0)
lon2d = ds["XLONG"].isel(Time=0)

# 2. Проходим по всем .csv и дописываем geom
for csv_file in tqdm(glob.glob(TRACKS_PATH)):
    df = pd.read_csv(csv_file)

    # предполагаю, что pxc_ind → индекс по x (west_east), pyc_ind → по y (south_north)
    y = df["pyc_ind"].astype(int).values
    x = df["pxc_ind"].astype(int).values

    # извлекаем по индексам из 2D‑массивов XLAT/XLONG
    df["latitude"] = lat2d.values[y, x]
    df["longitude"] = lon2d.values[y, x]

    # далее мы создаем дополнительные столбцы для параметров!!!!
    # вычисление расстояний от центра до каждой из трёх точек по пифагору: 
    df['distance1'] = np.sqrt((df['px1_ind'] - df['pxc_ind'])**2 + (df['py1_ind'] - df['pyc_ind'])**2)
    df['distance2'] = np.sqrt((df['px2_ind'] - df['pxc_ind'])**2 + (df['py2_ind'] - df['pyc_ind'])**2)
    df['distance3'] = np.sqrt((df['px3_ind'] - df['pxc_ind'])**2 + (df['py3_ind'] - df['pyc_ind'])**2)
    # Вычисляем 4-ю точку как симметричную 2-й точке относительно центра
    df['px4_ind'] = 2 * df['pxc_ind'] - df['px2_ind']
    df['py4_ind'] = 2 * df['pyc_ind'] - df['py2_ind']
    df['distance4'] = np.sqrt((df['px4_ind'] - df['pxc_ind'])**2 + (df['py4_ind'] - df['pyc_ind'])**2)
    
    # вычисление среднего ,макс, мин радиуса для каждого часа  + эксцентриситета. 
    df['mean_radius'] = df[['distance1', 'distance2', 'distance3','distance4']].mean(axis=1)
    df['max_distance'] = df[['distance1', 'distance2', 'distance3','distance4']].max(axis=1)
    df['min_distance'] = df[['distance1', 'distance2', 'distance3','distance4']].min(axis=1)
    df['eccentricity'] = df['min_distance'] / df['max_distance']
    # # теперь посчитаем расстояние треков по точкам:
#     df['dx'] = df['pxc_ind'].diff()  # разница по x между ЦЕНТРАМИ ежечасно ==> у нас столбец с дельтами потом. первый элемент - NaN
#     df['dy'] = df['pyc_ind'].diff()  # то же самое по y
#     df['segment_lengths'] = np.sqrt(df['dx']**2 + df['dy']**2)  # суммируем х и у как вектора ==> у нас чистый путь за час:
    

    df.to_csv(csv_file, index=False)
