import pandas as pd
import xarray as xr
import glob

TRACKS_PATH = "/storage/thalassa/users/vkoshkina/data/LoRes/LoRes/EddyClicker_tracks/*.csv"
WRF_PATH = "/storage/NAAD/NAAD/LoRes/2010/wrfout_d01_2010*"

# 1. Загружаем один WRF‑файл (для координат достаточно любой даты года)
wrf_files = sorted(glob.glob(WRF_PATH))
ds = xr.open_dataset(wrf_files[0])   # XLAT/XLONG: (Time, south_north, west_east) [web:16]

# Берём координаты без времени (Time=0)
lat2d = ds["XLAT"].isel(Time=0)
lon2d = ds["XLONG"].isel(Time=0)

# 2. Проходим по всем .csv и дописываем lon/lat
for csv_file in glob.glob(TRACKS_PATH):
    df = pd.read_csv(csv_file)

    # предполагаю, что pxc_ind → индекс по x (west_east), pyc_ind → по y (south_north)
    y = df["pyc_ind"].astype(int).values
    x = df["pxc_ind"].astype(int).values

    # извлекаем по индексам из 2D‑массивов XLAT/XLONG [web:8][web:16]
    df["latitude"] = lat2d.values[y, x]
    df["longitude"] = lon2d.values[y, x]

    df.to_csv(csv_file, index=False)
