import pandas as pd
import numpy as np
import xarray as xr
import glob

from tqdm import tqdm

# TRACKS_PATH = "/storage/thalassa/users/vkoshkina/data/LoRes/LoRes/EddyClicker_tracks_Egor_2010_params_r2d/hourly_data/*.csv"

TRACKS_PATH = "/storage/thalassa/users/vkoshkina/data/LoRes/LoRes/EddyClicker_tracks_Egor_2010_15params_2026-08-11_r2d/hourly_data/*.csv"


data_type='LoRes'


def calculate_velocity_rolling(track, data_type='LoRes'):
    """
    Использование pandas rolling для максимальной производительности.
    """
    length = len(track)
    
    if length == 0:
        return np.array([]), np.array([])
    
    if length == 1:
        return np.array([0.0]), np.array([0.0])
    
    # Координаты
    x = track['pxc_ind'].values.astype(float)
    y = track['pyc_ind'].values.astype(float)
    
    # Расстояния между точками
    dx = np.diff(x)
    dy = np.diff(y)
    distances = np.sqrt(dx**2 + dy**2)
    distances = np.insert(distances, 0, 0)
    
    # Скользящее среднее с окном 5 (центрированное)
    # min_periods=2 для учета границ
    rolling_mean = pd.Series(distances).rolling(
        window=5, 
        center=True, 
        min_periods=2
    ).mean()
    
    velocities = rolling_mean.values
    
    # Коэффициент перевода
    if data_type == 'ERA5':
        pixel_to_km = 111 * 0.25
        time_step = 1
    elif data_type == 'SMP':
        pixel_to_km = 6
        time_step = 1
    else:  # LoRes
        pixel_to_km = 77
        time_step = 3
    
    velocities = (velocities * pixel_to_km) / time_step
#     velocities[0] = 0.0
    velocities = np.nan_to_num(velocities, nan=0.0)
    
    # ========== УГЛЫ ==========
    dx_vec = np.diff(x, prepend=x[0])
    dy_vec = np.diff(y, prepend=y[0])
    
    dx_next = np.diff(x, append=x[-1])
    dy_next = np.diff(y, append=y[-1])
    dx_next[-1] = dx_vec[-1]
    dy_next[-1] = dy_vec[-1]
    
    cross = dx_vec * dy_next - dy_vec * dx_next
    dot = dx_vec * dx_next + dy_vec * dy_next
    angles = np.arctan2(cross, dot)
    angles[0] = 0.0
    angles[-1] = 0.0
    angles = np.nan_to_num(angles, nan=0.0)
    
    return velocities, angles

# 2. Проходим по всем .csv и дописываем vel
for csv_file in tqdm(glob.glob(TRACKS_PATH)):
    df = pd.read_csv(csv_file)
    
    velocities, angles = calculate_velocity_rolling(df, data_type)

    # Добавляем в DataFrame
    df['velocity'] = velocities
    df['angle_radians'] = angles
    
    df.to_csv(csv_file, index=False)