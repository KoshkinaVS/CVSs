from matplotlib import pyplot as plt
import pandas as pd 
import numpy as np
import math
import xarray as xr

import datetime
from datetime import timedelta

from tqdm import tqdm

import scipy as sp


from geopy.distance import geodesic
from geopy.distance import great_circle
from concurrent.futures import ThreadPoolExecutor

import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature

from shapely.geometry import Polygon
import matplotlib.patches as mpatches
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmaps

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import sys
import os

# Инициализация путей и параметров
path_init = f'/storage/thalassa/users/vkoshkina'
folder = 'scripts/CS_processing/CVS_tracking/after_70RAE/'
sys.path.insert(2, f'{path_init}/{folder}')

# path_dir_raw = f'/storage/OPENDATA/NAAD/LoRes/Surface/msl'
# ncfile = f'{path_dir_raw}/NAAD77km_msl_2010.nc'

# ds = xr.open_dataset(f'{ncfile}') #['HGT'][0]

level = 10

path_dir_data_ds = f"{path_init}/data/SMP/DBSCAN_02-04-10_with_wspd_smoothing/2019"

ncfile = f'{path_dir_data_ds}/sigma_2_DBSCAN_SMP_level_{level}_2019-02-05.nc'
ds = xr.open_dataset(ncfile)


def load_season_tracks(CS_tracks_list, year, months, path_data, time_th=3): 
    null_files = []
    
    for month in tqdm(months, desc='Loading NAAD data...'):
        if os.path.exists(path_data):
            ls = list(sorted(Path(f"{path_data}/{year}-{month:02d}").glob(f'*_track_*.csv')))
            
            if len(ls) != 0:
                for ii,ifile in enumerate(ls):
                    df = pd.read_csv(ifile, parse_dates=['datetime'])
                    if len(df) > time_th: 
                        df = df.drop(df.columns[0], axis=1)
                        df = df.dropna(how='any')
                        
                        CS_tracks_list.append(df)
                    else:
                        null_files.append(ifile)
    print(f'too short NAAD tracks in {year}: {len(null_files)}')
    
    return CS_tracks_list

def load_season_tracks_NOAA(CS_tracks_list, path_data, year, time_th=3): 
    
    null_files = []
    name_NOAA = f'AL*{year}.csv'
    ls = list(sorted(Path(f"{path_data}/").glob(name_NOAA)))

    if len(ls) != 0:
        for ii,ifile in enumerate(ls):
            df = pd.read_csv(ifile, encoding='ascii', parse_dates=['datetime'], 
                             dtype={'status': str, 'Name': str},
                             converters={
                                            'lat': convert_coord,  # Преобразуем широту
                                            'lon': convert_coord   # Преобразуем долготу
                                        })
            if len(df) > time_th:
                df = df.drop(df.columns[0], axis=1)
                CS_tracks_list.append(df)
            else:
                null_files.append(ifile)
    
    return CS_tracks_list
    
def load_season_tracks_EddyClicker(CS_tracks_list, path_data, year, time_th=3): 
    
    null_files = []
    name_NOAA = f'*.csv'
    ls = list(sorted(Path(f"{path_data}/").glob(name_NOAA)))

    if len(ls) != 0:
        for ii, ifile in enumerate(ls):
            df = pd.read_csv(ifile, parse_dates=['time'])
            
            # Переименовываем колонки
            df = df.rename(columns={
                'time': 'datetime',
                'pxc_ind': 'x',
                'pyc_ind': 'y'
            })
            
            if len(df) > time_th:
                CS_tracks_list.append(df)
            else:
                null_files.append(ifile)
    
    print(f'too short EC tracks in {year}: {len(null_files)}')
    
    return CS_tracks_list
    
# Функция преобразования координат
def convert_coord(coord):
    value = float(coord[:-1])  # Числовая часть
    direction = coord[-1]      # Последний символ (N, S, E, W)
    if direction in ['S', 'W']:
        value = -value  # Отрицательное значение для юга и запада
    return value

def get_tracks_dist(track_real, TC, x_name='lon', y_name='lat', manual_data='NOAA'):
    # Фильтруем данные по совпадающим датам
    TC_for_real = TC[np.isin(TC.datetime.values, track_real.datetime.values)]
    track_real_for_TC = track_real[np.isin(track_real.datetime.values, TC_for_real.datetime.values)]
    
    # Получаем пары декартовых координат (x, y)
    coord_pair = [(x, y) for x, y in zip(TC_for_real[x_name].values, TC_for_real[y_name].values)]
    coord_pair_real = [(x, y) for x, y in zip(track_real_for_TC[x_name].values, track_real_for_TC[y_name].values)]
    
    # Вычисляем евклидовы расстояния
    dist = []
    for i in range(len(coord_pair)):

        if manual_data == 'NOAA':
            distance = great_circle(coord_pair[i], coord_pair_real[i]).km
        else:
            x1, y1 = coord_pair[i]
            x2, y2 = coord_pair_real[i]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
        dist.append(distance)
    
    return dist

def calculate_rmcpd(track1, track2, time_col='datetime', lat_col='lat', lon_col='lon'):
    """
    Вычисляет reduced Mean of Closest Point Distances (rMCPD) между двумя треками.
    
    Parameters:
    -----------
    track1, track2 : pandas.DataFrame
        Треки с колонками времени, широты и долготы
    time_col : str
        Название колонки с временем
    lat_col, lon_col : str
        Названия колонок с координатами
        
    Returns:
    --------
    float
        rMCPD значение в километрах
    """
    # Объединяем треки по времени
    merged = pd.merge(track1, track2, on=time_col, suffixes=('_1', '_2'))
    
    if len(merged) == 0:
        return float('inf')  # Нет совпадающих временных точек
    
    # Вычисляем расстояния для каждой временной точки
    distances = []
    for _, row in merged.iterrows():
        coord1 = (row[f'{lat_col}_1'], row[f'{lon_col}_1'])
        coord2 = (row[f'{lat_col}_2'], row[f'{lon_col}_2'])
        distance = great_circle(coord1, coord2).km
        distances.append(distance)
    
    # Вычисляем reduced mean (исключаем выбросы)
    distances = np.array(distances)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    # Исключаем расстояния, превышающие mean + 2*std
    filtered_distances = distances[distances <= mean_dist + 2 * std_dist]
    
    if len(filtered_distances) == 0:
        return mean_dist  # Возвращаем обычное среднее, если все точки - выбросы
    
    return np.mean(filtered_distances)


def find_matching_tracks_fast(TC_dict, key, CS_track_NOAA, NAAD_tracks, max_distance_km=5, max_time_diff=3, similar_steps=5):
    # Предварительная обработка NOAA трека
    al_data = CS_track_NOAA.copy()
    al_data['datetime'] = pd.to_datetime(al_data['datetime'])
    al_data_sorted = al_data.sort_values('datetime')
    
    # Предварительно вычисляем временные границы для фильтрации
    min_time = al_data_sorted['datetime'].min() - pd.Timedelta(hours=max_time_diff)
    max_time = al_data_sorted['datetime'].max() + pd.Timedelta(hours=max_time_diff)
    
    track_matches = {}
    
    def process_track(track_data):
        # Быстрая проверка по времени перед полной обработкой
        track_times = track_data['datetime']
        if not ((track_times >= min_time) & (track_times <= max_time)).any():
            return None
            
        track_data = track_data.copy()
        track_data['datetime'] = pd.to_datetime(track_data['datetime'])
        track_data_sorted = track_data.sort_values('datetime')

        # Используем merge_asof с предварительно отсортированными данными
        merged = pd.merge_asof(
            al_data_sorted,
            track_data_sorted,
            on='datetime',
            direction='nearest',
            tolerance=pd.Timedelta(hours=max_time_diff),
            suffixes=('_al', '_track')
        ).dropna()

        if len(merged) == 0:
            return None

        # Векторизованное вычисление расстояния
        x_diff = merged['x_al'] - merged['x_track']
        y_diff = merged['y_al'] - merged['y_track']
        distances = np.sqrt(x_diff**2 + y_diff**2)
        
        valid_count = np.sum(distances <= max_distance_km)
        
        return valid_count if valid_count >= similar_steps else None

    # Обрабатываем треки с предварительной фильтрацией
    for idx, track in enumerate(NAAD_tracks):
        matches = process_track(track)
        if matches is not None:
            track_matches[idx] = matches
    
    # Сохраняем результаты
    unique_naad_indices = list(track_matches.keys())
    TC_dict[key]['NAAD_tracks'] = [NAAD_tracks[i] for i in unique_naad_indices]
    
    return TC_dict

# Основной цикл с оптимизациями
def process_all_tracks_fast(TC_dict, naad_tracks_preprocessed):
    """Обработка всех треков с оптимизацией"""
    
    # Создаем временной индекс для быстрого поиска
    time_index = {}
    for idx, track in enumerate(naad_tracks_preprocessed):
        for time_val in track['datetime']:
            if time_val not in time_index:
                time_index[time_val] = []
            time_index[time_val].append(idx)
    
    for key in tqdm(TC_dict.keys(), desc=f'Processing tracks'):
        CS_track_NOAA = TC_dict[key]['NOAA_track']
        NOAA_times = pd.to_datetime(CS_track_NOAA['datetime'])
        
        # Быстрый поиск подходящих треков через временной индекс
        matching_indices = set()
        for time_val in NOAA_times:
            if time_val in time_index:
                matching_indices.update(time_index[time_val])
        
        relevant_naad_tracks = [naad_tracks_preprocessed[i] for i in matching_indices]
        
        if relevant_naad_tracks:
            TC_dict = find_matching_tracks_fast(
                TC_dict, key, CS_track_NOAA, relevant_naad_tracks, 
                max_distance_km=5, max_time_diff=3, similar_steps=5
            )
    
    return TC_dict

def plot_closest_track_for_NOAA(TC_dict, key, NAAD_name, path_data, folder, data_type='LoRes', sigma=2, save=True):

    folder = f'{path_data}/{folder}'
    
    closest_tracks_NAAD = TC_dict[key][NAAD_name]
    if len(closest_tracks_NAAD) > 0:
    
        fig = plt.figure(figsize=(5, 5), dpi=150)
        ax = fig.add_subplot(111, projection=ccrs.Stereographic(central_latitude=45.0, central_longitude=-45))
        # ax.set_global()
        # gl = ax.gridlines(draw_labels=True,
        #              linewidth=1, color='grey', alpha=0.7, linestyle='--')
        # gl.xlabels_top = False
        # gl.ylabels_right = False
        ax.set_extent([-95, -13, 4, 79], ccrs.PlateCarree())

        # ax.coastlines(color='k', alpha=0.7, lw=1)
    
        # Добавление берегов с заливкой без границы
        land = cfeature.NaturalEarthFeature(
            category='physical',
            name='land',
            scale='110m',
            alpha=0.5,
            facecolor='lightgray',  # Заливка цветом 'lightgray'
            edgecolor='none'        # Без границы
        )
        
        # Добавление объекта land на ось
        ax.add_feature(land)

        lon_btm = ds.XLONG.isel(south_north=0)
        lat_btm = ds.XLAT.isel(south_north=0)
        lon_lft = ds.XLONG.isel(west_east=0)
        lat_lft = ds.XLAT.isel(west_east=0)
        lon_top = ds.XLONG.isel(south_north=-1)
        lat_top = ds.XLAT.isel(south_north=-1)
        lon_rgt = ds.XLONG.isel(west_east=-1)
        lat_rgt = ds.XLAT.isel(west_east=-1)
    
        ax.plot(lon_btm, lat_btm, color='tab:red', transform=ccrs.PlateCarree(), label=f"NAAD domain")
        ax.plot(lon_lft, lat_lft, color='tab:red', transform=ccrs.PlateCarree())
        ax.plot(lon_top, lat_top, color='tab:red', transform=ccrs.PlateCarree())
        ax.plot(lon_rgt, lat_rgt, color='tab:red', transform=ccrs.PlateCarree())

        # ax.contour(lon, lat, np.ones_like(lon), levels=[0.999, 1.001], 
        #    colors='red', linewidths=1, transform=ccrs.PlateCarree())
    
        CS_NOAA = TC_dict[key]['NOAA_track']
        ax.plot(CS_NOAA['lon'], CS_NOAA['lat'], 
                    c='k',
                    transform=ccrs.PlateCarree())
    
        
        # print(f'{key}: {len(closest_tracks_NAAD)} naad tracks')
        TC_name = TC_dict[key]['NOAA_track']['Name'].values[0].split()[0]
        TC_start = TC_dict[key]['NOAA_track']['datetime'].dt.strftime('%Y-%m-%dT%H').values[0]
        
        ax.set_title(f'{TC_name} at {TC_start}')

        for idx, CS_NAAD in enumerate(closest_tracks_NAAD):
            CS_start = CS_NAAD['datetime'].dt.strftime('%Y-%m-%dT%H').values[0]
            dist = TC_dict[key]['dist_btwn_tracks'][idx]
            ax.plot(CS_NAAD['lon'], CS_NAAD['lat'], 
                    # label=f'{CS_start} ({int(dist)} km)',
                    label=f'{CS_start} ({int(dist)} km)',  # Обновили подпись
                # c='red',
                transform=ccrs.PlateCarree())
            ax.legend(fontsize=7)   

        if save:
            # folder = f'{path_data}/TC_tracks/pics_diff_NAAD_NOAA_{data_type}/{folder_name}'

            # print('Plotting folder: ', folder)
            if not os.path.exists(f"{folder}"):
                os.makedirs(f"{folder}")
        
            fig.savefig(f'{folder}/{data_type}_{key}_{TC_name}_sigma_{sigma}.png', 
                        dpi=200, 
                        bbox_inches="tight", 
                        transparent=False)

def plot_closest_track_for_EC(TC_dict, key, NAAD_name, path_data, folder, data_type='LoRes', sigma=2, save=True):

    folder = f'{path_data}/{folder}'
    
    closest_tracks_NAAD = TC_dict[key][NAAD_name]
    if len(closest_tracks_NAAD) > 0:
    
        fig = plt.figure(figsize=(5, 5), dpi=150)
        ax = fig.add_subplot(111)

        ax.contourf(np.where(ds['HGT'] > 15, 1, np.nan), cmap='Greys', alpha=0.7)
        
       
        CS_NOAA = TC_dict[key]['NOAA_track']
        ax.plot(CS_NOAA['x'], CS_NOAA['y'], 
                    c='k')

        TC_name = key + 1

        
        TC_start = TC_dict[key]['NOAA_track']['datetime'].dt.strftime('%Y-%m-%dT%H').values[0]
        
        ax.set_title(f'{TC_name} at {TC_start}')

        ax.set_xlim(0,500)
        

        for idx, CS_NAAD in enumerate(closest_tracks_NAAD):
            CS_start = CS_NAAD['datetime'].dt.strftime('%Y-%m-%dT%H').values[0]
            dist = TC_dict[key]['dist_btwn_tracks'][idx]
            ax.plot(CS_NAAD['x'], CS_NAAD['y'], 
                    # label=f'{CS_start} ({int(dist)} km)',
                    label=f'{CS_start} ({int(dist)} km)',
                   )

        ax.scatter(CS_NOAA['x'].values[0], CS_NOAA['y'].values[0], 
                    c='g', s=7, zorder=10, label='manual start')
        ax.scatter(CS_NOAA['x'].values[-1], CS_NOAA['y'].values[-1], 
                    c='r', s=7, zorder=10, label='manual stop')

        ax.legend(fontsize=7)   
    

        if save:
            if not os.path.exists(f"{folder}"):
                os.makedirs(f"{folder}")
        
            fig.savefig(f'{folder}/{data_type}_{(TC_name):09d}_sigma_{sigma}.png', 
                        dpi=200, 
                        bbox_inches="tight", 
                        transparent=False)
            
def save_TC_merged(TC_dict, path_data, folder, NAAD_name='NOAA_track_NAAD', manual_data='NOAA'):
    folder = f'{path_data}/{folder}'
    
    if not os.path.exists(f"{folder}"):
        os.makedirs(f"{folder}")
    
    for idx, key in tqdm(enumerate(TC_dict.keys())):

        if len(TC_dict[key]['NAAD_tracks']) > 0:
        
            CS_track_NOAA = TC_dict[key]['NOAA_track_NAAD']

            if manual_data == 'NOAA':
                TC_name = TC_dict[key]['NOAA_track']['Name'].values[0].split()[0]
                CS_track_NOAA['Name'] = TC_name
                CS_track_NOAA['Id'] = TC_dict[key]['NOAA_track']['Id'].values[0]
            else:
                CS_track_NOAA['Id'] = key+1
    
            CS_start = CS_track_NOAA['datetime'].dt.strftime('%Y-%m-%dT%H').values[0]
            CS_track_NOAA.to_csv(f'{folder}/{(key+1):09d}_{CS_start}.csv')



def plot_false_alarms(false_alarm_tracks, folder):
    """
    Отрисовка False Alarms — NAAD-треков, не совпавших с NOAA.
    
    Параметры:
    - false_alarm_tracks: список DataFrame'ов (каждый — трек)
    - folder: папка для сохранения
    - ds: xarray.Dataset с координатами (для отрисовки домена NAAD), опционально
    - save: сохранять ли в файл
    - show_domain: показывать ли границу домена NAAD
    """
    if not false_alarm_tracks:
        print(f"No false alarms to plot.")
        return

    fig = plt.figure(figsize=(6, 6), dpi=150)
    ax = fig.add_subplot(111, projection=ccrs.Stereographic(central_latitude=45.0, central_longitude=-45))
    ax.set_extent([-95, -13, 4, 79], crs=ccrs.PlateCarree())

    # Добавляем заливку суши без границ
    land = cfeature.NaturalEarthFeature(
        category='physical',
        name='land',
        scale='110m',
        alpha=0.5,
        facecolor='lightgray',
        edgecolor='none'
    )
    ax.add_feature(land)

    lon_btm = ds.XLONG.isel(south_north=0)
    lat_btm = ds.XLAT.isel(south_north=0)
    lon_lft = ds.XLONG.isel(west_east=0)
    lat_lft = ds.XLAT.isel(west_east=0)
    lon_top = ds.XLONG.isel(south_north=-1)
    lat_top = ds.XLAT.isel(south_north=-1)
    lon_rgt = ds.XLONG.isel(west_east=-1)
    lat_rgt = ds.XLAT.isel(west_east=-1)
    ax.plot(lon_btm, lat_btm, color='tab:red', linewidth=1.2,
            transform=ccrs.PlateCarree(), label="NAAD domain")
    ax.plot(lon_lft, lat_lft, color='tab:red', linewidth=1.2, transform=ccrs.PlateCarree())
    ax.plot(lon_top, lat_top, color='tab:red', linewidth=1.2, transform=ccrs.PlateCarree())
    ax.plot(lon_rgt, lat_rgt, color='tab:red', linewidth=1.2, transform=ccrs.PlateCarree())

    # Отрисовка False Alarm треков
    for idx, track in enumerate(false_alarm_tracks):
        lons = track['lon'].values
        lats = track['lat'].values
        start_time = track['datetime'].iloc[0].strftime('%Y-%m-%d')

        ax.plot(lons, lats,
                transform=ccrs.PlateCarree(),
                color='red',
                linewidth=1.8,
                alpha=0.8,
                label=f"FA {start_time}" if idx == 0 else "")

    # Убираем дубли в легенде
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc='upper right')

    ax.set_title(f"False Alarms", pad=20)
    plt.tight_layout()

    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f"false_alarms.png")
    fig.savefig(filepath, dpi=200, bbox_inches="tight", transparent=False)
    plt.close(fig)
