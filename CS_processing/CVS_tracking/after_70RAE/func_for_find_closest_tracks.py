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

path_dir_raw = f'/storage/OPENDATA/NAAD/LoRes/Surface/msl'
ncfile = f'{path_dir_raw}/NAAD77km_msl_2010.nc'

ds = xr.open_dataset(f'{ncfile}') #['HGT'][0]


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

# Функция преобразования координат
def convert_coord(coord):
    value = float(coord[:-1])  # Числовая часть
    direction = coord[-1]      # Последний символ (N, S, E, W)
    if direction in ['S', 'W']:
        value = -value  # Отрицательное значение для юга и запада
    return value

def get_tracks_dist(track_real,TC):
#     TC_for_real = TC[np.isin(TC.datetime.values,track_real.date.values)]
#     track_real_for_TC = track_real[np.isin(track_real.date.values,TC_for_real.datetime.values)]

    TC_for_real = TC[np.isin(TC.datetime.values,track_real.datetime.values)]
    track_real_for_TC = track_real[np.isin(track_real.datetime.values,TC_for_real.datetime.values)]
    
    coord_pair = [(y,x) for x, y in zip(TC_for_real['lon'].values, TC_for_real['lat'].values)]
    coord_pair_real = [(y,x) for x, y in zip(track_real_for_TC['lon'].values, track_real_for_TC['lat'].values)]
    
    dist = []
    for i in range(len(coord_pair)):
        # print(coord_pair[i], coord_pair_real[i])
        dist.append(great_circle(coord_pair[i], coord_pair_real[i]).km)
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

# Функция для поиска ближайшего трека
def find_closest_track_many(TC_dict, key, CS_track_NOAA, NAAD_tracks, max_distance_km):

    
    print(key)
    
    CS_track_NAAD_list = []
    CS_names = []
    
    for t in range(len(CS_track_NOAA)):
        closest = None
        min_distance = float("inf")
        
        for idx, CS_track_NAAD in enumerate(NAAD_tracks):

            closest_time_track = CS_track_NAAD[CS_track_NAAD["datetime"] == CS_track_NOAA['datetime'][t]]

            if len(closest_time_track) == 0:
                continue

            # Вычисление расстояния
            noaa_coords = (CS_track_NOAA["lat"][t], CS_track_NOAA["lon"][t])
            naad_coords = (closest_time_track["lat"].values[0], closest_time_track["lon"].values[0])

            distance = geodesic(noaa_coords, naad_coords).kilometers

            if distance < min_distance and distance <= max_distance_km:
                closest = CS_track_NAAD
                min_distance = distance
                # print(noaa_coords, naad_coords, distance)
                CS_track_NAAD_list.append(CS_track_NAAD)
                CS_names.append(idx)

    CS_track_NAAD_list_fin = []
    for idx, CS_track_NAAD in zip(np.arange(len(NAAD_tracks)), CS_track_NAAD_list):
        # if idx not in CS_names:
        if CS_names.count(idx) >= 1:
            CS_track_NAAD_list_fin.append(CS_track_NAAD)
                    
    TC_dict[key]['NAAD_tracks'] = CS_track_NAAD_list_fin
    # TC_dict[key]['NAAD_tracks'] = CS_track_NAAD_list
    
    
    return TC_dict

def find_matching_tracks(TC_dict, key, CS_track_NOAA, NAAD_tracks, max_distance_km=300, max_time_diff=3, similar_steps=5):

    al_data = CS_track_NOAA

    track_files = NAAD_tracks

    matching_files = []

    for track_file in track_files:

        track_data = track_file

        # Проверка совпадений
        matches = 0
        consecutive_matches = 0
        max_consecutive = 0

        for _, al_row in al_data.iterrows():
            for _, track_row in track_data.iterrows():
                time_diff = abs(al_row['datetime'] - track_row['datetime'])
                if time_diff <= timedelta(hours=max_time_diff):
                    distance = geodesic([al_row['lat'], al_row['lon']], [track_row['lat'], track_row['lon']]).kilometers
                    
                    if distance <= max_distance_km:
                        matches += 1
                        consecutive_matches += 1
                        max_consecutive = max(max_consecutive, consecutive_matches)
                    else:
                        consecutive_matches = 0

        if max_consecutive >= similar_steps:
            matching_files.append(track_file)

    TC_dict[key]['NAAD_tracks'] = matching_files

    return TC_dict

# def find_matching_tracks_fast(TC_dict, key, CS_track_NOAA, NAAD_tracks, max_distance_km=500, max_time_diff=3, similar_steps=5):
#     al_data = CS_track_NOAA.copy()
#     al_data['datetime'] = pd.to_datetime(al_data['datetime'])
    
#     def process_track(track_data):
#         track_data = track_data.copy()
#         track_data['datetime'] = pd.to_datetime(track_data['datetime'])

#         merged = pd.merge_asof(
#             al_data.sort_values('datetime'),
#             track_data.sort_values('datetime'),
#             on='datetime',
#             direction='nearest',
#             tolerance=pd.Timedelta(hours=max_time_diff),
#             suffixes=('_al', '_track')
#         ).dropna()

#         if len(merged) == 0:
#             return False

#         merged['distance'] = merged.apply(
#             lambda row: great_circle(
#                 (row['lat_al'], row['lon_al']),
#                 (row['lat_track'], row['lon_track'])
#             ).km,
#             axis=1
#         )

#         valid = merged[merged['distance'] <= max_distance_km]
        
#         # Просто проверяем общее количество совпадений >= 3
#         return len(valid) >= similar_steps

#     with ThreadPoolExecutor() as executor:
#         results = list(executor.map(process_track, NAAD_tracks))
    
#     TC_dict[key]['NAAD_tracks'] = [track for track, is_match in zip(NAAD_tracks, results) if is_match]
#     return TC_dict

def find_matching_tracks_fast(TC_dict, key, CS_track_NOAA, NAAD_tracks, max_distance_km=500, max_time_diff=3, similar_steps=5):
    al_data = CS_track_NOAA.copy()
    al_data['datetime'] = pd.to_datetime(al_data['datetime'])
    
    # Словарь для хранения совпадений (ключ - индекс трека NAAD, значение - количество совпадений)
    track_matches = {}
    
    def process_track(track_data):
        track_data = track_data.copy()
        track_data['datetime'] = pd.to_datetime(track_data['datetime'])

        merged = pd.merge_asof(
            al_data.sort_values('datetime'),
            track_data.sort_values('datetime'),
            on='datetime',
            direction='nearest',
            tolerance=pd.Timedelta(hours=max_time_diff),
            suffixes=('_al', '_track')
        ).dropna()

        if len(merged) == 0:
            return None

        merged['distance'] = merged.apply(
            lambda row: great_circle(
                (row['lat_al'], row['lon_al']),
                (row['lat_track'], row['lon_track'])
            ).km,
            axis=1
        )

        valid = merged[merged['distance'] <= max_distance_km]
        
        # Возвращаем индекс трека и количество совпадений
        return len(valid)

    # Обрабатываем все треки и запоминаем количество совпадений
    for idx, track in enumerate(NAAD_tracks):
        matches = process_track(track)
        if matches and matches >= similar_steps:
            track_matches[idx] = matches
    
    # Убираем дубликаты (если один и тот же трек попал несколько раз)
    unique_naad_indices = set(track_matches.keys())
    TC_dict[key]['NAAD_tracks'] = [NAAD_tracks[i] for i in unique_naad_indices]
    
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

def save_TC_merged(TC_dict, path_data, folder, NAAD_name='NOAA_track_NAAD'):
    folder = f'{path_data}/{folder}'
    
    if not os.path.exists(f"{folder}"):
        os.makedirs(f"{folder}")
    
    for idx, key in tqdm(enumerate(TC_dict.keys())):

        if len(TC_dict[key]['NAAD_tracks']) > 0:
        
            CS_track_NOAA = TC_dict[key]['NOAA_track_NAAD']
            
            TC_name = TC_dict[key]['NOAA_track']['Name'].values[0].split()[0]
            CS_track_NOAA['Name'] = TC_name
            CS_track_NOAA['Id'] = TC_dict[key]['NOAA_track']['Id'].values[0]
            
    
            CS_start = CS_track_NOAA['datetime'].dt.strftime('%Y-%m-%dT%H').values[0]
            CS_track_NOAA.to_csv(f'{folder}/{key}_{TC_name}.csv')



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
