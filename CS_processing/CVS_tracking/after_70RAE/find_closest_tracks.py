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

from step_of_tracking import *

tracking_type = 'tracking_local_2_phase'
CVS_speed = 'adv_speed'

# pref_tracking = 'rad_and_bound'
pref_tracking = 'all_points_bound'

local_extr_name = 'local_extr_crit'

dist_m, our_level, x_unit, y_unit, u_unit, v_unit, time_unit, level_unit, crit, th, min_samples, eps, CS_points_th = get_init_params(data_type, json_file, print_info=True)


time_th = 3

similar_steps = 3
max_time_diff = 3  # Максимальная разница по времени
max_distance_km = 3*dist_m/1000  # Максимальная допустимая дистанция (км)

def load_season_tracks(CS_tracks_list, year, months, path_data, time_th=3): 
    null_files = []
    
    for month in tqdm(months):
        if os.path.exists(path_data):
            ls = list(sorted(Path(f"{path_data}/{year}-{month:02d}").glob(f'*_track_*.csv')))
            # print(f'tracks: {ls}')
            
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
            # print(ifile)
            df = pd.read_csv(ifile, encoding='ascii', parse_dates=['datetime'], 
                             dtype={'status': str, 'Name': str},
                             converters={
                                            'lat': convert_coord,  # Преобразуем широту
                                            'lon': convert_coord   # Преобразуем долготу
                                        })
            if len(df) > time_th:
                df = df.drop(df.columns[0], axis=1)
#             df = df.drop(df.columns[0], axis=1)
                CS_tracks_list.append(df)
            else:
                null_files.append(ifile)
    
    print(f'too short NOAA tracks in {year}: {len(null_files)}')
    
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
        dist.append(geodesic(coord_pair[i], coord_pair_real[i]).km)
    return dist

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
    

def plot_closest_track_for_NOAA(TC_dict, key, NAAD_name, save=True):

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
                    label=f'{CS_start} ({int(dist)} km)',
                # c='red',
                transform=ccrs.PlateCarree())
            ax.legend(fontsize=7)   

        if save:
            folder = f'{path_data}/TC_tracks/pics_diff_NAAD_NOAA/{tracking_type}_{CVS_speed}_{pref_tracking}'
            if not os.path.exists(f"{folder}"):
                os.makedirs(f"{folder}")
        
            fig.savefig(f'{folder}/{key}_{TC_name}.png', 
                        dpi=200, 
                        bbox_inches="tight", 
                        transparent=False)

def save_TC_merged(TC_dict, NAAD_name='NOAA_track_NAAD'):
    folder = f'{path_data}/TC_tracks/NAAD_NOAA_tracks_merged_{tracking_type}_{CVS_speed}_{pref_tracking}'
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

years = np.arange(1979,1986)
months = np.arange(1,13,1)
# months = np.arange(1,2,1)


path_dir_data = f'{path_init}/data/{data_type}'

path_data = f'{path_init}/data'

folder_NOAA = 'TC_tracks/splitted'

DBSCAN_name = f'DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_level_{level}_smoothing'
# DBSCAN_name = f'DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_with_uv_time_smoothing'


folder_NAAD = f'{path_dir_data}/{data_type}/{data_type}_tracks/{pref_tracking}/{tracking_type}_{CVS_speed}_{pref_tracking}/{DBSCAN_name}'

path_tracks_dir = f'{folder_NAAD}/tracks_{circ}'
# path_tracks_dir = f'{folder}/tracks_{circ}_params/'


CS_tracks_list_LoRes = []
for year in years:
    CS_tracks_list_LoRes = load_season_tracks(CS_tracks_list_LoRes, year, months, path_tracks_dir, time_th=time_th)

CS_tracks_list_NOAA = []
for year in years:
    CS_tracks_list_NOAA = load_season_tracks_NOAA(CS_tracks_list_NOAA, f'{path_data}/{folder_NOAA}', year, time_th=time_th)


# Создаем словарь
TC_dict = {
    CS_track['Id'].values[0]: {
        'NOAA_track': CS_track,  # Полный путь к файлу NOAA
        'NAAD_tracks': [],    # Пустой список для файлов NAAD, который будет заполняться позже
        'dist_btwn_tracks': []    # Пустой список для расстояний, который будет заполняться позже
        
    }
    for CS_track in CS_tracks_list_NOAA
}

CS_tracks_list_NAAD = CS_tracks_list_LoRes

for key in tqdm(TC_dict.keys()):
    
    CS_track_NOAA = TC_dict[key]['NOAA_track']
    NOAA_times = CS_track_NOAA['datetime']
    NAAD_tracks = [CS for CS in CS_tracks_list_NAAD if np.isin(CS['datetime'], NOAA_times).any()]
    # TC_dict = find_closest_track_many(TC_dict, key, CS_track_NOAA, NAAD_tracks, max_distance_km)
    TC_dict = find_matching_tracks(TC_dict, key, CS_track_NOAA, NAAD_tracks, max_distance_km, max_time_diff, similar_steps)


for key in tqdm(TC_dict.keys()):
    dist_list = []
    CS_track_NOAA = TC_dict[key]['NOAA_track']
    for idx in range(len(TC_dict[key]['NAAD_tracks'])):
        track_NAAD = TC_dict[key]['NAAD_tracks'][idx]
        if track_NAAD is not None:
            CS_track_NAAD = track_NAAD.rename(columns=lambda x: f"{x}_NAAD_{idx}" if x not in ['datetime'] else x)   
            
            # Объединяем таблицы на основе поля datetime (outer join для сохранения всех значений)
            CS_track_NOAA = pd.merge(CS_track_NOAA, CS_track_NAAD, on="datetime", how="outer")
            # Сортируем по дате для удобства
            CS_track_NOAA = CS_track_NOAA.sort_values("datetime").reset_index(drop=True)
    
            dist = get_tracks_dist(TC_dict[key]['NOAA_track'], track_NAAD)
            dist_list.append(np.nanmean(dist))
            
    TC_dict[key]['NOAA_track_NAAD'] = CS_track_NOAA
    TC_dict[key]['dist_btwn_tracks'] = dist_list


save_TC_merged(TC_dict, NAAD_name='NOAA_track_NAAD')

for idx, key in tqdm(enumerate(TC_dict.keys())):
    plot_closest_track_for_NOAA(TC_dict, key, NAAD_name='NAAD_tracks')