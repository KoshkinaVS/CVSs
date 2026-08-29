from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Initialize paths and parameters
path_init = f'/storage/thalassa/users/vkoshkina'
folder = 'scripts/CS_processing/CVS_tracking/after_70RAE/'
sys.path.insert(2, f'{path_init}/{folder}')

from step_of_tracking import *
from func_for_find_closest_tracks import *

from geopy.distance import great_circle
import pandas as pd

from datetime import datetime
import pandas as pd
from tqdm import tqdm

def load_TE_tracks(tracks_list, file_path, year, 
                   month_start=None, day_start=1, 
                   month_end=None, day_end=31, time_th=3):
    """
    Загружает треки из файла ERA5, фильтруя по сезону на основе ДАТЫ ПЕРВОЙ ТОЧКИ трека.
    
    Args:
        tracks_list (list): Список для добавления треков
        file_path (str): Путь к файлу с треками
        time_th (int): Минимальное количество точек для сохранения трека
        month_start (int): Начальный месяц периода (включительно)
        day_start (int): Начальный день периода (включительно)
        month_end (int): Конечный месяц периода (включительно)
        day_end (int): Конечный день периода (включительно)
        
    Returns:
        list: Обновлённый список DataFrame с треками, первый шаг которых в заданном периоде
    """
    current_track = []
    track_count = 0

    # Проверка корректности параметров
    if (month_start is None) or (month_end is None):
        print("Warning: month_start and month_end must be specified. Using all data.")
    # Можно добавить проверку валидности дней/месяцев, но опустим для краткости

    # Подсчёт строк для прогресс-бара
    with open(file_path, 'r') as f:
        total_lines = sum(1 for _ in f)

    with open(file_path, 'r') as f:
        pbar = tqdm(total=total_lines, desc="Processing tracks", unit="lines")

        for line in f:
            pbar.update(1)
            line = line.strip()
            if not line:
                continue

            if line.startswith('start'):
                # Обработка предыдущего трека
                if current_track:
                    df = pd.DataFrame(current_track, 
                                    columns=['i', 'j', 'lon', 'lat', 'wind', 'r2d',
                                             'year', 'month', 'day', 'hour'])
                    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
                    
                    # Проверка: достаточно ли точек
                    if len(df) <= time_th:
                        current_track = []
                        continue

                    # Берём ПЕРВУЮ точку трека
                    first_row = df.iloc[0]
                    first_month = int(first_row['month'])
                    first_day = int(first_row['day'])

                    # Проверяем, попадает ли первая точка в диапазон
                    if month_start is not None and month_end is not None:
                        # Преобразуем дату в "день года", но без привязки к году
                        try:
                            date_first = datetime(year=year, month=first_month, day=first_day)
                            date_start = datetime(year=year, month=month_start, day=day_start)
                            date_end = datetime(year=year, month=month_end, day=day_end)
                        except ValueError:
                            current_track = []
                            continue  # Неверная дата (например, 30 февраля)

                        # Обработка перехода через год (например, дек-янв)
                        if date_start <= date_end:
                            in_range = date_start <= date_first <= date_end
                        else:
                            # Период пересекает Новый год (например, ноябрь → февраль)
                            in_range = (date_first >= date_start) or (date_first <= date_end)

                        if not in_range:
                            current_track = []
                            continue

                    # Если всё ок — добавляем трек
                    tracks_list.append(df)
                    track_count += 1

                    current_track = []

            else:
                parts = line.split()
                if len(parts) == 10:
                    try:
                        current_track.append([float(x) for x in parts])
                    except (ValueError, TypeError):
                        continue  # Пропускаем некорректные строки

        # Обработка последнего трека
        if current_track:
            df = pd.DataFrame(current_track, 
                            columns=['i', 'j', 'lon', 'lat', 'wind', 'r2d',
                                     'year', 'month', 'day', 'hour'])
            df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])

            if len(df) <= time_th:
                pass
            else:
                first_row = df.iloc[0]
                first_month = int(first_row['month'])
                first_day = int(first_row['day'])

                if month_start is not None and month_end is not None:
                    try:
                        date_first = datetime(year=year, month=first_month, day=first_day)
                        date_start = datetime(year=year, month=month_start, day=day_start)
                        date_end = datetime(year=year, month=month_end, day=day_end)
                    except ValueError:
                        pass
                    else:
                        if date_start <= date_end:
                            in_range = date_start <= date_first <= date_end
                        else:
                            in_range = (date_first >= date_start) or (date_first <= date_end)

                        if not in_range:
                            pass
                        else:
                            tracks_list.append(df)
                            track_count += 1
                else:
                    tracks_list.append(df)
                    track_count += 1

        pbar.close()

    print(f"\nLoaded {track_count} tracks with more than {time_th} points "
          f"and first time step in period: {month_start}/{day_start} – {month_end}/{day_end}")
    return tracks_list


def find_matching_tracks_fast(TC_dict, key, CS_track_NOAA, NAAD_tracks, max_distance_km=500, max_time_diff=3, similar_steps=5):
    al_data = CS_track_NOAA.copy()
    al_data['datetime'] = pd.to_datetime(al_data['datetime'])
    
    # Словарь для хранения совпадений (ключ - индекс трека NAAD, значение - (количество совпадений, медианное расстояние))
    track_matches = {}
    
    def process_track(track_data):
        track_data = track_data.copy()
        track_data['datetime'] = pd.to_datetime(track_data['datetime'])

        # Объединение по ближайшему времени с допуском max_time_diff
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

        # Вычисление расстояний между точками
        merged['distance'] = merged.apply(
            lambda row: great_circle(
                (row['lat_al'], row['lon_al']),
                (row['lat_track'], row['lon_track'])
            ).km,
            axis=1
        )

        # Фильтрация точек, где расстояние <= max_distance_km
        valid = merged[merged['distance'] <= max_distance_km]

        if len(valid) == len(merged):
        
        # if len(valid) == similar_steps:
            return len(valid), np.nanmedian(merged['distance'])
        return None

    # Обработка всех треков NAAD
    for idx, track in enumerate(NAAD_tracks):
        result = process_track(track)
        if result:
            matches_count, median_dist = result
            # # Фильтрация по медианному расстоянию
            # if median_dist <= max_distance_km:
            #     track_matches[idx] = (matches_count, median_dist)

            track_matches[idx] = (matches_count, median_dist)
    
    # Отбираем только треки, удовлетворяющие условиям
    TC_dict[key]['NAAD_tracks'] = [
        NAAD_tracks[i] 
        for i in track_matches 
        # if track_matches[i][1] <= max_distance_km  # Проверка медианного расстояния
    ]
    
    return TC_dict

def calculate_detection_metrics(TC_dict, all_naad_tracks):
    hits = 0
    misses = 0
    false_alarms = 0
    false_alarm_tracks = []
    
    # 1. Подсчет hits и misses
    for noaa_id, track_data in TC_dict.items():
        if track_data['NAAD_tracks']:
            hits += 1
        else:
            misses += 1

    # 2. Определение false alarms
    # Собираем все NAAD треки, которые были сопоставлены
    matched_naad_tracks = []
    for track_data in TC_dict.values():
        matched_naad_tracks.extend(track_data['NAAD_tracks'])
    
    def is_track_in_list(track, track_list):
        """Custom function to check if a track exists in a list of tracks"""
        for t in track_list:
            if (track[['lat', 'lon', 'datetime']].equals(t[['lat', 'lon', 'datetime']])):  # adjust columns as needed
                return True
        return False
    
    false_alarm_tracks = [t for t in all_naad_tracks 
                         if not is_track_in_list(t, matched_naad_tracks)]
    false_alarms = len(false_alarm_tracks)
    
    return hits, misses, false_alarms, false_alarm_tracks

def plot_closest_track_for_NOAA(TC_dict, key, NAAD_name, path_data, folder, data_type='LoRes', sigma=2, postfix='', save=True):

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
                    label=f'{CS_start} ({int(dist)} km)',
                # c='red',
                transform=ccrs.PlateCarree())
            ax.legend(fontsize=7)   

        if save:
            # folder = f'{path_data}/TC_tracks/pics_diff_NAAD_NOAA_{data_type}/{folder_name}'
            
            if not os.path.exists(f"{folder}"):
                os.makedirs(f"{folder}")
        
            fig.savefig(f'{folder}/{key}_{TC_name}_sigma_{sigma}{postfix}.png', 
                        dpi=200, 
                        bbox_inches="tight", 
                        transparent=False)
    
def process_sigma(years, path_data, path_data_NOAA, data_type, sigma):
    
    folder_name = f'{output_dir}/NOAA_matching_test'
    os.makedirs(folder_name, exist_ok=True)
    
    for year in years:
        CS_tracks_list_NAAD = []
        filename = f"{path_data}/{data_type}_TC_tracks_{year}.txt"

        CS_tracks_list_NAAD = load_TE_tracks(CS_tracks_list_NAAD, filename, year, month_start=10, day_start=1, month_end=12, day_end=31)
        
        # Load NOAA tracks for current year
        CS_tracks_list_NOAA = []
        CS_tracks_list_NOAA = load_season_tracks_NOAA(CS_tracks_list_NOAA, path_data_NOAA, year, time_th=time_th)

        
        # Initialize dictionary for current year
        TC_dict = {
            CS_track['Id'].values[0]: {
                'NOAA_track': CS_track,
                'NAAD_tracks': [],
                'dist_btwn_tracks': []
            }
            for CS_track in CS_tracks_list_NOAA
        }

        keys = TC_dict.keys()
        keys = ['AL282005']
        # Find matching tracks
        for key in keys:
            CS_track_NOAA = TC_dict[key]['NOAA_track']
            NOAA_times = CS_track_NOAA['datetime']
            NAAD_tracks = [CS for CS in CS_tracks_list_NAAD if np.isin(CS['datetime'], NOAA_times).any()]
            TC_dict = find_matching_tracks_fast(TC_dict, key, CS_track_NOAA, NAAD_tracks, max_distance_km, max_time_diff, similar_steps)


        for key in tqdm(keys, desc=f'Calculating distances {year}'):
            dist_list = []
            CS_track_NOAA = TC_dict[key]['NOAA_track']
            for idx in range(len(TC_dict[key]['NAAD_tracks'])):
                track_NAAD = TC_dict[key]['NAAD_tracks'][idx]
                if track_NAAD is not None:
                    CS_track_NAAD = track_NAAD.rename(columns=lambda x: f"{x}_NAAD_{idx}" if x not in ['datetime'] else x)   
                    CS_track_NOAA = pd.merge(CS_track_NOAA, CS_track_NAAD, on="datetime", how="outer")
                    CS_track_NOAA = CS_track_NOAA.sort_values("datetime").reset_index(drop=True)
                    dist = get_tracks_dist(TC_dict[key]['NOAA_track'], track_NAAD)
                    dist_list.append(np.nanmean(dist))
            
            TC_dict[key]['NOAA_track_NAAD'] = CS_track_NOAA
            TC_dict[key]['dist_btwn_tracks'] = dist_list
        
        for idx, key in tqdm(enumerate(keys), 
                    desc=f'Plotting tracks for {year}',
                    total=len(TC_dict)):
            plot_closest_track_for_NOAA(TC_dict, key, 'NAAD_tracks', path_data, folder_name, data_type, sigma, postfix=f'_all_ts_with_{max_distance_km}')


dist_m, our_level, level, x_unit, y_unit, u_unit, v_unit, time_unit, level_unit, crit, th, min_samples, eps, CS_points_th = get_init_params(data_type, json_file, print_info=True)

time_th = 3
similar_steps = 3
max_time_diff = 1
# max_distance_km = 3*dist_m/1000

similar_steps = 1
max_distance_km = 250
max_distance_km = 500



years = np.arange(1979, 2019)
years = np.arange(2005, 2006)


months = np.arange(1, 13, 1)

sigmas = [
    4,
    2,
    0, 
]

path_data = f'{path_init}/data'
folder_NOAA = 'TC_tracks/splitted'



output_dir = f'/storage/thalassa/users/vkoshkina/data/TempestExtremes/{data_type}/NOAA_matching_results'

for sigma in sigmas:
    path_data_tracks = f'/storage/thalassa/users/vkoshkina/data/TempestExtremes/{data_type}/R2D_{data_type}_level_500_sigma_{sigma}/Tracks'
    
    print(f"\nProcessing sigma={sigma}...")
    process_sigma(years, path_data_tracks, f'{path_data}/{folder_NOAA}', data_type, sigma)
