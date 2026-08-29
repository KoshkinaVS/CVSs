from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import xarray as xr
import matplotlib.pyplot as plt
from geopy.distance import great_circle
from scipy.ndimage import label, generate_binary_structure
import warnings
warnings.filterwarnings("ignore")

# Initialize paths and parameters
path_init = '/storage/thalassa/users/vkoshkina'
data_type = 'LoRes'  # или 'ERA5' в зависимости от ваших данных

# Импорт пользовательских функций
folder = 'scripts/CS_processing/CVS_tracking/after_70RAE/'
sys.path.insert(2, f'{path_init}/{folder}')
from step_of_tracking import *
from func_for_find_closest_tracks import *

# Загрузка фоновых данных (если доступно)
try:
    ncfile = f'{path_init}/data/SMP/DBSCAN_02-04-10_with_wspd_smoothing/2019/sigma_2_DBSCAN_SMP_level_10_2019-02-05.nc'
    ds = xr.open_dataset(ncfile)
except:
    ds = None
    print("Background data not available, plotting without background")

def load_TE_tracks(tracks_list, file_path, time_th=3, start_date_bound=None, end_date_bound=None):
    """
    Загружает треки из файла TE с фильтрацией по времени
    """
    current_track = []
    track_count = 0
    filtered_track_count = 0
    
    with open(file_path, 'r') as f:
        total_lines = sum(1 for _ in f)
    
    with open(file_path, 'r') as f:
        pbar = tqdm(total=total_lines, desc="Processing TE tracks", unit="lines")
        
        for line in f:
            pbar.update(1)
            
            if line.startswith('start'):
                if current_track:
                    df = pd.DataFrame(current_track, 
                                    columns=['x', 'y', 'lon', 'lat', 'wind', 'r2d', 
                                             'year', 'month', 'day', 'hour'])
                    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
                    
                    if start_date_bound is not None and end_date_bound is not None:
                        track_start = df['datetime'].min()
                        track_end = df['datetime'].max()
                        if not (track_end < start_date_bound or track_start > end_date_bound):
                            if len(df) > time_th:
                                tracks_list.append(df)
                                filtered_track_count += 1
                    else:
                        if len(df) > time_th:
                            tracks_list.append(df)
                            track_count += 1
                    
                    current_track = []
            else:
                parts = line.strip().split()
                if len(parts) == 10:
                    current_track.append([float(x) for x in parts])
        
        if current_track:
            df = pd.DataFrame(current_track, 
                            columns=['x', 'y', 'lon', 'lat', 'wind', 'r2d', 
                                     'year', 'month', 'day', 'hour'])
            df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
            
            if start_date_bound is not None and end_date_bound is not None:
                track_start = df['datetime'].min()
                track_end = df['datetime'].max()
                if not (track_end < start_date_bound or track_start > end_date_bound):
                    if len(df) > time_th:
                        tracks_list.append(df)
                        filtered_track_count += 1
            else:
                if len(df) > time_th:
                    tracks_list.append(df)
                    track_count += 1
        
        pbar.close()
    
    if start_date_bound is not None and end_date_bound is not None:
        print(f"\nLoaded {filtered_track_count} tracks with more than {time_th} points (filtered by time)")
    else:
        print(f"\nLoaded {track_count} tracks with more than {time_th} points")
    
    return tracks_list

def load_eddyclicker_tracks(path_data_EC, year, time_th=3):
    """
    Загружает все треки EddyClicker для указанного года
    """
    CS_tracks_list = []
    name_pattern = f'*.csv'
    file_list = list(sorted(Path(f"{path_data_EC}/").glob(name_pattern)))

    for ifile in file_list:
        df = pd.read_csv(ifile, parse_dates=['time'])
        
        df = df.rename(columns={
            'time': 'datetime',
            'pxc_ind': 'x',
            'pyc_ind': 'y'
        })
        
        if len(df) > time_th:
            CS_tracks_list.append(df)
    
    return CS_tracks_list

def load_te_tracks_for_ec(EC_track, te_tracks_file, time_buffer_days=7, time_th=3):
    """
    Загружает TE треки для конкретного EC трека с фильтрацией по времени
    """
    EC_start = EC_track['datetime'].min()
    EC_end = EC_track['datetime'].max()
    start_date_bound = EC_start - pd.Timedelta(days=time_buffer_days)
    end_date_bound = EC_end + pd.Timedelta(days=time_buffer_days)
    
    te_tracks = []
    te_tracks = load_TE_tracks(te_tracks, te_tracks_file, time_th, start_date_bound, end_date_bound)
    
    print(f'{len(te_tracks)} TE tracks loaded for time range {start_date_bound} - {end_date_bound}')
    
    return te_tracks

def find_matching_tracks_ec(EC_track, TE_tracks, max_distance_km=50, max_time_diff=3, similar_steps=5):
    """
    Находит совпадающие TE треки для конкретного EC трека
    """
    ec_data = EC_track.copy()
    ec_data['datetime'] = pd.to_datetime(ec_data['datetime'])
    ec_data_sorted = ec_data.sort_values('datetime')
    
    matching_tracks = []
    
    for track in TE_tracks:
        track_data = track.copy()
        track_data['datetime'] = pd.to_datetime(track_data['datetime'])
        track_data_sorted = track_data.sort_values('datetime')

        merged = pd.merge_asof(
            ec_data_sorted,
            track_data_sorted,
            on='datetime',
            direction='nearest',
            tolerance=pd.Timedelta(hours=max_time_diff),
            suffixes=('_ec', '_te')
        ).dropna()

        if len(merged) == 0:
            continue

        if 'x_ec' in merged.columns and 'y_ec' in merged.columns:
            merged['distance'] = np.sqrt(
                (merged['x_ec'] - merged['x_te'])**2 + 
                (merged['y_ec'] - merged['y_te'])**2
            ) * 6
        else:
            merged['distance'] = merged.apply(
                lambda row: great_circle(
                    (row['lat_ec'], row['lon_ec']),
                    (row['lat_te'], row['lon_te'])
                ).km,
                axis=1
            )

        valid = merged[merged['distance'] <= max_distance_km]
        
        if len(valid) >= similar_steps:
            matching_tracks.append((track, np.nanmedian(valid['distance'])))
    
    return matching_tracks

def get_tracks_dist(track_real, TC, x_name='x', y_name='y'):
    """
    Вычисляет расстояния между двумя треками
    """
    common_dates = np.intersect1d(track_real['datetime'].values, TC['datetime'].values)
    
    if len(common_dates) == 0:
        return []
    
    TC_for_real = TC[TC['datetime'].isin(common_dates)]
    track_real_for_TC = track_real[track_real['datetime'].isin(common_dates)]
    
    dist = []
    for i in range(len(TC_for_real)):
        x1, y1 = TC_for_real.iloc[i][x_name], TC_for_real.iloc[i][y_name]
        x2, y2 = track_real_for_TC.iloc[i][x_name], track_real_for_TC.iloc[i][y_name]
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        dist.append(distance)
    
    return dist

def save_merged_track(EC_track, matching_tracks, path_save, track_id):
    """
    Сохраняет объединенный трек с данными EC и совпадающих треков
    """
    if not matching_tracks:
        return
    
    merged_track = EC_track.copy()
    
    for idx, (te_track, distance) in enumerate(matching_tracks):
        te_track_renamed = te_track.rename(columns=lambda x: f"{x}_TE_{idx}" if x not in ['datetime'] else x)
        merged_track = pd.merge(merged_track, te_track_renamed, on="datetime", how="outer")
    
    merged_track = merged_track.sort_values("datetime").reset_index(drop=True)
    merged_track['Id'] = track_id + 1
    
    CS_start = merged_track['datetime'].dt.strftime('%Y-%m-%dT%H').values[0]
    merged_track.to_csv(f'{path_save}/{(track_id+1):09d}_{CS_start}.csv', index=False)

def plot_EC_track_with_matches(EC_track, matching_tracks, path_save, track_id, data_type='LoRes', sigma=2):
    """
    Рисует и сохраняет график EC трека с совпадающими треками
    """
    if not matching_tracks:
        return
    
    fig = plt.figure(figsize=(5, 5), dpi=150)
    ax = fig.add_subplot(111)

    # Контур фона
    if ds is not None:
        try:
            ax.contourf(np.where(ds['HGT'] > 15, 1, np.nan), cmap='Greys', alpha=0.7)
        except:
            pass
    
    # Длительность EC трека
    ec_duration = int((EC_track['datetime'].max() - EC_track['datetime'].min()).total_seconds() / 3600)
    
    # EC трек
    ax.plot(EC_track['x'], EC_track['y'], c='k', linewidth=2, 
            label=f'manual ({ec_duration}h)')
    
    # Совпадающие TE треки
    for idx, (te_track, distance) in enumerate(matching_tracks):
        te_duration = int((te_track['datetime'].max() - te_track['datetime'].min()).total_seconds() / 3600)
        
        ax.plot(te_track['x'], te_track['y'], 
                label=f'TE {idx+1} ({int(distance)}km, {te_duration}h)')
    
    # Начальная и конечная точки
    ax.scatter(EC_track['x'].values[0], EC_track['y'].values[0], 
               c='g', s=20, zorder=10, label='start')
    ax.scatter(EC_track['x'].values[-1], EC_track['y'].values[-1], 
               c='r', s=20, zorder=10, label='end')
    
    # Настройки графика
    TC_start = EC_track['datetime'].dt.strftime('%Y-%m-%dT%H').values[0]
    ax.set_title(f'Track {track_id+1} at {TC_start}')
    # ax.set_xlim(0, 500)
    # ax.set_ylim(0, 500)

    ax.set_xlim(0, 400)
    ax.set_ylim(100, 500)

    ax.legend(fontsize=7)
    
    # Сохраняем график
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    
    fig.savefig(f'{path_save}/{data_type}_{(track_id+1):09d}_sigma_{sigma}.png', 
                dpi=200, 
                bbox_inches="tight", 
                transparent=False)
    
    plt.close(fig)

def calculate_ec_detection_metrics(EC_tracks_all, te_tracks_file, save_path, sigma, time_buffer_days=7, 
                                  save_plots=False, save_merged=False):
    """
    Вычисляет метрики обнаружения для EddyClicker треков с опциональным сохранением графиков и данных
    """
    hits = 0
    misses = 0
    all_matching_info = []
    
    # Создаем папки для сохранения
    if save_plots:
        plot_folder = f'{save_path}/pics/EC_TE_comparison_timefilter_1h/sigma_{sigma}'
        os.makedirs(plot_folder, exist_ok=True)
    
    if save_merged:
        merged_folder = f'{save_path}/EC_TE_merged_tracks_timefilter_1h/sigma_{sigma}'
        os.makedirs(merged_folder, exist_ok=True)
    
    for i, EC_track in enumerate(tqdm(EC_tracks_all, desc="Processing EC tracks")):
        TE_tracks = load_te_tracks_for_ec(EC_track, te_tracks_file, time_buffer_days)
        matching_tracks = find_matching_tracks_ec(EC_track, TE_tracks)
        
        if matching_tracks:
            hits += 1
            
            # Сохраняем графики
            if save_plots:
                plot_EC_track_with_matches(EC_track, matching_tracks, plot_folder, i, data_type, sigma)
            
            # Сохраняем объединенные треки
            if save_merged:
                save_merged_track(EC_track, matching_tracks, merged_folder, i)
            
            # Вычисляем расстояния для детальной статистики
            distances = []
            for te_track, _ in matching_tracks:
                dist = get_tracks_dist(EC_track, te_track)
                if dist:
                    distances.extend(dist)
            
            match_info = {
                'ec_track_id': i,
                'matches_count': len(matching_tracks),
                'avg_distance': np.mean(distances) if distances else None,
                'min_distance': min(distances) if distances else None,
                'max_distance': max(distances) if distances else None
            }
            all_matching_info.append(match_info)
        else:
            misses += 1
    
    # Расчет метрик
    pod = hits / (hits + misses) if (hits + misses) > 0 else 0
    
    metrics = {
        'sigma': sigma,
        'POD': pod,
        'Hits': hits,
        'Misses': misses,
        'Total_EC_Tracks': len(EC_tracks_all),
        'Matching_Info': all_matching_info
    }
    
    return metrics

def main():
    # Параметры обработки
    year = 2019
    sigmas = [4, 2, 0]
    time_th = 3
    time_buffer_days = 7
    
    # Параметры сохранения
    save_plots = True
    save_merged = True
    
    all_metrics = []
    
    for sigma in sigmas:
        print(f"\nProcessing sigma={sigma}...")
        
        # Загрузка треков EddyClicker
        path_data_EC = f'{path_init}/data/{data_type}/EddyClicker_tracks'
        EC_tracks_all = load_eddyclicker_tracks(path_data_EC, year, time_th)

        save_path = f'{path_init}/data/TempestExtremes/{data_type}'
        sigma_path = f'{save_path}/R2D_{data_type}_level_10_sigma_{sigma}'
        
        # Путь к файлу TE треков
        te_tracks_file = f'{sigma_path}/Tracks_timefilter_1h/{data_type}_TC_tracks_{year}.txt'
        
        # Вычисление метрик с сохранением графиков и данных
        metrics = calculate_ec_detection_metrics(EC_tracks_all, te_tracks_file, save_path, sigma, 
                                               time_buffer_days, save_plots, save_merged)
        all_metrics.append(metrics)
        
        print(f"Sigma {sigma}: POD = {metrics['POD']:.3f}, Hits = {metrics['Hits']}, Misses = {metrics['Misses']}")
    
    # Сохранение результатов
    metrics_df = pd.DataFrame([{k: v for k, v in m.items() if k != 'Matching_Info'} for m in all_metrics])
    output_file = f'{save_path}/EC_detection_metrics_timefilter_1h.csv'
    metrics_df.to_csv(output_file, index=False)
    
    # Сохранение детальной информации о совпадениях
    matching_info = []
    for metrics in all_metrics:
        for info in metrics['Matching_Info']:
            info['sigma'] = metrics['sigma']
            matching_info.append(info)
    
    matching_df = pd.DataFrame(matching_info)
    matching_file = f'{save_path}/EC_matching_details_timefilter_1h.csv'
    matching_df.to_csv(matching_file, index=False)
    
    print(f"\nMetrics saved to {output_file}")
    print(f"Matching details saved to {matching_file}")
    if save_plots:
        print(f"Plots saved to {save_path}/pics/EC_TE_comparison_timefilter_1h/")
    if save_merged:
        print(f"Merged tracks saved to {save_path}/EC_TE_merged_tracks_timefilter_1h/")

if __name__ == "__main__":
    main()