from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Initialize paths
path_init = f'/storage/thalassa/users/vkoshkina'
folder = 'scripts/CS_processing/CVS_tracking/after_70RAE/'
sys.path.insert(2, f'{path_init}/{folder}')

from step_of_tracking import *

def load_EC_tracks(path_data_EC, year, time_th=3):
    """
    Загружает все EddyClicker треки
    """
    CS_tracks_list = []
    name_pattern = f'*.csv'
    ls = list(sorted(Path(f"{path_data_EC}").glob(name_pattern)))

    if len(ls) != 0:
        for ifile in ls:
            df = pd.read_csv(ifile, parse_dates=['time'])
            
            # Переименовываем колонки
            df = df.rename(columns={
                'time': 'datetime',
                'pxc_ind': 'x',
                'pyc_ind': 'y'
            })
            
            if len(df) > time_th:
                CS_tracks_list.append(df)
    
    return CS_tracks_list

def load_merged_tracks(merged_folder):
    """
    Загружает все объединенные треки из merged_folder
    """
    merged_tracks = []
    ls = list(sorted(Path(merged_folder).glob('*.csv')))
    
    for ifile in ls:
        df = pd.read_csv(ifile, parse_dates=['datetime'])
        merged_tracks.append(df)
    
    return merged_tracks

def is_track_matched(merged_track):
    """
    Проверяет, есть ли совпадения для трека (есть ли колонки NAAD)
    """
    naad_columns = [col for col in merged_track.columns if 'NAAD' in col and ('x' in col or 'y' in col)]
    return len(naad_columns) > 0

def calculate_detection_metrics(EC_tracks_all, merged_tracks):
    """
    Вычисляет метрики обнаружения
    """
    hits = 0
    misses = 0
    false_alarms = 0
    
    # 1. Подсчет hits и misses для EC треков
    for ec_track in EC_tracks_all:
        # Находим соответствующий merged трек по ID или дате начала
        track_found = False
        for merged_track in merged_tracks:
            # Проверяем совпадение по дате начала и другим параметрам
            if (len(merged_track) > 0 and len(ec_track) > 0 and
                merged_track['datetime'].iloc[0] == ec_track['datetime'].iloc[0] and
                abs(merged_track['x'].iloc[0] - ec_track['x'].iloc[0]) < 10 and
                abs(merged_track['y'].iloc[0] - ec_track['y'].iloc[0]) < 10):
                
                if is_track_matched(merged_track):
                    hits += 1
                else:
                    misses += 1
                track_found = True
                break
        
        if not track_found:
            misses += 1
    
    # 2. Подсчет false alarms (NAAD треки без соответствующих EC треков)
    # Собираем все уникальные NAAD треки из merged файлов
    all_naad_tracks = set()
    
    for merged_track in merged_tracks:
        if is_track_matched(merged_track):
            # Извлекаем информацию о NAAD треках
            naad_columns = [col for col in merged_track.columns if 'NAAD' in col]
            naad_indices = set([col.split('_NAAD_')[-1] for col in naad_columns if 'NAAD_' in col])
            
            for idx in naad_indices:
                # Проверяем, есть ли данные для этого NAAD трека
                x_col = f'x_NAAD_{idx}'
                y_col = f'y_NAAD_{idx}'
                
                if x_col in merged_track.columns and y_col in merged_track.columns:
                    # Создаем уникальный идентификатор для NAAD трека
                    naad_data = merged_track[[x_col, y_col, 'datetime']].dropna()
                    if len(naad_data) > 0:
                        track_id = f"{naad_data[x_col].iloc[0]}_{naad_data[y_col].iloc[0]}_{naad_data['datetime'].iloc[0]}"
                        all_naad_tracks.add(track_id)
    
    false_alarms = len(all_naad_tracks) - hits  # NAAD треки без соответствующих EC треков
    
    return hits, misses, false_alarms

def calculate_detailed_metrics(EC_tracks_all, merged_tracks):
    """
    Вычисляет детализированные метрики
    """
    hits = 0
    misses = 0
    false_alarms = 0
    
    # Словари для хранения информации
    ec_track_info = {}
    naad_track_info = {}
    
    # Анализ EC треков
    for i, ec_track in enumerate(EC_tracks_all):
        ec_start = ec_track['datetime'].iloc[0]
        ec_key = f"EC_{i}_{ec_start}"
        ec_track_info[ec_key] = {'matched': False, 'naad_count': 0}
    
    # Анализ merged треков
    for merged_track in merged_tracks:
        if is_track_matched(merged_track):
            # Находим соответствующий EC трек
            ec_matched = False
            for i, ec_track in enumerate(EC_tracks_all):
                if (len(merged_track) > 0 and len(ec_track) > 0 and
                    merged_track['datetime'].iloc[0] == ec_track['datetime'].iloc[0] and
                    abs(merged_track['x'].iloc[0] - ec_track['x'].iloc[0]) < 10):
                    
                    ec_key = f"EC_{i}_{ec_track['datetime'].iloc[0]}"
                    ec_track_info[ec_key]['matched'] = True
                    ec_matched = True
                    hits += 1
                    break
            
            if not ec_matched:
                misses += 1
            
            # Подсчет NAAD треков
            naad_columns = [col for col in merged_track.columns if 'NAAD' in col and ('x' in col or 'y' in col)]
            naad_indices = set([col.split('_NAAD_')[-1] for col in naad_columns if 'NAAD_' in col])
            
            for idx in naad_indices:
                x_col = f'x_NAAD_{idx}'
                y_col = f'y_NAAD_{idx}'
                
                if x_col in merged_track.columns and y_col in merged_track.columns:
                    naad_data = merged_track[[x_col, y_col, 'datetime']].dropna()
                    if len(naad_data) > 0:
                        naad_key = f"NAAD_{idx}_{naad_data['datetime'].iloc[0]}"
                        naad_track_info[naad_key] = naad_track_info.get(naad_key, 0) + 1
    
    # Подсчет false alarms
    false_alarms = len(naad_track_info) - hits
    
    return hits, misses, false_alarms, ec_track_info, naad_track_info

def calculate_distance_metrics(merged_tracks):
    """
    Вычисляет статистику по расстояниям между треками
    """
    distances = []
    duration_ratios = []
    
    for merged_track in merged_tracks:
        if is_track_matched(merged_track):
            # Находим колонки NAAD
            naad_columns = [col for col in merged_track.columns if 'NAAD' in col and 'distance' in col]
            
            for col in naad_columns:
                dist_values = merged_track[col].dropna().values
                if len(dist_values) > 0:
                    distances.extend(dist_values)
            
            # Вычисляем соотношение длительностей
            ec_duration = (merged_track['datetime'].max() - merged_track['datetime'].min()).total_seconds() / 3600
            
            naad_indices = set([col.split('_NAAD_')[-1] for col in merged_track.columns if 'NAAD_' in col])
            for idx in naad_indices:
                datetime_col = f'datetime_NAAD_{idx}'
                if datetime_col in merged_track.columns:
                    naad_dates = merged_track[datetime_col].dropna()
                    if len(naad_dates) > 0:
                        naad_duration = (naad_dates.max() - naad_dates.min()).total_seconds() / 3600
                        if naad_duration > 0:
                            duration_ratios.append(ec_duration / naad_duration)
    
    return distances, duration_ratios

def save_metrics_report(hits, misses, false_alarms, distances, duration_ratios, output_file):
    """
    Сохраняет отчет с метриками
    """
    # Основные метрики
    pod = hits / (hits + misses) if (hits + misses) > 0 else 0
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else 0
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0
    
    metrics = {
        'Hits': hits,
        'Misses': misses,
        'False_Alarms': false_alarms,
        'POD': pod,
        'FAR': far,
        'CSI': csi,
        'Total_EC_Tracks': hits + misses,
        'Total_NAAD_Tracks': hits + false_alarms
    }
    
    # Статистика по расстояниям
    if distances:
        metrics.update({
            'Mean_Distance_km': np.mean(distances),
            'Median_Distance_km': np.median(distances),
            'Min_Distance_km': np.min(distances),
            'Max_Distance_km': np.max(distances),
            'Std_Distance_km': np.std(distances)
        })
    
    # Статистика по соотношению длительностей
    if duration_ratios:
        metrics.update({
            'Mean_Duration_Ratio': np.mean(duration_ratios),
            'Median_Duration_Ratio': np.median(duration_ratios),
            'Min_Duration_Ratio': np.min(duration_ratios),
            'Max_Duration_Ratio': np.max(duration_ratios)
        })
    
    # Сохраняем в CSV
    df_metrics = pd.DataFrame([metrics])
    df_metrics.to_csv(output_file, index=False)
    
    # Также сохраняем детальную статистику
    if distances:
        dist_stats = pd.DataFrame({'distances_km': distances})
        dist_stats.to_csv(output_file.replace('.csv', '_distances.csv'), index=False)
    
    if duration_ratios:
        ratio_stats = pd.DataFrame({'duration_ratios': duration_ratios})
        ratio_stats.to_csv(output_file.replace('.csv', '_durations.csv'), index=False)
    
    return metrics

def plot_metrics(metrics, output_dir):
    """
    Создает графики с метриками
    """
    # График основных метрик
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Метрики обнаружения
    detection_metrics = ['POD', 'FAR', 'CSI']
    values = [metrics[m] for m in detection_metrics]
    ax[0].bar(detection_metrics, values)
    ax[0].set_title('Detection Metrics')
    ax[0].set_ylim(0, 1)
    
    # Количество треков
    track_counts = ['Hits', 'Misses', 'False_Alarms']
    values = [metrics[m] for m in track_counts]
    ax[1].bar(track_counts, values)
    ax[1].set_title('Track Counts')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metrics_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

# Основной код выполнения
if __name__ == "__main__":
    # Параметры
    year = 2019
    data_type = 'SMP'  # или 'ERA5'
    sigma = 2
    tracking_type = 'tracking_local_2_phase'
    CVS_speed = 'adv_speed'
    
    # Пути
    path_dir_data = f'{path_init}/data'
    path_data_EC = f'{path_dir_data}/{data_type}/EddyClicker_tracks'
    merged_folder = f'{path_dir_data}/{data_type}/EC_tracks_merged_new/{data_type}_sigma_{sigma}/{tracking_type}_{CVS_speed}'
    output_dir = f'{path_dir_data}/{data_type}/EC_metrics_{sigma}'
    
    # Создаем выходную директорию
    os.makedirs(output_dir, exist_ok=True)
    
    # Загружаем все EC треки
    print("Loading EC tracks...")
    EC_tracks_all = load_EC_tracks(path_data_EC, year)
    print(f"Loaded {len(EC_tracks_all)} EC tracks")
    
    # Загружаем merged треки
    print("Loading merged tracks...")
    merged_tracks = load_merged_tracks(merged_folder)
    print(f"Loaded {len(merged_tracks)} merged tracks")
    
    # Вычисляем метрики
    print("Calculating metrics...")
    hits, misses, false_alarms = calculate_detection_metrics(EC_tracks_all, merged_tracks)
    
    # Вычисляем статистику по расстояниям
    distances, duration_ratios = calculate_distance_metrics(merged_tracks)
    
    # Сохраняем отчет
    output_file = f'{output_dir}/detection_metrics_sigma_{sigma}.csv'
    metrics = save_metrics_report(hits, misses, false_alarms, distances, duration_ratios, output_file)
    
    # Создаем графики
    plot_metrics(metrics, output_dir)
    
    print(f"\n=== Metrics Summary for sigma={sigma} ===")
    print(f"Hits: {hits}")
    print(f"Misses: {misses}")
    print(f"False Alarms: {false_alarms}")
    print(f"POD: {metrics['POD']:.3f}")
    print(f"FAR: {metrics['FAR']:.3f}")
    print(f"CSI: {metrics['CSI']:.3f}")
    
    if distances:
        print(f"\nDistance Statistics:")
        print(f"Mean: {np.mean(distances):.2f} km")
        print(f"Median: {np.median(distances):.2f} km")
        print(f"Min-Max: {np.min(distances):.2f} - {np.max(distances):.2f} km")
    
    print(f"\nDetailed report saved to: {output_file}")