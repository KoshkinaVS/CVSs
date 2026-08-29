import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
from geopy.distance import great_circle
import seaborn as sns
from tqdm import tqdm
import matplotlib.gridspec as gridspec

import xarray as xr

# Конвертация сеточных координат в географические
def grid_to_latlon(x, y, grid_spacing_km=6, origin_lat=0, origin_lon=0):
    """
    Конвертирует сеточные координаты в широту/долготу
    Приблизительно: 1 градус широты ~ 111 км
    """
    lat = origin_lat + y * grid_spacing_km / 111.0
    lon = origin_lon + x * grid_spacing_km / (111.0 * np.cos(np.radians(lat)))
    return lat, lon

def load_tracks_from_folder(folder_path, folder_name):
    """
    Загружает все треки из указанной папки
    """
    tracks_list = []
    csv_files = list(Path(folder_path).glob('*.csv'))
    
    print(f"Загрузка треков из {folder_name}...")
    for i, file_path in enumerate(tqdm(csv_files, desc=folder_name)):
        try:
            df = pd.read_csv(file_path, parse_dates=['time'])
            
            # Переименовываем колонки для единообразия
            df = df.rename(columns={
                'time': 'datetime',
                'pxc_ind': 'x',
                'pyc_ind': 'y'
            })
            
            # Добавляем информацию о треке
            df['track_id'] = i
            df['source'] = folder_name
            
            if len(df) >= 3:  # Минимальная длина трека
                tracks_list.append(df)
        except Exception as e:
            print(f"Ошибка при загрузке {file_path}: {e}")
            continue
    
    print(f"Загружено {len(tracks_list)} треков из {folder_name}")
    return tracks_list

def find_matching_tracks(ec_tracks, polina_tracks, max_distance_km=250, max_time_diff_hours=6):
    """
    Находит совпадающие треки между двумя наборами
    """
    matches = []
    
    for ec_id, ec_track in enumerate(tqdm(ec_tracks, desc="Сопоставление треков")):
        ec_times = set(ec_track['datetime'])
        
        best_match = None
        best_match_score = 0
        best_match_distance = float('inf')
        
        for polina_id, polina_track in enumerate(polina_tracks):
            # Проверка по времени
            polina_times = set(polina_track['datetime'])
            common_times = ec_times.intersection(polina_times)
            
            if len(common_times) == 0:
                continue
            
            # Вычисляем среднее расстояние между треками в общие моменты времени
            ec_subset = ec_track[ec_track['datetime'].isin(common_times)].sort_values('datetime')
            polina_subset = polina_track[polina_track['datetime'].isin(common_times)].sort_values('datetime')
            
            # Проверяем, что у нас одинаковое количество точек
            if len(ec_subset) != len(polina_subset):
                # Если разное количество, берем минимальное
                min_len = min(len(ec_subset), len(polina_subset))
                ec_subset = ec_subset.iloc[:min_len]
                polina_subset = polina_subset.iloc[:min_len]
            
            if len(ec_subset) == 0:
                continue
            
            # Векторизованное вычисление расстояний
            distances = []
            for i in range(len(ec_subset)):
                # Используем .iloc[i] для доступа по позиции
                x1 = ec_subset['x'].iloc[i]
                y1 = ec_subset['y'].iloc[i]
                x2 = polina_subset['x'].iloc[i]
                y2 = polina_subset['y'].iloc[i]
                
                dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) * 77  # 77 км на шаг сетки
                distances.append(dist)
            
            avg_distance = np.mean(distances)
            
            if avg_distance > max_distance_km:
                continue
            
            # Оценка качества совпадения
            time_overlap_ratio = len(common_times) / max(len(ec_times), len(polina_times))
            distance_score = 1 / (1 + avg_distance / max_distance_km)
            
            # Проверяем начало и конец треков
            ec_start = ec_track['datetime'].min()
            ec_end = ec_track['datetime'].max()
            polina_start = polina_track['datetime'].min()
            polina_end = polina_track['datetime'].max()
            
            start_diff = abs((ec_start - polina_start).total_seconds() / 3600)
            end_diff = abs((ec_end - polina_end).total_seconds() / 3600)
            
            # Комбинированная оценка
            combined_score = (time_overlap_ratio * 0.4 + 
                            distance_score * 0.4 +
                            (1 / (1 + start_diff / max_time_diff_hours)) * 0.1 +
                            (1 / (1 + end_diff / max_time_diff_hours)) * 0.1)
            
            if combined_score > best_match_score:
                best_match_score = combined_score
                best_match = polina_id
                best_match_distance = avg_distance
        
        if best_match is not None:
            matches.append({
                'ec_track_id': ec_id,
                'polina_track_id': best_match,
                'match_score': best_match_score,
                'avg_distance_km': best_match_distance,
                'ec_start': ec_track['datetime'].min(),
                'ec_end': ec_track['datetime'].max(),
                'polina_start': polina_tracks[best_match]['datetime'].min(),
                'polina_end': polina_tracks[best_match]['datetime'].max(),
                'ec_points': len(ec_track),
                'polina_points': len(polina_tracks[best_match]),
                'common_points': len(set(ec_track['datetime']).intersection(
                    set(polina_tracks[best_match]['datetime'])))
            })
    
    return matches




def plot_matched_tracks(ec_tracks, polina_tracks, matches, output_dir='matching_plots', ds=None):
    """
    Создает карты с сопоставленными треками в декартовых координатах с подложкой из ds
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if len(matches) == 0:
        print("Нет совпадений для визуализации")
        return
    
    # Создаем отдельные карты для каждой пары
    for i, match in enumerate(matches):
        try:
            # Создаем фигуру размером 5x5 как в примере
            fig = plt.figure(figsize=(5, 5), dpi=150)
            ax = fig.add_subplot(111)
            
            # Добавляем подложку из ds если доступна
            if ds is not None and 'HGT' in ds.variables:
                # Используем HGT для создания подложки (где HGT > 15 - суша)
                land_mask = xr.where(ds['HGT'] > 15, 1, np.nan)
                ax.contourf(land_mask, cmap='Greys', alpha=0.7, levels=[0.5, 1.5])
            
            ec_track = ec_tracks[match['ec_track_id']]
            polina_track = polina_tracks[match['polina_track_id']]
            
            # Рисуем EC трек (оригинальный) - черным цветом как в примере
            ax.plot(ec_track['x'], ec_track['y'], 
                   c='k', linewidth=1.5, label='Oboltus Track', zorder=5)
            

            # Рисуем Polina трек (сопоставленный)
            # Форматируем время начала для подписи как в примере: %Y-%m-%dT%H
            if 'datetime' in polina_track.columns:
                polina_start = pd.to_datetime(polina_track['datetime']).min()
                polina_start_str = polina_start.strftime('%Y-%m-%dT%H')
            else:
                polina_start_str = "unknown"
                polina_start = None
            
            if 'datetime' in ec_track.columns:
                ec_start = pd.to_datetime(ec_track['datetime']).min()
            else:
                ec_start = None
            
            # Вычисляем разницу во времени начала (в часах)
            if polina_start is not None and ec_start is not None:
                time_diff_hours = abs((polina_start - ec_start).total_seconds() / 3600)
                time_diff_str = f"{time_diff_hours:.1f}h"
            else:
                time_diff_str = "?"
            
            dist = match['avg_distance_km']
            ax.plot(polina_track['x'], polina_track['y'], 
                   label=f'Polina Track ({int(dist)} km, {time_diff_str})', 
                   linewidth=3, zorder=4,)
            
            # Отмечаем начало и конец EC трека как в примере
            if len(ec_track) > 0:
                ax.scatter(ec_track['x'].values[0], ec_track['y'].values[0], 
                          c='g', s=7, zorder=10, label='Oboltus start')
                ax.scatter(ec_track['x'].values[-1], ec_track['y'].values[-1], 
                          c='r', s=7, zorder=10, label='Oboltus stop')

            # Отмечаем начало и конец EC трека Polina
            if len(polina_track) > 0:
                ax.scatter(polina_track['x'].values[0], polina_track['y'].values[0], marker='*',
                          c='lightgreen', s=7, zorder=10, label='Polina start')
                ax.scatter(polina_track['x'].values[-1], polina_track['y'].values[-1], marker='*',
                          c='orange', s=7, zorder=10, label='Polina stop')
            
            # Настраиваем заголовок как в примере
            if 'datetime' in ec_track.columns:
                track_start = pd.to_datetime(ec_track['datetime']).min()
                start_str = track_start.strftime('%Y-%m-%dT%H')
                ax.set_title(f'Track {match["ec_track_id"]+1} at {start_str}')
            else:
                ax.set_title(f'Track {match["ec_track_id"]+1}')
            
            # Устанавливаем границы как в примере
            ax.set_xlim(0, 110)
            # Можно также установить ylim если нужно
            ax.set_ylim(0, 110)
            
            # Добавляем легенду с маленьким шрифтом как в примере
            ax.legend(fontsize=7, loc='upper right')
            
            # Сохраняем в формате как в примере
            track_num = match['ec_track_id'] + 1
            filename = f'{output_dir}/LoRes_{track_num:09d}_sigma_2_matched.png'
            fig.savefig(filename, dpi=200, bbox_inches="tight", transparent=False)
            plt.close(fig)
            
            print(f"Сохранено: {filename}")
            
        except Exception as e:
            print(f"Ошибка при создании графика для пары {i+1}: {e}")
            plt.close('all')
            continue



def plot_worst_matches_detailed(ec_tracks, polina_tracks, matches, output_dir='detailed_plots', 
                               ds=None, data_type='LoRes', sigma=2, n_worst=10):
    """
    Создает детальные графики для самых худших совпадений
    """
    if len(matches) == 0:
        print("Нет совпадений для детального анализа")
        return
    
    # Сортируем по качеству совпадения (от худших к лучшим)
    sorted_matches = sorted(matches, key=lambda x: x['match_score'])
    
    print(f"\nСоздание детальных графиков для {min(n_worst, len(sorted_matches))} ХУДШИХ совпадений:")
    print("="*60)
    
    for i, match in enumerate(sorted_matches[:n_worst]):
        ec_track = ec_tracks[match['ec_track_id']]
        polina_track = polina_tracks[match['polina_track_id']]
        
        print(f"\n{i+1}. EC Track {match['ec_track_id']} - Polina Track {match['polina_track_id']}")
        print(f"   Оценка качества: {match['match_score']:.3f}")
        print(f"   Среднее расстояние: {match['avg_distance_km']:.1f} км")
        print(f"   Общих точек: {match['common_points']}")
        
        plot_detailed_track_comparison(
            ec_track, polina_track, match,
            output_dir=output_dir, ds=ds,
            track_num=i+1, data_type=data_type, sigma=sigma,
            is_worst=True  # Добавляем флаг для худших
        )

def plot_detailed_track_comparison(ec_track, polina_track, match_info, 
                                   output_dir='detailed_plots', ds=None, 
                                   track_num=None, data_type='LoRes', sigma=2,
                                   is_worst=False):
    """
    Создает детальный график сравнения для одной пары со статистикой
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Создаем фигуру с сеткой для нескольких графиков
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 0.6], 
                               hspace=0.3, wspace=0.3)
        
        # ===== ГРАФИК 1: Карта с треками =====
        ax1 = plt.subplot(gs[0, :])
        
        # Добавляем подложку из ds если доступна
        if ds is not None and 'HGT' in ds.variables:
            land_mask = xr.where(ds['HGT'] > 15, 1, np.nan)
            ax1.contourf(land_mask, cmap='Greys', alpha=0.7, levels=[0.5, 1.5])
        
        # Получаем время начала для подписей
        if 'datetime' in ec_track.columns:
            ec_start = pd.to_datetime(ec_track['datetime']).min()
            ec_end = pd.to_datetime(ec_track['datetime']).max()
            ec_start_str = ec_start.strftime('%Y-%m-%dT%H')
            ec_end_str = ec_end.strftime('%Y-%m-%dT%H')
            ec_duration = (ec_end - ec_start).total_seconds() / 3600
        else:
            ec_start_str = "unknown"
            ec_end_str = "unknown"
            ec_duration = 0
            ec_start = None
            
        if 'datetime' in polina_track.columns:
            polina_start = pd.to_datetime(polina_track['datetime']).min()
            polina_end = pd.to_datetime(polina_track['datetime']).max()
            polina_start_str = polina_start.strftime('%Y-%m-%dT%H')
            polina_end_str = polina_end.strftime('%Y-%m-%dT%H')
            polina_duration = (polina_end - polina_start).total_seconds() / 3600
        else:
            polina_start_str = "unknown"
            polina_end_str = "unknown"
            polina_duration = 0
            polina_start = None
        
        # Вычисляем разницу во времени
        if polina_start is not None and ec_start is not None:
            time_diff_start = (polina_start - ec_start).total_seconds() / 3600
            time_diff_end = (polina_end - ec_end).total_seconds() / 3600
            
            if time_diff_start >= 0:
                time_diff_start_str = f"+{time_diff_start:.1f}h"
            else:
                time_diff_start_str = f"{time_diff_start:.1f}h"
                
            if time_diff_end >= 0:
                time_diff_end_str = f"+{time_diff_end:.1f}h"
            else:
                time_diff_end_str = f"{time_diff_end:.1f}h"
        else:
            time_diff_start_str = "?"
            time_diff_end_str = "?"
        
        # Определяем цвет рамки для худших совпадений
        if is_worst:
            frame_color = 'red'
            title_prefix = "ХУДШЕЕ СОВПАДЕНИЕ #"
        else:
            frame_color = 'blue'
            title_prefix = "Совпадение #"
        
        # Рисуем треки
        ax1.plot(ec_track['x'], ec_track['y'], 'b-', linewidth=2.5, 
                label=f'EC Track ({ec_start_str} - {ec_end_str}, {ec_duration:.1f}h)', zorder=5)
        ax1.plot(polina_track['x'], polina_track['y'], 'r--', linewidth=2, 
                label=f'Polina Track ({polina_start_str} - {polina_end_str}, {polina_duration:.1f}h)', zorder=4)
        
        # Отмечаем точки треков
        ax1.scatter(ec_track['x'], ec_track['y'], c='blue', s=20, alpha=0.5, zorder=3)
        ax1.scatter(polina_track['x'], polina_track['y'], c='red', s=20, alpha=0.5, zorder=3)
        
        # Отмечаем начало и конец
        if len(ec_track) > 0:
            ax1.scatter(ec_track['x'].values[0], ec_track['y'].values[0], 
                       c='lime', s=100, marker='o', edgecolor='black', 
                       linewidth=1.5, zorder=10, label='EC Start')
            ax1.scatter(ec_track['x'].values[-1], ec_track['y'].values[-1], 
                       c='orange', s=100, marker='s', edgecolor='black', 
                       linewidth=1.5, zorder=10, label='EC End')
        
        # Добавляем информацию о совпадении на карту
        info_box = (f'Среднее расстояние: {match_info["avg_distance_km"]:.1f} км\n'
                   f'Общих точек: {match_info["common_points"]}\n'
                   f'Оценка качества: {match_info["match_score"]:.3f}\n'
                   f'Разница начала: {time_diff_start_str}\n'
                   f'Разница конца: {time_diff_end_str}')
        
        # Для худших добавляем предупреждение
        if is_worst:
            info_box = f'⚠️ ПЛОХОЕ СОВПАДЕНИЕ ⚠️\n' + info_box
            bbox_color = 'lightcoral'
        else:
            bbox_color = 'wheat'
        
        ax1.text(0.02, 0.98, info_box, transform=ax1.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=bbox_color, alpha=0.9))
        
        ax1.set_xlabel('X координата')
        ax1.set_ylabel('Y координата')
        
        if track_num is not None:
            ax1.set_title(f'{title_prefix}{track_num}: EC {match_info["ec_track_id"]} vs Polina {match_info["polina_track_id"]}')
        else:
            ax1.set_title(f'EC {match_info["ec_track_id"]} vs Polina {match_info["polina_track_id"]}')
        
        ax1.legend(fontsize=8, loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # ===== ГРАФИК 2: Расстояние между треками по времени =====
        ax2 = plt.subplot(gs[1, 0])
        
        # Находим общие моменты времени
        if 'datetime' in ec_track.columns and 'datetime' in polina_track.columns:
            ec_times = set(ec_track['datetime'])
            polina_times = set(polina_track['datetime'])
            common_times = sorted(list(ec_times.intersection(polina_times)))
            
            if common_times:
                # Вычисляем расстояния для каждого общего момента времени
                distances = []
                time_labels = []
                
                for t in common_times:
                    ec_point = ec_track[ec_track['datetime'] == t].iloc[0]
                    polina_point = polina_track[polina_track['datetime'] == t].iloc[0]
                    
                    dist = np.sqrt((polina_point['x'] - ec_point['x'])**2 + 
                                  (polina_point['y'] - ec_point['y'])**2) * 6  # км
                    distances.append(dist)
                    time_labels.append(pd.to_datetime(t).strftime('%m-%d %H:%M'))
                
                # Рисуем график расстояний
                x_pos = range(len(time_labels))
                bars = ax2.bar(x_pos, distances, color='steelblue', alpha=0.7, width=0.6)
                
                # Подкрашиваем столбцы, где расстояние большое
                for j, (bar, d) in enumerate(zip(bars, distances)):
                    if d > 50:
                        bar.set_color('red')
                    elif d > 30:
                        bar.set_color('orange')
                
                ax2.axhline(y=match_info['avg_distance_km'], color='red', 
                           linestyle='--', linewidth=2, label=f'Среднее: {match_info["avg_distance_km"]:.1f} км')
                ax2.axhline(y=50, color='darkred', linestyle=':', linewidth=1.5, alpha=0.7, label='Порог 50 км')
                
                ax2.set_xlabel('Время')
                ax2.set_ylabel('Расстояние (км)')
                ax2.set_title('Расстояние между треками во времени')
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(time_labels, rotation=45, ha='right', fontsize=8)
                ax2.legend(fontsize=8)
                ax2.grid(True, alpha=0.3)
                
                # Добавляем значения над столбцами
                for j, (x, d) in enumerate(zip(x_pos, distances)):
                    ax2.text(x, d + 2, f'{d:.0f}', ha='center', va='bottom', fontsize=7)
        
        # ===== ГРАФИК 3: Метрики качества =====
        ax3 = plt.subplot(gs[1, 1])
        
        # Вычисляем метрики
        # 1. Временное перекрытие
        if 'datetime' in ec_track.columns and 'datetime' in polina_track.columns:
            ec_times = set(ec_track['datetime'])
            polina_times = set(polina_track['datetime'])
            common_count = len(ec_times.intersection(polina_times))
            time_overlap = common_count / max(len(ec_times), len(polina_times))
        else:
            time_overlap = match_info['common_points'] / max(match_info['ec_points'], match_info['polina_points'])
        
        # 2. Distance score (0-1, где 1 - идеально)
        distance_score = 1 / (1 + match_info['avg_distance_km'] / 20)  # 20 км - порог
        
        # 3. Temporal shift score
        if ec_start is not None and polina_start is not None:
            start_diff = abs((polina_start - ec_start).total_seconds() / 3600)
            end_diff = abs((pd.to_datetime(ec_track['datetime'].max()) - 
                          pd.to_datetime(polina_track['datetime'].max())).total_seconds() / 3600)
            temporal_score = 1 / (1 + (start_diff + end_diff) / 12)  # 12 часов - порог
        else:
            temporal_score = match_info['match_score']
        
        # 4. Path similarity score
        if len(ec_track) == len(polina_track) and len(ec_track) > 1:
            # Вычисляем корреляцию траекторий
            ec_dx = np.diff(ec_track['x'].values)
            ec_dy = np.diff(ec_track['y'].values)
            polina_dx = np.diff(polina_track['x'].values)
            polina_dy = np.diff(polina_track['y'].values)
            
            # Нормализуем векторы
            ec_norm = np.sqrt(ec_dx**2 + ec_dy**2)
            polina_norm = np.sqrt(polina_dx**2 + polina_dy**2)
            
            # Избегаем деления на ноль
            ec_norm = np.where(ec_norm == 0, 1, ec_norm)
            polina_norm = np.where(polina_norm == 0, 1, polina_norm)
            
            ec_dx_norm = ec_dx / ec_norm
            ec_dy_norm = ec_dy / ec_norm
            polina_dx_norm = polina_dx / polina_norm
            polina_dy_norm = polina_dy / polina_norm
            
            # Скалярное произведение нормализованных векторов
            dot_product = ec_dx_norm * polina_dx_norm + ec_dy_norm * polina_dy_norm
            path_similarity = np.mean(np.clip(dot_product, -1, 1))  # Косинусное сходство
            path_score = (path_similarity + 1) / 2  # Нормализуем в [0, 1]
        else:
            path_score = 0.5  # Значение по умолчанию
        
        metrics = ['Time Overlap', 'Distance', 'Temporal', 'Path']
        scores = [time_overlap, distance_score, temporal_score, path_score]
        
        # Цвета в зависимости от значения
        colors = []
        for s in scores:
            if s >= 0.8:
                colors.append('green')
            elif s >= 0.6:
                colors.append('yellowgreen')
            elif s >= 0.4:
                colors.append('orange')
            else:
                colors.append('red')
        
        bars = ax3.bar(metrics, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax3.set_ylim(0, 1)
        ax3.set_ylabel('Score (0-1)')
        ax3.set_title('Метрики качества сопоставления')
        ax3.axhline(y=0.8, color='green', linestyle=':', alpha=0.5, label='Отлично')
        ax3.axhline(y=0.6, color='orange', linestyle=':', alpha=0.5, label='Хорошо')
        ax3.axhline(y=0.4, color='red', linestyle=':', alpha=0.5, label='Удовл.')
        
        # Добавляем значения на столбцы
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{score:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax3.legend(fontsize=8, loc='lower right')
        ax3.set_xticklabels(metrics, fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Добавляем общий заголовок с указанием ранга
        if track_num is not None:
            if is_worst:
                plt.suptitle(f'ХУДШЕЕ СОВПАДЕНИЕ #{track_num} (EC {match_info["ec_track_id"]} - Polina {match_info["polina_track_id"]})', 
                            fontsize=14, y=1.02, color='red', weight='bold')
            else:
                plt.suptitle(f'Детальный анализ #{track_num}: EC {match_info["ec_track_id"]} - Polina {match_info["polina_track_id"]}', 
                            fontsize=14, y=1.02)
        else:
            plt.suptitle(f'EC {match_info["ec_track_id"]} - Polina {match_info["polina_track_id"]}', 
                        fontsize=14, y=1.02)
        
        # Сохраняем с соответствующим префиксом
        if track_num is not None:
            if is_worst:
                filename = f'{output_dir}/{data_type}_WORST_{track_num:02d}_sigma_{sigma}.png'
            else:
                filename = f'{output_dir}/{data_type}_detailed_{track_num:03d}_sigma_{sigma}.png'
        else:
            filename = f'{output_dir}/{data_type}_detailed_EC{match_info["ec_track_id"]:03d}_sigma_{sigma}.png'
        
        plt.tight_layout()
        fig.savefig(filename, dpi=200, bbox_inches="tight", transparent=False)
        plt.close(fig)
        
        print(f"   Детальный график сохранен: {filename}")
        
    except Exception as e:
        print(f"Ошибка при создании детального графика: {e}")
        plt.close('all')
        
def create_matching_statistics(ec_tracks, polina_tracks, matches):
    """
    Создает статистику сопоставления треков
    """
    stats = {
        'total_ec_tracks': len(ec_tracks),
        'total_polina_tracks': len(polina_tracks),
        'matched_pairs': len(matches),
        'ec_coverage': len(matches) / len(ec_tracks) * 100 if ec_tracks else 0,
        'polina_coverage': len(matches) / len(polina_tracks) * 100 if polina_tracks else 0
    }
    
    if matches:
        # Статистика по расстояниям
        distances = [m['avg_distance_km'] for m in matches]
        stats['avg_distance_km'] = np.mean(distances)
        stats['std_distance_km'] = np.std(distances)
        stats['min_distance_km'] = np.min(distances)
        stats['max_distance_km'] = np.max(distances)
        
        # Статистика по временному перекрытию
        common_points = [m['common_points'] for m in matches]
        stats['avg_common_points'] = np.mean(common_points)
        stats['total_common_points'] = np.sum(common_points)
        
        # Статистика по оценкам совпадения
        scores = [m['match_score'] for m in matches]
        stats['avg_match_score'] = np.mean(scores)
        stats['std_match_score'] = np.std(scores)
        
        # Анализ начала и конца треков
        start_diffs = []
        end_diffs = []
        for m in matches:
            start_diff = abs((m['ec_start'] - m['polina_start']).total_seconds() / 3600)
            end_diff = abs((m['ec_end'] - m['polina_end']).total_seconds() / 3600)
            start_diffs.append(start_diff)
            end_diffs.append(end_diff)
        
        stats['avg_start_diff_hours'] = np.mean(start_diffs)
        stats['avg_end_diff_hours'] = np.mean(end_diffs)
        
        # Категории качества
        excellent = sum(1 for d in distances if d < 20)
        good = sum(1 for d in distances if 20 <= d < 40)
        fair = sum(1 for d in distances if 40 <= d < 60)
        poor = sum(1 for d in distances if d >= 60)
        
        stats['quality_categories'] = {
            'excellent (<20 km)': excellent,
            'good (20-40 km)': good,
            'fair (40-60 km)': fair,
            'poor (>=60 km)': poor
        }
    
    return stats

def plot_statistics(ec_tracks, polina_tracks, matches, stats, output_dir='matching_plots'):
    """
    Создает визуализацию статистики с информацией о:
    - доле треков папки 1 (Oboltus) в папке Polina
    - смещении по времени начала (со знаком)
    - смещении по времени конца (со знаком)
    - отношении длин треков Oboltus/Polina
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Вычисляем дополнительные метрики для графиков
    if matches:
        # Вычисляем смещения начала и конца со знаком
        start_shifts = []
        end_shifts = []
        ec_lengths = []
        polina_lengths = []
        length_ratios = []  # Отношение Oboltus/Polina
        
        for m in matches:
            # Смещение начала
            start_shift = (m['polina_start'] - m['ec_start']).total_seconds() / 3600
            start_shifts.append(start_shift)
            
            # Смещение конца
            end_shift = (m['polina_end'] - m['ec_end']).total_seconds() / 3600
            end_shifts.append(end_shift)
            
            # Длины треков и их отношение
            ec_len = m['ec_points']
            polina_len = m['polina_points']
            ec_lengths.append(ec_len)
            polina_lengths.append(polina_len)
            
            # Отношение Oboltus/Polina (избегаем деления на 0)
            if polina_len > 0:
                length_ratios.append(ec_len / polina_len)
            else:
                length_ratios.append(np.nan)
    
    # Создаем фигуру с сеткой графиков (3 ряда по 3 графика)
    fig = plt.figure(figsize=(20, 18))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # ===== РЯД 1: Общая статистика и доли =====
    
    # 1. Доля треков EC/Oboltus
    ax1 = fig.add_subplot(gs[0, 0])
    
    ec_matched = len(matches)
    ec_unmatched = stats['total_ec_tracks'] - ec_matched
    
    ax1.pie([ec_matched, ec_unmatched], 
            labels=[f'EC совпало\n{ec_matched}', f'EC не совпало\n{ec_unmatched}'],
            colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%', startangle=90)
    ax1.set_title(f'EC/Oboltus треки (всего: {stats["total_ec_tracks"]})', fontsize=12)
    
    # 2. Доля треков Polina
    ax2 = fig.add_subplot(gs[0, 1])
    
    polina_matched = len(set([m['polina_track_id'] for m in matches]))
    polina_unmatched = stats['total_polina_tracks'] - polina_matched
    
    ax2.pie([polina_matched, polina_unmatched], 
            labels=[f'Polina совпало\n{polina_matched}', f'Polina не совпало\n{polina_unmatched}'],
            colors=['skyblue', 'pink'], autopct='%1.1f%%', startangle=90)
    ax2.set_title(f'Polina треки (всего: {stats["total_polina_tracks"]})', fontsize=12)
    
    # 3. Общая статистика текстом
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    # Создаем информационное поле
    info_text = (
        f"ОБЩАЯ СТАТИСТИКА\n"
        f"{'='*25}\n\n"
        f"EC/Oboltus треков: {stats['total_ec_tracks']}\n"
        f"Polina треков: {stats['total_polina_tracks']}\n"
        f"Найдено совпадений: {stats['matched_pairs']}\n\n"
        f"Доля EC в Polina:\n"
        f"  {stats['ec_coverage']:.1f}% EC треков имеют совпадение\n"
        f"  {polina_matched/stats['total_polina_tracks']*100:.1f}% Polina треков имеют совпадение\n\n"
        f"Среднее расстояние: {stats.get('avg_distance_km', 0):.1f} км\n"
        f"Средняя оценка: {stats.get('avg_match_score', 0):.3f}"
    )
    
    ax3.text(0.1, 0.5, info_text, transform=ax3.transAxes,
            fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # ===== РЯД 2: Временные смещения =====
    
    # 4. Гистограмма смещений начала
    ax4 = fig.add_subplot(gs[1, 0])
    if matches and start_shifts:
        bins = np.linspace(-24, 24, 25)
        n, bins, patches = ax4.hist(start_shifts, bins=bins, edgecolor='black', alpha=0.7)
        
        for i, patch in enumerate(patches):
            if bins[i] < 0:
                patch.set_facecolor('lightcoral')
            else:
                patch.set_facecolor('lightgreen')
        
        ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax4.axvline(x=np.mean(start_shifts), color='red', linestyle='--', 
                   label=f'Среднее: {np.mean(start_shifts):.1f} ч')
        
        ax4.text(0.02, 0.75, f'РАНЬШЕ: {sum(s < 0 for s in start_shifts)} треков\n(Polina начала до Oboltus)', 
                transform=ax4.transAxes, color='darkred', fontsize=9)
        ax4.text(0.55, 0.75, f'ПОЗЖЕ: {sum(s > 0 for s in start_shifts)} треков\n(Polina начала после Oboltus)', 
                transform=ax4.transAxes, color='darkgreen', fontsize=9)
        
        ax4.set_xlabel('Смещение начала трека Polina относительно Oboltus (часы)')
        ax4.set_ylabel('Количество пар')
        ax4.set_title('СМЕЩЕНИЕ НАЧАЛА ТРЕКОВ\n(-) Polina раньше Oboltus | (+) Polina позже Oboltus')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)

    # 5. Гистограмма смещений конца
    ax5 = fig.add_subplot(gs[1, 1])
    if matches and end_shifts:
        bins = np.linspace(-24, 24, 25)
        n, bins, patches = ax5.hist(end_shifts, bins=bins, edgecolor='black', alpha=0.7)
        
        for i, patch in enumerate(patches):
            if bins[i] < 0:
                patch.set_facecolor('lightcoral')
            else:
                patch.set_facecolor('lightgreen')
        
        ax5.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax5.axvline(x=np.mean(end_shifts), color='red', linestyle='--', 
                   label=f'Среднее: {np.mean(end_shifts):.1f} ч')
        
        ax5.text(0.02, 0.75, f'РАНЬШЕ: {sum(s < 0 for s in end_shifts)} треков\n(Polina закончила до Oboltus)', 
                transform=ax5.transAxes, color='darkred', fontsize=9)
        ax5.text(0.55, 0.75, f'ПОЗЖЕ: {sum(s > 0 for s in end_shifts)} треков\n(Polina закончила после Oboltus)', 
                transform=ax5.transAxes, color='darkgreen', fontsize=9)
        
        ax5.set_xlabel('Смещение конца трека Polina относительно Oboltus (часы)')
        ax5.set_ylabel('Количество пар')
        ax5.set_title('СМЕЩЕНИЕ КОНЦА ТРЕКОВ\n(-) Polina раньше Oboltus | (+) Polina позже Oboltus')
        ax5.legend(loc='upper right')
        ax5.grid(True, alpha=0.3)
        
    # 6. Гистограмма отношения длин Oboltus/Polina (НОВЫЙ ГРАФИК)
    ax9 = fig.add_subplot(gs[1, 2])
    if matches and length_ratios:
        # Убираем NaN значения
        clean_ratios = [r for r in length_ratios if not np.isnan(r)]
        
        if clean_ratios:
            # Создаем бины от 0 до 3 с шагом 0.2
            bins = np.linspace(0, 3, 16)
            n, bins, patches = ax9.hist(clean_ratios, bins=bins, edgecolor='black', alpha=0.7)
            
            # Раскрашиваем в зависимости от значения
            for i, patch in enumerate(patches):
                ratio_center = (bins[i] + bins[i+1]) / 2
                if ratio_center < 0.8:
                    patch.set_facecolor('lightcoral')  # Oboltus короче
                elif ratio_center > 1.2:
                    patch.set_facecolor('lightgreen')  # Oboltus длиннее
                else:
                    patch.set_facecolor('lightblue')   # Примерно равны
            
            ax9.axvline(x=1.0, color='black', linestyle='-', linewidth=2, label='Равные длины')
            ax9.axvline(x=np.mean(clean_ratios), color='red', linestyle='--', 
                       label=f'Среднее: {np.mean(clean_ratios):.2f}')
            
            # Добавляем статистику
            shorter = sum(r < 0.8 for r in clean_ratios)
            equal = sum(0.8 <= r <= 1.2 for r in clean_ratios)
            longer = sum(r > 1.2 for r in clean_ratios)
            
            ax9.text(0.02, 0.75, f'Oboltus КОРОЧЕ: {shorter} треков\n(ratio < 0.8)', 
                    transform=ax9.transAxes, color='darkred', fontsize=9)
            ax9.text(0.35, 0.75, f'ПРИМЕРНО РАВНЫ: {equal} треков\n(0.8-1.2)', 
                    transform=ax9.transAxes, color='darkblue', fontsize=9)
            ax9.text(0.70, 0.75, f'Oboltus ДЛИННЕЕ: {longer} треков\n(ratio > 1.2)', 
                    transform=ax9.transAxes, color='darkgreen', fontsize=9)
            
            ax9.set_xlabel('Отношение длин Oboltus / Polina')
            ax9.set_ylabel('Количество пар')
            ax9.set_title('ОТНОШЕНИЕ ДЛИН ТРЕКОВ\n<1 Oboltus короче | >1 Oboltus длиннее')
            ax9.legend(loc='upper right')
            ax9.grid(True, alpha=0.3)
    
    plt.suptitle(f'ДЕТАЛЬНАЯ СТАТИСТИКА СОПОСТАВЛЕНИЯ (всего пар: {len(matches)})', 
                fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'matching_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Сохраняем сырые данные в CSV
    if matches:
        shifts_df = pd.DataFrame({
            'ec_track_id': [m['ec_track_id'] for m in matches],
            'polina_track_id': [m['polina_track_id'] for m in matches],
            'ec_start': [m['ec_start'] for m in matches],
            'polina_start': [m['polina_start'] for m in matches],
            'start_shift_hours': start_shifts,
            'ec_end': [m['ec_end'] for m in matches],
            'polina_end': [m['polina_end'] for m in matches],
            'end_shift_hours': end_shifts,
            'ec_length': ec_lengths,
            'polina_length': polina_lengths,
            'length_ratio': length_ratios,  # Отношение Oboltus/Polina
            'avg_distance_km': [m['avg_distance_km'] for m in matches],
            'match_score': [m['match_score'] for m in matches]
        })
        shifts_df.to_csv(os.path.join(output_dir, 'temporal_shifts.csv'), index=False)
        print(f"  - Данные о временных смещениях и длинах: {output_dir}/temporal_shifts.csv")
        
def save_matching_results(ec_tracks, polina_tracks, matches, stats, output_dir='matching_results'):
    """
    Сохраняет результаты сопоставления в CSV файлы
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Сохраняем информацию о совпадениях
    matches_df = pd.DataFrame(matches)
    matches_df.to_csv(os.path.join(output_dir, 'matching_pairs.csv'), index=False)
    
    # Сохраняем статистику
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(os.path.join(output_dir, 'matching_statistics.csv'), index=False)
    
    # Сохраняем детальную информацию по каждому треку EC
    ec_matching_info = []
    for ec_id, ec_track in enumerate(ec_tracks):
        matching_track = next((m for m in matches if m['ec_track_id'] == ec_id), None)
        
        info = {
            'ec_track_id': ec_id,
            'ec_start': ec_track['datetime'].min(),
            'ec_end': ec_track['datetime'].max(),
            'ec_points': len(ec_track),
            'ec_center_x': ec_track['x'].mean(),
            'ec_center_y': ec_track['y'].mean(),
            'has_match': matching_track is not None
        }
        
        if matching_track:
            info.update({
                'polina_track_id': matching_track['polina_track_id'],
                'match_score': matching_track['match_score'],
                'avg_distance_km': matching_track['avg_distance_km'],
                'common_points': matching_track['common_points'],
                'polina_start': matching_track['polina_start'],
                'polina_end': matching_track['polina_end']
            })
        
        ec_matching_info.append(info)
    
    ec_df = pd.DataFrame(ec_matching_info)
    ec_df.to_csv(os.path.join(output_dir, 'ec_tracks_matching_info.csv'), index=False)
    
    print(f"\nРезультаты сохранены в папку: {output_dir}")




def main():
    """
    Основная функция для сопоставления треков из двух папок
    """
    print("="*60)
    print("СОПОСТАВЛЕНИЕ ТРЕКОВ ИЗ ПАПОК EddyClicker_tracks И EddyClicker_Polina")
    print("="*60)
    
    # Пути к папкам
    path_init = '/storage/thalassa/users/vkoshkina'
    folder1_path = f'{path_init}/data/LoRes/LoRes/EddyClicker_tracks'
    folder2_path = f'{path_init}/data/LoRes/LoRes/EddyClicker_Polina'
    
    # Проверяем существование папок
    if not os.path.exists(folder1_path):
        print(f"ОШИБКА: Папка не найдена: {folder1_path}")
        folder1_path = input("Введите путь к папке EddyClicker_tracks: ")
    
    if not os.path.exists(folder2_path):
        print(f"ОШИБКА: Папка не найдена: {folder2_path}")
        folder2_path = input("Введите путь к папке EddyClicker_Polina: ")
    
    print(f"\nПапка 1 (EC): {folder1_path}")
    print(f"Папка 2 (Polina): {folder2_path}")
    
    # Загружаем треки
    ec_tracks = load_tracks_from_folder(folder1_path, "EddyClicker_tracks")
    polina_tracks = load_tracks_from_folder(folder2_path, "EddyClicker_Polina")
    
    if not ec_tracks or not polina_tracks:
        print("Ошибка: Не удалось загрузить треки из одной из папок")
        return

    # После загрузки треков добавьте загрузку ds
    path_dir_data_ds = f"{path_init}/data/LoRes/LoRes/DBSCAN_02-04-10_smoothing_sigma_2_daily/2010"
    ncfile = f'{path_dir_data_ds}/sigma_2_DBSCAN_LoRes_level_12_2010-12-31.nc'
    try:
        ds = xr.open_dataset(ncfile)
        print(f"Загружен netCDF файл для подложки: {ncfile}")
    except Exception as e:
        print(f"Ошибка при загрузке netCDF: {e}")
        ds = None
        
        # Сопоставляем треки
        print("\n" + "="*60)
        print("СОПОСТАВЛЕНИЕ ТРЕКОВ")
        print("="*60)
    
    matches = find_matching_tracks(ec_tracks, polina_tracks, 
                                  max_distance_km=50,  # Максимальное расстояние в км
                                  max_time_diff_hours=6)  # Максимальная разница во времени
    
    print(f"\nНайдено совпадений: {len(matches)}")
    
    # Создаем статистику
    stats = create_matching_statistics(ec_tracks, polina_tracks, matches)
    
    # Выводим основную статистику
    print("\n" + "="*60)
    print("СТАТИСТИКА СОПОСТАВЛЕНИЯ")
    print("="*60)
    print(f"Всего EC треков: {stats['total_ec_tracks']}")
    print(f"Всего Polina треков: {stats['total_polina_tracks']}")
    print(f"Найдено совпадений: {stats['matched_pairs']}")
    print(f"Покрытие EC треков: {stats['ec_coverage']:.1f}%")
    print(f"Покрытие Polina треков: {stats['polina_coverage']:.1f}%")
    
    if matches:
        print(f"\nСреднее расстояние между треками: {stats['avg_distance_km']:.1f} ± {stats['std_distance_km']:.1f} км")
        print(f"Средняя оценка качества: {stats['avg_match_score']:.3f}")
        print(f"Среднее смещение начала: {stats['avg_start_diff_hours']:.1f} ч")
        print(f"Среднее смещение конца: {stats['avg_end_diff_hours']:.1f} ч")
        
        print("\nКатегории качества:")
        for cat, count in stats['quality_categories'].items():
            print(f"  {cat}: {count} ({count/len(matches)*100:.1f}%)")
    
    # Создаем выходные папки
    output_dir = f'{path_init}/data/TC_tracks/EC_inner_matching_results'
    plots_dir = os.path.join(output_dir, 'plots')
    
    # Сохраняем результаты
    save_matching_results(ec_tracks, polina_tracks, matches, stats, output_dir)
    
    # Создаем визуализации
    print("\n" + "="*60)
    print("СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
    print("="*60)
    
    plot_statistics(ec_tracks, polina_tracks, matches, stats, plots_dir)
    
    if matches:
        plot_matched_tracks(ec_tracks, polina_tracks, matches, plots_dir, ds=ds)

    # После создания обычных визуализаций, создаем детальные для худших совпадений
    if matches:
        print("\n" + "="*60)
        print("СОЗДАНИЕ ДЕТАЛЬНЫХ ГРАФИКОВ ДЛЯ ХУДШИХ СОВПАДЕНИЙ")
        print("="*60)
        
        worst_dir = os.path.join(output_dir, 'worst_matches_detailed')
        plot_worst_matches_detailed(ec_tracks, polina_tracks, matches, 
                                    output_dir=worst_dir, ds=ds,
                                    data_type='LoRes', sigma=2, n_worst=10)
    
    print(f"\nГотово! Все результаты сохранены в папке: {output_dir}")
    print(f"  - Статистика: {output_dir}/matching_statistics.csv")
    print(f"  - Пары совпадений: {output_dir}/matching_pairs.csv")
    print(f"  - Визуализации: {plots_dir}/")

if __name__ == "__main__":
    main()