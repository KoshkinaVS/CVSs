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

def find_matching_tracks(polina_tracks, oboltus_tracks, max_distance_km=500, grid_step_km=77):
    """
    Находит ВСЕ совпадающие треки по пересекающимся временным интервалам
    Теперь сохраняет все совпадения, а не только лучшее
    """
    all_matches = []
    
    for polina_id, polina_track in enumerate(tqdm(polina_tracks, desc="Сопоставление треков (Polina -> Oboltus)")):
        polina_times = set(polina_track['datetime'])
        polina_start = polina_track['datetime'].min()
        polina_end = polina_track['datetime'].max()
        
        for oboltus_id, oboltus_track in enumerate(oboltus_tracks):
            oboltus_start = oboltus_track['datetime'].min()
            oboltus_end = oboltus_track['datetime'].max()
            
            # Проверяем пересечение временных интервалов
            if not ((polina_start <= oboltus_end) and (oboltus_start <= polina_end)):
                continue
            
            # Находим общие временные метки
            common_times = polina_times.intersection(set(oboltus_track['datetime']))
            
            if len(common_times) == 0:
                continue
            
            # Берем общие точки
            polina_subset = polina_track[polina_track['datetime'].isin(common_times)].sort_values('datetime')
            oboltus_subset = oboltus_track[oboltus_track['datetime'].isin(common_times)].sort_values('datetime')
            
            # Убеждаемся, что у нас одинаковое количество точек
            min_len = min(len(polina_subset), len(oboltus_subset))
            
            if min_len == 0:
                continue
            
            # Вычисляем среднее расстояние
            total_distance = 0
            for i in range(min_len):
                x1 = float(polina_subset['x'].iloc[i])
                y1 = float(polina_subset['y'].iloc[i])
                x2 = float(oboltus_subset['x'].iloc[i])
                y2 = float(oboltus_subset['y'].iloc[i])
                
                dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) * grid_step_km
                total_distance += dist
            
            avg_distance = total_distance / min_len
            
            if avg_distance > max_distance_km:
                continue
            
            # Сохраняем ВСЕ совпадения, удовлетворяющие условиям
            all_matches.append({
                'polina_track_id': polina_id,
                'oboltus_track_id': oboltus_id,
                'avg_distance_km': avg_distance,
                'common_points': min_len,
                'polina_start': polina_start,
                'polina_end': polina_end,
                'oboltus_start': oboltus_start,
                'oboltus_end': oboltus_end,
                'polina_points': len(polina_track),
                'oboltus_points': len(oboltus_track)
            })
    
    print(f"\nНайдено всего совпадений: {len(all_matches)}")
    
    # Группируем по Polina для информации
    polina_match_counts = {}
    for m in all_matches:
        polina_id = m['polina_track_id']
        if polina_id not in polina_match_counts:
            polina_match_counts[polina_id] = 0
        polina_match_counts[polina_id] += 1
    
    polinas_with_multiple = sum(1 for count in polina_match_counts.values() if count > 1)
    print(f"Polina треков с несколькими совпадениями: {polinas_with_multiple}")
    
    return all_matches

    
def plot_matched_tracks(oboltus_tracks, polina_tracks, matches, output_dir='matching_plots', ds=None):
    """
    Создает карты для каждого трека Polina со всеми его совпадениями в Oboltus
    Polina - черная линия, Oboltus - жирные цветные линии
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if len(matches) == 0:
        print("Нет совпадений для визуализации")
        return
    
    # Группируем совпадения по Polina трекам
    polina_matches = {}
    for match in matches:
        polina_id = match['polina_track_id']
        if polina_id not in polina_matches:
            polina_matches[polina_id] = []
        polina_matches[polina_id].append(match)
    
    print(f"\nСоздание графиков для {len(polina_matches)} Polina треков...")
    
    # Создаем график для каждого Polina трека
    for polina_idx, (polina_id, polina_match_list) in enumerate(polina_matches.items()):
        try:
            # Создаем фигуру
            fig = plt.figure(figsize=(5, 5), dpi=150)
            ax = fig.add_subplot(111)
            
            # Добавляем подложку из ds если доступна
            if ds is not None and 'HGT' in ds.variables:
                land_mask = xr.where(ds['HGT'] > 15, 1, np.nan)
                ax.contourf(land_mask, cmap='Greys', alpha=0.7, levels=[0.5, 1.5])
            
            # Получаем Polina трек
            polina_track = polina_tracks[polina_id]
            
            # Рисуем Polina трек черным цветом
            ax.plot(polina_track['x'], polina_track['y'], 
                   c='k', linewidth=1.5, label='Polina Track', zorder=5)
            
            # Получаем время начала Polina трека для заголовка и расчетов
            if 'datetime' in polina_track.columns:
                polina_start = pd.to_datetime(polina_track['datetime']).min()
                start_str = polina_start.strftime('%Y-%m-%dT%H')
            else:
                polina_start = None
                start_str = "unknown"
            
            # Отмечаем начало и конец Polina трека
            if len(polina_track) > 0:
                ax.scatter(polina_track['x'].values[0], polina_track['y'].values[0], 
                          c='g', s=7, zorder=10, label='Polina start')
                ax.scatter(polina_track['x'].values[-1], polina_track['y'].values[-1], 
                          c='r', s=7, zorder=10, label='Polina stop')
            
            # Рисуем все совпавшие Oboltus треки разными цветами
            # Используем яркую цветовую схему для хорошей видимости
            colors = plt.cm.tab10(np.linspace(0, 1, len(polina_match_list)))
            
            for match_idx, (match, color) in enumerate(zip(polina_match_list, colors)):
                oboltus_id = match['oboltus_track_id']
                oboltus_track = oboltus_tracks[oboltus_id]
                
                # Получаем время начала Oboltus трека
                if 'datetime' in oboltus_track.columns:
                    oboltus_start = pd.to_datetime(oboltus_track['datetime']).min()
                    oboltus_start_str = oboltus_start.strftime('%Y-%m-%dT%H')
                else:
                    oboltus_start = None
                    oboltus_start_str = "unknown"
                
                # Вычисляем разницу во времени
                if oboltus_start is not None and polina_start is not None:
                    time_diff_hours = abs((oboltus_start - polina_start).total_seconds() / 3600)
                    time_diff_str = f"{time_diff_hours:.1f}h"
                else:
                    time_diff_str = "?"
                
                dist = match['avg_distance_km']
                
                # Создаем подпись для легенды
                label = f'Oboltus #{match_idx+1} ({int(dist)} km, {time_diff_str})'
                
                # Рисуем Oboltus трек жирной цветной линией
                ax.plot(oboltus_track['x'], oboltus_track['y'], 
                       color=color, linewidth=3, label=label, zorder=4)
                
                # Отмечаем начало и конец Oboltus трека
                if len(oboltus_track) > 0:
                    ax.scatter(oboltus_track['x'].values[0], oboltus_track['y'].values[0], 
                              marker='*', color=color, s=7, zorder=10)
                    ax.scatter(oboltus_track['x'].values[-1], oboltus_track['y'].values[-1], 
                              marker='*', color=color, s=7, zorder=10)
            
            # Настраиваем заголовок
            ax.set_title(f'Polina Track {polina_id+1} at {start_str}')
            
            # Устанавливаем границы как в исходной функции
            ax.set_xlim(0, 110)
            ax.set_ylim(0, 110)
            
            # Добавляем легенду
            ax.legend(fontsize=7, loc='upper right')
            
            # Сохраняем
            filename = f'{output_dir}/Polina_{polina_id+1:09d}_all_matches.png'
            fig.savefig(filename, dpi=200, bbox_inches="tight", transparent=False)
            plt.close(fig)
            
            if (polina_idx + 1) % 10 == 0:  # Прогресс каждые 10 графиков
                print(f"  Создано {polina_idx + 1}/{len(polina_matches)} графиков")
            
        except Exception as e:
            print(f"Ошибка при создании графика для Polina {polina_id}: {e}")
            plt.close('all')
            continue
    
    print(f"Готово! Создано {len(polina_matches)} графиков (по одному на каждый Polina трек)")
    
def plot_worst_matches_detailed(oboltus_tracks, polina_tracks, matches, output_dir='detailed_plots', 
                               ds=None, data_type='LoRes', sigma=2, n_worst=10):
    """
    Создает детальные графики для самых худших совпадений из ВСЕХ найденных
    """
    if len(matches) == 0:
        print("Нет совпадений для детального анализа")
        return
    
    # Сортируем по расстоянию (от худших к лучшим)
    sorted_matches = sorted(matches, key=lambda x: x['avg_distance_km'], reverse=True)
    
    print(f"\nСоздание детальных графиков для {min(n_worst, len(sorted_matches))} ХУДШИХ совпадений из {len(matches)}:")
    print("="*60)
    
    for i, match in enumerate(sorted_matches[:n_worst]):
        polina_track = polina_tracks[match['polina_track_id']]
        oboltus_track = oboltus_tracks[match['oboltus_track_id']]
        
        print(f"\n{i+1}. Polina Track {match['polina_track_id']} - Oboltus Track {match['oboltus_track_id']}")
        print(f"   Среднее расстояние: {match['avg_distance_km']:.1f} км")
        print(f"   Общих точек: {match['common_points']}")
        
        plot_detailed_track_comparison(
            polina_track, oboltus_track, match,
            output_dir=output_dir, ds=ds,
            track_num=i+1, data_type=data_type, sigma=sigma,
            is_worst=True
        )
        
def plot_detailed_track_comparison(polina_track, oboltus_track, match_info, 
                                   output_dir='detailed_plots', ds=None, 
                                   track_num=None, data_type='LoRes', sigma=2,
                                   is_worst=False):
    """
    Создает детальный график сравнения для одной пары со статистикой
    Polina - датасет 1, Oboltus - датасет 2
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
            
        if 'datetime' in oboltus_track.columns:
            oboltus_start = pd.to_datetime(oboltus_track['datetime']).min()
            oboltus_end = pd.to_datetime(oboltus_track['datetime']).max()
            oboltus_start_str = oboltus_start.strftime('%Y-%m-%dT%H')
            oboltus_end_str = oboltus_end.strftime('%Y-%m-%dT%H')
            oboltus_duration = (oboltus_end - oboltus_start).total_seconds() / 3600
        else:
            oboltus_start_str = "unknown"
            oboltus_end_str = "unknown"
            oboltus_duration = 0
            oboltus_start = None
        
        # Вычисляем разницу во времени
        if oboltus_start is not None and polina_start is not None:
            time_diff_start = (oboltus_start - polina_start).total_seconds() / 3600
            time_diff_end = (oboltus_end - polina_end).total_seconds() / 3600
            
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
        ax1.plot(polina_track['x'], polina_track['y'], 'b-', linewidth=2.5, 
                label=f'Polina Track ({polina_start_str} - {polina_end_str}, {polina_duration:.1f}h)', zorder=5)
        ax1.plot(oboltus_track['x'], oboltus_track['y'], 'r--', linewidth=2, 
                label=f'Oboltus Track ({oboltus_start_str} - {oboltus_end_str}, {oboltus_duration:.1f}h)', zorder=4)
        
        # Отмечаем точки треков
        ax1.scatter(polina_track['x'], polina_track['y'], c='blue', s=20, alpha=0.5, zorder=3)
        ax1.scatter(oboltus_track['x'], oboltus_track['y'], c='red', s=20, alpha=0.5, zorder=3)
        
        # Отмечаем начало и конец Polina трека
        if len(polina_track) > 0:
            ax1.scatter(polina_track['x'].values[0], polina_track['y'].values[0], 
                       c='lime', s=100, marker='o', edgecolor='black', 
                       linewidth=1.5, zorder=10, label='Polina Start')
            ax1.scatter(polina_track['x'].values[-1], polina_track['y'].values[-1], 
                       c='orange', s=100, marker='s', edgecolor='black', 
                       linewidth=1.5, zorder=10, label='Polina End')
        
        # Добавляем информацию о совпадении на карту
        info_box = (f'Среднее расстояние: {match_info["avg_distance_km"]:.1f} км\n'
                   f'Общих точек: {match_info["common_points"]}\n'
                   # f'Оценка качества: {match_info["match_score"]:.3f}\n'
                   f'Разница начала: {time_diff_start_str}\n'
                   f'Разница конца: {time_diff_end_str}')
        
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
            ax1.set_title(f'{title_prefix}{track_num}: Polina {match_info["polina_track_id"]} vs Oboltus {match_info["oboltus_track_id"]}')
        else:
            ax1.set_title(f'Polina {match_info["polina_track_id"]} vs Oboltus {match_info["oboltus_track_id"]}')
        
        ax1.legend(fontsize=8, loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # ===== ГРАФИК 2: Расстояние между треками по времени =====
        ax2 = plt.subplot(gs[1, 0])
        
        # Находим общие моменты времени
        if 'datetime' in polina_track.columns and 'datetime' in oboltus_track.columns:
            polina_times = set(polina_track['datetime'])
            oboltus_times = set(oboltus_track['datetime'])
            common_times = sorted(list(polina_times.intersection(oboltus_times)))
            
            if common_times:
                # Вычисляем расстояния для каждого общего момента времени
                distances = []
                time_labels = []
                
                for t in common_times:
                    polina_point = polina_track[polina_track['datetime'] == t].iloc[0]
                    oboltus_point = oboltus_track[oboltus_track['datetime'] == t].iloc[0]
                    
                    dist = np.sqrt((oboltus_point['x'] - polina_point['x'])**2 + 
                                  (oboltus_point['y'] - polina_point['y'])**2) * 6  # км
                    distances.append(dist)
                    time_labels.append(pd.to_datetime(t).strftime('%m-%d %H:%M'))
                
                # Рисуем график расстояний
                x_pos = range(len(time_labels))
                bars = ax2.bar(x_pos, distances, color='steelblue', alpha=0.7, width=0.6)
                
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
                
                for j, (x, d) in enumerate(zip(x_pos, distances)):
                    ax2.text(x, d + 2, f'{d:.0f}', ha='center', va='bottom', fontsize=7)
        
        # ===== ГРАФИК 3: Метрики качества =====
        ax3 = plt.subplot(gs[1, 1])
        
        # Вычисляем метрики
        if 'datetime' in polina_track.columns and 'datetime' in oboltus_track.columns:
            polina_times = set(polina_track['datetime'])
            oboltus_times = set(oboltus_track['datetime'])
            common_count = len(polina_times.intersection(oboltus_times))
            time_overlap = common_count / max(len(polina_times), len(oboltus_times))
        else:
            time_overlap = match_info['common_points'] / max(match_info['polina_points'], match_info['oboltus_points'])
        
        distance_score = 1 / (1 + match_info['avg_distance_km'] / 20)
        
        if polina_start is not None and oboltus_start is not None:
            start_diff = abs((oboltus_start - polina_start).total_seconds() / 3600)
            end_diff = abs((pd.to_datetime(polina_track['datetime'].max()) - 
                          pd.to_datetime(oboltus_track['datetime'].max())).total_seconds() / 3600)
            temporal_score = 1 / (1 + (start_diff + end_diff) / 12)
        else:
            temporal_score = 0.5
        
        if len(polina_track) == len(oboltus_track) and len(polina_track) > 1:
            polina_dx = np.diff(polina_track['x'].values)
            polina_dy = np.diff(polina_track['y'].values)
            oboltus_dx = np.diff(oboltus_track['x'].values)
            oboltus_dy = np.diff(oboltus_track['y'].values)
            
            polina_norm = np.sqrt(polina_dx**2 + polina_dy**2)
            oboltus_norm = np.sqrt(oboltus_dx**2 + oboltus_dy**2)
            
            polina_norm = np.where(polina_norm == 0, 1, polina_norm)
            oboltus_norm = np.where(oboltus_norm == 0, 1, oboltus_norm)
            
            polina_dx_norm = polina_dx / polina_norm
            polina_dy_norm = polina_dy / polina_norm
            oboltus_dx_norm = oboltus_dx / oboltus_norm
            oboltus_dy_norm = oboltus_dy / oboltus_norm
            
            dot_product = polina_dx_norm * oboltus_dx_norm + polina_dy_norm * oboltus_dy_norm
            path_similarity = np.mean(np.clip(dot_product, -1, 1))
            path_score = (path_similarity + 1) / 2
        else:
            path_score = 0.5
        
        metrics = ['Time Overlap', 'Distance', 'Temporal', 'Path']
        scores = [time_overlap, distance_score, temporal_score, path_score]
        
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
        
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{score:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax3.legend(fontsize=8, loc='lower right')
        ax3.set_xticklabels(metrics, fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
        
        if track_num is not None:
            if is_worst:
                plt.suptitle(f'ХУДШЕЕ СОВПАДЕНИЕ #{track_num} (Polina {match_info["polina_track_id"]} - Oboltus {match_info["oboltus_track_id"]})', 
                            fontsize=14, y=1.02, color='red', weight='bold')
            else:
                plt.suptitle(f'Детальный анализ #{track_num}: Polina {match_info["polina_track_id"]} - Oboltus {match_info["oboltus_track_id"]}', 
                            fontsize=14, y=1.02)
        else:
            plt.suptitle(f'Polina {match_info["polina_track_id"]} - Oboltus {match_info["oboltus_track_id"]}', 
                        fontsize=14, y=1.02)
        
        if track_num is not None:
            if is_worst:
                filename = f'{output_dir}/{data_type}_WORST_{track_num:02d}_sigma_{sigma}.png'
            else:
                filename = f'{output_dir}/{data_type}_detailed_{track_num:03d}_sigma_{sigma}.png'
        else:
            filename = f'{output_dir}/{data_type}_detailed_Polina{match_info["polina_track_id"]:03d}_sigma_{sigma}.png'
        
        plt.tight_layout()
        fig.savefig(filename, dpi=200, bbox_inches="tight", transparent=False)
        plt.close(fig)
        
        print(f"   Детальный график сохранен: {filename}")
        
    except Exception as e:
        print(f"Ошибка при создании детального графика: {e}")
        plt.close('all')
        
def create_matching_statistics(polina_tracks, oboltus_tracks, matches):
    """
    Создает статистику сопоставления треков для ВСЕХ совпадений
    """
    # Уникальные треки, участвующие в совпадениях
    unique_polina = set([m['polina_track_id'] for m in matches])
    unique_oboltus = set([m['oboltus_track_id'] for m in matches])
    
    stats = {
        'total_polina_tracks': len(polina_tracks),
        'total_oboltus_tracks': len(oboltus_tracks),
        'total_pairs': len(matches),
        'matched_pairs': len(matches),  # для совместимости
        'unique_polina_matched': len(unique_polina),
        'unique_oboltus_matched': len(unique_oboltus),
        'polina_coverage': len(unique_polina) / len(polina_tracks) * 100 if polina_tracks else 0,
        'oboltus_coverage': len(unique_oboltus) / len(oboltus_tracks) * 100 if oboltus_tracks else 0
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
        
        # Статистика по количеству совпадений на Polina трек
        polina_match_counts = {}
        for m in matches:
            polina_id = m['polina_track_id']
            polina_match_counts[polina_id] = polina_match_counts.get(polina_id, 0) + 1
        
        stats['avg_matches_per_polina'] = np.mean(list(polina_match_counts.values()))
        stats['max_matches_per_polina'] = max(polina_match_counts.values())
        
        # Анализ начала и конца треков (СО ЗНАКОМ)
        start_shifts = []
        end_shifts = []
        start_diffs_abs = []
        end_diffs_abs = []
        
        for m in matches:
            # Смещение со знаком
            start_shift = (m['oboltus_start'] - m['polina_start']).total_seconds() / 3600
            end_shift = (m['oboltus_end'] - m['polina_end']).total_seconds() / 3600
            start_shifts.append(start_shift)
            end_shifts.append(end_shift)
            
            # Абсолютные значения для статистики
            start_diffs_abs.append(abs(start_shift))
            end_diffs_abs.append(abs(end_shift))
        
        stats['avg_start_shift_hours'] = np.mean(start_shifts)  # среднее со знаком
        stats['avg_end_shift_hours'] = np.mean(end_shifts)      # среднее со знаком
        stats['avg_start_diff_hours'] = np.mean(start_diffs_abs)  # среднее абсолютное
        stats['avg_end_diff_hours'] = np.mean(end_diffs_abs)      # среднее абсолютное
        
        # Категории качества
        excellent = sum(1 for d in distances if d < 50)
        good = sum(1 for d in distances if 50 <= d < 100)
        fair = sum(1 for d in distances if 100 <= d < 200)
        poor = sum(1 for d in distances if d >= 200)
        
        stats['quality_categories'] = {
            'excellent (<50 km)': excellent,
            'good (50-100 km)': good,
            'fair (100-200 km)': fair,
            'poor (>=200 km)': poor
        }
    else:
        # Значения по умолчанию, если нет совпадений
        stats['avg_distance_km'] = 0
        stats['std_distance_km'] = 0
        stats['min_distance_km'] = 0
        stats['max_distance_km'] = 0
        stats['avg_common_points'] = 0
        stats['total_common_points'] = 0
        stats['avg_matches_per_polina'] = 0
        stats['max_matches_per_polina'] = 0
        stats['avg_start_shift_hours'] = 0
        stats['avg_end_shift_hours'] = 0
        stats['avg_start_diff_hours'] = 0
        stats['avg_end_diff_hours'] = 0
        stats['quality_categories'] = {}
    
    return stats

    
def plot_statistics(polina_tracks, oboltus_tracks, matches, stats, output_dir='matching_plots'):
    """
    Создает визуализацию статистики
    Polina - датасет 1, Oboltus - датасет 2
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if matches:
        start_shifts = []
        end_shifts = []
        polina_lengths = []
        oboltus_lengths = []
        length_ratios = []  # Отношение Polina/Oboltus
        
        for m in matches:
            start_shift = (m['oboltus_start'] - m['polina_start']).total_seconds() / 3600
            start_shifts.append(start_shift)
            
            end_shift = (m['oboltus_end'] - m['polina_end']).total_seconds() / 3600
            end_shifts.append(end_shift)
            
            polina_len = m['polina_points']
            oboltus_len = m['oboltus_points']
            polina_lengths.append(polina_len)
            oboltus_lengths.append(oboltus_len)
            
            if oboltus_len > 0:
                length_ratios.append(polina_len / oboltus_len)
            else:
                length_ratios.append(np.nan)
    
    fig = plt.figure(figsize=(20, 18))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # ===== РЯД 1: Общая статистика и доли =====
    
    # 1. Доля треков Polina
    ax1 = fig.add_subplot(gs[0, 0])
    polina_matched = len(matches)
    polina_unmatched = stats['total_polina_tracks'] - polina_matched
    
    ax1.pie([polina_matched, polina_unmatched], 
            labels=[f'Polina совпало\n{polina_matched}', f'Polina не совпало\n{polina_unmatched}'],
            colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%', startangle=90)
    ax1.set_title(f'Polina треки (всего: {stats["total_polina_tracks"]})', fontsize=12)
    
    # 2. Доля треков Oboltus
    ax2 = fig.add_subplot(gs[0, 1])
    oboltus_matched = len(set([m['oboltus_track_id'] for m in matches]))
    oboltus_unmatched = stats['total_oboltus_tracks'] - oboltus_matched
    
    ax2.pie([oboltus_matched, oboltus_unmatched], 
            labels=[f'Oboltus совпало\n{oboltus_matched}', f'Oboltus не совпало\n{oboltus_unmatched}'],
            colors=['skyblue', 'pink'], autopct='%1.1f%%', startangle=90)
    ax2.set_title(f'Oboltus треки (всего: {stats["total_oboltus_tracks"]})', fontsize=12)
    
    # 3. Общая статистика текстом
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    info_text = (
        f"ОБЩАЯ СТАТИСТИКА\n"
        f"{'='*25}\n\n"
        f"Polina треков: {stats['total_polina_tracks']}\n"
        f"Oboltus треков: {stats['total_oboltus_tracks']}\n"
        f"Найдено совпадений: {stats['matched_pairs']}\n\n"
        f"Покрытие Polina:\n"
        f"  {stats['polina_coverage']:.1f}% Polina треков имеют совпадение\n"
        f"Покрытие Oboltus:\n"
        f"  {stats['oboltus_coverage']:.1f}% Oboltus треков имеют совпадение\n\n"
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
        
        ax4.text(0.02, 0.75, f'РАНЬШЕ: {sum(s < 0 for s in start_shifts)} треков\n(Oboltus начался до Polina)', 
                transform=ax4.transAxes, color='darkred', fontsize=9)
        ax4.text(0.55, 0.75, f'ПОЗЖЕ: {sum(s > 0 for s in start_shifts)} треков\n(Oboltus начался после Polina)', 
                transform=ax4.transAxes, color='darkgreen', fontsize=9)
        
        ax4.set_xlabel('Смещение начала трека Oboltus относительно Polina (часы)')
        ax4.set_ylabel('Количество пар')
        ax4.set_title('СМЕЩЕНИЕ НАЧАЛА ТРЕКОВ\n(-) Oboltus раньше Polina | (+) Oboltus позже Polina')
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
        
        ax5.text(0.02, 0.75, f'РАНЬШЕ: {sum(s < 0 for s in end_shifts)} треков\n(Oboltus закончился до Polina)', 
                transform=ax5.transAxes, color='darkred', fontsize=9)
        ax5.text(0.55, 0.75, f'ПОЗЖЕ: {sum(s > 0 for s in end_shifts)} треков\n(Oboltus закончился после Polina)', 
                transform=ax5.transAxes, color='darkgreen', fontsize=9)
        
        ax5.set_xlabel('Смещение конца трека Oboltus относительно Polina (часы)')
        ax5.set_ylabel('Количество пар')
        ax5.set_title('СМЕЩЕНИЕ КОНЦА ТРЕКОВ\n(-) Oboltus раньше Polina | (+) Oboltus позже Polina')
        ax5.legend(loc='upper right')
        ax5.grid(True, alpha=0.3)
        
    # 6. Гистограмма отношения длин Polina/Oboltus
    ax6 = fig.add_subplot(gs[1, 2])
    if matches and length_ratios:
        clean_ratios = [r for r in length_ratios if not np.isnan(r)]
        
        if clean_ratios:
            bins = np.linspace(0, 3, 16)
            n, bins, patches = ax6.hist(clean_ratios, bins=bins, edgecolor='black', alpha=0.7)
            
            for i, patch in enumerate(patches):
                ratio_center = (bins[i] + bins[i+1]) / 2
                if ratio_center < 0.8:
                    patch.set_facecolor('lightcoral')
                elif ratio_center > 1.2:
                    patch.set_facecolor('lightgreen')
                else:
                    patch.set_facecolor('lightblue')
            
            ax6.axvline(x=1.0, color='black', linestyle='-', linewidth=2, label='Равные длины')
            ax6.axvline(x=np.mean(clean_ratios), color='red', linestyle='--', 
                       label=f'Среднее: {np.mean(clean_ratios):.2f}')
            
            shorter = sum(r < 0.8 for r in clean_ratios)
            equal = sum(0.8 <= r <= 1.2 for r in clean_ratios)
            longer = sum(r > 1.2 for r in clean_ratios)
            
            ax6.text(0.02, 0.75, f'Polina КОРОЧЕ: {shorter} треков\n(ratio < 0.8)', 
                    transform=ax6.transAxes, color='darkred', fontsize=9)
            ax6.text(0.35, 0.75, f'ПРИМЕРНО РАВНЫ: {equal} треков\n(0.8-1.2)', 
                    transform=ax6.transAxes, color='darkblue', fontsize=9)
            ax6.text(0.70, 0.75, f'Polina ДЛИННЕЕ: {longer} треков\n(ratio > 1.2)', 
                    transform=ax6.transAxes, color='darkgreen', fontsize=9)
            
            ax6.set_xlabel('Отношение длин Polina / Oboltus')
            ax6.set_ylabel('Количество пар')
            ax6.set_title('ОТНОШЕНИЕ ДЛИН ТРЕКОВ\n<1 Polina короче | >1 Polina длиннее')
            ax6.legend(loc='upper right')
            ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'ДЕТАЛЬНАЯ СТАТИСТИКА СОПОСТАВЛЕНИЯ (всего пар: {len(matches)})', 
                fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'matching_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # if matches:
    #     shifts_df = pd.DataFrame({
    #         'polina_track_id': [m['polina_track_id'] for m in matches],
    #         'oboltus_track_id': [m['oboltus_track_id'] for m in matches],
    #         'polina_start': [m['polina_start'] for m in matches],
    #         'oboltus_start': [m['oboltus_start'] for m in matches],
    #         'start_shift_hours': start_shifts,
    #         'polina_end': [m['polina_end'] for m in matches],
    #         'oboltus_end': [m['oboltus_end'] for m in matches],
    #         'end_shift_hours': end_shifts,
    #         'polina_length': polina_lengths,
    #         'oboltus_length': oboltus_lengths,
    #         'length_ratio': length_ratios,
    #         'avg_distance_km': [m['avg_distance_km'] for m in matches],
    #         'match_score': [m['match_score'] for m in matches]
    #     })
    #     shifts_df.to_csv(os.path.join(output_dir, 'temporal_shifts.csv'), index=False)
    #     print(f"  - Данные о временных смещениях и длинах: {output_dir}/temporal_shifts.csv")

        
def save_matching_results(polina_tracks, oboltus_tracks, matches, stats, output_dir='matching_results'):
    """
    Сохраняет результаты сопоставления в CSV файлы
    Теперь включает информацию о всех совпадениях
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Сохраняем информацию о всех совпадениях
    matches_df = pd.DataFrame(matches)
    matches_df.to_csv(os.path.join(output_dir, 'all_matching_pairs.csv'), index=False)
    
    # Создаем сводку по каждому Polina треку (со списком всех совпадений)
    polina_summary = []
    for polina_id, polina_track in enumerate(polina_tracks):
        # Находим все совпадения для этого Polina трека
        polina_matches = [m for m in matches if m['polina_track_id'] == polina_id]
        
        # Сортируем по расстоянию
        polina_matches.sort(key=lambda x: x['avg_distance_km'])
        
        # Создаем строки для каждого совпадения
        for rank, match in enumerate(polina_matches):
            summary = {
                'polina_track_id': polina_id,
                'polina_start': polina_track['datetime'].min(),
                'polina_end': polina_track['datetime'].max(),
                'polina_points': len(polina_track),
                'match_rank': rank + 1,
                'total_matches_for_polina': len(polina_matches),
                'oboltus_track_id': match['oboltus_track_id'],
                'avg_distance_km': match['avg_distance_km'],
                'common_points': match['common_points'],
                'oboltus_start': match['oboltus_start'],
                'oboltus_end': match['oboltus_end']
            }
            polina_summary.append(summary)
    
    polina_df = pd.DataFrame(polina_summary)
    polina_df.to_csv(os.path.join(output_dir, 'polina_tracks_all_matches.csv'), index=False)
    
    # Сохраняем статистику
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(os.path.join(output_dir, 'matching_statistics.csv'), index=False)
    
    print(f"\nРезультаты сохранены в папку: {output_dir}")
    print(f"  - Всего пар: {len(matches)}")
    print(f"  - Уникальных Polina треков с совпадениями: {len(set([m['polina_track_id'] for m in matches]))}")
    print(f"  - Уникальных Oboltus треков с совпадениями: {len(set([m['oboltus_track_id'] for m in matches]))}")
    
def main():
    """
    Основная функция для сопоставления треков из двух папок
    Polina - датасет 1, ищем совпадения в Oboltus (EddyClicker_tracks)
    """
    print("="*60)
    print("СОПОСТАВЛЕНИЕ ТРЕКОВ: Polina (датасет 1) vs Oboltus (датасет 2)")
    print("="*60)
    
    # Пути к папкам
    path_init = '/storage/thalassa/users/vkoshkina'
    polina_folder = f'{path_init}/data/LoRes/LoRes/EddyClicker_Polina'  # Polina - датасет 1
    oboltus_folder = f'{path_init}/data/LoRes/LoRes/EddyClicker_tracks'  # Oboltus - датасет 2
    
    # Проверяем существование папок
    if not os.path.exists(polina_folder):
        print(f"ОШИБКА: Папка Polina не найдена: {polina_folder}")
        polina_folder = input("Введите путь к папке EddyClicker_Polina: ")
    
    if not os.path.exists(oboltus_folder):
        print(f"ОШИБКА: Папка Oboltus не найдена: {oboltus_folder}")
        oboltus_folder = input("Введите путь к папке EddyClicker_tracks: ")
    
    print(f"\nПапка Polina (датасет 1): {polina_folder}")
    print(f"Папка Oboltus (датасет 2): {oboltus_folder}")
    
    # Загружаем треки
    polina_tracks = load_tracks_from_folder(polina_folder, "Polina")  # Датасет 1
    oboltus_tracks = load_tracks_from_folder(oboltus_folder, "Oboltus")  # Датасет 2
    
    if not polina_tracks or not oboltus_tracks:
        print("Ошибка: Не удалось загрузить треки из одной из папок")
        return

    # Загружаем ds для подложки
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
    
    matches = find_matching_tracks(polina_tracks, oboltus_tracks, 
                                  max_distance_km=500, grid_step_km=77) 
    
    print(f"\nНайдено совпадений: {len(matches)}")
    
    # Создаем статистику (обновим названия в функции create_matching_statistics)
    stats = create_matching_statistics(polina_tracks, oboltus_tracks, matches)
    
    # Выводим основную статистику
    print("\n" + "="*60)
    print("СТАТИСТИКА СОПОСТАВЛЕНИЯ")
    print("="*60)
    print(f"Всего Polina треков: {stats['total_polina_tracks']}")
    print(f"Всего Oboltus треков: {stats['total_oboltus_tracks']}")
    print(f"Найдено совпадений: {stats['total_pairs']}")
    print(f"Покрытие Polina треков: {stats['polina_coverage']:.1f}%")
    print(f"Покрытие Oboltus треков: {stats['oboltus_coverage']:.1f}%")
    
    if matches:
        print(f"\nСреднее расстояние между треками: {stats['avg_distance_km']:.1f} ± {stats['std_distance_km']:.1f} км")
        print(f"Среднее смещение начала: {stats['avg_start_diff_hours']:.1f} ч")
        print(f"Среднее смещение конца: {stats['avg_end_diff_hours']:.1f} ч")
        
        print("\nКатегории качества:")
        for cat, count in stats['quality_categories'].items():
            print(f"  {cat}: {count} ({count/len(matches)*100:.1f}%)")
    
    # Создаем выходные папки
    output_dir = f'{path_init}/data/TC_tracks/Polina_Oboltus_matching_results_all'
    plots_dir = os.path.join(output_dir, 'plots')
    
    # Сохраняем результаты
    save_matching_results(polina_tracks, oboltus_tracks, matches, stats, output_dir)
    
    # Создаем визуализации
    print("\n" + "="*60)
    print("СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
    print("="*60)
    
    plot_statistics(polina_tracks, oboltus_tracks, matches, stats, plots_dir)
    
    if matches:
        plot_matched_tracks(oboltus_tracks, polina_tracks, matches, plots_dir, ds=ds)

    # Создаем детальные графики для худших совпадений
    if matches:
        print("\n" + "="*60)
        print("СОЗДАНИЕ ДЕТАЛЬНЫХ ГРАФИКОВ ДЛЯ ХУДШИХ СОВПАДЕНИЙ")
        print("="*60)
        
        worst_dir = os.path.join(output_dir, 'worst_matches_detailed')
        plot_worst_matches_detailed(oboltus_tracks, polina_tracks, matches, 
                                    output_dir=worst_dir, ds=ds,
                                    data_type='LoRes', sigma=2, n_worst=10)
    
    print(f"\nГотово! Все результаты сохранены в папке: {output_dir}")
    print(f"  - Статистика: {output_dir}/matching_statistics.csv")
    print(f"  - Пары совпадений: {output_dir}/matching_pairs.csv")
    print(f"  - Визуализации: {plots_dir}/")

if __name__ == "__main__":
    main()