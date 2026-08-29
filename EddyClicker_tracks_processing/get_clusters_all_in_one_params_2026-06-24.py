import pandas as pd
import xarray as xr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import os
import sys
warnings.filterwarnings('ignore')

path_init = f'/storage/thalassa/users/vkoshkina'
folder = 'scripts/CS_processing/CVS_clustering/'
sys.path.insert(2, f'{path_init}/{folder}')

from func_for_CVS_clusters import *
from func_for_metrics import *

# # Добавляем выбор сезона
# print('Выберите сезон (winter/summer/year/all): ')
# input_season = input().strip().lower()
input_season = 'year'

print('Выберите метод кластеризации:')
print('1. KMeans')
print('2. GMM')
print('(по умолчанию KMeans): ', end='')
choice1 = input().strip()
clustering_type = 'kmeans' if choice1 in ['1', ''] else 'gmm'

print('Выберите метод нормализации:')
print('1. StandardScaler')
print('2. RobustScaler')
print('(по умолчанию StandardScaler): ', end='')
choice2 = input().strip()
scaler_type = 'standard' if choice2 in ['1', ''] else 'robust'

# Проверяем валидность ввода
valid_seasons = ['winter', 'summer', 'year', 'all']
if input_season not in valid_seasons:
    raise ValueError(f"Неправильный сезон. Допустимые значения: {valid_seasons}")

path_data_dir = f'{path_init}/data'  

data_type = 'LoRes'  # или 'LoRes', в зависимости от того, что вы сейчас обрабатываете
if data_type == 'LoRes':
    path_data_dir = f'{path_data_dir}/{data_type}'


params_type = 'points_clustering'

# params_type = 'max_wspd_day'


# tracks_params_folder = 'EddyClicker_tracks_Egor_2010_params_r2d'

tracks_params_folder = 'EddyClicker_tracks_Egor_2010_15params_2028-08-10_r2d'

path_tracks_dir = f'{path_data_dir}/{data_type}/{tracks_params_folder}'

    
output_dir = f"{path_tracks_dir}/EddyClicker_clustering/{scaler_type}_scaler/EddyClicker_{clustering_type}_results_2026-08-10_no_pca/{params_type}"


# Входной файл - объединенный CSV со всеми точками
input_file = f'{path_tracks_dir}/EddyClicker_tracks_Egor_2010_params_all_in_one.csv'


os.makedirs(f"{output_dir}/cluster_tables", exist_ok=True)
os.makedirs(f"{output_dir}/cluster_pics", exist_ok=True)



if data_type == 'LoRes':
    path_dir_raw = f'/storage/NAAD/NAAD/LoRes/2010'
    ncfile = f'{path_dir_raw}/wrfout_d01_2010-01-01_00:00:00'

    # 2026-07-14 (ибо по старому адресу пустота)
    path_dir_raw = f'/storage/thalassa/users/vkoshkina/data/LoRes/LoRes/DBSCAN_02-04-10_with_wspd_smoothing'
    ncfile = f'{path_dir_raw}/sigma_2_DBSCAN_LoRes_level_12_2010-08.nc'
    
    ground_ds = xr.open_dataset(f'{ncfile}')['HGT'] #[0]
    ground = np.where(ground_ds > 5, 1, np.nan)
elif data_type == 'ERA5':
    path_dir_raw = f'/storage/thalassa/users/vkoshkina/data/ERA5'
    ncfile = f'{path_dir_raw}/ERA5_lsm_cropped.nc'
    
    ground_ds_ERA5 = xr.open_dataset(f'{ncfile}')['var172'][0]
    ground = np.where(ground_ds_ERA5 > 0.5, 1, np.nan)


track_to_pmc = {
    1102: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL032010'},  # перекрытие 30.0 ч, dist=94.1 км
    1157: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL122010'},  # перекрытие 72.0 ч, dist=98.2 км
    1168: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL092010'},  # перекрытие 93.0 ч, dist=190.0 км
    1172: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL122010'},  # перекрытие 24.0 ч, dist=153.0 км
    1178: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL142010'},  # перекрытие 135.0 ч, dist=117.3 км
    1183: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL152010'},  # перекрытие 27.0 ч, dist=91.2 км
    12: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL062010'},  # перекрытие 267.0 ч, dist=188.1 км
    1294: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL172010'},  # перекрытие 96.0 ч, dist=101.4 км
    14: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL072010'},  # перекрытие 291.0 ч, dist=104.3 км
    1497: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL192010'},  # перекрытие 84.0 ч, dist=119.4 км
    1623: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL212010'},  # перекрытие 114.0 ч, dist=189.5 км
    1624: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL212010'},  # перекрытие 126.0 ч, dist=137.4 км
    1631: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL202010'},  # перекрытие 18.0 ч, dist=178.1 км
    1727: {'база': 'ERA5 tracks (zenodo)', 'тип': 'ПМЦ', 'id_в_базе': '458'},  # перекрытие 18.0 ч, dist=151.2 км
    1729: {'база': 'Cyclone infos (zenodo)', 'тип': 'ПМЦ', 'id_в_базе': '457'},  # перекрытие 22.8 ч, dist=160.9 км
    1877: {'база': 'Climatology NH (Stoll et al.)', 'тип': 'ПМЦ', 'id_в_базе': '120101106120'},  # перекрытие 47.0 ч, dist=134.9 км
    2130: {'база': 'Rojo et al. 2019 (PANGAEA)', 'тип': 'ПМЦ', 'id_в_базе': '143.g'},  # перекрытие 24.1 ч, dist=120.0 км
    2314: {'база': 'ERA5 tracks (zenodo)', 'тип': 'ПМЦ', 'id_в_базе': '479'},  # перекрытие 39.0 ч, dist=162.7 км
    2358: {'база': 'Cyclone infos (zenodo)', 'тип': 'ПМЦ', 'id_в_базе': '481'},  # перекрытие 31.9 ч, dist=145.8 км
    276: {'база': 'Climatology NH (Stoll et al.)', 'тип': 'ПМЦ', 'id_в_базе': '120100112610'},  # перекрытие 22.0 ч, dist=139.3 км
    38: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL062010'},  # перекрытие 24.0 ч, dist=61.1 км
    383: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL082010'},  # перекрытие 81.0 ч, dist=139.3 км
    491: {'база': 'Climatology NH (Stoll et al.)', 'тип': 'ПМЦ', 'id_в_базе': '120100200570'},  # перекрытие 29.0 ч, dist=174.4 км
    500: {'база': 'Climatology NH (Stoll et al.)', 'тип': 'ПМЦ', 'id_в_базе': '120100200210'},  # перекрытие 25.0 ч, dist=180.1 км
    537: {'база': 'ERA5 tracks (zenodo)', 'тип': 'ПМЦ', 'id_в_базе': '419'},  # перекрытие 47.0 ч, dist=164.4 км
    568: {'база': 'Cyclone infos (zenodo)', 'тип': 'ПМЦ', 'id_в_базе': '424'},  # перекрытие 18.1 ч, dist=73.7 км
    644: {'база': 'Climatology NH (Stoll et al.)', 'тип': 'ПМЦ', 'id_в_базе': '120100213970'},  # перекрытие 42.0 ч, dist=165.9 км
    66: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL112010'},  # перекрытие 351.0 ч, dist=92.7 км
    710: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL042010'},  # перекрытие 39.0 ч, dist=189.4 км
    732: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL042010'},  # перекрытие 33.0 ч, dist=168.2 км
    757: {'база': 'NOAA HURDAT2', 'тип': 'ТЦ', 'id_в_базе': 'AL052010'},  # перекрытие 105.0 ч, dist=134.0 км
    77: {'база': 'Climatology NH (Stoll et al.)', 'тип': 'ПМЦ', 'id_в_базе': '120100903050'},  # перекрытие 21.0 ч, dist=177.0 км
}


param_cols = [
    'R2D_max', 
    'mean_radius', 
    'velocity',

    'SLP_diff_cent_95',
    'U10_mean', 
    # 'U500_mean', 
    'U850_mean', 
    'U500_U850_frac', 'U500_minus_U850',
    'PV_850_mean',
    # 'PV_500_mean',
    # 'T2_minus_T500_mean', 
    'T2_minus_T850_mean', 'T850_disp',
    # 'trop_height', 'pbl_height', 
    'pbl_trop_frac',
    'w_850', 
    # 'w_500', 
    'RH_850', 
    'RAIN_HOURLY_sum', 
    # 'RAIN_HOURLY_95', 
]


print(f'всего {len(param_cols)} параметров для кластеризации')


def plot_points_by_clusters(df_clustered, n_clusters, colors, ground, output_dir, season, 
                           data_type='LoRes', plot_TC=False, track_to_pmc=None):
    """Визуализирует точки по кластерам на карте"""
    
    # Определяем ID ТЦ/ПМЦ
    tc_ids = set()
    pmc_ids = set()
    if plot_TC and track_to_pmc:
        for tid, info in track_to_pmc.items():
            if info.get('тип') == 'ТЦ':
                tc_ids.add(tid)
            elif info.get('тип') == 'ПМЦ':
                pmc_ids.add(tid)
    
    cols = min(4, n_clusters)
    rows = (n_clusters + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()
    
    for cluster_idx in range(n_clusters):
        ax = axes[cluster_idx]
        
        # Точки этого кластера
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_idx]
        
        if len(cluster_data) == 0:
            ax.axis('off')
            continue
        
        # Отрисовка фона (суша)
        plot_ground(ax, ground)
        
        # Определяем цвет кластера
        color = colors(cluster_idx) if callable(colors) else colors[cluster_idx % len(colors)]
        
        # Обычные точки
        ax.scatter(cluster_data['x'], cluster_data['y'],
                  c=color, s=2, alpha=0.3)
        
        # Точки ТЦ (черные)
        if plot_TC and tc_ids:
            tc_data = cluster_data[cluster_data['track_id'].isin(tc_ids)]
            if len(tc_data) > 0:
                ax.scatter(tc_data['x'], tc_data['y'],
                          c='black', s=5, alpha=0.8, zorder=10)
        
        # Точки ПМЦ (синие)
        if plot_TC and pmc_ids:
            pmc_data = cluster_data[cluster_data['track_id'].isin(pmc_ids)]
            if len(pmc_data) > 0:
                ax.scatter(pmc_data['x'], pmc_data['y'],
                          c='blue', s=5, alpha=0.8, zorder=11)
        
        ax.set_title(f'{season}: Кластер {cluster_idx} (точек: {len(cluster_data)})', fontsize=10)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

        if data_type == 'ERA5':
            ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        
        # Легенда
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=color, lw=2, label=f'Cluster {cluster_idx}')
        ]
        if plot_TC and tc_ids and len(cluster_data[cluster_data['track_id'].isin(tc_ids)]) > 0:
            legend_elements.append(Line2D([0], [0], color='black', lw=2, label='TC'))
        if plot_TC and pmc_ids and len(cluster_data[cluster_data['track_id'].isin(pmc_ids)]) > 0:
            legend_elements.append(Line2D([0], [0], color='blue', lw=2, label='PMC'))
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Выключаем лишние subplot'ы
    for i in range(n_clusters, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    output_file = f"{output_dir}/cluster_pics/nclusters_{n_clusters}_{season}_points_all_clusters.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def plot_points_by_clusters(df_clustered, n_clusters, colors, ground, output_dir, season, 
                           data_type='LoRes', plot_TC=False, track_to_pmc=None,
                           plot_monthly_dist=False, months_order=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                           monthly_mode='relative'):
    """Визуализирует точки по кластерам на карте с опциональными гистограммами по месяцам"""
    
    # Проверка параметров
    if plot_monthly_dist and df_clustered is None:
        raise ValueError("Для отрисовки месячных распределений необходим df_clustered с колонкой 'datetime'")
    
    if monthly_mode not in ('relative', 'absolute'):
        raise ValueError("monthly_mode должен быть 'relative' или 'absolute'")
    
    # Определяем ID ТЦ/ПМЦ
    tc_ids = set()
    pmc_ids = set()
    if plot_TC and track_to_pmc:
        for tid, info in track_to_pmc.items():
            if info.get('тип') == 'ТЦ':
                tc_ids.add(tid)
            elif info.get('тип') == 'ПМЦ':
                pmc_ids.add(tid)
    
    # Подготовка данных для месячных распределений
    if plot_monthly_dist:
        # Проверяем наличие колонки datetime
        if 'datetime' not in df_clustered.columns:
            raise ValueError("df_clustered должен содержать колонку 'datetime'")
        
        # Добавляем колонку с месяцем
        df_clustered['Month'] = df_clustered['datetime'].dt.strftime('%b')
        
        # Группируем по кластеру и месяцу
        df_grouped = df_clustered.groupby(['cluster', 'Month']).size().unstack(fill_value=0)
        df_grouped = df_grouped.reindex(columns=months_order, fill_value=0)
        
        if monthly_mode == 'relative':
            # Проценты внутри кластера
            df_plot = df_grouped.div(df_grouped.sum(axis=1), axis=0) * 100
            y_label = 'Freq (%)'
            global_max = None
        else:
            # Абсолютные значения
            df_plot = df_grouped
            y_label = 'Count'
            global_max = df_plot.values.max() if df_plot.size > 0 else 0
    
    # Создаем сетку графиков
    if plot_monthly_dist:
        # Каждый кластер занимает 2 строки: карта + гистограмма
        cols = min(3, n_clusters)
        rows = (n_clusters + cols - 1) // cols
        
        # Используем GridSpec для гибкого размещения
        gs = plt.GridSpec(2 * rows, cols, height_ratios=[2, 1] * rows, 
                         hspace=0.25, wspace=0.25)
        fig = plt.figure(figsize=(5 * cols, 3.5 * 2 * rows), dpi=150)
    else:
        # Стандартная сетка только с картами
        cols = min(4, n_clusters)
        rows = (n_clusters + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
        axes = axes.flatten()
    
    # Цикл по кластерам
    for cluster_idx in range(n_clusters):
        if plot_monthly_dist:
            # Получаем позиции для карты и гистограммы
            r = cluster_idx // cols
            c = cluster_idx % cols
            ax = fig.add_subplot(gs[2 * r, c])
            ax_bar = fig.add_subplot(gs[2 * r + 1, c])
        else:
            ax = axes[cluster_idx]
            ax_bar = None
        
        # Точки этого кластера
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_idx]
        
        if len(cluster_data) == 0:
            if plot_monthly_dist:
                ax.axis('off')
                ax_bar.axis('off')
            else:
                ax.axis('off')
            continue
        
        # Отрисовка фона (суша)
        plot_ground(ax, ground)
        
        # Определяем цвет кластера
        color = colors(cluster_idx) if callable(colors) else colors[cluster_idx % len(colors)]
        
        # Обычные точки
        ax.scatter(cluster_data['x'], cluster_data['y'],
                  c=color, s=2, alpha=0.3)
        
        # Точки ТЦ (черные)
        if plot_TC and tc_ids:
            tc_data = cluster_data[cluster_data['track_id'].isin(tc_ids)]
            if len(tc_data) > 0:
                ax.scatter(tc_data['x'], tc_data['y'],
                          c='black', s=5, alpha=0.8, zorder=10)
        
        # Точки ПМЦ (синие)
        if plot_TC and pmc_ids:
            pmc_data = cluster_data[cluster_data['track_id'].isin(pmc_ids)]
            if len(pmc_data) > 0:
                ax.scatter(pmc_data['x'], pmc_data['y'],
                          c='blue', s=5, alpha=0.8, zorder=11)
        
        # Заголовок
        title = f'{season}: Кластер {cluster_idx} (точек: {len(cluster_data)})'
        if plot_TC:
            n_tc = len(cluster_data[cluster_data['track_id'].isin(tc_ids)])
            n_pmc = len(cluster_data[cluster_data['track_id'].isin(pmc_ids)])
            if n_tc > 0 or n_pmc > 0:
                title += f' (TC: {n_tc}, PMC: {n_pmc})'
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        
        if data_type == 'ERA5':
            ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        
        # Легенда для карты
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=color, lw=2, label=f'Cluster {cluster_idx}')
        ]
        if plot_TC and tc_ids and len(cluster_data[cluster_data['track_id'].isin(tc_ids)]) > 0:
            legend_elements.append(Line2D([0], [0], color='black', lw=2, label='TC'))
        if plot_TC and pmc_ids and len(cluster_data[cluster_data['track_id'].isin(pmc_ids)]) > 0:
            legend_elements.append(Line2D([0], [0], color='blue', lw=2, label='PMC'))
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        # Отрисовка гистограммы по месяцам
        if plot_monthly_dist and cluster_idx in df_plot.index:
            freqs = df_plot.loc[cluster_idx].reindex(months_order, fill_value=0)
            bar_color = color
            
            bars = ax_bar.bar(months_order, freqs.values,
                             color=bar_color, edgecolor='black', linewidth=0.5, alpha=0.9)
            
            # Подписи значений на столбцах
            for bar, height in zip(bars, freqs.values):
                if monthly_mode == 'relative':
                    cond = height > 0.5
                    text_val = f'{height:.1f}'
                else:
                    cond = height > 0
                    text_val = f'{int(height)}'
                if cond:
                    ax_bar.text(bar.get_x() + bar.get_width() / 2, height + (0.5 if monthly_mode == 'relative' else 0.2),
                               text_val, ha='center', va='bottom', fontsize=7)
            
            # Настройки гистограммы
            ax_bar.tick_params(axis='x', labelrotation=45, labelsize=8)
            ax_bar.set_ylabel(y_label, fontsize=9)
            
            if monthly_mode == 'relative':
                ax_bar.set_ylim(0, min(35, freqs.max() * 1.2))
            else:
                # единый лимит по всем кластерам
                ax_bar.set_ylim(0, global_max * 1.1 if global_max > 0 else 1)
            
            ax_bar.grid(True, alpha=0.3, axis='y')
            
            # Подпись для гистограммы
            if monthly_mode == 'relative':
                ax_bar.set_title('Распределение по месяцам (отн.)', fontsize=9, pad=2)
            else:
                ax_bar.set_title('Распределение по месяцам (абс.)', fontsize=9, pad=2)
        elif plot_monthly_dist:
            ax_bar.axis('off')
    
    # Выключаем лишние subplot'ы
    if not plot_monthly_dist:
        for i in range(n_clusters, len(axes)):
            axes[i].axis('off')
    
    # Общий заголовок
    if plot_monthly_dist:
        plt.suptitle(f'{season.capitalize()} сезон: точки и распределение по месяцам', 
                    fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Сохранение
    if plot_monthly_dist:
        suffix = '_monthly_rel' if monthly_mode == 'relative' else '_monthly_abs'
        output_file = f"{output_dir}/cluster_pics/nclusters_{n_clusters}_{season}_points_all_clusters{suffix}.png"
    else:
        output_file = f"{output_dir}/cluster_pics/nclusters_{n_clusters}_{season}_points_all_clusters.png"
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    
# def plot_points_by_clusters(df_clustered, n_clusters, colors, ground, output_dir, season, 
#                            data_type='LoRes', plot_TC=False, track_to_pmc=None):
#     """Визуализирует точки по кластерам на карте с выделением старта (зеленый) и конца (красный) треков"""
    
#     # Определяем ID ТЦ/ПМЦ
#     tc_ids = set()
#     pmc_ids = set()
#     if plot_TC and track_to_pmc:
#         for tid, info in track_to_pmc.items():
#             if info.get('тип') == 'ТЦ':
#                 tc_ids.add(tid)
#             elif info.get('тип') == 'ПМЦ':
#                 pmc_ids.add(tid)
    
#     cols = min(4, n_clusters)
#     rows = (n_clusters + cols - 1) // cols
#     fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
#     axes = axes.flatten()
    
#     for cluster_idx in range(n_clusters):
#         ax = axes[cluster_idx]
        
#         # Точки этого кластера
#         cluster_data = df_clustered[df_clustered['cluster'] == cluster_idx]
        
#         if len(cluster_data) == 0:
#             ax.axis('off')
#             continue
        
#         # Отрисовка фона (суша)
#         plot_ground(ax, ground)
        
#         # Определяем цвет кластера
#         color = colors(cluster_idx) if callable(colors) else colors[cluster_idx % len(colors)]
        
#         # Обычные точки
#         ax.scatter(cluster_data['x'], cluster_data['y'],
#                   c=color, s=2, alpha=0.3)
        
#         # Находим стартовые и конечные точки для каждого трека в этом кластере
#         start_indices = []
#         end_indices = []
        
#         for track_id in cluster_data['track_id'].unique():
#             track_data = cluster_data[cluster_data['track_id'] == track_id]
#             # Сортируем по времени (предполагаем, что time колонка есть и трек упорядочен)
#             track_data_sorted = track_data.sort_values('time')
            
#             if len(track_data_sorted) > 0:
#                 # Индекс первой точки - старт
#                 start_indices.append(track_data_sorted.iloc[0].name)
#                 # Индекс последней точки - конец
#                 end_indices.append(track_data_sorted.iloc[-1].name)
        
#         # Контуры для стартовых точек (зеленый) - рисуем поверх существующих точек
#         if start_indices:
#             start_df = cluster_data.loc[start_indices]
#             ax.scatter(start_df['x'], start_df['y'],
#                       facecolors='none', edgecolors='limegreen', 
#                       s=8, linewidths=0.8, zorder=6)
        
#         # Контуры для конечных точек (красный)
#         if end_indices:
#             end_df = cluster_data.loc[end_indices]
#             ax.scatter(end_df['x'], end_df['y'],
#                       facecolors='none', edgecolors='red', 
#                       s=8, linewidths=0.8, zorder=6)
        
#         # Точки ТЦ (черные)
#         if plot_TC and tc_ids:
#             tc_data = cluster_data[cluster_data['track_id'].isin(tc_ids)]
#             if len(tc_data) > 0:
#                 ax.scatter(tc_data['x'], tc_data['y'],
#                           c='black', s=5, alpha=0.8, zorder=10)
        
#         # Точки ПМЦ (синие)
#         if plot_TC and pmc_ids:
#             pmc_data = cluster_data[cluster_data['track_id'].isin(pmc_ids)]
#             if len(pmc_data) > 0:
#                 ax.scatter(pmc_data['x'], pmc_data['y'],
#                           c='blue', s=5, alpha=0.8, zorder=11)
        
#         ax.set_title(f'{season}: Кластер {cluster_idx} (точек: {len(cluster_data)})', fontsize=10)
#         ax.set_xlabel('x')
#         ax.set_ylabel('y')
#         ax.set_aspect('equal')

#         if data_type == 'ERA5':
#             ax.invert_yaxis()
#         ax.grid(True, alpha=0.3)
        
#         # Легенда
#         from matplotlib.lines import Line2D
#         legend_elements = [
#             Line2D([0], [0], color=color, lw=2, label=f'Cluster {cluster_idx}')
#         ]
#         if start_indices:
#             legend_elements.append(Line2D([0], [0], marker='o', color='w', 
#                                          markerfacecolor='none', markeredgecolor='limegreen', 
#                                          markersize=4, label='Start'))
#         if end_indices:
#             legend_elements.append(Line2D([0], [0], marker='o', color='w', 
#                                          markerfacecolor='none', markeredgecolor='red', 
#                                          markersize=4, label='End'))
#         if plot_TC and tc_ids and len(cluster_data[cluster_data['track_id'].isin(tc_ids)]) > 0:
#             legend_elements.append(Line2D([0], [0], color='black', lw=2, label='TC'))
#         if plot_TC and pmc_ids and len(cluster_data[cluster_data['track_id'].isin(pmc_ids)]) > 0:
#             legend_elements.append(Line2D([0], [0], color='blue', lw=2, label='PMC'))
#         ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
#     # Выключаем лишние subplot'ы
#     for i in range(n_clusters, len(axes)):
#         axes[i].axis('off')
    
#     plt.tight_layout()
#     output_file = f"{output_dir}/cluster_pics/nclusters_{n_clusters}_{season}_points_all_clusters.png"
#     plt.savefig(output_file, dpi=150, bbox_inches='tight')
#     plt.close()
    
    
    
    
def process_season(season_name, months, n_clusters_range, clustering_type='gmm', scaler_type='standart'):
    metrics_list_season = []
    
    print(f"\nЗагрузка данных из {input_file}...")
    df_season = pd.read_csv(input_file, parse_dates=['time'])
    print(f"Всего точек: {len(df_season)}")

    df_season['datetime'] = df_season['time']
    df_season['x'] = df_season['pxc_ind']
    df_season['y'] = df_season['pyc_ind']

    df_season = df_season.dropna(subset=param_cols)
    print(f"Всего точек без nan: {len(df_season)}")
    

        
    for n_clusters in tqdm(n_clusters_range, desc=f"{season_name} ({clustering_type})"):
        colors = plt.cm.get_cmap('tab20', n_clusters)
        



        # # ---------- 2. NaN → 0 для выбранных параметров ----------
        # # какие поля хотим заполнять нулями (пример)
        # fill0_cols = [
        #     'mucape_95', 'mcin_95', 'helicity_95', 'pw_95',
        #     # добавь сюда остальные нужные параметры
        # ]
        # # оставляем только реально существующие в df_season
        # fill0_cols = [c for c in fill0_cols if c in df_season.columns]
        # if fill0_cols:
        #     df_season[fill0_cols] = df_season[fill0_cols].fillna(0)
        #     print(f'NaN → 0 для колонок: {fill0_cols}')

        # ---------- 3. дальше как у тебя ----------
        numeric_cols = [c for c in param_cols if df_season[c].notna().all()]
        print(f'всего {len(numeric_cols)} параметров для кластеризации (не None)')  
        print('Параметры:', numeric_cols)

        df_season, cluster_groups, gmm_metrics = get_clusters_universal(
            df_season, numeric_cols, n_clusters,
            clustering_type=clustering_type, scaler_type=scaler_type
        )
        
        # Для точек мы создаем "треки" из отдельных точек, группируя их по track_id
        clusters_full_tracks = {}
        
        # Получаем уникальные track_id и их точки
        for cluster in cluster_groups.keys():
            cluster_indices = cluster_groups[cluster]
            cluster_df = df_season.iloc[cluster_indices]
            
            # Группируем по track_id, чтобы создать "треки" из точек одного трека
            tracks_dict = {}
            for track_id in cluster_df['track_id'].unique():
                track_points = cluster_df[cluster_df['track_id'] == track_id].copy()
                # Переименовываем колонки в формат, ожидаемый функциями визуализации
                track_points_for_plot = track_points.rename(columns={
                    'pxc_ind': 'x',
                    'pyc_ind': 'y'
                })
                tracks_dict[track_id] = track_points_for_plot
            
            clusters_full_tracks[cluster] = list(tracks_dict.values())
        
        # Проверяем наличие ТЦ/ПМЦ
        all_marked_ids = set(track_to_pmc.keys())
        
        present_ids = set()
        for clust, tracks in clusters_full_tracks.items():
            for tr in tracks:
                if 'track_id' in tr.columns:
                    present_ids.update(tr['track_id'].unique())
        
        present = all_marked_ids & present_ids
        missing = all_marked_ids - present_ids
        
        print(f'Всего в словаре: {len(all_marked_ids)}')
        print(f'Есть в данных: {len(present)}, {sorted(present)}')
        print(f'Нет в данных: {len(missing)}, {sorted(missing)}')
        

        # Визуализация точек на карте
        plot_points_by_clusters(
            df_season, n_clusters, colors, ground, output_dir,
            season=season_name, data_type=data_type,
            plot_TC=True, track_to_pmc=track_to_pmc,
            plot_monthly_dist=True, 
            # monthly_mode='relative',
            monthly_mode='absolute',
            
        )
        
        
#         plot_cluster_boxplots_all(df_season, numeric_cols, n_clusters, colors, output_dir, 
#                                 season=season_name)
        
        
        plot_path = plot_cluster_boxplots_all_with_TC_PMC(
                                                df=df_season,
                                                numeric_cols=numeric_cols,
                                                n_clusters=int(n_clusters),
                                                colors=colors,
                                                path_data_tracks=output_dir,
                                                season=season_name,
                                                clustering_type=clustering_type,
                                                track_to_cyclone=track_to_pmc,
                                            )
        
#         print(plot_path)
        
        

#         print(f'Getting metrics>>>')

#         X_for_metrics = df_season[numeric_cols].dropna().values  # ← numpy array
#         labels_for_metrics = df_season['cluster'].loc[df_season[numeric_cols].dropna().index].values
        
#         metrics = compute_metrics(X_for_metrics, labels_for_metrics, gmm_metrics, clustering_type, 
# #                                     compute_classifiability_flag=True,
# #                                     compute_reproducibility_flag=True,
#                                  )
        
#         metrics.update({
#             'n_clusters': n_clusters,
#             'season': season_name,
#             'clustering_type': clustering_type
#         })
#         metrics_list_season.append(metrics)
        
        # Сохранение
        df_season.to_csv(f"{output_dir}/cluster_tables/{season_name}_n{n_clusters}_{clustering_type}.csv", 
                        index=False)
    
    return metrics_list_season

# Основной блок выполнения (исправленный)
n_clusters_range = range(2, 15)
metrics_file = f"{output_dir}/clustering_metrics_internal.csv"

if input_season == 'all':
    print("🚀 Запуск последовательной обработки всех сезонов...")
    seasons_config = {
        'winter': {'months': [1, 2, 3]},
        'summer': {'months': [7, 8, 9]},
        'year': {'months': list(range(1, 13))}
    }
    
    all_metrics = []
    for season_name, config in seasons_config.items():
        print(f"🔍 Обработка сезона: {season_name}")
        try:
            season_metrics = process_season(season_name, config['months'], n_clusters_range, clustering_type)
            all_metrics.extend(season_metrics)
            print(f"✅ Сезон {season_name} завершен")
        except Exception as e:
            print(f"❌ Ошибка в сезоне {season_name}: {e}")
            import traceback
            traceback.print_exc()
    
    metrics_df_new = pd.DataFrame(all_metrics)
    
else:
    # Обработка одного сезона
    if input_season == 'winter':
        months = [1, 2, 3]
    elif input_season == 'summer':
        months = [7, 8, 9]
    else:  # year
        months = list(range(1, 13))
    
    print(f"🔍 Обработка сезона: {input_season}")
    metrics_list = process_season(input_season, months, n_clusters_range, clustering_type=clustering_type, scaler_type=scaler_type)
    metrics_df_new = pd.DataFrame(metrics_list)

# ✅ Сохранение метрик
save_metrics(metrics_df_new, input_season, metrics_file)

# ✅ Визуализация (отдельная функция)
plot_clustering_metrics(metrics_df_new, input_season, output_dir, clustering_type)

print(f"✅ Анализ для {scaler_type} {clustering_type} завершен!")