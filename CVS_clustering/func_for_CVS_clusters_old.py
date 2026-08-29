from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture


from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import itertools

import hdbscan
from sklearn.manifold import TSNE
from itertools import combinations


import matplotlib.pyplot as plt
import seaborn as sns

path_init = f'/storage/thalassa/users/vkoshkina'
folder = 'scripts/CS_processing/CVS_tracking/after_70RAE/'

sys.path.insert(2, f'{path_init}/{folder}')

from step_of_tracking import *

def load_season_tracks(CS_tracks_list, path_list, basenames_list, year, months, path_data, time_th=3): 
    null_files = []
    
    for month in months:
        if os.path.exists(path_data):
            ls = list(sorted(Path(f"{path_data}/{year}-{month:02d}").glob(f'*_track_*.csv')))
            
            if len(ls) != 0:
                for ii, ifile in tqdm(enumerate(ls), total=len(ls), desc=f"Loading tracks for {year}-{month:02d}"):
                    df = pd.read_csv(ifile, parse_dates=['datetime'])
                    if len(df) > time_th: 
                        # df = df.drop(df.columns[0], axis=1)
                        df = df.dropna(how='any')
                        
                        CS_tracks_list.append(df)
                        path_list.append(ifile)
                        basenames_list.append(os.path.basename(ifile))
                    else:
                        null_files.append(ifile)
    print(f'Too short tracks for {year}: {len(null_files)}')
                        
    return CS_tracks_list, path_list, basenames_list

def load_season_df(path_tracks_dir, years, months, season):
    df_winter = pd.DataFrame()

    for year in tqdm(years, total=len(years), desc=f"Loading tracks for {season} months"):
        for month in months:
            df_1 = pd.read_csv(f'{path_tracks_dir}/max_crit_day_CVS_{year}-{month:02d}.csv')
            df_winter = pd.concat([df_winter, df_1], ignore_index=True)
    return df_winter

def load_season_df_EC(path_tracks_dir, months, season, params_type='max_wspd_day'):

    df_season = pd.read_csv(f'{path_tracks_dir}/all_tracks_with_params_{params_type}.csv', parse_dates=['datetime'])
    df_season = df_season[np.isin(df_season.datetime.dt.month, months)]
    
    # df_season['track_id'] = (
    #         df_season['name']
    #         .str.replace('.csv', '', regex=False)
    #         .str.lstrip('0')
    #         .astype(int)
    #     )
           
    return df_season

def is_over_ocean(x, y, land_mask):
    
    # Значение маски (0 = океан, 1 = суша)
    # return land_mask.isel(west_east=x, south_north=y).values[0] == 1.
    return land_mask[y,x] != 1.
    

def filter_ocean_tracks(CS_tracks_list, path_list, basenames_list, land_mask, ocean_threshold=0.5):
    """
    Фильтрует треки, оставляя только те, которые находятся над океаном достаточное время
    
    Args:
        CS_tracks_list: список треков
        land_mask_path: путь к файлу с маской суши (NetCDF)
        ocean_threshold: минимальная доля времени над океаном (по умолчанию 0.5)
    
    Returns:
        Список треков, удовлетворяющих условию
    """

    
    filtered_tracks = []
    filtered_paths = []
    filtered_names = []

    
    for idx, track in enumerate(CS_tracks_list):
        if isinstance(track, dict):
            # Для треков в формате словаря
            x = track['x']
            y = track['y']
        else:
            # Для треков в формате DataFrame
            x = track['x'].values
            y = track['y'].values
        
        # Считаем количество точек над океаном
        ocean_points = sum(is_over_ocean(int(xx), int(yy), land_mask) for xx, yy in zip(x, y))
        ocean_ratio = ocean_points / len(x)
        
        if ocean_ratio >= ocean_threshold:
            filtered_tracks.append(track)
            filtered_paths.append(path_list[idx])
            filtered_names.append(basenames_list[idx])
            
    
    return filtered_tracks, filtered_paths, filtered_names

def split_tracks_by_season(tracks_list):
    """
    Разделяет треки на зимние (январь-март) и летние (июнь-сентябрь)
    
    Returns:
        tuple: (winter_tracks, summer_tracks)
    """
    winter_tracks = []
    summer_tracks = []
    
    for track in tracks_list:
        if isinstance(track, dict):
            # Для треков в формате словаря
            first_date = track['time'][0]
        else:
            # Для треков в формате DataFrame
            first_date = track['datetime'].iloc[0]
        
        month = first_date.month
        
        if month in [1, 2, 3]:  # январь-март
            winter_tracks.append(track)
        elif month in [7, 8, 9]:  # июнь-сентябрь
            summer_tracks.append(track)
    
    return winter_tracks, summer_tracks
    
def get_max_crit_day_values(CS, df, idx, param_cols):
    CS_max = CS[CS['crit'] == np.nanmax(CS['crit'])]
    df = pd.concat([df, CS_max], ignore_index=True)
    return df, CS_max.index

def pca_analysis(data):
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)
    return data_pca

def ca_analysis(data, n_clusters=12):
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(data)
    return clusters

# def group_cyclones(cyclones, clusters, path_list, basenames_list):
#     cluster_dict = {}
#     names_dict = {}
#     for i in range(len(clusters)):
#         if clusters[i] not in cluster_dict:
#             cluster_dict[clusters[i]] = []
#             names_dict[clusters[i]] = []
#         cluster_dict[clusters[i]].append(i)
#         names_dict[clusters[i]].append((path_list[i], basenames_list[i]))
#     return cluster_dict, names_dict

# Group cyclones into clusters based on their spatial characteristics
def group_cyclones(cyclones, clusters):
    cluster_dict = {}
    for i in range(len(clusters)):
        if clusters[i] not in cluster_dict:
            cluster_dict[clusters[i]] = []
        cluster_dict[clusters[i]].append(i)
    return cluster_dict

def preprocess_data(df, numeric_cols, scaler_type='standart'):
    # Нормализация + PCA для уменьшения размерности
    # Robust Scaler uses the median and interquartile range (IQR) instead of the mean and standard deviation, which are susceptible to outliers
    X = df[numeric_cols].dropna().values

    if scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    
    X_scaled = scaler.fit_transform(X)
    
    # Уменьшение размерности сохраняя 95% дисперсии
    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(X_scaled)
        
    print(f"PCA: {X.shape[1]} → {X_reduced.shape[1]} компонент (сохранено {pca.explained_variance_ratio_.sum():.1%} дисперсии)")
    
    return X_reduced

def get_clusters_universal(df, numeric_cols, n_clusters=10, clustering_type='kmeans', random_state=42, scaler_type='standart'):
    """Универсальная кластеризация"""
    # df = df.copy()
    df = df.dropna(subset=numeric_cols).copy()
    
    X = preprocess_data(df, numeric_cols, scaler_type)
    
    if clustering_type == 'gmm':
        clusterer = GaussianMixture(
            n_components=n_clusters,
            covariance_type='full',
            max_iter=200,
            n_init=10,
            reg_covar=1e-6,
            random_state=random_state
        )
        clusters = clusterer.fit_predict(X)
        gmm_metrics = {
            'aic': clusterer.aic(X),
            'bic': clusterer.bic(X),
            'log_likelihood': clusterer.score(X)
        }
        
    elif clustering_type == 'kmeans':  # KMeans
        clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
        clusters = clusterer.fit_predict(X)
        gmm_metrics = {}
    
    # Присваиваем метки
    clean_idx = df[numeric_cols].dropna().index
    df.loc[clean_idx, 'cluster'] = clusters
    
    print(f"{clustering_type.upper()}: {df['cluster'].value_counts().sort_index().dropna()}")
    
    cluster_groups = group_cyclones(df, df['cluster'].values)
    
    return df, cluster_groups, gmm_metrics

def get_clusters(df, numeric_cols, n_clusters=10):

    df = df.dropna
    X = preprocess_data(df, numeric_cols)

    clusterer = KMeans(n_clusters=n_clusters)
    clusters = clusterer.fit_predict(X)
    df['cluster'] = clusters
    
    print(df['cluster'].value_counts().sort_index())

    cluster_groups = group_cyclones(df, clusters)
    
    
    return df, cluster_groups

def get_clusters_KMeansConstrained(df, numeric_cols, n_clusters=10, min_size=20):
    df = df.dropna()
    X = preprocess_data(df, numeric_cols)

    clusterer = KMeans(n_clusters, random_state=42)
    clusters = clusterer.fit_predict(X)
    df['cluster'] = clusters

    print("До:", df['cluster'].value_counts().sort_index())

    # Постобработка малых кластеров
    cluster_counts = pd.Series(clusters).value_counts()
    small_clusters = cluster_counts[cluster_counts < min_size].index.tolist()
    
    if small_clusters:
        large_indices = [i for i in range(n_clusters) if i not in small_clusters]
        large_centers = clusterer.cluster_centers_[large_indices]
        
        small_mask = np.isin(clusters, small_clusters)
        small_points = X[small_mask]
        
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(small_points, large_centers)
        new_labels = large_indices[np.argmin(distances, axis=1)]
        
        clusters[small_mask] = new_labels
        df['cluster'] = clusters

    
    # ✅ ПЕРЕИНДЕКСАЦИЯ: 0,1,2,...,N где N=число кластеров после слияния
    unique_clusters = sorted(set(df['cluster'].values))
    n_final_clusters = len(unique_clusters)
    cluster_map = {old: new for new, old in enumerate(unique_clusters)}
    
    df['cluster'] = df['cluster'].map(cluster_map)
    
    print(f"Финальные метки (0-{n_final_clusters-1}):")
    print(df['cluster'].value_counts().sort_index())
    
    cluster_groups = group_cyclones(df, df['cluster'].values)
    return df, cluster_groups

def get_clusters_GMM(df, numeric_cols, n_clusters=10, random_state=42):
    """GMM кластеризация с оптимальными параметрами"""
    df = df.dropna()
    X = preprocess_data(df, numeric_cols)
    
    # GMM параметры (оптимизированы для атмосферных данных)
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type='full',      # Полные ковариационные матрицы
        max_iter=200,                # Больше итераций для сходимости
        n_init=10,                   # Многократная инициализация
        reg_covar=1e-6,              # Регуляризация для стабильности
        random_state=random_state
    )
    
    clusters = gmm.fit_predict(X)
    df.loc[df[numeric_cols].dropna().index, 'cluster'] = clusters  # Только для чистых строк
    
    print(f"GMM: n_clusters={n_clusters}")
    print(df['cluster'].value_counts().sort_index().dropna())
    
    # AIC/BIC для оценки качества
    aic = gmm.aic(X)
    bic = gmm.bic(X)
    print(f"AIC: {aic:.1f}, BIC: {bic:.1f}")
    
    cluster_groups = group_cyclones(df, df['cluster'].values)
    
    return df, cluster_groups


def compute_metrics(X, labels, gmm_metrics=None, clustering_type='gmm'):
    """Обновленные метрики с учетом GMM"""
    valid_idx = ~(np.isnan(labels) | np.any(np.isnan(X), axis=1))
    X_clean = X[valid_idx]
    labels_clean = labels[valid_idx].astype(int)
    
    metrics = {
        'silhouette': silhouette_score(X_clean, labels_clean),
        'connectivity': compute_connectivity(X_clean, labels_clean, L=20),
        'hubert_g': compute_huberts_g_statistic(X_clean, labels_clean),
        'stability': compute_stability(X_clean, labels_clean, n_subsamples=20, 
                                     n_clusters=len(np.unique(labels_clean)))
    }
    
    # GMM-специфичные метрики
    if gmm_metrics:
        metrics.update(gmm_metrics)
    
    return metrics

def save_metrics(metrics_df, input_season, metrics_file):
    """Универсальное сохранение метрик"""
    if os.path.exists(metrics_file):
        metrics_df_existing = pd.read_csv(metrics_file)
        if input_season == 'all':
            metrics_df_combined = metrics_df
            print("✅ Все сезоны пересчитаны")
        else:
            metrics_df_existing = metrics_df_existing[metrics_df_existing['season'] != input_season]
            metrics_df_combined = pd.concat([metrics_df_existing, metrics_df], ignore_index=True)
            print(f"✅ Метрики дозаписаны для {input_season}")
        metrics_df_combined.to_csv(metrics_file, index=False)
    else:
        metrics_df.to_csv(metrics_file, index=False)
        print(f"✅ Новый файл метрик для {input_season if input_season != 'all' else 'всех сезонов'}")

def plot_clustering_metrics(metrics_df, input_season, output_dir, clustering_type):
    """Универсальная визуализация метрик"""
    sns.set_style("whitegrid")
    
    # GMM-специфичные метрики (если есть)
    gmm_metrics = ['aic', 'bic', 'log_likelihood'] if 'aic' in metrics_df.columns else []
    common_metrics = ['silhouette', 'connectivity', 'hubert_g', 'stability']
    
    if input_season == 'all':
        seasons_to_plot = ['winter', 'summer', 'year']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        # Сводный график
        fig = plt.figure(figsize=(16, 12))
        plot_metrics_subplot(fig, metrics_df, seasons_to_plot, colors, common_metrics, gmm_metrics)
        plt.suptitle(f'Clustering Metrics - {clustering_type.upper()} (All Seasons)', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/metrics_{clustering_type}_all_seasons.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Отдельные по сезонам
        for season_name in seasons_to_plot:
            plot_season_metrics(metrics_df, season_name, output_dir, clustering_type, gmm_metrics)
            
    else:
        plot_season_metrics(metrics_df, input_season, output_dir, clustering_type, gmm_metrics)

def plot_metrics_subplot(fig, metrics_df, seasons_to_plot, colors, common_metrics, gmm_metrics):
    """Подграфики для метрик (ИСПРАВЛЕН)"""
    all_metrics = common_metrics + gmm_metrics[:2]  # AIC/BIC
    
    ncols = 3 if gmm_metrics else 2
    for i, metric in enumerate(all_metrics):
        plt.subplot(2, ncols, i + 1)  # ✅ subplot, НЕ subplots!
        for season_name, color in zip(seasons_to_plot, colors):
            data_season = metrics_df[metrics_df['season'] == season_name]
            plt.plot(data_season['n_clusters'], data_season[metric], 
                     marker='o', label=season_name.capitalize(), color=color, linewidth=2)
        
        plt.xlabel('Number of Clusters')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(metric.replace('_', ' ').title())
        plt.xticks(range(2, 15))
        plt.grid(True, alpha=0.3)
        if i == 0: plt.legend()


def plot_season_metrics(metrics_df, season_name, output_dir, clustering_type, gmm_metrics):
    """График для одного сезона (ИСПРАВЛЕН)"""
    data = metrics_df[metrics_df['season'] == season_name]
    common_metrics = ['silhouette', 'connectivity', 'hubert_g', 'stability']
    all_metrics = common_metrics + (['aic', 'bic'] if gmm_metrics else [])
    
    # ✅ ПРАВИЛЬНЫЙ синтаксис subplots
    n_metrics = len(all_metrics)
    nrows = 2
    ncols = 3 if n_metrics > 4 else 2  # AIC/BIC добавляют 2 графика
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 10))
    axes = axes.ravel() if n_metrics > 1 else [axes]
    
    for i, metric in enumerate(all_metrics):
        axes[i].plot(data['n_clusters'], data[metric], marker='o', color='#1f77b4', linewidth=2)
        axes[i].set_xlabel('Number of Clusters')
        axes[i].set_ylabel(metric.replace('_', ' ').title())
        axes[i].set_title(f"{metric.replace('_', ' ').title()} - {season_name.capitalize()}")
        axes[i].grid(True, alpha=0.3)
    
    # Убираем пустые subplot'ы
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_{clustering_type}_{season_name}.png", dpi=300, bbox_inches='tight')
    plt.close()


    
# Функция для загрузки полного трека по пути из path
def load_full_track(path):
    try:
        return pd.read_csv(path)
    except:
        print(f"Не удалось загрузить файл: {path}")
        return None

def adding_cluster_tracks(df, cluster_groups):

    # Создаем словарь для хранения полных треков по кластерам
    clusters_full_tracks = {cluster: [] for cluster in cluster_groups.keys()}
    
    # Загружаем полные треки для каждого кластера
    for cluster in cluster_groups.keys():
        print(f"Загрузка треков для кластера {cluster}...")
        for idx in tqdm(cluster_groups[cluster]):
            path = df.iloc[idx]['path']
            name = df.iloc[idx]['name']
            full_track = load_full_track(path)
            track_id_str = name.replace('.csv', '').lstrip('0')
            
            if track_id_str == '':
                track_id = -1
            else:
                track_id = int(track_id_str)
    
            full_track['track_id'] = track_id
            
            if full_track is not None:
                full_track = full_track.rename(columns={
                                'pxc_ind': 'x',
                                'pyc_ind': 'y',
                            })
                clusters_full_tracks[cluster].append(full_track)

    return clusters_full_tracks


def plot_ground(ax, ground):
    xxx = np.arange(ground.shape[1])
    yyy = np.arange(ground.shape[0])
    xx, yy = np.meshgrid(xxx, yyy)
    
    mc = ax.contourf(xx, yy, ground, 
                # cmap='grey', 
                cmap='Oranges', 
                alpha=0.3
                 )

def plot_tracks_by_clusters(clusters_full_tracks, n_clusters, colors, ground, path_data_tracks, season, data_type='ERA5', track_type='track'):
    # Визуализация треков по кластерам
    for cluster, tracks in tqdm(clusters_full_tracks.items(), total=n_clusters, desc=f"Plot tracks for {season} clusters"):
        if not tracks:  # Пропускаем пустые кластеры
            continue
        
        # Создаем новую фигуру
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Отрисовываем фон (замените на ваши данные)
        plot_ground(ax, ground)
        
        # Рисуем все треки кластера
        for track in tracks:
            if track_type == 'track':
                ax.plot(track['x'], track['y'], 
                    color=colors(cluster), 
                    alpha=0.3, 
                    linewidth=1.0,
                    # marker='o', 
                    markersize=2)
            elif track_type == 'start':
                ax.scatter(track['x'][0], track['y'][0], 
                color=colors(cluster), 
                alpha=0.5, 
                s=5,
                  )
        
        # Настройки графика
        ax.set_title(f'{season}: кластер {cluster} (треков: {len(tracks)})', fontsize=12)
        ax.set_xlabel('Долгота', fontsize=10)
        ax.set_ylabel('Широта', fontsize=10)
        
        # Добавляем цветную метку кластера
        ax.scatter([], [], color=colors(cluster), label=f'Cluster {cluster}')
        ax.legend()
        if data_type == 'ERA5':
            ax.invert_yaxis()
        ax.grid()
        
        # # Для географических данных лучше использовать равное соотношение осей
        # ax.set_aspect('equal', adjustable='datalim')
        
        plt.tight_layout()
        output_dir = f"{path_data_tracks}/cluster_pics/nclusters_{n_clusters}"
        os.makedirs(output_dir, exist_ok=True)
        
        
        plt.savefig(f'{output_dir}/nclusters_{n_clusters}_{season}_{track_type}_cluster_{cluster}.png', )
        # plt.show()


def plot_tracks_by_clusters_all(clusters_full_tracks, n_clusters, colors, ground, path_data_tracks, season, data_type='ERA5', track_type='track', clustering_type='kmeans', clustering_params=None):
    # Определяем размер сетки подграфиков
    cols = min(4, n_clusters)  # максимум 4 столбца
    rows = (n_clusters + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()  # делаем плоский список осей

    # for cluster_idx, cluster in enumerate(clusters_full_tracks.keys()):
    for cluster_idx, cluster in enumerate(range(n_clusters)):
    
        ax = axes[cluster_idx]
        tracks = clusters_full_tracks[cluster]
        
        if not tracks:
            ax.axis('off')  # выключаем пустые subplot'ы
            continue
        
        plot_ground(ax, ground)

        for track in tracks:
            if track_type == 'track':
                ax.plot(track['x'], track['y'], 
                        color=colors(cluster), 
                        alpha=0.3, 
                        linewidth=1.0)
            elif track_type == 'start':
                ax.scatter(track['x'][0], track['y'][0], 
                           color=colors(cluster), 
                           alpha=0.5, 
                           s=5)

        ax.set_title(f'{season}: Кластер {cluster} (треков: {len(tracks)})', fontsize=10)
        ax.set_xlabel('Долгота')
        ax.set_ylabel('Широта')
        if data_type == 'ERA5':
            ax.invert_yaxis()
        ax.grid(True)
        ax.scatter([], [], color=colors(cluster), label=f'Cluster {cluster}')
        ax.legend(loc='upper right')

    # Выключаем лишние subplot'ы
    for i in range(len(clusters_full_tracks), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if clustering_type == 'hdbscan':
        output_dir = f"{path_data_tracks}/cluster_pics/nclusters_{n_clusters}_mcs{clustering_params[0]}_ms{clustering_params[1]}"
        filename = f"{output_dir}/nclusters_{n_clusters}_mcs{clustering_params[0]}_ms{clustering_params[1]}_{season}_{track_type}_all_clusters.png"
    else:
        output_dir = f"{path_data_tracks}/cluster_pics/nclusters_{n_clusters}"
        filename = f"{output_dir}/nclusters_{n_clusters}_{season}_{track_type}_all_clusters.png"
        
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_tracks_by_clusters_all_with_TC(
    clusters_full_tracks,
    n_clusters,
    colors,
    ground,
    path_data_tracks,
    season,
    data_type='ERA5',
    track_type='track',
    clustering_type='kmeans',
    clustering_params=None,
    plot_TC=False,
    track_to_cyclone=None
):
    """
    Визуализация треков по кластерам в сетке подграфиков.
    Если plot_TC=True и track_to_cyclone задан, ТЦ отмечаются черным цветом
    только на картах тех кластеров, куда они попали.
    """
    if track_to_cyclone is None:
        plot_TC = False

    tropical_ids = set(track_to_cyclone.keys())

    # 1. Для каждого кластера считаем число ТЦ и собираем его ТЦ‑треки
    cluster_tc_counts = {cluster: 0 for cluster in range(n_clusters)}
    cluster_tropical_tracks = {cluster: [] for cluster in range(n_clusters)}

    for cluster, tracks in clusters_full_tracks.items():
        for track in tracks:
            if not isinstance(track, pd.DataFrame):
                continue
            if 'track_id' not in track.columns:
                continue
            if 'x' not in track.columns or 'y' not in track.columns:
                continue

            track_id = track['track_id'].iloc[0]
            if track_id < 0:
                continue

            if track_id in tropical_ids:
                cluster_tropical_tracks[cluster].append(track)
                cluster_tc_counts[cluster] += 1

    print(f"DEBUG: cluster_tc_counts = {cluster_tc_counts}")

    # 2. Строим сетку графиков
    cols = min(4, n_clusters)
    rows = (n_clusters + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()

    for cluster_idx in range(n_clusters):
        ax = axes[cluster_idx]
        tracks = clusters_full_tracks.get(cluster_idx, [])

        if not tracks:
            ax.axis('off')
            continue

        plot_ground(ax, ground)

        # 2.1. Отрисовка обычных треков по кластерам
        for track in tracks:
            if not isinstance(track, pd.DataFrame):
                continue
            if 'x' not in track.columns or 'y' not in track.columns:
                continue

            x = track['x'].values
            y = track['y'].values

            if track_type == 'track':
                ax.plot(
                    x, y,
                    color=colors(cluster_idx),
                    alpha=0.3,
                    linewidth=1.0
                )
            elif track_type == 'start':
                ax.scatter(
                    x[0], y[0],
                    color=colors(cluster_idx),
                    alpha=0.5,
                    s=5
                )

        # 2.2. Отрисовка ТЦ‑треков только для этого кластера
        if plot_TC and cluster_tropical_tracks[cluster_idx]:
            for track in cluster_tropical_tracks[cluster_idx]:
                if not isinstance(track, pd.DataFrame):
                    continue
                if 'x' not in track.columns or 'y' not in track.columns:
                    continue
                x = track['x'].values
                y = track['y'].values
                ax.plot(
                    x, y,
                    color='black',
                    linewidth=1.0,
                    zorder=10
                )

        # 2.3. Подпись с числом ТЦ в кластере
        n_tc = cluster_tc_counts[cluster_idx]
        title = f'{season}: Кластер {cluster_idx} (треков: {len(tracks)})'
        if plot_TC:
            title += f' ({n_tc} TC)'
        ax.set_title(title, fontsize=10)

        ax.set_xlabel('Долгота')
        ax.set_ylabel('Широта')
        if data_type == 'ERA5':
            ax.invert_yaxis()
        ax.grid(True)

        # Легенда
        ax.scatter([], [], color=colors(cluster_idx), label=f'Cluster {cluster_idx}')
        if plot_TC and cluster_tropical_tracks[cluster_idx]:
            ax.plot([], [], color='black', label='Tropical Cyclone', linewidth=2)
        ax.legend(loc='upper right')

    # Выключаем лишние subplot'ы
    for i in range(len(clusters_full_tracks), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if clustering_type == 'hdbscan':
        output_dir = f"{path_data_tracks}/cluster_pics/nclusters_{n_clusters}_mcs{clustering_params[0]}_ms{clustering_params[1]}"
        filename = f"{output_dir}/nclusters_{n_clusters}_mcs{clustering_params[0]}_ms{clustering_params[1]}_{season}_{track_type}_all_clusters.png"
    else:
        output_dir = f"{path_data_tracks}/cluster_pics/nclusters_{n_clusters}"
        filename = f"{output_dir}/nclusters_{n_clusters}_{season}_{track_type}_all_clusters.png"

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(filename, dpi=150)
    plt.close()

# новая функция с гистограммкой по месяцам
def plot_tracks_by_clusters_all_with_TC(
    clusters_full_tracks,
    n_clusters,
    colors,
    ground,
    path_data_tracks,
    season,
    data_type='ERA5',
    track_type='track',
    clustering_type='kmeans',
    clustering_params=None,
    plot_TC=False,
    track_to_cyclone=None,
    plot_monthly_dist=False,  # Новый параметр для отрисовки месячных распределений
    df_season=None,  # DataFrame с данными для месячных распределений
    months_order=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
):
    """
    Визуализация треков по кластерам в сетке подграфиков.
    
    Параметры:
    ----------
    plot_monthly_dist : bool
        Если True, отрисовывает гистограммы распределения по месяцам для каждого кластера
    df_season : pandas.DataFrame, optional
        DataFrame с колонками ['cluster', 'datetime'] для построения месячных распределений
    months_order : list
        Порядок месяцев для отображения на гистограммах
    """
    if plot_monthly_dist and df_season is None:
        raise ValueError("Для отрисовки месячных распределений необходимо передать df_season")
    
    if track_to_cyclone is None:
        plot_TC = False

    tropical_ids = set(track_to_cyclone.keys()) if track_to_cyclone else set()

    # 1. Для каждого кластера считаем число ТЦ и собираем его ТЦ‑треки
    cluster_tc_counts = {cluster: 0 for cluster in range(n_clusters)}
    cluster_tropical_tracks = {cluster: [] for cluster in range(n_clusters)}

    for cluster, tracks in clusters_full_tracks.items():
        for track in tracks:
            if not isinstance(track, pd.DataFrame):
                continue
            if 'track_id' not in track.columns:
                continue
            if 'x' not in track.columns or 'y' not in track.columns:
                continue

            track_id = track['track_id'].iloc[0]
            if track_id < 0:
                continue

            if track_id in tropical_ids:
                cluster_tropical_tracks[cluster].append(track)
                cluster_tc_counts[cluster] += 1

    # 2. Подготовка данных для месячных распределений (если нужно)
    if plot_monthly_dist:
        # Добавляем колонку с месяцем, если её нет
        if 'Month' not in df_season.columns:
            df_season['Month'] = df_season['datetime'].dt.strftime('%b')
        
        # Группируем и считаем % по месяцам для каждого кластера
        df_grouped = df_season.groupby(['cluster', 'Month']).size().unstack(fill_value=0)
        df_grouped = df_grouped.reindex(columns=months_order, fill_value=0)
        df_percent = df_grouped.div(df_grouped.sum(axis=1), axis=0) * 100

    # 3. Строим сетку графиков
    if plot_monthly_dist:
        # Каждый кластер занимает 2 строки: карта + гистограмма
        cols = min(3, n_clusters)  # максимум 3 столбца для лучшей читаемости
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

    # 4. Цикл по кластерам
    for cluster_idx in range(n_clusters):
        if plot_monthly_dist:
            # Получаем позиции для карты и гистограммы
            r = cluster_idx // cols
            c = cluster_idx % cols
            ax_map = fig.add_subplot(gs[2 * r, c])
            ax_bar = fig.add_subplot(gs[2 * r + 1, c])
        else:
            ax_map = axes[cluster_idx]
            ax_bar = None

        tracks = clusters_full_tracks.get(cluster_idx, [])

        if not tracks:
            if plot_monthly_dist:
                ax_map.axis('off')
                ax_bar.axis('off')
            else:
                ax_map.axis('off')
            continue

        # Отрисовка карты
        plot_ground(ax_map, ground)

        # Отрисовка обычных треков
        for track in tracks:
            if not isinstance(track, pd.DataFrame):
                continue
            if 'x' not in track.columns or 'y' not in track.columns:
                continue

            x = track['x'].values
            y = track['y'].values

            if track_type == 'track':
                ax_map.plot(
                    x, y,
                    color=colors(cluster_idx) if callable(colors) else colors[cluster_idx % len(colors)],
                    alpha=0.3,
                    linewidth=1.0
                )
            elif track_type == 'start':
                ax_map.scatter(
                    x[0], y[0],
                    color=colors(cluster_idx) if callable(colors) else colors[cluster_idx % len(colors)],
                    alpha=0.5,
                    s=5
                )

        # Отрисовка ТЦ‑треков
        if plot_TC and cluster_tropical_tracks[cluster_idx]:
            for track in cluster_tropical_tracks[cluster_idx]:
                if not isinstance(track, pd.DataFrame):
                    continue
                if 'x' not in track.columns or 'y' not in track.columns:
                    continue
                x = track['x'].values
                y = track['y'].values
                ax_map.plot(
                    x, y,
                    color='black',
                    linewidth=1.0,
                    zorder=10
                )

        # Настройки карты
        n_tc = cluster_tc_counts[cluster_idx]
        title = f'{len(tracks)} треков'
        if plot_TC:
            title += f' ({n_tc} TC)'
        
        # Подпись на карте
        ax_map.text(0.02, 0.96, title,
                    transform=ax_map.transAxes,
                    fontsize=10,
                    fontweight='bold',
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.85, edgecolor='none'))

        ax_map.set_xlabel('Долгота')
        ax_map.set_ylabel('Широта')
        if data_type == 'ERA5':
            ax_map.invert_yaxis()
        ax_map.grid(True, alpha=0.3)

        # Легенда для карты
        from matplotlib.lines import Line2D
        legend_elements = []
        
        # Цвет кластера
        legend_elements.append(Line2D([0], [0], color=colors(cluster_idx) if callable(colors) 
                                     else colors[cluster_idx % len(colors)], lw=2, label=f'Cluster {cluster_idx}'))
        if plot_TC and cluster_tropical_tracks[cluster_idx]:
            legend_elements.append(Line2D([0], [0], color='black', lw=2, label='Tropical Cyclone'))
        
        ax_map.legend(handles=legend_elements, loc='upper right', fontsize=8)

        # Отрисовка гистограммы по месяцам
        if plot_monthly_dist and cluster_idx in df_percent.index:
            freqs = df_percent.loc[cluster_idx].reindex(months_order, fill_value=0)
            bar_color = colors(cluster_idx) if callable(colors) else colors[cluster_idx % len(colors)]
            
            bars = ax_bar.bar(months_order, freqs.values,
                             color=bar_color, edgecolor='black', linewidth=0.5, alpha=0.9)

            # Подписи значений на столбцах
            for bar, height in zip(bars, freqs.values):
                if height > 0.5:
                    ax_bar.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                               f'{height:.1f}', ha='center', va='bottom', fontsize=7)

            # Настройки гистограммы
            ax_bar.tick_params(axis='x', labelrotation=45, labelsize=8)
            ax_bar.set_ylabel('Freq (%)', fontsize=9)
            ax_bar.set_ylim(0, min(35, freqs.max() * 1.2))  # Динамический лимит
            ax_bar.grid(True, alpha=0.3, axis='y')
            
            # Подпись для гистограммы
            ax_bar.set_title(f'Распределение по месяцам', fontsize=9, pad=2)
        elif plot_monthly_dist:
            ax_bar.axis('off')

    # 5. Обработка пустых подграфиков
    if plot_monthly_dist:
        # Для GridSpec ничего делать не нужно
        pass
    else:
        for i in range(len(clusters_full_tracks), len(axes)):
            axes[i].axis('off')

    # 6. Общий заголовок
    if plot_monthly_dist:
        plt.suptitle(f'{season.capitalize()} сезон: треки и распределение по месяцам', 
                    fontsize=14, fontweight='bold')
    else:
        plt.suptitle(f'{season.capitalize()} сезон: треки по кластерам', 
                    fontsize=14, fontweight='bold')

    plt.tight_layout()

    # 7. Сохранение
    if clustering_type == 'hdbscan':
        output_dir = f"{path_data_tracks}/cluster_pics/nclusters_{n_clusters}_mcs{clustering_params[0]}_ms{clustering_params[1]}"
        base_filename = f"nclusters_{n_clusters}_mcs{clustering_params[0]}_ms{clustering_params[1]}_{season}_{track_type}"
    else:
        output_dir = f"{path_data_tracks}/cluster_pics/nclusters_{n_clusters}"
        base_filename = f"nclusters_{n_clusters}_{season}_{track_type}"

    if plot_monthly_dist:
        filename = f"{output_dir}/{base_filename}_with_monthly_dist.png"
    else:
        filename = f"{output_dir}/{base_filename}_all_clusters.png"

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_tracks_by_clusters_all_with_TC(
    clusters_full_tracks,
    n_clusters,
    colors,
    ground,
    path_data_tracks,
    season,
    data_type='ERA5',
    track_type='track',
    clustering_type='kmeans',
    clustering_params=None,
    plot_TC=False,
    track_to_cyclone=None,
    plot_monthly_dist=False,  # Новый параметр для отрисовки месячных распределений
    df_season=None,  # DataFrame с данными для месячных распределений
    months_order=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    monthly_mode='relative',  # 'relative' или 'absolute'
):
    """
    Визуализация треков по кластерам в сетке подграфиков.
    
    Параметры:
    ----------
    plot_monthly_dist : bool
        Если True, отрисовывает гистограммы распределения по месяцам для каждого кластера
    df_season : pandas.DataFrame, optional
        DataFrame с колонками ['cluster', 'datetime'] для построения месячных распределений
    months_order : list
        Порядок месяцев для отображения на гистограммах
    monthly_mode : {'relative', 'absolute'}
        'relative' — проценты внутри кластера (как сейчас),
        'absolute' — абсолютное число треков; ось Y нормируется на максимум по всем кластерам
    """
    if plot_monthly_dist and df_season is None:
        raise ValueError("Для отрисовки месячных распределений необходимо передать df_season")
    
    if monthly_mode not in ('relative', 'absolute'):
        raise ValueError("monthly_mode должен быть 'relative' или 'absolute'")
    
    if track_to_cyclone is None:
        plot_TC = False

    tropical_ids = set(track_to_cyclone.keys()) if track_to_cyclone else set()

    # 1. Для каждого кластера считаем число ТЦ и собираем его ТЦ‑треки
    cluster_tc_counts = {cluster: 0 for cluster in range(n_clusters)}
    cluster_tropical_tracks = {cluster: [] for cluster in range(n_clusters)}

    for cluster, tracks in clusters_full_tracks.items():
        for track in tracks:
            if not isinstance(track, pd.DataFrame):
                continue
            if 'track_id' not in track.columns:
                continue
            if 'x' not in track.columns or 'y' not in track.columns:
                continue

            track_id = track['track_id'].iloc[0]
            if track_id < 0:
                continue

            if track_id in tropical_ids:
                cluster_tropical_tracks[cluster].append(track)
                cluster_tc_counts[cluster] += 1

    # 2. Подготовка данных для месячных распределений (если нужно)
    if plot_monthly_dist:
        # Добавляем колонку с месяцем, если её нет
        if 'Month' not in df_season.columns:
            df_season['Month'] = df_season['datetime'].dt.strftime('%b')
        
        # Группируем: абсолютные числа по месяцам
        df_grouped = df_season.groupby(['cluster', 'Month']).size().unstack(fill_value=0)
        df_grouped = df_grouped.reindex(columns=months_order, fill_value=0)

        if monthly_mode == 'relative':
            # Проценты внутри кластера
            df_percent = df_grouped.div(df_grouped.sum(axis=1), axis=0) * 100
            df_plot = df_percent
            y_label = 'Freq (%)'
            # max_y для относительного: по каждому кластеру отдельно (как у тебя сейчас, динамика ниже)
            global_max = None
        else:
            # Абсолютные значения, нормировка оси Y на максимум по всем кластерам
            df_plot = df_grouped
            y_label = 'Count'
            global_max = df_plot.values.max() if df_plot.size > 0 else 0

    # 3. Строим сетку графиков
    if plot_monthly_dist:
        # Каждый кластер занимает 2 строки: карта + гистограмма
        cols = min(3, n_clusters)  # максимум 3 столбца для лучшей читаемости
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

    # 4. Цикл по кластерам
    for cluster_idx in range(n_clusters):
        if plot_monthly_dist:
            # Получаем позиции для карты и гистограммы
            r = cluster_idx // cols
            c = cluster_idx % cols
            ax_map = fig.add_subplot(gs[2 * r, c])
            ax_bar = fig.add_subplot(gs[2 * r + 1, c])
        else:
            ax_map = axes[cluster_idx]
            ax_bar = None

        tracks = clusters_full_tracks.get(cluster_idx, [])

        if not tracks:
            if plot_monthly_dist:
                ax_map.axis('off')
                ax_bar.axis('off')
            else:
                ax_map.axis('off')
            continue

        # Отрисовка карты
        plot_ground(ax_map, ground)

        # Отрисовка обычных треков
        for track in tracks:
            if not isinstance(track, pd.DataFrame):
                continue
            if 'x' not in track.columns or 'y' not in track.columns:
                continue

            x = track['x'].values
            y = track['y'].values

            if track_type == 'track':
                ax_map.plot(
                    x, y,
                    color=colors(cluster_idx) if callable(colors) else colors[cluster_idx % len(colors)],
                    alpha=0.3,
                    linewidth=1.0
                )
            elif track_type == 'start':
                ax_map.scatter(
                    x[0], y[0],
                    color=colors(cluster_idx) if callable(colors) else colors[cluster_idx % len(colors)],
                    alpha=0.5,
                    s=5
                )

        # Отрисовка ТЦ‑треков
        if plot_TC and cluster_tropical_tracks[cluster_idx]:
            for track in cluster_tropical_tracks[cluster_idx]:
                if not isinstance(track, pd.DataFrame):
                    continue
                if 'x' not in track.columns or 'y' not in track.columns:
                    continue
                x = track['x'].values
                y = track['y'].values
                ax_map.plot(
                    x, y,
                    color='black',
                    linewidth=1.0,
                    zorder=10
                )

        # Настройки карты
        n_tc = cluster_tc_counts[cluster_idx]
        title = f'{len(tracks)} треков'
        if plot_TC:
            title += f' ({n_tc} TC)'
        
        # Подпись на карте
        ax_map.text(0.02, 0.96, title,
                    transform=ax_map.transAxes,
                    fontsize=10,
                    fontweight='bold',
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.85, edgecolor='none'))

        ax_map.set_xlabel('Долгота')
        ax_map.set_ylabel('Широта')
        if data_type == 'ERA5':
            ax_map.invert_yaxis()
        ax_map.grid(True, alpha=0.3)

        # Легенда для карты
        from matplotlib.lines import Line2D
        legend_elements = []
        
        # Цвет кластера
        legend_elements.append(Line2D([0], [0], color=colors(cluster_idx) if callable(colors) 
                                     else colors[cluster_idx % len(colors)], lw=2, label=f'Cluster {cluster_idx}'))
        if plot_TC and cluster_tropical_tracks[cluster_idx]:
            legend_elements.append(Line2D([0], [0], color='black', lw=2, label='Tropical Cyclone'))
        
        ax_map.legend(handles=legend_elements, loc='upper right', fontsize=8)

        # Отрисовка гистограммы по месяцам
        if plot_monthly_dist and cluster_idx in df_plot.index:
            freqs = df_plot.loc[cluster_idx].reindex(months_order, fill_value=0)
            bar_color = colors(cluster_idx) if callable(colors) else colors[cluster_idx % len(colors)]
            
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

    # 5. Обработка пустых подграфиков
    if plot_monthly_dist:
        pass
    else:
        for i in range(len(clusters_full_tracks), len(axes)):
            axes[i].axis('off')

    # 6. Общий заголовок
    if plot_monthly_dist:
        plt.suptitle(f'{season.capitalize()} сезон: треки и распределение по месяцам', 
                    fontsize=14, fontweight='bold')
    else:
        plt.suptitle(f'{season.capitalize()} сезон: треки по кластерам', 
                    fontsize=14, fontweight='bold')

    plt.tight_layout()

    # 7. Сохранение
    if clustering_type == 'hdbscan':
        output_dir = f"{path_data_tracks}/cluster_pics/nclusters_{n_clusters}_mcs{clustering_params[0]}_ms{clustering_params[1]}"
        base_filename = f"nclusters_{n_clusters}_mcs{clustering_params[0]}_ms{clustering_params[1]}_{season}_{track_type}"
    else:
        output_dir = f"{path_data_tracks}/cluster_pics/nclusters_{n_clusters}"
        base_filename = f"nclusters_{n_clusters}_{season}_{track_type}"

    if plot_monthly_dist:
        suffix = '_monthly_rel' if monthly_mode == 'relative' else '_monthly_abs'
        filename = f"{output_dir}/{base_filename}_with_monthly_dist{suffix}.png"
    else:
        filename = f"{output_dir}/{base_filename}_all_clusters.png"

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()



def plot_cluster_boxplots(df, numeric_cols, n_clusters, path_data_tracks, season):
    """
    Рисует отдельные boxplot-графики для каждой из numeric_cols,
    где каждый график показывает распределение значений по кластерам.
    """
    output_dir = f"{path_data_tracks}/cluster_pics/nclusters_{n_clusters}"
    os.makedirs(output_dir, exist_ok=True)

    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='cluster', y=col, data=df, palette="Set3")
        plt.title(f'{season}: Распределение "{col}" по кластерам')
        plt.xlabel('Кластер')
        plt.ylabel('Значение')
        plt.xticks(range(n_clusters))
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(f"{output_dir}/nclusters_{n_clusters}_{season}_boxplot_{col}.png")
        plt.close()



def plot_cluster_boxplots_all(df, numeric_cols, n_clusters, colors, path_data_tracks, season, clustering_type='kmeans', clustering_params=None):
    """
    Рисует один общий график (subplots) с boxplot'ами для всех numeric_cols,
    где каждый subplot показывает распределение значений по кластерам.
    Используется заданная цветовая палитра.
    """
    if clustering_type == 'hdbscan':
        output_dir = f"{path_data_tracks}/cluster_pics/nclusters_{n_clusters}_mcs{clustering_params[0]}_ms{clustering_params[1]}"
        plot_path = f"{output_dir}/nclusters_{n_clusters}_mcs{clustering_params[0]}_ms{clustering_params[1]}_{season}_all_boxplots.png"
    else:
        output_dir = f"{path_data_tracks}/cluster_pics/nclusters_{n_clusters}"
        plot_path = f"{output_dir}/nclusters_{n_clusters}_{season}_all_boxplots.png"
        
        

    # Задаём палитру цветов
    # colors = plt.cm.get_cmap('tab20', n_clusters)
    palette = [colors(i) for i in range(n_clusters)]

    # Определяем размер сетки графиков
    n_cols_subplot = 4  # количество графиков в строке
    n_rows_subplot = (len(numeric_cols) + n_cols_subplot - 1) // n_cols_subplot

    fig, axes = plt.subplots(n_rows_subplot, n_cols_subplot, figsize=(25, 5 * n_rows_subplot))
    axes = axes.flatten()  # делаем массив axes плоским для удобного доступа

    for ax, col in zip(axes, numeric_cols):
        sns.boxplot(x='cluster', y=col, data=df, palette=palette, ax=ax)
        ax.set_title(f'{season}: Распределение "{col}" по кластерам')
        ax.set_xlabel('Кластер')
        ax.set_ylabel('Значение')
        ax.set_xticks(range(n_clusters))
        ax.grid(True)

    # Убираем лишние пустые subplot'ы, если они есть
    for ax in axes[len(numeric_cols):]:
        ax.axis('off')

    plt.tight_layout()
    

        
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(plot_path)
    plt.close()


#### metrics ####

def compute_silhouette(X, labels):
    """Silhouette Width — максимизировать"""
    if len(np.unique(labels)) < 2:
        return -1  # Невозможно посчитать при одном кластере
    return silhouette_score(X, labels)


def compute_connectivity(X, labels, L=20):
    """Connectivity — минимизировать
    X: признаковое пространство (n_samples, n_features)
    labels: метки кластеров
    L: количество ближайших соседей для проверки
    """
    # Вычисляем матрицу расстояний
    dist_matrix = squareform(pdist(X, metric='euclidean'))
    n_samples = X.shape[0]
    connectivity = 0.0

    for i in range(n_samples):
        # Получаем L ближайших соседей (исключая сам объект)
        neighbors = np.argsort(dist_matrix[i])[1:L+1]
        # Проверяем, в одном ли кластере с текущим объектом
        for nn in neighbors:
            if labels[nn] != labels[i]:
                rank = np.where(np.argsort(dist_matrix[i]) == nn)[0][0]
                connectivity += 1.0 / rank

    return connectivity


def compute_huberts_g_statistic(X, labels):
    """Hubert’s G Statistic — максимизировать
    Корреляция между исходной матрицей расстояний и идеальной (0 если в одном кластере, 1 иначе)
    """
    dist_matrix = squareform(pdist(X, metric='euclidean'))
    n = len(labels)
    # Создаём идеальную матрицу: 0 если в одном кластере, 1 иначе
    cluster_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cluster_matrix[i, j] = 0 if labels[i] == labels[j] else 1

    # Убираем диагональ (или можно оставить — pdist её не включает)
    triu_indices = np.triu_indices(n, k=1)
    g_statistic, _ = pearsonr(dist_matrix[triu_indices], cluster_matrix[triu_indices])
    return g_statistic


def compute_stability(X, labels, n_subsamples=20, subsample_size=0.8, n_clusters=10):
    """Stability — максимизировать
    Стабильность кластеров через ARI между подвыборками и исходной кластеризацией
    """
    n_samples = X.shape[0]
    ari_scores = []

    for _ in range(n_subsamples):
        # Случайная подвыборка с перекрытием
        idx = np.random.choice(n_samples, size=int(n_samples * subsample_size), replace=True)
        X_sub = X[idx]
        labels_sub = labels[idx]

        # Перекластеризуем подвыборку (с тем же числом кластеров)
        from sklearn.cluster import KMeans
        kmeans_sub = KMeans(n_clusters=n_clusters, n_init=10, random_state=None)
        labels_sub_pred = kmeans_sub.fit_predict(X_sub)

        # Сравниваем с "истинными" метками на подвыборке (из исходной кластеризации)
        ari = adjusted_rand_score(labels_sub, labels_sub_pred)
        ari_scores.append(ari)

    return np.mean(ari_scores)