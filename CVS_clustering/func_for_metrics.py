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

from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import cdist

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


def get_ordered_metrics(metrics_df):
    """Фиксированный порядок метрик: max → min → Class/Repr"""
    maximize_metrics = [
        'silhouette',
        'hubert_g',
        'stability',
        'calinski_harabasz',
        'classifiability',
        'reproducibility',
    ]
    minimize_metrics = ['connectivity', 'davies_bouldin']

    all_metrics = maximize_metrics + minimize_metrics
    available = [m for m in all_metrics if m in metrics_df.columns]
    return available

def get_pvalue_colors(data, metric):
    """
    Возвращает массив цветов для точек по p-value.
      p <= 0.05        — тёмно-зелёный
      0.05 < p <= 0.10 — бледно-зелёный
      p > 0.10         — серый
    Ожидает колонку '<metric>_p'.
    """
    p_col = f"{metric}_p"
    if p_col not in data.columns:
        # если p-value нет, всё рисуем одним цветом
        return None

    p_vals = data[p_col].values
    colors = np.empty(p_vals.shape, dtype=object)

    colors[p_vals > 0.10] = 'gray'
    colors[(p_vals > 0.05) & (p_vals <= 0.10)] = '#90EE90'  # бледно-зелёный
    colors[p_vals <= 0.05] = 'green'                        # тёмно-зелёный

    return colors

def plot_clustering_metrics(metrics_df, input_season, output_dir, clustering_type, scaler_type, save=True):
    """Универсальная визуализация метрик"""
    sns.set_style("whitegrid")
    
    # GMM-специфичные метрики (если есть)
    gmm_metrics = ['aic', 'bic', 'log_likelihood'] if 'aic' in metrics_df.columns else []
    
    # Фиксированный порядок общих метрик
    common_metrics = get_ordered_metrics(metrics_df)
    
    if input_season == 'all':
        seasons_to_plot = ['winter', 'summer', 'year']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        # Сводный график: 2 строки × 4 столбца (макс → мин → WSS/BSS)
        fig = plt.figure(figsize=(18, 8))
        plot_metrics_subplot(fig, metrics_df, seasons_to_plot, colors, common_metrics)
        plt.suptitle(f'Clustering Metrics - {clustering_type.upper()} (All Seasons)', fontsize=16)
        plt.tight_layout()
        if save:
            plt.savefig(f"{output_dir}/metrics_{clustering_type}_{scaler_type}_all_seasons.png",
                        dpi=300, bbox_inches='tight')
            plt.close()
        
        # Отдельные по сезонам
        for season_name in seasons_to_plot:
            plot_season_metrics(metrics_df, season_name, output_dir,
                                clustering_type, scaler_type, gmm_metrics, common_metrics)
    else:
        plot_season_metrics(metrics_df, input_season, output_dir,
                            clustering_type, scaler_type, gmm_metrics, common_metrics)


def plot_metrics_subplot(fig, metrics_df, seasons_to_plot, colors, common_metrics):
    """Подграфики для метрик (сводный по сезонам, 2×4)"""
    all_metrics = common_metrics.copy()
    
    nrows, ncols = 2, 4
    max_plots = nrows * ncols
    
    for i, metric in enumerate(all_metrics[:max_plots]):
        plt.subplot(nrows, ncols, i + 1)
        
        for season_name, color in zip(seasons_to_plot, colors):
            data_season = metrics_df[metrics_df['season'] == season_name]
            if metric not in data_season.columns:
                continue

            # обычные метрики — как раньше
            if metric not in ['classifiability', 'reproducibility']:
                plt.plot(data_season['n_clusters'], data_season[metric],
                         marker='o', label=season_name.capitalize(),
                         color=color, linewidth=2)
            else:
                # для classifiability/reproducibility берём цвет точки по p-value
                p_colors = get_pvalue_colors(data_season, metric)
                if p_colors is None:
                    # p-value нет: fallback на обычную линию
                    plt.plot(data_season['n_clusters'], data_season[metric],
                             marker='o', label=season_name.capitalize(),
                             color=color, linewidth=2)
                else:
                    # рисуем линию серым фоном + точки по p-value
                    x = data_season['n_clusters'].values
                    y = data_season[metric].values

                    # тонкая линия (чтобы видеть тренд)
                    plt.plot(x, y, '-', color='lightgray', linewidth=1)

                    # точки: цвет определяется p-value
                    for xi, yi, ci in zip(x, y, p_colors):
                        plt.scatter(xi, yi,
                                    color=ci,
                                    edgecolor='none',
                                    s=25,
                                    label=season_name.capitalize() if ci == 'black' and xi == x[0] else None)

        plt.xlabel('Number of Clusters')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(metric.replace('_', ' ').title())
        plt.xticks(sorted(metrics_df['n_clusters'].unique()))
        plt.grid(True, alpha=0.3)
        if i == 0:
            plt.legend()


def plot_season_metrics(metrics_df, season_name, output_dir,
                        clustering_type, scaler_type, gmm_metrics, common_metrics):
    """График для одного сезона: 2×4 + (опц.) AIC/BIC"""
    data = metrics_df[metrics_df['season'] == season_name]
    
    base_metrics = [m for m in common_metrics if m in data.columns]
    
    extra_metrics = []
    if gmm_metrics:
        for m in ['aic', 'bic']:
            if m in data.columns:
                extra_metrics.append(m)
    
    all_metrics = base_metrics + extra_metrics
    if len(all_metrics) == 0:
        return
    
    n_metrics = len(all_metrics)
    ncols = 4
    nrows = int(np.ceil(n_metrics / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).ravel()
    
    for i, metric in enumerate(all_metrics):
        if metric not in data.columns:
            continue

        ax = axes[i]
        x = data['n_clusters'].values
        y = data[metric].values

        if metric not in ['classifiability', 'reproducibility']:
            ax.plot(x, y, marker='o', color='#1f77b4', linewidth=2)
        else:
            p_colors = get_pvalue_colors(data, metric)
            if p_colors is None:
                ax.plot(x, y, marker='o', color='#1f77b4', linewidth=2)
            else:
                # тонкая линия для тренда
                ax.plot(x, y, '-', color='lightgray', linewidth=1)
                # точки по p-value
                for xi, yi, ci in zip(x, y, p_colors):
                    ax.scatter(xi, yi, color=ci, edgecolor='none', s=25)

        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f"{metric.replace('_', ' ').title()} - {season_name.capitalize()}")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted(data['n_clusters'].unique()))
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_{clustering_type}_{scaler_type}.png",
                dpi=300, bbox_inches='tight')
    plt.close()


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

        
#### metrics ####

# def compute_metrics(X, labels, gmm_metrics=None, clustering_type='gmm'):
#     """Обновленные метрики с учетом GMM + DB + CH + WSS/BSS"""
#     valid_idx = ~(np.isnan(labels) | np.any(np.isnan(X), axis=1))
#     X_clean = X[valid_idx]
#     labels_clean = labels[valid_idx].astype(int)

#     wss, bss = compute_within_between_ss(X_clean, labels_clean)

#     metrics = {
#         'silhouette': compute_silhouette(X_clean, labels_clean),
#         'connectivity': compute_connectivity(X_clean, labels_clean, L=20),
#         'hubert_g': compute_huberts_g_statistic(X_clean, labels_clean),
#         'stability': compute_stability(
#             X_clean, labels_clean,
#             n_subsamples=20,
#             n_clusters=len(np.unique(labels_clean))
#         ),
#         'davies_bouldin': compute_davies_bouldin(X_clean, labels_clean),
#         'calinski_harabasz': compute_calinski_harabasz(X_clean, labels_clean),
#         'within_ss': wss,
#         'between_ss': bss
#     }

#     if gmm_metrics:
#         metrics.update(gmm_metrics)

#     return metrics

def compute_metrics(X, labels, gmm_metrics=None,
                    clustering_type='gmm',
                    n_clusters=None,
                    noise_M_class=20,
                    noise_R_repr=20,
                    noise_M_repr=10,
                    noise_N=50,
                    random_state=None,
                    compute_classifiability_flag=False,
                    compute_reproducibility_flag=False):
    """
    Обновленные метрики:
    + classifiability / reproducibility
    + их p-values относительно Gaussian-шума.
    Флаги compute_classifiability_flag и compute_reproducibility_flag
    позволяют отключать соответствующие расчёты.
    """
    valid_idx = ~(np.isnan(labels) | np.any(np.isnan(X), axis=1))
    X_clean = X[valid_idx]
    labels_clean = labels[valid_idx].astype(int)

    if n_clusters is None:
        n_clusters = len(np.unique(labels_clean))

    wss, bss = compute_within_between_ss(X_clean, labels_clean)

    # Базовые метрики
    metrics = {
        'silhouette': compute_silhouette(X_clean, labels_clean),
        'connectivity': compute_connectivity(X_clean, labels_clean, L=20),
        'hubert_g': compute_huberts_g_statistic(X_clean, labels_clean),
        'stability': compute_stability(
            X_clean, labels_clean,
            n_subsamples=20,
            n_clusters=n_clusters
        ),
        'davies_bouldin': compute_davies_bouldin(X_clean, labels_clean),
        'calinski_harabasz': compute_calinski_harabasz(X_clean, labels_clean),
        'within_ss': wss,
        'between_ss': bss,
    }

    # Индекс классифицируемости + p-value (по флагу)
    if compute_classifiability_flag:
        classif_val, classif_p = compute_classifiability_with_noise(
            X_clean, n_clusters,
            M=noise_M_class, N_noise=noise_N,
            random_state=random_state
        )
        metrics['classifiability'] = classif_val
        metrics['classifiability_p'] = classif_p

    # Индекс воспроизводимости + p-value (по флагу)
    if compute_reproducibility_flag:
        repr_val, repr_p = compute_reproducibility_with_noise(
            X_clean, n_clusters,
            R=noise_R_repr, M=noise_M_repr,
            N_noise=noise_N,
            subsample_frac=0.5,
            random_state=random_state
        )
        metrics['reproducibility'] = repr_val
        metrics['reproducibility_p'] = repr_p

    # GMM-метрики, если есть
    if gmm_metrics:
        metrics.update(gmm_metrics)

    return metrics



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

def compute_davies_bouldin(X, labels):
    """Davies–Bouldin — минимизировать"""
    if len(np.unique(labels)) < 2:
        return np.nan
    return davies_bouldin_score(X, labels)


def compute_within_between_ss(X, labels):
    """
    Внутри- и межкластерная сумма квадратов (WSS, BSS)
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 1:
        return np.nan, np.nan

    overall_center = X.mean(axis=0)
    within_ss = 0.0
    between_ss = 0.0

    for lab in unique_labels:
        cluster_points = X[labels == lab]
        if cluster_points.shape[0] == 0:
            continue
        center = cluster_points.mean(axis=0)

        # WSS
        dists = cdist(cluster_points, center[None, :], metric='euclidean')
        within_ss += (dists ** 2).sum()

        # BSS
        center_dist = np.linalg.norm(center - overall_center)
        between_ss += cluster_points.shape[0] * (center_dist ** 2)

    return within_ss, between_ss


def compute_calinski_harabasz(X, labels):
    """Calinski–Harabasz — максимизировать"""
    if len(np.unique(labels)) < 2:
        return np.nan
    return calinski_harabasz_score(X, labels)


from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import itertools

def compute_classifiability_index(X, n_clusters, M=20, random_state=None):
    """
    c*(K) = средний ARI между M k-means разбиениями (разные инициализации).
    Максимизировать.
    """
    X = np.asarray(X)
    n_samples = X.shape[0]
    if n_clusters < 2 or n_clusters >= n_samples:
        return np.nan

    rng = np.random.RandomState(random_state)
    labels_list = []

    for m in range(M):
        rs = None if random_state is None else rng.randint(0, 10**9)
        km = KMeans(
            n_clusters=n_clusters,
            n_init=1,
            init="k-means++",
            random_state=rs
        )
        labels_list.append(km.fit_predict(X))

    ari_vals = []
    for i, j in itertools.combinations(range(len(labels_list)), 2):
        ari_vals.append(adjusted_rand_score(labels_list[i], labels_list[j]))

    return np.mean(ari_vals) if ari_vals else np.nan


def evaluate_significance_vs_noise_max(real_value, noise_values):
    """
    p-value для метрики, которую максимизируем:
    p = доля шумовых значений >= real_value.
    """
    noise_values = np.asarray(noise_values)
    return np.mean(noise_values >= real_value)


def compute_classifiability_with_noise(X_real, n_clusters,
                                       M=20, N_noise=50,
                                       random_state=None):
    rng = np.random.RandomState(random_state)

    c_real = compute_classifiability_index(
        X_real, n_clusters, M=M, random_state=rng.randint(0, 10**9)
    )

    noise_vals = []
    for _ in tqdm(range(N_noise), desc=f"class_noise K={n_clusters}", leave=False):
        X_noise = generate_gaussian_noise_like_X(X_real, rng=rng)
        c_noise = compute_classifiability_index(
            X_noise, n_clusters, M=M, random_state=rng.randint(0, 10**9)
        )
        noise_vals.append(c_noise)

    c_real_p = evaluate_significance_vs_noise_max(c_real, noise_vals)
    return c_real, c_real_p

def _best_kmeans_partition(X, n_clusters, M=20, random_state=None):
    """
    Эталонное разбиение: берем запуск с минимальной inertia.
    """
    X = np.asarray(X)
    rng = np.random.RandomState(random_state)
    best_inertia = np.inf
    best_labels = None

    for m in range(M):
        rs = None if random_state is None else rng.randint(0, 10**9)
        km = KMeans(
            n_clusters=n_clusters,
            n_init=1,
            init="k-means++",
            random_state=rs
        )
        km.fit(X)
        if km.inertia_ < best_inertia:
            best_inertia = km.inertia_
            best_labels = km.labels_.copy()

    return best_labels


def compute_reproducibility_index(X, n_clusters,
                                  R=20, M=10,
                                  subsample_frac=0.5,
                                  random_state=None):
    """
    Средний ARI между эталонным разбиением полной выборки
    и эталонными разбиениями R подвыборок (50% точек).
    Максимизировать.
    """
    X = np.asarray(X)
    n_samples = X.shape[0]
    if n_clusters < 2 or n_clusters >= n_samples:
        return np.nan

    rng = np.random.RandomState(random_state)

    # Эталон на полной выборке
    labels_ref = _best_kmeans_partition(
        X, n_clusters, M=M,
        random_state=rng.randint(0, 10**9)
    )

    ari_vals = []
    for _ in range(R):
        idx = rng.choice(n_samples,
                         size=int(n_samples * subsample_frac),
                         replace=False)
        X_sub = X[idx]

        labels_sub_best = _best_kmeans_partition(
            X_sub, n_clusters, M=M,
            random_state=rng.randint(0, 10**9)
        )

        labels_ref_sub = labels_ref[idx]
        ari_vals.append(adjusted_rand_score(labels_ref_sub, labels_sub_best))

    return np.mean(ari_vals) if ari_vals else np.nan


def compute_reproducibility_with_noise(X_real, n_clusters,
                                       R=20, M=10, N_noise=50,
                                       subsample_frac=0.5,
                                       random_state=None):
    rng = np.random.RandomState(random_state)

    r_real = compute_reproducibility_index(
        X_real, n_clusters, R=R, M=M,
        subsample_frac=subsample_frac,
        random_state=rng.randint(0, 10**9)
    )

    noise_vals = []
    for _ in tqdm(range(N_noise), desc=f"repr_noise K={n_clusters}", leave=False):
        X_noise = generate_gaussian_noise_like_X(X_real, rng=rng)
        r_noise = compute_reproducibility_index(
            X_noise, n_clusters, R=R, M=M,
            subsample_frac=subsample_frac,
            random_state=rng.randint(0, 10**9)
        )
        noise_vals.append(r_noise)

    r_real_p = evaluate_significance_vs_noise_max(r_real, noise_vals)
    return r_real, r_real_p


def generate_gaussian_noise_like_X(X, rng=None):
    """
    Шумовая выборка той же размерности, что X:
    многомерное нормальное распределение с тем же mean и cov.
    X: (n_samples, n_features)
    """
    X = np.asarray(X)
    n_samples, n_features = X.shape
    if rng is None:
        rng = np.random.RandomState()

    mu = np.mean(X, axis=0)               # (n_features,)
    Sigma = np.cov(X, rowvar=False)       # (n_features, n_features)

    # Генерируем multivariate normal
    X_noise = rng.multivariate_normal(mu, Sigma, size=n_samples)
    return X_noise