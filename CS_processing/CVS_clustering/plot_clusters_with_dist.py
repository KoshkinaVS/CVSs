import cartopy.crs as ccrs
from scipy.stats import binned_statistic_2d
import math
from matplotlib.colors import ListedColormap

from func_for_CVS_clusters import *

data_type = 'LoRes'
sigma = 2

# Загрузка треков из файлов по кластерам
def load_tracks_from_paths(df, cluster_col='cluster', path_col='path'):
    clusters_full_tracks = {}
    for cluster in tqdm(df[cluster_col].unique()):
        tracks = []
        cluster_df = df[df[cluster_col] == cluster]
        for _, row in cluster_df.iterrows():
            # Загрузка трека из файла (предполагаем, что файл содержит колонки 'lat', 'lon')
            track_df = pd.read_csv(row[path_col])
            tracks.append(track_df[['lat', 'lon']].values.T)  # (lat, lon)
        clusters_full_tracks[cluster] = tracks
    return clusters_full_tracks


path_init = f'/storage/thalassa/users/vkoshkina'
path_data = f'{path_init}/data'  
path_data_tracks = f'{path_data}/TC_tracks/{data_type}/{data_type}_sigma_{sigma}'
output_dir = f"{path_data_tracks}/cluster_results_august"

input_season = 'year'
n_clusters = 9

df_season = pd.read_csv(f"{output_dir}/cluster_tables/{input_season}_nclusters_{n_clusters}_tracks.csv", parse_dates=['datetime'])
df_season['Month'] = df_season['datetime'].dt.strftime('%b').str[:3]  # или просто J, F, M...

# Загрузка треков
clusters_full_tracks = load_tracks_from_paths(df_season)


# --- ОБЩИЕ ПАРАМЕТРЫ ---
colors = [
    '#e41a1c',  # красный
    '#377eb8',  # синий
    '#723FDA',  # фиолетовый
    '#17DEC7',  # циан
    '#BD0FB7',  # фуксия
    '#ff7f00',  # оранжевый
    '#A6DA3F',  # жёлтый
    '#f781bf',  # розовый
    '#096C39',  # зелёный
]

# Параметры карты
lon_min, lon_max = -95, -13
lat_min, lat_max = 4, 79
cell_size = 1
lon_bins = np.arange(lon_min, lon_max + cell_size, cell_size)
lat_bins = np.arange(lat_min, lat_max + cell_size, cell_size)


# --- ПОДГОТОВКА ДАННЫХ ДЛЯ ГИСТОГРАММ ---
months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Предполагается, что у вас есть df с колонками ['cluster', 'Month']
# Если нет — нужно подготовить его отдельно, например:
# df['cluster'] = ... ; df['Month'] = ...
# В данном примере предположим, что `df` уже есть.

# Группируем и считаем % по месяцам
df_grouped = df_season.groupby(['cluster', 'Month']).size().unstack(fill_value=0)
df_grouped = df_grouped.reindex(columns=months_order, fill_value=0)
df_percent = df_grouped.div(df_grouped.sum(axis=1), axis=0) * 100

# --- РАЗМЕЩЕНИЕ ---
clusters = sorted(clusters_full_tracks.keys())
n_clusters = len(clusters)

# Рассчитываем сетку: по 2 ячейки на кластер (карта + барчарт)
cols = min(3, n_clusters)
rows = (n_clusters + cols - 1) // cols
total_subplots = n_clusters * 2  # ×2 — карта и гистограмма

# GridSpec: 2*rows × cols → каждый кластер занимает 2 строки
gs = plt.GridSpec(2 * rows, cols, height_ratios=[2, 1] * rows, hspace=0.15, wspace=0.2)

fig = plt.figure(figsize=(5 * cols, 3.5 * 2 * rows), dpi=300)

# --- ЦИКЛ ПО КЛАСТЕРАМ ---
for i, cluster in enumerate(clusters):
    r = i // cols
    c = i % cols

    # --- 1. Карта (верхняя ячейка) ---
    ax_map = fig.add_subplot(gs[2 * r, c], projection=projection)

    tracks = clusters_full_tracks[cluster]
    if tracks:
        # Собираем точки для отрисовки треков
        for track in tracks:
            if cluster == 4:
                alpha = 0.3
            else:
                alpha = 0.1
            ax_map.plot(track[1], track[0],
                        c=colors[cluster],
                        transform=ccrs.PlateCarree(),
                        alpha=alpha)

        ax_map.set_extent([-80, 0, 4, 79], ccrs.PlateCarree())
        ax_map.coastlines(color='k', alpha=0.7, lw=1)

        gl = ax_map.gridlines(draw_labels=True, linewidth=0.5, color='grey', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = plt.FixedLocator(np.arange(-80, 1, 10))
        gl.ylocator = plt.FixedLocator(np.arange(10, 80, 10))
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}

        # ax_map.set_title(f'C{cluster+1} ({len(tracks)})', fontsize=12, fontweight='bold', pad=6)
        # --- Подпись вместо заголовка (типа легенды слева вверху) ---
        ax_map.text(0.02, 0.96, f'C{cluster+1} ({len(tracks)})',
                    transform=ax_map.transAxes,
                    fontsize=11,
                    fontweight='bold',
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.85, edgecolor='none'))

    else:
        ax_map.axis('off')

    # --- 2. Гистограмма (нижняя ячейка) ---
    ax_bar = fig.add_subplot(gs[2 * r + 1, c])

    if cluster in df_percent.index:
        freqs = df_percent.loc[cluster].reindex(months_order, fill_value=0)
        color = colors[cluster % len(colors)]
        bars = ax_bar.bar(months_order, freqs.values,
                          color=color, edgecolor='black', linewidth=0.5, alpha=0.9)

        # Подписи значений
        for bar, height in zip(bars, freqs.values):
            if height > 0.5:
                ax_bar.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                            f'{height:.1f}', ha='center', va='bottom', fontsize=7, color='black')

        # Только нижний ряд — подписи месяцев
        # if r == rows - 1:
        ax_bar.tick_params(axis='x', labelrotation=45, labelsize=8)
        # else:
        #     ax_bar.set_xticklabels([])

        # Только левый столбец — ось Y
        if c == 0:
            ax_bar.set_ylabel('Freq (%)', fontsize=9)
            ax_bar.set_yticklabels([])
            
        else:
            ax_bar.set_yticklabels([])
            ax_bar.tick_params(left=False)

        ax_bar.set_ylim(0, 25)
    else:
        ax_bar.axis('off')

# --- ЗАГРУЗКА ПУСТЫХ ЯЧЕЕК ---
# Ничего дополнительно делать не нужно — GridSpec уже фиксирован.

# plt.suptitle('Cyclone Tracks and Monthly Frequency by Cluster', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.02, 1, 0.96])

plot_path = f"{output_dir}/maps_with_innerannual_2026.png"
plt.savefig(plot_path)

plt.show()