import seaborn as sns
import matplotlib.pyplot as plt

from func_for_CVS_clusters import *


tracking_type = 'tracking_local_2_phase'
pref_tracking = 'all_points_bound'
CVS_speed = 'adv_speed'
circ = 'C'
n_clusters = 5
years = np.arange(1979, 2025)
months = np.arange(1, 13, 1)
path_data = f'{path_init}/data'  
DBSCAN_name = f'DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_level_{level}_smoothing'
results_dir = f"{tracking_type}_{CVS_speed}_{pref_tracking}"

if data_type == 'HiRes':
    path_data_tracks = f'{path_data}/TC_tracks/my_tracking_results/{pref_tracking}/{results_dir}/{data_type}_tracks_sigma_{sigma}'
elif data_type == 'ERA5':
    path_data_tracks = f"{path_data}/TC_tracks/{data_type}/my_tracking_results/{pref_tracking}/{results_dir}/{data_type}_tracks_sigma_{sigma}"
else:
    path_data_tracks = f'{path_data}/{data_type}/{data_type}/{data_type}_tracks/{pref_tracking}/{results_dir}/{DBSCAN_name}'

path_tracks_dir = f'{path_data_tracks}/tracks_{circ}_params'

param_cols = [
    'lat', 'lon', 
    'rad', 'crit', 
    # 'track_len',
    'msl_min', 'mlhf_max', 'mshf_max', 'wspd_max', 't2m_median', 't_median',
    'w_median', 'tp_median']

CS_tracks_list_LoRes = []
path_list = []
basenames_list = []

for year in years:
    CS_tracks_list_LoRes, path_list, basenames_list = load_season_tracks(
        CS_tracks_list_LoRes, path_list, basenames_list, year, months, path_tracks_dir, time_th=24)

df_max = pd.DataFrame()

for idx, CS in enumerate(tqdm(CS_tracks_list_LoRes)):
    df_max = get_max_crit_day_values(CS, df_max, idx, param_cols)



corr_matrix = df_max[param_cols[2:]].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title("Матрица корреляций параметров")
# plt.show()

folder = f'{path_tracks_dir}/pics/'
os.makedirs(folder, exist_ok=True)
plt.savefig(
    f"{folder}/corr_matrix.png", 
    dpi=200, 
    bbox_inches="tight", 
    transparent=False
)
plt.close()

cyclones = np.array(df_max[param_cols[2:]])

# Apply PCA
data_pca = pca_analysis(cyclones)
# Perform CA
clusters = ca_analysis(data_pca, n_clusters)
cluster_groups, names_groups = group_cyclones(cyclones, clusters, path_list, basenames_list)

# Создаем список для хранения информации о кластерах и файлах
cluster_idx_list = []
names_list = []

for cluster in cluster_groups.keys():
    for cluster_idx in cluster_groups[cluster]:
        cluster_idx_list.append(cluster)
        names_list.append(names_groups[cluster][cluster_groups[cluster].index(cluster_idx)])

year = 1979

# Сохраняем результат в CSV
cluster_names = pd.DataFrame(data={'path': [p[0] for p in names_list], 'file_name': [p[1] for p in names_list], 'cluster': cluster_idx_list}) 
# cluster_names.to_csv(f'{path_tracks_dir}/{n_clusters}_clusters_names-{year}_test.csv', index=False)
cluster_names.to_csv(f'{path_tracks_dir}/{n_clusters}_clusters_names_test.csv', index=False)
