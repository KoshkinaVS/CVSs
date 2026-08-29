import numpy as np
import pandas as pd
import xarray as xr
from sklearn.cluster import DBSCAN
from skimage.feature import peak_local_max
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")



def get_R2D(ds, name_crit, time_unit, time_name, level_name, time, our_level=0):
      
    it = np.argmin(np.abs((pd.to_datetime(ds[time_unit]) - time).total_seconds()))
    var = ds[name_crit].isel({time_name: it,level_name: our_level})

    return var

def get_coords_for_DBSCAN(data, circ='C'):
    
    if circ == 'C':
        data = xr.where(data <= 0, np.nan, data)
    elif circ == 'AC':
        data = xr.where(data >= 0, np.nan, data)

    mask = data.notnull()
    y, x = np.where(mask)
    
    var = np.array([x,y]).T

    return var


# get pd.df with cluster label for each point
def clustering_DBSCAN_C(coords, eps=10., min_samples=10, size_filter=10, metric='euclidean'):
    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = db.fit_predict(coords)
    
    # Создаем DataFrame и фильтруем
    pd_coords = pd.DataFrame({'x': coords[:,0], 'y': coords[:,1], 'cluster': labels})
    pd_coords = pd_coords[pd_coords['cluster'] != -1]
    
    # Группировка и фильтрация с помощью transform (быстрее чем отдельные операции)
    cluster_sizes = pd_coords.groupby('cluster').transform('size')
    pd_coords = pd_coords[cluster_sizes >= size_filter]
    
    # Перенумерация кластеров
    unique_clusters = pd_coords['cluster'].unique()
    mapping = {old: new for new, old in enumerate(unique_clusters, 1)}
    pd_coords['cluster'] = pd_coords['cluster'].map(mapping)
    
    return pd_coords

def DBSCAN_processing(ds, name_crit, time_unit, time_name, level_name, itime, eps, min_samples, size_filter, our_level=0):

    r2d = get_R2D(ds, name_crit, time_unit, time_name, level_name, itime, our_level=our_level)
    
    r2d_coords_C = get_coords_for_DBSCAN(r2d, circ='C')
    r2d_coords_AC = get_coords_for_DBSCAN(r2d, circ='AC')
    
    pd_coords_C = clustering_DBSCAN_C(r2d_coords_C, eps=eps, min_samples=min_samples, size_filter=size_filter)
    pd_coords_AC = clustering_DBSCAN_C(r2d_coords_AC, eps=eps, min_samples=min_samples, size_filter=size_filter)

    pd_coords_AC['cluster'] = -pd_coords_AC['cluster']

    pd_coords = pd.concat([pd_coords_C,pd_coords_AC], ignore_index=True)

    crits = r2d.values[pd_coords.y.astype(int), pd_coords.x.astype(int)]
    pd_coords['crit'] = crits.flatten()

    stat = get_stat(pd_coords, bounds=False)
        
    return pd_coords, stat

# примитивная статистика, центр на основе максимума критерия
def get_stat(coords_Q_cluster, bounds=False):
    # Используем групповые операции
    stats = coords_Q_cluster.groupby('cluster').agg(
        x=('x', lambda g: g.iloc[np.abs(coords_Q_cluster.loc[g.index, 'crit']).argmax()]),
        y=('y', lambda g: g.iloc[np.abs(coords_Q_cluster.loc[g.index, 'crit']).argmax()]),
        crit=('crit', lambda g: g.iloc[np.abs(g).argmax()]),
        rad_eff=('crit', lambda g: np.sqrt(len(g)/np.pi))
    ).reset_index()
    
    return stats.dropna()

def get_local_max(ds_HiRes, stat, coords, crit, config, min_dist=1):
    
    coords['Time'] = 0
    coords_ds = coords.set_index(['Time', 'y', 'x'])
    
    dates = coords.Time.unique()
    new_index = pd.MultiIndex.from_product([list(dates), 
                                            np.arange(0,len(ds_HiRes[config['y_name']])), 
                                            np.arange(0,len(ds_HiRes[config['x_name']]))
                                           ], names=["Time", "y", "x" ])

    coords_ds = coords_ds.reindex(new_index, fill_value=0.)
    cluster_ds = xr.Dataset.from_dataframe(coords_ds[[crit, 'cluster']])
    
    cluster_vals_pos = np.where(cluster_ds['crit'][0].values > 0, cluster_ds['crit'][0].values, 0)
    cluster_vals_neg = np.where(cluster_ds['crit'][0].values < 0, -cluster_ds['crit'][0].values, 0)
    
    
    coordinates_pos = peak_local_max(cluster_vals_pos, min_distance=min_dist, exclude_border=False)
    coordinates_neg = peak_local_max(cluster_vals_neg, min_distance=min_dist, exclude_border=False)
    
    
    crit_values_pos = [float(cluster_ds['crit'][0,y,x].values) 
                   for y,x in zip(coordinates_pos[:,0], coordinates_pos[:,1])]
    cluster_values_pos = [float(cluster_ds['cluster'][0,y,x].values) 
                      for y,x in zip(coordinates_pos[:,0], coordinates_pos[:,1])]
    
    crit_values_neg = [float(cluster_ds['crit'][0,y,x].values) 
                   for y,x in zip(coordinates_neg[:,0], coordinates_neg[:,1])]
    cluster_values_neg = [float(cluster_ds['cluster'][0,y,x].values) 
                      for y,x in zip(coordinates_neg[:,0], coordinates_neg[:,1])]
    

    local_max_pos = pd.DataFrame(data={'x': coordinates_pos[:,1],
                              'y': coordinates_pos[:,0],
                              'crit': crit_values_pos,
                              'cluster': cluster_values_pos,
                                  })
    
    local_max_neg = pd.DataFrame(data={'x': coordinates_neg[:,1],
                              'y': coordinates_neg[:,0],
                              'crit': crit_values_neg,
                              'cluster': cluster_values_neg,
                                  })
    
    local_max = pd.concat([local_max_pos, local_max_neg])
    
    unique_pairs_xy = set(zip(local_max['x'], local_max['y']))
    
    mask = ~stat.apply(lambda row: (row['x'], row['y']) in unique_pairs_xy, axis=1)
    filtered_stat = stat[mask]
    del filtered_stat['rad_eff']
    
    all_extr_coords = pd.concat([local_max, filtered_stat])

    all_extr_coords.reset_index(drop=True, inplace=True)

    return all_extr_coords


def pd_ds_magic(ds_HiRes, coords_HiRes, t, crit, config):

    name_crit   = config['name_crit']
    time_unit   = config['time_unit']
    time_name   = config['time_name']
    level_name  = config['level_name']
    eps         = config['eps']
    min_samples = config['min_samples']
    size_filter = config['size_filter']
    min_dist    = config['min_dist']
    y_name = config['y_name']
    x_name = config['x_name']
    
    x_len = len(ds_HiRes[x_name]) 
    y_len = len(ds_HiRes[y_name])    
    
    coords_HiRes[time_name] = t
    coords_HiRes_ds = coords_HiRes.set_index([time_name, 'y', 'x'])

    dates = coords_HiRes[time_name].unique()
    new_index = pd.MultiIndex.from_product([list(dates), np.arange(y_len), np.arange(x_len)],
                                           names=[time_name, y_name, x_name ])
    coords_HiRes_ds = coords_HiRes_ds.reindex(new_index, fill_value=None)

    cluster_ds = xr.Dataset.from_dataframe(coords_HiRes_ds[crit])
    cluster_ds = cluster_ds.expand_dims(dim={level_name: 1})
    cluster_ds = cluster_ds.transpose(time_name, level_name, y_name, x_name)
    
    return cluster_ds

def get_DBSCAN_ds(ds_HiRes, config, our_level=0, data_type='LoRes'):
    # Инициализация
    times = pd.to_datetime(ds_HiRes[config['time_unit']].values)
    datasets = []
    
    # Обработка каждого временного шага
    for idx, t in enumerate((times)):
        # Кластеризация
        coords, stat = DBSCAN_processing(ds_HiRes, config['name_crit'], config['time_unit'], 
                                       config['time_name'], config['level_name'], t, 
                                       config['eps'], config['min_samples'], 
                                       config['size_filter'], our_level)
        
        # Локальные максимумы
        local_max = get_local_max(ds_HiRes, stat, coords, 'crit', config, config['min_dist'])
        local_max['rad_eff'] = stat.set_index('cluster')['rad_eff'].loc[local_max['cluster']].values
        
        # Создание Dataset
        ds_cluster = pd_ds_magic(ds_HiRes, coords, idx, ['cluster'], config)
        ds_centers = pd_ds_magic(ds_HiRes, stat, idx, ['crit', 'cluster', 'rad_eff'], config)
        ds_local = pd_ds_magic(ds_HiRes, local_max, idx, ['crit', 'cluster', 'rad_eff'], config)
        
        # Объединение
        ds_cluster['center'] = ds_centers['crit']
        ds_cluster['center_cluster'] = ds_centers['cluster']
        ds_cluster['rad_eff'] = ds_centers['rad_eff']
        ds_cluster['local_extr_crit'] = ds_local['crit']
        ds_cluster['local_extr_cluster'] = ds_local['cluster']
        ds_cluster['local_extr_rad_eff'] = ds_local['rad_eff']
        
        datasets.append(ds_cluster)
    
    # Объединение всех временных шагов
    result = xr.concat(datasets, dim=config['time_name'])
    
    # Настройка атрибутов и типов данных
    attrs_config = {
        'cluster': ('CS cluster number for groups of points (neg - AC, pos - C)', 'CS cluster'),
        'center': ('CS center based on max(abs(R2D))', 'CS max center (R2D value)'),
        'center_cluster': ('CS cluster number for max center (neg - AC, pos - C)', 'CS max center (cluster value)'),
        'rad_eff': ('CS center effective radius', 'CS max center rad_eff'),
        'local_extr_crit': (f'CS local R2D extrema value R2D with min_dist={config["min_dist"]}', 
                           'CS local R2D extrema value (AC < 0 , C > 0)'),
        'local_extr_cluster': (f'CS local R2D extrema cluster number with min_dist={config["min_dist"]}', 
                              'CS local R2D extrema cluster number (AC < 0 , C > 0)'),
        'local_extr_rad_eff': (f'CS local R2D extrema radius with min_dist={config["min_dist"]}', 
                              'CS local R2D extrema radius in gridpoints')
    }
    
    for var, (desc, long_name) in attrs_config.items():
        result[var].attrs.update({'description': desc, 'long_name': long_name})
        if var in ['cluster', 'center_cluster', 'local_extr_cluster']:
            result[var] = result[var].astype(np.int16)
        else:
            result[var] = result[var].astype(np.float32)

    
    return result