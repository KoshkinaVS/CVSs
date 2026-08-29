import pandas as pd 
import numpy as np
import math
import xarray as xr

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


import datetime
from datetime import timedelta

import scipy as sp

from tqdm import tqdm

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


from sklearn.cluster import DBSCAN
from skimage.feature import peak_local_max


from pathlib import Path
import sys
import os

path_init = f'/storage/thalassa/users/vkoshkina'

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, f'{path_init}/scripts/vortex_identification')
# sys.path.insert(1, './vortex_identification')

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
    
    db1 = DBSCAN(eps=eps, min_samples=min_samples, metric=metric) # для декартовой сетки 
    
    y_db1 = db1.fit_predict(coords)
#     y_db1 = db1.fit(coords).labels_
    
    
    
    pd_coords = pd.DataFrame(data={'x': coords[:,0], 
                                   'y': coords[:,1],
                                  })
    
    pd_coords['cluster'] = y_db1
    
    pd_coords = pd_coords.drop(np.where(pd_coords['cluster'] == -1)[0])
    clusters = pd_coords.groupby('cluster').count()
    true_clusters = clusters[clusters['x'] >= size_filter]
    true_index = true_clusters.index.values
    
    pd_coords = pd_coords.where(pd_coords.cluster.isin(true_index))
    pd_coords = pd_coords.dropna()
    
    # Получаем уникальные значения и сортируем их
    unique_vals_pos = sorted(pd_coords['cluster'].unique())
    
    # Создаем отображение уникальных значений на новые значения
    mapping = {val: i+1 for i, val in enumerate(unique_vals_pos)}
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

    crits  = [float(r2d[int(y),int(x)].values) for x,y in zip(pd_coords.x.values, pd_coords.y.values)]

    pd_coords['crit']  = crits

    stat = get_stat(pd_coords, bounds=False)
        
    return pd_coords, stat

# примитивная статистика, центр на основе максимума критерия
def get_stat(coords_Q_cluster, bounds=False):

    centroid_idx = []
    
    crit_centroid_x = []
    crit_centroid_y = []
    
    radius_eff = []
    
    rortex = []
    
    for n in coords_Q_cluster['cluster'].unique():
    
        coords_Q_c = coords_Q_cluster[coords_Q_cluster['cluster'] == n]
        
        N_points = len(coords_Q_c)
        rad_eff = np.sqrt(N_points/np.pi)  # *dist_m/1000
        radius_eff.append(rad_eff)
        
        centroid_idx.append(n)

        crit_max = np.nanmax(np.abs(coords_Q_c.crit))
        center = coords_Q_c[np.abs(coords_Q_c['crit']) == crit_max]
        
        x_cr = center.x[center.index == center.index[0]].values[0]
        y_cr = center.y[center.index == center.index[0]].values[0]
        crit_value = center.crit[center.index == center.index[0]].values[0]
        
        x_c = int(x_cr)
        y_c = int(y_cr)
        
        
        rortex.append(float(crit_value))
        
        crit_centroid_x.append(x_c)
        crit_centroid_y.append(y_c)
        
    
    stat = pd.DataFrame({'cluster': centroid_idx, 
                         'x': crit_centroid_x, 'y': crit_centroid_y,  
                         'rad_eff': radius_eff,
                         'crit': rortex,
                        })
    
    stat = stat.dropna()   

    return stat

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

    x_name = config["x_name"]
    y_name = config["y_name"]

    coords['Time'] = 0
    coords_ds = coords.set_index(['Time', 'y', 'x'])
    
    dates = coords.Time.unique()
    new_index = pd.MultiIndex.from_product([list(dates), 
                                            np.arange(0,len(ds_HiRes[y_name])), 
                                            np.arange(0,len(ds_HiRes[x_name]))
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

    x_name = config["x_name"]
    y_name = config["y_name"]
    
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

def get_DBSCAN_ds(ds_HiRes, config, our_level=0):
    name_crit   = config['name_crit']
    time_unit   = config['time_unit']
    time_name   = config['time_name']
    level_name  = config['level_name']
    eps         = config['eps']
    min_samples = config['min_samples']
    size_filter = config['size_filter']
    min_dist    = config['min_dist']
    data_type    = config['data_type']
    
    
    itime = pd.Timestamp(ds_HiRes[time_unit].values[0])
    coords_HiRes, stat_HiRes = DBSCAN_processing(ds_HiRes, name_crit, time_unit, time_name, level_name, itime, eps, min_samples, size_filter, our_level=our_level) 

    local_max = get_local_max(ds_HiRes, stat_HiRes, coords_HiRes, 'crit', config, min_dist=min_dist)

    local_max['rad_eff'] = [stat_HiRes['rad_eff'][stat_HiRes['cluster'] == clstr].values[0] 
                            for clstr in local_max['cluster']]


    cluster_ds = pd_ds_magic(ds_HiRes, coords_HiRes, 0, ['cluster'], config)
    cluster_ds_centers = pd_ds_magic(ds_HiRes, stat_HiRes, 0, ['crit', 'cluster', 'rad_eff'], config)
    cluster_ds_local_max = pd_ds_magic(ds_HiRes, local_max, 0, ['crit', 'cluster', 'rad_eff'], config)


    cluster_ds['center'] = cluster_ds_centers['crit']
    cluster_ds['center_cluster'] = cluster_ds_centers['cluster']
    cluster_ds['rad_eff'] = cluster_ds_centers['rad_eff']

    cluster_ds['local_extr_crit'] = cluster_ds_local_max['crit']
    cluster_ds['local_extr_cluster'] = cluster_ds_local_max['cluster']
    cluster_ds['local_extr_rad_eff'] = cluster_ds_local_max['rad_eff']


    for idx,t in enumerate(tqdm(ds_HiRes[time_unit].values[1:])):
        t = pd.Timestamp(t)

        coords_HiRes, stat_HiRes = DBSCAN_processing(ds_HiRes, name_crit, time_unit, time_name, level_name, t, eps, min_samples, size_filter, our_level=our_level) 
        local_max = get_local_max(ds_HiRes, stat_HiRes, coords_HiRes, 'crit', config, min_dist=min_dist)
        local_max['rad_eff'] = [stat_HiRes['rad_eff'][stat_HiRes['cluster'] == clstr].values[0] 
                            for clstr in local_max['cluster']]


        cluster_ds_1 = pd_ds_magic(ds_HiRes, coords_HiRes, idx+1, ['cluster'], config)
        cluster_ds_centers = pd_ds_magic(ds_HiRes, stat_HiRes, idx+1, ['crit', 'cluster', 'rad_eff'], config)
        cluster_ds_local_max = pd_ds_magic(ds_HiRes, local_max, idx+1, ['crit', 'cluster', 'rad_eff'], config)

        cluster_ds_1['center'] = cluster_ds_centers['crit']
        cluster_ds_1['center_cluster'] = cluster_ds_centers['cluster']
        cluster_ds_1['rad_eff'] = cluster_ds_centers['rad_eff']

        cluster_ds_1['local_extr_crit'] = cluster_ds_local_max['crit']
        cluster_ds_1['local_extr_cluster'] = cluster_ds_local_max['cluster']
        cluster_ds_1['local_extr_rad_eff'] = cluster_ds_local_max['rad_eff']

        cluster_ds = xr.concat([cluster_ds, cluster_ds_1], time_name)  


#     r2d = ds_HiRes['R2D'][:,our_level:our_level+1]
#     r2d_max = r2d.max()
#     r2d_normalized = r2d / r2d_max
#     r2d_scaled = r2d_normalized * 127
#     r2d_int8 = r2d_scaled.astype(np.int8)

#     # r2d_scaled = r2d_normalized * 32767
#     # r2d_int8 = r2d_scaled.astype(np.int16)

#     cluster_ds['R2D'] = r2d_int8
#     cluster_ds['R2D'].attrs['description'] = 'Rortex criterion 2D discrete (neg - AC, pos - C)'
#     cluster_ds['R2D'].attrs['long_name'] = 'Rortex 2D discrete'

    cluster_ds['cluster'] = cluster_ds['cluster'].astype(np.int16) # LoRes влезает в int8, а вот HiRes нет (я так думала)
    cluster_ds['cluster'].attrs['description'] = 'CS cluster number for groups of points (neg - AC, pos - C)'
    cluster_ds['cluster'].attrs['long_name'] = 'CS cluster'

    cluster_ds['center'] = cluster_ds['center'].astype(np.float32)
    cluster_ds['center'].attrs['description'] = 'CS center based on max(abs(R2D))'
    cluster_ds['center'].attrs['long_name'] = 'CS max center (R2D value)'

    cluster_ds['center_cluster'] = cluster_ds['center_cluster'].astype(np.int16) 
    cluster_ds['center_cluster'].attrs['description'] = 'CS cluster number for max center (neg - AC, pos - C)'
    cluster_ds['center_cluster'].attrs['long_name'] = 'CS max center (cluster value)'

    cluster_ds['rad_eff'] = cluster_ds['rad_eff'].astype(np.float32)
    cluster_ds['rad_eff'].attrs['description'] = 'CS center effective radius'
    cluster_ds['rad_eff'].attrs['long_name'] = 'CS max center rad_eff'

    cluster_ds['local_extr_crit'] = cluster_ds['local_extr_crit'].astype(np.float32)
    cluster_ds['local_extr_crit'].attrs['description'] = f'CS local R2D extrema value R2D with min_dist={min_dist}'
    cluster_ds['local_extr_crit'].attrs['long_name'] = 'CS local R2D extrema value (AC < 0 , C > 0)'

    cluster_ds['local_extr_cluster'] = cluster_ds['local_extr_cluster'].astype(np.int16)
    cluster_ds['local_extr_cluster'].attrs['description'] = f'CS local R2D extrema cluster number with min_dist={min_dist}'
    cluster_ds['local_extr_cluster'].attrs['long_name'] = 'CS local R2D extrema cluster number (AC < 0 , C > 0)'

    cluster_ds['local_extr_rad_eff'] = cluster_ds['local_extr_rad_eff'].astype(np.float32)
    cluster_ds['local_extr_rad_eff'].attrs['description'] = f'CS local R2D extrema radius with min_dist={min_dist}'
    cluster_ds['local_extr_rad_eff'].attrs['long_name'] = 'CS local R2D extrema radius in gridpoints'

    if data_type != 'ERA5':
        del cluster_ds.coords['south_north']
        del cluster_ds.coords['west_east']

    return cluster_ds