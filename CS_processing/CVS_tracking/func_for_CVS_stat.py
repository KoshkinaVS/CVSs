from matplotlib import pyplot as plt
import pandas as pd 
import numpy as np
import math
import xarray as xr
from numpy import linalg as LA

# from matplotlib import animation
import datetime
from datetime import timedelta

import scipy as sp
# from scipy.ndimage import label, generate_binary_structure
# from scipy import interpolate

import seaborn as sns

# import cartopy.crs as ccrs
# from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from shapely.geometry import Polygon
import matplotlib.patches as mpatches
from matplotlib import gridspec

from tqdm import tqdm

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# from mpl_toolkits.axes_grid1 import make_axes_locatable

import cmaps

from sklearn.cluster import DBSCAN


from pathlib import Path

import datetime
from datetime import timedelta

from tqdm import tqdm

import sys
import os
# caution: path[0] is reserved for script path (or '' in REPL)
# sys.path.insert(1, '/storage/kubrick/vkoshkina/scripts/vortex_identification')
# sys.path.insert(1, './vortex_identification')

# get coords for clustering after th, X.shape = (y,x), crit - name like in ds Data variables
def for_DBSCAN(X, threshold, crit):
    X_arr = X.to_dataframe().dropna(how='any')
#     print(f'points before th: {len(X_arr[crit])}')
    X_arr[crit][np.abs(X_arr[crit]) < threshold] = None
    X_arr = X_arr.dropna(how='any')
#     print(f'points after th: {len(X_arr[crit])}')
    
    coords = X_arr[crit].index.to_frame(name=['y', 'x'], index=False)
    coords['lon'] = X_arr.XLONG.values
    coords['lat'] = X_arr.XLAT.values
    coords['crit'] = X_arr[crit].values  
    
    return coords

# get pd.df with cluster label for each point
def clustering_DBSCAN_rortex_C(coords_lambda2, eps=10., min_samples=10, metric='euclidean', circ='AC'):
    
    db1 = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=4) # для декартовой сетки 
    
    if circ == 'C':
        coords_lambda2_pos = coords_lambda2[coords_lambda2.crit > 0]
    else:
        coords_lambda2_pos = coords_lambda2[coords_lambda2.crit < 0]
        
    coords_pos = np.array([coords_lambda2_pos.x, coords_lambda2_pos.y]).T
    
    y_db1_pos = db1.fit_predict(coords_pos)

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_lambda2_pos = len(set(y_db1_pos)) - (1 if -1 in y_db1_pos else 0)
    n_noise_lambda2_pos = list(y_db1_pos).count(-1)

#     print(f'Estimated number of clusters ({circ}): %d' % n_clusters_lambda2_pos)    
#     print(f'Estimated number of noise points ({circ}): %d' % n_noise_lambda2_pos)


    coords_lambda2_pos['cluster'] = y_db1_pos
       
    return coords_lambda2_pos.reset_index(), n_clusters_lambda2_pos

# filter out noise points, CSs < given number
def DBSCAN_filter(coords_lambda2, points=40):
    coords_lambda2 = coords_lambda2.drop(np.where(coords_lambda2['cluster'] == -1)[0])
    clusters = coords_lambda2.groupby('cluster').count()
    true_clusters = clusters[clusters['lat'] >= points]
    true_index = true_clusters.index.values
    
    n_clusters = len(true_clusters)
    
    coords_lambda2 = coords_lambda2.where(coords_lambda2.cluster.isin(true_index))
    coords_lambda2 = coords_lambda2.dropna()
    return coords_lambda2, n_clusters


# примитивная статистика, центр на основе максимума критерия
def get_stat(coords_Q_cluster, n_clusters_Q, dist_m, circ='C', CS_points_th=25):

    centroid_idx = []
    
    centroid_x = []
    centroid_y = []
    
    crit_centroid_x = []
    crit_centroid_y = []
    
    radius_eff = []
    
    for n in range(n_clusters_Q):
    
        coords_Q_c = coords_Q_cluster[coords_Q_cluster['cluster'] == n]
        
        if len(coords_Q_c) >= CS_points_th:
            
            N_points = len(coords_Q_c)
            rad_eff = np.sqrt(N_points/np.pi)  # *dist_m/1000
            radius_eff.append(rad_eff)
            
            centroid_idx.append(n)

            if circ == 'AC':
                crit_max = np.min(coords_Q_c.crit)
            else:
                crit_max = np.max(coords_Q_c.crit)
            center = coords_Q_c[coords_Q_c['crit'] == crit_max]
            crit_centroid_x.append(float(center.x))
            crit_centroid_y.append(float(center.y))


        else:
            
            
            centroid_idx.append(None)
            
            centroid_x.append(None)
            centroid_y.append(None)
            
            crit_centroid_x.append(None)
            crit_centroid_y.append(None)
            
            radius_eff.append(None)
        
    
    stat = pd.DataFrame({'cluster': centroid_idx, 
#                          'x': centroid_x, 'y': centroid_y,              
                         'x': crit_centroid_x, 'y': crit_centroid_y,                         
                         'rad_eff': radius_eff,
                        })
    
    stat = stat.dropna()   

    return stat


