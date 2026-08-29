from step_of_tracking import *


# tracking_type = 'tracking_local_global'

# CVS_speed = 'no_speed'
# CVS_speed = 'adv_speed'
# CVS_speed = 'bg_speed'
# # CVS_speed = 'adv_bg_speed'

# трекинг с учетом смещения в новую точку по скорости
def step_of_tracking(cluster_idx, CS_tracks_list, clstr_len, ds, data_type, path_data_dir, our_time, 
                                          speed_level, CVS_speed,
                     circ='C', dt_step=3600, multi_levels=False):
    
    stat_Q = get_stat_global_max(ds, our_time, circ)
    local_max = get_stat_local_max(ds, our_time, circ)

    # радиус поиска
    hw_scale = 1.5
    
    # граница домена
    bound_size = 0
    
    for i in range(len(CS_tracks_list)):
        
        x_init = CS_tracks_list[i]['x'][-1]
        y_init = CS_tracks_list[i]['y'][-1]
        
        rad_init = CS_tracks_list[i]['rad'][-1]
        crit_init = CS_tracks_list[i]['crit'][-1]
        
        # if ~np.isnan(x_init) and x_init != (len(ds[x_unit]) - 1) and y_init != (len(ds[y_unit]) - 1) and x_init != 0 and y_init != 0:
        # отступаем bound_size клеточек от границы
        if ~np.isnan(x_init) and x_init < (len(ds[x_unit]) - bound_size) and y_init < (len(ds[y_unit]) - bound_size) and x_init > bound_size - 1 and y_init > bound_size - 1:
            
            x, y = get_next_loc_cases(CS_tracks_list, ds, dt_step, i, our_time, x_init, y_init, hw_scale, rad_init, speed_level, CVS_speed)
            
            local_max['dist'] = dist(x, y, local_max.x, local_max.y)

            coords_Q_min_dist  = local_max[local_max['dist'] < hw_scale*rad_init]
    
            if len(coords_Q_min_dist) > 0:
            
                min_dist = np.nanmin(coords_Q_min_dist['dist'])
#                 min_delta_crit = np.min(coords_Q_min_dist['crit'] - crit_init)
                
                cluster = coords_Q_min_dist[coords_Q_min_dist['dist'] == min_dist]
#                 cluster = coords_Q_min_dist[(coords_Q_min_dist['crit'] - crit_init) == min_delta_crit]
                
                            
                # внимание - используется первое значение с мин расстоянием
                x_c = int(cluster.x.values[0])
                y_c = int(cluster.y.values[0])
                
                rad = (cluster.rad_eff.values[0])
                
                track_len_new = dist(x_c, y_c, x_init, y_init)

                CS_tracks_list[i]['x'].append(x_c)
                CS_tracks_list[i]['y'].append(y_c)
                
                
                if data_type == 'ERA5':
                    # CS_tracks_list[i]['lon'].append(float(ds.XLONG[y_c,x_c].values))
                    # CS_tracks_list[i]['lat'].append(float(ds.XLAT[y_c,x_c].values))
                    CS_tracks_list[i]['lat'].append(float(ds.latitude[y_c].values))
                    CS_tracks_list[i]['lon'].append(float(ds.longitude[x_c].values))
                elif data_type == 'SMP':
                    CS_tracks_list[i]['lon'].append(float(ds.XLONG[y_c,x_c].values))
                    CS_tracks_list[i]['lat'].append(float(ds.XLAT[y_c,x_c].values))
                else:
                    CS_tracks_list[i]['lon'].append(float(ds.XLONG[our_time,y_c,x_c].values))
                    CS_tracks_list[i]['lat'].append(float(ds.XLAT[our_time,y_c,x_c].values))
                
                CS_tracks_list[i]['rad'].append(rad)
                CS_tracks_list[i]['crit'].append(cluster.crit.values[0])
        
                
                track_l = CS_tracks_list[i]['track_len'][-1]
                
                CS_tracks_list[i]['track_len'].append(track_l + track_len_new)

                CS_tracks_list[i]['t'].append(our_time)
                CS_tracks_list[i]['time'].append(ds[time_name][our_time].values)
                
                # удаляем строчку с использованными координатами
                if len(stat_Q) != 0:
                    stat_Q = stat_Q[stat_Q['cluster'] != cluster['cluster'].values[0]]
                    local_max = local_max[local_max['cluster'] != cluster['cluster'].values[0]]
                    
            else:
                CS_tracks_list[i]['x'].append(np.nan)
                CS_tracks_list[i]['y'].append(np.nan)
                
                CS_tracks_list[i]['lon'].append(np.nan)
                CS_tracks_list[i]['lat'].append(np.nan)
                
                CS_tracks_list[i]['rad'].append(np.nan)
                CS_tracks_list[i]['crit'].append(np.nan)

                CS_tracks_list[i]['t'].append(np.nan)
                CS_tracks_list[i]['time'].append(np.nan)
                CS_tracks_list[i]['track_len'].append(np.nan)       
                
        else:
            CS_tracks_list[i]['x'].append(np.nan)
            CS_tracks_list[i]['y'].append(np.nan)
            
            CS_tracks_list[i]['lon'].append(np.nan)
            CS_tracks_list[i]['lat'].append(np.nan)
            
            CS_tracks_list[i]['rad'].append(np.nan)
            CS_tracks_list[i]['crit'].append(np.nan)

            CS_tracks_list[i]['t'].append(np.nan)
            CS_tracks_list[i]['time'].append(np.nan)
            CS_tracks_list[i]['track_len'].append(np.nan)
                
    CS_tracks_list_new = []
    
    for CS in CS_tracks_list:
        if not np.isnan(CS['x'][-1]):
            CS_tracks_list_new.append(CS)
        else:
            if np.sum(~np.isnan(CS['x'])) >= 3:
                # добавить сохранение на этом шаге
                cluster_idx = save_track_csv(cluster_idx, CS, path_data_dir)
                save_track_txt(cluster_idx, CS, path_data_dir)

    if len(stat_Q) != 0:
        CS_tracks_list, clstr_len = track_init(ds, clstr_len, data_type, stat_Q.reset_index(), CS_tracks_list_new, our_time, time_name)

    # sorted_CS_tracks_list = sorted(CS_tracks_list, key=lambda x: len(x['track_len']), reverse=True)

    CS_tracks_list.sort(key=lambda x: (x['crit'][-1]), reverse=True)
    
    return cluster_idx, CS_tracks_list, clstr_len