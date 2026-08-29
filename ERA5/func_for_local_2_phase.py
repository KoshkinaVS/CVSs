from step_of_tracking import *


# tracking_type = 'tracking_local_2_phase'

# CVS_speed = 'no_speed'
# CVS_speed = 'adv_speed'
# CVS_speed = 'bg_speed'
# CVS_speed = 'adv_bg_speed'


# трекинг с учетом смещения в новую точку по скорости
def step_of_tracking(cluster_idx, CS_tracks_list, clstr_len, ds, data_type, path_data_dir, our_time, 
                     speed_level, CVS_speed,
                     circ='C', dt_step=3600):

    r2d_coords = get_all_coords(ds, our_time, circ)
    # stat_Q = get_stat_global_max(ds, our_time, circ)
    local_max = get_stat_local_max(ds, our_time, circ)
    
    local_max = update_rad(local_max, r2d_coords)

    # радиус поиска
    hw_scale = 1.5
    # граница домена
    bound_size = 0
    # Максимальное перемещение в grid cells
    max_allowed_distance = 50
    
    for i in range(len(CS_tracks_list)):

        local_max['dist'] = 9999.
        local_max['x'] = local_max['x'].astype(int)
        local_max['y'] = local_max['y'].astype(int)
        
        x_init = CS_tracks_list[i]['x'][-1]
        y_init = CS_tracks_list[i]['y'][-1]
        
        rad_init = CS_tracks_list[i]['rad'][-1]
        crit_init = CS_tracks_list[i]['crit'][-1]
        
        # if ~np.isnan(x_init) and x_init != (len(ds[x_unit]) - 1) and y_init != (len(ds[y_unit]) - 1) and x_init != 0 and y_init != 0:
        # отступаем bound_size клеточек от границы
        if ~np.isnan(x_init) and x_init < (len(ds[x_unit]) - bound_size) and y_init < (len(ds[y_unit]) - bound_size) and x_init > bound_size - 1 and y_init > bound_size - 1:

            x, y = get_next_loc_cases(CS_tracks_list, ds, dt_step, i, our_time, x_init, y_init, hw_scale, rad_init, speed_level, CVS_speed)
            
            # local_max['dist'] = dist(x, y, local_max.x.values, local_max.y.values)
            local_max['dist'] = np.sqrt((x - local_max['x'])*(x - local_max['x']) + (y - local_max['y'])*(y - local_max['y']))
            

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

                if track_len_new > max_allowed_distance:
                    # Перемещение слишком большое - завершаем трек
                    print(f"Track {i} ({CVS_speed}): excessive movement ({track_len_new:.1f} units, while min_dist={min_dist} for x_0={x_init} -> x_speed={x} -> x_new={x_c}, y_0={y_init} -> y_speed={y} -> y_new={y_c})")
                    print(coords_Q_min_dist)

                    CS_tracks_list[i] = append_nan_to_track(CS_tracks_list[i])
                    continue

                CS_tracks_list[i]['x'].append(x_c)
                CS_tracks_list[i]['y'].append(y_c)
                
                
                if data_type == 'ERA5':
                    # CS_tracks_list[i]['lon'].append(float(ds.XLONG[y_c,x_c].values))
                    # CS_tracks_list[i]['lat'].append(float(ds.XLAT[y_c,x_c].values))
                    CS_tracks_list[i]['lat'].append(float(ds.latitude[y_c].values))
                    CS_tracks_list[i]['lon'].append(float(ds.longitude[x_c].values))
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
                if len(local_max) != 0:
                # if len(stat_Q) != 0:
                    
                    # stat_Q = stat_Q[stat_Q['cluster'] != cluster['cluster'].values[0]]
                    # local_max = local_max[local_max['cluster'] != cluster['cluster'].values[0]]
                    local_max = local_max[local_max.index != cluster.index.values[0]]
                    local_max = local_max.reset_index(drop=True)
                    
                    # local_max['dist'] = 9999
                    
                    
            else:
                CS_tracks_list[i] = append_nan_to_track(CS_tracks_list[i]) 
                
        else:
            CS_tracks_list[i] = append_nan_to_track(CS_tracks_list[i])
                
    CS_tracks_list_new = []
    
    for CS in CS_tracks_list:
        if not np.isnan(CS['x'][-1]):
            CS_tracks_list_new.append(CS)
        else:
            if np.sum(~np.isnan(CS['x'])) >= 3:
                # добавить сохранение на этом шаге
                cluster_idx = save_track_csv(cluster_idx, CS, path_data_dir)
                

    # if len(stat_Q) != 0:
        # CS_tracks_list, clstr_len = track_init(ds, clstr_len, data_type, stat_Q.reset_index(), CS_tracks_list_new, our_time, time_unit)
    
    if len(local_max) != 0:
        CS_tracks_list, clstr_len = track_init(ds, clstr_len, data_type, local_max.reset_index(drop=True), CS_tracks_list_new, our_time, time_name)

    # sorted_CS_tracks_list = sorted(CS_tracks_list, key=lambda x: len(x['track_len']), reverse=True)

    CS_tracks_list.sort(key=lambda x: (x['crit'][-1]), reverse=True)

    return cluster_idx, CS_tracks_list, clstr_len