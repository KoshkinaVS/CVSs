from step_of_tracking import *

tracking_type = 'tracking_local_only'

# трекинг с учетом смещения в новую точку по скорости
def step_of_tracking(CS_tracks_list, ds, data_type, path_data_dir, our_time, circ='C', dt_step=3600):
       
#     stat_Q = get_stat_global_max(ds, our_time, circ)
    local_max = get_stat_local_max(ds, our_time, circ)

    # радиус поиска
    hw_scale = 1.5
    
    
    for i in range(len(CS_tracks_list)):
        
        x_init = CS_tracks_list[i]['x'][-1]
        y_init = CS_tracks_list[i]['y'][-1]
        
        rad_init = CS_tracks_list[i]['rad'][-1]
        crit_init = CS_tracks_list[i]['crit'][-1]
        
        if ~np.isnan(x_init) and x_init != (len(ds[x_unit]) - 1) and y_init != (len(ds[y_unit]) - 1) and x_init != 0 and y_init != 0:
                
            if len(CS_tracks_list[i]['x']) == 1:
                x, y = get_new_loc_mean_speed(ds, data_type, dt_step, x_init, y_init, int(hw_scale*np.round(rad_init)), our_time, speed_level)
            else:
                x_prpr = CS_tracks_list[i]['x'][-2]
                y_prpr = CS_tracks_list[i]['y'][-2]
                x, y = get_new_loc_CS_speed(ds, dt_step, x_init, y_init, x_prpr, y_prpr)

            
            local_max['dist'] = dist(x, y, local_max.x, local_max.y)
#             local_max['dist'] = dist(x_init, y_init, local_max.x, local_max.y).values

            coords_Q_min_dist  = local_max[local_max['dist'] < hw_scale*rad_init]
    
            if len(coords_Q_min_dist) > 0:
            
                min_dist = np.nanmin(coords_Q_min_dist['dist'])
#                 min_delta_crit = np.min(coords_Q_min_dist['crit'] - crit_init)
                
                cluster = coords_Q_min_dist[coords_Q_min_dist['dist'] == min_dist]
#                 cluster = coords_Q_min_dist[(coords_Q_min_dist['crit'] - crit_init) == min_delta_crit]
                
#                 stat_Q_c = stat_Q[stat_Q['cluster'] == cluster['cluster'].values[0]]
                            
                # внимание - используется первое значение с мин расстоянием
                x_c = int(cluster.x.values[0])
                y_c = int(cluster.y.values[0])
                
                rad = (cluster.rad_eff.values[0])

                # print(f'{ds[time_unit][our_time].values}')
                
                # print(f'before: {x_init, y_init}, after: {x_c, y_c}, dist: {min_dist}')
                
                track_len_new = dist(x_c, y_c, x_init, y_init)

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
                CS_tracks_list[i]['time'].append(ds[time_unit][our_time].values)
                
                # удаляем строчку с использованными координатами
                if len(local_max) != 0:
                    local_max = local_max[local_max.index != cluster.index[0]]                    
                
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
            if len(CS['x']) >= 4:
                # добавить сохранение на этом шаге
                save_track_csv(CS, path_data_dir)
                
                # CS_tracks_list_new.append(CS)

    if len(local_max) != 0:
        CS_tracks_list = track_init(ds, len(CS_tracks_list), data_type, local_max.reset_index(), CS_tracks_list_new, our_time, time_unit)

    # sorted_CS_tracks_list = sorted(CS_tracks_list, key=lambda x: len(x['track_len']), reverse=True)

    CS_tracks_list.sort(key=lambda x: (len(x['track_len']), x['crit'][-1]), reverse=True)

    return CS_tracks_list