from step_of_tracking import *


# tracking_type = 'tracking_local_2_phase'

# CVS_speed = 'no_speed'
# CVS_speed = 'adv_speed'
# CVS_speed = 'bg_speed'
# CVS_speed = 'adv_bg_speed'


# трекинг с учетом смещения в новую точку по скорости
def step_of_tracking(cluster_idx, CS_tracks_list, clstr_len, ds, data_type, path_data_dir, our_time, 
                     speed_level, CVS_speed,
                     circ='C', dt_step=3600, multi_levels=False):

    r2d_coords = get_all_coords(ds, our_time, circ)
    # stat_Q = get_stat_global_max(ds, our_time, circ)
    local_max = get_stat_local_max(ds, our_time, circ)
    
    local_max = update_rad(local_max, r2d_coords)

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

            # Новая логика для avg_cone
            if CVS_speed == 'avg_cone':
                # Проверяем, что есть минимум 2 предыдущих точки для расчета скорости
                if len(CS_tracks_list[i]['x']) >= 2:  # Нужно 2 точки: текущая, предыдущая
                    # Берем предыдущую и пред-предыдущую точки
                    x_prev = CS_tracks_list[i]['x'][-1]
                    y_prev = CS_tracks_list[i]['y'][-1]
                    x_prev2 = CS_tracks_list[i]['x'][-2]
                    y_prev2 = CS_tracks_list[i]['y'][-2]
                    
                    # Рассчитываем скорость перемещения (пикселей за dt_step)
                    u = (x_prev - x_prev2) / dt_step if dt_step > 0 else 0  # скорость по x (пиксели/сек)
                    v = (y_prev - y_prev2) / dt_step if dt_step > 0 else 0  # скорость по y (пиксели/сек)
                    
                    # Скорость в м/с (с учетом разрешения сетки)
                    speed_ms = np.sqrt(u**2 + v**2) * dist_m  # м/с
                    
                    # Ожидаемая позиция через dt_step
                    x_exp = x_prev + u * dt_step
                    y_exp = y_prev + v * dt_step
                    
                    # Получаем параметры конуса на основе скорости
                    cone_params = get_search_cone_parameters(speed_ms, dt_step, hw_scale, rad_init)
                    
                    # Динамический радиус поиска
                    search_radius = cone_params['search_radius']
                    
                    # Вычисляем расстояние от ожидаемой позиции до всех кандидатов
                    local_max['dist'] = dist(x_exp, y_exp, local_max.x, local_max.y)
                    
                    # Базовый отбор по радиусу
                    candidates = local_max[local_max['dist'] < search_radius].copy()
                    
                    if len(candidates) > 0:
                        # Вычисляем направление предыдущего движения
                        prev_direction = np.degrees(np.arctan2(x_prev - x_prev2, y_prev - y_prev2)) % 360
                        
                        # Для каждого кандидата вычисляем изменение направления
                        angle_changes = []
                        for idx, candidate in candidates.iterrows():
                            # Вектор от предыдущей точки к кандидату
                            dx = candidate.x - x_prev
                            dy = candidate.y - y_prev
                            
                            # Направление к кандидату
                            candidate_direction = np.degrees(np.arctan2(dx, dy)) % 360
                            
                            # Изменение направления относительно предыдущего движения
                            angle_change = abs(candidate_direction - prev_direction)
                            angle_change = min(angle_change, 360 - angle_change)
                            angle_changes.append(angle_change)
                        
                        candidates['angle_change'] = angle_changes
                        
                        # Фильтруем по максимальному изменению направления
                        candidates = candidates[candidates['angle_change'] <= cone_params['max_angle_change']]
                        
                        # Дополнительная фильтрация по форме конуса
                        if len(candidates) > 0 and cone_params['search_shape'] != 'full_circle':
                            # Вычисляем углы относительно ожидаемой позиции
                            angles_from_exp = np.degrees(
                                np.arctan2(candidates.x - x_exp, candidates.y - y_exp)
                            ) % 360
                            
                            if cone_params['search_shape'] == 'three_quarter_circle':
                                # Исключаем задний сектор (противоположное направление)
                                back_direction = (prev_direction + 180) % 360
                                angle_diff_from_back = np.abs(angles_from_exp - back_direction)
                                angle_diff_from_back = np.minimum(angle_diff_from_back, 360 - angle_diff_from_back)
                                candidates = candidates[angle_diff_from_back > 45]
                                
                            elif cone_params['search_shape'] == 'semicircle':
                                # Только передний полукруг
                                angle_diff_from_prev = np.abs(angles_from_exp - prev_direction)
                                angle_diff_from_prev = np.minimum(angle_diff_from_prev, 360 - angle_diff_from_prev)
                                candidates = candidates[angle_diff_from_prev <= 90]
                                
                            elif cone_params['search_shape'] in ['broad_cone', 'long_cone']:
                                # Узкий конус
                                cone_half_angle = cone_params['max_angle_change'] / 2
                                angle_diff_from_prev = np.abs(angles_from_exp - prev_direction)
                                angle_diff_from_prev = np.minimum(angle_diff_from_prev, 360 - angle_diff_from_prev)
                                candidates = candidates[angle_diff_from_prev <= cone_half_angle]
                        
                        if len(candidates) > 0:
                            # Выбираем кандидата с минимальным расстоянием до ожидаемой позиции
                            min_dist_idx = candidates['dist'].idxmin()
                            cluster = candidates.loc[[min_dist_idx]]
                        else:
                            # Нет кандидатов в конусе - трек заканчивается
                            cluster = None
                    else:
                        # Нет кандидатов в радиусе - трек заканчивается
                        cluster = None
                else:
                    # Недостаточно истории для расчета скорости (меньше 3 точек)
                    # Используем стандартный поиск без учета скорости
                    x_center, y_center = get_next_loc_cases(CS_tracks_list, ds, dt_step, i, our_time, 
                                                             x_init, y_init, hw_scale, rad_init, 
                                                             speed_level, 'no_speed')
                    local_max['dist'] = dist(x_center, y_center, local_max.x, local_max.y)
                    candidates = local_max[local_max['dist'] < hw_scale * rad_init]
                    
                    if len(candidates) > 0:
                        min_dist = np.nanmin(candidates['dist'])
                        cluster = candidates[candidates['dist'] == min_dist]
                    else:
                        cluster = None
                        
            else:
                # Старая логика для других CVS_speed
                x, y = get_next_loc_cases(CS_tracks_list, ds, dt_step, i, our_time, x_init, y_init, 
                                          hw_scale, rad_init, speed_level, CVS_speed)
                
                local_max['dist'] = dist(x, y, local_max.x, local_max.y)
                candidates = local_max[local_max['dist'] < hw_scale * rad_init]
                
                if len(candidates) > 0:
                    min_dist = np.nanmin(candidates['dist'])
                    cluster = candidates[candidates['dist'] == min_dist]
                else:
                    cluster = None
                    
            # Обработка найденного кластера
            if cluster is not None and len(cluster) > 0:
                x_c = int(cluster.x.values[0])
                y_c = int(cluster.y.values[0])
                rad = cluster.rad_eff.values[0]
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
                if len(local_max) != 0:
                # if len(stat_Q) != 0:
                    
                    # stat_Q = stat_Q[stat_Q['cluster'] != cluster['cluster'].values[0]]
                    # local_max = local_max[local_max['cluster'] != cluster['cluster'].values[0]]
                    local_max = local_max[local_max.index != cluster.index.values[0]]
                    
                    
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
                

    # if len(stat_Q) != 0:
        # CS_tracks_list, clstr_len = track_init(ds, clstr_len, data_type, stat_Q.reset_index(), CS_tracks_list_new, our_time, time_unit)
    
    if len(local_max) != 0:
        CS_tracks_list, clstr_len = track_init(ds, clstr_len, data_type, local_max.reset_index(), CS_tracks_list_new, our_time, time_name)

    # sorted_CS_tracks_list = sorted(CS_tracks_list, key=lambda x: len(x['track_len']), reverse=True)

    CS_tracks_list.sort(key=lambda x: (x['crit'][-1]), reverse=True)

    return cluster_idx, CS_tracks_list, clstr_len