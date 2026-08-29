from step_of_tracking import *
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

# Константы для метрики на сфере
R_EARTH = 6371000  # Радиус Земли в метрах
DEG_TO_RAD = np.pi / 180

def calculate_distance_km(lon1, lat1, lon2, lat2):
    """
    Вычисление расстояния между двумя точками на сфере в километрах
    """
    # Преобразование в радианы
    lon1_rad = np.radians(lon1)
    lat1_rad = np.radians(lat1)
    lon2_rad = np.radians(lon2)
    lat2_rad = np.radians(lat2)
    
    # Вычисление гаверсинуса
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R_EARTH * c / 1000  # расстояние в километрах

def get_search_radius_km(speed_ms, dt_step, base_radius_km, hw_scale=1.5):
    """
    Расчет динамического радиуса поиска в километрах
    
    Parameters:
    -----------
    speed_ms : float
        Скорость движения в м/с
    dt_step : float
        Шаг по времени в секундах
    base_radius_km : float
        Базовый радиус области в километрах
    hw_scale : float
        Масштабный коэффициент
    """
    # Ожидаемое смещение за время dt
    expected_displacement_km = speed_ms * dt_step / 1000
    
    # Радиус поиска = ожидаемое смещение + базовый радиус * масштаб
    search_radius_km = expected_displacement_km + base_radius_km * hw_scale
    
    return max(search_radius_km, base_radius_km * hw_scale / 2)

# Основная функция
def step_of_tracking(cluster_idx, CS_tracks_list, clstr_len, ds, data_type, path_data_dir, our_time, 
                     speed_level, CVS_speed,
                     circ='C', dt_step=3600, multi_levels=False):
    
    # Получаем координаты локальных максимумов в географических координатах
    local_max = get_stat_local_max(ds, our_time, circ)
    
    # Конвертируем в географические координаты
    if data_type == 'ERA5':
        # Получаем широты и долготы для индексов сетки
        lons = ds.longitude.values
        lats = ds.latitude.values
        
        # Добавляем географические координаты в local_max
        local_max['lon'] = local_max['x'].apply(lambda x: lons[int(x)] if not np.isnan(x) else np.nan)
        local_max['lat'] = local_max['y'].apply(lambda y: lats[int(y)] if not np.isnan(y) else np.nan)
        
        # Конвертируем радиус из пикселей в километры
        # Радиус в local_max предположительно в градусах или пикселях
        # Переводим в километры на средней широте
        if 'rad_eff' in local_max.columns:
            mean_lat = np.nanmean(local_max['lat'].values)
            deg_to_km = (R_EARTH * DEG_TO_RAD / 1000) * np.cos(np.radians(mean_lat))
            local_max['rad_km'] = local_max['rad_eff'] * deg_to_km
    
    # Параметры поиска
    hw_scale = 1.5
    bound_size = 0
    
    for i in range(len(CS_tracks_list)):
        
        # Последняя известная позиция (в географических координатах)
        lon_init = CS_tracks_list[i]['lon'][-1]
        lat_init = CS_tracks_list[i]['lat'][-1]
        rad_init_km = CS_tracks_list[i]['rad_km'][-1] if 'rad_km' in CS_tracks_list[i] else CS_tracks_list[i]['rad'][-1]
        crit_init = CS_tracks_list[i]['crit'][-1]
        
        # Проверка валидности позиции
        if ~np.isnan(lon_init) and ~np.isnan(lat_init):
            # Конвертируем в индексы сетки для проверки границ
            lons = ds.longitude.values
            lats = ds.latitude.values
            
            # Находим ближайшие индексы
            x_init = np.argmin(np.abs(lons - lon_init))
            y_init = np.argmin(np.abs(lats - lat_init))
            
            # Проверка границ домена
            if (x_init < (len(lons) - bound_size) and y_init < (len(lats) - bound_size) 
                and x_init > bound_size - 1 and y_init > bound_size - 1):
                
                # Новая логика для avg_cone
                if CVS_speed == 'avg_cone':
                    # Проверяем, что есть минимум 2 предыдущих точки для расчета скорости
                    if len(CS_tracks_list[i]['lon']) >= 2:
                        # Берем предыдущие точки
                        lon_prev = CS_tracks_list[i]['lon'][-1]
                        lat_prev = CS_tracks_list[i]['lat'][-1]
                        lon_prev2 = CS_tracks_list[i]['lon'][-2]
                        lat_prev2 = CS_tracks_list[i]['lat'][-2]
                        
                        # Рассчитываем скорость перемещения (км/с)
                        dist_km = calculate_distance_km(lon_prev, lat_prev, lon_prev2, lat_prev2)
                        u = dist_km / dt_step if dt_step > 0 else 0  # скорость в км/с
                        
                        # Направление движения
                        # Для простоты используем среднюю скорость (требует уточнения по компонентам)
                        speed_km_s = u
                        
                        # Ожидаемая позиция через dt_step (в километрах)
                        # Нужно пересчитать в смещение по широте/долготе
                        # Это приближение работает для небольших расстояний
                        bearing = calculate_bearing(lon_prev2, lat_prev2, lon_prev, lat_prev)
                        lon_exp, lat_exp = calculate_new_position(lon_prev, lat_prev, speed_km_s * dt_step, bearing)
                        
                        # Получаем параметры конуса на основе скорости
                        avg_speed = speed_km_s * 1000  # конвертируем в м/с для функции
                        cone_params = get_search_cone_parameters(avg_speed, dt_step, hw_scale, rad_init_km / 1000)
                        
                        # Динамический радиус поиска (км)
                        search_radius_km = get_search_radius_km(avg_speed, dt_step, rad_init_km, hw_scale)
                        
                        # Вычисляем расстояние в км от ожидаемой позиции до всех кандидатов
                        local_max['dist_km'] = local_max.apply(
                            lambda row: calculate_distance_km(lon_exp, lat_exp, row['lon'], row['lat']), 
                            axis=1
                        )
                        
                        # Базовый отбор по радиусу
                        candidates = local_max[local_max['dist_km'] < search_radius_km].copy()
                        
                        if len(candidates) > 0:
                            # Вычисляем направление предыдущего движения
                            prev_direction = bearing
                            
                            # Для каждого кандидата вычисляем изменение направления
                            angle_changes = []
                            for idx, candidate in candidates.iterrows():
                                # Направление от предыдущей точки к кандидату
                                candidate_bearing = calculate_bearing(
                                    lon_prev, lat_prev, candidate['lon'], candidate['lat']
                                )
                                
                                # Изменение направления
                                angle_change = abs(candidate_bearing - prev_direction)
                                angle_change = min(angle_change, 360 - angle_change)
                                angle_changes.append(angle_change)
                            
                            candidates['angle_change'] = angle_changes
                            
                            # Фильтруем по максимальному изменению направления
                            candidates = candidates[candidates['angle_change'] <= cone_params['max_angle_change']]
                            
                            if len(candidates) > 0:
                                # Выбираем кандидата с минимальным расстоянием
                                min_dist_idx = candidates['dist_km'].idxmin()
                                cluster = candidates.loc[[min_dist_idx]]
                            else:
                                cluster = None
                        else:
                            cluster = None
                    else:
                        # Недостаточно истории - используем стандартный поиск
                        x_center, y_center = get_next_loc_cases(
                            CS_tracks_list, ds, dt_step, i, our_time, 
                            x_init, y_init, hw_scale, rad_init_km / deg_to_km, 
                            speed_level, 'no_speed'
                        )
                        
                        # Конвертируем в географические координаты
                        lon_center = lons[int(x_center)]
                        lat_center = lats[int(y_center)]
                        
                        local_max['dist_km'] = local_max.apply(
                            lambda row: calculate_distance_km(lon_center, lat_center, row['lon'], row['lat']), 
                            axis=1
                        )
                        candidates = local_max[local_max['dist_km'] < hw_scale * rad_init_km]
                        
                        if len(candidates) > 0:
                            min_dist = np.nanmin(candidates['dist_km'])
                            cluster = candidates[candidates['dist_km'] == min_dist]
                        else:
                            cluster = None
                        
                else:
                    # Старая логика для других CVS_speed
                    x, y = get_next_loc_cases(CS_tracks_list, ds, dt_step, i, our_time, x_init, y_init, 
                                              hw_scale, rad_init_km / deg_to_km, speed_level, CVS_speed)
                    
                    lon_center = lons[int(x)]
                    lat_center = lats[int(y)]
                    
                    local_max['dist_km'] = local_max.apply(
                        lambda row: calculate_distance_km(lon_center, lat_center, row['lon'], row['lat']), 
                        axis=1
                    )
                    candidates = local_max[local_max['dist_km'] < hw_scale * rad_init_km]
                    
                    if len(candidates) > 0:
                        min_dist = np.nanmin(candidates['dist_km'])
                        cluster = candidates[candidates['dist_km'] == min_dist]
                    else:
                        cluster = None
                
                # Обработка найденного кластера
                if cluster is not None and len(cluster) > 0:
                    # Найденные координаты (в градусах)
                    lon_c = cluster.lon.values[0]
                    lat_c = cluster.lat.values[0]
                    
                    # Находим ближайшие индексы сетки
                    x_c = np.argmin(np.abs(lons - lon_c))
                    y_c = np.argmin(np.abs(lats - lat_c))
                    
                    rad_km = cluster.rad_km.values[0] if 'rad_km' in cluster.columns else cluster.rad_eff.values[0] * deg_to_km
                    track_len_new = calculate_distance_km(lon_c, lat_c, lon_init, lat_init)
                    
                    # Сохраняем индексы сетки для совместимости
                    CS_tracks_list[i]['x'].append(x_c)
                    CS_tracks_list[i]['y'].append(y_c)
                    
                    # Сохраняем географические координаты
                    CS_tracks_list[i]['lon'].append(lon_c)
                    CS_tracks_list[i]['lat'].append(lat_c)
                    
                    # Сохраняем радиус в километрах
                    CS_tracks_list[i]['rad_km'].append(rad_km)
                    CS_tracks_list[i]['rad'] = CS_tracks_list[i].get('rad', [])
                    CS_tracks_list[i]['rad'].append(rad_km / deg_to_km)  # обратно в градусы/пиксели
                    
                    CS_tracks_list[i]['crit'].append(cluster.crit.values[0])
                    
                    # Обновляем длину трека
                    track_l = CS_tracks_list[i]['track_len'][-1] if CS_tracks_list[i]['track_len'] else 0
                    CS_tracks_list[i]['track_len'].append(track_l + track_len_new)
                    
                    # Обновляем время
                    CS_tracks_list[i]['t'].append(our_time)
                    CS_tracks_list[i]['time'].append(ds[time_name][our_time].values)
                    
                    # Удаляем использованную строчку
                    if len(local_max) != 0:
                        local_max = local_max[local_max.index != cluster.index.values[0]]
                    
                else:
                    # Не найдено продолжение - добавляем NaN
                    CS_tracks_list[i]['x'].append(np.nan)
                    CS_tracks_list[i]['y'].append(np.nan)
                    CS_tracks_list[i]['lon'].append(np.nan)
                    CS_tracks_list[i]['lat'].append(np.nan)
                    CS_tracks_list[i]['rad'].append(np.nan)
                    CS_tracks_list[i]['rad_km'].append(np.nan)
                    CS_tracks_list[i]['crit'].append(np.nan)
                    CS_tracks_list[i]['t'].append(np.nan)
                    CS_tracks_list[i]['time'].append(np.nan)
                    CS_tracks_list[i]['track_len'].append(np.nan)
            
            else:
                # Выход за границы
                add_nan_entries(CS_tracks_list[i])
        else:
            # Невалидная позиция
            add_nan_entries(CS_tracks_list[i])
    
    # Фильтрация завершенных треков
    CS_tracks_list_new = []
    for CS in CS_tracks_list:
        if not np.isnan(CS['lon'][-1]) and not np.isnan(CS['lat'][-1]):
            CS_tracks_list_new.append(CS)
        else:
            if np.sum(~np.isnan(CS['lon'])) >= 3:
                cluster_idx = save_track_csv(cluster_idx, CS, path_data_dir)
                save_track_txt(cluster_idx, CS, path_data_dir)
    
    # Инициализация новых треков
    if len(local_max) != 0:
        CS_tracks_list, clstr_len = track_init(
            ds, clstr_len, data_type, local_max.reset_index(), 
            CS_tracks_list_new, our_time, time_name
        )
    
    # Сортировка по критичности
    if CS_tracks_list:
        CS_tracks_list.sort(key=lambda x: x['crit'][-1] if not np.isnan(x['crit'][-1]) else 0, reverse=True)
    
    return cluster_idx, CS_tracks_list, clstr_len

# Вспомогательные функции
def calculate_bearing(lon1, lat1, lon2, lat2):
    """
    Вычисление начального азимута между двумя точками на сфере
    """
    lon1_rad = np.radians(lon1)
    lon2_rad = np.radians(lon2)
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    
    dlon = lon2_rad - lon1_rad
    
    x = np.sin(dlon) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
    
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360

def calculate_new_position(lon, lat, distance_km, bearing):
    """
    Вычисление новой позиции на сфере по начальной точке, расстоянию и азимуту
    """
    R = R_EARTH / 1000  # Радиус Земли в километрах
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    bearing_rad = np.radians(bearing)
    
    angular_distance = distance_km / R
    
    lat_new_rad = np.arcsin(
        np.sin(lat_rad) * np.cos(angular_distance) + 
        np.cos(lat_rad) * np.sin(angular_distance) * np.cos(bearing_rad)
    )
    
    lon_new_rad = lon_rad + np.arctan2(
        np.sin(bearing_rad) * np.sin(angular_distance) * np.cos(lat_rad),
        np.cos(angular_distance) - np.sin(lat_rad) * np.sin(lat_new_rad)
    )
    
    return np.degrees(lon_new_rad) % 360, np.degrees(lat_new_rad)

def add_nan_entries(track):
    """Добавление NaN записей в трек"""
    track['x'].append(np.nan)
    track['y'].append(np.nan)
    track['lon'].append(np.nan)
    track['lat'].append(np.nan)
    track['rad'].append(np.nan)
    if 'rad_km' in track:
        track['rad_km'].append(np.nan)
    track['crit'].append(np.nan)
    track['t'].append(np.nan)
    track['time'].append(np.nan)
    track['track_len'].append(np.nan)