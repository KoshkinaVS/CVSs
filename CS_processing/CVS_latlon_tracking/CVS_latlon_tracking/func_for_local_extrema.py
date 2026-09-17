"""
ERA5 lat/lon версия func_for_local_extrema.py (tracking_type = 'tracking_local_global').

Логика метода не изменена: кандидаты на продолжение трека ищутся среди
локальных экстремумов (get_stat_local_max), а новые треки заводятся из
оставшихся глобальных максимумов (get_stat_global_max) — как и в исходном
файле, при совпадении кластер убирается из обоих пулов.

Что изменено относительно ../func_for_local_extrema.py:
  - расстояние "текущая точка трека -> кандидат" считается по честным
    географическим координатам (haversine, км), а не по индексам сетки —
    см. latlon_utils.py за подробным объяснением, почему это важно для
    ERA5 на высоких широтах;
  - порог поиска (hw_scale * rad_init) переводится из клеток сетки в км
    на текущей широте перед сравнением;
  - lon/lat совпавшей точки берутся напрямую из таблицы local_max (для
    ERA5 они там уже есть — get_stat_local_max сама их считает), без
    повторного обращения к ds.XLONG/ds.XLAT (WRF-специфичным полям,
    которых в ERA5-датасете просто нет);
  - track_len копится в километрах (реальная длина смещения), а не в
    клетках сетки.

Формат CS_tracks_list (x, y, lon, lat, rad, crit, t, time, track_len) и
формат сохраняемых треков (save_track_csv/save_track_txt) не меняются —
эти функции по-прежнему берутся из step_of_tracking.py.
"""
from step_of_tracking import *
from latlon_utils import haversine_distance_km, index_to_lonlat, rad_gridpoints_to_km, nearest_within_radius


def step_of_tracking(cluster_idx, CS_tracks_list, clstr_len, ds, data_type, path_data_dir, our_time,
                      CVS_speed,
                      circ='C', dt_step=3600):

    if data_type != 'ERA5':
        raise NotImplementedError(
            "ERA5_latlon/func_for_local_extrema.py рассчитан только на data_type='ERA5' "
            "(регулярная широтно-долготная сетка); для остальных типов данных "
            "используйте оригинальный ../func_for_local_extrema.py"
        )

    stat_Q = get_stat_global_max(ds, our_time, circ)
    local_max = get_stat_local_max(ds, our_time, circ)

    # радиус поиска (множитель к rad_init)
    hw_scale = 1.5
    # граница домена (отступ в клетках)
    bound_size = 0

    for i in range(len(CS_tracks_list)):

        x_init = CS_tracks_list[i]['x'][-1]
        y_init = CS_tracks_list[i]['y'][-1]
        lon_init = CS_tracks_list[i]['lon'][-1]
        lat_init = CS_tracks_list[i]['lat'][-1]

        rad_init = CS_tracks_list[i]['rad'][-1]
        crit_init = CS_tracks_list[i]['crit'][-1]

        # отступаем bound_size клеточек от границы
        if ~np.isnan(x_init) and x_init < (len(ds[x_unit]) - bound_size) and y_init < (len(ds[y_unit]) - bound_size) and x_init > bound_size - 1 and y_init > bound_size - 1:

            # ожидаемая следующая точка трека (с учётом CVS_speed), в индексах сетки
            x, y = get_next_loc_cases(CS_tracks_list, ds, dt_step, i, our_time, x_init, y_init,
                                       rad_init, CVS_speed)
            lon_exp, lat_exp = index_to_lonlat(ds, x, y)

            radius_km = hw_scale * rad_gridpoints_to_km(rad_init, lat_exp, ds)
            cluster = nearest_within_radius(local_max, lon_exp, lat_exp, radius_km)

            if cluster is not None:

                x_c = int(cluster['x'])
                y_c = int(cluster['y'])
                lon_c = float(cluster['lon'])
                lat_c = float(cluster['lat'])

                rad = cluster['rad_eff']

                track_len_new = haversine_distance_km(lon_init, lat_init, lon_c, lat_c)

                CS_tracks_list[i]['x'].append(x_c)
                CS_tracks_list[i]['y'].append(y_c)

                CS_tracks_list[i]['lon'].append(lon_c)
                CS_tracks_list[i]['lat'].append(lat_c)

                CS_tracks_list[i]['rad'].append(rad)
                CS_tracks_list[i]['crit'].append(cluster['crit'])

                track_l = CS_tracks_list[i]['track_len'][-1]
                CS_tracks_list[i]['track_len'].append(track_l + track_len_new)

                CS_tracks_list[i]['t'].append(our_time)
                CS_tracks_list[i]['time'].append(ds[time_name][our_time].values)

                # удаляем строчку с использованными координатами из обоих пулов
                if len(stat_Q) != 0:
                    stat_Q = stat_Q[stat_Q['cluster'] != cluster['cluster']]
                    local_max = local_max[local_max['cluster'] != cluster['cluster']]

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
                cluster_idx = save_track_csv(cluster_idx, CS, path_data_dir)
                save_track_txt(cluster_idx, CS, path_data_dir)

    if len(stat_Q) != 0:
        CS_tracks_list, clstr_len = track_init(ds, clstr_len, data_type, stat_Q.reset_index(), CS_tracks_list_new, our_time)

    CS_tracks_list.sort(key=lambda x: (x['crit'][-1]), reverse=True)

    return cluster_idx, CS_tracks_list, clstr_len
