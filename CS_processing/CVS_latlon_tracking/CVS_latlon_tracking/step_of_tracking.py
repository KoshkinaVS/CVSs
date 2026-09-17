"""
Общая инфраструктура трекинга КВС по данным ERA5 (регулярная широтно-
долготная сетка). Её используют все func_for_*.py в этой папке через
`from step_of_tracking import *`.

Это ERA5-only копия для CVS_latlon_tracking/. Общий файл
../../CVS_tracking/after_70RAE/step_of_tracking.py обслуживает и другие
ваши проекты (WRF HiRes/LoRes/GPN/SMP, GLORYS, ALT) и хранит ветки под
каждый из них — здесь эти ветки убраны, оставлена только ERA5. Если
понадобится трекинг по другим типам данных, берите общий файл, не эту
копию. Полный список отличий — CHANGELOG.md рядом.
"""
import os
import warnings
from functools import partial
from typing import Callable, Dict

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")


# === Параметры сетки/данных ERA5 =====================================
# ВАЖНО, проверьте перед первым запуском на сервере: time_name — это имя
# временнОго измерения в реальных .nc-файлах после DBSCAN/R2D. Здесь стоит
# 'Time' (как было в общем step_of_tracking.py), но ERA5/config_ERA5.json
# (конфиг для расчёта самого R2D/DBSCAN) называет его 'time' с маленькой
# буквы — если у вас в файлах измерение 'time', замените константу ниже,
# иначе ds.isel({time_name: ...}) упадёт с KeyError. Проверить можно так:
#   python3 -c "import xarray as xr; print(xr.open_dataset('файл.nc').dims)"
time_name = 'Time'

x_unit = 'longitude'
y_unit = 'latitude'
level_unit = 'level'
u_unit = 'u'
v_unit = 'v'

# Имя поля критерия завихренности в .nc после DBSCAN (см. ERA5/config_ERA5.json:
# "name_crit") и индекс единственного уровня давления в этих файлах. Вынесены
# в константы, а не хардкожены заново в каждой функции ниже (и в
# plot_tracking_results.py, который их тоже импортирует) — раньше 'R2D' и 0
# повторялись как магические литералы в 3 местах.
CRIT_FIELD = 'R2D'
LEVEL_INDEX = 0

dt_step = 3600  # секунд, шаг по времени ERA5 (1 час)

# Шаг сетки ERA5 в метрах (0.25° ~ 111 км/градус). Используется только в
# get_new_loc_mean_speed для перевода фоновой скорости (м/с) в смещение по
# индексам сетки (CVS_speed='bg_speed'/'adv_bg_speed'). В отличие от
# latlon_utils.rad_gridpoints_to_km здесь НЕТ поправки на cos(широта) по
# долготе — см. README_adaptation.md, раздел "что осознанно не менялось";
# это давняя, не lat/lon-специфичная часть трекинга, вынесенная за рамки
# текущей адаптации.
dist_m = 111_000 * 0.25

circ = 'C'  # 'C' — циклонические КВС, 'AC' — антициклонические


# === Регистрация методов трекинга ====================================
TRACKING_METHODS: Dict[str, Callable] = {
    'tracking_local_global': None,
    'tracking_local_2_phase': None,
    'tracking_global_only': None,
    'tracking_local_only': None,
}


def initialize_tracking_methods():
    """Импортирует и регистрирует все методы трекинга (func_for_*.py)."""
    from func_for_local_extrema import step_of_tracking as local_extrema_tracking
    from func_for_local_2_phase import step_of_tracking as local_2_phase_tracking
    from func_for_global_only import step_of_tracking as global_only_tracking
    from func_for_local_only import step_of_tracking as local_only_tracking

    TRACKING_METHODS['tracking_local_global'] = local_extrema_tracking
    TRACKING_METHODS['tracking_local_2_phase'] = local_2_phase_tracking
    TRACKING_METHODS['tracking_global_only'] = global_only_tracking
    TRACKING_METHODS['tracking_local_only'] = local_only_tracking


def get_tracking_function(tracking_type: str, **kwargs):
    """Возвращает функцию трекинга с частично применёнными параметрами."""
    if tracking_type not in TRACKING_METHODS:
        raise ValueError(f"Unknown tracking type: {tracking_type}")
    return partial(TRACKING_METHODS[tracking_type], **kwargs)


def initialize_tracks(ds, tracking_type, data_type, CS_tracks_list, our_time=0, circ=circ):
    """Инициализация треков (на t=0 первого файла) в зависимости от типа трекинга."""
    if tracking_type == 'tracking_local_only':
        local_max = get_stat_local_max(ds, our_time, circ)
        return track_init(ds, 0, data_type, local_max, CS_tracks_list, our_time)
    elif tracking_type == 'tracking_local_2_phase':
        r2d_coords = get_all_coords(ds, our_time, circ)
        local_max = get_stat_local_max(ds, our_time, circ)
        local_max = update_rad(local_max, r2d_coords)
        return track_init(ds, 0, data_type, local_max, CS_tracks_list, our_time)
    else:
        stat_Q = get_stat_global_max(ds, our_time, circ)
        return track_init(ds, 0, data_type, stat_Q, CS_tracks_list, our_time)


def save_track_csv(cluster_idx, TC, path_data_dir):
    start_date = TC['time'][0]
    if isinstance(start_date, np.datetime64):
        start_date = pd.Timestamp(start_date)

    year, month, day, hour = start_date.year, start_date.month, start_date.day, start_date.hour

    df_track = pd.DataFrame({
        "t": TC['t'][:-1],
        "datetime": TC['time'][:-1],
        "x": TC['x'][:-1],
        "y": TC['y'][:-1],
        "lat": TC['lat'][:-1],
        "lon": TC['lon'][:-1],
        "rad": TC['rad'][:-1],
        "crit": TC['crit'][:-1],
        "track_len": TC['track_len'][:-1],
    })

    month_dir = f"{path_data_dir}/{year}-{month:02d}/"
    os.makedirs(month_dir, exist_ok=True)

    df_track.to_csv(f'{month_dir}/{cluster_idx:06d}_track_{year}-{month:02d}-{day:02d}T{hour:02d}.csv')
    cluster_idx += 1

    return cluster_idx


def save_track_txt(cluster_idx, TC, path_data_dir):
    """Сохраняет трек в формате TempestExtremes (единый годовой файл на месяц)."""
    start_date = TC['time'][0]
    if isinstance(start_date, np.datetime64):
        start_date = pd.Timestamp(start_date)

    year, month = start_date.year, start_date.month

    yearly_dir = f"{path_data_dir}/txt_yearly/"
    os.makedirs(yearly_dir, exist_ok=True)
    yearly_file = os.path.join(yearly_dir, f'tracks_{year}-{month:02d}.txt')

    # Считаем только точки, где t и x не NaN
    npoints = sum(
        1 for i in range(len(TC['t']))
        if not np.isnan(TC['t'][i]) and not np.isnan(TC['x'][i])
    )

    with open(yearly_file, 'a') as f:
        f.write(f"start\t{npoints}\t{start_date.year}\t{start_date.month}\t{start_date.day}\t{start_date.hour}\n")

        for i in range(len(TC['t']) - 1):
            if np.isnan(TC['t'][i]):
                continue

            time_point = TC['time'][i]
            if isinstance(time_point, np.datetime64):
                time_point = pd.Timestamp(time_point)

            f.write(f"\t{int(TC['x'][i])}\t{int(TC['y'][i])}\t"
                    f"{TC['lon'][i]:.6f}\t{TC['lat'][i]:.6f}\t"
                    f"{TC['crit'][i]:.6e}\t{TC['rad'][i]:.6e}\t"
                    f"{int(TC['track_len'][i])}\t"
                    f"{time_point.year}\t{time_point.month}\t{time_point.day}\t{time_point.hour}\n")


def get_all_coords(ds, t, circ):
    """Координаты (x,y индексы + lon/lat) всех точек кластеров R2D на шаге t."""
    ds_mini = ds.isel({time_name: t, level_unit: LEVEL_INDEX})[['cluster', CRIT_FIELD]]
    X_arr = ds_mini.to_dataframe().dropna(how='any')
    index_df = X_arr['cluster'].index.to_frame(index=False)

    coords = pd.DataFrame({
        'y': X_arr.index.codes[0],  # индекс по latitude
        'x': X_arr.index.codes[1],  # индекс по longitude
        'lat': index_df['latitude'].values,
        'lon': index_df['longitude'].values,
        'crit': X_arr[CRIT_FIELD].values,
        'cluster': X_arr['cluster'].values,
    })

    coords = coords[coords['cluster'] > 0] if circ == 'C' else coords[coords['cluster'] < 0]
    return coords.reset_index(drop=True)


def get_stat_local_max(ds, t, circ):
    """Локальные экстремумы R2D (для tracking_local_only / tracking_local_2_phase)."""
    ds_mini = ds.isel({time_name: t, level_unit: LEVEL_INDEX})[
        ['local_extr_crit', 'local_extr_cluster', 'local_extr_rad_eff']
    ]
    X_arr = ds_mini.to_dataframe().dropna(how='any')
    index_df = X_arr['local_extr_crit'].index.to_frame(index=False)

    coords = pd.DataFrame({
        'y': X_arr.index.codes[0],
        'x': X_arr.index.codes[1],
        'lat': index_df['latitude'].values,
        'lon': index_df['longitude'].values,
        'crit': X_arr['local_extr_crit'].values,
        'cluster': X_arr['local_extr_cluster'].values,
        'rad_eff': X_arr['local_extr_rad_eff'].values,
    })

    coords = coords.dropna(how='any')
    coords = coords[coords['crit'] > 0] if circ == 'C' else coords[coords['crit'] < 0]
    return coords.sort_values(by='crit', ascending=False).reset_index(drop=True)


def get_stat_global_max(ds, t, circ):
    """Глобальные максимумы R2D по кластеру (для tracking_global_only / tracking_local_global)."""
    ds_mini = ds.isel({time_name: t, level_unit: LEVEL_INDEX})
    X_arr = ds_mini.to_dataframe().dropna(how='any')
    index_df = X_arr['center'].index.to_frame(index=False)

    coords = pd.DataFrame({
        'y': X_arr.index.codes[0],
        'x': X_arr.index.codes[1],
        'lat': index_df['latitude'].values,
        'lon': index_df['longitude'].values,
        'crit': X_arr['center'].values,
        'cluster': X_arr['center_cluster'].values,
        'rad_eff': X_arr['rad_eff'].values,
    })

    coords = coords.dropna(how='any')
    coords = coords[coords['crit'] > 0] if circ == 'C' else coords[coords['crit'] < 0]
    return coords.sort_values(by='crit', ascending=False).reset_index(drop=True)


def find_local_minimum_near(bp, crit_field, blue_points):
    """Ищет ближайший локальный минимум по пути от blue-точки к красной (для update_rad)."""
    dists = np.linalg.norm(blue_points - bp, axis=1)
    dists = np.where(dists == 0, 999, dists)
    min_idx = np.argmin(dists)
    point_near = blue_points[min_idx]

    path = np.linspace(bp, point_near, num=100)

    tree = cKDTree(crit_field[:, :2])
    _, indices = tree.query(path)
    values = crit_field[indices, 2]

    min_idx = np.argmin(np.abs(values))
    return path[min_idx]


def update_rad(local_max, r2d_coords, big_CVS=True):
    """Уточняет rad_eff локальных экстремумов по границе кластера в R2D (в клетках сетки)."""
    clusters = np.unique(local_max['cluster'])
    local_rad_df = pd.DataFrame(data=None)

    for cluster in clusters:
        CVS = r2d_coords[r2d_coords['cluster'] == cluster]
        CVS_extr = local_max[local_max['cluster'] == cluster]

        CVS_max_rad = CVS_extr['rad_eff'].max()

        blue_points = CVS_extr[['x', 'y']].values
        crit_field = CVS[['x', 'y', 'crit']].values

        if len(blue_points) > 1:
            green_points = [find_local_minimum_near(bp, crit_field, blue_points) for bp in blue_points]
            radii = np.linalg.norm(blue_points - green_points, axis=1)
        else:
            radii = CVS_extr['rad_eff'].values

        if big_CVS:
            max_crit_idx = CVS_extr['crit'].idxmax()

            radii_modified = radii.copy()
            if max_crit_idx in CVS_extr.index:
                pos_in_cluster = np.where(CVS_extr.index == max_crit_idx)[0][0]
                radii_modified[pos_in_cluster] = CVS_max_rad

            rad_df = pd.DataFrame(radii_modified, columns=['rad_eff'], index=CVS_extr.index)
        else:
            cluster_idx = local_max[local_max['cluster'] == cluster].index
            rad_df = pd.DataFrame(radii, columns=['rad_eff'], index=cluster_idx)

        local_rad_df = pd.concat([local_rad_df, rad_df])

    local_max['rad_eff'] = local_rad_df['rad_eff'].sort_index().values
    # запрещаем радиусам вырождаться
    local_max.loc[local_max['rad_eff'] < 1, 'rad_eff'] = 1.

    return local_max


def dist(p1_x, p1_y, p2_x, p2_y):
    """Евклидово расстояние между точками в индексах сетки."""
    return np.sqrt((p1_x - p2_x) ** 2 + (p1_y - p2_y) ** 2)


def get_new_loc_mean_speed(ds, dt, x, y, hw, t):
    """Перемещение в новую точку исходя из средней скорости потока (фоновая адвекция)."""
    y_int, x_int = int(np.round(y)), int(np.round(x))

    y_in, x_in = max(y_int - hw, 0), max(x_int - hw, 0)
    y_out = -1 if (y_int + hw) >= len(ds[y_unit]) else y_int + hw
    x_out = -1 if (x_int + hw) >= len(ds[x_unit]) else x_int + hw

    u, v = compute_mean_uv_for_track(ds, t, y_in, y_out, x_in, x_out)

    x_new = x + u / dist_m * dt
    y_new = y + v / dist_m * dt

    x_new = min(max(x_new, 0), len(ds[x_unit]) - 1)
    y_new = min(max(y_new, 0), len(ds[y_unit]) - 1)

    return x_new, y_new


def compute_mean_uv_for_track(ds, idx, y_in, y_out, x_in, x_out, level=0):
    """Средняя скорость потока в окне [y_in:y_out, x_in:x_out] на шаге idx."""
    u = ds[u_unit][idx, level, y_in:y_out, x_in:x_out]
    v = ds[v_unit][idx, level, y_in:y_out, x_in:x_out]
    return np.nanmean(u), np.nanmean(v)


def get_new_loc_CS_speed(ds, dt, x_pr, y_pr, x_prpr, y_prpr):
    """Перемещение в новую точку исходя из скорости самого КВС между предыдущими шагами."""
    u = (x_pr - x_prpr) / dt
    v = (y_pr - y_prpr) / dt

    x_new = x_pr + u * dt
    y_new = y_pr + v * dt

    x_new = min(max(x_new, 0), len(ds[x_unit]) - 1)
    y_new = min(max(y_new, 0), len(ds[y_unit]) - 1)

    return x_new, y_new


# Полуширина окна (в клетках сетки) для усреднения фонового ветра в
# get_new_loc_mean_speed. Раньше это был параметр hw_scale самой
# get_next_loc_cases — но она получала hw_scale=1.5 от вызывающих
# func_for_*.py (порог поиска кандидата) и тут же затирала его этой же
# константой прямо в первой строке функции, так что переданное значение
# никогда не использовалось. Здесь это разделено явно: у поиска кандидата
# и у окна усреднения ветра — разные, независимые масштабы.
BG_SPEED_BOX_HW_SCALE = 3


def get_next_loc_cases(CS_tracks_list, ds, dt_step, i, our_time, x_init, y_init, rad_init, CVS_speed='no_speed'):
    """Ожидаемая следующая точка трека (в индексах сетки) с учётом CVS_speed."""
    if CVS_speed == 'adv_speed':
        if len(CS_tracks_list[i]['x']) == 1:
            x, y = x_init, y_init
        else:
            x_prpr, y_prpr = CS_tracks_list[i]['x'][-2], CS_tracks_list[i]['y'][-2]
            x, y = get_new_loc_CS_speed(ds, dt_step, x_init, y_init, x_prpr, y_prpr)

    elif CVS_speed == 'bg_speed':
        if len(CS_tracks_list[i]['x']) == 1:
            hw = int(BG_SPEED_BOX_HW_SCALE * np.round(rad_init))
            x, y = get_new_loc_mean_speed(ds, dt_step, x_init, y_init, hw, our_time)
        else:
            x_prpr, y_prpr = CS_tracks_list[i]['x'][-2], CS_tracks_list[i]['y'][-2]
            x, y = get_new_loc_CS_speed(ds, dt_step, x_init, y_init, x_prpr, y_prpr)

    elif CVS_speed == 'adv_bg_speed':
        hw = int(BG_SPEED_BOX_HW_SCALE * np.round(rad_init))
        if len(CS_tracks_list[i]['x']) == 1:
            x, y = get_new_loc_mean_speed(ds, dt_step, x_init, y_init, hw, our_time)
        else:
            x_prpr, y_prpr = CS_tracks_list[i]['x'][-2], CS_tracks_list[i]['y'][-2]
            x_adv, y_adv = get_new_loc_CS_speed(ds, dt_step, x_init, y_init, x_prpr, y_prpr)
            x_bg, y_bg = get_new_loc_mean_speed(ds, dt_step, x_init, y_init, hw, our_time)
            x, y = (x_adv + x_bg) / 2.0, (y_adv + y_bg) / 2.0

    else:  # 'no_speed'
        x, y = x_init, y_init

    return x, y


################################################
################   сам трекинг    #############
################################################

def track_init(ds, clstr_len, data_type, stat_Q, CS_tracks_list, our_time):
    """Инициализация точек-начал треков по DBSCAN-кластерам."""
    for i in range(len(stat_Q)):
        x, y = int(stat_Q.x.values[i]), int(stat_Q.y.values[i])

        CS_coords = {
            'cluster': clstr_len + i,
            't': [our_time],
            'time': [ds[time_name][our_time].values],
            'x': [x],
            'y': [y],
            'lon': [float(stat_Q.lon.values[i])],
            'lat': [float(stat_Q.lat.values[i])],
            'rad': [stat_Q.rad_eff.values[i]],
            'crit': [stat_Q.crit.values[i]],
            'track_len': [0],
        }

        CS_tracks_list.append(CS_coords)

    clstr_len = CS_tracks_list[-1]['cluster']
    return CS_tracks_list, clstr_len
