"""
Геометрические утилиты для трекинга ВВС (вихрей) по регулярной
широтно-долготной сетке ERA5 (0.25° или иного шага).

Зачем этот файл
---------------
Исходные func_for_*.py (в ../) считают расстояние между точкой трека и
кандидатами на продолжение трека как евклидово расстояние между индексами
сетки: dist(x, y, x2, y2) = sqrt((x-x2)^2 + (y-y2)^2), а порог поиска задаётся
как hw_scale*rad_init, где rad_init тоже в "клетках" (rad_eff = sqrt(N/pi),
см. ERA5/func_for_DBSCAN_opt.py::get_stat). Это корректно для равномерных
декартовых сеток (WRF x/y, GLORYS и т.п.), где 1 клетка = одно и то же
расстояние в метрах по обеим осям.

Для ERA5 сетка регулярна в градусах (шаг по долготе == шаг по широте), но НЕ
в километрах: 1° долготы физически короче 1° широты в cos(широта) раз.
На широтах полярных мезоциклонов (60-80°N) это даёт множитель cos(70°)~0.34 —
то есть "расстояние" в клетках по долготе, посчитанное как в WRF, занижает
реальный разнос точек по долготе почти в 3 раза. Из-за этого поиск по кругу
в индексном пространстве превращается в реальности в вытянутый по долготе
эллипс, и трекинг может как хватать слишком далёкие по факту вихри, так и
терять вихри, реально близкие, но разнесённые по широте.

Этот модуль переводит индексный трекинг в трекинг по честным
географическим координатам:
  - haversine_distance_km       — расстояние по дуге большого круга, км;
  - index_to_lonlat              — перевод (возможно, дробных) индексов сетки
                                    x/y, которые возвращает get_next_loc_cases,
                                    в реальные lon/lat (линейная интерполяция
                                    по регулярной сетке);
  - grid_step_deg                — фактический шаг сетки по lon/lat, градусы
                                    (определяется по данным, а не хардкодится,
                                    чтобы работать и для 0.25°, и для других
                                    разрешений ERA5);
  - rad_gridpoints_to_km         — перевод "радиуса" вихря rad_eff (в клетках,
                                    как его считает DBSCAN) в километры с
                                    учётом cos(широта);
  - nearest_within_radius        — поиск ближайшего кандидата в пределах
                                    физического радиуса (км).

Что НЕ меняется
----------------
Сам DBSCAN/R2D (см. ERA5/func_for_DBSCAN_opt.py) по-прежнему кластеризует
и считает rad_eff в индексах сетки — это осознанно оставлено как есть
(см. обсуждение с пользователем): здесь адаптируется только шаг трекинга
(привязка последовательных точек трека друг к другу), а не сама
идентификация вихрей.
"""
import numpy as np

# Радиус Земли, км (тот же порядок точности, что и в существующем
# черновике func_for_local_2_phase_cone_latlon.py)
R_EARTH_KM = 6371.0

# км на градус по меридиану/экватору — то же приближение, что уже используется
# в step_of_tracking.py для ERA5 (dist_m = 111 * 1000 * 0.25, т.е. 111 км/град)
KM_PER_DEG = 111.0


def haversine_distance_km(lon1, lat1, lon2, lat2):
    """
    Расстояние по дуге большого круга между точками (lon1, lat1) и
    (lon2, lat2) в километрах. Все аргументы в градусах; поддерживают
    скаляры, numpy-массивы и pandas.Series (broadcasting как в numpy).
    """
    lon1_r = np.radians(lon1)
    lat1_r = np.radians(lat1)
    lon2_r = np.radians(lon2)
    lat2_r = np.radians(lat2)

    dlon = lon2_r - lon1_r
    dlat = lat2_r - lat1_r

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arcsin(np.sqrt(a))

    return R_EARTH_KM * c


def grid_step_deg(ds):
    """
    Фактический шаг регулярной сетки ERA5 по долготе и широте, градусы.
    Берётся из самих координат ds, а не хардкодится — так один и тот же
    код работает и для 0.25°, и для 0.5°/1° ERA5.
    """
    lons = np.asarray(ds.longitude.values, dtype=float)
    lats = np.asarray(ds.latitude.values, dtype=float)

    dlon = float(np.abs(np.median(np.diff(lons)))) if len(lons) > 1 else 0.25
    dlat = float(np.abs(np.median(np.diff(lats)))) if len(lats) > 1 else 0.25

    return dlon, dlat


def index_to_lonlat(ds, x, y):
    """
    Перевод (возможно, дробных) индексов сетки x (по долготе), y (по широте)
    в реальные координаты lon/lat, линейной интерполяцией по регулярной
    сетке ds.longitude / ds.latitude. Нужен, потому что get_next_loc_cases
    (step_of_tracking.py) по-прежнему работает и возвращает "ожидаемую"
    следующую точку трека в индексном пространстве (в т.ч. с учётом скорости
    смещения) — здесь мы лишь переводим её результат в градусы, чтобы
    дальше искать кандидатов по честному расстоянию.
    """
    lons = np.asarray(ds.longitude.values, dtype=float)
    lats = np.asarray(ds.latitude.values, dtype=float)

    idx_lon = np.arange(len(lons))
    idx_lat = np.arange(len(lats))

    x_clipped = np.clip(x, 0, len(lons) - 1)
    y_clipped = np.clip(y, 0, len(lats) - 1)

    lon = float(np.interp(x_clipped, idx_lon, lons))
    lat = float(np.interp(y_clipped, idx_lat, lats))

    return lon, lat


def rad_gridpoints_to_km(rad_gridpoints, lat_deg, ds):
    """
    Перевод "радиуса" вихря rad_eff (в клетках сетки, rad_eff = sqrt(N/pi),
    см. ERA5/func_for_DBSCAN_opt.py::get_stat) в километры, на широте
    lat_deg.

    Клетка сетки ERA5 имеет размер dlon_deg x dlat_deg градусов, что в
    километрах даёт разные по долготе и широте величины:
        dx_km = KM_PER_DEG * dlon_deg * cos(lat)
        dy_km = KM_PER_DEG * dlat_deg
    Чтобы не подменять одну грубую (изотропную по клеткам) оценку другой
    столь же грубой (изотропной по км), но зато сохранить площадь,
    заложенную в исходное rad_eff = sqrt(N_points/pi) (т.е.
    pi*rad_km^2 == N_points*dx_km*dy_km), используется геометрическое
    среднее двух масштабов:
        rad_km = rad_gridpoints * sqrt(dx_km * dy_km)
    """
    dlon_deg, dlat_deg = grid_step_deg(ds)

    dx_km = KM_PER_DEG * dlon_deg * np.cos(np.radians(lat_deg))
    dy_km = KM_PER_DEG * dlat_deg

    cell_scale_km = np.sqrt(max(dx_km * dy_km, 0.0))

    return rad_gridpoints * cell_scale_km


def nearest_within_radius(df, lon_exp, lat_exp, radius_km):
    """
    Среди строк df (должен содержать колонки 'lon', 'lat') находит
    ближайшую по honest-расстоянию (haversine) к точке (lon_exp, lat_exp)
    в пределах radius_km. Возвращает pandas.Series (одну строку df, с
    добавленным полем 'dist_km' и сохранённым исходным индексом df в
    .name — чтобы можно было исключить эту строку из пула кандидатов по
    индексу/'cluster' так же, как это делали исходные func_for_*.py) или
    None, если подходящих кандидатов нет.
    """
    if len(df) == 0:
        return None

    d = haversine_distance_km(lon_exp, lat_exp, df['lon'].values, df['lat'].values)

    within = d <= radius_km
    if not np.any(within):
        return None

    d_masked = np.where(within, d, np.inf)
    i = int(np.argmin(d_masked))

    row = df.iloc[i].copy()
    row['dist_km'] = d_masked[i]

    return row
