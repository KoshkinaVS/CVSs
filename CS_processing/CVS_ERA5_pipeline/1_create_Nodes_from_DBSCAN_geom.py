"""
Stage B (геометрический вариант) — конвертация DBSCAN nc-файлов в узлы для
TempestExtremes/StitchNodes, где позиция узла = ГЕОМЕТРИЧЕСКИЙ ЦЕНТРОИД
(среднее по координатам всех точек кластера), а не глобальный экстремум
('_global', как в 1_create_Nodes_from_DBSCAN.py) и не набор локальных
экстремумов ('_local').

Идея (заметка 28.08, чек-лист лета 2026): если трекинг ведётся только по
глобальному экстремуму, при перекачке энергии между разными частями одной
структуры трек "прыгает". Центроид кластера смещается плавнее.

Откуда берутся данные
----------------------
compute_DBSCAN_latlon_with_rad.py (см. func_for_DBSCAN_opt.py::get_DBSCAN_ds)
кладёт в daily nc следующие поля (time, level, lat, lon):
  - cluster:        DBSCAN-метка кластера В КАЖДОЙ точке, входящей в кластер
                     (0 = фон/шум; знак: + циклонический, - антициклонический)
  - center / center_cluster / rad_eff:
                     ненулевые ТОЛЬКО в одной точке на кластер — точке
                     глобального экстремума |R2D| (это и есть источник для
                     '_global' в 1_create_Nodes_from_DBSCAN.py)
  - local_extr_crit / local_extr_cluster / local_extr_rad_eff:
                     то же, но в НЕСКОЛЬКИХ точках локальных экстремумов
                     (источник для '_local')
  - R2D (name_crit): полное 2D-поле R2D (NaN вне кластеров)
  - wspd:            полное 2D-поле скорости ветра (везде, не только в кластерах)

Здесь для геометрического варианта используется 'cluster' (полное членство
в кластере, а не только точка-экстремум), чтобы посчитать центроид, плюс
R2D/wspd (полные поля) — значения в узле берутся в ближайшей к центроиду
точке сетки этих полей, а радиус — переиспользуется из rad_eff кластера (тот
же rad_eff, что и для '_global'/'_local' — баг с локальными радиусами
(func_for_DBSCAN_opt.py::update_rad) сейчас сознательно не трогаем, см.
обсуждение по "динамическому размеру вихря").

ВАЖНО (сделанное здесь допущение, стоит свериться): значение R2D/wspd в узле
берётся из ПОЛНОГО поля в ближайшей к центроиду точке сетки, а не как
значение в точке глобального экстремума кластера. Для компактных структур
разница небольшая; для сильно вытянутых/серповидных структур центроид может
попасть в точку с более слабым R2D, чем пиковый — это физически ожидаемо
(центроид ведь не обязан совпадать с максимумом), но именно поэтому для
StitchNodes с --prioritize -r2d соединение по геометрическим узлам может вести
себя иначе, чем по глобальным экстремумам.

Формат выходного txt — тот же, что у 1_create_Nodes_from_DBSCAN.py (иначе
StitchNodes не запустится): одна строка метаданных в час
    {year}\t{month}\t{day}\t{n_extrema}\t{hour}
и по одной строке на узел:
    \t{lon_idx}\t{lat_idx}\t{lon:.6f}\t{lat:.6f}\t{rad:.6e}\t{crit:.6e}\t{wspd:.6e}
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

warnings.filterwarnings("ignore")

SOURCE_SIZE_FILTER = 10  # см. 1_create_Nodes_from_DBSCAN.py: DBSCAN считается один раз с этим (самым мягким) порогом


def _circular_mean_lon(lons_deg: np.ndarray) -> float:
    """Среднее по долготе с учётом перехода через -180/180 (векторное среднее)."""
    rad = np.radians(lons_deg)
    mean_angle = np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))
    return float(np.degrees(mean_angle))


def convert_nc_to_txt_format_geom(nc_file_path, output_txt_path, date_str=None, append_mode=False,
                                   min_cluster_size=None):
    """
    min_cluster_size : int or None
        Целевой size_filter (10/25/49/...), применяется как доп. фильтр по
        числу точек в кластере поверх поля 'cluster' исходного nc (тот же
        приём, что и в 1_create_Nodes_from_DBSCAN.py::get_valid_cluster_ids -
        nc всегда считается один раз с SOURCE_SIZE_FILTER=10, а 25/49
        получаются здесь фильтрацией, без повторного запуска DBSCAN).
        None = не фильтровать (взять все кластеры как есть, т.е. как раньше).
    """
    ds = xr.open_dataset(nc_file_path)

    if date_str is None:
        match = re.search(r'(\d{4})-(\d{2})-(\d{2})', str(nc_file_path))
        if match:
            year, month, day = match.groups()
            date_str = f"{year}-{month}-{day}"
        else:
            raise ValueError("Не удалось определить дату. Укажите date_str.")

    dt = pd.to_datetime(date_str)
    year, month, day = dt.year, dt.month, dt.day

    lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
    lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
    lon_coord = ds[lon_name].values
    lat_coord = ds[lat_name].values

    time_name = 'Time' if 'Time' in ds.coords else 'time'
    times = pd.to_datetime(ds[time_name].values)
    n_times = len(times)

    cluster_full = ds['cluster'].values          # (time, level, lat, lon) — членство В КАЖДОЙ точке кластера
    center_cluster = ds['center_cluster'].values  # ненулевое только в точке глобального экстремума
    rad_eff_field = ds['rad_eff'].values          # то же положение, что и center_cluster
    r2d_field = ds['R2D'].values                  # полное поле R2D (NaN вне кластеров)
    wspd_field = ds['wspd'].values                # полное поле скорости ветра

    mode = 'a' if append_mode else 'w'

    with open(output_txt_path, mode) as f:
        for t_idx in range(n_times):
            current_time = times[t_idx]

            cluster_t = cluster_full[t_idx, 0, :, :]
            center_cluster_t = center_cluster[t_idx, 0, :, :]
            rad_eff_t = rad_eff_field[t_idx, 0, :, :]
            r2d_t = r2d_field[t_idx, 0, :, :]
            wspd_t = wspd_field[t_idx, 0, :, :]

            # rad_eff кластера хранится только в его точке-экстремуме -> строим lookup {cluster_id: rad_eff}
            extr_i, extr_j = np.where(center_cluster_t != 0)
            rad_by_cluster = {
                int(center_cluster_t[i, j]): float(rad_eff_t[i, j])
                for i, j in zip(extr_i, extr_j)
                if np.isfinite(rad_eff_t[i, j])
            }

            cluster_ids, cluster_counts = np.unique(cluster_t, return_counts=True)
            nonzero_mask = cluster_ids != 0
            cluster_ids = cluster_ids[nonzero_mask]
            cluster_counts = cluster_counts[nonzero_mask]

            if min_cluster_size is not None:
                size_ok = cluster_counts >= min_cluster_size
                cluster_ids = cluster_ids[size_ok]

            nodes = []
            for cid in cluster_ids:
                mask_i, mask_j = np.where(cluster_t == cid)
                if mask_i.size == 0:
                    continue

                centroid_lat = float(np.mean(lat_coord[mask_i]))
                centroid_lon = _circular_mean_lon(lon_coord[mask_j])

                # ближайшая к центроиду точка сетки — для чтения R2D/wspd и для (lon_idx, lat_idx)
                lat_idx = int(np.argmin(np.abs(lat_coord - centroid_lat)))
                lon_idx = int(np.argmin(np.abs(((lon_coord - centroid_lon + 180.0) % 360.0) - 180.0)))

                crit_val = float(r2d_t[lat_idx, lon_idx])
                if not np.isfinite(crit_val):
                    # центроид попал в "дыру" вне кластера (вытянутая/серповидная структура) —
                    # берём знак кластера (C/AC) и запасное значение R2D из ближайшей КЛАСТЕРНОЙ точки
                    dist = (lat_coord[mask_i] - centroid_lat) ** 2 + \
                           (((lon_coord[mask_j] - centroid_lon + 180.0) % 360.0 - 180.0)) ** 2
                    nearest = np.argmin(dist)
                    crit_val = float(r2d_t[mask_i[nearest], mask_j[nearest]])
                    if not np.isfinite(crit_val):
                        continue

                wspd_val = float(wspd_t[lat_idx, lon_idx])
                rad_val = rad_by_cluster.get(int(cid), np.nan)
                if not np.isfinite(rad_val):
                    continue

                nodes.append({
                    'lon_idx': lon_idx, 'lat_idx': lat_idx,
                    'lon': centroid_lon, 'lat': centroid_lat,
                    'rad': rad_val, 'crit': crit_val, 'wspd': wspd_val,
                })

            f.write(f"{year}\t{month}\t{day}\t{len(nodes)}\t{current_time.hour}\n")
            for node in nodes:
                f.write(
                    f"\t{node['lon_idx']}\t{node['lat_idx']}\t{node['lon']:.6f}\t{node['lat']:.6f}\t"
                    f"{node['rad']:.6e}\t{node['crit']:.6e}\t{node['wspd']:.6e}\n"
                )

    ds.close()


def convert_entire_dataset_geom(nc_dir, output_dir, years_range, eps, min_samples, size_filter, sigma, data_type='ERA5',
                                 min_cluster_size=None):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pattern = f"sigma_{sigma}_DBSCAN_{data_type}_*.nc"
    nc_files = sorted(Path(nc_dir).glob(pattern))

    if not nc_files:
        print(f"Файлы по шаблону {pattern} не найдены в {nc_dir}")
        return

    print(f"Найдено {len(nc_files)} файлов")

    if years_range is not None:
        start_year, end_year = years_range
        filtered = []
        for nc_file in nc_files:
            m = re.search(r'(\d{4})-(\d{2})-(\d{2})', str(nc_file))
            if m and start_year <= int(m.group(1)) <= end_year:
                filtered.append(nc_file)
        nc_files = filtered
        print(f"После фильтрации по годам {start_year}-{end_year}: {len(nc_files)} файлов")
        if not nc_files:
            return

    files_by_month = {}
    for nc_file in nc_files:
        m = re.search(r'(\d{4})-(\d{2})-(\d{2})', str(nc_file))
        if m:
            year, month, _day = m.groups()
            files_by_month.setdefault(f"{year}-{month}", []).append(nc_file)

    for month_key, files in tqdm(files_by_month.items(), desc="Месяцы"):
        year, month = month_key.split('-')
        output_filename = f"{data_type}_R2D_extr_{year}-{month}.txt"
        output_path = Path(output_dir) / output_filename
        # Пишем во временный файл, переименовываем в output_path только если
        # ВЕСЬ месяц прошёл без ошибок - см. 1_create_Nodes_from_DBSCAN.py
        # (одинаковый фикс в обоих скриптах): иначе output_path.exists() на
        # следующем запуске может принять за "готово" частично записанный
        # (после сбоя/kill посреди месяца) файл.
        # pid в имени - см. 1_create_Nodes_from_DBSCAN.py: без него два
        # параллельных процесса (например, --stage eprime и --stage tracks),
        # оба гоняющие Stage B(size_filter=10), могли бы одновременно писать
        # в один и тот же tmp-файл и портить друг друга.
        tmp_path = output_path.with_suffix(output_path.suffix + f".partial.pid{os.getpid()}")

        if output_path.exists():
            print(f"Файл {output_filename} уже существует, пропускаем месяц {month_key}")
            continue

        failed_files = []
        wrote_any = False
        for nc_file in sorted(files):
            try:
                convert_nc_to_txt_format_geom(str(nc_file), str(tmp_path), append_mode=wrote_any,
                                               min_cluster_size=min_cluster_size)
                wrote_any = True
            except Exception as e:
                print(f"Ошибка при обработке {nc_file}: {e}")
                failed_files.append(nc_file)
                continue

        if failed_files:
            print(
                f"Месяц {month_key}: {len(failed_files)} из {len(files)} дней не обработаны "
                f"({[Path(f).name for f in failed_files]}) - итоговый файл НЕ создан, "
                f"частичный результат оставлен в {tmp_path} для диагностики. "
                f"Месяц будет пересобран заново при следующем запуске."
            )
            continue

        tmp_path.rename(output_path)  # атомарная публикация - только полностью собранный месяц
        print(f"Файл сохранён: {output_path}")


def run_for_combo(eps, size_filter, sigma=2, data_type='ERA5', level_hPa=850,
                   region_name=None, years=(2010, 2010), path_init='/storage/thalassa/users/vkoshkina/data',
                   min_samples=4):
    """Один прогон Stage B (geom) для конкретной комбинации (eps, size_filter).
    extr_type здесь всегда 'geom'. Как и в 1_create_Nodes_from_DBSCAN.py::run_for_combo,
    DBSCAN (Stage A) читается всегда из папки с SOURCE_SIZE_FILTER=10, а целевой
    size_filter применяется здесь доп. фильтром по числу точек в кластере."""
    if region_name is None:
        region_name = f'NA_for_TC_{level_hPa}hPa'

    extr_type = 'geom'

    nc_dir = f'{path_init}/ERA5/DBSCAN_{eps:02d}-{min_samples:02d}-{SOURCE_SIZE_FILTER:02d}_{region_name}_sigma_{sigma}_rad'

    combo_dir = (
        f'{path_init}/TempestExtremes/ERA5/R2D_ERA5_{region_name}_sigma_{sigma}/'
        f'{eps:02d}-{min_samples:02d}-{size_filter:02d}_{extr_type}'
    )
    output_dir = f'{combo_dir}/R2D_txt_files_{years[0]}'

    convert_entire_dataset_geom(
        nc_dir=nc_dir,
        output_dir=output_dir,
        years_range=years,
        eps=eps,
        min_samples=min_samples,
        size_filter=size_filter,
        sigma=sigma,
        data_type=data_type,
        min_cluster_size=size_filter,
    )


if __name__ == "__main__":
    eps = 1
    size_filter = 25  # целевой порог отбора вихрей: 10 / 25 / 49 (фильтр применяется здесь, не в Stage A)

    sigma = 2
    data_type = 'ERA5'
    level_hPa = 850

    years = (2010, 2010)

    run_for_combo(
        eps=eps,
        size_filter=size_filter,
        sigma=sigma,
        data_type=data_type,
        level_hPa=level_hPa,
        years=years,
    )
