"""
Stage B (global/local) — конвертация DBSCAN nc-файлов в узлы для TE/StitchNodes.

2026-08-30: обновлено под перебор пайплайна (eps x size_filter x extr_type):

1. Новая схема папок (та же, что уже сделана в 1_create_Nodes_from_DBSCAN_geom.py):
     {path_init}/TempestExtremes/ERA5/R2D_ERA5_{region_name}_sigma_{sigma}/
         {eps:02d}-{min_samples:02d}-{size_filter:02d}_{extr_type}/R2D_txt_files_{year}/
   вместо старой '.../R2D_txt_files_2010_{size_filter}points{extr_type}'.

2. DBSCAN (Stage A) больше не считается заново для каждого size_filter.
   size_filter — это ПОСТ-фильтр по числу точек в кластере поверх уже
   готовой кластеризации (см. func_for_DBSCAN_opt.py::clustering_DBSCAN_C:
   сначала DBSCAN.fit_predict по (eps, min_samples), потом
   `cluster_sizes >= size_filter`). Он не меняет форму/состав кластеров,
   которые проходят фильтр — только отбрасывает более мелкие. Значит nc-файлы,
   посчитанные один раз с САМЫМ МЯГКИМ порогом (size_filter=10), уже содержат
   как подмножество ровно те же кластеры (тех же формы/значений), что дал бы
   отдельный прогон DBSCAN с size_filter=25 или 49 — их нужно только
   дополнительно отфильтровать по числу точек, здесь, на этапе создания узлов,
   не пересчитывая DBSCAN.

   Поэтому здесь всегда читаем nc-источник из папки с size_filter=10
   (SOURCE_SIZE_FILTER), а целевой size_filter (10/25/49) применяется как
   дополнительный фильтр по полю 'cluster' (полное членство) при отборе
   экстремумов — см. get_valid_cluster_ids()/min_cluster_size.

Откуда берутся данные
----------------------
compute_DBSCAN_latlon_with_rad.py (см. func_for_DBSCAN_opt.py::get_DBSCAN_ds)
кладёt в daily nc следующие поля (time, level, lat, lon):
  - cluster:              членство В КАЖДОЙ точке кластера (0 = фон/шум;
                           знак: + циклонический, - антициклонический)
  - center / center_cluster / rad_eff:
                           ненулевые ТОЛЬКО в точке глобального экстремума
                           |R2D| кластера; center_cluster хранит id этого
                           кластера в этой точке (источник для extr_type='_global')
  - local_extr_crit / local_extr_cluster / local_extr_rad_eff:
                           то же, но в НЕСКОЛЬКИХ точках локальных экстремумов
                           (источник для extr_type='_local')
  - wspd:                  полное 2D-поле скорости ветра

Формат выходного txt (не менялся, читается StitchNodes и
4_compute_era5_params_for_nodes.py):
    {year}\t{month}\t{day}\t{n_extrema}\t{hour}
    \t{lon_idx}\t{lat_idx}\t{lon:.6f}\t{lat:.6f}\t{rad:.6e}\t{crit:.6e}\t{wspd:.6e}
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import os
import re

SOURCE_SIZE_FILTER = 10  # DBSCAN нужно посчитать только с этим (самым мягким) порогом


def get_valid_cluster_ids(cluster_full_t: np.ndarray, min_cluster_size: int) -> set:
    """Множество id кластеров (знак: + C, - AC) поля 'cluster' в текущий час,
    у которых число точек >= min_cluster_size. None -> фильтр не применяется
    (используется как есть, т.е. как в исходном nc, порог size_filter=10)."""
    if min_cluster_size is None:
        return None
    ids, counts = np.unique(cluster_full_t, return_counts=True)
    mask = (ids != 0) & (counts >= min_cluster_size)
    return set(ids[mask].tolist())


def convert_nc_to_txt_format(nc_file_path, output_txt_path, date_str=None, append_mode=False,
                              extr_type='_local', min_cluster_size=None):
    """
    Конвертирует NetCDF файл с данными о локальных экстремумах в текстовый формат.

    Parameters:
    -----------
    nc_file_path : str
        Путь к NetCDF файлу
    output_txt_path : str
        Путь для сохранения текстового файла
    date_str : str, optional
        Дата в формате 'YYYY-MM-DD'
    append_mode : bool
        Если True, добавляет данные в конец файла (для группировки по месяцам)
    extr_type : '_local' | '_global'
        '_local' -> local_extr_crit/local_extr_rad_eff/local_extr_cluster
        иначе    -> center/rad_eff/center_cluster (глобальный экстремум кластера)
    min_cluster_size : int or None
        Целевой size_filter (10/25/49/...), применяется здесь как доп. фильтр
        по числу точек в кластере поверх поля 'cluster' исходного nc (см. докстринг
        модуля). None = не фильтровать (как раньше, доверяем nc как есть).
    """

    # Открываем NetCDF файл
    ds = xr.open_dataset(nc_file_path)

    # Определяем дату
    if date_str is None:
        match = re.search(r'(\d{4})-(\d{2})-(\d{2})', nc_file_path)
        if match:
            year, month, day = match.groups()
            date_str = f"{year}-{month}-{day}"
        else:
            if 'Time' in ds.coords or 'time' in ds.coords:
                time_coord = ds['Time'] if 'Time' in ds.coords else ds['time']
                dt = pd.to_datetime(time_coord.values[0])
                date_str = dt.strftime('%Y-%m-%d')
            else:
                raise ValueError("Не удалось определить дату. Укажите date_str.")

    dt = pd.to_datetime(date_str)
    year = dt.year
    month = dt.month
    day = dt.day

    # Получаем координаты
    if 'longitude' in ds.coords:
        lon_coord = ds['longitude']
    elif 'lon' in ds.coords:
        lon_coord = ds['lon']
    else:
        raise ValueError("Координата долготы не найдена")

    if 'latitude' in ds.coords:
        lat_coord = ds['latitude']
    elif 'lat' in ds.coords:
        lat_coord = ds['lat']
    else:
        raise ValueError("Координата широты не найдена")

    # Полное поле членства (для доп. фильтра по size_filter)
    cluster_full = ds['cluster'].values  # (time, level, lat, lon)

    if extr_type == '_local':
        local_extr_data = ds['local_extr_crit'].values  # (time, level, lat, lon)
        local_extr_rad = ds['local_extr_rad_eff'].values  # (time, level, lat, lon)
        local_extr_cluster_id = ds['local_extr_cluster'].values  # (time, level, lat, lon)
    else:
        local_extr_data = ds['center'].values  # (time, level, lat, lon)
        local_extr_rad = ds['rad_eff'].values  # (time, level, lat, lon)
        local_extr_cluster_id = ds['center_cluster'].values  # (time, level, lat, lon)

    wspd_data = ds['wspd'].values  # (time, level, lat, lon)

    # Получаем временные шаги
    if 'Time' in ds.coords:
        time_coord = ds['Time']
    elif 'time' in ds.coords:
        time_coord = ds['time']
    else:
        time_coord = None

    if time_coord is not None:
        n_times = len(time_coord)
        times = pd.to_datetime(time_coord.values)
    else:
        n_times = 1
        times = [dt]

    # Определяем режим записи
    mode = 'a' if append_mode else 'w'

    # Открываем файл для записи
    with open(output_txt_path, mode) as f:
        # Проходим по временным шагам (часам)
        for t_idx in range(n_times):
            current_time = times[t_idx] if time_coord is not None else dt

            # Извлекаем данные для текущего времени (level=0)
            extr_crit = local_extr_data[t_idx, 0, :, :]
            extr_rad = local_extr_rad[t_idx, 0, :, :]
            extr_wspd = wspd_data[t_idx, 0, :, :]
            extr_cluster_id = local_extr_cluster_id[t_idx, 0, :, :]
            cluster_full_t = cluster_full[t_idx, 0, :, :]

            ###### тут берутся только циклонические!!!!
            # Находим индексы ячеек с экстремумами
            valid_mask = (extr_crit > 0) & ~np.isnan(extr_crit)

            # Доп. фильтр по целевому size_filter (10/25/49...), см. докстринг модуля:
            # тот же nc (посчитанный один раз с size_filter=10) переиспользуется для
            # всех целевых порогов, здесь просто отбрасываем более мелкие кластеры.
            valid_ids = get_valid_cluster_ids(cluster_full_t, min_cluster_size)
            if valid_ids is not None:
                id_ok = np.isin(extr_cluster_id, list(valid_ids)) if valid_ids else np.zeros_like(valid_mask, dtype=bool)
                valid_mask = valid_mask & id_ok

            valid_indices = np.where(valid_mask)

            # Собираем все экстремумы
            extrema_list = []
            for i, j in zip(valid_indices[0], valid_indices[1]):
                lon_idx = j
                lat_idx = i

                lon_val = float(lon_coord[lon_idx].values)
                lat_val = float(lat_coord[lat_idx].values)
                rad_val = float(extr_rad[i, j])
                crit_val = float(extr_crit[i, j])
                wspd_val = float(extr_wspd[i, j])

                extrema_list.append({
                    'lon_idx': lon_idx,
                    'lat_idx': lat_idx,
                    'lon': lon_val,
                    'lat': lat_val,
                    'rad': rad_val,
                    'crit': crit_val,
                    'wspd': wspd_val,
                })

            n_extrema = len(extrema_list)

            # Записываем строку с метаданными: год, месяц, день, кол-во экстремумов, час
            f.write(f"{year}\t{month}\t{day}\t{n_extrema}\t{current_time.hour}\n")

            # Записываем каждый экстремум
            for ext in extrema_list:
                f.write(f"\t{ext['lon_idx']}\t{ext['lat_idx']}\t{ext['lon']:.6f}\t{ext['lat']:.6f}\t{ext['rad']:.6e}\t{ext['crit']:.6e}\t{ext['wspd']:.6e}\n")

    ds.close()
    print(f"Добавлено {n_times} временных шагов из {nc_file_path}")


def convert_entire_dataset(nc_dir, output_dir, years_range, eps, min_samples, size_filter, sigma,
                            data_type='ERA5', extr_type='_local', min_cluster_size=None):
    """
    Конвертирует все NetCDF файлы в текстовый формат, группируя по месяцам.
    Каждый временной шаг (час) сохраняется отдельно.

    min_cluster_size: целевой size_filter (10/25/49), применяемый как доп.
        фильтр поверх nc, посчитанного один раз с SOURCE_SIZE_FILTER=10
        (см. докстринг модуля). Обычно = size_filter.
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pattern = f"sigma_{sigma}_DBSCAN_{data_type}_*.nc"
    nc_files = sorted(Path(nc_dir).glob(pattern))

    if not nc_files:
        print(f"Файлы по шаблону {pattern} не найдены в {nc_dir}")
        return

    print(f"Найдено {len(nc_files)} файлов")

    if years_range is not None:
        start_year, end_year = years_range
        filtered_files = []
        for nc_file in nc_files:
            match = re.search(r'(\d{4})-(\d{2})-(\d{2})', str(nc_file))
            if match:
                year = int(match.group(1))
                if start_year <= year <= end_year:
                    filtered_files.append(nc_file)
        nc_files = filtered_files
        print(f"После фильтрации по годам {start_year}-{end_year}: {len(nc_files)} файлов")

        if not nc_files:
            print("Нет файлов для обработки после фильтрации по годам")
            return

    files_by_month = {}
    for nc_file in nc_files:
        match = re.search(r'(\d{4})-(\d{2})-(\d{2})', str(nc_file))
        if match:
            year, month, day = match.groups()
            month_key = f"{year}-{month}"
            if month_key not in files_by_month:
                files_by_month[month_key] = []
            files_by_month[month_key].append(nc_file)

    for month_key, files in files_by_month.items():
        year, month = month_key.split('-')
        output_filename = f"{data_type}_R2D_extr_{year}-{month}.txt"
        output_path = Path(output_dir) / output_filename
        # Пишем во временный файл и переименовываем в output_path только если
        # ВЕСЬ месяц прошёл без ошибок - иначе output_path.exists() на
        # следующем запуске может принять за "готово" частично записанный
        # (после сбоя/kill посреди месяца) файл. Само по себе наличие файла
        # не означает, что в нём есть все дни - гарантирует это только
        # атомарное переименование в конце ниже.
        # pid в имени - на случай если этот же месяц одновременно считают два
        # процесса (например, при разнесении Stage E' и Stage C на параллельные
        # запуски grid_runner'а, оба из которых гоняют Stage B(size_filter=10)):
        # без pid они писали бы в один и тот же tmp-файл и портили друг друга.
        tmp_path = output_path.with_suffix(output_path.suffix + f".partial.pid{os.getpid()}")

        if output_path.exists():
            print(f"Файл {output_filename} уже существует. Пропускаем обработку месяца {month_key}")
            continue  # Пропускаем этот месяц

        print(f"\nОбработка месяца {month_key}. Файлов: {len(files)}")

        failed_files = []
        wrote_any = False
        for nc_file in sorted(files):
            try:
                convert_nc_to_txt_format(
                    str(nc_file),
                    str(tmp_path),
                    append_mode=wrote_any,
                    extr_type=extr_type,
                    min_cluster_size=min_cluster_size,
                )
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

        print(f"Файл сохранен: {output_path}")


def convert_single_file(nc_file_path, output_txt_path, extr_type='_local', min_cluster_size=None):
    """
    Конвертирует один NetCDF файл в текстовый формат.
    """
    convert_nc_to_txt_format(nc_file_path, output_txt_path, append_mode=False,
                              extr_type=extr_type, min_cluster_size=min_cluster_size)


def run_for_combo(eps, size_filter, extr_type, sigma=2, data_type='ERA5', level_hPa=850,
                   region_name=None, years=(2010, 2010), path_init='/storage/thalassa/users/vkoshkina/data',
                   min_samples=4):
    """Один прогон Stage B для конкретной комбинации (eps, size_filter, extr_type).

    extr_type: '_global' | '_local' (внутренний селектор полей в nc).
    Папка на выходе называется по 'global'/'local' (без подчёркивания), как и
    у geom-варианта (1_create_Nodes_from_DBSCAN_geom.py), т.е. итоговая схема
    одинакова для всех трёх extr_type.

    DBSCAN (Stage A) читается всегда из папки с SOURCE_SIZE_FILTER=10 —
    целевой size_filter применяется здесь как доп. фильтр по числу точек в
    кластере (см. докстринг модуля), без повторного запуска DBSCAN.
    """
    if region_name is None:
        region_name = f'NA_for_TC_{level_hPa}hPa'

    extr_suffix = extr_type.lstrip('_')  # '_global' -> 'global', '_local' -> 'local'

    # Stage A: всегда самый мягкий порог (10) — источник для всех size_filter
    nc_dir = f'{path_init}/ERA5/DBSCAN_{eps:02d}-{min_samples:02d}-{SOURCE_SIZE_FILTER:02d}_{region_name}_sigma_{sigma}_rad'

    combo_dir = (
        f'{path_init}/TempestExtremes/ERA5/R2D_ERA5_{region_name}_sigma_{sigma}/'
        f'{eps:02d}-{min_samples:02d}-{size_filter:02d}_{extr_suffix}'
    )
    output_dir = f'{combo_dir}/R2D_txt_files_{years[0]}'

    convert_entire_dataset(
        nc_dir=nc_dir,
        output_dir=output_dir,
        years_range=years,
        eps=eps,
        min_samples=min_samples,
        size_filter=size_filter,
        sigma=sigma,
        data_type=data_type,
        extr_type=extr_type,
        min_cluster_size=size_filter,
    )


# Пример использования
if __name__ == "__main__":
    eps = 1
    size_filter = 25  # целевой порог отбора вихрей: 10 / 25 / 49 (фильтр применяется здесь, не в Stage A)

    sigma = 2
    data_type = 'ERA5'
    level_hPa = 850

    extr_type = '_global'
    extr_type = '_local'

    years = (2010, 2010)

    run_for_combo(
        eps=eps,
        size_filter=size_filter,
        extr_type=extr_type,
        sigma=sigma,
        data_type=data_type,
        level_hPa=level_hPa,
        years=years,
    )

    # Пример перебора всех size_filter на одном DBSCAN-источнике (eps=1, sigma=2, global):
    # for size_filter in (10, 25, 49):
    #     run_for_combo(eps=1, size_filter=size_filter, extr_type='_global', sigma=2, years=(2010, 2010))
