"""
Stage B (NAAD, LoRes/HiRes) — конвертация Stage-A DBSCAN nc-файлов (WRF-сетка)
в узлы для TE/StitchNodes. Аналог 1_create_Nodes_from_DBSCAN.py (ERA5), но
адаптирован под WRF-специфику NAAD — см. подробности в наад_config.py и ниже.

Место в пайплайне
------------------
Stage A (DBSCAN identification на NAAD, CS_processing/CVS_identification/
в репозитории KoshkinaVS/CVSs, config_NAAD.json) уже посчитан ОТДЕЛЬНО, не
этим скриптом — он лежит в NC_DIR_TEMPLATE_BY_TYPE (naad_config.py) и кладёт
в daily nc (по одному файлу в день, как у ERA5) поля (Time, interp_level,
south_north, west_east) - размерность уровня подтверждена как 'interp_level'
(не 'bottom_top' из config_NAAD.json - см. п.3 ниже), размер 1:
    cluster, center / center_cluster / rad_eff,
    local_extr_crit / local_extr_cluster / local_extr_rad_eff, wspd
— имена/семантика ПОДТВЕРЖДЕНЫ на реальном файле (ncdump от пользователя,
2026-09-05), совпадают с тем, что ожидалось по аналогии с ERA5.

Чем это отличается от ERA5-версии (1_create_Nodes_from_DBSCAN.py)
--------------------------------------------------------------------
1. Широта/долгота — НЕ 1D координаты lat[i]/lon[j], а 2D переменные
   XLAT[i,j]/XLONG[i,j] (WRF-сетка south_north x west_east). Индексировать их
   по отдельности (как lon_coord[lon_idx] у ERA5) НЕЛЬЗЯ — нужен ОБА индекса
   сразу. См. _load_latlon_2d() ниже.
2. Время — 2026-09-05, ПОДТВЕРЖДЕНО на реальном файле NAAD (ncdump от
   пользователя): переменная называется 'Time' (не 'XTIME', как можно было
   подумать по config_NAAD.json), int64, CF-формат "hours since
   1970-01-01 00:00:00" (calendar=proleptic_gregorian) — xarray обычно сам
   декодирует такое в datetime64 при open_dataset, _load_times() это
   использует напрямую; если вдруг не декодировалось (raw осталось int) —
   декодирует вручную по атрибуту units. 'XTIME' оставлен как второй по
   приоритету вариант (вдруг в другом файле/варианте NAAD называется иначе),
   дата из имени файла — крайний случай, с явным предупреждением в консоли.
3. Порядок размерностей полей — ПОДТВЕРЖДЕНО: (Time, interp_level,
   south_north, west_east), НЕ 'bottom_top', как можно было подумать по
   config_NAAD.json (там, видимо, описан внутренний формат Stage A ДО
   интерполяции на уровень, а не итоговый сохранённый nc). _field_2d() ниже
   не хардкодит имя уровневой размерности — определяет её как "всё, что не
   Time/south_north/west_east" и требует, чтобы она была размера 1 (иначе
   явная ошибка, а не тихо неверный уровень).
4. dist_m (физический шаг сетки в метрах, naad_config.DIST_M_BY_TYPE) — здесь
   не используется напрямую (формат Nodes-txt не меняется, 'rad' по-прежнему
   пишется в ЯЧЕЙКАХ, как и у ERA5, для совместимости со Stage C/D), но
   доступен через naad_config.rad_km() для последующего анализа/фильтрации.

Формат выходного txt — БЕЗ ИЗМЕНЕНИЙ (тот же, что читает StitchNodes и
3_create_csv_tracks_from_StitchNodes.py):
    {year}\t{month}\t{day}\t{n_extrema}\t{hour}
    \t{lon_idx}\t{lat_idx}\t{lon:.6f}\t{lat:.6f}\t{rad:.6e}\t{crit:.6e}\t{wspd:.6e}

Пример запуска
--------------
python 1_create_Nodes_from_DBSCAN_NAAD.py --data-type LoRes --size-filter 10 \
    --extr-type local --year 2010
python 1_create_Nodes_from_DBSCAN_NAAD.py --data-type HiRes --size-filter 10 \
    --extr-type local --year 2010

2026-09-05: первый реальный прогон на сервере нашёл 2 расхождения с
config_NAAD.json (оба исправлены здесь): переменная времени - 'Time', не
'XTIME'; размерность уровня - 'interp_level', не 'bottom_top'. Оба места
(_load_times/_try_time_var и _field_2d) теперь не хардкодят имя, а определяют
его по факту - см. комментарии по месту.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import naad_config as cfg

TIME_DIM = "Time"
Y_DIM = "south_north"
X_DIM = "west_east"
# Имя уровневой размерности НЕ хардкодим - на реальных файлах NAAD (2026-09-05)
# это оказалось 'interp_level' (размер 1), а не 'bottom_top' из config_NAAD.json
# (тот конфиг, видимо, про внутренний формат Stage A до интерполяции на
# уровень). См. _field_2d() ниже - определяет её автоматически.


def get_valid_cluster_ids(cluster_full_t: np.ndarray, min_cluster_size) -> set | None:
    """Множество id кластеров (знак: + C, - AC) поля 'cluster' в текущий срез,
    у которых число точек >= min_cluster_size. None -> фильтр не применяется
    (используется как есть, т.е. как в исходном nc, порог SOURCE_SIZE_FILTER).
    Идентично get_valid_cluster_ids() в 1_create_Nodes_from_DBSCAN.py (ERA5)."""
    if min_cluster_size is None:
        return None
    ids, counts = np.unique(cluster_full_t, return_counts=True)
    mask = (ids != 0) & (counts >= min_cluster_size)
    return set(ids[mask].tolist())


def _load_latlon_2d(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """XLAT/XLONG как 2D-массивы (south_north, west_east). Если в файле они
    заведены с размерностью Time (статичная сетка, но формально 3D) - берём
    первый временной срез."""
    if "XLAT" not in ds.variables or "XLONG" not in ds.variables:
        raise ValueError(
            f"Не найдены XLAT/XLONG в файле (доступные переменные: {list(ds.variables)}) - "
            "проверьте, что это Stage-A DBSCAN nc для NAAD, а не что-то другое."
        )
    xlat_da = ds["XLAT"]
    xlong_da = ds["XLONG"]

    if TIME_DIM in xlat_da.dims:
        xlat = xlat_da.isel({TIME_DIM: 0}).transpose(Y_DIM, X_DIM).values
        xlong = xlong_da.isel({TIME_DIM: 0}).transpose(Y_DIM, X_DIM).values
    else:
        xlat = xlat_da.transpose(Y_DIM, X_DIM).values
        xlong = xlong_da.transpose(Y_DIM, X_DIM).values

    if xlat.ndim != 2 or xlong.ndim != 2:
        raise ValueError(f"Неожиданная размерность XLAT/XLONG после transpose: {xlat.shape}/{xlong.shape}")
    return xlat, xlong


def _decode_cf_time(raw: np.ndarray, units: str):
    """Ручной запасной декодер CF-времени вида '<единица> since <референс>'
    (у нас — 'hours since 1970-01-01 00:00:00') - нужен только если xarray
    почему-то НЕ декодировал 'Time'/'XTIME' сам при open_dataset (обычно
    декодирует такие CF-переменные автоматически, это просто подстраховка)."""
    match = re.match(r"\s*(\w+)\s+since\s+(.+)", units)
    if not match:
        return None
    unit_word, ref_str = match.groups()
    pd_unit = {"hours": "h", "hour": "h", "minutes": "m", "minute": "m",
               "seconds": "s", "second": "s", "days": "D", "day": "D"}.get(unit_word.lower())
    if pd_unit is None:
        return None
    try:
        ref = pd.Timestamp(ref_str.strip())
        return pd.DatetimeIndex(ref + pd.to_timedelta(raw.astype("int64"), unit=pd_unit))
    except Exception:
        return None


def _try_time_var(ds: xr.Dataset, var_name: str, n_times: int):
    """Пробует прочитать времена из переменной var_name: сначала как уже
    декодированный xarray'ом datetime64 (обычный случай для CF 'hours
    since ...'), потом - ручной CF-декодинг по атрибуту units (на случай,
    если decode_times почему-то не сработал)."""
    if var_name not in ds.variables:
        return None
    raw = ds[var_name].values
    try:
        times = pd.DatetimeIndex(pd.to_datetime(raw))
        if len(times) == n_times and 1900 <= times[0].year <= 2100:
            return times
    except Exception:
        pass
    units = ds[var_name].attrs.get("units")
    if units:
        decoded = _decode_cf_time(np.asarray(raw), units)
        if decoded is not None and len(decoded) == n_times and 1900 <= decoded[0].year <= 2100:
            return decoded
    return None


def _load_times(ds: xr.Dataset, nc_file_path: str) -> pd.DatetimeIndex:
    """Времена срезов файла. 2026-09-05, подтверждено на реальном файле NAAD:
    основной источник - переменная 'Time' (CF, 'hours since 1970-01-01
    00:00:00', calendar=proleptic_gregorian) - см. _try_time_var()/
    _decode_cf_time() выше. 'XTIME' проверяется вторым (на случай другого
    варианта NAAD-файлов), дата из имени файла + равномерный шаг
    DEFAULT_TIMESTEP_HOURS - крайний случай, с явным предупреждением."""
    n_times = ds.sizes.get(TIME_DIM, 1)

    times = _try_time_var(ds, TIME_DIM, n_times)
    if times is not None:
        return times

    times = _try_time_var(ds, "XTIME", n_times)
    if times is not None:
        return times

    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", str(nc_file_path))
    if not match:
        raise ValueError(
            f"Не удалось разобрать XTIME И не нашлась дата в имени файла {nc_file_path} - "
            "укажите дату вручную или почините _load_times()."
        )
    year, month, day = match.groups()
    dt0 = pd.Timestamp(year=int(year), month=int(month), day=int(day))
    print(
        f"ПРЕДУПРЕЖДЕНИЕ: не удалось разобрать время из 'Time'/'XTIME' в {nc_file_path} - "
        f"использую дату из имени файла ({dt0.date()}) + равномерный шаг "
        f"{cfg.DEFAULT_TIMESTEP_HOURS}ч на {n_times} срез(ов). Это НЕ ожидаемый путь для "
        f"реальных файлов NAAD (обычно 'Time' декодируется напрямую, см. _try_time_var()) - "
        f"если видите это предупреждение регулярно, разберитесь, почему."
    )
    return pd.date_range(dt0, periods=n_times, freq=f"{cfg.DEFAULT_TIMESTEP_HOURS}h")


def _field_2d(da: xr.DataArray, t_idx: int) -> np.ndarray:
    """2D-срез (south_north, west_east) поля на конкретный момент времени.

    Уровневая размерность (если есть) определяется ДИНАМИЧЕСКИ, а не по
    хардкод-имени - 2026-09-05 выяснилось на реальном файле NAAD, что она
    называется 'interp_level' (размер 1), а не 'bottom_top', как можно было
    подумать по config_NAAD.json. Здесь просто берём "всё, что не
    Time/south_north/west_east" - работает независимо от того, как эта
    размерность называется, лишь бы она была размера 1 (иначе явная ошибка,
    а не молча взятый не тот уровень)."""
    extra_dims = [d for d in da.dims if d not in (TIME_DIM, Y_DIM, X_DIM)]

    if not extra_dims:
        da = da.transpose(TIME_DIM, Y_DIM, X_DIM)
        return da.values[t_idx, :, :]

    if len(extra_dims) == 1:
        level_dim = extra_dims[0]
        if da.sizes[level_dim] != 1:
            raise ValueError(
                f"Поле {da.name!r} имеет размерность уровня {level_dim!r} размера "
                f"{da.sizes[level_dim]} (>1) - неясно, какой уровень брать. Если в файле "
                "правда несколько уровней, добавьте явный выбор уровня в run_for_combo()/CLI."
            )
        da = da.transpose(TIME_DIM, level_dim, Y_DIM, X_DIM)
        return da.values[t_idx, 0, :, :]

    raise ValueError(
        f"Неожиданный набор размерностей у поля {da.name!r}: {da.dims} "
        f"(ожидалось Time + south_north + west_east + не более одной доп. размерности)"
    )


def convert_nc_to_txt_format_naad(
    nc_file_path,
    output_txt_path,
    append_mode: bool = False,
    extr_type: str = "_local",
    min_cluster_size=None,
) -> int:
    """NAAD-аналог convert_nc_to_txt_format() из 1_create_Nodes_from_DBSCAN.py
    (ERA5). Возвращает число обработанных временных срезов."""

    ds = xr.open_dataset(nc_file_path)

    xlat, xlong = _load_latlon_2d(ds)
    times = _load_times(ds, str(nc_file_path))
    n_times = len(times)

    cluster_full = ds["cluster"]

    if extr_type == "_local":
        crit_da = ds["local_extr_crit"]
        rad_da = ds["local_extr_rad_eff"]
        cluster_id_da = ds["local_extr_cluster"]
    else:
        crit_da = ds["center"]
        rad_da = ds["rad_eff"]
        cluster_id_da = ds["center_cluster"]

    wspd_da = ds["wspd"]

    mode = "a" if append_mode else "w"
    with open(output_txt_path, mode) as f:
        for t_idx in range(n_times):
            current_time = times[t_idx]

            extr_crit = _field_2d(crit_da, t_idx)
            extr_rad = _field_2d(rad_da, t_idx)
            extr_wspd = _field_2d(wspd_da, t_idx)
            extr_cluster_id = _field_2d(cluster_id_da, t_idx)
            cluster_full_t = _field_2d(cluster_full, t_idx)

            ###### тут берутся только циклонические (как и в ERA5-версии) !!!!
            valid_mask = (extr_crit > 0) & ~np.isnan(extr_crit)

            valid_ids = get_valid_cluster_ids(cluster_full_t, min_cluster_size)
            if valid_ids is not None:
                id_ok = np.isin(extr_cluster_id, list(valid_ids)) if valid_ids else np.zeros_like(valid_mask, dtype=bool)
                valid_mask = valid_mask & id_ok

            valid_indices = np.where(valid_mask)

            extrema_list = []
            for i, j in zip(valid_indices[0], valid_indices[1]):
                # i = индекс south_north (аналог lat_idx у ERA5), j = west_east (lon_idx)
                lon_val = float(xlong[i, j])
                lat_val = float(xlat[i, j])
                rad_val = float(extr_rad[i, j])
                crit_val = float(extr_crit[i, j])
                wspd_val = float(extr_wspd[i, j])

                extrema_list.append({
                    "lon_idx": int(j),
                    "lat_idx": int(i),
                    "lon": lon_val,
                    "lat": lat_val,
                    "rad": rad_val,
                    "crit": crit_val,
                    "wspd": wspd_val,
                })

            n_extrema = len(extrema_list)
            f.write(f"{current_time.year}\t{current_time.month}\t{current_time.day}\t{n_extrema}\t{current_time.hour}\n")
            for ext in extrema_list:
                f.write(
                    f"\t{ext['lon_idx']}\t{ext['lat_idx']}\t{ext['lon']:.6f}\t{ext['lat']:.6f}\t"
                    f"{ext['rad']:.6e}\t{ext['crit']:.6e}\t{ext['wspd']:.6e}\n"
                )

    ds.close()
    print(f"Добавлено {n_times} временных шагов из {nc_file_path}")
    return n_times


def _month_key_for_file(nc_file: Path) -> str | None:
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", nc_file.name)
    if match:
        year, month, _day = match.groups()
        return f"{year}-{month}"
    # Fallback: имя файла без даты - открываем и смотрим на XTIME/дату первого среза.
    try:
        with xr.open_dataset(nc_file) as ds:
            times = _load_times(ds, str(nc_file))
            return f"{times[0].year}-{times[0].month:02d}"
    except Exception as e:
        print(f"Не удалось определить месяц для {nc_file}: {e}")
        return None


def convert_entire_dataset_naad(
    nc_dir: str,
    output_dir: str,
    data_type: str,
    extr_type: str = "_local",
    min_cluster_size=None,
) -> None:
    """NAAD-аналог convert_entire_dataset() (ERA5). nc_dir уже year-scoped
    (см. naad_config.nc_dir_for_year) - в отличие от ERA5 здесь НЕ нужен
    years_range, вся папка = один год."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    nc_files = sorted(Path(nc_dir).glob("*.nc"))
    if not nc_files:
        print(f"nc-файлы не найдены в {nc_dir}")
        return

    print(f"Найдено {len(nc_files)} файлов в {nc_dir}")

    files_by_month: dict[str, list[Path]] = {}
    for nc_file in nc_files:
        month_key = _month_key_for_file(nc_file)
        if month_key is None:
            continue
        files_by_month.setdefault(month_key, []).append(nc_file)

    for month_key, files in files_by_month.items():
        year, month = month_key.split("-")
        output_filename = f"{data_type}_R2D_extr_{year}-{month}.txt"
        output_path = Path(output_dir) / output_filename
        # Атомарная публикация по месяцу - см. докстринг 1_create_Nodes_from_DBSCAN.py
        # (ERA5) про недописанные после kill файлы; тот же приём здесь.
        tmp_path = output_path.with_suffix(output_path.suffix + f".partial.pid{os.getpid()}")

        if output_path.exists():
            print(f"Файл {output_filename} уже существует. Пропускаем {month_key}")
            continue

        print(f"\nОбработка месяца {month_key}. Файлов: {len(files)}")

        failed_files = []
        wrote_any = False
        for nc_file in sorted(files):
            try:
                convert_nc_to_txt_format_naad(
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

        tmp_path.rename(output_path)
        print(f"Файл сохранён: {output_path}")


def run_for_combo(
    data_type: str,
    year: int,
    size_filter: int,
    extr_type: str = "_local",
    sigma: int = cfg.DEFAULT_SIGMA,
    eps: int = cfg.DEFAULT_EPS,
    min_samples: int = cfg.DEFAULT_MIN_SAMPLES,
    region_name: str = cfg.DEFAULT_REGION_NAME,
    path_init: str = cfg.PATH_INIT,
) -> str:
    """Один прогон Stage B для NAAD: (data_type, year, size_filter, extr_type).
    Схема выходных папок идентична ERA5-версии (см. naad_config.py), чтобы
    2_run_StitchNodes_by_year.py/2_run_StitchNodes_by_year_NAAD.py и
    3_create_csv_tracks_from_StitchNodes.py подхватили результат без правок.
    """
    extr_suffix = extr_type.lstrip("_")

    nc_dir = cfg.nc_dir_for_year(
        data_type=data_type, year=year, eps=eps, min_samples=min_samples,
        size_filter=cfg.SOURCE_SIZE_FILTER, sigma=sigma,
    )

    combo_dir = (
        f"{path_init}/TempestExtremes/{data_type}/R2D_{data_type}_{region_name}_sigma_{sigma}/"
        f"{eps:02d}-{min_samples:02d}-{size_filter:02d}_{extr_suffix}"
    )
    output_dir = f"{combo_dir}/R2D_txt_files_{year}"

    convert_entire_dataset_naad(
        nc_dir=nc_dir,
        output_dir=output_dir,
        data_type=data_type,
        extr_type=extr_type,
        min_cluster_size=size_filter,
    )
    return combo_dir


def main():
    parser = argparse.ArgumentParser(
        description="Stage B (NAAD): DBSCAN nc -> узлы для StitchNodes (LoRes/HiRes)"
    )
    parser.add_argument("--data-type", choices=["LoRes", "HiRes"], required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--size-filter", type=int, default=cfg.SOURCE_SIZE_FILTER,
                         help="Целевой порог отбора вихрей (пост-фильтр по числу точек в кластере, "
                              f"поверх Stage A, посчитанного с SOURCE_SIZE_FILTER={cfg.SOURCE_SIZE_FILTER})")
    parser.add_argument("--extr-type", choices=["global", "local"], default="local",
                         help="Без подчёркивания в CLI (как у Stage C) - внутри скрипта всё равно "
                              "используется как '_global'/'_local' (см. extr_type в run_for_combo)")
    parser.add_argument("--sigma", type=int, default=cfg.DEFAULT_SIGMA)
    parser.add_argument("--eps", type=int, default=cfg.DEFAULT_EPS)
    parser.add_argument("--min-samples", type=int, default=cfg.DEFAULT_MIN_SAMPLES)
    parser.add_argument("--region-name", default=cfg.DEFAULT_REGION_NAME)
    parser.add_argument("--path-init", default=cfg.PATH_INIT)
    args = parser.parse_args()

    combo_dir = run_for_combo(
        data_type=args.data_type,
        year=args.year,
        size_filter=args.size_filter,
        extr_type=f"_{args.extr_type}",
        sigma=args.sigma,
        eps=args.eps,
        min_samples=args.min_samples,
        region_name=args.region_name,
        path_init=args.path_init,
    )
    print(f"\nДля Stage C (2_run_StitchNodes_by_year_NAAD.py) передайте:")
    print(f"  --data-type {args.data_type} --region-name {args.region_name} --sigma {args.sigma} "
          f"--eps {args.eps} --min-samples {args.min_samples} --size-filter {args.size_filter} "
          f"--extr-type {args.extr_type} --years {args.year}")
    print(f"(combo_dir = {combo_dir})")


if __name__ == "__main__":
    main()
