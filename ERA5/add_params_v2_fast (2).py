"""
Обработка данных ERA5 и расчёт параметров циклонов — ускоренная версия (v2).

Что изменилось относительно add_params_2026-08-19_optimized (v1) и почему это
должно быть быстрее, БЕЗ изменения статистик (MEDIAN/PERCENTILE_95/STD/SUM/POINT
остаются ровно теми же, что и раньше — от NodeFileEditor отказались, см.
обсуждение):

1. ФАЙЛЫ БОЛЬШЕ НЕ ПЕРЕОТКРЫВАЮТСЯ НА КАЖДЫЙ ЧАС/ПЕРЕМЕННУЮ.
   В v1 `_load_grib` открывал GRIB-файл заново под каждую (переменная, уровень,
   час) — то есть один и тот же дневной GRIB-файл открывался до
   6 переменных x 2 уровня x 24 часа = 288 раз. Теперь GRIB открывается
   ОДИН РАЗ на (день, уровень) через `filter_by_keys` только по
   typeOfLevel+level (без shortName) — cfgrib сам соберёт t,u,v,q,z,r в один
   Dataset, т.к. они лежат на одной сетке. Итог: 2 открытия GRIB на день
   вместо 288.
   То же самое для NetCDF (mslp/uv10m/t2/precip/blh/tropopause): раньше
   `with xr.open_dataset(...)` вызывался на каждый час, теперь Dataset
   открывается один раз на файл (месяц) и держится в LRU-кэше, читаются
   только маленькие срезы по времени.

2. НЕТ ПОВТОРНЫХ ВЫЧИСЛЕНИЙ НА ТОЧКУ ТРЕКА.
   В v1 `np.hypot(u, v)` для ветра на 10 м / 850 / 500 гПа считался заново
   для КАЖДОЙ точки трека, хотя это одно и то же 2D-поле для всего часа.
   Если через час проходит 50 точек разных треков — это 50 одинаковых
   пересчётов hypot на полной сетке 1440x721. Теперь эти производные поля
   считаются один раз в `loader.load(time)` и переиспользуются.

3. ОДИН ПРОХОД ПО КАЛЕНДАРНЫМ ДНЯМ ВСЕГО ПЕРИОДА (а не по батчам треков).
   Раньше треки обрабатывались батчами по дате СТАРТА трека, но точки одного
   трека растянуты на много дней вперёд — соседние батчи неизбежно повторно
   трогали одни и те же дни ERA5 (трек из батча N доживал до дат, которые уже
   "принадлежат" батчу N+1). Теперь весь период проходится как ЕДИНЫЙ
   отсортированный список календарных дней: каждый день ERA5 (= один
   GRIB-файл) открывается РОВНО ОДИН РАЗ за весь прогон, дни идут по порядку
   (максимальная повторная используемость месячных NetCDF-файлов), задачи
   раздаются по процессам через persistent `multiprocessing.Pool.imap`
   (результаты возвращаются строго в хронологическом порядке дней, даже если
   воркеры считают их параллельно и не по порядку).

4. ДВУХПРОХОДНАЯ СХЕМА ДЛЯ ЭКОНОМИИ ПАМЯТИ (вместо чтения всех ~26000
   треков сразу).
   Проход 1: читаем time/lat/lon/rad каждого трека, раскладываем точки по
   дням, запоминаем последний день каждого трека — полный DataFrame не
   хранится. Проход 2: идём по дням; как только текущий день оказывается
   последним днём какого-то трека, его CSV перечитывается заново (чтобы
   сохранить исходные колонки), в него вписываются посчитанные параметры,
   он сохраняется и сразу выгружается из памяти. В памяти одновременно
   держатся только результаты треков, которые ещё "не закрылись" — обычно
   небольшое скользящее окно, а не все 26000 треков разом.

Логика агрегации (SpatialAggregatorERA5), список RESULT_PARAMS, формулы для
каждого параметра и radius_deg = row['rad'] * radius_multiplier — не менялись,
скопированы из v1 без изменений.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import date
from enum import Enum
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


# ============================================================================
# Настройки
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    SUM = "sum"
    STD = "std"
    PERCENTILE_95 = 95
    PERCENTILE_5 = 5
    POINT = "point"
    DELTA = "delta"


RESULT_PARAMS = [
    "SLP_diff_cent_95",
    "U10_mean",
    "U850_mean",
    "U500_U850_frac",
    "U500_minus_U850",
    "PV_850_mean",
    "T2_minus_T850_mean",
    "T850_disp",
    "pbl_trop_frac",
    "w_850",
    "RH_850",
    "RAIN_HOURLY_sum",
]


# ============================================================================
# Пути к ERA5
# ============================================================================


def get_data_paths():
    return {
        "mslp": {
            "path": "/storage/thalassa/DATA/ERA5/mslp",
            "pattern": "era5_mslp_{year}-{month:02d}.nc",
            "var": "msl",
            "type": "surface",
        },
        "uv10m": {
            "path": "/storage/thalassa/DATA/ERA5/uv10m",
            "pattern": "era5_uv10m_{year}-{month:02d}.nc",
            "var": ["u10", "v10"],
            "type": "surface",
        },
        "t2": {
            "path": "/storage/thalassa/DATA/ERA5/t2",
            "pattern": "era5_t2_{year}-{month:02d}.nc",
            "var": "t2m",
            "type": "surface",
        },
        "precip": {
            "path": "/storage/thalassa/DATA/ERA5/precip",
            "pattern": "ERA5_precip_{year}-{month:02d}.nc",
            "var": "tp",
            "type": "surface",
        },
        "boundary_layer": {
            "path": "/storage/thalassa/DATA/ERA5/boundary_layer_height",
            "pattern": "era5_boundary_layer_height_{year}-{month:02d}.nc",
            "var": "blh",
            "type": "surface",
        },
        "pl_grib": {
            "path": "/storage/thalassa/DATA/ERA5/PL/grib",
            "pattern": "{year}/era5_pl_{year}-{month:02d}-{day:02d}.grib",
            "var": ["t", "u", "v", "q", "z", "r"],
            "type": "pressure",
            "levels": [850, 500],
        },
        "pl_pv_omega": {
            "path": "/storage/thalassa/DATA/ERA5/PL/NC/pv-omega_850and500hPa",
            "pattern": "{year}/era5_pl_pv-omega_{year}-{month:02d}.nc",
            "var": ["pv", "w"],
            "type": "pressure",
            "levels": [850, 500],
        },
        "tropopause": {
            "path": "/storage/thalassa/users/vkoshkina/data/ERA5/tropopause",
            "pattern": "{year}/era5_{year}_{month:02d}_{day:02d}.nc",
            "var": ["clp_z", "dyn_z", "wmo_1st_z"],
            "type": "tropopause",
        },
    }


def get_file_path(data_config: dict, time: pd.Timestamp) -> str:
    return f"{data_config['path']}/{data_config['pattern'].format(year=time.year, month=time.month, day=time.day)}"


# ============================================================================
# Утилиты xarray
# ============================================================================


def safe_values(data):
    if data is None:
        return None
    if isinstance(data, np.ndarray):
        return data
    if hasattr(data, "values"):
        return data.values
    if hasattr(data, "magnitude"):
        return data.magnitude
    return np.asarray(data)


def _find_time_dim(ds: xr.Dataset) -> Optional[str]:
    for dim in ("time", "valid_time"):
        if dim in ds.dims or dim in ds.coords:
            return dim
    return None


def _select_time(ds: xr.Dataset, time: pd.Timestamp) -> xr.Dataset:
    time_dim = _find_time_dim(ds)
    if time_dim is not None:
        return ds.sel({time_dim: time}, method="nearest")
    return ds


def _select_level(ds: xr.Dataset, level: Optional[int]) -> xr.Dataset:
    if level is None:
        return ds

    for dim in ("isobaricInhPa", "pressure_level"):
        if dim in ds.dims or dim in ds.coords:
            return ds.sel({dim: level}, method="nearest")

    return ds


def _find_var(ds: xr.Dataset, var_name: str) -> Optional[str]:
    if var_name in ds.data_vars:
        return var_name

    for var in ds.data_vars:
        if var_name in var or var in var_name:
            return var

    return None


# ============================================================================
# Кэш открытых Dataset (не срезов!) — главное ускорение I/O.
# ============================================================================


class _OpenDatasetLRU:
    """
    LRU-кэш ОТКРЫТЫХ xr.Dataset (не значений).

    В отличие от v1, где кэшировался уже выбранный временной срез, здесь
    кэшируется сам открытый файл. Это резко уменьшает число открытий файла
    (syscalls + разбор GRIB/NetCDF заголовков), при этом чтение конкретного
    часа всё равно лёгкое, т.к. .sel(...).load() читает только нужный срез.
    """

    def __init__(self, max_open: int = 8):
        self.max_open = max_open
        self._store: "OrderedDict[tuple, xr.Dataset]" = OrderedDict()

    def get_or_open(self, key: tuple, opener):
        if key in self._store:
            ds = self._store.pop(key)
            self._store[key] = ds
            return ds

        ds = opener()
        self._store[key] = ds

        while len(self._store) > self.max_open:
            _, old_ds = self._store.popitem(last=False)
            try:
                old_ds.close()
            except Exception:
                pass

        return ds

    def clear(self):
        for ds in self._store.values():
            try:
                ds.close()
            except Exception:
                pass
        self._store.clear()


# ============================================================================
# Загрузчик ERA5
# ============================================================================


class ERA5DataLoader:
    """
    Загрузчик ERA5.

    Кэшируются ОТКРЫТЫЕ Dataset (по файлу, для GRIB — по файлу+уровню), а не
    отдельные временные срезы. Один и тот же файл может переиспользоваться
    сотни раз (все часы месяца/суток) без повторного открытия.
    """

    def __init__(self, data_paths: dict, max_open_datasets: int = 8):
        self.data_paths = data_paths
        self._ds_cache = _OpenDatasetLRU(max_open=max_open_datasets)
        self._missing_files = set()

    # ------------------------------------------------------------------
    # NetCDF (поверхностные переменные, pv/omega, tropopause)
    # ------------------------------------------------------------------

    def _get_nc_dataset(self, file_path: str) -> Optional[xr.Dataset]:
        if file_path in self._missing_files:
            return None
        if not Path(file_path).exists():
            self._missing_files.add(file_path)
            return None

        key = ("nc", file_path)
        return self._ds_cache.get_or_open(key, lambda: xr.open_dataset(file_path))

    def _load_nc(
        self,
        param_type: str,
        time: pd.Timestamp,
        var_name: str,
        level: Optional[int] = None,
    ) -> Optional[xr.DataArray]:
        config = self.data_paths[param_type]
        file_path = get_file_path(config, time)

        try:
            ds = self._get_nc_dataset(file_path)
            if ds is None:
                return None

            selected = _select_time(ds, time)
            selected = _select_level(selected, level)
            actual_var = _find_var(selected, var_name)
            if actual_var is None:
                return None

            return selected[actual_var].load()

        except Exception as exc:
            logger.debug("Ошибка загрузки %s из %s: %s", param_type, file_path, exc)
            return None

    # ------------------------------------------------------------------
    # GRIB (850/500 гПа: t, u, v, q, z, r)
    # ------------------------------------------------------------------

    def _get_grib_dataset(self, file_path: str, level: int) -> Optional[xr.Dataset]:
        cache_key_missing = (file_path, level)
        if cache_key_missing in self._missing_files:
            return None
        if not Path(file_path).exists():
            self._missing_files.add(cache_key_missing)
            return None

        key = ("grib", file_path, level)

        def _opener():
            # Фильтруем ТОЛЬКО по typeOfLevel+level (без shortName) — тогда
            # cfgrib сам объединит t,u,v,q,z,r (все на одной сетке/уровне) в
            # один Dataset, и файл открывается один раз на весь день+уровень,
            # а не один раз на каждую переменную.
            return xr.open_dataset(
                file_path,
                engine="cfgrib",
                backend_kwargs={
                    "filter_by_keys": {
                        "typeOfLevel": "isobaricInhPa",
                        "level": level,
                    },
                    "indexpath": "",
                },
            )

        try:
            return self._ds_cache.get_or_open(key, _opener)
        except Exception as exc:
            logger.debug(
                "Ошибка открытия GRIB %s, level=%s: %s", file_path, level, exc
            )
            self._missing_files.add(cache_key_missing)
            return None

    def _load_grib_group(
        self, time: pd.Timestamp, level: int, var_names: List[str]
    ) -> Dict[str, Optional[xr.DataArray]]:
        config = self.data_paths["pl_grib"]
        file_path = get_file_path(config, time)

        ds = self._get_grib_dataset(file_path, level)
        result = {v: None for v in var_names}
        if ds is None:
            return result

        try:
            selected = _select_time(ds, time)
        except Exception as exc:
            logger.debug("Ошибка выбора времени в GRIB %s: %s", file_path, exc)
            return result

        for var_name in var_names:
            try:
                actual_var = _find_var(selected, var_name)
                if actual_var is None:
                    continue
                result[var_name] = selected[actual_var].load()
            except Exception as exc:
                logger.debug(
                    "Ошибка чтения переменной %s (level=%s) из %s: %s",
                    var_name,
                    level,
                    file_path,
                    exc,
                )

        return result

    def _load_pv_omega(
        self, time: pd.Timestamp, var_name: str, level: int
    ) -> Optional[xr.DataArray]:
        return self._load_nc("pl_pv_omega", time, var_name, level)

    # ------------------------------------------------------------------
    # Главная точка входа
    # ------------------------------------------------------------------

    def load(self, time: pd.Timestamp) -> Dict[str, Optional[Any]]:
        """
        Загружает весь набор ERA5, необходимый для одного timestamp, плюс
        заранее считает производные поля (скорость ветра и т.п.), чтобы не
        пересчитывать их отдельно для каждой точки трека этого часа.
        """
        time = pd.Timestamp(time)

        data: Dict[str, Optional[Any]] = {}

        # Поверхность
        data["mslp"] = self._load_nc("mslp", time, "msl")
        data["u10"] = self._load_nc("uv10m", time, "u10")
        data["v10"] = self._load_nc("uv10m", time, "v10")
        data["t2"] = self._load_nc("t2", time, "t2m")
        data["precip"] = self._load_nc("precip", time, "tp")
        data["blh"] = self._load_nc("boundary_layer", time, "blh")
        data["tropopause"] = self._load_nc("tropopause", time, "dyn_z")

        # Pressure levels — один open на уровень, а не на каждую переменную.
        pl850 = self._load_grib_group(time, 850, ["t", "u", "v", "r"])
        data["t850"] = pl850["t"]
        data["u850"] = pl850["u"]
        data["v850"] = pl850["v"]
        data["r850"] = pl850["r"]

        pl500 = self._load_grib_group(time, 500, ["u", "v"])
        data["u500"] = pl500["u"]
        data["v500"] = pl500["v"]

        # PV и omega — отдельный NC-файл.
        data["pv850"] = self._load_pv_omega(time, "pv", 850)
        data["w850"] = self._load_pv_omega(time, "w", 850)

        # ------------------------------------------------------------------
        # Производные поля — считаем ОДИН раз на timestamp, а не на точку.
        # ------------------------------------------------------------------
        data["_wind10"] = None
        if data["u10"] is not None and data["v10"] is not None:
            data["_wind10"] = np.hypot(safe_values(data["u10"]), safe_values(data["v10"]))

        data["_wind850"] = None
        data["_du_500_850"] = None
        data["_dv_500_850"] = None
        data["_wind500"] = None
        if data["u850"] is not None and data["v850"] is not None:
            u850_np = safe_values(data["u850"])
            v850_np = safe_values(data["v850"])
            data["_wind850"] = np.hypot(u850_np, v850_np)

            if data["u500"] is not None and data["v500"] is not None:
                u500_np = safe_values(data["u500"])
                v500_np = safe_values(data["v500"])
                data["_wind500"] = np.hypot(u500_np, v500_np)
                data["_du_500_850"] = u500_np - u850_np
                data["_dv_500_850"] = v500_np - v850_np

        return data

    def clear_cache(self):
        self._ds_cache.clear()
        self._missing_files.clear()


# ============================================================================
# Пространственный агрегатор (не изменён относительно v1)
# ============================================================================


class SpatialAggregatorERA5:
    """
    Агрегация по точкам регулярной ERA5-сетки.

    Сначала выбирается прямоугольное окно индексов вокруг ближайшей точки
    центра, затем внутри него остаются точки, расстояние до которых меньше
    radius_deg.
    """

    def __init__(
        self,
        center_lat: float,
        center_lon: float,
        radius_deg: float,
        grid_lat: np.ndarray,
        grid_lon: np.ndarray,
    ):
        self.center_lat = float(center_lat)
        self.center_lon = float(center_lon)
        self.radius_deg = float(radius_deg)

        self.lat = np.asarray(grid_lat)
        self.lon = np.asarray(grid_lon)

        if self.lat.ndim != 1 or self.lon.ndim != 1:
            raise ValueError("Ожидаются одномерные координаты latitude/longitude")

        self.lat_idx = int(np.argmin(np.abs(self.lat - self.center_lat)))
        self.lon_idx = int(np.argmin(np.abs(self._lon_difference(self.lon, self.center_lon))))

        self._lat_indices, self._lon_indices = self._make_local_window()
        self._mask = self._make_radius_mask()

        self._lat_idx_2d, self._lon_idx_2d = np.meshgrid(
            self._lat_indices,
            self._lon_indices,
            indexing="ij",
        )
        self._lat_idx_flat = self._lat_idx_2d[self._mask]
        self._lon_idx_flat = self._lon_idx_2d[self._mask]

    @staticmethod
    def _lon_difference(lon, center_lon):
        return (lon - center_lon + 180.0) % 360.0 - 180.0

    def _make_local_window(self):
        lat_min = self.center_lat - self.radius_deg
        lat_max = self.center_lat + self.radius_deg

        lat_indices = np.flatnonzero((self.lat >= lat_min) & (self.lat <= lat_max))

        lon_diff = np.abs(self._lon_difference(self.lon, self.center_lon))
        lon_indices = np.flatnonzero(lon_diff <= self.radius_deg)

        if lat_indices.size == 0:
            lat_indices = np.array([self.lat_idx])
        if lon_indices.size == 0:
            lon_indices = np.array([self.lon_idx])

        return lat_indices, lon_indices

    def _make_radius_mask(self):
        lat_local = self.lat[self._lat_indices][:, None]
        lon_local = self.lon[self._lon_indices][None, :]

        dlat = lat_local - self.center_lat
        dlon = self._lon_difference(lon_local, self.center_lon)

        dlon_metric = dlon * np.cos(np.deg2rad(self.center_lat))
        distance_deg = np.sqrt(dlat**2 + dlon_metric**2)

        return distance_deg <= self.radius_deg

    def _values_in_radius(self, data) -> np.ndarray:
        data_np = safe_values(data)
        if data_np is None:
            return np.empty(0, dtype=float)

        values = data_np[np.ix_(self._lat_indices, self._lon_indices)]
        values = values[self._mask]
        return values[np.isfinite(values)]

    def aggregate(self, data, method: AggregationMethod) -> float:
        if data is None:
            return np.nan

        data_np = safe_values(data)
        if data_np is None or data_np.size == 0:
            return np.nan

        if method == AggregationMethod.POINT:
            return float(data_np[self.lat_idx, self.lon_idx])

        values = self._values_in_radius(data_np)
        if values.size == 0:
            return np.nan

        if method == AggregationMethod.MEAN:
            return float(np.mean(values))
        if method == AggregationMethod.MEDIAN:
            return float(np.median(values))
        if method == AggregationMethod.MIN:
            return float(np.min(values))
        if method == AggregationMethod.MAX:
            return float(np.max(values))
        if method == AggregationMethod.SUM:
            return float(np.sum(values))
        if method == AggregationMethod.STD:
            return float(np.std(values, ddof=1))
        if method == AggregationMethod.DELTA:
            return float(np.percentile(values, 95) - np.percentile(values, 5))
        if isinstance(method.value, (int, float)):
            return float(np.percentile(values, method.value))

        return float(np.median(values))

    def aggregate_vector(self, real_part, imag_part):
        if real_part is None or imag_part is None:
            return np.nan, np.nan, np.nan, np.nan

        real_np = safe_values(real_part)
        imag_np = safe_values(imag_part)

        real_values = real_np[np.ix_(self._lat_indices, self._lon_indices)][self._mask]
        imag_values = imag_np[np.ix_(self._lat_indices, self._lon_indices)][self._mask]

        valid = np.isfinite(real_values) & np.isfinite(imag_values)
        if not np.any(valid):
            return np.nan, np.nan, np.nan, np.nan

        mean_real = float(np.mean(real_values[valid]))
        mean_imag = float(np.mean(imag_values[valid]))
        mean_angle = float(np.degrees(np.arctan2(mean_imag, mean_real)) % 360.0)
        mean_magnitude = float(np.hypot(mean_real, mean_imag))

        return mean_real, mean_imag, mean_angle, mean_magnitude


# ============================================================================
# Расчёт параметров одного положения трека
# ============================================================================


class CycloneProcessorERA5:
    def __init__(
        self,
        data_paths: dict,
        radius_multiplier: float = 4.0,
        max_open_datasets: int = 8,
    ):
        self.data_paths = data_paths
        self.radius_multiplier = radius_multiplier
        self.loader = ERA5DataLoader(data_paths, max_open_datasets=max_open_datasets)

    @staticmethod
    def _get_lat_lon(data_dict: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        for key, data in data_dict.items():
            if key.startswith("_"):
                continue
            if data is not None:
                lat_name = "latitude" if "latitude" in data.coords else "lat"
                lon_name = "longitude" if "longitude" in data.coords else "lon"
                return data[lat_name].values, data[lon_name].values

        raise ValueError("Не найдено ни одного ERA5 поля с координатами")

    def _process_point(
        self,
        row: pd.Series,
        era5: Dict[str, Optional[Any]],
    ) -> Dict[str, Any]:
        center_lat = float(row["lat"])
        center_lon = float(row["lon"])

        radius_deg = float(row["rad"]) * self.radius_multiplier

        grid_lat, grid_lon = self._get_lat_lon(era5)
        aggregator = SpatialAggregatorERA5(
            center_lat=center_lat,
            center_lon=center_lon,
            radius_deg=radius_deg,
            grid_lat=grid_lat,
            grid_lon=grid_lon,
        )

        results = {param: np.nan for param in RESULT_PARAMS}

        mslp = era5["mslp"]
        t2 = era5["t2"]
        precip = era5["precip"]
        blh = era5["blh"]
        tropopause = era5["tropopause"]

        t850 = era5["t850"]
        r850 = era5["r850"]
        pv850 = era5["pv850"]
        w850 = era5["w850"]

        # Уже готовые производные поля (посчитаны один раз на timestamp).
        wind10 = era5["_wind10"]
        wind850 = era5["_wind850"]
        wind500 = era5["_wind500"]
        du_500_850 = era5["_du_500_850"]
        dv_500_850 = era5["_dv_500_850"]

        # ------------------------------------------------------------------
        # SLP
        # ------------------------------------------------------------------
        if mslp is not None:
            slp_center = aggregator.aggregate(mslp, AggregationMethod.POINT)
            slp_95 = aggregator.aggregate(mslp, AggregationMethod.PERCENTILE_95)
            results["SLP_diff_cent_95"] = 0.01 * (slp_95 - slp_center)

        # ------------------------------------------------------------------
        # 10 m wind
        # ------------------------------------------------------------------
        if wind10 is not None:
            results["U10_mean"] = aggregator.aggregate(wind10, AggregationMethod.MEDIAN)

        # ------------------------------------------------------------------
        # 850/500 hPa wind
        # ------------------------------------------------------------------
        if wind850 is not None:
            results["U850_mean"] = aggregator.aggregate(wind850, AggregationMethod.MEDIAN)

            if wind500 is not None:
                u500_mean = aggregator.aggregate(wind500, AggregationMethod.MEDIAN)

                if np.isfinite(results["U850_mean"]) and results["U850_mean"] > 0:
                    results["U500_U850_frac"] = u500_mean / results["U850_mean"]

            if du_500_850 is not None and dv_500_850 is not None:
                _, _, _, mean_dV = aggregator.aggregate_vector(du_500_850, dv_500_850)
                results["U500_minus_U850"] = mean_dV

        # ------------------------------------------------------------------
        # PV
        # ------------------------------------------------------------------
        if pv850 is not None:
            results["PV_850_mean"] = aggregator.aggregate(
                safe_values(pv850) * 1_000_000,
                AggregationMethod.MEDIAN,
            )

        # ------------------------------------------------------------------
        # T2 - T850 и дисперсия T850
        # ------------------------------------------------------------------
        if t2 is not None and t850 is not None:
            t2_med = aggregator.aggregate(t2, AggregationMethod.MEDIAN)
            t850_med = aggregator.aggregate(t850, AggregationMethod.MEDIAN)
            results["T2_minus_T850_mean"] = t2_med - t850_med
            results["T850_disp"] = aggregator.aggregate(t850, AggregationMethod.STD)

        # ------------------------------------------------------------------
        # Boundary layer / tropopause
        # ------------------------------------------------------------------
        if blh is not None and tropopause is not None:
            blh_val = aggregator.aggregate(blh, AggregationMethod.MEDIAN)
            trop_val = aggregator.aggregate(tropopause, AggregationMethod.MEDIAN)
            results["pbl_trop_frac"] = (
                blh_val * 1000.0 / trop_val if np.isfinite(trop_val) and trop_val > 0 else np.nan
            )

        # ------------------------------------------------------------------
        # Omega, RH, precipitation
        # ------------------------------------------------------------------
        if w850 is not None:
            results["w_850"] = aggregator.aggregate(w850, AggregationMethod.PERCENTILE_95)

        if r850 is not None:
            results["RH_850"] = aggregator.aggregate(r850, AggregationMethod.MEDIAN)

        if precip is not None:
            results["RAIN_HOURLY_sum"] = aggregator.aggregate(precip, AggregationMethod.SUM)

        return results

    # ------------------------------------------------------------------
    # Обработка одних суток (используется и в один процесс, и в воркере).
    # ------------------------------------------------------------------

    def process_day_points(
        self, day_points: List[Tuple[pd.Timestamp, int, int, float, float, float]]
    ) -> List[Tuple[int, int, Dict[str, Any]]]:
        """
        day_points: список (time, track_id, row_idx, lat, lon, rad) — все точки
        всех треков, приходящиеся на один календарный день.

        Возвращает список (track_id, row_idx, results_dict).
        """
        out: List[Tuple[int, int, Dict[str, Any]]] = []

        by_time: Dict[pd.Timestamp, list] = defaultdict(list)
        for time, track_id, row_idx, lat, lon, rad in day_points:
            by_time[time].append((track_id, row_idx, lat, lon, rad))

        for current_time, items in by_time.items():
            try:
                era5 = self.loader.load(current_time)
            except Exception as exc:
                logger.exception("Ошибка загрузки ERA5 для %s: %s", current_time, exc)
                continue

            if not any(
                v is not None for k, v in era5.items() if not k.startswith("_")
            ):
                continue

            for track_id, row_idx, lat, lon, rad in items:
                row = pd.Series({"lat": lat, "lon": lon, "rad": rad})
                try:
                    results = self._process_point(row, era5)
                except Exception as exc:
                    logger.exception(
                        "Ошибка обработки точки track=%s row=%s time=%s: %s",
                        track_id,
                        row_idx,
                        current_time,
                        exc,
                    )
                    results = {param: np.nan for param in RESULT_PARAMS}
                out.append((track_id, row_idx, results))

        # ВАЖНО: кэш датасетов НЕ закрываем здесь. process_day_points может
        # вызываться много раз подряд на одном и том же воркере (для разных
        # календарных дней одного батча треков), и месячные NetCDF-файлы
        # (mslp/uv10m/t2/precip/blh) переиспользуются между соседними днями.
        # Память и так ограничена LRU (max_open_datasets) внутри loader-а.
        return out


# ============================================================================
# Воркер для multiprocessing (модульная функция — обязательна для pickling).
# ============================================================================

_WORKER_STATE: Dict[str, Any] = {}


def _worker_init(data_paths: dict, radius_multiplier: float, max_open_datasets: int):
    """Вызывается один раз на процесс — создаёт свой процессор с чистым кэшем."""
    _WORKER_STATE["processor"] = CycloneProcessorERA5(
        data_paths,
        radius_multiplier=radius_multiplier,
        max_open_datasets=max_open_datasets,
    )


def _worker_process_day(day_points):
    processor: CycloneProcessorERA5 = _WORKER_STATE["processor"]
    return processor.process_day_points(day_points)


# ============================================================================
# Проход 1: лёгкое чтение точек треков (без хранения полных DataFrame)
# ============================================================================


def _read_track_points_only(
    track_file: str,
) -> Optional[Tuple[List[pd.Timestamp], List[float], List[float], List[float]]]:
    """
    Читает CSV трека и возвращает только (time, lat, lon, rad) по точкам.
    Полный DataFrame НЕ сохраняется — он будет прочитан заново на этапе
    записи результата (см. _flush_track), чтобы не держать в памяти все
    ~26000 треков одновременно.
    """
    try:
        df = pd.read_csv(track_file)

        if "time" in df.columns:
            times = pd.to_datetime(df["time"])
        elif "datetime" in df.columns:
            times = pd.to_datetime(df["datetime"])
        else:
            logger.error("Нет time/datetime: %s", track_file)
            return None

        if not {"lat", "lon", "rad"}.issubset(df.columns):
            logger.error("В %s отсутствует одна из колонок lat/lon/rad", track_file)
            return None

        return (
            list(times),
            df["lat"].astype(float).tolist(),
            df["lon"].astype(float).tolist(),
            df["rad"].astype(float).tolist(),
        )

    except Exception as exc:
        logger.error("Ошибка чтения %s: %s", track_file, exc)
        return None


def _flush_track(
    track_file: str,
    results_map: Dict[int, Dict[str, Any]],
    output_dir: Path,
):
    """
    Перечитывает исходный CSV трека (чтобы сохранить все его колонки),
    проставляет посчитанные параметры и сохраняет результат. Вызывается
    ровно один раз на трек — как только обработаны все дни, на которые
    приходятся его точки.
    """
    try:
        df = pd.read_csv(track_file)
        for param in RESULT_PARAMS:
            df[param] = np.nan

        for row_idx, results in results_map.items():
            for param, value in results.items():
                df.at[row_idx, param] = value

        output_file = output_dir / Path(track_file).name
        df.to_csv(output_file, index=False)

    except Exception as exc:
        logger.error("Ошибка записи результата для %s: %s", track_file, exc)


# ============================================================================
# Главный драйвер: один проход по календарным дням всего периода.
# ============================================================================


def process_all_tracks(
    track_files,
    output_dir: Path,
    data_paths: dict,
    radius_multiplier: float = 4.0,
    n_workers: int = 4,
    max_open_datasets: int = 8,
):
    """
    Идея: каждый календарный день ERA5 должен обрабатываться РОВНО ОДИН РАЗ
    за весь прогон (а не по разу на каждый "батч треков", как раньше — это
    приводило к повторному открытию одних и тех же файлов, если трек
    пересекал границу батча).

    Проход 1 (лёгкий): читаем time/lat/lon/rad всех треков, группируем точки
    по календарному дню, запоминаем последний день каждого трека. Полные
    DataFrame не хранятся.

    Проход 2: идём по дням СТРОГО по хронологии (persistent Pool.imap — он
    отдаёт результаты в порядке отправки задач, даже если воркеры считают их
    параллельно и не по порядку). Результаты копятся в
    results_by_track[track_idx][row_idx]. Как только текущий день оказывается
    последним днём какого-то трека — CSV этого трека перечитывается, в него
    вписываются посчитанные параметры, он сохраняется и сразу выгружается из
    памяти.

    В памяти одновременно держатся только результаты треков, ещё не
    "закрывшихся" (обычно небольшое скользящее окно — треки живут дни/недели,
    а не весь период целиком).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Проход 1/2: чтение точек %d треков (без хранения DataFrame)", len(track_files))

    day_buckets: Dict[date, list] = defaultdict(list)
    track_last_day: Dict[int, date] = {}
    n_points = 0

    for track_idx, track_file in enumerate(tqdm(track_files, desc="Чтение треков")):
        res = _read_track_points_only(track_file)
        if res is None:
            continue

        times, lats, lons, rads = res
        if not times:
            continue

        last_day = None
        for row_idx, (t, lat, lon, rad) in enumerate(zip(times, lats, lons, rads)):
            t = pd.Timestamp(t)
            d = t.date()
            day_buckets[d].append((t, track_idx, row_idx, lat, lon, rad))
            n_points += 1
            if last_day is None or d > last_day:
                last_day = d

        if last_day is not None:
            track_last_day[track_idx] = last_day

    if not day_buckets:
        logger.warning("Нет точек для обработки")
        return

    # После этого момента полные строки точек больше не нужны отдельно —
    # они уже разложены по дням в day_buckets.

    flush_schedule: Dict[date, List[int]] = defaultdict(list)
    for track_idx, last_day in track_last_day.items():
        flush_schedule[last_day].append(track_idx)

    sorted_days = sorted(day_buckets.keys())
    day_tasks = [day_buckets[d] for d in sorted_days]

    logger.info(
        "Всего точек: %d; треков с точками: %d; уникальных дней: %d; воркеров: %d",
        n_points,
        len(track_last_day),
        len(sorted_days),
        n_workers,
    )

    pool = None
    sequential_processor = None
    ctx = mp.get_context("spawn")

    if n_workers > 1:
        pool = ctx.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(data_paths, radius_multiplier, max_open_datasets),
        )
    else:
        sequential_processor = CycloneProcessorERA5(
            data_paths,
            radius_multiplier=radius_multiplier,
            max_open_datasets=max_open_datasets,
        )

    results_by_track: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)

    try:
        logger.info("Проход 2/2: обработка по дням (хронологически, каждый день — один раз)")

        if pool is not None:
            # imap (НЕ imap_unordered) — результаты приходят строго в порядке
            # отправки задач, то есть в хронологическом порядке дней, хотя
            # сами воркеры считают дни параллельно и не обязательно по порядку.
            result_iter = pool.imap(_worker_process_day, day_tasks)
        else:
            result_iter = (
                sequential_processor.process_day_points(day_points) for day_points in day_tasks
            )

        for current_day, day_result in tqdm(
            zip(sorted_days, result_iter), total=len(sorted_days), desc="Дни"
        ):
            for track_idx, row_idx, results in day_result:
                results_by_track[track_idx][row_idx] = results

            # Закрываем треки, для которых сегодняшний день — последний.
            for track_idx in flush_schedule.get(current_day, []):
                results_map = results_by_track.pop(track_idx, {})
                _flush_track(track_files[track_idx], results_map, output_dir)

    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # На случай гонки/неучтённых треков — досохраняем всё, что осталось.
    for track_idx, results_map in results_by_track.items():
        _flush_track(track_files[track_idx], results_map, output_dir)

    logger.info("Обработка завершена.")


# ============================================================================
# Точка входа
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Расчёт параметров циклонов по ERA5 (v2, ускоренная)")
    parser.add_argument("--year", type=int, default=2010)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, mp.cpu_count() - 1),
        help="Число процессов (по одному на набор суток). 1 = последовательно, для отладки.",
    )
    parser.add_argument(
        "--max-open-datasets",
        type=int,
        default=8,
        help="Сколько открытых xr.Dataset держать в LRU-кэше на процесс.",
    )
    args = parser.parse_args()

    data_paths = get_data_paths()

    year = args.year

    path_init = "/storage/thalassa/users/vkoshkina"
    path_dir_data = f"{path_init}/data"

    size_filter = 25
    extr_type = "_global"
    postfix = f"_range_1_5_18h_12h_2010_{size_filter}points{extr_type}"

    TRACKS_PATH = (
        f"{path_dir_data}/TempestExtremes/ERA5/"
        "R2D_ERA5_NA_for_TC_850hPa_sigma_2/"
        f"csv_Tracks{postfix}"
    )

    OUTPUT_PATH = Path(f"{TRACKS_PATH}_params")
    output_dir = OUTPUT_PATH / "hourly_data"

    track_files = glob(f"{TRACKS_PATH}/*_track_{year}*.csv")

    if not track_files:
        print(f"Файлы треков не найдены в {TRACKS_PATH}")
    else:
        print(f"Найдено {len(track_files)} треков")
        print(f"Выходная директория: {output_dir}")
        print(f"Воркеров: {args.workers}")

        process_all_tracks(
            track_files,
            output_dir,
            data_paths,
            radius_multiplier=4.0,
            n_workers=args.workers,
            max_open_datasets=args.max_open_datasets,
        )

        print("Обработка завершена!")
