"""
Обработка данных ERA5 и расчёт параметров циклонов.

Основные оптимизации по сравнению с исходной версией:
1. ERA5-файлы не открываются заново для каждого часа/переменной.
2. Данные группируются по timestamp: один ERA5 timestamp используется всеми
   треками, проходящими через этот час.
3. Для одного источника ERA5 и timestamp нужные переменные читаются один раз.
4. Пространственная область берётся как окно индексов вокруг центра, после чего
   применяется простая маска по расстоянию на регулярной сетке.
5. Маска радиуса вычисляется один раз на точку трека и затем используется для
   всех параметров.
6. Результаты сначала собираются в numpy-массивы, а не записываются в DataFrame
   через df.at на каждом шаге.

ВАЖНО:
- В исходном коде было center_lon = row['lat']; здесь исправлено на row['lon'].
- Сохранено исходное правило: radius_deg = row['rad'] * 4.
  Проверьте, что именно такое преобразование соответствует определению `rad`
  в ваших CSV.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from glob import glob
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
# Кэш ERA5
# ============================================================================


class ERA5DataLoader:
    """
    Загрузчик ERA5 с кэшем.

    Кэшируется уже выбранный временной срез, а не только путь к файлу.
    Это важно: один и тот же timestamp может использоваться большим числом
    треков.

    max_cache_size ограничивает память. При обработке timestamp-ов по порядку
    старые срезы автоматически удаляются.
    """

    def __init__(self, data_paths: dict, max_cache_size: int = 24):
        self.data_paths = data_paths
        self.max_cache_size = max_cache_size
        self._cache: OrderedDict[tuple, Optional[xr.DataArray]] = OrderedDict()
        self._missing_cache = set()

    def _cache_get(self, key):
        if key not in self._cache:
            return None, False
        value = self._cache.pop(key)
        self._cache[key] = value
        return value, True

    def _cache_put(self, key, value):
        if key in self._cache:
            self._cache.pop(key)
        self._cache[key] = value
        while len(self._cache) > self.max_cache_size:
            _, old_value = self._cache.popitem(last=False)
            # DataArray содержит ссылки на numpy/xarray objects; удаляем ссылку.
            del old_value

    def _load_nc(
        self,
        param_type: str,
        time: pd.Timestamp,
        var_name: str,
        level: Optional[int] = None,
    ) -> Optional[xr.DataArray]:
        config = self.data_paths[param_type]
        file_path = get_file_path(config, time)
        key = (param_type, file_path, time, var_name, level)

        cached, found = self._cache_get(key)
        if found:
            return cached

        if key in self._missing_cache:
            return None

        if not Path(file_path).exists():
            self._missing_cache.add(key)
            return None

        try:
            with xr.open_dataset(file_path) as ds:
                selected = _select_time(ds, time)
                selected = _select_level(selected, level)
                actual_var = _find_var(selected, var_name)

                if actual_var is None:
                    self._missing_cache.add(key)
                    return None

                # Важно: после выхода из with данные должны остаться доступными.
                data = selected[actual_var].load()

            self._cache_put(key, data)
            return data

        except Exception as exc:
            logger.debug("Ошибка загрузки %s из %s: %s", param_type, file_path, exc)
            self._missing_cache.add(key)
            return None

    def _load_grib(
        self,
        time: pd.Timestamp,
        var_name: str,
        level: int,
    ) -> Optional[xr.DataArray]:
        config = self.data_paths["pl_grib"]
        file_path = get_file_path(config, time)
        key = ("pl_grib", file_path, time, var_name, level)

        cached, found = self._cache_get(key)
        if found:
            return cached

        if key in self._missing_cache:
            return None

        if not Path(file_path).exists():
            self._missing_cache.add(key)
            return None

        try:
            # Для cfgrib фильтруем сразу по переменной и уровню.
            # Это значительно лучше, чем открывать весь GRIB без фильтра.
            ds = xr.open_dataset(
                file_path,
                engine="cfgrib",
                backend_kwargs={
                    "filter_by_keys": {
                        "shortName": var_name,
                        "typeOfLevel": "isobaricInhPa",
                    },
                    "indexpath": "",
                },
            )

            try:
                selected = _select_time(ds, time)
                selected = _select_level(selected, level)
                actual_var = _find_var(selected, var_name)
                if actual_var is None:
                    self._missing_cache.add(key)
                    return None

                data = selected[actual_var].load()
            finally:
                ds.close()

            self._cache_put(key, data)
            return data

        except Exception as exc:
            logger.debug(
                "Ошибка загрузки GRIB %s, level=%s из %s: %s",
                var_name,
                level,
                file_path,
                exc,
            )
            self._missing_cache.add(key)
            return None

    def _load_pv_omega(
        self,
        time: pd.Timestamp,
        var_name: str,
        level: int,
    ) -> Optional[xr.DataArray]:
        return self._load_nc("pl_pv_omega", time, var_name, level)

    def load(self, time: pd.Timestamp) -> Dict[str, Optional[xr.DataArray]]:
        """
        Загружает весь набор ERA5, необходимый для одного timestamp.

        Это единственная функция, которую должен вызывать расчёт параметров.
        """
        time = pd.Timestamp(time)

        data = {}

        # Поверхность
        data["mslp"] = self._load_nc("mslp", time, "msl")
        data["u10"] = self._load_nc("uv10m", time, "u10")
        data["v10"] = self._load_nc("uv10m", time, "v10")
        data["t2"] = self._load_nc("t2", time, "t2m")
        data["precip"] = self._load_nc("precip", time, "tp")
        data["blh"] = self._load_nc("boundary_layer", time, "blh")
        data["tropopause"] = self._load_nc("tropopause", time, "dyn_z")

        # Pressure levels
        data["t850"] = self._load_grib(time, "t", 850)
        data["u850"] = self._load_grib(time, "u", 850)
        data["v850"] = self._load_grib(time, "v", 850)
        data["r850"] = self._load_grib(time, "r", 850)

        data["u500"] = self._load_grib(time, "u", 500)
        data["v500"] = self._load_grib(time, "v", 500)

        # PV и omega находятся в отдельном NC-файле.
        data["pv850"] = self._load_pv_omega(time, "pv", 850)
        data["w850"] = self._load_pv_omega(time, "w", 850)

        return data

    def clear_cache(self):
        self._cache.clear()
        self._missing_cache.clear()


# ============================================================================
# Пространственный агрегатор
# ============================================================================


class SpatialAggregatorERA5:
    """
    Агрегация по точкам регулярной ERA5-сетки.

    В отличие от исходного варианта здесь не строится Haversine-маска по всей
    двумерной сетке для каждого параметра.

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

        # ERA5 regular grid.
        self.lat = np.asarray(grid_lat)
        self.lon = np.asarray(grid_lon)

        if self.lat.ndim != 1 or self.lon.ndim != 1:
            raise ValueError("Ожидаются одномерные координаты latitude/longitude")

        self.lat_idx = int(np.argmin(np.abs(self.lat - self.center_lat)))
        self.lon_idx = int(np.argmin(np.abs(self._lon_difference(self.lon, self.center_lon))))

        self._lat_indices, self._lon_indices = self._make_local_window()
        self._mask = self._make_radius_mask()

        # Индексы после применения маски.
        self._lat_idx_2d, self._lon_idx_2d = np.meshgrid(
            self._lat_indices,
            self._lon_indices,
            indexing="ij",
        )
        self._lat_idx_flat = self._lat_idx_2d[self._mask]
        self._lon_idx_flat = self._lon_idx_2d[self._mask]

    @staticmethod
    def _lon_difference(lon, center_lon):
        """Кратчайшая разница долгот в диапазоне [-180, 180]."""
        return (lon - center_lon + 180.0) % 360.0 - 180.0

    def _make_local_window(self):
        """
        Сначала ограничиваем поиск квадратом radius_deg.

        Это дешёвая операция и обычно оставляет очень маленькую часть ERA5
        сетки. После неё _make_radius_mask уточняет расстояние.
        """
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
        """
        Маска внутри уже маленького локального окна.

        Здесь используется расстояние в градусах на регулярной сетке:
        lat/lon считаются декартовыми координатами. Для небольших радиусов
        в несколько ячеек это намного дешевле Haversine и соответствует
        отбору ближайших ячеек ERA5.
        """
        lat_local = self.lat[self._lat_indices][:, None]
        lon_local = self.lon[self._lon_indices][None, :]

        dlat = lat_local - self.center_lat
        dlon = self._lon_difference(lon_local, self.center_lon)

        # Учитываем сжатие долготы с широтой.
        dlon_metric = dlon * np.cos(np.deg2rad(self.center_lat))
        distance_deg = np.sqrt(dlat**2 + dlon_metric**2)

        return distance_deg <= self.radius_deg

    def _values_in_radius(self, data) -> np.ndarray:
        data_np = safe_values(data)
        if data_np is None:
            return np.empty(0, dtype=float)

        # Данные предполагаются (lat, lon).
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

        values = self._values_in_radius(data)
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
    def __init__(self, data_paths: dict, radius_multiplier: float = 4.0):
        self.data_paths = data_paths
        self.radius_multiplier = radius_multiplier
        self.loader = ERA5DataLoader(data_paths)

    @staticmethod
    def _get_lat_lon(data_dict: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Получить координаты из первого доступного ERA5 DataArray."""
        for data in data_dict.values():
            if data is not None:
                lat_name = "latitude" if "latitude" in data.coords else "lat"
                lon_name = "longitude" if "longitude" in data.coords else "lon"
                return data[lat_name].values, data[lon_name].values

        raise ValueError("Не найдено ни одного ERA5 поля с координатами")

    def _process_point(
        self,
        row: pd.Series,
        era5: Dict[str, Optional[xr.DataArray]],
    ) -> Dict[str, Any]:
        current_time = row["time"]
        center_lat = float(row["lat"])
        center_lon = float(row["lon"])

        # Сохраняем исходную логику пользователя:
        # rad из CSV -> радиус в градусах через умножение на 4.
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
        u10 = era5["u10"]
        v10 = era5["v10"]
        t2 = era5["t2"]
        precip = era5["precip"]
        blh = era5["blh"]
        tropopause = era5["tropopause"]

        t850 = era5["t850"]
        u850 = era5["u850"]
        v850 = era5["v850"]
        r850 = era5["r850"]
        pv850 = era5["pv850"]
        w850 = era5["w850"]
        u500 = era5["u500"]
        v500 = era5["v500"]

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
        if u10 is not None and v10 is not None:
            u10_np = safe_values(u10)
            v10_np = safe_values(v10)
            wind_speed = np.hypot(u10_np, v10_np)
            results["U10_mean"] = aggregator.aggregate(
                wind_speed, AggregationMethod.MEDIAN
            )

        # ------------------------------------------------------------------
        # 850/500 hPa wind
        # ------------------------------------------------------------------
        if u850 is not None and v850 is not None:
            u850_np = safe_values(u850)
            v850_np = safe_values(v850)
            wind_speed_850 = np.hypot(u850_np, v850_np)
            results["U850_mean"] = aggregator.aggregate(
                wind_speed_850, AggregationMethod.MEDIAN
            )

            if u500 is not None and v500 is not None:
                u500_np = safe_values(u500)
                v500_np = safe_values(v500)
                wind_speed_500 = np.hypot(u500_np, v500_np)

                u500_mean = aggregator.aggregate(
                    wind_speed_500, AggregationMethod.MEDIAN
                )

                if np.isfinite(results["U850_mean"]) and results["U850_mean"] > 0:
                    results["U500_U850_frac"] = (
                        u500_mean / results["U850_mean"]
                    )

                du = u500_np - u850_np
                dv = v500_np - v850_np
                _, _, _, mean_dV = aggregator.aggregate_vector(du, dv)
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
            results["T850_disp"] = aggregator.aggregate(
                t850, AggregationMethod.STD
            )

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
            results["w_850"] = aggregator.aggregate(
                w850, AggregationMethod.PERCENTILE_95
            )

        if r850 is not None:
            results["RH_850"] = aggregator.aggregate(
                r850, AggregationMethod.MEDIAN
            )

        if precip is not None:
            results["RAIN_HOURLY_sum"] = aggregator.aggregate(
                precip, AggregationMethod.SUM
            )

        return results

    def process_all_tracks(self, track_files, output_dir: Path):
        """
        Главный оптимизированный цикл.

        Ключевой принцип: сначала собираем все точки всех треков и группируем
        их по времени. ERA5 загружается один раз на timestamp, после чего этот
        набор данных используется всеми треками этого часа.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # ---------------------------------------------------------------
        # 1. Загружаем треки и добавляем служебный _track_id.
        # ---------------------------------------------------------------
        tracks = {}
        points = []

        logger.info("Чтение %d файлов треков", len(track_files))

        for track_id, track_file in enumerate(track_files):
            try:
                df = pd.read_csv(track_file)

                if "time" in df.columns:
                    df["time"] = pd.to_datetime(df["time"])
                elif "datetime" in df.columns:
                    df["time"] = pd.to_datetime(df["datetime"])
                else:
                    logger.error("Нет time/datetime: %s", track_file)
                    continue

                if not {"lat", "lon", "rad"}.issubset(df.columns):
                    logger.error(
                        "В %s отсутствует одна из колонок lat/lon/rad", track_file
                    )
                    continue

                for param in RESULT_PARAMS:
                    df[param] = np.nan

                tracks[track_id] = {
                    "file": track_file,
                    "df": df,
                }

                # Сохраняем только необходимые данные для группировки.
                for row_idx, row in df[["time", "lat", "lon", "rad"]].iterrows():
                    points.append(
                        (
                            pd.Timestamp(row["time"]),
                            track_id,
                            row_idx,
                        )
                    )

            except Exception as exc:
                logger.error("Ошибка чтения %s: %s", track_file, exc)

        if not points:
            logger.warning("Нет точек для обработки")
            return

        # ---------------------------------------------------------------
        # 2. Группировка точек по времени.
        # ---------------------------------------------------------------
        points_df = pd.DataFrame(points, columns=["time", "track_id", "row_idx"])
        points_df = points_df.sort_values("time")

        grouped = points_df.groupby("time", sort=False)

        logger.info(
            "Всего точек: %d; уникальных timestamp: %d",
            len(points_df),
            points_df["time"].nunique(),
        )

        # ---------------------------------------------------------------
        # 3. ERA5 загружается один раз на timestamp.
        # ---------------------------------------------------------------
        for current_time, group in tqdm(
            grouped,
            total=points_df["time"].nunique(),
            desc="ERA5 timestamps",
        ):
            try:
                era5 = self.loader.load(current_time)

                if not any(value is not None for value in era5.values()):
                    continue

                for point in group.itertuples(index=False):
                    track_id = point.track_id
                    row_idx = point.row_idx
                    df = tracks[track_id]["df"]

                    results = self._process_point(df.loc[row_idx], era5)

                    for param, value in results.items():
                        df.at[row_idx, param] = value

            except Exception as exc:
                logger.exception(
                    "Ошибка обработки timestamp %s: %s", current_time, exc
                )

        # ---------------------------------------------------------------
        # 4. Сохраняем готовые треки.
        # ---------------------------------------------------------------
        for track_info in tqdm(tracks.values(), desc="Сохранение треков"):
            output_file = output_dir / Path(track_info["file"]).name
            track_info["df"].to_csv(output_file, index=False)
            logger.info("Обработан трек: %s", output_file)


# ============================================================================
# Точка входа
# ============================================================================


if __name__ == "__main__":
    data_paths = get_data_paths()

    year = 2010

    path_init = "/storage/thalassa/users/vkoshkina"
    path_dir_data = f"{path_init}/data"

    size_filter = 25
    extr_type = '_global'
    postfix = f'_range_1_5_18h_12h_2010_{size_filter}points{extr_type}'


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

        processor = CycloneProcessorERA5(
            data_paths,
            radius_multiplier=4.0,
        )

        processor.process_all_tracks(track_files, output_dir)

        print("Обработка завершена!")
