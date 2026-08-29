"""
Модуль для обработки данных ERA5 и расчета параметров циклонов
"""

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import logging
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time
from glob import glob
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== Константы и перечисления ====================

class AggregationMethod(Enum):
    """Методы агрегации данных"""
    MEAN = 'mean'
    MEDIAN = 'median'
    MIN = 'min'
    MAX = 'max'
    SUM = 'sum'
    STD = 'std'
    PERCENTILE_95 = 95
    PERCENTILE_5 = 5
    POINT = 'point'
    DELTA = 'delta'

@dataclass
class ERA5Config:
    """Конфигурация ERA5"""
    radius_degrees: float = 1.0

# ==================== Утилиты ====================

def safe_values(data):
    """Безопасное извлечение numpy массива из DataArray"""
    if data is None:
        return None
    if hasattr(data, 'values'):
        return data.values
    if hasattr(data, 'magnitude'):
        return data.magnitude
    return np.array(data)

# ==================== Структура data_paths ====================

def get_data_paths():
    """Возвращает структуру путей к данным ERA5"""
    return {
        # Приземные параметры (месячные файлы)
        'mslp': {
            'path': '/storage/thalassa/DATA/ERA5/mslp',
            'pattern': 'era5_mslp_{year}-{month:02d}.nc',
            'var': 'msl',
            'type': 'surface'
        },
        'uv10m': {
            'path': '/storage/thalassa/DATA/ERA5/uv10m',
            'pattern': 'era5_uv10m_{year}-{month:02d}.nc',
            'var': ['u10', 'v10'],
            'type': 'surface'
        },
        't2': {
            'path': '/storage/thalassa/DATA/ERA5/t2',
            'pattern': 'era5_t2_{year}-{month:02d}.nc',
            'var': 't2m',
            'type': 'surface'
        },
        'precip': {
            'path': '/storage/thalassa/DATA/ERA5/precip',
            'pattern': 'ERA5_precip_{year}-{month:02d}.nc',
            'var': 'tp',
            'type': 'surface'
        },
        'boundary_layer': {
            'path': '/storage/thalassa/DATA/ERA5/boundary_layer_height',
            'pattern': 'era5_boundary_layer_height_{year}-{month:02d}.nc',
            'var': 'blh',
            'type': 'surface'
        },
        
        # Уровни давления (GRIB файлы - посуточные)
        'pl_grib': {
            'path': '/storage/thalassa/DATA/ERA5/PL/grib',
            'pattern': '{year}/era5_pl_{year}-{month:02d}-{day:02d}.grib',
            'var': ['t', 'u', 'v', 'q', 'z', 'r'],
            'type': 'pressure',
            'levels': [850, 500]
        },
        
        # PV и omega (месячные файлы)
        'pl_pv_omega': {
            'path': '/storage/thalassa/DATA/ERA5/PL/NC/pv-omega_850and500hPa',
            'pattern': '{year}/era5_pl_pv-omega_{year}-{month:02d}.nc',
            'var': ['pv', 'w'],
            'type': 'pressure',
            'levels': [850, 500]
        },
        
        # Тропопауза (посутoчные файлы)
        'tropopause': {
            'path': '/storage/thalassa/users/vkoshkina/data/ERA5/tropopause',
            'pattern': '{year}/era5_{year}_{month:02d}_{day:02d}.nc',
            'var': ['clp_z', 'dyn_z', 'wmo_1st_z'],
            'type': 'tropopause'
        }
    }

# ==================== Функции загрузки данных ====================

def get_file_path(data_config: dict, time: pd.Timestamp) -> str:
    """Получить путь к файлу для заданного времени"""
    pattern = data_config['pattern']
    file_path = pattern.format(
        year=time.year,
        month=time.month,
        day=time.day
    )
    return f"{data_config['path']}/{file_path}"

def load_era5_data(data_paths: dict, param_type: str, time: pd.Timestamp, 
                   var_name: str = None, level: int = None) -> Optional[xr.DataArray]:
    """Загрузка данных ERA5"""
    if param_type not in data_paths:
        logger.warning(f"Параметр {param_type} не найден в data_paths")
        return None
    
    config = data_paths[param_type]
    file_path = get_file_path(config, time)
    
    if not Path(file_path).exists():
        logger.debug(f"Файл не найден: {file_path}")
        return None
    
    try:
        # Загрузка файла
        if file_path.endswith('.grib'):
            ds = xr.open_dataset(file_path, engine='cfgrib')
        else:
            ds = xr.open_dataset(file_path)
        
        # Определяем измерение времени
        time_dim = None
        for dim in ['time', 'valid_time']:
            if dim in ds.dims:
                time_dim = dim
                break
        
        # Выбор времени
        if time_dim:
            ds = ds.sel({time_dim: time}, method='nearest')
        
        # Выбор уровня давления
        if level:
            if 'isobaricInhPa' in ds.dims:
                ds = ds.sel(isobaricInhPa=level, method='nearest')
            elif 'pressure_level' in ds.dims:
                ds = ds.sel(pressure_level=level, method='nearest')
        
        # Выбор переменной
        if var_name:
            if var_name in ds.data_vars:
                data = ds[var_name]
            else:
                # Ищем похожую переменную
                for var in ds.data_vars:
                    if var_name in var or var in var_name:
                        data = ds[var]
                        break
                else:
                    ds.close()
                    return None
        else:
            # Берем первую переменную
            data = ds[list(ds.data_vars.keys())[0]]
        
        return data
        
    except Exception as e:
        logger.debug(f"Ошибка загрузки {param_type} из {file_path}: {e}")
        return None

# ==================== Класс для загрузки данных ====================

class ERA5DataLoader:
    """Класс для загрузки данных ERA5"""
    
    def __init__(self, data_paths: dict):
        self.data_paths = data_paths
        self._cache = {}
    
    def load_mslp(self, time: pd.Timestamp) -> Optional[xr.DataArray]:
        return load_era5_data(self.data_paths, 'mslp', time, var_name='msl')
    
    def load_uv10m(self, time: pd.Timestamp) -> Tuple[Optional[xr.DataArray], Optional[xr.DataArray]]:
        u10 = load_era5_data(self.data_paths, 'uv10m', time, var_name='u10')
        v10 = load_era5_data(self.data_paths, 'uv10m', time, var_name='v10')
        return u10, v10
    
    def load_t2(self, time: pd.Timestamp) -> Optional[xr.DataArray]:
        return load_era5_data(self.data_paths, 't2', time, var_name='t2m')
    
    def load_precip(self, time: pd.Timestamp) -> Optional[xr.DataArray]:
        return load_era5_data(self.data_paths, 'precip', time, var_name='tp')
    
    def load_boundary_layer(self, time: pd.Timestamp) -> Optional[xr.DataArray]:
        return load_era5_data(self.data_paths, 'boundary_layer', time, var_name='blh')
    
    def load_tropopause(self, time: pd.Timestamp) -> Optional[xr.DataArray]:
        # Используем динамическую тропопаузу (dyn_z)
        return load_era5_data(self.data_paths, 'tropopause', time, var_name='dyn_z')
    
    def load_pressure_level(self, time: pd.Timestamp, level: int, var_name: str) -> Optional[xr.DataArray]:
        """Загрузка данных на уровне давления"""
        # Проверяем в GRIB
        if var_name in self.data_paths['pl_grib']['var']:
            return load_era5_data(self.data_paths, 'pl_grib', time, var_name=var_name, level=level)
        
        # Проверяем в PV/omega
        if var_name in self.data_paths['pl_pv_omega']['var']:
            return load_era5_data(self.data_paths, 'pl_pv_omega', time, var_name=var_name, level=level)
        
        return None

# ==================== Пространственный агрегатор ====================

class SpatialAggregatorERA5:
    """Класс для пространственной агрегации данных ERA5 по координатам"""
    
    def __init__(self, center_lat: float, center_lon: float, radius_degrees: float):
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.radius_degrees = radius_degrees
        self._lat_grid = None
        self._lon_grid = None
        
    def _get_radius_mask(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Создание маски радиуса"""
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        center_lat_rad = np.radians(self.center_lat)
        center_lon_rad = np.radians(self.center_lon)
        
        dlat = lat_rad - center_lat_rad
        dlon = lon_rad - center_lon_rad
        
        a = np.sin(dlat/2)**2 + np.cos(center_lat_rad) * np.cos(lat_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance_deg = np.degrees(c)
        
        return distance_deg <= self.radius_degrees
    
    def _get_coordinate_grids(self, data: xr.DataArray):
        """Получение сеток координат"""
        if self._lat_grid is None:
            lat = data.latitude.values
            lon = data.longitude.values
            self._lon_grid, self._lat_grid = np.meshgrid(lon, lat)
        return self._lat_grid, self._lon_grid
    
    def aggregate(self, data: xr.DataArray, method: AggregationMethod) -> float:
        """Агрегация данных в радиусе"""
        if data is None or data.size == 0:
            return np.nan
        
        data_np = safe_values(data)
        lat_grid, lon_grid = self._get_coordinate_grids(data)
        mask = self._get_radius_mask(lat_grid, lon_grid)
        
        values = data_np[mask]
        valid_values = values[~np.isnan(values)]
        
        if len(valid_values) == 0:
            return np.nan
        
        if method == AggregationMethod.MEAN:
            return np.nanmean(valid_values)
        elif method == AggregationMethod.MEDIAN:
            return np.nanmedian(valid_values)
        elif method == AggregationMethod.MIN:
            return np.nanmin(valid_values)
        elif method == AggregationMethod.MAX:
            return np.nanmax(valid_values)
        elif method == AggregationMethod.SUM:
            return np.nansum(valid_values)
        elif method == AggregationMethod.STD:
            return np.nanstd(valid_values, ddof=1)
        elif method == AggregationMethod.POINT:
            lat_idx = np.argmin(np.abs(data.latitude.values - self.center_lat))
            lon_idx = np.argmin(np.abs(data.longitude.values - self.center_lon))
            return data_np[lat_idx, lon_idx]
        elif method == AggregationMethod.DELTA:
            return np.nanpercentile(valid_values, 95) - np.nanpercentile(valid_values, 5)
        elif isinstance(method.value, (int, float)):
            return np.nanpercentile(valid_values, method.value)
        else:
            return np.nanmedian(valid_values)
    
    def aggregate_vector(self, real_part, imag_part) -> Tuple[float, float, float, float]:
        """Векторное усреднение"""
        if real_part is None or imag_part is None:
            return np.nan, np.nan, np.nan, np.nan
        
        real_np = safe_values(real_part) if not isinstance(real_part, np.ndarray) else real_part
        imag_np = safe_values(imag_part) if not isinstance(imag_part, np.ndarray) else imag_part
        
        # Получаем маску от real_part
        lat_grid, lon_grid = self._get_coordinate_grids(real_part)
        mask = self._get_radius_mask(lat_grid, lon_grid)
        
        valid_mask = mask & ~np.isnan(real_np) & ~np.isnan(imag_np)
        
        if not np.any(valid_mask):
            return np.nan, np.nan, np.nan, np.nan
        
        mean_real = np.nanmean(real_np[mask])
        mean_imag = np.nanmean(imag_np[mask])
        mean_angle = np.degrees(np.arctan2(mean_imag, mean_real)) % 360
        mean_magnitude = np.sqrt(mean_real**2 + mean_imag**2)
        
        return mean_real, mean_imag, mean_angle, mean_magnitude

# ==================== Основной класс обработчика ====================

class CycloneProcessorERA5:
    """Основной класс для обработки циклонов с данными ERA5"""
    
    def __init__(self, data_paths: dict, radius_degrees: float = 1.0):
        self.data_paths = data_paths
        self.radius_degrees = radius_degrees
        self.loader = ERA5DataLoader(data_paths)
    
    def process_track(self, track_file: str, output_dir: Path) -> Optional[pd.DataFrame]:
        """Обработка одного трека"""
        try:
            df = pd.read_csv(track_file)
            
            # Определяем колонку с временем
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
            elif 'datetime' in df.columns:
                df['time'] = pd.to_datetime(df['datetime'])
            else:
                logger.error("В треке нет колонки time или datetime")
                return None
            
            # Определяем колонки с координатами
            if 'latitude' in df.columns and 'longitude' in df.columns:
                pass
                # df['latitude'] = df['lat']
                # df['longitude'] = df['lon']
            elif 'lat' in df.columns and 'lon' in df.columns:
                # df['latitude'] = df['lat']
                # df['longitude'] = df['lon']
                pass
            else:
                logger.error("В треке нет координат")
                return None
            
            # Добавляем колонки для результатов
            result_params = [
                'SLP_diff_cent_95', 'U10_mean', 'U850_mean', 
                'U500_U850_frac', 'U500_minus_U850', 'PV_850_mean',
                'T2_minus_T850_mean', 'T850_disp', 'pbl_trop_frac',
                'w_850', 'RH_850', 'RAIN_HOURLY_sum'
            ]
            for param in result_params:
                df[param] = np.nan
            
            # Обработка каждого часа
            for i in tqdm(range(len(df)), desc='single track'):
                results = self._process_hour(df.iloc[i])
                for param, value in results.items():
                    if param in df.columns:
                        df.at[i, param] = value
            
            # Сохранение
            output_file = output_dir / Path(track_file).name
            df.to_csv(output_file, index=False)
            logger.info(f"Обработан трек: {output_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"Ошибка обработки трека {track_file}: {e}")
            return None
    
    def _process_hour(self, row: pd.Series) -> Dict[str, Any]:
        """Обработка одного часа"""
        current_time = row['time']
        center_lat = float(row['lat'])
        center_lon = float(row['lat'])
        rad = float(row['rad'])*4 # in degrees (need to check) 2026-08-17
        
        
        aggregator = SpatialAggregatorERA5(center_lat, center_lon, rad)
        results = {}
        
        # 1. Загружаем все данные
        mslp = self.loader.load_mslp(current_time)
        u10, v10 = self.loader.load_uv10m(current_time)
        t2 = self.loader.load_t2(current_time)
        precip = self.loader.load_precip(current_time)
        blh = self.loader.load_boundary_layer(current_time)
        tropopause = self.loader.load_tropopause(current_time)
        
        # Данные на уровнях давления
        t850 = self.loader.load_pressure_level(current_time, 850, 't')
        u850 = self.loader.load_pressure_level(current_time, 850, 'u')
        v850 = self.loader.load_pressure_level(current_time, 850, 'v')
        r850 = self.loader.load_pressure_level(current_time, 850, 'r')
        pv850 = self.loader.load_pressure_level(current_time, 850, 'pv')
        w850 = self.loader.load_pressure_level(current_time, 850, 'w')
        
        u500 = self.loader.load_pressure_level(current_time, 500, 'u')
        v500 = self.loader.load_pressure_level(current_time, 500, 'v')
        
        # 2. Расчет параметров
        # SLP_diff_cent_95
        if mslp is not None:
            slp_center = aggregator.aggregate(mslp, AggregationMethod.POINT)
            slp_95 = aggregator.aggregate(mslp, AggregationMethod.PERCENTILE_95)
            results['SLP_diff_cent_95'] = 0.01*(slp_95 - slp_center)
        
        # U10_mean
        if u10 is not None and v10 is not None:
            wind_speed = np.sqrt(safe_values(u10)**2 + safe_values(v10)**2)
            results['U10_mean'] = aggregator.aggregate(wind_speed, AggregationMethod.MEDIAN)
        
        # U850_mean, U500_U850_frac, U500_minus_U850
        if u850 is not None and v850 is not None:
            wind_speed_850 = np.sqrt(safe_values(u850)**2 + safe_values(v850)**2)
            results['U850_mean'] = aggregator.aggregate(wind_speed_850, AggregationMethod.MEDIAN)
            
            if u500 is not None and v500 is not None:
                wind_speed_500 = np.sqrt(safe_values(u500)**2 + safe_values(v500)**2)
                u500_mean = aggregator.aggregate(wind_speed_500, AggregationMethod.MEDIAN)
                
                if results['U850_mean'] > 0:
                    results['U500_U850_frac'] = u500_mean / results['U850_mean']
                
                du = safe_values(u500) - safe_values(u850)
                dv = safe_values(v500) - safe_values(v850)
                _, _, _, mean_dV = aggregator.aggregate_vector(du, dv)
                results['U500_minus_U850'] = mean_dV
        
        # PV_850_mean
        if pv850 is not None:
            results['PV_850_mean'] = aggregator.aggregate(pv850*1000000, AggregationMethod.MEDIAN)
        
        # T2_minus_T850_mean, T850_disp
        if t2 is not None and t850 is not None:
            t2_med = aggregator.aggregate(t2, AggregationMethod.MEDIAN)
            t850_med = aggregator.aggregate(t850, AggregationMethod.MEDIAN)
            results['T2_minus_T850_mean'] = t2_med - t850_med
            results['T850_disp'] = aggregator.aggregate(t850, AggregationMethod.STD)
        
        # pbl_trop_frac
        if blh is not None and tropopause is not None:
            blh_val = aggregator.aggregate(blh, AggregationMethod.MEDIAN)
            trop_val = aggregator.aggregate(tropopause, AggregationMethod.MEDIAN)
            results['pbl_trop_frac'] = blh_val * 1000 / trop_val if trop_val > 0 else np.nan
        
        # w_850
        if w850 is not None:
            results['w_850'] = aggregator.aggregate(w850, AggregationMethod.PERCENTILE_95)
        
        # RH_850
        if r850 is not None:
            results['RH_850'] = aggregator.aggregate(r850, AggregationMethod.MEDIAN)
        
        # RAIN_HOURLY_sum
        if precip is not None:
            results['RAIN_HOURLY_sum'] = aggregator.aggregate(precip, AggregationMethod.SUM)
        
        return results

# ==================== Точка входа ====================

if __name__ == "__main__":
    
    # 1. Получаем структуру путей
    data_paths = get_data_paths()

    year = 2010
    
    # 2. Настройка путей к трекам

    path_init = f'/storage/thalassa/users/vkoshkina'
    path_dir_data = f'{path_init}/data'
    TRACKS_PATH = f"{path_dir_data}/TempestExtremes/ERA5/R2D_ERA5_NA_for_TC_850hPa_sigma_2/csv_Tracks_range_1_5_18h_12h" 
    OUTPUT_PATH = Path(f"{TRACKS_PATH}_params")

    
    # 3. Поиск файлов треков
    track_files = glob(f"{TRACKS_PATH}/*_track_{year}*.csv")
    
    if not track_files:
        print(f"Файлы треков не найдены в {TRACKS_PATH}")
        print("Пожалуйста, укажите правильный путь к трекам")
    else:
        # 4. Создаем выходную директорию
        output_dir = OUTPUT_PATH / "hourly_data"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 5. Создаем процессор
        processor = CycloneProcessorERA5(data_paths, radius_degrees=1.0)
        
        # 6. Обрабатываем все треки
        print(f"Найдено {len(track_files)} треков")
        print(f"Выходная директория: {output_dir}")
        
        for track_file in tqdm(track_files, desc="Обработка треков"):
            processor.process_track(track_file, output_dir)
        
        print("Обработка завершена!")