"""
Модуль для обработки данных WRF и расчета параметров циклонов
"""

import pandas as pd
import numpy as np
import xarray as xr
from netCDF4 import Dataset
from wrf import getvar, interplevel
from metpy.calc import potential_temperature, equivalent_potential_temperature
from metpy.calc import gradient, brunt_vaisala_frequency
from metpy.units import units
from multiprocessing import Pool, cpu_count
from pathlib import Path
import logging
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time
from glob import glob
import re
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def safe_values(data):
    """Безопасное извлечение numpy массива из DataArray"""
    if data is None:
        return None
    if hasattr(data, 'values'):
        return data.values
    if hasattr(data, 'magnitude'):
        return data.magnitude
    return np.array(data)
    
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

class WRFDataLevel(Enum):
    """Уровни давления WRF"""
    SURFACE = 'surface'
    LEVEL_925 = 925
    LEVEL_850 = 850
    LEVEL_700 = 700
    LEVEL_500 = 500
    LEVEL_300 = 300
    LEVEL_200 = 200

@dataclass
class WRFConfig:
    """Конфигурация WRF"""
    bottom_pressure: int = 1000  # гПа
    top_pressure: int = 300      # гПа
    min_tropopause_height: int = 3000  # м
    pvu_threshold: float = 2.0
    
@dataclass
class CalculationConfig:
    """Конфигурация расчетов"""
    radius_pixels: float = 10.0
    n_levels_for_ivt: int = 100
    tau_hours: List[int] = field(default_factory=lambda: [2, 4, 6, 12, 24])

# ==================== Базовые утилиты ====================

def parse_wrf_time(time_bytes: bytes) -> pd.Timestamp:
    """Парсинг времени из WRF файла"""
    time_str = time_bytes.tobytes().decode('utf-8').strip().replace('_', ' ')
    
    # ========== УЛУЧШЕННЫЙ ПАРСИНГ ==========
    # Пробуем разные форматы
    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M', 
        '%Y-%m-%d %H'
    ]
    
    for fmt in formats:
        try:
            return pd.to_datetime(time_str, format=fmt)
        except ValueError:
            continue
    
    # Если ничего не подошло, используем общий парсинг
    return pd.to_datetime(time_str)
    # =======================================

def find_wrf_file_for_time(target_time: pd.Timestamp, wrf_files: List[str]) -> Optional[str]:
    """Поиск WRF файла для заданного времени"""
    target_datetime = pd.to_datetime(target_time)

    
    for wrf_file in wrf_files:
        if "2019-01-01_00:00:00" in wrf_file:
            continue
        
        filename = Path(wrf_file).name
        
        if filename.endswith('.nc'):
            filename = filename[:-3]
        
        parts = filename.split('_')
        
        if len(parts) >= 4:
            date_part = parts[2] + '_' + parts[3]
        else:
            continue
        
        try:
            if ':' in date_part and date_part.count(':') == 2:
                file_start_time = pd.to_datetime(date_part, format='%Y-%m-%d_%H:%M:%S')
            else:
                file_start_time = pd.to_datetime(date_part, format='%Y-%m-%d_%H')
            
            # ========== ИЗМЕНЕНИЕ: ДИАПАЗОН ДО 24 ЧАСОВ ==========
            # Для суточных файлов проверяем, что время попадает в диапазон
            file_end_time = file_start_time + pd.Timedelta(hours=24)
            
            # Добавляем небольшую погрешность в 1 секунду
            if file_start_time <= target_datetime < file_end_time:
                return wrf_file
            # =====================================================
            
        except Exception as e:
            logger.warning(f"Ошибка парсинга даты из {filename}: {e}")
            continue
    
    logger.warning(f"Не найден WRF файл для времени {target_datetime}")
    return None

# ==================== Классы для работы с данными ====================

class DataExtractor:
    """Класс для извлечения данных из WRF"""
    
    def __init__(self, ds: Dataset, time_idx: int):
        self.ds = ds
        self.time_idx = time_idx
        self._cache = {}
        
    def get(self, var_name: str, **kwargs) -> np.ndarray:
        """Получение переменной с кэшированием"""
        cache_key = f"{var_name}_{self.time_idx}_{kwargs}"
        if cache_key not in self._cache:
            data = getvar(self.ds, var_name, timeidx=self.time_idx, **kwargs)
            self._cache[cache_key] = safe_values(data)  # <-- ДОБАВИТЬ safe_values
        return self._cache[cache_key]
    
    def get_pressure(self) -> np.ndarray:
        """Получение поля давления"""
        return self.get('pressure')
    
    def get_temperature(self) -> np.ndarray:
        """Получение температуры"""
        return self.get('temp')
    
    def get_wind(self) -> Tuple[np.ndarray, np.ndarray]:
        """Получение U и V компонент ветра"""
        u = self.get('ua', units="m s-1")
        v = self.get('va', units="m s-1")
        return u, v
    
    def get_wind_speed(self, level: WRFDataLevel) -> np.ndarray:
        """Получение скорости ветра на заданном уровне"""
        u, v = self.get_wind()
        pressure = self.get_pressure()
        
        u_interp = interplevel(u, pressure, float(level.value), meta=True)
        v_interp = interplevel(v, pressure, float(level.value), meta=True)
        
        u_np = safe_values(u_interp)  # <-- использовать safe_values
        v_np = safe_values(v_interp)  # <-- использовать safe_values
        
        return np.sqrt(u_np**2 + v_np**2)
    
    def get_temperature_at_level(self, level: WRFDataLevel) -> np.ndarray:
        """Получение температуры на заданном уровне"""
        temp = self.get_temperature()
        pressure = self.get_pressure()
        result = interplevel(temp, pressure, float(level.value), meta=True)
        return safe_values(result)  # <-- ДОБАВИТЬ safe_values
    
    def get_theta_at_level(self, level: WRFDataLevel) -> np.ndarray:
        """Получение потенциальной температуры на заданном уровне"""
        theta = self.get('theta')
        pressure = self.get_pressure()
        result = interplevel(theta, pressure, float(level.value), meta=True)
        return safe_values(result)  # <-- ДОБАВИТЬ safe_values
    
    def get_theta_e_at_level(self, level: WRFDataLevel) -> np.ndarray:
        """Получение эквивалентной потенциальной температуры на заданном уровне"""
        theta_e = self.get('theta_e')
        pressure = self.get_pressure()
        result = interplevel(theta_e, pressure, float(level.value), meta=True)
        return safe_values(result)  # <-- ДОБАВИТЬ safe_values
    
    def get_interpolated_at_level(self, var: np.ndarray, level: WRFDataLevel) -> np.ndarray:
        """Общая интерполяция переменной на уровень"""
        pressure = self.get_pressure()
        result = interplevel(var, pressure, float(level.value), meta=True)
        return safe_values(result)  # <-- ДОБАВИТЬ safe_values

class SpatialAggregator:
    """Класс для пространственной агрегации данных"""
    
    def __init__(self, center_lat_idx: int, center_lon_idx: int, radius_pixels: float):
        self.center_lat_idx = center_lat_idx
        self.center_lon_idx = center_lon_idx
        self.radius_pixels = radius_pixels
        self._mask = None
        
    def _get_radius_mask(self, data_shape: Tuple[int, int]) -> np.ndarray:
        """Создание маски радиуса"""
        if self._mask is None or self._mask.shape != data_shape:
            ny, nx = data_shape
            y_coords, x_coords = np.ogrid[:ny, :nx]
            distances = np.sqrt((y_coords - self.center_lat_idx)**2 + 
                              (x_coords - self.center_lon_idx)**2)
            self._mask = distances <= self.radius_pixels
        return self._mask
    
    def aggregate(self, data: np.ndarray, method: AggregationMethod) -> float:
        """Агрегация данных в радиусе"""
        # ========== ВАЖНО: КОНВЕРТИРУЕМ ВХОДНЫЕ ДАННЫЕ ==========
        data = safe_values(data)  # <-- ДОБАВИТЬ ЭТУ СТРОКУ!
        # =======================================================
        
        if data is None or data.size == 0:
            return np.nan
            
        mask = self._get_radius_mask(data.shape)
        values = data[mask]
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
            return data[self.center_lat_idx, self.center_lon_idx]
        elif method == AggregationMethod.DELTA:
            return np.nanpercentile(valid_values, 95) - np.nanpercentile(valid_values, 5)
        elif isinstance(method.value, (int, float)):
            return np.nanpercentile(valid_values, method.value)
        else:
            return np.nanmedian(valid_values)
    
    def aggregate_vector(self, real_part: np.ndarray, imag_part: np.ndarray) -> Tuple[float, float, float, float]:
        """Векторное усреднение"""
        # ========== КОНВЕРТИРУЕМ ВХОДНЫЕ ДАННЫЕ ==========
        real_part = safe_values(real_part)
        imag_part = safe_values(imag_part)
        # ================================================
        
        mask = self._get_radius_mask(real_part.shape)
        valid_mask = mask & ~np.isnan(real_part) & ~np.isnan(imag_part)
        
        if not np.any(valid_mask):
            return np.nan, np.nan, np.nan, np.nan
        
        mean_real = np.nanmean(real_part[mask])
        mean_imag = np.nanmean(imag_part[mask])
        mean_angle = np.degrees(np.arctan2(mean_imag, mean_real)) % 360
        mean_magnitude = np.sqrt(mean_real**2 + mean_imag**2)
        
        return mean_real, mean_imag, mean_angle, mean_magnitude

# ==================== Расчетные классы ====================

class ThermodynamicCalculator:
    """Расчет термодинамических параметров"""
    
    def __init__(self, ds, time_idx: int, config: WRFConfig):
        self.ds = ds
        self.time_idx = time_idx
        self.config = config
        self.extractor = DataExtractor(ds, time_idx)
    
    def calculate_mcao_kolstad(self, theta_sst: np.ndarray, 
                               theta_500: np.ndarray, 
                               theta_700: np.ndarray, 
                               slp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Расчет MCAO по Kolstad"""
        with np.errstate(divide='ignore', invalid='ignore'):
            mcao_500 = (theta_sst - theta_500) / (slp*100 - 50000)
            mcao_700 = (theta_sst - theta_700) / (slp*100 - 70000)
        return np.nan_to_num(mcao_500), np.nan_to_num(mcao_700)
    
    def calculate_mcao_bracegirdle(self, theta_sst: np.ndarray, 
                                   theta_700: np.ndarray, 
                                   z_700: np.ndarray) -> np.ndarray:
        """Расчет MCAO по Bracegirdle"""
        L = 7.5e5
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.nan_to_num((L / z_700) * (np.log(theta_sst) - np.log(theta_700)))
    
    def calculate_george_index(self, t_850: np.ndarray, t_700: np.ndarray, 
                               t_500: np.ndarray, td_850: np.ndarray, 
                               td_700: np.ndarray) -> np.ndarray:
        """Расчет индекса Джорджа"""
        return (t_850 - t_500) + td_850 - (t_700 - td_700)
    
    def calculate_lifted_index(self, t2: np.ndarray, t_500: np.ndarray, 
                              pressure: np.ndarray, dewpoint: np.ndarray,
                              center_lat_idx: int, center_lon_idx: int, 
                              radius_pixels: float) -> float:
        """Расчет индекса поднятия"""
        li_field = np.full_like(t_500, np.nan)
        ny, nx = t2.shape
        
        # Создаем маску радиуса
        y_coords, x_coords = np.ogrid[:ny, :nx]
        distances = np.sqrt((y_coords - center_lat_idx)**2 + 
                          (x_coords - center_lon_idx)**2)
        radius_mask = distances <= radius_pixels
        
        for lat_idx, lon_idx in zip(*np.where(radius_mask)):
            p_profile = pressure[:, lat_idx, lon_idx]
            if hasattr(p_profile, 'values'):
                p_profile = p_profile.values
            
            if np.any(np.isnan(p_profile)):
                continue
            
            t2_sfc = float(t2[lat_idx, lon_idx].values) - 273.15
            td_sfc = float(dewpoint[lat_idx, lon_idx].values)
            t500_env = float(t_500[lat_idx, lon_idx].values)
            
            # Подъем частицы
            prof = self._parcel_profile(p_profile, t2_sfc, td_sfc)
            idx_500 = np.argmin(np.abs(p_profile - 500))
            t500_parcel = prof[idx_500].to('K').magnitude
            
            li_value = t500_env - t500_parcel
            li_field[lat_idx, lon_idx] = li_value
        
        valid_mask = radius_mask & ~np.isnan(li_field)
        if np.any(valid_mask):
            return np.nanpercentile(li_field[valid_mask], 95)
        return np.nan
    
    def _parcel_profile(self, pressure: np.ndarray, t_sfc: float, td_sfc: float):
        """Профиль поднимающейся частицы"""
        from metpy.calc import parcel_profile
        return parcel_profile(pressure * units('hPa'), 
                            t_sfc * units.degC, 
                            td_sfc * units.degC)

class IVTCalculator:
    """Расчет интегрального переноса влаги"""
    
    def __init__(self, ds, time_idx: int, config: WRFConfig):
        self.ds = ds
        self.time_idx = time_idx
        self.config = config
        self.g = 9.81  # м/с²
        
    def calculate(self, bottom_pressure: int = 1000, top_pressure: int = 300) -> Dict[str, np.ndarray]:
        """Расчет IVT и компонент"""
        q = getvar(self.ds, 'QVAPOR', timeidx=self.time_idx)
        u = getvar(self.ds, 'ua', timeidx=self.time_idx)
        v = getvar(self.ds, 'va', timeidx=self.time_idx)
        pressure = getvar(self.ds, 'pressure', timeidx=self.time_idx)
        
        # КОНВЕРТИРУЕМ
        q = safe_values(q)
        u = safe_values(u)
        v = safe_values(v)
        pressure = safe_values(pressure)
        
        # Уровни давления для интегрирования
        p_levels = np.linspace(bottom_pressure, top_pressure, 100)
        
        # Интерполяция
        qv_interp = interplevel(q, pressure, p_levels)
        u_interp = interplevel(u, pressure, p_levels)
        v_interp = interplevel(v, pressure, p_levels)
        
        # Компоненты переноса
        qu = qv_interp * u_interp
        qv = qv_interp * v_interp
        
        # Интегрирование
        dp = np.diff(p_levels * 100.0)
        dp = np.append(dp, dp[-1]).reshape(100, 1, 1)
        
        ivt_u = np.sum(qu * dp / self.g, axis=0)
        ivt_v = np.sum(qv * dp / self.g, axis=0)
        ivt = np.sqrt(ivt_u**2 + ivt_v**2)
        
        return {
            'integral': ivt,
            'u_dir': ivt_u,
            'v_dir': ivt_v
        }

class TropopauseCalculator:
    """Расчет высоты тропопаузы"""
    
    def __init__(self, config: WRFConfig):
        self.config = config
    
    def calculate(self, z: np.ndarray, potential_vorticity: np.ndarray) -> np.ndarray:
        """Расчет высоты тропопаузы по поверхности 2 PVU"""
        # КОНВЕРТИРУЕМ ВХОДНЫЕ ДАННЫЕ
        z = safe_values(z)
        potential_vorticity = safe_values(potential_vorticity)
        
        ny, nx = potential_vorticity.shape[1], potential_vorticity.shape[2]
        trop_height = np.full((ny, nx), np.nan)
        
        for i in range(ny):
            for j in range(nx):
                pv_profile = potential_vorticity[:, i, j]  # уже numpy
                z_profile = z[:, i, j]  # уже numpy
                sort_idx = np.argsort(z_profile)
                
                pv_sorted = pv_profile[sort_idx]
                z_sorted = z_profile[sort_idx]
                high_levels = z_sorted >= self.config.min_tropopause_height
                
                if not np.any(high_levels):
                    continue
                
                pv_high = pv_sorted[high_levels]
                z_high = z_sorted[high_levels]
                idx = np.where(pv_high >= self.config.pvu_threshold)[0]
                
                if len(idx) > 0:
                    first_idx = idx[0]
                    if first_idx > 0:
                        pv1, pv2 = pv_high[first_idx - 1], pv_high[first_idx]
                        z1, z2 = z_high[first_idx - 1], z_high[first_idx]
                        if pv2 - pv1 != 0:
                            fraction = (self.config.pvu_threshold - pv1) / (pv2 - pv1)
                            trop_height[i, j] = z1 + fraction * (z2 - z1)
                        else:
                            trop_height[i, j] = z1
                    else:
                        trop_height[i, j] = z_high[first_idx]
        
        return trop_height

class BergeronsIndexCalculator:
    """Расчет индекса Бержерона"""
    
    @staticmethod
    def calculate(df: pd.DataFrame, tau_hours: int = 12) -> List[float]:
        """Расчет индекса Бержерона для часовых данных"""
        bergeron_values = [np.nan] * len(df)
        sin45 = np.sin(np.radians(45))
        
        df_sorted = df.sort_values('time').reset_index(drop=True)
        times = pd.Series(df_sorted['time'].values)
        lats = df_sorted['latitude'].values
        slp = df_sorted['SLP_center'].values
        
        half_tau = tau_hours // 2
        
        for i in range(len(df_sorted)):
            if i - half_tau < 0 or i + half_tau >= len(df_sorted):
                continue
            
            time_minus = times.iloc[i - half_tau]
            time_plus = times.iloc[i + half_tau]
            time_current = times.iloc[i]
            
            diff_minus = (time_current - time_minus).total_seconds() / 3600
            diff_plus = (time_plus - time_current).total_seconds() / 3600
            
            if abs(diff_minus - half_tau) < 0.1 and abs(diff_plus - half_tau) < 0.1:
                mean_lat = (lats[i - half_tau] + lats[i + half_tau]) / 2
                sin_lat = np.sin(np.radians(mean_lat))
                lat_factor = sin45 / sin_lat
                delta_p = slp[i - half_tau] - slp[i + half_tau]
                bergeron_values[i] = (delta_p / 12) * lat_factor
        
        return bergeron_values

# ==================== Основной класс обработчика ====================

class CycloneProcessor:
    """Основной класс для обработки циклонов"""
    
    def __init__(self, wrf_files: List[str], config: WRFConfig = None, calc_config: CalculationConfig = None):
        self.wrf_files = wrf_files
        self.config = config or WRFConfig()
        self.calc_config = calc_config or CalculationConfig()
        
    def process_track(self, track_file: str, output_dir: Path) -> Optional[pd.DataFrame]:
        """Обработка одного трека"""
        try:
            df = pd.read_csv(track_file)

            if EC_tracks:
                #### 2026-08-10 - for EC tracks
                df['time'] = pd.to_datetime(df['time'])
            else:
                #### 2026-08-10 - for auto tracks
                df['time'] = pd.to_datetime(df['datetime'])
    
                df['pyc_ind'] = df['x']
                df['pxc_ind'] = df['y']
                df['mean_radius'] = df['rad']
            
            
            
            # Добавление колонок для результатов
            result_params = self._get_result_params()
            for param in result_params:
                df[param] = np.nan
            
            # Обработка каждого часа
            for i in range(len(df)):
                results = self._process_hour(i, df.iloc[i], df)
                for param, value in results.items():
                    if param in df.columns:
                        df.at[i, param] = value
            
            # # Расчет индексов Бержерона
            # bergeron_results = self._calculate_bergeron_indexes(df)
            # for param, values in bergeron_results.items():
            #     df[param] = values
            
            # Сохранение
            output_file = output_dir / Path(track_file).name
            df.to_csv(output_file, index=False)
            logger.info(f"Обработан трек: {output_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"Ошибка обработки трека {track_file}: {e}")
            return None
    
    def _process_hour(self, idx: int, row: pd.Series, storm_data: pd.DataFrame) -> Dict[str, Any]:
        """Обработка одного часа"""
        current_time = row['time']
        
        # ========== ДОБАВИТЬ ЭТУ ПРОВЕРКУ ==========
        # Исключаем дату 2019-01-01
        if current_time.date() == pd.Timestamp('2019-01-01').date():
            logger.debug(f"Пропускаем время {current_time} (исключенная дата)")
            return {}
        # ===========================================
        
        radius_pixels = float(row['mean_radius'])
        
        wrf_file = find_wrf_file_for_time(current_time, self.wrf_files)
        if not wrf_file:
            return {}
        
        with Dataset(wrf_file) as ds:
            time_idx = self._find_time_index(ds, current_time)
            if time_idx is None:
                return {}
            
            center_lat_idx = int(row['pyc_ind'])
            center_lon_idx = int(row['pxc_ind'])
            
            results = {}
            extractor = DataExtractor(ds, time_idx)
            aggregator = SpatialAggregator(center_lat_idx, center_lon_idx, radius_pixels)
            
            # Расчет всех параметров
            results.update(self._calculate_basic_params(extractor, aggregator))
            results.update(self._calculate_thermodynamic_params(extractor, aggregator))
            results.update(self._calculate_dynamics_params(extractor, aggregator, idx, storm_data))
            results.update(self._calculate_precipitation_params(extractor, aggregator, ds, time_idx, current_time))
            results.update(self._calculate_ivt_params(extractor, aggregator, ds, time_idx))
            
            return results
    
    def _find_time_index(self, ds: Dataset, target_time: pd.Timestamp) -> Optional[int]:
        """Поиск индекса времени в WRF файле"""
        times = ds.variables['Times'][:]
        wrf_times = [parse_wrf_time(t) for t in times]
        
        for idx, wrf_time in enumerate(wrf_times):
            if wrf_time == target_time:
                return idx
        return None
    
    def _get_result_params(self) -> List[str]:
        """Список параметров для расчета"""
        return [
            # 'SLP_center', 'SLP_median', 'SLP_delta', 'SLP_diff_cent_med', 'SLP_95', 
            'SLP_center', 'SLP_95', 'SLP_diff_cent_95',
            'U10_mean', 'U500_mean', 'U850_mean', 'U500_U850_frac', 'U500_minus_U850',
            # 'U500_poleward', 
            'PV_850_mean', 'PV_500_mean', 
            # 'T2_mean', 'T500_mean', 'T700_mean', 'T850_mean', 
            'T2_minus_T500_mean', 'T2_minus_T850_mean', 'T850_disp',
            # 'T2_minus_T700_mean',
            # 'TH2_mean', 'TH2_minus_theta_500_mean', 'TH2_minus_theta_700_mean', 'TH2_minus_theta_850_mean',
            # 'SST_mean', 'theta_e_700_mean', 'theta_e_850_mean', 'TH850',
            # 'SST_minus_T500_mean', 'SST_minus_T700_mean', 'theta_SST_minus_theta_500_mean', 
            # 'theta_SST_minus_theta_700_mean', 'theta_SST_minus_theta_850_mean',
            # 'theta_e_SST_minus_theta_e_500_mean', 'theta_e_SST_minus_theta_e_700_mean',
            # 'theta_e_SST_minus_theta_e_850_mean',
            # 'MCAO1_500_mean', 'MCAO1_700_mean', 'MCAO2_mean',
            # 'grad_theta_e_850_mean', 'T2_delta', 'T850_delta', 'T700_delta', 'T500_delta',
            # 'TH500_delta', 'TH700_delta', 'TH850_delta',
            # 'PBL_med', 'rel_vor_850_med',
            'trop_height', 'pbl_height', 'pbl_trop_frac',
            # 'theta_trop_med', 'delta_theta_trop_theta_sst', 'pressure_trop', 'wspd_trop',
            # 'HFX_rad', 'LH_rad', 'mucape_95', 'mcin_95', 'helicity_95', 'pw_95', 'pw_sum',
            # 'w_925', 
            'w_850', 'w_500', 

            # 'N_500',
            # 'propagation_speed', 'differential_wind_vector', 'vertical_shear_strength',
            # 'alpha_d', 'alpha_p', 'vertical_shear_angle', 'vertical_shear_vector_u', 'vertical_shear_vector_v',
            # 'wind_shear_10m_500', 'T2_disp', 'SST_disp', 'TH850_disp',
            'RH_850', 'RAIN_HOURLY_sum', 'RAIN_HOURLY_95', 'RAIN_HOURLY_med',
            # 'george_index', 'LI_rad', 'z500_rad', 'U200_95', 'Q850_95',
            # 'LCL_95', 'LFC_95', 'LFC_LCL', 'mcin_95',
            # 'DBZ_sfc_500_mean', 'rh_95', 'updraft_helicity',
            # 'INTEGR_VAPOR_TRANSP', 'U_VAPOR_TRANS', 'V_VAPOR_TRANS'
        ]
    
    def _calculate_basic_params(self, extractor: DataExtractor, aggregator: SpatialAggregator) -> Dict[str, float]:
        """Расчет базовых параметров"""
        results = {}
        
        # SLP
        slp = extractor.get('slp')
        results['SLP_center'] = aggregator.aggregate(slp, AggregationMethod.POINT)
        results['SLP_95'] = aggregator.aggregate(slp, AggregationMethod.PERCENTILE_95)
        results['SLP_diff_cent_95'] = results['SLP_95'] - results['SLP_center']
        
        # Ветер на 10м
        wind_10m = extractor.get('uvmet10')
        wind_10m = safe_values(wind_10m)  # <-- ДОБАВИТЬ
        wind_speed = np.sqrt(wind_10m[0]**2 + wind_10m[1]**2)
        results['U10_mean'] = aggregator.aggregate(wind_speed, AggregationMethod.MEDIAN)
        
        # Ветер на 500 гПа
        u500, v500 = extractor.get_wind()
        pressure = extractor.get_pressure()
        u500_interp = interplevel(u500, pressure, 500., meta=True)
        v500_interp = interplevel(v500, pressure, 500., meta=True)
        u500_np = safe_values(u500_interp)  # <-- использовать safe_values
        v500_np = safe_values(v500_interp)  # <-- использовать safe_values
        wind_speed_500 = np.sqrt(u500_np**2 + v500_np**2)
        results['U500_mean'] = aggregator.aggregate(wind_speed_500, AggregationMethod.MEDIAN)
    
        # Ветер на 850 гПа
        u850, v850 = extractor.get_wind()
        u850_interp = interplevel(u850, pressure, 850., meta=True)
        v850_interp = interplevel(v850, pressure, 850., meta=True)
        u850_np = safe_values(u850_interp)  # <-- использовать safe_values
        v850_np = safe_values(v850_interp)  # <-- использовать safe_values
        wind_speed_850 = np.sqrt(u850_np**2 + v850_np**2)
        results['U850_mean'] = aggregator.aggregate(wind_speed_850, AggregationMethod.MEDIAN)
    
        results['U500_U850_frac'] = results['U500_mean']/results['U850_mean']
        
        return results
    
    def _calculate_thermodynamic_params(self, extractor: DataExtractor, 
                                       aggregator: SpatialAggregator) -> Dict[str, float]:
        """Расчет термодинамических параметров"""
        results = {}
        
        # Температуры
        temp = extractor.get('temp')
        pressure = extractor.get_pressure()
        t_500 = interplevel(temp, pressure, 500., meta=True)
        t_850 = interplevel(temp, pressure, 850., meta=True)

        t_500_med = aggregator.aggregate(t_500, AggregationMethod.MEDIAN)
        t_850_med = aggregator.aggregate(t_850, AggregationMethod.MEDIAN)
        
        # t_500 = extractor.get_temperature_at_level(WRFDataLevel.LEVEL_500)
        # t_850 = extractor.get_temperature_at_level(WRFDataLevel.LEVEL_850)
        t2 = extractor.get('T2')
        
        # Конвертируем t2, если еще не сконвертирован
        t2 = safe_values(t2)  # <-- ДОБАВИТЬ (на случай если get() не сконвертировал)
        
        t2_med = aggregator.aggregate(t2, AggregationMethod.MEDIAN)

        results['T2_minus_T500_mean'] = t2_med - t_500_med
        results['T2_minus_T850_mean'] = t2_med - t_850_med
        
        # results['T2_minus_T500_mean'] = aggregator.aggregate(t2 - t_500, AggregationMethod.MEDIAN)
        # results['T2_minus_T850_mean'] = aggregator.aggregate(t2 - t_850, AggregationMethod.MEDIAN)
        results['T850_disp'] = aggregator.aggregate(t_850, AggregationMethod.STD)
        
        # RH
        rh = extractor.get('rh')
        pressure = extractor.get_pressure()
        rh_850 = interplevel(rh, pressure, 850., meta=True)
        rh_850 = safe_values(rh_850)  # <-- ДОБАВИТЬ safe_values
        results['RH_850'] = aggregator.aggregate(rh_850, AggregationMethod.MEDIAN)
        
        return results
    
    def _calculate_dynamics_params(self, extractor: DataExtractor, 
                                  aggregator: SpatialAggregator,
                                  idx: int, storm_data: pd.DataFrame) -> Dict[str, float]:
        """Расчет динамических параметров"""
        results = {}
    
        # Тропопауза
        z = extractor.get('z')
        pot_vorticity = extractor.get('pvo')
        trop_calc = TropopauseCalculator(self.config)
        trop_height = trop_calc.calculate(z, pot_vorticity)
        results['trop_height'] = aggregator.aggregate(trop_height, AggregationMethod.MEDIAN)
    
        pblh = extractor.get('PBLH')
        pblh = safe_values(pblh)  # <-- ДОБАВИТЬ safe_values
        results['pbl_height'] = aggregator.aggregate(pblh, AggregationMethod.MEDIAN)
    
        results['pbl_trop_frac'] = results['pbl_height']/results['trop_height']
        
        # Вертикальная скорость
        w = extractor.get('wa')
        pressure = extractor.get_pressure()
        w_500 = interplevel(w, pressure, 500., meta=True)
        w_850 = interplevel(w, pressure, 850., meta=True)
        w_500 = safe_values(w_500)  # <-- ДОБАВИТЬ safe_values
        w_850 = safe_values(w_850)  # <-- ДОБАВИТЬ safe_values
    
        results['w_500'] = aggregator.aggregate(w_500, AggregationMethod.PERCENTILE_95)
        results['w_850'] = aggregator.aggregate(w_850, AggregationMethod.PERCENTILE_95)
        
        # Ветровой сдвиг
        u, v = extractor.get_wind()
        u_850 = interplevel(u, pressure, 850., meta=True)
        v_850 = interplevel(v, pressure, 850., meta=True)
        u_500 = interplevel(u, pressure, 500., meta=True)
        v_500 = interplevel(v, pressure, 500., meta=True)
        
        # КОНВЕРТИРУЕМ ВСЕ
        u_500 = safe_values(u_500)  # <-- ДОБАВИТЬ
        v_500 = safe_values(v_500)  # <-- ДОБАВИТЬ
        u_850 = safe_values(u_850)  # <-- ДОБАВИТЬ
        v_850 = safe_values(v_850)  # <-- ДОБАВИТЬ
    
        du = u_500 - u_850
        dv = v_500 - v_850
        
        mean_du, mean_dv, mean_alpha_d, mean_dV = aggregator.aggregate_vector(du, dv)
        results['U500_minus_U850'] = mean_dV


        pv = extractor.get('pvo')
        
        pv_850 = interplevel(pv, pressure, 850., meta=True)
        pv_500 = interplevel(pv, pressure, 500., meta=True)
            
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        results['PV_850_mean'] = aggregator.aggregate(pv_850, AggregationMethod.MEDIAN)
        results['PV_500_mean'] = aggregator.aggregate(pv_500, AggregationMethod.MEDIAN)
        
        return results
    
    def _calculate_precipitation_params(self, extractor: DataExtractor,
                                       aggregator: SpatialAggregator,
                                       ds: Dataset, time_idx: int,
                                       current_time: pd.Timestamp) -> Dict[str, float]:
        """Расчет параметров осадков"""
        results = {}
        
        # RAIN_HOURLY
        rain_data = self._get_rain_with_previous(ds, time_idx, current_time)
        if rain_data is not None:
            results['RAIN_HOURLY_95'] = aggregator.aggregate(rain_data, AggregationMethod.PERCENTILE_95)
            results['RAIN_HOURLY_med'] = aggregator.aggregate(rain_data, AggregationMethod.MEDIAN)
            results['RAIN_HOURLY_sum'] = aggregator.aggregate(rain_data, AggregationMethod.SUM)
        else:
            results['RAIN_HOURLY_95'] = np.nan
            results['RAIN_HOURLY_med'] = np.nan
            results['RAIN_HOURLY_sum'] = np.nan
        
        return results
    
    def _get_rain_with_previous(self, ds: Dataset, time_idx: int, current_time: pd.Timestamp) -> Optional[np.ndarray]:
        """Получение часовых осадков с учетом предыдущего файла"""
        if time_idx > 0:
            RAINC = getvar(ds, "RAINC", timeidx=time_idx)
            RAINNC = getvar(ds, "RAINNC", timeidx=time_idx)
            RAINSH = getvar(ds, "RAINSH", timeidx=time_idx)
            
            RAINC_p = getvar(ds, "RAINC", timeidx=time_idx-1)
            RAINNC_p = getvar(ds, "RAINNC", timeidx=time_idx-1)
            RAINSH_p = getvar(ds, "RAINSH", timeidx=time_idx-1)
        else:
            prev_file = find_wrf_file_for_time(current_time - pd.Timedelta(hours=1), self.wrf_files)
            if not prev_file:
                return None
            
            with Dataset(prev_file) as ds_prev:
                last_idx = len(ds_prev.dimensions['Time']) - 1
                
                RAINC = getvar(ds, "RAINC", timeidx=0)
                RAINNC = getvar(ds, "RAINNC", timeidx=0)
                RAINSH = getvar(ds, "RAINSH", timeidx=0)
                
                RAINC_p = getvar(ds_prev, "RAINC", timeidx=last_idx)
                RAINNC_p = getvar(ds_prev, "RAINNC", timeidx=last_idx)
                RAINSH_p = getvar(ds_prev, "RAINSH", timeidx=last_idx)
        
        # КОНВЕРТИРУЕМ ВСЕ
        RAINC = safe_values(RAINC)
        RAINNC = safe_values(RAINNC)
        RAINSH = safe_values(RAINSH)
        RAINC_p = safe_values(RAINC_p)
        RAINNC_p = safe_values(RAINNC_p)
        RAINSH_p = safe_values(RAINSH_p)
        
        return (RAINC - RAINC_p) + (RAINNC - RAINNC_p) + (RAINSH - RAINSH_p)
        
    def _calculate_ivt_params(self, extractor: DataExtractor,
                             aggregator: SpatialAggregator,
                             ds: Dataset, time_idx: int) -> Dict[str, float]:
        """Расчет параметров IVT"""
        results = {}
        
        ivt_calc = IVTCalculator(ds, time_idx, self.config)
        ivt_results = ivt_calc.calculate()
        
        results['INTEGR_VAPOR_TRANSP'] = aggregator.aggregate(ivt_results['integral'], AggregationMethod.PERCENTILE_95)
        results['U_VAPOR_TRANS'] = aggregator.aggregate(ivt_results['u_dir'], AggregationMethod.PERCENTILE_95)
        results['V_VAPOR_TRANS'] = aggregator.aggregate(ivt_results['v_dir'], AggregationMethod.PERCENTILE_95)
        
        return results
    
    def _calculate_bergeron_indexes(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Расчет индексов Бержерона"""
        results = {}
        for tau in self.calc_config.tau_hours:
            results[f'bergeron_{tau}h'] = BergeronsIndexCalculator.calculate(df, tau)
        return results

# ==================== Параллельная обработка ====================

class CycloneProcessorParallel:
    """Параллельная обработка циклонов"""
    
    def __init__(self, wrf_files: List[str], output_dir: Path, n_processes: int = None):
        self.wrf_files = wrf_files
        self.output_dir = output_dir
        self.n_processes = n_processes or min(cpu_count(), 16)
        
    def process_all(self, track_files: List[Path]) -> List[Dict]:
        """Обработка всех треков параллельно"""
        output_dir = self.output_dir / "hourly_data"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stats_dir = self.output_dir / "statistics"
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Обработка {len(track_files)} треков с {self.n_processes} процессами")
        
        args_list = [(track_file, self.wrf_files, output_dir, self.n_processes) for track_file in track_files]
        
        all_stats = []
        start_time = time.time()
        
        with Pool(processes=self.n_processes) as pool:
            results = list(tqdm(
                pool.imap_unordered(self._process_single_track_wrapper, args_list),
                total=len(track_files),
                desc="Обработка треков"
            ))
        
        for result in results:
            if result:
                all_stats.append(result)
        
        elapsed = time.time() - start_time
        logger.info(f"Обработка завершена за {elapsed:.1f} сек")
        logger.info(f"Успешно обработано: {len(all_stats)}/{len(track_files)}")
        
        # Сохранение статистики
        if all_stats:
            stats_df = pd.DataFrame(all_stats)
            stats_file = stats_dir / "all_tracks_statistics.csv"
            stats_df.to_csv(stats_file, index=False)
            logger.info(f"Статистика сохранена в {stats_file}")
        
        return all_stats
    
    @staticmethod
    def _process_single_track_wrapper(args: Tuple[Path, List[str], Path, int]) -> Optional[Dict]:
        """Обертка для обработки одного трека"""
        track_file, wrf_files, output_dir, n_procs = args
        try:
            processor = CycloneProcessor(wrf_files)
            df = processor.process_track(str(track_file), output_dir)
            if df is not None:
                return CycloneProcessorParallel._calculate_track_stats(df, track_file.name)
        except Exception as e:
            logger.error(f"Ошибка обработки {track_file}: {e}")
        return None
    
    @staticmethod
    def _calculate_track_stats(df: pd.DataFrame, filename: str) -> Dict:
        """Расчет статистики трека"""
        stats = {'filename': filename}
        
        # Базовые параметры для статистики
        params = [
            'SLP_center', 'SLP_95', 'SLP_diff_cent_95',
            'U10_mean', 'U500_mean', 'U850_mean', 'U500_U850_frac', 'U500_minus_U850',
            'PV_850_mean','PV_500_mean',
            'T2_minus_T500_mean', 'T2_minus_T850_mean', 'T850_disp',
            'trop_height', 'pbl_height', 'pbl_trop_frac',
            'w_850', 'w_500', 
            'RH_850', 'RAIN_HOURLY_sum', 'RAIN_HOURLY_95', 'RAIN_HOURLY_med',

        ]
        
        for param in params:
            if param in df.columns and df[param].notna().sum() > 0:
                stats[f'{param}_mean'] = df[param].mean()
                stats[f'{param}_min'] = df[param].min()
                stats[f'{param}_max'] = df[param].max()
        
        # # Время жизни
        # time_diff = df['time'].max() - df['time'].min()
        # stats['lifetime_hours'] = time_diff.total_seconds() / 3600
        
        # # Длина трека
        # if 'track_len' in df.columns:
        #     stats['track_length_km'] = df['track_len'].max()
        
        return stats

# ==================== Точка входа ====================

# if __name__ == "__main__":

#     EC_tracks = True

    
#     track_files = glob( "/storage/thalassa/users/vkoshkina/data/LoRes/LoRes/EddyClicker_Egor_2010/*.csv")
#     wrf_files = sorted(glob("/storage/NAAD/NAAD/LoRes/2010/wrfout_d01_2010*"))
#     output_dir = Path("/storage/thalassa/users/vkoshkina/data/LoRes/LoRes/EddyClicker_tracks_Egor_2010_15params_2026-08-11")


#     # track_files = glob( "/storage/kubrick/nikitenko/tracks_hourly_geometry/*.csv")
#     # wrf_files = sorted(glob("/storage/buffer/SMP/MODELS/WRF/OUTPUT/2019/wrfout_d01_2019*"))
#     # output_dir = Path("/storage/thalassa/users/vkoshkina/data/SMP/EddyClicker_tracks_2019_15params_2026-08-11")
    
    
#     # Обработка
#     processor = CycloneProcessorParallel(wrf_files, output_dir)
#     stats = processor.process_all([Path(f) for f in track_files])




if __name__ == "__main__":


    EC_tracks = False
    
    months = np.arange(1,13)

    data_type = 'LoRes'
    data_type = 'SMP'
    
    sigma = 2
    circ = 'C'
    
    pref_tracking = 'update_2026-05-13'
    CVS_speed = 'adv_speed'
    tracking_type = 'tracking_local_2_phase'
    results_dir = f"{tracking_type}_{CVS_speed}_{pref_tracking}"
    
    if data_type == 'LoRes':
        years = np.arange(1979,2019)
        years = np.arange(2010,2011)
        
        path_dir_data = '/storage/thalassa/users/vkoshkina/data/LoRes/'
        TRACKS_PATH = f"{path_dir_data}/{data_type}/{data_type}_tracks_sigma_{sigma}/{pref_tracking}/{results_dir}/tracks_{circ}/"
        OUTPUT_PATH = f"{path_dir_data}/{data_type}/{data_type}_tracks/{data_type}_tracks_1979-2018_15params_2026-08-11"
    elif data_type == 'SMP':
        years = np.arange(2019,2020)
        
        path_dir_data = '/storage/thalassa/users/vkoshkina/data/'
        TRACKS_PATH = f"{path_dir_data}/{data_type}/{data_type}_tracks_sigma_{sigma}/{pref_tracking}/{results_dir}/tracks_{circ}/"
        OUTPUT_PATH = f"{path_dir_data}/{data_type}/{data_type}_tracks/{data_type}_tracks_2019_15params_2026-08-11"

    for year in tqdm(years, total=len(years)):
        for month in months:
            TRACKS_PATH_monthly = glob(f"{TRACKS_PATH}/{year}-{month:02d}/*.csv")

            # print(TRACKS_PATH)
            
            if data_type == 'LoRes':
                WRF_PATH = sorted(glob(f"/storage/NAAD/NAAD/LoRes/{year}/wrfout_d01_{year}*"))
            elif data_type == 'SMP':
                WRF_PATH = sorted(glob(f"/storage/buffer/SMP/MODELS/WRF/OUTPUT/{year}/wrfout_d01_{year}*"))

            
            OUTPUT_PATH_monthly = Path(f"{OUTPUT_PATH}/{year}-{month:02d}")
            
            # Обработка
            processor = CycloneProcessorParallel(WRF_PATH, OUTPUT_PATH_monthly)
            stats = processor.process_all([Path(f) for f in TRACKS_PATH_monthly])
    