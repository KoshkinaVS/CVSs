import pandas as pd
import numpy as np
import xarray as xr
import os
from glob import glob
from tqdm import tqdm
import warnings
from netCDF4 import Dataset
from wrf import getvar, interplevel, ll_to_xy
warnings.filterwarnings('ignore')


from metpy.calc import lcl, parcel_profile
from metpy.units import units
from metpy.calc import potential_temperature
from metpy.calc import equivalent_potential_temperature
from metpy.calc import gradient
from metpy.calc import brunt_vaisala_frequency


from multiprocessing import Pool, cpu_count
import traceback
import time
from geopy.distance import distance



def get_rain_with_previous(ds_current, time_idx, current_time, wrf_files):
    """
    Получает осадки для текущего часа и разницу с предыдущим
    Если time_idx=0 - ищет последний час в предыдущем файле
    """
    
    if time_idx > 0:
        # Простой случай - есть предыдущий час в этом же файле
        RAINC = getvar(ds_current, "RAINC", timeidx=time_idx)
        RAINNC = getvar(ds_current, "RAINNC", timeidx=time_idx)
        RAINSH = getvar(ds_current, "RAINSH", timeidx=time_idx)
        
        RAINC_p = getvar(ds_current, "RAINC", timeidx=time_idx-1)
        RAINNC_p = getvar(ds_current, "RAINNC", timeidx=time_idx-1)
        RAINSH_p = getvar(ds_current, "RAINSH", timeidx=time_idx-1)
        
    else:
        # Первый час в файле - ищем предыдущий файл
        prev_file = find_previous_wrf_file(current_time, wrf_files)
        
        if prev_file:
            # print(f"    Берем последний час из предыдущего файла: {os.path.basename(prev_file)}")
            ds_prev = Dataset(prev_file)
            
            # Последний таймстеп в предыдущем файле
            last_idx = len(ds_prev.dimensions['Time']) - 1
            
            RAINC = getvar(ds_current, "RAINC", timeidx=0)
            RAINNC = getvar(ds_current, "RAINNC", timeidx=0)
            RAINSH = getvar(ds_current, "RAINSH", timeidx=0)
            
            RAINC_p = getvar(ds_prev, "RAINC", timeidx=last_idx)
            RAINNC_p = getvar(ds_prev, "RAINNC", timeidx=last_idx)
            RAINSH_p = getvar(ds_prev, "RAINSH", timeidx=last_idx)
            
            ds_prev.close()
        else:
            # Нет предыдущего файла - не можем посчитать
            print(f"RAIN: Нет предыдущего файла для {current_time}")
            return None, None, None
    
    # Разница
    delta_RAINC = RAINC - RAINC_p
    delta_RAINNC = RAINNC - RAINNC_p
    delta_RAINSH = RAINSH - RAINSH_p
    
    RAIN_HOURLY = delta_RAINC + delta_RAINNC + delta_RAINSH
    
    return RAIN_HOURLY

def find_previous_wrf_file(current_time, wrf_files):
    """Находит предыдущий WRF файл по времени"""
    current_dt = pd.to_datetime(current_time)
    
    # Предыдущий день
    prev_day = current_dt - pd.Timedelta(days=1)
    prev_day_str = prev_day.strftime('%Y-%m-%d')
    
    for wrf_file in wrf_files:
        if prev_day_str in wrf_file:
            return wrf_file
    
    return None



def get_tropopause_height_by_2pvu(z, pot_vorticity,min_height=3000):
    """
    Получение высоты тропопаузы по поверхности 2 PVU
    """
    
    # Находим высоту, где PV = 2 PVU (тропопауза)
    ny, nx = pot_vorticity.shape[1], pot_vorticity.shape[2]
    trop_height = np.full((ny, nx), np.nan)  # массив для высоты тропопаузы

    # Порог для тропопаузы (2 PVU)
    pvu_threshold = 2.0

    for i in range(ny):
        for j in range(nx):
            # Профиль PV и высоты в точке
            pv_profile = pot_vorticity[:, i, j].values
            z_profile = z[:, i, j].values
            sort_idx = np.argsort(z_profile)

            pv_sorted = pv_profile[sort_idx]
            z_sorted = z_profile[sort_idx]
            high_levels = z_sorted >= min_height
            if not np.any(high_levels):
                continue
            pv_high = pv_sorted[high_levels]
            z_high = z_sorted[high_levels]
            # Ищем уровень, где PV пересекает порог
            idx = np.where(pv_high >= pvu_threshold)[0]
           
            if len(idx) > 0:
                first_idx = idx[0]
                
                if first_idx > 0:
                    # Линейная Интерполяция между уровнями для точного значения
                    pv1 =   pv_high [first_idx - 1]
                    pv2 =   pv_high [first_idx]
                    z1 =   z_high[first_idx - 1]
                    z2 =   z_high[first_idx]
                    
                    if pv2 - pv1 != 0:
                        fraction = (pvu_threshold - pv1) / (pv2 - pv1)
                        trop_height[i, j] = z1 + fraction * (z2 - z1)
                    else:
                        trop_height[i, j] = z1
                else:
                    trop_height[i, j] =   z_high[first_idx]

    return  trop_height

'IVT = (1/g) * ∫(q * V) dp'
def calculate_ivt(ds, time_idx, bottom_pressure=1000, top_pressure=300,type=''):
    """
    Расчет Integrated Vapor Transport (IVT) для WRF output
    - bottom_pressure: нижний уровень интегригования(гПа) - 1000 гпа
    - top_pressure: верхний уровень интегригования (гПа) - 300 гпа
    """
    # Константы
    g = 9.81  # м/с²
    
    # Получаем 3D переменные
    q = getvar(ds, 'QVAPOR', timeidx=time_idx)  # удельная влажность (кг/кг)
    u = getvar(ds, 'ua', timeidx=time_idx)       # U-компонента ветра (м/с)
    v = getvar(ds, 'va', timeidx=time_idx)       # V-компонента ветра (м/с)
    pressure = getvar(ds, 'pressure', timeidx=time_idx)  # давление (гПа)
    
    # Создаем уровни давления для интегригования
    p_levels = np.linspace(bottom_pressure, top_pressure, 100)  # 100 уровней
    
    # Интерполируем все поля на уровни давления
    qv_interp = interplevel(q, pressure, p_levels)
    u_interp = interplevel(u, pressure, p_levels)
    v_interp = interplevel(v, pressure, p_levels)
    
    # Скорость ветра на каждом уровне
    wind_speed = np.sqrt(u_interp**2 + v_interp**2)
    
    # Компоненты переноса
    qu = qv_interp * u_interp  # перенос в направлении U
    qv = qv_interp * v_interp  # перенос в направлении V
    
    # Интегрируем по вертикали
    # dp в Па между уровнями
    dp = np.diff(p_levels * 100.0)  # перепад давления между уровнями в Па
    dp = np.append(dp, dp[-1])  # выравниваем размерность
    dp = dp.reshape(100, 1, 1)
    # Интеграл (сумма по вертикали)
    ivt_u = np.sum(qu * dp / g, axis=0)  # компонента U
    ivt_v = np.sum(qv * dp / g, axis=0)  # компонента V
    
    # Общий IVT (магнитуда)
    ivt = np.sqrt(ivt_u**2 + ivt_v**2)
    # Возвращаем результаты!!!!!!!!!
    if type=='integral':
        return ivt           
    elif type == 'U_dir':
        return ivt_u   
    elif type == 'V_dir':
        return ivt_v   

def calculate_li(center_lat_idx,center_lon_idx,t_500,T2,pressure,radius_pixels,dewpoint_sst):
 
    # СОЗДАЕМ ПОЛЕ ДЛЯ LI (заполняем NaN)
    li_field = np.full_like(t_500, np.nan)
    # Создаем маску радиуса
    ny, nx = T2.shape
    y_coords, x_coords = np.ogrid[:ny, :nx]
    distances = np.sqrt((y_coords - center_lat_idx)**2 + (x_coords - center_lon_idx)**2)
    radius_mask = distances <= radius_pixels
    # Находим все точки в радиусе
    points_in_radius = list(zip(*np.where(radius_mask)))

    # Счетчик для отладки
    valid_points = 0

    # Для каждой точки в радиусе считаем свой LI
    for lat_idx, lon_idx in points_in_radius:
        
        # Профиль давления в этой точке
        p_profile_vals = pressure[:, lat_idx, lon_idx]
        if hasattr(p_profile_vals, 'values'):
            p_profile_vals = p_profile_vals.values
            
        # if np.min(p_profile_vals) <= 500 <= np.max(p_profile_vals):
        #     print(f" Уровень 500 гПа ВНУТРИ профиля")
        #     print(f"   Профиль от {np.min(p_profile_vals):.1f} до {np.max(p_profile_vals):.1f} hPa")
            
        # else:
        #     print(f" Уровень 500 гПа ВНЕ профиля!")
        #     print(f"   Профиль от {np.min(p_profile_vals):.1f} до {np.max(p_profile_vals):.1f} hPa")

        # Проверка на NaN
        if np.any(np.isnan(p_profile_vals)):
            continue
        
        # Приземные параметры в точке
        t2_sfc = float(T2[lat_idx, lon_idx].values) -273.15 # в цельсиях
        td_sfc = float(dewpoint_sst[lat_idx, lon_idx].values)  # в цельсиях
            # Получаем температуру окружающего воздуха на 500 гПа в этой точке
        if hasattr(t_500, 'values'):
            t500_env = float(t_500[lat_idx, lon_idx].values)
        else:
            t500_env = float(t_500[lat_idx, lon_idx])

        # Поднимаем ЧАСТИЦУ
        prof = parcel_profile(p_profile_vals * units('hPa'), t2_sfc * units.degC, td_sfc * units.degC)
        
        # Индекс уровня 500 гПа
        idx_500 = np.argmin(np.abs(p_profile_vals - 500))
        t500_parcel = prof[idx_500].to('K').magnitude
        # print(t500_parcel)
        # print(f"На 500 гПа температура: {t500_parcel-273.15:.1f}°C")

        li_value = t500_env - t500_parcel
        #ПРИСВАИВАЕМ УЗЛЫ С ЗНАЧЕНИЕМ LI в пределах радиуса!!
        li_field[lat_idx, lon_idx] = li_value
        valid_points += 1
        
    if valid_points > 0:
        # Создаем маску для валидных точек в радиусе
        valid_mask = radius_mask & ~np.isnan(li_field)               
        if np.any(valid_mask):
            li_in_rad= np.percentile(li_field[valid_mask],95)
            #print(li_in_rad)
    # print(f"T500 окружающий: {t500_env:.1f}K = {t500_env-273.15:.1f}°C")
    # print(f"T500 частица: {t500_parcel:.1f}K = {t500_parcel-273.15:.1f}°C")
    # print(f"T приземная: {t2_sfc:.1f}°C")
    # print(f"Td приземная: {td_sfc:.1f}°C")
    return li_in_rad


def calculate_mcao_kolstad_fast(theta_sst, theta_500, theta_700, slp):
    #тут все в паскали переводим!!!!! для репрезентативности
    """Расчет MCAO по Kolstad."""
    with np.errstate(divide='ignore', invalid='ignore'):
        mcao_500 = (theta_sst - theta_500) / (slp*100 - 50000)
        mcao_700 = (theta_sst - theta_700) / (slp*100 - 70000)
    return np.nan_to_num(mcao_500), np.nan_to_num(mcao_700)

def calculate_mcao_bracegirdle_fast(theta_sst, theta_700, z_700):
    """Расчет MCAO по Bracegirdle."""
    L = 7.5e5
    with np.errstate(divide='ignore', invalid='ignore'):
        mcao2 = (L / (z_700)) * (np.log(theta_sst) - np.log(theta_700))
    return np.nan_to_num(mcao2)


def calculate_bergeron_hourly(df, tau_hours=12):
    """
    Расчет индекса Бержерона для часовых данных.
    
    Args:
        df: DataFrame с колонками 'time', 'latitude', 'SLP_center'
        tau_hours: интервал (2,4,6,12 часов)
    
    Returns:
        list: значения индекса Бержерона
    """
    
    bergeron_values = [np.nan] * len(df)
    
    # Константа sin(45°)
    sin45 = np.sin(np.radians(45))
    
    # Сортируем по времени (хотя должны быть уже отсортированы)
    df_sorted = df.sort_values('time').reset_index(drop=True)
    times = pd.Series(df_sorted['time'].values)
    lats = df_sorted['latitude'].values
    slp = df_sorted['SLP_center'].values
    
    half_tau = tau_hours // 2  # для четного τ
    
    # print(f"      Расчет для τ={tau_hours}ч (half={half_tau}ч)")
    
    valid_count = 0
    
    for i in range(len(df_sorted)):
        # Проверяем, что есть данные за half_tau до и после
        if i - half_tau < 0 or i + half_tau >= len(df_sorted):
            continue
        
        # Для часовых данных индексы считаются точно
        idx_minus = i - half_tau
        idx_plus = i + half_tau
        
        # Проверяем временной шаг (должен быть ровно half_tau часов)
        time_minus = times.iloc[idx_minus]
        time_plus = times.iloc[idx_plus]
        time_current = times.iloc[i]
        
        # Проверяем, что интервалы точно по half_tau часов
        diff_minus = (time_current - time_minus).total_seconds() / 3600
        diff_plus = (time_plus - time_current).total_seconds() / 3600
        
        if abs(diff_minus - half_tau) < 0.1 and abs(diff_plus - half_tau) < 0.1:
            
            # Средняя широта между точками ДО и ПОСЛЕ
            mean_lat = (lats[idx_minus] + lats[idx_plus]) / 2
            
            # Широтный фактор
            sin_lat = np.sin(np.radians(mean_lat))
            lat_factor = sin45 / sin_lat
            
            # Изменение давления
            delta_p = slp[idx_minus] - slp[idx_plus]
            
            # Индекс Бержерона
            bergeron = (delta_p / 12) * lat_factor
            bergeron_values[i] = bergeron
            valid_count += 1
            
    #         if valid_count <= 3:  #  примеры
    #             print(f"        i={i}: t={time_current}, φ={mean_lat:.1f}°, "
    #                   f"Δp={delta_p:.1f}, Berg={bergeron:.2f}")
    
    # print(f"      Рассчитано точек: {valid_count}/{len(df)}")
    return bergeron_values


def calculate_all_bergeron(df):
    """
    Расчет индексов для разных масштабов в арктическом регионе (70-80°N).
    """
    results = {}
    
    # Все масштабы имеют смысл
    tau_options = [2, 4, 6, 12,24]

    for tau in tau_options:
        bergeron = calculate_bergeron_hourly(df, tau_hours=tau)
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!и сразу записываем как ежечасные данные!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        results[f'bergeron_{tau}h'] = bergeron
        
        # Статистика
        valid_values = [v for v in bergeron if not np.isnan(v)]
        # if valid_values:
        #     print(f"      τ={tau:2d}ч: {len(valid_values):3d} точек, "
        #           f"средний={np.mean(valid_values):6.2f}, "
        #           f"мин={np.min(valid_values):6.2f}, "
        #           f"макс={np.max(valid_values):6.2f}")
    
    return results


def find_grid_indices_fast(target_lat, target_lon, lats, lons):
    """Быстрый поиск индексов сетки."""
    lat_diff = np.abs(lats - target_lat)

    lon_diff = np.abs(lons - target_lon)

    total_diff = lat_diff + lon_diff
    return np.unravel_index(np.argmin(total_diff), lats.shape)
def parse_wrf_time(time_bytes):
    """Парсит время из WRF файла."""
    time_str = time_bytes.tobytes().decode('utf-8').strip()
    # Формат: 2019-01-01_00:00:00 → заменяем _ на пробел
    time_str = time_str.replace('_', ' ')
    return pd.to_datetime(time_str, format='%Y-%m-%d %H:%M:%S')
def find_wrf_file_for_time(target_time, wrf_files):
    """Находит WRF файл, который содержит заданное время."""
    target_datetime = target_time
    
    #  ИСКЛЮЧАЕМ ПРОБЛЕМНЫЙ ФАЙЛ 2019-01-01
    filtered_wrf_files = [f for f in wrf_files if "2019-01-01_00:00:00" not in f]
    
    # print(f" Поиск WRF для {target_datetime}")
    
    for wrf_file in filtered_wrf_files:
        try:
            filename = os.path.basename(wrf_file)
            date_part = filename.split('_')[2] + '_' + filename.split('_')[3]
            file_start_time = pd.to_datetime(date_part, format='%Y-%m-%d_%H:%M:%S')
            file_end_time = file_start_time + pd.Timedelta(hours=23)
            
            # Проверяем, попадает ли целевое время в диапазон файла
            if file_start_time <= target_datetime <= file_end_time:
                # print(f"    Найден файл: {filename}")
                return wrf_file
                
        except Exception as e:
            print(f"    Ошибка обработки файла {filename}: {e}")
            continue
    
    print(f"    Не найден WRF файл для времени {target_datetime}")
    return None

''''ОБРАБОТКА ЗНАЧЕНИЙ POLEAWRDS ОТ РАДИУСА'''
def get_poleward_sector(data, center_lat_idx, center_lon_idx, radius_pixels, lats, method='mean'):
    """Быстрое извлечение значений в северном секторе."""
    try:
    
        data = data.values if hasattr(data, 'values') else data
        lats = lats.values if hasattr(lats, 'values') else lats
        
        ny, nx = data.shape
        y_coords, x_coords = np.ogrid[:ny, :nx]
        distances = np.sqrt((y_coords - center_lat_idx)**2 + (x_coords - center_lon_idx)**2)
        radius_mask = distances <= radius_pixels
        north_mask = lats > lats[center_lat_idx, center_lon_idx]
        combined_mask = radius_mask & north_mask
        #для первого таймстепа лишь 1 точка
        if not np.any(combined_mask):
            return np.nan
        #и для остальных данных
        data_in_sector = data[combined_mask]
        valid_data = data_in_sector[~np.isnan(data_in_sector)]
        if method == 'mean':
            return np.mean(valid_data)
        elif method == 'median':
            return np.median(valid_data)
        elif isinstance(method, (int, float)):
            return np.percentile(valid_data, method)
    
    except Exception as e:
        print(f"    Ошибка в get_poleward_sector: {e}")
        return np.nan

''''ОБРАБОТКА ЗНАЧЕНИЙ В ПРЕДЕЛАХ РАДИУСА ИЛИ В ТОЧКЕ'''
def get_values_in_radius(data, center_lat_idx, center_lon_idx, radius_pixels, method='', param_name=''):
    try:
        # Извлекаем numpy массив
        if hasattr(data, 'values'):
            data_values = data.values
        else:
            data_values = data
         # Для скалярных величин берем модуль комплексных чисел
        if np.iscomplexobj(data_values):
            data_values = np.abs(data_values)

        # print(f"Данные: {data_values.shape}, центр: ({center_lat_idx}, {center_lon_idx})")
        
        #  ДЛЯ 2D ДАННЫХ
        if data_values.ndim == 2:
            return get_single_level_values(data_values, center_lat_idx, center_lon_idx, radius_pixels, method)
        
        else:
            print(f"Неподдерживаемая размерность: {data_values.ndim} для {param_name}")
            return np.nan
            
    except Exception as e:
        print(f"       Ошибка в get_values_in_radius: {e}")
        return np.nan


def get_values_in_radius(data, center_lat_idx, center_lon_idx, radius_pixels, method='', param_name=''):
    try:
        # Извлекаем numpy массив
        if hasattr(data, 'values'):
            data_values = data.values
        elif isinstance(data, tuple):
            # Если это кортеж (например, от getvar), берем первый элемент
            data_values = data[0] if len(data) > 0 else np.array([])
        else:
            data_values = data
        
        # Проверяем, что данные не пустые
        if data_values is None or len(data_values) == 0:
            return np.nan
        
        # Для скалярных величин берем модуль комплексных чисел
        if np.iscomplexobj(data_values):
            data_values = np.abs(data_values)
        
        # Для 2D данных
        if data_values.ndim == 2:
            return get_single_level_values(data_values, center_lat_idx, center_lon_idx, radius_pixels, method)
        else:
            # Для 1D или других размерностей
            if data_values.ndim == 0:  # скаляр
                return float(data_values)
            elif data_values.ndim == 1:  # 1D массив
                return np.nanmedian(data_values) if method == 'median' else np.nanmean(data_values)
            else:
                print(f"Неподдерживаемая размерность: {data_values.ndim} для {param_name}")
                return np.nan
            
    except Exception as e:
        print(f"Ошибка в get_values_in_radius для {param_name}: {e}")
        return np.nan

def get_single_level_values(data_2d, center_lat_idx, center_lon_idx, radius_pixels, method=''):
    """Обработка одного уровня (2D данных)."""
    try:
        ny, nx = data_2d.shape
        
        if isinstance(radius_pixels, str):
            if radius_pixels.lower() == 'point':
                method = 'point'
                radius_pixels = 0


        # Создаем маску радиуса
        y_coords, x_coords = np.ogrid[:ny, :nx]
        distances = np.sqrt((y_coords - center_lat_idx)**2 + (x_coords - center_lon_idx)**2)
        radius_mask = distances <= radius_pixels
        
        if not np.any(radius_mask):
            return np.nan
        
        data_in_radius = data_2d[radius_mask]
        valid_data = data_in_radius[~np.isnan(data_in_radius)]
        
        if len(valid_data) == 0:
            return np.nan
        
        # Обработка разных методов
        if method == 'mean':
            return np.nanmean(valid_data)
        elif method == 'median':
            return np.nanmedian(valid_data)
        elif method == 'min':
            return np.nanmin(valid_data)
        elif method == 'max':
            return np.nanmax(valid_data)
        elif method == 'sum':
            return np.nansum(valid_data)
        elif method == 'std':
            return np.nanstd(valid_data, ddof=1)
        elif method == 'point':
            return data_2d[center_lat_idx, center_lon_idx]
        elif method == 'delta':
            # Разница между 95-м и 5-м перцентилями
            p5 = np.nanpercentile(valid_data, 5)
            p95 = np.nanpercentile(valid_data, 95)
            return p95 - p5
        elif isinstance(method, (int, float)):
            # Если method - число, используем как процентиль
            return np.nanpercentile(valid_data, method)
        
    except Exception as e:
        print(f"Ошибка в _get_single_level_values: {e}")
        return np.nan

def get_single_level_values(data_2d, center_lat_idx, center_lon_idx, radius_pixels, method=''):
    """Обработка одного уровня (2D данных)."""
    try:
        # Проверяем входные данные
        if data_2d is None:
            return np.nan
            
        ny, nx = data_2d.shape
        
        # Проверяем индексы центра
        if (center_lat_idx < 0 or center_lat_idx >= ny or 
            center_lon_idx < 0 or center_lon_idx >= nx):
            return np.nan
        
        if isinstance(radius_pixels, str):
            if radius_pixels.lower() == 'point':
                method = 'point'
                radius_pixels = 0
        
        # Создаем маску радиуса
        y_coords, x_coords = np.ogrid[:ny, :nx]
        distances = np.sqrt((y_coords - center_lat_idx)**2 + (x_coords - center_lon_idx)**2)
        radius_mask = distances <= radius_pixels
        
        if not np.any(radius_mask):
            return np.nan
        
        data_in_radius = data_2d[radius_mask]
        valid_data = data_in_radius[~np.isnan(data_in_radius)]
        
        if len(valid_data) == 0:
            return np.nan
        
        # Обработка разных методов
        if method == 'mean':
            return np.nanmean(valid_data)
        elif method == 'median':
            return np.nanmedian(valid_data)
        elif method == 'min':
            return np.nanmin(valid_data)
        elif method == 'max':
            return np.nanmax(valid_data)
        elif method == 'sum':
            return np.nansum(valid_data)
        elif method == 'std':
            return np.nanstd(valid_data, ddof=1)
        elif method == 'point':
            return data_2d[center_lat_idx, center_lon_idx]
        elif method == 'delta':
            # Разница между 95-м и 5-м перцентилями
            p5 = np.nanpercentile(valid_data, 5)
            p95 = np.nanpercentile(valid_data, 95)
            return p95 - p5
        elif isinstance(method, (int, float)):
            # Если method - число, используем как процентиль
            return np.nanpercentile(valid_data, method)
        else:
            # По умолчанию возвращаем медиану
            return np.nanmedian(valid_data)
        
    except Exception as e:
        print(f"Ошибка в get_single_level_values: {e}")
        return np.nan
    
''''ПРАВИЛЬНОЕ УСРЕДНЕНИЕ ДЛЯ ТЕРМИЧЕСКОГО ВЕТРА В ПРЕДЕЛАХ РАДИУСА!!!! Надо доработать потом'''
def calculate_vector_mean_in_radius(real_part, imag_part, center_lat_idx, center_lon_idx, radius_pixels):
    """Векторное усреднение в радиусе (для правильного расчёта направлений)"""
    ny, nx = real_part.shape
    y_coords, x_coords = np.ogrid[:ny, :nx]
    distances = np.sqrt((y_coords - center_lat_idx)**2 + (x_coords - center_lon_idx)**2)
    radius_mask = distances <= radius_pixels
    
    # Усредняем компоненты
    valid_mask = radius_mask & ~np.isnan(real_part) & ~np.isnan(imag_part)
    if not np.any(valid_mask):
        return np.nan, np.nan, np.nan, np.nan
    
    mean_real = np.nanmean(real_part[radius_mask])
    mean_imag = np.nanmean(imag_part[radius_mask])
    
    # Угол из усреднённых компонент
    mean_angle = np.degrees(np.arctan2(mean_imag, mean_real)) % 360
    mean_magnitude = np.sqrt(mean_real**2 + mean_imag**2)
    
    return mean_real, mean_imag, mean_angle, mean_magnitude

''' ФУНКЦИИ ОБРАБОТКИ ТРЕКОВ'''

def process_all_tracks(track_files_pattern, wrf_files_dir, output_dir, data_type='EC', n_processes=None):
    """Обработка всех треков с параллелизацией по трекам"""
    
    # Создание директорий
    os.makedirs(output_dir, exist_ok=True)
    hourly_dir = os.path.join(output_dir, "hourly_data")
    stats_dir = os.path.join(output_dir, "statistics")
    os.makedirs(hourly_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    
    # Поиск файлов
    track_files = glob(track_files_pattern)
    wrf_files = sorted(glob(wrf_files_dir))
    
    print(f"ЗАГРУЗКА ФАЙЛОВ:")
    print(f"   • Треков: {len(track_files)}")
    print(f"   • WRF файлов: {len(wrf_files)}")
    
    if n_processes is None:
        n_processes = min(cpu_count(), len(track_files))
    
    print(f"\n ПАРАЛЛЕЛЬНАЯ ОБРАБОТКА:")
    print(f"   • Всего треков для обработки: {len(track_files)}")
    print(f"   • Количество процессов: {n_processes}")
    
    # Подготовка аргументов для параллельной обработки
    args_list = []
    for track_file in track_files:
        args_list.append((track_file, wrf_files, hourly_dir, data_type))
    
    # Параллельная обработка треков
    all_track_stats = []
    start_time = time.time()
    
    with Pool(processes=n_processes) as pool:
        # Используем imap_unordered для прогресс-бара
        results = list(tqdm(
            pool.imap_unordered(process_single_track_wrapper, args_list),
            total=len(track_files),
            desc=" Обработка треков"
        ))
    
    # Сбор результатов
    for result in results:
        if result and not result[0].empty:
            df, hourly_file = result
            track_stats = calculate_track_statistics(df, os.path.basename(hourly_file))
            all_track_stats.append(track_stats)

    elapsed_time = time.time() - start_time
    
    # Сохранение статистики
    if all_track_stats:
        stats_df = pd.DataFrame(all_track_stats)
        stats_output = os.path.join(stats_dir, "all_tracks_statistics.csv")
        stats_df.to_csv(stats_output, index=False)
        
        print(f"\n{'='*60}")
        print("ИТОГОВАЯ СТАТИСТИКА")
        print(f"{'='*60}")
        print(f"   • Всего треков: {len(track_files)}")
        print(f"   • Успешно обработано: {len(all_track_stats)}")
        print(f"   • Пропущено: {len(track_files) - len(all_track_stats)}")
        print(f"   • Общее время: {elapsed_time:.1f} сек")
        print(f"   • Файл статистики: {stats_output}")
    
    return all_track_stats

def process_single_track_wrapper(args):
    """Обертка для обработки одного трека в параллельном процессе"""
    track_file, wrf_files, output_dir, data_type = args
    try:
        return process_single_track(track_file, wrf_files, output_dir, data_type)
    except Exception as e:
        print(f"\n[ERROR] Ошибка обработки трека {os.path.basename(track_file)}:")
        traceback.print_exc()
        return None



def calculate_track_statistics(df, track_filename):
    """Вычисляет статистику за всю жизнь циклона."""
    stats = {'filename': track_filename}
    
    # Определяем агрегации для каждого параметра
    param_aggregations = {
        #pressures
        'SLP_center': ['mean', 'min', 'max'],
        'SLP_median': ['mean', 'min', 'max'],
        'SLP_delta': ['mean', 'min', 'max'],
        'SLP_95': ['mean', 'min', 'max'],
        'SLP_diff(cent-med)': ['mean', 'min', 'max'],
        'SLP_diff(cent-95)': ['mean', 'min', 'max'],

        'PV_850_mean_cyclone': ['mean', 'min', 'max'],
        #temps
        'T500_mean': ['mean', 'min', 'max'],
        'T700_mean': ['mean', 'min', 'max'],
        'SST_mean': ['mean', 'min', 'max'],
        'theta_e_700_mean': ['mean', 'min', 'max'],
        'theta_e_850_mean': ['mean', 'min', 'max'],
        'SST_minus_T500_mean': ['mean', 'min', 'max'],
        'SST_minus_T700_mean': ['mean', 'min', 'max'],
        'theta_SST_minus_theta_500_mean': ['mean', 'min', 'max'],
        'theta_SST_minus_theta_700_mean': ['mean', 'min', 'max'], 
        'theta_SST_minus_theta_850_mean': ['mean', 'min', 'max'],
        'theta_e_SST_minus_theta_e_500_mean': ['mean', 'min', 'max'],
        'theta_e_SST_minus_theta_e_700_mean': ['mean', 'min', 'max'],
        'theta_e_SST_minus_theta_e_850_mean': ['mean', 'min', 'max'],
        'MCAO1_500_mean': ['mean', 'min', 'max'],
        'MCAO1_700_mean': ['mean', 'min', 'max'],
        'MCAO2_mean': ['mean', 'min', 'max'],
        'grad_theta_e_850_mean': ['mean', 'min', 'max'],
        'rel_vor_850_med': ['mean', 'min', 'max'],

        #wspeeds
        'U10_mean_cyclone': ['mean', 'min', 'max'],
        'U500_mean': ['mean', 'min', 'max'],
        'U500_poleward_cyclone': ['mean', 'min', 'max'],
        #trop
        'trop_height': ['mean', 'min', 'max'],
        'theta_trop_med': ['mean', 'min', 'max'],
        'delta_theta_trop-theta_sst': ['mean', 'min', 'max'],   
        'pressure_trop': ['mean', 'min', 'max'],
        'wspd_trop': ['mean', 'min', 'max'],
     
        '..............custom params.............'

        'HFX_rad': ['mean', 'min', 'max'], 
        'LH_rad': ['mean', 'min', 'max'], 

        'mucape_95': ['mean', 'min', 'max'],
        'cape_sur': ['mean', 'min', 'max'],

        'helicity_rad': ['mean', 'min', 'max'], 
        'pw_rad': ['mean', 'min', 'max'],
        'w_10m_mean': ['mean', 'min', 'max'], 
        'SLP_diff_cyclone': ['mean', 'min', 'max'],
        'N_mean': ['mean', 'min', 'max'],
        'propagation_speed': ['mean', 'min', 'max'],
        'differential_wind_vector': ['mean', 'min', 'max'],
        'vertical_shear_strength': ['mean', 'min', 'max'],
        'alpha_d': ['mean', 'min', 'max'],
        'alpha_p': ['mean', 'min', 'max'],
        'vertical_shear_angle': ['mean', 'min', 'max'],
        'vertical_shear_vector_u': ['mean', 'min', 'max'],
        'vertical_shear_vector_v': ['mean', 'min', 'max'],
        'wind_shear_10m_500': ['mean', 'min', 'max'],

        
        'RAIN_HOURLY': ['mean', 'min', 'max'],

        'PBL_med': ['mean', 'min', 'max'],
        #dispersions
        'T850': ['mean', 'min', 'max'],
        'T850_disp': ['mean', 'min', 'max'],
        'T2': ['mean', 'min', 'max'], 
        'T2_disp': ['mean', 'min', 'max'],
        'SST_disp': ['mean', 'min', 'max'],
        #Bergeron indexes
        'bergeron_2h': ['mean', 'min', 'max'],
        'bergeron_4h': ['mean', 'min', 'max'],
        'bergeron_6h': ['mean', 'min', 'max'],
        'bergeron_12h': ['mean', 'min', 'max'],
        'bergeron_24h': ['mean', 'min', 'max'],
        #others from list
        'george_index': ['mean', 'min', 'max'],
        'LI_rad': ['mean', 'min', 'max'],
        'z500_rad': ['mean', 'min', 'max'],
        'U850_95': ['mean', 'min', 'max'],
        'U200_95': ['mean', 'min', 'max'],
        'Q850_95': ['mean', 'min', 'max'],
        
        'LCL_95':['mean', 'min', 'max'],
        'LFC_95':['mean', 'min', 'max'],
        'LFC-LCL':['mean', 'min', 'max'],
        
        'mcin_95':['mean', 'min', 'max'],
        'cin_sur': ['mean', 'min', 'max'],

        'DBZ_sfc_500_mean':['mean', 'min', 'max'],

        'rh_95':['mean', 'min', 'max'],
        'updraft_helicity':['mean', 'min', 'max'],

        'INTEGR_VAPOR_TRANSP':['mean', 'min', 'max'],
        'U_VAPOR_TRANS':['mean', 'min', 'max'],
        'V_VAPOR_TRANS':['mean', 'min', 'max'],
     
    }

    # Вычисляем статистику для каждого параметра
    for param, aggregations in param_aggregations.items():
        if param in df.columns and df[param].notna().sum() > 0:
            for agg_func in aggregations:
                if agg_func == 'mean':
                    stats[f'{param}_mean'] = df[param].mean()
                elif agg_func == 'min':
                    stats[f'{param}_min'] = df[param].min()
                elif agg_func == 'max':
                    stats[f'{param}_max'] = df[param].max()

    # Добавляем ВРЕМЯ ЖИЗНИ ВИХРЯ (в часах)
    time_diff = df['time'].max() - df['time'].min()
    stats['lifetime_hours'] = time_diff.total_seconds() / 3600
    coords = list(zip(df['latitude'], df['longitude']))
    # Добавляем КИЛОМЕТРАЖ ВИХРЯ (в ячейках сетки)
    stats['track_length_km'] = df['track_len'].max()
    
    return stats


def process_single_track(track_file, wrf_files, output_dir, data_type='EC'):
    """Обработка одного трека с ПОСЛЕДОВАТЕЛЬНОЙ обработкой часов."""
    try:
        df = pd.read_csv(track_file)


        if data_type == 'LoRes':
            df = df.rename(columns={
                                'datetime': 'time',
                                'x': 'pxc_ind',
                                'y': 'pyc_ind', 
                                'lat': 'latitude',
                                'lon': 'longitude',
                                'rad': 'mean_radius'
                            })
        df['time'] = pd.to_datetime(df['time'])

        track_filename = os.path.basename(track_file)
        
        print(f"    Трек загружен: {len(df)} часов")
        
        # Параметры для расчета
        all_params = [
            'SLP_center', 'SLP_median', 'SLP_delta', 'SLP_diff(cent-med)','SLP_95','SLP_diff(cent-95)',
            
            'U10_mean', 'U500_mean', 'U500_poleward', 'PV_850_mean', 'PV_500_mean', 'wspd_850_mean',
            
            'T2_mean', 'T500_mean', 'T700_mean', 'T850_mean', 'T2_minus_T500_mean', 'T2_minus_T700_mean',
            'TH2_mean', 'TH2_minus_theta_500_mean', 'TH2_minus_theta_700_mean', 'TH2_minus_theta_850_mean',
            
            'SST_mean','theta_e_700_mean', 'theta_e_850_mean', 'TH850',
            
            'SST_minus_T500_mean', 'SST_minus_T700_mean','theta_SST_minus_theta_500_mean', 'theta_SST_minus_theta_700_mean', 
            
            'theta_SST_minus_theta_850_mean','theta_e_SST_minus_theta_e_500_mean', 'theta_e_SST_minus_theta_e_700_mean',
            'theta_e_SST_minus_theta_e_850_mean',
            'MCAO1_500_mean', 'MCAO1_700_mean', 'MCAO2_mean', 
            
            'grad_theta_e_850_mean', 'T2_delta', 'T850_delta', 'T700_delta', 'T500_delta', 'TH500_delta', 'TH700_delta', 'TH850_delta',
            
            
            'PBL_med', 'rel_vor_850_med',

            #trop
            'trop_height',
            'theta_trop_med',
            'delta_theta_trop-theta_sst',   
            'pressure_trop',
            'wspd_trop',

            #custom_params
            'HFX_rad', 'LH_rad', 'mucape_95', 'mcin_95', 
            'helicity_95', 'pw_95', 'pw_sum',
            
            'w_925', 'w_850', 'w_500',
            
            'N_500',
            
            'propagation_speed','differential_wind_vector',
            'vertical_shear_strength','alpha_d','alpha_p','vertical_shear_angle', 'vertical_shear_vector_u','vertical_shear_vector_v',
            'wind_shear_10m_500',
            'T2_disp' ,'SST_disp','TH850_disp',
            
            'RAIN_HOURLY_sum', 'RAIN_HOURLY_95',
            #bergeron_indexes
#             'bergeron_2h', 'bergeron_4h', 'bergeron_6h', 'bergeron_12h','bergeron_24h',
            #others form list
            'george_index','LI_rad','z500_rad','U200_95','Q850_95','LCL_95','LFC_95','LFC-LCL','mcin_95',
            'DBZ_sfc_500_mean',
            'rh_95','updraft_helicity',
            'INTEGR_VAPOR_TRANSP', 'U_VAPOR_TRANS', 'V_VAPOR_TRANS',
        ]
        
        # Создаем пустые колонки
        for param in all_params:
            df[param] = np.nan
        
        # ПОСЛЕДОВАТЕЛЬНАЯ ОБРАБОТКА КАЖДОГО ЧАСА
        # print(f"    Последовательная обработка {len(df)} часов")
        
        successful_hours = 0
        for i in range(len(df)):
            try:
                row = df.iloc[i]
                hour_idx, results = process_single_hour(
                    i,  # index
                    row,  # Series
                    wrf_files,
                    df
                )
                
                if results and len(results) > 0:
                    successful_hours += 1
                    for param, value in results.items():
                        df.at[hour_idx, param] = value
            except Exception as e:
                continue
         # !!!!!!!!!!!РАСЧЕТ ИНДЕКСОВ БЕРЖЕРОНА с учетом реальной широты!!!!!!!!!!!!!!!! 
         # + c наличием необходимых переменных
        bergeron_results = calculate_all_bergeron(df)

        # Записываем результаты
        for param_name, values in bergeron_results.items():
            df[param_name] = values
        
        # ОБЩЕЕ Сохранение
        output_file = os.path.join(output_dir, f"{track_filename}")
        df.to_csv(output_file, index=False)
        
        print(f"    Файл сохранен: {os.path.basename(output_file)}")
        # print(f"    Обработано часов: {successful_hours}/{len(df)}")
        
        return df, output_file
        
    except Exception as e:
        print(f"    Ошибка обработки трека: {e}")
        return pd.DataFrame(), None


def process_single_hour(i, row, wrf_files, storm_data):
    """Обрабатывает один час данных."""
  
    try:
        current_time = row['time']
        if current_time.year == 1979 and current_time.month == 1 and current_time.day == 1:
            if current_time.hour == 0:
                # Для самого первого часа используем накопленные значения
                pass  # Продолжаем, но get_rain_with_previous вернет накопленные осадки
        radius_pixels = float(row['mean_radius'])
        
        if current_time.date() == pd.Timestamp('2019-01-01').date() and current_time.hour >= 2:
            return i, {}
        
        selected_file = find_wrf_file_for_time(current_time, wrf_files)
        
        if not selected_file:
            return i, {}
        
        with Dataset(selected_file) as ds:
            times = ds.variables['Times'][:]
            wrf_times = [parse_wrf_time(t) for t in times]
            
            time_idx = None
            for idx, wrf_time in enumerate(wrf_times):
                if wrf_time == current_time:
                    time_idx = idx
                    break
            

            center_lon_idx = int(row['pxc_ind'])
            center_lat_idx = int(row['pyc_ind'])

            lats = ds.variables['XLAT'][time_idx] if 'XLAT' in ds.variables else None
            lons = ds.variables['XLONG'][time_idx] if 'XLONG' in ds.variables else None

        
            results = {}

            try:
                # 1. SLP (hpa)
                slp = getvar(ds, "slp", timeidx=time_idx)
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                results['SLP_center'] = get_values_in_radius(slp, center_lat_idx, center_lon_idx, 'point')
                results['SLP_median'] = get_values_in_radius(slp, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                results['SLP_delta'] = get_values_in_radius(slp, center_lat_idx, center_lon_idx, radius_pixels, 'delta')
                
                results['SLP_diff(cent-med)'] = results['SLP_median'] - results['SLP_center']
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                # 2. Ветер на 10м (m/s)
                wind_10m = getvar(ds, "uvmet10", timeidx=time_idx)
                wind_speed = np.sqrt(wind_10m[0]**2 + wind_10m[1]**2)
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                results['U10_mean'] = get_values_in_radius(wind_speed, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                # 3. Давление и уровни (hpa)
                pressure = getvar(ds, "pressure", timeidx=time_idx)
                
                # 4. Ветер на 500 гПа (m/s) - ВЫЧИСЛЯЕМ МОДУЛЬ
                u = getvar(ds, "ua", timeidx=time_idx, units="m s-1")
                v = getvar(ds, "va", timeidx=time_idx, units="m s-1")
                
                # Интерполируем
                u_500_interp = interplevel(u, pressure, 500., meta=True)
                v_500_interp = interplevel(v, pressure, 500., meta=True)
  
                u_500_np = u_500_interp.values if hasattr(u_500_interp, "values") else u_500_interp
                v_500_np = v_500_interp.values if hasattr(v_500_interp, "values") else v_500_interp
                
#                 wind_speed_500 = np.sqrt(u_500_np**2 + v_500_np**2)

                wspd = getvar(ds, "wspd_wdir", timeidx=time_idx, units="m s-1")[0]
                wspd_500_interp = interplevel(wspd, pressure, 500., meta=True)
                                        
                results['U500_mean'] = get_values_in_radius(wspd_500_interp, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                results['U500_poleward'] = get_poleward_sector(wspd_500_interp, center_lat_idx, center_lon_idx, radius_pixels, lats, 'median')
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

                # 5. Завихренность на 850 гПа (PVU)
                pot_vorticity = getvar(ds, 'pvo', timeidx=time_idx)  
                pot_vorticity_850 = interplevel(pot_vorticity,     # поле для интерполяции
                     pressure,      # давление
                     850.,   # целевой уровень (гПа)
                     meta=True)  # сохранить метаданные
                
                pot_vorticity_500 = interplevel(pot_vorticity,     # поле для интерполяции
                     pressure,      # давление
                     500.,   # целевой уровень (гПа)
                     meta=True)  # сохранить метаданные
                    
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                results['PV_850_mean'] = get_values_in_radius(pot_vorticity_850, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                results['PV_500_mean'] = get_values_in_radius(pot_vorticity_500, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                # 6. Температуры (K)
                temperature = getvar(ds, 'temp', timeidx=time_idx)

                t_500 = interplevel(temperature,     # поле для интерполяции
                     pressure,      # давление
                     500.,   # целевой уровень (гПа)
                     meta=True)  # сохранить метаданные
                
                t_700 = interplevel(temperature,     # поле для интерполяции
                     pressure,      # давление
                     700.,   # целевой уровень (гПа)
                     meta=True)  # сохранить метаданные
                
                t_850 = interplevel(temperature,     # поле для интерполяции
                     pressure,      # давление
                     850.,   # целевой уровень (гПа)
                     meta=True)  # сохранить метаданные
                    
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                results['T500_mean'] = get_values_in_radius(t_500, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                results['T700_mean'] = get_values_in_radius(t_700, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                results['T850_mean'] = get_values_in_radius(t_850, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                
                results['T500_delta'] = get_values_in_radius(t_500, center_lat_idx, center_lon_idx, radius_pixels, 'delta')
                results['T700_delta'] = get_values_in_radius(t_700, center_lat_idx, center_lon_idx, radius_pixels, 'delta')
                results['T850_delta'] = get_values_in_radius(t_850, center_lat_idx, center_lon_idx, radius_pixels, 'delta')
                
                
          
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                # 7. SST (K)
                sst_data = getvar(ds, "SST", timeidx=time_idx)
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                results['SST_mean'] = get_values_in_radius(sst_data, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                # Разности SST и температур (K)
                sst_minus_t500 = sst_data - t_500
                sst_minus_t700 = sst_data - t_700
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                results['SST_minus_T500_mean'] = get_values_in_radius(sst_minus_t500, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                results['SST_minus_T700_mean'] = get_values_in_radius(sst_minus_t700, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                
                # 7. T2 (K)
                t2_data = getvar(ds, "T2", timeidx=time_idx)
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                results['T2_mean'] = get_values_in_radius(t2_data, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                results['T2_delta'] = get_values_in_radius(t2_data, center_lat_idx, center_lon_idx, radius_pixels, 'delta')
                
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                # Разности T2 и температур (K)
                t2_minus_t500 = t2_data - t_500
                t2_minus_t700 = t2_data - t_700
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                results['T2_minus_T500_mean'] = get_values_in_radius(t2_minus_t500, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                results['T2_minus_T700_mean'] = get_values_in_radius(t2_minus_t700, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                # 8. Потенциальные температуры SST (K)
                '''Получаем данные для MetPy (экв. потенц температуры + потенц sst)'''
                # Конвертируем в units для MetPy функций

                pressure_mp = pressure * units('hPa')
                temperature_mp = temperature * units('kelvin')
                psfc_data = getvar(ds, "PSFC", timeidx=time_idx)/100 #hpa! Для тета Metpy
                psfc_mp = psfc_data * units('hPa')
                sst_mp = sst_data * units('kelvin')

                # для SST (K)
                theta_sst_mp = potential_temperature(psfc_mp, sst_mp)
                theta_sst = np.array(theta_sst_mp.magnitude) if hasattr(theta_sst_mp, 'magnitude') else np.array(theta_sst_mp)
                
  
                # и для уровней
                theta_3d = getvar(ds, 'theta', timeidx=time_idx)

                theta_500 = interplevel(theta_3d,     # поле для интерполяции
                     pressure,      # давление
                     500.,   # целевой уровень (гПа)
                     meta=True)  # сохранить метаданные

                theta_700 = interplevel(theta_3d,     # поле для интерполяции
                     pressure,      # давление
                     700.,   # целевой уровень (гПа)
                     meta=True)  # сохранить метаданные

                theta_850 = interplevel(theta_3d,     # поле для интерполяции
                     pressure,      # давление
                     850.,   # целевой уровень (гПа)
                     meta=True)  # сохранить метаданные

                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                results['theta_SST_minus_theta_500_mean'] = get_values_in_radius(theta_sst - theta_500, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                results['theta_SST_minus_theta_700_mean'] = get_values_in_radius(theta_sst - theta_700, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                results['theta_SST_minus_theta_850_mean'] = get_values_in_radius(theta_sst - theta_850, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                
                results['TH500_delta'] = get_values_in_radius(theta_500, center_lat_idx, center_lon_idx, radius_pixels, 'delta')
                results['TH700_delta'] = get_values_in_radius(theta_700, center_lat_idx, center_lon_idx, radius_pixels, 'delta')
                results['TH850_delta'] = get_values_in_radius(theta_850, center_lat_idx, center_lon_idx, radius_pixels, 'delta')
                
                
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                
                # 8. Потенциальные температуры TH2 (K)
                
                th2 = ds.variables["TH2"][time_idx, :, :]
                
                results['TH2_mean'] = get_values_in_radius(th2, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                
                
                results['TH2_minus_theta_500_mean'] = get_values_in_radius(th2 - theta_500, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                results['TH2_minus_theta_700_mean'] = get_values_in_radius(th2 - theta_700, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                results['TH2_minus_theta_850_mean'] = get_values_in_radius(th2 - theta_850, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                
                
                # 9. MCAO kolstad (K/PA)
                slp_np = slp.values
                mcao_500, mcao_700 = calculate_mcao_kolstad_fast(theta_sst, theta_500, theta_700, slp_np)
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                results['MCAO1_500_mean'] = get_values_in_radius(mcao_500, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                results['MCAO1_700_mean'] = get_values_in_radius(mcao_700, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                # MCAO Bracegirdle (K/PA)
                z = getvar(ds, "z")
                z_700 = interplevel(z,     # поле для интерполяции
                     pressure,      # давление
                     700.,   # целевой уровень (гПа)
                     meta=True)  # сохранить метаданные

                mcao2 = calculate_mcao_bracegirdle_fast(theta_sst, theta_700, z_700)
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                results['MCAO2_mean'] = get_values_in_radius(mcao2, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                # 10. Эквивалентные потенциальные температуры (METPY)
                #берем точку росы для экв. потенц. темератур по всем уровням
                dewpoint_data = getvar(ds, "td", timeidx=time_idx)
                #и на 2м для экв. потенц. температур sst
                dewpoint_sst = getvar(ds, "td2", timeidx=time_idx)

#                 '''ФИЛЬТРУЕМ DEWPOINT'''
#                 # Конвертируем в Kelvin и ограничиваем точку росы температурой (для уровней)
#                 dewpoint_kelvin = dewpoint_data + 273.15
#                 #dewpoint_kelvin = np.minimum(dewpoint_kelvin, temperature_3d)
#                 dewpoint_mp = dewpoint_kelvin * units('kelvin')
                
#                 # Рассчитываем theta_e для уровней
#                 theta_e_mp = equivalent_potential_temperature(pressure_mp, temperature_mp, dewpoint_mp)
#                 theta_e = theta_e_mp.values
                
                
                theta_e = getvar(ds, "theta_e", timeidx=time_idx)
                

                # Получаем theta_e на уровнях
                theta_e_850 = interplevel(theta_e,     # поле для интерполяции
                     pressure,      # давление
                     850.,   # целевой уровень (гПа)
                     meta=True)  # сохранить метаданные
                
                theta_e_700 = interplevel(theta_e,     # поле для интерполяции
                     pressure,      # давление
                     700.,   # целевой уровень (гПа)
                     meta=True)  # сохранить метаданные
                
                theta_e_500 = interplevel(theta_e,     # поле для интерполяции
                     pressure,      # давление
                     500.,   # целевой уровень (гПа)
                     meta=True)  # сохранить метаданные
                
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                results['theta_e_700_mean'] = get_values_in_radius(theta_e_700, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                results['theta_e_850_mean'] = get_values_in_radius(theta_e_850, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                # 11. Theta_e для SST (METPY)
                dewpoint_sst_kelvin = dewpoint_sst + 273.15
            
                dewpoint_sst_mp = dewpoint_sst_kelvin * units('kelvin')
                theta_e_sst_mp = equivalent_potential_temperature(psfc_mp, sst_mp, dewpoint_sst_mp)
                theta_e_sst = theta_e_sst_mp.values
            
                # Рассчитываем разности
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                results['theta_e_SST_minus_theta_e_500_mean'] = get_values_in_radius(
                    theta_e_sst - theta_e_500, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                results['theta_e_SST_minus_theta_e_700_mean'] = get_values_in_radius(
                    theta_e_sst - theta_e_700, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                results['theta_e_SST_minus_theta_e_850_mean'] = get_values_in_radius(
                    theta_e_sst - theta_e_850, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                # 12. Градиент Theta_e на 850 гПа
                theta_e_850_with_units = theta_e_850 * units('kelvin')
                
                dx = ds.DX/1000 * units.kilometer 
                dy = ds.DY/1000 * units.kilometer 
                
#                 dx = 6 * units.kilometer 
#                 dy = 6 * units.kilometer 
                    
                grad_x, grad_y = gradient(theta_e_850_with_units, deltas=[dy, dx])
                grad_theta_e_magnitude = np.sqrt(grad_x**2 + grad_y**2).magnitude
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                results['grad_theta_e_850_mean'] = get_values_in_radius(
                    grad_theta_e_magnitude, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                
                
                # 13. Получаем высоту планетарного пограничного слоя
                pblh = getvar(ds, 'PBLH', timeidx=time_idx)  # уже в метрах
                results['PBL_med'] = get_values_in_radius(
                    pblh, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                
                 # 14. Получаем высоту тропопаузы и считаем в ней штолловские параметры (потенц. темпы)
                trop_height_array = get_tropopause_height_by_2pvu(z, pot_vorticity,min_height=3000)
                # print(f"    Диапазон высот тропопаузы: min={np.nanmin(trop_height_array):.2f} м, max={np.nanmax(trop_height_array):.2f} м")

                trop_height = get_values_in_radius(trop_height_array, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                results['trop_height']=trop_height
                # print(f"    Медиана в радиусе: {trop_height:.2f} м")
                
                theta_trop=interplevel(theta_3d, z, trop_height)
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                results['theta_trop_med'] = get_values_in_radius(  
                    theta_trop, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

                theta_sst_minus_theta_trop=theta_sst-theta_trop
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                results['delta_theta_trop-theta_sst'] = get_values_in_radius(  
                    theta_sst_minus_theta_trop, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                press_trop=interplevel(pressure, z, trop_height)
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                results['pressure_trop'] = get_values_in_radius(  
                     press_trop, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
            
                u_trop_interp= interplevel(u,  z, trop_height, meta=True)
                v_trop_interp = interplevel(v,  z, trop_height, meta=True)
                u_trop_np = u_trop_interp.values if hasattr(u_trop_interp, 'values') else u_trop_interp
                v_trop_np = v_trop_interp.values if hasattr(v_trop_interp, 'values') else v_trop_interp

                wind_speed_trop = np.sqrt(u_trop_np**2 + v_trop_np**2)
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                results['wspd_trop'] = get_poleward_sector(wind_speed_trop, center_lat_idx, center_lon_idx, radius_pixels, lats, 95)
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

                # 14. Relative vorticity 850
                avo = getvar(ds, 'avo', timeidx=time_idx)
                f= getvar(ds, 'F', timeidx=time_idx) 
                # print(f' параметр кориолиса - {f}')
                rv=avo-f
                rv_850 = interplevel(rv,     # поле для интерполяции
                     pressure,      # давление
                     850.,   # целевой уровень (гПа)
                     meta=True)  # сохранить метаданные
                
                results['rel_vor_850_med'] = get_values_in_radius(  
                    rv_850, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                # print(f"медиана RV - {results['rel_vor_850_med']}")
                


                # 0. DELTA SLP
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                results['SLP_95'] = get_values_in_radius(slp, center_lat_idx, center_lon_idx, radius_pixels, 95)
                results['SLP_diff(cent-95)'] = results['SLP_95'] - results['SLP_center']
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                
                # 1. HFX
                hfx = getvar(ds, "HFX", timeidx=time_idx)
                results['HFX_rad'] = get_values_in_radius(hfx, center_lat_idx, center_lon_idx, radius_pixels, 95)
                
                # 2. LH
                lh = getvar(ds, "LH", timeidx=time_idx)
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                results['LH_rad'] = get_values_in_radius(lh, center_lat_idx, center_lon_idx, radius_pixels, 95)
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                
                # 3. MU-CAPE, LCL, LFC, MCIN, + CAPE,CIN
                cape_data = getvar(ds, "cape_2d", timeidx=time_idx,meta=True)
                # print(cape_data)
                try:
                    if hasattr(cape_data, 'shape') and len(cape_data.shape) == 3:
        
                        mu_cape = cape_data[0, :, :]
                    
                        mu_cape_result = get_values_in_radius(mu_cape, center_lat_idx, center_lon_idx, radius_pixels, 95)
                
                        if not np.isnan(mu_cape_result):
                            ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                            results['mucape_95'] = mu_cape_result
                            ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                except:
                    results['mucape_95'] = 0

                try:
                    if hasattr(cape_data, 'shape') and len(cape_data.shape) == 3:
        
                        mcin = cape_data[1, :, :]
                        mcin_result = get_values_in_radius(mcin, center_lat_idx, center_lon_idx, radius_pixels, 95)
                
                        if not np.isnan(mcin_result):
                            ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                            results['mcin_95'] = mcin_result
                            ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                except:
                    results['mcin_95'] = 0


                try:
                    if hasattr(cape_data, 'shape') and len(cape_data.shape) == 3:
                        LCL = cape_data[2, :, :]
                        LCL_result = get_values_in_radius(LCL, center_lat_idx, center_lon_idx, radius_pixels, 95)

                        if not np.isnan(LCL_result):
                            ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                            results['LCL_95'] = LCL_result
                            ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                except:
                    results['LCL_95'] = np.nan

                try:
                    if hasattr(cape_data, 'shape') and len(cape_data.shape) == 3:
                     
                        LFC = cape_data[3, :, :]
                        LFC_result = get_values_in_radius(LFC, center_lat_idx, center_lon_idx, radius_pixels, 95)
                      
                        if not np.isnan(LFC_result):
                            ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                            results['LFC_95'] = LFC_result
                            ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                except:
                    results['LFC_95'] = np.nan

#                 '+ 3D CAPE and CIN'
#                 try:
#                     cape3d=getvar(ds, "cape_3d", timeidx=time_idx, meta=True)
#                     cape, cin = cape3d[0,:,:], cape3d[1,:,:]
#                     # surface vals
#                     cape_sur=cape[0,:,:]
#                     cin_sur=cin[0,:,:]
#                     results['cape_sur'] = get_values_in_radius(cape_sur, center_lat_idx, center_lon_idx, radius_pixels, 95)
#                     results['cin_sur'] = get_values_in_radius(cin_sur, center_lat_idx, center_lon_idx, radius_pixels, 95)
#                 except:
#                     results['cape_sur'] = np.nan
#                     results['cin_sur'] = np.nan

                

                # 4. HELICITY
                try:
                    srhel = getvar(ds, "helicity", timeidx=time_idx)
                    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                    results['helicity_95'] = get_values_in_radius(srhel, center_lat_idx, center_lon_idx, radius_pixels, 95)
                    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                except:
                    results['helicity_95'] = 0
                
                # 5. Precipitable Water
                pw = getvar(ds, "pw", timeidx=time_idx)
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                results['pw_95'] = get_values_in_radius(pw, center_lat_idx, center_lon_idx, radius_pixels, 95)
                results['pw_sum'] = get_values_in_radius(pw, center_lat_idx, center_lon_idx, radius_pixels, 'sum')
                
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                
                # 6. VERTICAL VELOCITY
                w = getvar(ds, "wa", timeidx=time_idx)
                
                w_850 = interplevel(w,     # поле для интерполяции
                     pressure,      # давление
                     850.,   # целевой уровень (гПа)
                     meta=True)  # сохранить метаданные
                
                w_500 = interplevel(w, pressure, 500., meta=True)
                w_925 = interplevel(w, pressure, 925., meta=True)
                
                
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                results['w_850'] = get_values_in_radius(w_850, center_lat_idx, center_lon_idx, radius_pixels, 95)
                results['w_500'] = get_values_in_radius(w_500, center_lat_idx, center_lon_idx, radius_pixels, 95)
                results['w_925'] = get_values_in_radius(w_925, center_lat_idx, center_lon_idx, radius_pixels, 95)
                
                
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                
#                 # 7. N (500-925 HPA)
#                 theta_3d = getvar(ds, 'theta', timeidx=time_idx)
#                 pres_3d = getvar(ds, "pressure", timeidx=time_idx)
            
                
#                 theta_500 = interplevel(theta_3d, pres_3d, 500.)
#                 theta_925 = interplevel(theta_3d, pres_3d, 925.)
#                 z_500 = interplevel(z, pres_3d, 500.)
#                 z_925 = interplevel(z, pres_3d, 925.)
                
#                 g = 9.81
#                 theta_mean = (theta_500 + theta_925) / 2.0
#                 delta_theta = theta_500 - theta_925
#                 delta_z = z_500 - z_925
#                 delta_z = np.where(delta_z == 0, 1e-10, delta_z)
                
#                 N2 = (g / theta_mean) * (delta_theta / delta_z)

#                 '''ПОДРОБНЕЕ ПОЧЕКАТЬ ПРО ВБ ЧАСТОТУ'''
#                 N = np.sqrt(np.maximum(N2, 0))
#                 ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#                 results['N_mean'] = get_values_in_radius(N, center_lat_idx, center_lon_idx, radius_pixels, 'median')
                
                
                
                z = getvar(ds, "z", timeidx=time_idx) # msl (boolean): Set to False to return AGL values. True is for MSL.

                
                N = brunt_vaisala_frequency(z * units.meter, theta_3d * units.kelvin)
                N_500 = interplevel(N, pressure, 500., meta=True)
#                 N_500 = N_500.values if hasattr(N_500, 'values') else N_500
                
                results['N_500'] = get_values_in_radius(N_500, center_lat_idx, center_lon_idx, radius_pixels, 'median', 'N_500')
                
                
                


                # ВЕТРОВОЙ СДВИГ ПО МАГНИТУДЕ СКОРОСТИ МЕЖДУ 10 м И 500 ГПА
                try:
                    # 10 м ветер
                    wind10m = getvar(ds, "uvmet10", timeidx=time_idx)  # [2, y, x], м/с
                    u10m = wind10m[0, :, :]
                    v10m = wind10m[1, :, :]
                    speed10m = np.sqrt(u10m**2 + v10m**2)
                    
                   

                    speed500 = np.sqrt(u_500_np**2 + v_500_np**2)
                
                    # модуль разности скоростей
                    dV_speed_10m_500 = np.abs(speed500 - speed10m)
                
                    # усреднение в радиусе вокруг центра циклона
                    wind_shear_10m_500 = get_values_in_radius(
                        dV_speed_10m_500,
                        center_lat_idx,
                        center_lon_idx,
                        radius_pixels,
                        "median",   # или "95" по вкусу
                        'wind_shear_10m_500'
                    )
                
                    results["wind_shear_10m_500"] = wind_shear_10m_500
                
                except Exception as e:
                    results["wind_shear_10m_500"] = np.nan
                
                # 8. Вертикальный сдвиг ветра

                # на 500 гпа ветер уже считался, добавим значения на 925 гпа!
                u_925 = interplevel(u, pressure, 925., meta=True)
                v_925 = interplevel(v, pressure, 925., meta=True)

                u_925_np = u_925.values if hasattr(u_925, 'values') else u_925
                v_925_np = v_925.values if hasattr(v_925, 'values') else v_925
                
                # Разностные компоненты
                du = u_500_np - u_925_np  # Δu (реальная часть)
                dv = v_500_np - v_925_np  # Δv (мнимая часть)
                # Векторное усреднение в радиусе
                mean_du, mean_dv, mean_alpha_d, mean_dV_magnitude = calculate_vector_mean_in_radius(
                    du, dv, center_lat_idx, center_lon_idx, radius_pixels
                )

                # 10. Вектор вертикального сдвига
                if not np.isnan(mean_dV_magnitude):
                    # Средний Δz в радиусе
                    z_500 = interplevel(z, pressure, 500.)
                    z_925 = interplevel(z, pressure, 925.)
                
                    delta_z = z_500 - z_925
                    delta_z = np.where(delta_z == 0, 1e-10, delta_z)
                    delta_z_mean = get_values_in_radius(delta_z, center_lat_idx, center_lon_idx, radius_pixels, 'median', 'delta_z_mean')
                    # 10. Магнитуда вертикального сдвига
                    if delta_z_mean != 0 and not np.isnan(delta_z_mean):
                        shear_magnitude = mean_dV_magnitude / abs(delta_z_mean)
                        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                        results['vertical_shear_strength'] = shear_magnitude
                        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                        results['differential_wind_vector'] = mean_dV_magnitude
                
                # 9 . Угол вертикального сдвига ветра alpha_d
                if not np.isnan(mean_alpha_d):
                    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                    results['alpha_d'] = mean_alpha_d
                    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

                
                # 11. НАПРАВЛЕНИЯ РАСПРОСТРАНЕНИЯ ЦИКЛОНА alpha_p + 12. Скорость распространенифя циклона
                try:
                    current_idx = i
                    
                    if current_idx > 0:
                        prev_row = storm_data.iloc[current_idx - 1]
                        curr_row = storm_data.iloc[current_idx]
                        
                        # ИСПРАВЛЕННЫЙ РАСЧЁТ НАПРАВЛЕНИЯ
                        dlat = curr_row['latitude'] - prev_row['latitude']
                        dlon = curr_row['longitude'] - prev_row['longitude']
                        
                        mean_lat = (curr_row['latitude'] + prev_row['latitude']) / 2
                        dy = dlat * 111.0  # км на градус широты
                        dx = dlon * 111.0 * np.cos(np.radians(mean_lat))  # км на градус долготы
                        distance_km = np.sqrt(dx**2 + dy**2)
                        
                        # arctan2(dy, dx) - dy: северная компонента, dx: восточная
                        alpha_p = np.degrees(np.arctan2(dy, dx))
                        alpha_p = alpha_p % 360
                        
                        dt_hours = 1.0
                        propagation_speed = distance_km / dt_hours
                        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                        results['alpha_p'] = alpha_p
                        results['propagation_speed'] = propagation_speed
                        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                        # Расчет угла вертикального сдвиза α = [αd - αp] (mod 360°)
                        if (not np.isnan(results.get('alpha_d')) and not np.isnan(alpha_p) and 
                            not np.isnan(results.get('vertical_shear_strength'))):
                            
                            alpha_shear = (results['alpha_d'] - alpha_p) % 360
                            alpha_shear = np.where(alpha_shear > 180, alpha_shear - 360, alpha_shear)
                            ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                            results['vertical_shear_angle'] = alpha_shear
                            ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

                            # Вектор сдвига по формуле (4)
                            alpha_rad = np.radians(alpha_shear)
                            shear_mag = results['vertical_shear_strength']
                            
                            shear_vector_u = shear_mag * np.cos(alpha_rad)  # вдоль движения
                            shear_vector_v = shear_mag * np.sin(alpha_rad)  # поперёк движения
                            
                            ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                            results['vertical_shear_vector_u'] = shear_vector_u
                            results['vertical_shear_vector_v'] = shear_vector_v
                            ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                except Exception as e:

                    pass


                # 12. Загружаем T2
                results['T2_disp'] = get_values_in_radius(t2_data, center_lat_idx, center_lon_idx, radius_pixels, 'std', 'T2_disp')
                
                # 13. T850 + его дисперсия

                results['TH850'] = get_values_in_radius(theta_850, center_lat_idx, center_lon_idx, radius_pixels, 'median', 'TH850')
                results['TH850_disp'] = get_values_in_radius(theta_850, center_lat_idx, center_lon_idx, radius_pixels, 'std', 'TH850_disp')
                
                
                # 14. SST дисперсия
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                results['SST_disp'] = get_values_in_radius(sst_data, center_lat_idx, center_lon_idx, radius_pixels, 'std', 'SST_disp')
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

                    
                # 15. '''George k - index'''
                'K = (T850 - T500) + Td850 - (T700 - Td700)'
            
                td_850 = interplevel(dewpoint_data, pressure, 850., meta=True) 
                td_700 = interplevel(dewpoint_data, pressure, 700., meta=True)

                t_500c=t_500 -  273.15
                t_700c=t_700 -  273.15
                t_850c=t_850 -  273.15

                k_index = (t_850c - t_500c) + td_850 - (t_700c - td_700)
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                results['george_index'] = get_values_in_radius(k_index, center_lat_idx, center_lon_idx, radius_pixels, 95, 'george_index')
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

                # 16. '''Lifted Index (LI)'''
                'LI = T500(окружающий воздух) --- T500(поднятая частица)     -95 процентиль-' 
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                results['LI_rad']=calculate_li(center_lat_idx,center_lon_idx,t_500,t2_data,pressure,radius_pixels,dewpoint_sst)
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                # 17.'''geopotential height 500hpa'''

                z_500 = interplevel(z,     # поле для интерполяции
                     pressure,      # давление
                     500.,   # целевой уровень (гПа)
                     meta=True)  # сохранить метаданные
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                results['z500_rad']=get_values_in_radius(z_500, center_lat_idx, center_lon_idx, radius_pixels, 95, 'z500_rad')
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                # 18. Ветер на 850 гПа (m/s) - ВЫЧИСЛЯЕМ МОДУЛЬ

                # # Интерполируем
                u_850_interp = interplevel(u, pressure, 850., meta=True)
                v_850_interp = interplevel(v, pressure, 850., meta=True)
                u_850_np = u_850_interp.values 
                v_850_np = v_850_interp.values 
  
                wind_speed_850 = np.sqrt(u_850_np**2 + v_850_np**2)
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                results['wspd_850_mean'] = get_values_in_radius(wind_speed_850, center_lat_idx, center_lon_idx, radius_pixels, 'median', 'wspd_850_mean')
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                # 19. Ветер на 200 гПа (m/s) - ВЫЧИСЛЯЕМ МОДУЛЬ
    
                # # Интерполируем
                u_200_interp= interplevel(u, pressure, 200., meta=True)
                v_200_interp = interplevel(v, pressure, 200., meta=True)
                u_200_np = u_200_interp.values 
                v_200_np = v_200_interp.values 
  
                wind_speed_200 = np.sqrt(u_200_np**2 + v_200_np**2)
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                results['U200_95'] = get_values_in_radius(wind_speed_200, center_lat_idx, center_lon_idx, radius_pixels, 95, 'U200_95')
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                # # 18.'''accumulated surface precipitation =RAINC + RAINNC'''
                rain_hourly = get_rain_with_previous(
                    ds, time_idx, current_time, wrf_files
                )

                if rain_hourly is not None:
                    results['RAIN_HOURLY_95'] = get_values_in_radius(
                        rain_hourly, center_lat_idx, center_lon_idx, radius_pixels, 95, 'RAIN_HOURLY_95'
                    )
                    results['RAIN_HOURLY_sum'] = get_values_in_radius(
                        rain_hourly, center_lat_idx, center_lon_idx, radius_pixels, 'sum', 'RAIN_HOURLY_sum',
                    )
                else:
                    results['RAIN_HOURLY_95'] = np.nan
                    results['RAIN_HOURLY_sum'] = np.nan
                    

                # print (f'rain_total {rain_hourly.values} \n rain_shape {rain_hourly.shape} \n rain_dim {rain_hourly.ndim}')

    
                 # 20. МАССОВАЯ ДОЛЯ ВОДЯНОГО ПАРА
                q=getvar(ds, 'QVAPOR', timeidx=time_idx)
                q_850=interplevel(q, pressure, 850., meta=True)
                results['Q850_95'] = get_values_in_radius(q_850, center_lat_idx, center_lon_idx, radius_pixels, 95, 'Q850_95')
       
                # 21. LCL-LFK
                DIFF_LFC_LCL=LCL_result-LFC_result
                results['LFC-LCL'] = DIFF_LFC_LCL
                # 22. DBZ mean: sfc - 500 hpa
       
                dbz= getvar(ds, 'dbz', timeidx=0)
                mask_500_dbz = pressure >= 500.0
                dbz_masked = np.where(mask_500_dbz, dbz, np.nan)
                dbz_vert_avg = np.nanmean(dbz_masked, axis=0)
                # print(dbz_vert_avg)
                results['DBZ_sfc_500_mean'] = get_values_in_radius(dbz_vert_avg, center_lat_idx, center_lon_idx, radius_pixels, 95, 'DBZ_sfc_500_mean')
          
                #23. RH mean: sfc - 850 hpa
                rh=getvar(ds, 'rh', timeidx=time_idx)
                mask_850_rh = pressure >= 850.0
                rh_masked = np.where(mask_850_rh, rh, np.nan)
                rh_vert_avg = np.nanmean(rh_masked, axis=0)
                # print(rh_vert_avg)
                results['rh_95'] = get_values_in_radius(rh_vert_avg, center_lat_idx, center_lon_idx, radius_pixels, 95, 'rh_95')

                #24. Updraft helicity
                try:
                    up_hel=getvar(ds,'updraft_helicity', timeidx=time_idx)
                    results['updraft_helicity'] = get_values_in_radius(up_hel, center_lat_idx, center_lon_idx, radius_pixels, 95, 'updraft_helicity')
                except: 
                    results['updraft_helicity'] = np.nan

                #25. INTEGRATED VAPOR TRANSPORT (FULL,U,V)
                ivt=calculate_ivt(ds, time_idx, bottom_pressure=1000, top_pressure=300,type='integral')
                ivt_U=calculate_ivt(ds, time_idx, bottom_pressure=1000, top_pressure=300,type='U_dir')
                ivt_V=calculate_ivt(ds, time_idx, bottom_pressure=1000, top_pressure=300,type='V_dir')

                results['INTEGR_VAPOR_TRANSP'] = get_values_in_radius(ivt, center_lat_idx, center_lon_idx, radius_pixels, 95, 'INTEGR_VAPOR_TRANSP')
                results['U_VAPOR_TRANS'] = get_values_in_radius(ivt_U, center_lat_idx, center_lon_idx, radius_pixels, 95, 'U_VAPOR_TRANS')
                results['V_VAPOR_TRANS'] = get_values_in_radius(ivt_V, center_lat_idx, center_lon_idx, radius_pixels, 95, 'V_VAPOR_TRANS')
     
            except Exception as e:
                print(f" Ошибка расчета параметров для часа {i}: {e}")
            return i, results
            
    except Exception as e:
        print(f"Ошибка обработки часа {i}: {str(e)[:200]}")
        traceback.print_exc()
        return i, {}
    
