from matplotlib import pyplot as plt
import pandas as pd 
import numpy as np
import math
import xarray as xr
from numpy import linalg as LA

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import sys
import os

import cmaps

from scipy.ndimage import gaussian_filter


def compute_S_A_2d(du_dx, du_dy, dv_dx, dv_dy):
    """
    Расчет 2D тензора скоростей деформации S и тензора завихренности A
    
    Параметры:
        du_dx, du_dy, dv_dx, dv_dy: 4D массивы [time, level, lat, lon] 
                                    компонент тензора градиента скорости
    
    Возвращает:
        S, A: 2D тензоры деформации и завихренности [2, 2, time, level, lat, lon]
    """
    s12 = 0.5 * (du_dy + dv_dx)
    
    S = np.array([
        [du_dx, s12],
        [s12, dv_dy]
    ])
    
    a12 = 0.5 * (du_dy - dv_dx)
    
    diag_shape = du_dx.shape
    diag_0 = np.zeros(diag_shape)
    
    A = np.array([
        [diag_0, a12],
        [-a12, diag_0]
    ])
    
    return S, A

def compute_Q_2d(du_dx, du_dy, dv_dx, dv_dy, S=None, A=None, normalize=True):
    """
    Расчет Q-критерия для 2D случая
    
    Параметры:
        du_dx, du_dy, dv_dx, dv_dy: 4D массивы [time, level, lat, lon]
                                    компонент тензора градиента скорости
        S, A: опционально, тензоры деформации и завихренности (если уже вычислены)
        normalize: флаг нормализации (логарифмирование)
    
    Возвращает:
        Q: массив Q-критерия [time, level, lat, lon]
    """
    if S is None or A is None:
        S, A = compute_S_A_2d(du_dx, du_dy, dv_dx, dv_dy)
    
    norm_S = np.sqrt(np.sum(S**2, axis=(0, 1)))
    norm_A = np.sqrt(np.sum(A**2, axis=(0, 1)))
    
    Q = 0.5 * (norm_A**2 - norm_S**2)
    
    if normalize:
        Q = crit_log(Q)
    
    return Q

def compute_delta_2d(du_dx, du_dy, dv_dx, dv_dy, S=None, A=None, normalize=True):
    """
    Расчет delta-критерия для 2D случая
    
    Параметры:
        du_dx, du_dy, dv_dx, dv_dy: 4D массивы [time, level, lat, lon]
                                    компонент тензора градиента скорости
        normalize: флаг нормализации (логарифмирование)
    
    Возвращает:
        delta: массив delta-критерия [time, level, lat, lon]
    """
    # R = ∂u/∂x * ∂v/∂y - ∂v/∂x * ∂u/∂y (детерминант 2x2)
    R = du_dx * dv_dy - dv_dx * du_dy
    
    if S is None or A is None:
        S, A = compute_S_A_2d(du_dx, du_dy, dv_dx, dv_dy)
    Q = compute_Q_2d(du_dx, du_dy, dv_dx, dv_dy, S, A, normalize=False)
    
    delta = (Q / 3)**3 + (0.5 * R)**2
    
    if normalize:
        delta = crit_log(delta)
    
    return delta

def compute_lambda2_2d(du_dx, du_dy, dv_dx, dv_dy, S=None, A=None, normalize=True):
    """
    Расчет lambda2-критерия для 2D случая
    
    Параметры:
        du_dx, du_dy, dv_dx, dv_dy: 4D массивы [time, level, lat, lon]
                                    компонент тензора градиента скорости
        S, A: опционально, тензоры деформации и завихренности (если уже вычислены)
        normalize: флаг нормализации (логарифмирование)
    
    Возвращает:
        lambda2: массив lambda2-критерия [time, level, lat, lon]
    """
    if S is None or A is None:
        S, A = compute_S_A_2d(du_dx, du_dy, dv_dx, dv_dy)
    
    # Перестановка осей для матричного умножения
    S_perm = S.transpose(2, 3, 4, 5, 0, 1)
    A_perm = A.transpose(2, 3, 4, 5, 0, 1)
    
    # S^2 + A^2
    S_sq = np.matmul(S_perm, S_perm)
    A_sq = np.matmul(A_perm, A_perm)
    SA = S_sq + A_sq
    
    # Для 2D матрицы второе собственное значение
    shape = SA.shape[:-2]
    lambda2 = np.zeros(shape)
    
    for idx in np.ndindex(shape):
        if np.isnan(SA[idx + (1, 1)]):
            lambda2[idx] = np.nan
        else:
            eigenvals = np.linalg.eigvalsh(SA[idx])
            lambda2[idx] = eigenvals[1]
    
    lambda2 = -lambda2
    
    if normalize:
        lambda2 = crit_log(lambda2)
    
    return lambda2

def regular_grid_gradient(field, dx, dy):
    """
    Расчет градиента для 4D массива [time, level, lat, lon]
    
    Параметры:
        field: 4D массив компоненты ветра (u или v)
        dx: 2D массив расстояний по долготе [lat, lon] в метрах
        dy: скалярное расстояние по широте в метрах
    
    Возвращает:
        df_dx, df_dy: градиенты по долготе и широте
    """
    # Градиент по долготе (x-направление)
    df_dx = np.gradient(field, 1., axis=-1)/ dx[None,None,:,None]
    
    # Градиент по широте (y-направление)
    df_dy = np.gradient(field, dy, axis=-2) 
    
    return df_dx, df_dy


def compute_omega(du_dx, du_dy, dv_dx, dv_dy):
    """
    Расчет вертикальной завихренности (omega_z)
    
    Параметры:
        du_dx: ∂u/∂x - градиент u по долготе
        du_dy: ∂u/∂y - градиент u по широте
        dv_dx: ∂v/∂x - градиент v по долготе
        dv_dy: ∂v/∂y - градиент v по широте
    
    Возвращает:
        omega_z: вертикальная компонента завихренности
    """
    omega_z = dv_dx - du_dy
    return omega_z


def swirling_strength(du_dlon, du_dlat, dv_dlon, dv_dlat):
    """
    Векторизованный расчет λ_ci (swirling strength)
    
    Параметры:
        du_dlon: ∂u/∂x - градиент u по долготе
        du_dlat: ∂u/∂y - градиент u по широте
        dv_dlon: ∂v/∂x - градиент v по долготе
        dv_dlat: ∂v/∂y - градиент v по широте
    
    Возвращает:
        lambda_ci: swirling strength (мнимая часть собственных значений)
                   Невихревые точки заменяются на NaN
    """
    a = du_dlon  # ∂u/∂x
    b = du_dlat  # ∂u/∂y
    c = dv_dlon  # ∂v/∂x
    d = dv_dlat  # ∂v/∂y
    
    # Характеристическое уравнение: λ² - (a+d)λ + (ad-bc) = 0
    trace = a + d
    det = a*d - b*c
    
    # Дискриминант (комплексный, если trace² < 4det)
    discriminant = trace**2 - 4*det
    
    # Swirling strength - мнимая часть собственных значений
    lambda_ci = np.sqrt(np.maximum(-discriminant, 0)) / 2
    # Замена невихревых точек на NaN
    lambda_ci[lambda_ci <= 0] = np.nan
    
    return lambda_ci


def compute_rortex_2d(sw_str_2d, omega_2d):
    """
    Расчет 2D Rortex-критерия
    
    Параметры:
        sw_str_2d: swirling strength (λ_ci)
        omega_2d: вертикальная завихренность (ω_z)
    
    Возвращает:
        R_2d: Rortex критерий (положительные значения - циклон,
              отрицательные - антициклон)
    """
    R_2d = (1 - np.sqrt(1 - 4*sw_str_2d*sw_str_2d/(omega_2d*omega_2d))) * omega_2d
    return R_2d


def unification(ds):
    """
    Добавление атрибутов R2D
    
    Параметры:
        ds: xarray Dataset для обработки
    
    Возвращает:
        ds: xarray Dataset с обновленными атрибутами
    """
    ds[crit_name].attrs['description'] = 'Rortex criterion 2D'
    ds[crit_name].attrs['long_name'] = 'Rortex 2D'
    
    ds.attrs = {}
    
    return ds


import numpy as np
import xarray as xr
import os
from scipy.ndimage import gaussian_filter


def process_single_file(ncfile, data_type, smooth=False, sigma=0):
    """
    Обработка одного NetCDF файла для расчета Rortex 2D критерия
    
    Параметры:
        ncfile: путь к NetCDF файлу
        data_type: тип данных (строка для именования выходного файла)
        smooth: флаг применения гауссового сглаживания (по умолчанию False)
        sigma: параметр сигмы для гауссового сглаживания (по умолчанию 0)
    
    Возвращает:
        None (сохраняет результат в NetCDF файл)
    """
    # Открытие датасета
    ds = xr.open_dataset(ncfile)

    if data_type == 'ALT':
        ds = ds.expand_dims(depth=[0], axis=1)
    
    print(f'dims: {ds.dims}')
    
    # Расчет расстояния по долготе в метрах (зависит от широты)
    dx = dlon * DEG_TO_M * np.cos(np.radians(ds.latitude.values))
    
    # Определение компонент скорости (на уровне высоты)
    # ВАЖНО: код принимает скорости 4D (уровень высоты - отдельный dim, не редуцирован)
    u = ds[params[0]]
    v = ds[params[1]]
    
    # Применение гауссового сглаживания с сигмой
    sigma_2d = (0, 0, sigma, sigma)
    
    if smooth:
        u_smooth = gaussian_filter(u.values, sigma=sigma_2d)
        v_smooth = gaussian_filter(v.values, sigma=sigma_2d)
    else:
        u_smooth = u
        v_smooth = v
    
    print('smoothing done')
    
    # Вычисление градиентов ветра
    du_dx, du_dy = regular_grid_gradient(u_smooth, dx, dy)
    dv_dx, dv_dy = regular_grid_gradient(v_smooth, dx, dy)
    print('grads done')

    # 4. Вычисляем тензоры S и A (для последующих критериев)
    S, A = compute_S_A_2d(du_dx, du_dy, dv_dx, dv_dy)
    
    # 5. Вычисляем Q-критерий
    Q = compute_Q_2d(du_dx, du_dy, dv_dx, dv_dy, S, A, normalize=False)
    print('Q done')
    
    # 6. Вычисляем delta-критерий
    delta = compute_delta_2d(du_dx, du_dy, dv_dx, dv_dy, S, A, normalize=False)
    print('delta done')
    
    # 7. Вычисляем lambda2-критерий
    lambda2 = compute_lambda2_2d(du_dx, du_dy, dv_dx, dv_dy, S, A, normalize=False)
    print('lambda2 done')
    
    # Расчет завихренности
    omega_2d = compute_omega(du_dx, du_dy, dv_dx, dv_dy)
    print('omega done')
    
    # Расчет swirling strength и Rortex 2D
    sw_str_2d = swirling_strength(du_dx, du_dy, dv_dx, dv_dy)
    print('swirling strength done')
    
    r2d = compute_rortex_2d(sw_str_2d, omega_2d)
    print('rortex done')

    # Создание копии датасета с компонентами скорости
    ds_smooth = ds[[params[0], params[1]]].copy()

    ds_smooth['omega_2d'] = ({time_name: len(ds[time_name]), 
                             level_name: len(ds[level_name]), 
                             y_name: len(ds[y_name]), 
                             x_name: len(ds[x_name])}, 
                            omega_2d.astype(np.float32))
    
    # Добавление рассчитанного критерия в датасет
    ds_smooth[name_crit] = ({
        time_name: len(ds[time_name]),
        level_name: len(ds[level_name]),
        y_name: len(ds[y_name]),
        x_name: len(ds[x_name])
    }, r2d.astype(np.float32))

    # Добавляем в ds_smooth только положительные значения
    ds_smooth['lambda2'] = ({time_name: len(ds[time_name]), 
                             level_name: len(ds[level_name]), 
                             y_name: len(ds[y_name]), 
                             x_name: len(ds[x_name])}, 
                            np.where(lambda2 > 0, lambda2, np.nan).astype(np.float32))
    
    ds_smooth['delta'] = ({time_name: len(ds[time_name]), 
                           level_name: len(ds[level_name]), 
                           y_name: len(ds[y_name]), 
                           x_name: len(ds[x_name])}, 
                          np.where(delta > 0, delta, np.nan).astype(np.float32))
    
    ds_smooth['Q'] = ({time_name: len(ds[time_name]), 
                       level_name: len(ds[level_name]), 
                       y_name: len(ds[y_name]), 
                       x_name: len(ds[x_name])}, 
                      np.where(Q > 0, Q, np.nan).astype(np.float32))

    print('Saving>>>')
    
    # Создание директории для сохранения результатов
    folder = f'{path_dir_data}/ocean_eddies/'
    if not os.path.exists(f'{folder}'):
        os.makedirs(f'{folder}')

    # Сохранение результата в NetCDF файл
    ds_smooth.to_netcdf(f'{folder}/sigma_{sigma}_{name_crit}_{data_type}.nc', mode='w')
    print(f'Result saved to: {folder}/sigma_{sigma}_{name_crit}_{data_type}.nc')


if __name__ == "__main__":
    """
    Основная точка входа для запуска скрипта
    """
    data_type = 'GLORYS'

    data_type = 'ALT'

    
    path_init = f'/storage/thalassa/users/vkoshkina'
    path_dir_data = f'{path_init}/data'

    if data_type == 'GLORYS':
        ncfile = f'{path_dir_data}/ocean_eddies/BarKara_glorys_aug2023.nc'
        params = ['uo', 'vo']
    else:
        ncfile = f'{path_dir_data}/ocean_eddies/BarKara_altimetry_aug2023.nc'
        params = ['ugos', 'vgos']
    
    name_crit = 'R2D'
    level_name = 'depth'
    time_name = 'time'
    time_unit = 'time'
    x_name = 'longitude'
    y_name = 'latitude'
    
    
    sigma = 0
    smooth = False
    
    # sigma = 2
    # smooth = True
    
    # Константы
    DEG_TO_M = 111 * 1000  # 1 градус ~ 111 км
    dlon = 1/12  # шаг по долготе
    dlat = 1/12  # шаг по широте
    
    # Расстояния между точками сетки (в метрах)
    dy = dlat * DEG_TO_M  # постоянное по широте

    process_single_file(ncfile, data_type, smooth=False)

