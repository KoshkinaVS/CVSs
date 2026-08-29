import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.patches as mpatches
import os
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from tqdm import tqdm

import huracanpy



postfix = '_range_1_5_18h_12h'


# Данные для конкретного ТЦ

year = 2003
id_IBTrACS = '2003286N09323'


year = 2018
id_IBTrACS = '2018186N10325'

year = 2011
id_IBTrACS = '2011266N08343' 

year = 2001
id_IBTrACS = '2001245N13326' 



path_init = f'/storage/thalassa/users/vkoshkina'
path_dir_data = f'{path_init}/data/'

base_folder = f'{path_dir_data}/TempestExtremes/ERA5/R2D_ERA5_NA_for_TC_500hPa_sigma_2/Tracks_R2D_txt_files'
base_folder = f'{base_folder}{postfix}'
years = np.arange(year, year+1)

ref_path_dir = f'{path_dir_data}/TC_tracks/IBTrACS'
ibtracs_df = pd.read_csv(f'{ref_path_dir}/ibtracs.NA.list.v04r01.csv', parse_dates=['ISO_TIME'],
                        skiprows=[1],  # Пропускаем вторую строку (индекс 1)
)

min_overlap = 4

all_tracks_TC = huracanpy.load(
    source="ibtracs",
    # ibtracs_subset="NA",
)
# all_tracks_TC = all_tracks_TC.where(all_tracks_TC.time.dt.year < 2019, drop=True)

# Фильтруем по бассейну "NA" (North Atlantic)
na_tracks = all_tracks_TC.where(all_tracks_TC.basin == "NA", drop=True)
na_tracks.track_id.hrcn.nunique()

na_tracks_filtered = na_tracks.where(
    (na_tracks.lon > -110) & (na_tracks.lon < 15) & 
    (na_tracks.lat > 0) & (na_tracks.lat < 71),
    drop=True
)



# Словарь для хранения треков по годам
tracks_by_year = {}
all_tracks = []


for year in tqdm(years, desc="Loading years"):
    try:
        # Загружаем январь
        filename = f'{base_folder}/ERA5_TC_tracks_{year}.txt'
        tracks_year = huracanpy.load(
            filename,
            source="tempestextremes",
            variable_names=['rad', 'r2d',],
        )
        
        
        # Сохраняем
        tracks_by_year[year] = tracks_year
        all_tracks.append(tracks_year)
        
    except FileNotFoundError:
        print(f"Year {year} not found, skipping...")
        continue

# Объединяем все годы
tracks_all_years = huracanpy.concat_tracks(all_tracks)

# Информация о загруженных данных
print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print(f"Years loaded: {list(tracks_by_year.keys())}")
print(f"Total track points: {len(tracks_all_years)}")
print(f"Time range: {tracks_all_years.time.min().values} to {tracks_all_years.time.max().values}")
print(f"Number of unique storms: {tracks_all_years.track_id.hrcn.nunique()}")


matches_TE_NA = huracanpy.assess.match([na_tracks_filtered, tracks_all_years], names=['IBTrACS', 'TE_NA'], 
                                 max_dist=200, 
                                 mean_dist=120, 
                                 min_overlap=min_overlap, 
                                 # consecutive_overlap=True,
                                 tracks1_is_ref=True)



# Создаем фигуру с двумя сабплотами
fig = plt.figure(figsize=(12, 6), dpi=150)

# Левый сабплот - общая карта
ax1 = fig.add_subplot(121, projection=ccrs.Stereographic(central_latitude=45.0, central_longitude=-45))
ax1.set_global()

# Правый сабплот - детальный вид
ax2 = fig.add_subplot(122, projection=ccrs.Stereographic(central_latitude=45.0, central_longitude=-45))


def plot_r2d(ax, ds, our_time, our_level, name):
    ax.set_global()
    gl = ax.gridlines(draw_labels=True,
                 linewidth=1, color='grey', alpha=0.7, linestyle='--')
    
    gl.xlabels_top = False
    gl.ylabels_right = False

    vmax = 0.5*np.nanmax(np.abs(ds['R2D'][our_time]))
    ax.contourf(ds['longitude'], ds['latitude'], 
                ds['R2D'][our_time,our_level],
                # lambda_ci[our_time,0],
                # c='grey',
                vmin=-vmax,
                vmax=vmax,
                cmap='PiYG',
                transform=ccrs.PlateCarree())
        
        
    ax.set_extent([-100, 5, 2, 80], 
                          ccrs.PlateCarree())
    
    ax.coastlines(color='k', alpha=0.7, lw=1)
    ax.set_title(name)

    
# Функция для отрисовки изобар (surface pressure)
def plot_isobars(ax, year, month, day, hour, center_lon=None, center_lat=None, zoom=False):
    """
    Рисует изобары из файлов surface pressure
    """
    path_sp = '/storage/thalassa/DATA/ERA5/mslp/'
    ncfile = f'{path_sp}/era5_mslp_{year}-{month:02d}.nc'
    
    if not os.path.exists(ncfile):
        print(f"Файл surface pressure не найден: {ncfile}")
        return
    
    try:
        ds_sp = xr.open_dataset(ncfile)
        
        # Находим ближайшее время
        target_time = np.datetime64(f'{year}-{month:02d}-{day:02d} {hour:02d}:00:00')
        
        # Находим индекс ближайшего времени
        time_idx = np.argmin(np.abs(ds_sp.time.values - target_time))
        
        # Извлекаем данные давления
        sp_data = ds_sp.msl.isel(time=time_idx)
        
        # Конвертируем из Па в гПа (гектопаскали)
        sp_hpa = sp_data / 100
        
        # Создаем уровни изобар (от 960 до 1040 гПа с шагом 4)
        levels = np.arange(860, 1020, 1)
        
        # Если это зум (правый график), вырезаем область
        if zoom and center_lon is not None and center_lat is not None:
            lon_min = center_lon - 5
            lon_max = center_lon + 5
            lat_min = center_lat - 5
            lat_max = center_lat + 5
            
            # Преобразуем долготы из формата 0-360 в -180-180 для сравнения
            # Для этого создаем копию координат долготы
            lon_coords = sp_hpa.longitude.values.copy()
            
            # Преобразуем долготы > 180 в отрицательные
            lon_coords_shifted = np.where(lon_coords > 180, lon_coords - 360, lon_coords)
            
            # Создаем новый DataArray с преобразованными долготами
            sp_hpa_shifted = sp_hpa.copy()
            sp_hpa_shifted['longitude'] = lon_coords_shifted
            
            # Теперь вырезаем область с использованием where
            sp_hpa_subset = sp_hpa_shifted.where(
                (sp_hpa_shifted.longitude >= lon_min) & 
                (sp_hpa_shifted.longitude <= lon_max) &
                (sp_hpa_shifted.latitude >= lat_min) & 
                (sp_hpa_shifted.latitude <= lat_max),
                drop=True
            )
            
            # Проверяем, что данные не пустые
            if sp_hpa_subset.shape[0] < 2 or sp_hpa_subset.shape[1] < 2:
                print(f"Предупреждение: слишком маленькая область для изобар: {sp_hpa_subset.shape}")
                print(f"  lon: {lon_min:.1f} - {lon_max:.1f}, lat: {lat_min:.1f} - {lat_max:.1f}")
                ds_sp.close()
                return
            
            lon_vals = sp_hpa_subset.longitude.values
            lat_vals = sp_hpa_subset.latitude.values
            sp_vals = sp_hpa_subset.values
        else:
            lon_vals = sp_hpa.longitude.values
            lat_vals = sp_hpa.latitude.values
            sp_vals = sp_hpa.values
        
        # Рисуем изобары (черные контуры) только на правом графике (zoom=True)
        if zoom:
            # Проверяем, что массив имеет правильную форму
            if sp_vals.shape[0] >= 2 and sp_vals.shape[1] >= 2:
                contour = ax.contour(lon_vals, lat_vals, sp_vals,
                                    levels=levels,
                                    colors='cyan',
                                    linewidths=1,
                                    alpha=0.6,
                                    transform=ccrs.PlateCarree(),
                                    zorder=4)
                
                # Добавляем подписи на изобарах
                ax.clabel(contour, inline=True, fontsize=6, fmt='%d', zorder=5)
                
            else:
                print(f"Предупреждение: недостаточный размер данных для изобар: {sp_vals.shape}")
        
        ds_sp.close()
        
    except Exception as e:
        print(f"Ошибка при загрузке/отрисовке изобар: {e}")
        import traceback
        traceback.print_exc()
        if 'ds_sp' in locals():
            ds_sp.close()
            
# Функция для получения траектории из xarray по track_id
def get_trajectory_from_xarray(track_id, ds_tracks):
    """Извлекает траекторию из xarray Dataset по track_id"""
    track_mask = ds_tracks['track_id'] == track_id
    track_data = ds_tracks.isel(record=track_mask)
    
    # Сортируем по времени
    track_df = track_data.to_dataframe()
    track_df = track_df.sort_values('time')
    
    return track_df

# Функция для создания цветов для разных траекторий
def get_color_generator():
    """Генератор цветов для разных траекторий"""
    colors = ['blue', 'red', 'purple', 'orange', 'brown', 'pink', 
              'cyan', 'magenta', 'olive', 'teal', 'navy', 'coral']
    for color in colors:
        yield color

def read_te_points(year, month, day, hour):
    """
    Читает точки из файла TempestExtremes для заданной даты и часа
    Возвращает DataFrame с колонками: lon, lat, radius, r2d
    """
    path_te_txt = f'{path_dir_data}/TempestExtremes/ERA5/R2D_ERA5_NA_for_TC_500hPa_sigma_2/R2D_txt_files'
    filename = f'ERA5_R2D_extr_{year}-{month:02d}.txt'
    filepath = f'{path_te_txt}/{filename}'
    
    if not os.path.exists(filepath):
        print(f"Файл не найден: {filepath}")
        return pd.DataFrame()
    
    # Читаем файл
    points = []
    found = False
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            # Парсим заголовок: год месяц день кол-во_точек час
            parts = line.split()
            if len(parts) >= 5:
                try:
                    file_year = int(parts[0])
                    file_month = int(parts[1])
                    file_day = int(parts[2])
                    n_points = int(parts[3])
                    file_hour = int(parts[4])
                    
                    # Проверяем, совпадает ли дата с искомой
                    if (file_year == year and file_month == month and 
                        file_day == day and file_hour == hour):
                        
                        print(f"Найден блок: {year}-{month:02d}-{day:02d} {hour:02d}:00, точек: {n_points}")
                        found = True
                        
                        # Читаем точки
                        for j in range(n_points):
                            i += 1
                            if i >= len(lines):
                                break
                            point_line = lines[i].strip().split()
                            if len(point_line) >= 6:
                                try:
                                    # Формат: индекс_i индекс_j долгота широта радиус r2d
                                    lon = float(point_line[2])
                                    lat = float(point_line[3])
                                    radius = float(point_line[4]) * 0.25  # Умножаем на 0.25
                                    r2d = float(point_line[5])
                                    points.append({
                                        'lon': lon,
                                        'lat': lat,
                                        'radius': radius,
                                        'r2d': r2d
                                    })
                                except Exception as e:
                                    print(f"Ошибка парсинга точки {j}: {point_line}")
                        break  # Выходим из цикла, так как нашли нужную дату
                except Exception as e:
                    pass
            i += 1
    
    if not found:
        print(f"Данные для {year}-{month:02d}-{day:02d} {hour:02d}:00 НЕ НАЙДЕНЫ")
    
    print(f"Загружено {len(points)} точек для {year}-{month:02d}-{day:02d} {hour:02d}:00")
    return pd.DataFrame(points)

# Функция для отрисовки точек TE на карте
def plot_te_points(ax, year, month, day, hour, center_lon, center_lat, zoom=False):
    """Рисует точки TE на указанной оси"""
    # Читаем точки для данной даты
    te_points = read_te_points(year, month, day, hour)
    
    if te_points.empty:
        return
    
    # Для зума (правый график) фильтруем точки в пределах 5 градусов
    if zoom:
        mask = (np.abs(te_points['lon'] - center_lon) <= 5) & \
               (np.abs(te_points['lat'] - center_lat) <= 5)
        te_points = te_points[mask]
    
    if te_points.empty:
        return
    
    # Рисуем точки (звездочки черного цвета)
    ax.scatter(te_points['lon'], te_points['lat'],
              marker='*', color='black', s=10,
              transform=ccrs.PlateCarree(),
              alpha=0.7, zorder=3,
              label=f'TE points ({len(te_points)})')
    
    # Рисуем круги радиуса вокруг каждой точки
    for _, point in te_points.iterrows():
        # Создаем круг радиуса
        circle = mpatches.Circle((point['lon'], point['lat']), 
                                radius=point['radius'],
                                transform=ccrs.PlateCarree(),
                                fill=False, 
                                color='gray', 
                                linewidth=0.5, 
                                alpha=0.5,
                                zorder=2)
        ax.add_patch(circle)


detected_TC = ibtracs_df[ibtracs_df['SID'] == id_IBTrACS]
detected_TC = detected_TC.sort_values('ISO_TIME')

# ========== СОЗДАЕМ РАВНОМЕРНУЮ СЕТКУ ВРЕМЕНИ С ШАГОМ 1 ЧАС ==========
print("\nСоздание равномерной сетки времени с шагом 1 час...")

# Определяем временной диапазон
start_time = detected_TC['ISO_TIME'].min()
end_time = detected_TC['ISO_TIME'].max()

# Создаем равномерную сетку с шагом 1 час (используем 'h' вместо 'H')
time_grid = pd.date_range(start=start_time, end=end_time, freq='1h')

print(f"Исходное количество точек IBTrACS: {len(detected_TC)}")
print(f"Новая сетка времени (1 час): {len(time_grid)} точек")
print(f"Диапазон: {start_time} - {end_time}")

# Функция для получения позиции ТЦ на заданное время (без интерполяции)
def get_tc_position_at_time(target_time, tc_data):
    """Возвращает позицию ТЦ на заданное время (без интерполяции)"""
    # Преобразуем время в numpy datetime64 для совместимости
    times = tc_data['ISO_TIME'].values
    target_time_np = np.datetime64(target_time)
    
    # Находим индекс последней точки, которая <= target_time
    # Используем searchsorted с numpy array
    idx = np.searchsorted(times.astype('datetime64[ns]'), target_time_np, side='right') - 1
    
    # Если выходим за границы
    if idx < 0:
        idx = 0
    if idx >= len(times):
        idx = len(times) - 1
    
    return tc_data['LON'].iloc[idx], tc_data['LAT'].iloc[idx], times[idx]

# Создаем DataFrame с повторяющимися значениями IBTrACS для каждого часа
interpolated_data = []

for time in time_grid:
    # Получаем позицию для текущего времени
    lon, lat, tc_time = get_tc_position_at_time(time, detected_TC)
    interpolated_data.append({
        'ISO_TIME': time,
        'LON': lon,
        'LAT': lat,
        'NAME': detected_TC['NAME'].iloc[0],
        'SID': id_IBTrACS,
        'TC_TIME': tc_time  # Сохраняем реальное время из IBTrACS
    })

detected_TC_interp = pd.DataFrame(interpolated_data)
print(f"Создано {len(detected_TC_interp)} интерполированных точек")
# ========================================================================

# Получаем соответствующие id_TE_NA из matches_TE_NA
te_ids = matches_TE_NA[matches_TE_NA['id_IBTrACS'] == id_IBTrACS]['id_TE_NA'].unique()
print(f"Найдено {len(te_ids)} связанных TE траекторий для {id_IBTrACS}")

# Пути к данным
path_init = f'/storage/thalassa/users/vkoshkina'
path_dir_data = f'{path_init}/data'
path_dir_raw = f'{path_dir_data}/ERA5/R2D_ERA5_NA_for_TC_500hPa_sigma_2/daily'

# Создаем директорию для сохранения кадров
output_dir = f'{path_dir_data}/TC_tracks/pics/anim_ERA5/{id_IBTrACS}_msl'
os.makedirs(output_dir, exist_ok=True)

# Генератор цветов для TE траекторий
color_gen = get_color_generator()
te_colors = {te_id: next(color_gen) for te_id in te_ids}

# Функция для отрисовки всех траекторий TE
def plot_te_trajectories(ax, current_time, center_lon, center_lat, zoom=False):
    """Рисует все TE траектории на указанной оси"""
    for te_id in te_ids:
        # Извлекаем траекторию для данного te_id
        te_track = get_trajectory_from_xarray(te_id, tracks_all_years)
        
        if not te_track.empty:
            # Фильтруем траекторию до текущего момента времени
            te_track_until_now = te_track[te_track['time'] <= current_time]
            
            if not te_track_until_now.empty:
                # Рисуем траекторию
                color = te_colors[te_id]
                ax.plot(te_track_until_now['lon'], te_track_until_now['lat'],
                       color=color, linewidth=1.5, 
                        # linestyle='--',
                       transform=ccrs.PlateCarree(),
                       alpha=0.9,
                       label=f'TE {te_id}')
                
                # Если это зум (правый график), рисуем только траектории в пределах области
                if zoom:
                    # Проверяем, есть ли точки в пределах 5 градусов
                    mask = (np.abs(te_track_until_now['lon'] - center_lon) <= 5) & \
                           (np.abs(te_track_until_now['lat'] - center_lat) <= 5)
                    if mask.any():
                        ax.plot(te_track_until_now['lon'][mask], 
                               te_track_until_now['lat'][mask],
                               color=color, linewidth=2,
                               transform=ccrs.PlateCarree(),
                               alpha=0.9)
                
                # Отмечаем текущую позицию TE, если она близка к центру
                if not te_track_until_now.empty:
                    last_point = te_track_until_now.iloc[-1]
                    dist_lon = abs(last_point['lon'] - center_lon)
                    dist_lat = abs(last_point['lat'] - center_lat)
                    if dist_lon <= 5 and dist_lat <= 5:
                        ax.plot(last_point['lon'], last_point['lat'],
                               'o', color=color, markersize=4,
                               transform=ccrs.PlateCarree())

                        
def update_and_save(time_index, index_t):
    # Очищаем оси
    ax1.clear()
    ax2.clear()
    
    # Настройка левого графика - фиксированный размер
    ax1.set_global()
    ax1.add_feature(cfeature.LAND, facecolor='#f0e0c0', alpha=0.3)
    ax1.coastlines(color='k', alpha=0.7, lw=1)
    
    # ФИКСИРОВАННЫЙ ЭКСТЕНТ для левого графика
    FIXED_LON_MIN = -80
    FIXED_LON_MAX = -30
    FIXED_LAT_MIN = 15
    FIXED_LAT_MAX = 55
    
    ax1.set_extent([FIXED_LON_MIN, FIXED_LON_MAX, FIXED_LAT_MIN, FIXED_LAT_MAX], ccrs.PlateCarree())
    
    # Получаем текущее время и позицию из интерполированных данных
    current_time = detected_TC_interp['ISO_TIME'].iloc[time_index]
    current_lon = detected_TC_interp['LON'].iloc[time_index]
    current_lat = detected_TC_interp['LAT'].iloc[time_index]
    real_tc_time = detected_TC_interp['TC_TIME'].iloc[time_index]  # Реальное время из IBTrACS
    
    # Извлекаем год, месяц, день и час из ISO_TIME
    year = current_time.year
    month = current_time.month
    day = current_time.day
    our_time = current_time.hour
    
    # Формируем путь к файлу для конкретной даты
    ncfile = f'{path_dir_raw}/{year}/sigma_2_R2D_{year}-{month:02d}-{day:02d}.nc'
    
    # Загружаем поле R2D если файл существует
    if os.path.exists(ncfile):
        ds = xr.open_dataset(ncfile)
        
        vmax = 0.5 * np.nanmax(np.abs(ds['R2D'][our_time]))
        
        # Отображение R2D на обоих графиках
        plot_r2d(ax1, ds, our_time=our_time, our_level=0, name='500hPa')
        plot_r2d(ax2, ds, our_time=our_time, our_level=0, name='500hPa')
        
        ds.close()
    
    # Рисуем изобары на правом графике (с зумом)
    plot_isobars(ax2, year, month, day, our_time, center_lon=current_lon, center_lat=current_lat, zoom=True)
    
    # Получаем имя ТЦ
    tc_name = detected_TC_interp['NAME'].iloc[0]
    
    # Рисуем траекторию IBTrACS на левом графике 
    # Используем ТОЛЬКО реальные точки IBTrACS (не все часы)
    # Находим все реальные точки IBTrACS до текущего времени
    TC_part_real = detected_TC[detected_TC['ISO_TIME'] <= real_tc_time]
    
    # Но для отображения на графике используем интерполированные точки (для непрерывности)
    # Берем все точки с начала до текущего времени
    start_idx = max(0, time_index - 100)  # Берем достаточно много для полной траектории
    TC_part_all = detected_TC_interp.iloc[start_idx:time_index+1]
    
    # Рисуем полную траекторию (интерполированную) - пунктиром для обозначения
    ax1.plot(TC_part_all['LON'], TC_part_all['LAT'], 
             color='gray', linewidth=1.0, linestyle=':',
             transform=ccrs.PlateCarree(),
             alpha=0.5)
    
    # Рисуем РЕАЛЬНЫЕ точки IBTrACS (жирной линией)
    if not TC_part_real.empty:
        ax1.plot(TC_part_real['LON'], TC_part_real['LAT'], 
                 color='k', linewidth=2.5, 
                 transform=ccrs.PlateCarree(),
                 label=f'IBTrACS {tc_name}')
    
    # Отмечаем текущую позицию (из интерполяции)
    # Проверяем, совпадает ли текущее время с реальной точкой IBTrACS
    is_real_point = (current_time == real_tc_time)
    marker_style = 'ko' if is_real_point else 'ko'
    marker_size = 4 if is_real_point else 3
    
    ax1.plot(current_lon, current_lat, marker_style, markersize=marker_size, 
             transform=ccrs.PlateCarree())
    
    # ========== ДОБАВЛЯЕМ КРАСНУЮ РАМКУ 5x5 ГРАДУСОВ ==========
    lon_min_box = current_lon - 5
    lon_max_box = current_lon + 5
    lat_min_box = current_lat - 5
    lat_max_box = current_lat + 5
    
    # Создаем прямоугольник
    rect = mpatches.Rectangle(
        (lon_min_box, lat_min_box), 
        width=10, height=10,
        transform=ccrs.PlateCarree(),
        fill=False,
        edgecolor='red',
        linewidth=1.5,
        # linestyle='--',
        alpha=0.8,
        zorder=5
    )
    ax1.add_patch(rect)
    
    
    # Рисуем TE траектории на левом графике
    plot_te_trajectories(ax1, current_time, current_lon, current_lat, zoom=False)
    
    # Рисуем TE точки на левом графике (только текущий момент)
    plot_te_points(ax1, year, month, day, our_time, current_lon, current_lat, zoom=False)
    
    # Добавляем сетку для левого графика
    gl1 = ax1.gridlines(draw_labels=True, linewidth=0.5, color='grey', alpha=0.5, linestyle='--')
    gl1.top_labels = False
    gl1.right_labels = False
    gl1.xformatter = LONGITUDE_FORMATTER
    gl1.yformatter = LATITUDE_FORMATTER
    
    # Добавляем легенду
    ax1.legend(loc='upper left', fontsize=8)
    
    # Заголовок левого графика
    ax1.set_title(f'TC {tc_name} - {current_time.strftime("%Y-%m-%d %H:%M")} UTC')
    
    # Настройка правого графика - детальный вид (всегда 10x10 градусов)
    lon_min = current_lon - 5
    lon_max = current_lon + 5
    lat_min = current_lat - 5
    lat_max = current_lat + 5
    
    ax2.set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())
    ax2.add_feature(cfeature.LAND, facecolor='#f0e0c0', alpha=0.3)
    ax2.coastlines(color='k', alpha=0.7, lw=1)
    
    # Рисуем траекторию IBTrACS на детальном графике
    # Сначала пунктиром интерполированную траекторию
    ax2.plot(TC_part_all['LON'], TC_part_all['LAT'], 
             color='gray', linewidth=1.0, linestyle=':',
             transform=ccrs.PlateCarree(),
             alpha=0.5)
    
    # Жирной линией реальные точки
    if not TC_part_real.empty:
        ax2.plot(TC_part_real['LON'], TC_part_real['LAT'], 
                 color='k', linewidth=2.5, 
                 transform=ccrs.PlateCarree())
    
    # Отмечаем текущую позицию
    ax2.plot(current_lon, current_lat, marker_style, markersize=10, 
             transform=ccrs.PlateCarree())
    
    # Рисуем TE траектории на правом графике (только в пределах 5 градусов)
    plot_te_trajectories(ax2, current_time, current_lon, current_lat, zoom=True)
    
    # Рисуем TE точки на правом графике (только в пределах 5 градусов)
    plot_te_points(ax2, year, month, day, our_time, current_lon, current_lat, zoom=True)
    
    # Добавляем сетку для детального графика
    gl2 = ax2.gridlines(draw_labels=True, linewidth=0.5, color='grey', alpha=0.5, linestyle='--')
    gl2.top_labels = False
    gl2.right_labels = False
    gl2.xformatter = LONGITUDE_FORMATTER
    gl2.yformatter = LATITUDE_FORMATTER
    
    # Заголовок детального графика
    n_te_nearby = sum([1 for te_id in te_ids if has_te_nearby(te_id, current_lon, current_lat, current_time)])
    ax2.set_title(f'Zoom (5° radius) - {n_te_nearby} TE nearby\nLon: {current_lon:.1f}°, Lat: {current_lat:.1f}°')
    
    plt.tight_layout()
    
    # Сохраняем кадр
    output_filename = f'{output_dir}/{id_IBTrACS}_track_{index_t:05d}.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"Сохранен кадр {index_t}: {output_filename}")
    
    return ax1, ax2
    
# Функция для проверки, есть ли TE траектория поблизости
def has_te_nearby(te_id, center_lon, center_lat, current_time):
    """Проверяет, есть ли TE траектория в радиусе 5 градусов от центра"""
    te_track = get_trajectory_from_xarray(te_id, tracks_all_years)
    if te_track.empty:
        return False
    
    te_track_until_now = te_track[te_track['time'] <= current_time]
    if te_track_until_now.empty:
        return False
    
    # Проверяем последнюю точку
    last_point = te_track_until_now.iloc[-1]
    dist_lon = abs(last_point['lon'] - center_lon)
    dist_lat = abs(last_point['lat'] - center_lat)
    
    return dist_lon <= 5 and dist_lat <= 5

# Основной цикл по временным шагам (используем интерполированные данные)
for index_t in range(len(detected_TC_interp)):
    print(f"Обработка шага {index_t} - Время: {detected_TC_interp['ISO_TIME'].iloc[index_t]}")    
    update_and_save(index_t, index_t)

plt.close(fig)
print(f"Все кадры сохранены в {output_dir}")

# Дополнительно: Создаем анимацию из сохраненных кадров
try:
    import imageio
    import glob
    
    # Собираем все PNG файлы в правильном порядке
    png_files = sorted(glob.glob(f'{output_dir}/{id_IBTrACS}_track_*.png'))
    
    if png_files:
        # Создаем GIF
        gif_filename = f'{output_dir}/{id_IBTrACS}_track_animation.gif'
        with imageio.get_writer(gif_filename, mode='I', duration=0.5) as writer:
            for filename in png_files:
                image = imageio.imread(filename)
                writer.append_data(image)
        print(f"Создана анимация: {gif_filename}")
except ImportError:
    print("Для создания GIF установите imageio: pip install imageio")
except Exception as e:
    print(f"Ошибка при создании анимации: {e}")