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

year = 2011
# Данные для конкретного ТЦ
id_IBTrACS = '2003286N09323'
# id_IBTrACS = '2018186N10325'
id_IBTrACS = '2001245N13326' 
id_IBTrACS = '2011266N08343' 



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
        
        
    ax.set_extent([-100, 17, 2, 85], 
                          ccrs.PlateCarree())
    
    ax.coastlines(color='k', alpha=0.7, lw=1)
    ax.set_title(name)

    
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
        # Показываем все доступные даты в файле для отладки
        print("Доступные даты в файле:")
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines[:50]:  # Проверяем первые 50 строк
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        f_year = int(parts[0])
                        f_month = int(parts[1])
                        f_day = int(parts[2])
                        f_hour = int(parts[4])
                        n_pts = int(parts[3])
                        print(f"  {f_year}-{f_month:02d}-{f_day:02d} {f_hour:02d}:00 - {n_pts} точек")
                    except:
                        pass
    
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

# Получаем соответствующие id_TE_NA из matches_TE_NA
te_ids = matches_TE_NA[matches_TE_NA['id_IBTrACS'] == id_IBTrACS]['id_TE_NA'].unique()
print(f"Найдено {len(te_ids)} связанных TE траекторий для {id_IBTrACS}")

# Загружаем xarray Dataset с траекториями (если еще не загружен)
# Предполагаем, что ds_tracks уже загружен
# ds_tracks = xr.open_dataset('path_to_tracks.nc')

# Сортируем detected_TC по времени
detected_TC = detected_TC.sort_values('ISO_TIME')

# Пути к данным
path_init = f'/storage/thalassa/users/vkoshkina'
path_dir_data = f'{path_init}/data'
path_dir_raw = f'{path_dir_data}/ERA5/R2D_ERA5_NA_for_TC_500hPa_sigma_2/daily'

# Создаем директорию для сохранения кадров
output_dir = f'{path_dir_data}/TC_tracks/pics/anim_ERA5/{id_IBTrACS}'
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

# Функция для обновления графика на каждом шаге
def update_and_save(time_index, index_t):
    # Очищаем оси
    ax1.clear()
    ax2.clear()
    
    # Настройка левого графика
    ax1.set_global()
    ax1.add_feature(cfeature.LAND, facecolor='#f0e0c0', alpha=0.3)
    ax1.coastlines(color='k', alpha=0.7, lw=1)
    ax1.set_extent([-100, 5, 0, 82], ccrs.PlateCarree())
    
    # Получаем текущее время и позицию
    current_time = detected_TC['ISO_TIME'].iloc[time_index]
    current_lon = detected_TC['LON'].iloc[time_index]
    current_lat = detected_TC['LAT'].iloc[time_index]
    
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
    
    # Получаем имя ТЦ
    tc_name = detected_TC['NAME'].iloc[0]
    
    # Рисуем траекторию IBTrACS (красным)
    TC_part = detected_TC[detected_TC['ISO_TIME'] <= current_time]
    ax1.plot(TC_part['LON'], TC_part['LAT'], 
             color='k', linewidth=2.5, 
             transform=ccrs.PlateCarree(),
             label=f'IBTrACS {tc_name}')
    
    # Отмечаем текущую позицию IBTrACS
    ax1.plot(current_lon, current_lat, 'ko', markersize=4, 
             transform=ccrs.PlateCarree())
    
    # Рисуем TE траектории на левом графике
    plot_te_trajectories(ax1, current_time, current_lon, current_lat, zoom=False)
    
    # Рисуем TE точки на левом графике
    plot_te_points(ax1, year, month, day, our_time, current_lon, current_lat, zoom=False)
    
    # Добавляем легенду
    ax1.legend(loc='upper left', fontsize=8)
    
    # Заголовок левого графика
    ax1.set_title(f'TC {tc_name} - {current_time.strftime("%Y-%m-%d %H:%M")} UTC')
    
    # Настройка правого графика - детальный вид
    lon_min = current_lon - 5
    lon_max = current_lon + 5
    lat_min = current_lat - 5
    lat_max = current_lat + 5
    
    ax2.set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())
    ax2.add_feature(cfeature.LAND, facecolor='#f0e0c0', alpha=0.3)
    ax2.coastlines(color='k', alpha=0.7, lw=1)
    
    # Рисуем траекторию IBTrACS на детальном графике
    ax2.plot(TC_part['LON'], TC_part['LAT'], 
             color='k', linewidth=2.5, 
             transform=ccrs.PlateCarree())
    ax2.plot(current_lon, current_lat, 'ko', markersize=10, 
             transform=ccrs.PlateCarree())
    
    # Рисуем TE траектории на правом графике (только те, что в пределах 5 градусов)
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



# Основной цикл по временным шагам
for index_t in range(len(detected_TC)):
    print(f"Обработка шага {index_t} - Время: {detected_TC['ISO_TIME'].iloc[index_t]}")    
    update_and_save(index_t, index_t)

plt.close(fig)
print(f"Все кадры сохранены в {output_dir}")

# Дополнительно: Создаем анимацию из сохраненных кадров (опционально)
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