import huracanpy

from tqdm import tqdm  # для отображения прогресс-бара (опционально)
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import os

import cmaps

path_init = f'/storage/thalassa/users/vkoshkina'
path_dir_data = f'{path_init}/data/'



postfix = '_range_1_5_18h_12h'

R2D_files = True
R2D_files = False


if R2D_files:
    R2D_fix = '_R2D_txt_files'
else:
    R2D_fix = ''
    
base_folder = f'{path_dir_data}/TempestExtremes/ERA5/R2D_ERA5_NA_for_TC_500hPa_sigma_2/Tracks{R2D_fix}{postfix}'


all_tracks_TC = huracanpy.load(
    source="ibtracs",
    # ibtracs_subset="NA",
)
# all_tracks_TC = all_tracks_TC.where(all_tracks_TC.time.dt.year < 1999, drop=True)

# Фильтруем по бассейну "NA" (North Atlantic)
na_tracks = all_tracks_TC.where(all_tracks_TC.basin == "NA", drop=True)

# # # (Опционально) Дополнительная фильтрация по широте и долготе для уверенности
# na_tracks_filtered = na_tracks.where(
#     (na_tracks.lon > -110) & (na_tracks.lon < 15) & 
#     (na_tracks.lat > 0) & (na_tracks.lat < 71),
#     drop=True
# )

na_tracks_filtered = na_tracks

print(f"ibtracs loaded: {na_tracks_filtered.track_id.hrcn.nunique()}")


# Словарь для хранения треков по годам
tracks_by_year = {}
all_tracks = []

years = np.arange(1979, 2025)

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


matches = huracanpy.assess.match([na_tracks_filtered, tracks_all_years], names=['IBTrACS', 'TE_NA'], 
                                 max_dist=200, 
                                 mean_dist=120, 
                                 min_overlap=4, tracks1_is_ref=True)

filtered_matches = matches.loc[
    matches.groupby('id_IBTrACS')['temp'].idxmax()
]

POD = huracanpy.assess.pod(matches, ref=na_tracks_filtered, ref_name='IBTrACS')

print(f'POD={POD}')






# 1. Находим ID треков IBTrACS, которые НЕ сопоставились
all_ibtracs_ids = set(na_tracks_filtered.track_id.values)
matched_ids = set(matches['id_IBTrACS'].values)
unmatched_ids = all_ibtracs_ids - matched_ids

print(f"Всего треков IBTrACS: {len(all_ibtracs_ids)}")
print(f"Сопоставилось треков: {len(matched_ids)}")
print(f"Не сопоставилось треков: {len(unmatched_ids)}")

# 2. Отфильтровываем треки
unmatched_tracks = na_tracks_filtered.where(
    na_tracks_filtered.track_id.isin(list(unmatched_ids)), 
    drop=True
)

matched_tracks = na_tracks_filtered.where(
    na_tracks_filtered.track_id.isin(list(matched_ids)), 
    drop=True
)

# 3. Получаем список уникальных годов
matched_tracks['year'] = matched_tracks.time.dt.year
years = np.unique(matched_tracks.year.values)
years = [int(y) for y in years if y >= 1979]
years = sorted(years)

print(f"Годы для отображения: {years}")

# 4. Создаем словарь для быстрого поиска соответствий
ibtracs_to_te = {}
for idx, row in matches.iterrows():
    ibtracs_id = row['id_IBTrACS']
    te_id = row['id_TE_NA']
    if ibtracs_id not in ibtracs_to_te:
        ibtracs_to_te[ibtracs_id] = []
    ibtracs_to_te[ibtracs_id].append(te_id)

# 5. Строим отдельную карту для КАЖДОГО трека IBTrACS
total_tracks = 0

for year in years:
    print(f"\nОбработка {year} года...")
    
    # Фильтруем треки IBTrACS для этого года
    ibtracs_year = matched_tracks.where(matched_tracks.year == year, drop=True)
    
    # Получаем ID треков IBTrACS для этого года
    ibtracs_ids_year = set(ibtracs_year.track_id.values)
    
    print(f"  IBTrACS треков в {year}: {len(ibtracs_ids_year)}")
    
    # Для каждого IBTrACS трека создаем отдельную карту
    for track_idx, ibtracs_id in enumerate(ibtracs_ids_year):
        total_tracks += 1
        
        # Получаем данные IBTrACS трека
        ibtracs_track = ibtracs_year.where(
            ibtracs_year.track_id == ibtracs_id, 
            drop=True
        ).sortby('time')
        
        # Находим соответствующие TE треки
        te_tracks_for_this = []
        te_ids_found = set()
        
        if ibtracs_id in ibtracs_to_te:
            te_ids = ibtracs_to_te[ibtracs_id]
            for te_id in te_ids:
                if te_id not in te_ids_found:
                    track_data = tracks_all_years.where(
                        tracks_all_years.track_id == te_id, 
                        drop=True
                    )
                    if len(track_data) > 0:
                        te_tracks_for_this.append(track_data)
                        te_ids_found.add(te_id)
        
        # Объединяем TE треки
        if te_tracks_for_this:
            tracks_te = huracanpy.concat_tracks(te_tracks_for_this)
            unique_te_ids = np.unique(tracks_te.track_id.values)
            n_te = len(unique_te_ids)
        else:
            tracks_te = None
            n_te = 0
        
        # Создаем карту
        fig = plt.figure(figsize=(12, 8), dpi=150)
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        
        # Настройка карты
        # Определяем экстент вокруг трека для лучшего обзора
        lat_min = ibtracs_track.lat.min().values - 10
        lat_max = ibtracs_track.lat.max().values + 10
        lon_min = ibtracs_track.lon.min().values - 10
        lon_max = ibtracs_track.lon.max().values + 10
        
        # Ограничиваем экстент разумными пределами
        lat_min = max(-90, lat_min)
        lat_max = min(90, lat_max)
        lon_min = max(-180, lon_min)
        lon_max = min(180, lon_max)
        
        # Если трек маленький, расширяем экстент
        if (lat_max - lat_min) < 5:
            center_lat = (lat_min + lat_max) / 2
            lat_min = center_lat - 10
            lat_max = center_lat + 10
        if (lon_max - lon_min) < 5:
            center_lon = (lon_min + lon_max) / 2
            lon_min = center_lon - 10
            lon_max = center_lon + 10
        
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
        
        # Рисуем IBTrACS трек (черным, жирным)
        ax.plot(ibtracs_track.lon, ibtracs_track.lat, 
                color='black', linewidth=3.0, transform=ccrs.PlateCarree(),
                label='IBTrACS')
        
        # Отмечаем начало и конец трека
        ax.plot(ibtracs_track.lon[0], ibtracs_track.lat[0], 
                'go', markersize=8, transform=ccrs.PlateCarree(),
                label='Genesis')
        ax.plot(ibtracs_track.lon[-1], ibtracs_track.lat[-1], 
                'rs', markersize=8, transform=ccrs.PlateCarree(),
                label='Lysis')

            
        # Рисуем TE треки (разными цветами) - без синих и голубых оттенков
        if tracks_te is not None:
            unique_te_ids = np.unique(tracks_te.track_id.values)
            n_tracks = len(unique_te_ids)
            
            # # Только теплые и яркие цвета (без синего, голубого, циана)
            # # Все цвета хорошо контрастируют с океаном и сушей
            # map_colors = [
            #     '#FF1493',  # Горячий розовый
            #     '#CC00FF',  # Фиолетовый
            #     # '#FF69B4',  # Ярко-розовый
            #     '#00CC00',  # Ярко-зеленый
                
            #     '#FF0000',  # Ярко-красный
            #     '#FF8C00',  # Темно-оранжевый
            #     '#FFD700',  # Золотой
            #     '#FF4500',  # Оранжево-красный
            #     '#7FFF00',  # Шартрез (желто-зеленый)
            #     '#FF6347',  # Томатный
            #     '#FF00FF',  # Маджента
            #     '#FFA500',  # Оранжевый
            #     '#FF1493',  # Розовый
            #     '#ADFF2F',  # Желто-зеленый
            #     '#FF4040',  # Ярко-красный
            #     '#FFB6C1',  # Светло-розовый
            # ]
            
            # # Если треков больше 10, используем больше цветов из списка
            # if n_tracks <= len(map_colors):
            #     colors = map_colors[:n_tracks]
            # else:
            #     colors = map_colors * (n_tracks // len(map_colors) + 1)
            #     colors = colors[:n_tracks]
    
            colors = [cmaps.grads_default(i) for i in range(n_tracks)]
            
            for i, te_id in enumerate(unique_te_ids):
                te_track = tracks_te.where(
                    tracks_te.track_id == te_id, 
                    drop=True
                ).sortby('time')
                
                ax.plot(te_track.lon, te_track.lat, 
                        color=colors[i], linewidth=1.5, alpha=0.8,
                        transform=ccrs.PlateCarree(),
                        label=f'TE {te_id}')
        
        # Добавляем легенду
        ax.legend(loc='upper left', fontsize=8)
        ax.set_extent([-110, 15, 0, 73], ccrs.PlateCarree())
        
        
        # Заголовок с информацией
        storm_name = ibtracs_track.attrs.get('name', 'Unknown') if hasattr(ibtracs_track, 'attrs') else 'Unknown'
        plt.title(f'IBTrACS {ibtracs_id} ({storm_name}) - {year}\n'
                  f'Черный: IBTrACS, Цветные: TempestExtremes ({n_te} треков)')
        
        plt.tight_layout()

        output_dir = f'{path_dir_data}/TempestExtremes/ERA5/R2D_ERA5_NA_for_TC_500hPa_sigma_2/TC_comparison_huracanpy{R2D_fix}{postfix}'
        # Создаем папку для сохранения картинок, если её нет
        os.makedirs(output_dir, exist_ok=True)
        
        # Сохраняем картинку
        filename = f'{output_dir}/IBTrACS_{ibtracs_id}_{year}_track{track_idx+1}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()  # Закрываем фигуру, чтобы не занимала память
        
        print(f"    {total_tracks}. Сохранен: {filename}")

print("\n" + "="*60)
print(f"ГОТОВО! Всего сохранено {total_tracks} картинок в папке '{output_dir}'")
print("="*60)


