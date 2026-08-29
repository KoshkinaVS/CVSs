import huracanpy
from tqdm import tqdm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import os

path_init = f'/storage/thalassa/users/vkoshkina'
path_dir_data = f'{path_init}/data/'

base_folder = f'{path_dir_data}/TempestExtremes/ERA5/R2D_ERA5_NA_for_TC_500hPa_sigma_2/Tracks_R2D_txt_files'

postfix_list = ['', '_range_2', '_range_1', '_range_1_prioritize',
                '_range_1_5_18h_12h',
               ]

all_tracks_TC = huracanpy.load(
    source="ibtracs",
    # ibtracs_subset="NA",
)

fin_year = 2018
all_tracks_TC = all_tracks_TC.where(all_tracks_TC.time.dt.year < fin_year + 1, drop=True)

# Фильтруем по бассейну "NA" (North Atlantic)
na_tracks = all_tracks_TC.where(all_tracks_TC.basin == "NA", drop=True)

na_tracks_filtered = na_tracks

print(f"ibtracs loaded: {na_tracks_filtered.track_id.hrcn.nunique()}")

years = np.arange(1979, 2025)


def load_TE_for_years(base_folder, years):
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
    return tracks_all_years

# Создаем список для хранения результатов
results_list = []

for postfix in postfix_list:
    print(f"\n{'='*60}")
    print(f"Processing: {postfix if postfix else 'default'}")
    print(f"{'='*60}")
    
    tracks_all_years = load_TE_for_years(f'{base_folder}{postfix}', years)
    
    matches = huracanpy.assess.match([na_tracks_filtered, tracks_all_years], 
                                     names=['IBTrACS', 'TE_NA'], 
                                     max_dist=200, 
                                     mean_dist=120, 
                                     min_overlap=3, 
                                     tracks1_is_ref=True)
    
    filtered_matches = matches.loc[
        matches.groupby('id_IBTrACS')['temp'].idxmax()
    ]
    
    POD = huracanpy.assess.pod(matches, ref=na_tracks_filtered, ref_name='IBTrACS')
    
    # 1. Находим ID треков IBTrACS, которые НЕ сопоставились
    all_ibtracs_ids = set(na_tracks_filtered.track_id.values)
    matched_ids = set(matches['id_IBTrACS'].values)
    unmatched_ids = all_ibtracs_ids - matched_ids
    
    print(f"POD = {POD:.4f}")
    print(f"Сопоставилось треков: {len(matched_ids)}")
    print(f"Не сопоставилось треков: {len(unmatched_ids)}")
    
    # 1. Рассчитываем общую длину каждого трека в na_tracks_filtered (референсный датасет)
    na_df = na_tracks_filtered.to_dataframe().reset_index()
    
    # Считаем количество записей для каждого track_id
    na_track_lengths = na_df.groupby('track_id').size().reset_index(name='IBTrACS_len')
    
    # 2. Агрегируем temp из matches по id_IBTrACS (суммируем количество совпавших таймстепов)
    matches_agg = filtered_matches.groupby('id_IBTrACS')['temp'].sum().reset_index(name='temp_total')
    
    # 3. Объединяем с длинами треков из референсного датасета
    merged = pd.merge(matches_agg, na_track_lengths, 
                      left_on='id_IBTrACS', right_on='track_id', 
                      how='inner')
    
    # 4. Рассчитываем процент покрытия
    merged['coverage_percent'] = (merged['temp_total'] / merged['IBTrACS_len']) * 100
    
    # 5. Выводим статистику распределения
    print(f"\n=== Статистика распределения покрытия ===")
    print(merged['coverage_percent'].describe())
    
    # Сохраняем результаты для этого постфикса
    result_entry = {
        'postfix': postfix if postfix else 'default',
        'POD': POD,
        'matched_tracks': len(matched_ids),
        'unmatched_tracks': len(unmatched_ids),
        'total_ibtracs_tracks': len(all_ibtracs_ids),
        'mean_coverage': merged['coverage_percent'].mean(),
        'median_coverage': merged['coverage_percent'].median(),
        'std_coverage': merged['coverage_percent'].std(),
        'min_coverage': merged['coverage_percent'].min(),
        'max_coverage': merged['coverage_percent'].max(),
        'coverage_25th': merged['coverage_percent'].quantile(0.25),
        'coverage_75th': merged['coverage_percent'].quantile(0.75),
    }
    results_list.append(result_entry)

# Создаем DataFrame со всеми результатами
results_df = pd.DataFrame(results_list)

# Сохраняем результаты в CSV файл
folder_TE = f'{path_dir_data}/TempestExtremes/ERA5/R2D_ERA5_NA_for_TC_500hPa_sigma_2'

output_file = f'{folder_TE}/compare_TC_Tracks_R2D_txt_files_1980-{fin_year}.csv'
results_df.to_csv(output_file, index=False)
print(f"\n{'='*60}")
print(f"Results saved to: {output_file}")
print(f"{'='*60}")


# Выводим итоговую таблицу
print("\n" + "="*60)
print("SUMMARY TABLE:")
print("="*60)
print(results_df.to_string())

# Дополнительно: сохраняем детальную информацию по каждому треку для каждого постфикса
# (опционально, если нужно больше деталей)
detailed_results = []

for postfix in postfix_list:
    print(f"\nProcessing detailed data for: {postfix if postfix else 'default'}")
    
    tracks_all_years = load_TE_for_years(f'{base_folder}{postfix}', years)
    
    matches = huracanpy.assess.match([na_tracks_filtered, tracks_all_years], 
                                     names=['IBTrACS', 'TE_NA'], 
                                     max_dist=200, 
                                     mean_dist=120, 
                                     min_overlap=4, 
                                     tracks1_is_ref=True)
    
    filtered_matches = matches.loc[
        matches.groupby('id_IBTrACS')['temp'].idxmax()
    ]
    
    na_df = na_tracks_filtered.to_dataframe().reset_index()
    na_track_lengths = na_df.groupby('track_id').size().reset_index(name='IBTrACS_len')
    
    matches_agg = filtered_matches.groupby('id_IBTrACS')['temp'].sum().reset_index(name='temp_total')
    
    merged = pd.merge(matches_agg, na_track_lengths, 
                      left_on='id_IBTrACS', right_on='track_id', 
                      how='inner')
    
    merged['coverage_percent'] = (merged['temp_total'] / merged['IBTrACS_len']) * 100
    merged['postfix'] = postfix if postfix else 'default'
    
    detailed_results.append(merged)

# Объединяем все детальные результаты
detailed_df = pd.concat(detailed_results, ignore_index=True)

# Сохраняем детальные результаты
detailed_output_file = f'{folder_TE}/compare_TC_Tracks_R2D_txt_files_detailed_1980-{fin_year}.csv'
detailed_df.to_csv(detailed_output_file, index=False)
print(f"\nDetailed results saved to: {detailed_output_file}")