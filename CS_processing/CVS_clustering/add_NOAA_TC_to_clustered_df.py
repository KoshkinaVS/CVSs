import pandas as pd
from pathlib import Path
from glob import glob

def validate_track_name(name: str) -> str:
    """
    Проверяет префикс ID в имени файла до '_track'.
    Если len(prefix) == 8 — возвращает base как есть.
    Иначе — дополняет prefix до 8 знаков и возвращает полное имя.
    
    '003568_track_1979-06-11T06.csv' -> '00003568_track_1979-06-11T06.csv'
    '00003568_track_1979-06-11T06.csv' -> '00003568_track_1979-06-11T06.csv'
    """
    base = Path(name).name
    prefix, *rest = base.split('_track', 1)
    if len(prefix) == 8:
        return base
    padded_prefix = prefix.zfill(8)
    new_base = f"{padded_prefix}_track{rest[0] if rest else ''}"
    return new_base


path_init = '/storage/thalassa/users/vkoshkina'
path_data = f'{path_init}/data'
data_type = 'LoRes'
sigma = 2

# ---------- 1. NOAA + NAAD ----------
noaa_dir = f'{path_data}/TC_tracks/NAAD_NOAA_matching_tables'
noaa_name = f'noaa_naad_matching_summary_full_sigma_{sigma}.csv'

df_match = pd.read_csv(f'{noaa_dir}/{noaa_name}')

# отфильтровать нужные строки
df_match_filt = df_match[
    (df_match['tracking_type'] == 'tracking_local_2_phase') &
    (df_match['CVS_speed'] == 'adv_speed') &
    (df_match['sigma'] == 2)
].copy()  # .copy() сразу

# track_id_8 для NOAA
df_match_filt['track_id_8'] = df_match_filt['naad_file_path'].astype(str).map(
    lambda x: Path(x).name.split('_track')[0].zfill(8)
)

# Только нужное для merge (все строки!)
noaa_merge = df_match_filt[['track_id_8', 'noaa_id']].copy()
print(f"NOAA треков: {len(noaa_merge)}")


print(f"naad_file_path не пусто: {noaa_merge['track_id_8'].count()}")

# ---------- 2. Кластерные треки ----------
path_data_tracks = f'{path_data}/TC_tracks/{data_type}/{data_type}_sigma_{sigma}'
output_dir = f"{path_data_tracks}/cluster_results_august"

input_season = 'year'
n_clusters = 9

df_season = pd.read_csv(
    f"{output_dir}/cluster_tables/{input_season}_nclusters_{n_clusters}_tracks.csv",
    parse_dates=['datetime']
).copy()

df_season['track_id_8'] = df_season['name'].astype(str).map(
        lambda x: Path(x).name.split('_track')[0].zfill(8)
    )

# ---------- DEBUG ----------
n_common = len(set(noaa_merge['track_id_8']) & set(df_season['track_id_8']))
print(f"n_common track_id_8 = {n_common}")
print("Примеры:", list(set(noaa_merge['track_id_8']) & set(df_season['track_id_8']))[:5])

# ---------- 3. Merge ----------
df_season = df_season.merge(
    noaa_merge,
    how='left',
    on='track_id_8'
)
df_season['NOAA_TC'] = df_season['noaa_id'].notna()
print(f"Добавлено матчей: {df_season['NOAA_TC'].sum()}")


# ---------- 4. Сохранить ----------
out_path = f"{output_dir}/cluster_tables/{input_season}_nclusters_{n_clusters}_tracks_with_NOAA.csv"
df_season.to_csv(out_path, index=False)
print('Сохранено в', out_path)
