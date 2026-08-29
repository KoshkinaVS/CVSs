import os
from tqdm import tqdm
import calendar

path_init = '/storage/thalassa/users/vkoshkina/data/TempestExtremes'

months = range(1, 13)       # от 1 до 12

sigmas = [
    4,
    2,
    0,
]


print('data type: ')
data_type = input() 

if data_type == 'LoRes' or data_type == 'HiRes':
    level = 12
    years = range(1979, 2019)   
    u_name = 'ue'
    v_name = 've'
    search_range = 2.0
elif data_type == 'SMP':
    level = 10
    years = range(2019, 2020)
    u_name = 'ua'
    v_name = 'va'
    search_range = 0.5

elif data_type == 'GPN':
    level = 22
    level = 12
    
    years = range(2022, 2023)   
    months = range(2, 3)       # !!!!!!!!!!!!!
    u_name = 'ua'
    v_name = 'va'
    search_range = 0.4
    
elif data_type == 'ERA5':
    level = 500
    years = range(1979, 2025)   
    u_name = 'u'
    v_name = 'v'
    search_range = 1.5


if data_type == 'HiRes':
    search_range = 0.4

stitchnodes_cmd = "StitchNodes"  # Убедись, что доступен в PATH

for sigma in tqdm(sigmas):
    sigma_dir = f"{path_init}/{data_type}/R2D_{data_type}_level_{level}_sigma_{sigma}"
    
    # Поддиректории

    input_list_dir = f"{sigma_dir}/Input_list_mergedist_05"
    nodes_dir = f"{sigma_dir}/Nodes_mergedist_05"
    tracks_dir = f"{sigma_dir}/Tracks_05"

    input_list_dir = f"{sigma_dir}/Input_list_timefilter_1h"
    nodes_dir = f"{sigma_dir}/Nodes_mergedist_timefilter_1h"
    tracks_dir = f"{sigma_dir}/Tracks_timefilter_1h"
    
    # input_list_dir = f"{sigma_dir}/Input_list"
    # tracks_dir = f"{sigma_dir}/Tracks"
    # nodes_dir = f"{sigma_dir}/Nodes"
    
    # Создаём нужные директории
    os.makedirs(input_list_dir, exist_ok=True)
    os.makedirs(tracks_dir, exist_ok=True)
    os.makedirs(nodes_dir, exist_ok=True)

    for year in tqdm(years):
        base_path_out = f"{nodes_dir}/{data_type}_R2D_extr"  # месячные файлы .txt

        # Пути к файлам списков и треков
        output_list_file = f"{input_list_dir}/{data_type}_R2D_extr_{year}.txt"
        output_tracks_file = f"{tracks_dir}/{data_type}_TC_tracks_{year}.csv"

        # Генерация списка файлов (полные пути к monthly .txt)
        with open(output_list_file, 'w') as fout:
            for month in months:

                if data_type == 'SMP' or data_type == 'GPN':
                    num_days = calendar.monthrange(year, month)[1]
        
                    for day in range(1, num_days + 1):
                        output_nc = f"{base_path_out}_{year}-{month:02d}-{day:02d}.txt"
                        fout.write(f"{output_nc}\n")
                else:
                    output_nc = f"{base_path_out}_{year}-{month:02d}.txt"
                    fout.write(f"{output_nc}\n")

        # Переход в директорию sigma_dir для выполнения команды
        original_dir = os.getcwd()
        try:
            os.chdir(sigma_dir)
            print(f"Перешли в директорию: {sigma_dir}")

            cmd = (
                f"{stitchnodes_cmd} "
                f"--in_list {output_list_file} "
                f"--out {output_tracks_file} "
                f"--in_fmt \"lon,lat,wind,r2d\" "
                f"--range {search_range} "
                f"--mintime \"12h\" "
                f"--maxgap \"3h\" "
                f"--min_endpoint_dist 0.5 "
                f"--out_file_format \"csv\" "
            )

            print(f"Выполняется: {cmd}")
            exit_code = os.system(cmd)

            if exit_code == 0:
                print(f"Успешно: треки для sigma={sigma}, {year} сохранены в {output_tracks_file}")
            else:
                print(f"Ошибка при обработке sigma={sigma}, год {year}")

        except Exception as e:
            print(f"Ошибка при выполнении команды для sigma={sigma}, год {year}: {e}")
        finally:
            os.chdir(original_dir)  # Возвращаемся в исходную директорию