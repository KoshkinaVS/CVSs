import os
from tqdm import tqdm
import calendar

path_init = '/storage/thalassa/users/vkoshkina/data/TempestExtremes'

months = range(1, 13)       # от 1 до 12

sigmas = [
#     4,
    2,
#     0,
]


# print('data type: ')
# data_type = input() 

data_type = 'ERA5'

maxgap = 3

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

elif data_type == 'GLORYS':
    level = 8
    # level = 15
    
    years = range(2023, 2024)   
    # years = range(1979, 1980)   
    
    u_name = 'uo'
    v_name = 'vo'
    search_range = 0.25
    search_range = 0.15 # 2026-08-04
    
    sigmas = [0]
elif data_type == 'ALT':
    level = 0
    years = range(2023, 2024)   
    
    u_name = 'ugos'
    v_name = 'vgos'
    search_range = 0.25
    search_range = 0.15 # 2026-08-04
    
    sigmas = [0]
elif data_type == 'ERA5':
    level = 500
    years = range(1979, 2025)   
    # years = range(1979, 1980)   
    years = range(2010, 2011)   
    
    
    u_name = 'u'
    v_name = 'v'
    search_range = 1.5 # # 2026-07-14 - translation speed of the fastest EXs <= 140 km per hour (Bernhardt&DeGaetano,2012;Lodiseetal.,2022).
    
    # #### 2026-07-09
    # search_range = 1.
    # #### 2026-07-13
    # search_range = 2.
    


if data_type == 'HiRes':
    search_range = 0.4

stitchnodes_cmd = "StitchNodes"  # Убедись, что доступен в PATH


print('size_filter: ')
size_filter = int(input())


for sigma in tqdm(sigmas):
    # sigma_dir = f"{path_init}/{data_type}/R2D_{data_type}_level_{level}_sigma_{sigma}"
    
    #### 2026-07-16

    region = 'Arctic'
    region = 'NA'
    # region = 'BarKara'
    
    
    if region == 'Arctic':
        # Арктика
        level_hPa = 850
        region_name = f'Arctic_{level_hPa}hPa'
    elif region == 'BarKara':
        region_name = f'BarKara_level_{level}'
    else:
        # Атлантика
        level_hPa = 500
        level_hPa = 850
        region_name = f'NA_for_TC_{level_hPa}hPa'
    
    sigma_dir = f"{path_init}/{data_type}/R2D_{data_type}_{region_name}_sigma_{sigma}"
    
    # Поддиректории

#     input_list_dir = f"{sigma_dir}/Input_list_mergedist_05"
#     nodes_dir = f"{sigma_dir}/Nodes_mergedist_05"
#     tracks_dir = f"{sigma_dir}/Tracks_05"

#     input_list_dir = f"{sigma_dir}/Input_list_timefilter_1h"
#     nodes_dir = f"{sigma_dir}/Nodes_mergedist_timefilter_1h"
#     tracks_dir = f"{sigma_dir}/Tracks_timefilter_1h"
    
    input_list_dir = f"{sigma_dir}/Input_list_for_tracking"
    tracks_dir = f"{sigma_dir}/Tracks"
    nodes_dir = f"{sigma_dir}/Nodes"

    input_list_dir = f"{sigma_dir}/R2D_txt_files_list_for_tracking"
    nodes_dir = f"{sigma_dir}/R2D_txt_files"

    extr_type = '_global'
    # extr_type = '_local'
    # extr_type = ''
    
    
    # size_filter = 10
    nodes_dir = f"{sigma_dir}/R2D_txt_files_2010_{size_filter}points{extr_type}"
    
    # tracks_dir = f"{sigma_dir}/Tracks_R2D_txt_files_range_015_144h_72h"
    tracks_dir = f"{sigma_dir}/Tracks_R2D_txt_files_range_1_5_18h_{maxgap}h_2010_{size_filter}points{extr_type}"
    

    # input_list_dir = f"{sigma_dir}/Input_list_for_tracking"
    # tracks_dir = f"{sigma_dir}/Tracks_range_1_5_18h_12h"
    # nodes_dir = f"{sigma_dir}/Nodes_mergedist_075"
    
    
    # Создаём нужные директории
    os.makedirs(input_list_dir, exist_ok=True)
    os.makedirs(tracks_dir, exist_ok=True)
    os.makedirs(nodes_dir, exist_ok=True)

    for year in tqdm(years):

        if data_type == 'GLORYS' or data_type == 'ALT':
            # ========== ИЗМЕНЕНИЕ: Теперь указываем путь к годовому файлу ==========
            # Вместо месячных файлов используем один годовой файл
            annual_file = f"{nodes_dir}/{data_type}_R2D_extr_{year}.txt"
            
            # Пути к файлам списков и треков
            output_list_file = f"{input_list_dir}/{data_type}_R2D_extr_{year}.txt"
            output_tracks_file = f"{tracks_dir}/{data_type}_TC_tracks_{year}.txt"
            
            # Проверяем, существует ли годовой файл с данными
            if not os.path.exists(annual_file):
                print(f"Предупреждение: файл {annual_file} не найден, пропускаем {year}")
                continue
    
            # ========== ИЗМЕНЕНИЕ: Создаем список из одного файла ==========
            with open(output_list_file, 'w') as fout:
                fout.write(f"{annual_file}\n")
            
            # Проверяем, что список не пустой
            if os.path.getsize(output_list_file) == 0:
                print(f"Внимание: список файлов для {year} пуст, пропускаем")
                continue
        else:
            base_path_out = f"{nodes_dir}/{data_type}_R2D_extr"  # месячные файлы .txt
    
            # Пути к файлам списков и треков
            output_list_file = f"{input_list_dir}/{data_type}_R2D_extr_{year}.txt"
            output_tracks_file = f"{tracks_dir}/{data_type}_TC_tracks_{year}.txt"
    
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
                f"--in_fmt \"lon,lat,rad,r2d,wind\" "
                f"--range {search_range} "
                # f"--mintime \"12h\" " # попробовать 18h - чтобы точно не ловить суточные
                # f"--maxgap \"3h\" "
                f"--mintime \"18h\" " # 2026-07-14 - как в статье Han,Y.,& Ullrich,P.A.(2025).
                # f"--maxgap \"12h\" " # 2026-07-14 - как в статье Han,Y.,& Ullrich,P.A.(2025).
                f"--maxgap \"{maxgap}h\" " # 2026-08-25 - стараемся получить надежные треки по часовым данным

                # f"--mintime \"144h\" " # 2026-08-03 - ocean
                # f"--maxgap \"72h\" " # 2026-08-03 - ocean
            
                # f"--prioritize -r2d"  # Приоритет соединения с узлом, имеющим МАКСИМАЛЬНОЕ r2d
                # f"--min_endpoint_dist 0.5 " # это если фильтр по стационирующим КВС
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